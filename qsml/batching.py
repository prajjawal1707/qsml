"""QSML 2.2 - Adaptive batching and hybrid CPU/GPU QUBO search."""

from __future__ import annotations

import math
import os
import time
from concurrent.futures import ThreadPoolExecutor

from .qubo import QUBOEngine
from .qubo_gpu import GPUQUBOEngine


class BatchedGPUQUBOEngine:
    """
    Runs GPU candidate search in memory-aware chunks.

    Each chunk is an independent candidate-search call. The best
    solution across all chunks is retained.
    """

    def __init__(
        self,
        memory_fraction: float = 0.20,
        max_batch_size: int = 65536,
        minimum_batch_size: int = 1,
    ):
        if not 0 < memory_fraction <= 0.5:
            raise ValueError("memory_fraction must be greater than 0 and <= 0.5")

        if max_batch_size < 1 or minimum_batch_size < 1:
            raise ValueError("Batch sizes must be positive.")

        if minimum_batch_size > max_batch_size:
            raise ValueError("Minimum batch size cannot exceed maximum.")

        self.memory_fraction = memory_fraction
        self.max_batch_size = max_batch_size
        self.minimum_batch_size = minimum_batch_size

    @staticmethod
    def available_vram_bytes() -> int:
        """Return currently free CUDA memory, or raise if unavailable."""
        try:
            import cupy as cp

            if cp.cuda.runtime.getDeviceCount() < 1:
                raise RuntimeError("No CUDA GPU is available.")

            free_bytes, _ = cp.cuda.runtime.memGetInfo()
            return int(free_bytes)

        except Exception as exc:
            raise RuntimeError(
                f"Cannot inspect GPU memory: {exc}"
            ) from exc

    @staticmethod
    def estimate_bytes_per_candidate(num_variables: int) -> int:
        """
        Conservative planning estimate, not a measurement of the
        GPUQUBOEngine's exact internal allocations.

        Actual GPU memory use depends on CuPy dtypes and implementation.
        """
        if not isinstance(num_variables, int) or num_variables < 1:
            raise ValueError("num_variables must be a positive integer.")

        return max(256, num_variables * 32 + 128)

    def choose_batch_size(
        self,
        num_variables: int,
        requested_batch_size: int | None = None,
        free_vram_bytes: int | None = None,
    ) -> int:
        """Choose a conservative candidate batch size from free VRAM."""
        if free_vram_bytes is None:
            free_vram_bytes = self.available_vram_bytes()

        if free_vram_bytes <= 0:
            raise MemoryError("No free GPU memory is available.")

        usable_bytes = int(free_vram_bytes * self.memory_fraction)
        bytes_per_candidate = self.estimate_bytes_per_candidate(
            num_variables
        )

        memory_limited = usable_bytes // bytes_per_candidate

        if memory_limited < 1:
            raise MemoryError(
                "Available VRAM is below the minimum estimated batch budget."
            )

        batch_size = min(
            int(memory_limited),
            self.max_batch_size,
        )

        if requested_batch_size is not None:
            if requested_batch_size < 1:
                raise ValueError("requested_batch_size must be positive.")
            batch_size = min(batch_size, requested_batch_size)

        return max(
            min(batch_size, self.max_batch_size),
            self.minimum_batch_size,
        )

    @staticmethod
    def plan_batches(total_candidates: int, batch_size: int) -> list[int]:
        """Split a candidate budget into deterministic batch sizes."""
        if total_candidates < 1:
            raise ValueError("total_candidates must be positive.")

        if batch_size < 1:
            raise ValueError("batch_size must be positive.")

        full_batches, remainder = divmod(total_candidates, batch_size)

        batches = [batch_size] * full_batches

        if remainder:
            batches.append(remainder)

        return batches

    def solve(
        self,
        num_variables: int,
        terms: dict[tuple[int, int], float],
        candidates: int = 32768,
        seed: int = 42,
        requested_batch_size: int | None = None,
        free_vram_bytes: int | None = None,
    ) -> dict:
        if candidates < 1:
            raise ValueError("candidates must be positive.")

        batch_size = self.choose_batch_size(
            num_variables=num_variables,
            requested_batch_size=requested_batch_size,
            free_vram_bytes=free_vram_bytes,
        )

        batches = self.plan_batches(candidates, batch_size)

        engine = GPUQUBOEngine()
        best_solution = None
        best_energy = float("inf")
        total_evaluations = 0
        batch_results = []

        started = time.perf_counter()

        for batch_index, batch_candidates in enumerate(batches):
            batch_seed = seed + batch_index

            result = engine.solve(
                num_variables=num_variables,
                terms=terms,
                candidates=batch_candidates,
                seed=batch_seed,
            )

            energy = float(result.energy)
            evaluations = int(result.evaluations)

            total_evaluations += evaluations

            batch_results.append({
                "batch": batch_index + 1,
                "candidates": batch_candidates,
                "seed": batch_seed,
                "energy": energy,
                "evaluations": evaluations,
            })

            if energy < best_energy:
                best_energy = energy
                best_solution = list(result.solution)

        elapsed = time.perf_counter() - started

        if best_solution is None:
            raise RuntimeError("GPU search produced no candidate solution.")

        return {
            "backend": "GPU",
            "method": "VRAM-aware chunked random candidate search",
            "solution": best_solution,
            "energy": best_energy,
            "evaluations": total_evaluations,
            "requested_candidates": candidates,
            "batch_size": batch_size,
            "batch_count": len(batches),
            "batch_results": batch_results,
            "elapsed_seconds": elapsed,
        }


class HybridQUBOEngine:
    """
    Runs CPU simulated annealing and chunked GPU search concurrently.

    Both engines optimize the same QUBO objective. The lower-energy
    result is selected after both searches finish.
    """

    def __init__(
        self,
        gpu_memory_fraction: float = 0.20,
        max_gpu_batch_size: int = 65536,
    ):
        self.cpu_engine = QUBOEngine()

        self.gpu_engine = BatchedGPUQUBOEngine(
            memory_fraction=gpu_memory_fraction,
            max_batch_size=max_gpu_batch_size,
        )

    def solve(
        self,
        num_variables: int,
        terms: dict[tuple[int, int], float],
        gpu_candidates: int = 32768,
        cpu_iterations: int = 12000,
        cpu_restarts: int = 8,
        seed: int = 42,
        gpu_batch_size: int | None = None,
    ) -> dict:
        if num_variables < 1:
            raise ValueError("num_variables must be positive.")

        if not terms:
            raise ValueError("At least one QUBO term is required.")

        started = time.perf_counter()

        def run_cpu():
            result = self.cpu_engine.solve(
                num_variables=num_variables,
                terms=terms,
                iterations=cpu_iterations,
                restarts=cpu_restarts,
                seed=seed,
            )

            return {
                "backend": "CPU",
                "method": "Simulated annealing",
                "solution": list(result.solution),
                "energy": float(result.energy),
                "evaluations": int(result.evaluations),
            }

        def run_gpu():
            return self.gpu_engine.solve(
                num_variables=num_variables,
                terms=terms,
                candidates=gpu_candidates,
                seed=seed,
                requested_batch_size=gpu_batch_size,
            )

        with ThreadPoolExecutor(max_workers=2) as executor:
            cpu_future = executor.submit(run_cpu)
            gpu_future = executor.submit(run_gpu)

            # Calling result() propagates errors rather than silently
            # returning an incomplete hybrid result.
            cpu_result = cpu_future.result()
            gpu_result = gpu_future.result()

        elapsed = time.perf_counter() - started

        if cpu_result["energy"] <= gpu_result["energy"]:
            selected = cpu_result
        else:
            selected = gpu_result

        return {
            "backend": "HYBRID",
            "method": "Concurrent CPU simulated annealing + GPU batching",
            "solution": selected["solution"],
            "energy": selected["energy"],
            "selected_backend": selected["backend"],
            "elapsed_seconds": elapsed,
            "cpu_result": cpu_result,
            "gpu_result": gpu_result,
        }
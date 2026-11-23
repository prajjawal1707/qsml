"""QSML 1.4 - Adaptive optimization engine."""

import time

from .qubo import QUBOEngine


class AdaptiveOptimizer:

    def __init__(self):
        self.cpu_engine = QUBOEngine()

    def available_backends(self):
        backends = ["CPU"]

        try:
            import cupy as cp

            if cp.cuda.runtime.getDeviceCount() > 0:
                backends.append("GPU")
        except Exception:
            pass

        return backends

    def choose_backend(self, num_variables, candidates=4096):
        """Select a backend using workload size and GPU availability."""

        available = self.available_backends()

        if "GPU" in available and num_variables >= 12 and candidates >= 8192:
            return "GPU"

        return "CPU"

    def solve(
        self,
        num_variables,
        terms,
        backend="auto",
        iterations=12000,
        restarts=8,
        candidates=4096,
        seed=42,
    ):
        available = self.available_backends()

        if backend.lower() == "auto":
            selected_backend = self.choose_backend(
                num_variables,
                candidates,
            )
        else:
            selected_backend = backend.upper()

        if selected_backend not in ("CPU", "GPU"):
            raise ValueError("Backend must be auto, CPU, or GPU")

        if selected_backend not in available:
            raise RuntimeError(f"{selected_backend} backend is unavailable")

        start = time.perf_counter()

        if selected_backend == "CPU":
            result = self.cpu_engine.solve(
                num_variables=num_variables,
                terms=terms,
                iterations=iterations,
                restarts=restarts,
                seed=seed,
            )
            method = "Simulated annealing"

        else:
            from .qubo_gpu import GPUQUBOEngine

            engine = GPUQUBOEngine()

            result = engine.solve(
                num_variables=num_variables,
                terms=terms,
                candidates=candidates,
                seed=seed,
            )
            method = "GPU parallel random candidate search"

        elapsed = time.perf_counter() - start

        return {
            "backend": selected_backend,
            "method": method,
            "solution": result.solution,
            "energy": result.energy,
            "evaluations": result.evaluations,
            "elapsed_seconds": elapsed,
        }

# qsml/v2_engine.py
from __future__ import annotations

import math
import os
import time
import inspect
import platform
from dataclasses import dataclass, asdict
from typing import Any

import numpy as np
import psutil

from .runtime import UnifiedRuntime
from .adaptive import AdaptiveOptimizer
from .decision import DecisionEngine


# ============================================================
# QSML 1.7 — RESOURCE MANAGER
# ============================================================

@dataclass
class ResourceSnapshot:
    cpu_count: int
    ram_total_gb: float
    ram_available_gb: float
    gpu_available: bool
    gpu_name: str
    gpu_free_gb: float
    gpu_total_gb: float


class ResourceManager:
    """Inspects available classical compute resources."""

    def snapshot(self) -> ResourceSnapshot:
        memory = psutil.virtual_memory()

        gpu_available = False
        gpu_name = "Not available"
        gpu_free = 0.0
        gpu_total = 0.0

        try:
            import cupy as cp
            device = cp.cuda.Device()
            props = cp.cuda.runtime.getDeviceProperties(device.id)

            name = props.get("name", "CUDA GPU")
            if isinstance(name, bytes):
                name = name.decode(errors="replace")

            free_bytes, total_bytes = cp.cuda.runtime.memGetInfo()

            gpu_available = True
            gpu_name = str(name)
            gpu_free = free_bytes / (1024 ** 3)
            gpu_total = total_bytes / (1024 ** 3)
        except Exception:
            pass

        return ResourceSnapshot(
            cpu_count=os.cpu_count() or 1,
            ram_total_gb=round(memory.total / (1024 ** 3), 2),
            ram_available_gb=round(memory.available / (1024 ** 3), 2),
            gpu_available=gpu_available,
            gpu_name=gpu_name,
            gpu_free_gb=round(gpu_free, 2),
            gpu_total_gb=round(gpu_total, 2),
        )

    def estimate_statevector_gb(self, qubits: int) -> float:
        if qubits < 1:
            raise ValueError("Qubit count must be at least 1.")
        # complex128 = 16 bytes per amplitude
        return (2 ** qubits) * 16 / (1024 ** 3)

    def choose_backend(self, requested: str, qubits: int = 1) -> str:
        requested = requested.upper()
        if requested not in {"AUTO", "CPU", "GPU"}:
            raise ValueError("ENGINE must be AUTO, CPU, or GPU.")

        resources = self.snapshot()
        required_gb = self.estimate_statevector_gb(qubits)

        if requested == "CPU":
            if required_gb > resources.ram_available_gb * 0.5:
                raise MemoryError("State-vector estimate exceeds the safe RAM budget.")
            return "CPU"

        if requested == "GPU":
            if not resources.gpu_available:
                raise RuntimeError("GPU was requested, but CUDA is unavailable.")
            if required_gb > resources.gpu_free_gb * 0.5:
                raise MemoryError("State-vector estimate exceeds the safe VRAM budget.")
            return "GPU"

        # AUTO: prefer GPU only when it has enough memory.
        if resources.gpu_available and required_gb <= resources.gpu_free_gb * 0.5:
            return "GPU"

        if required_gb <= resources.ram_available_gb * 0.5:
            return "CPU"

        raise MemoryError("The requested state vector exceeds safe available memory.")


# ============================================================
# QSML 1.5 — UNIFIED LANGUAGE PARSER
# ============================================================

@dataclass
class QSMLProgram:
    engine: str = "AUTO"
    qubits: int = 3
    marked_state: str = "101"
    iterations: int = 2
    shots: int = 1000
    decide: bool = True


class QSMLParser:
    """Small, strict parser for the QSML 2.0 search program."""

    def parse(self, source: str) -> QSMLProgram:
        program = QSMLProgram()
        seen_register = False

        for line_number, raw_line in enumerate(source.splitlines(), start=1):
            line = raw_line.split("#", 1)[0].strip()
            if not line:
                continue

            parts = line.split()
            command = parts[0].upper()
            args = parts[1:]

            try:
                if command == "ENGINE":
                    self._require(args, 1, command)
                    program.engine = args[0].upper()
                    if program.engine not in {"AUTO", "CPU", "GPU"}:
                        raise ValueError("ENGINE must be AUTO, CPU, or GPU.")

                elif command == "REGISTER":
                    self._require(args, 1, command)
                    program.qubits = int(args[0])
                    if not 1 <= program.qubits <= 30:
                        raise ValueError("REGISTER supports 1–30 qubits.")
                    seen_register = True

                elif command == "SUPERPOSITION":
                    self._require(args, 1, command)
                    if args[0].upper() != "ALL":
                        raise ValueError("Only SUPERPOSITION ALL is supported.")

                elif command == "MARK":
                    self._require(args, 1, command)
                    program.marked_state = args[0]

                elif command == "AMPLIFY":
                    self._require(args, 1, command)
                    program.iterations = int(args[0])
                    if program.iterations < 0:
                        raise ValueError("AMPLIFY iterations cannot be negative.")

                elif command == "MEASURE":
                    self._require(args, 1, command)
                    program.shots = int(args[0])
                    if program.shots < 1:
                        raise ValueError("MEASURE shots must be positive.")

                elif command == "DECIDE":
                    self._require(args, 1, command)
                    if args[0].upper() != "BEST":
                        raise ValueError("Only DECIDE BEST is supported.")
                    program.decide = True

                elif command == "SHOW":
                    self._require(args, 1, command)
                    if args[0].upper() != "RESULTS":
                        raise ValueError("Only SHOW RESULTS is supported.")

                else:
                    raise ValueError(f"Unknown instruction: {command}")

            except Exception as exc:
                raise ValueError(f"QSML line {line_number}: {exc}") from exc

        if not seen_register:
            raise ValueError("Program must contain REGISTER.")
        if len(program.marked_state) != program.qubits:
            raise ValueError("MARK state length must equal REGISTER size.")
        if any(bit not in "01" for bit in program.marked_state):
            raise ValueError("MARK state must contain only 0 and 1.")

        return program

    @staticmethod
    def _require(args: list[str], count: int, command: str) -> None:
        if len(args) != count:
            raise ValueError(f"{command} expects {count} argument(s).")


# ============================================================
# QSML 1.5 + 1.7 — UNIFIED EXECUTION
# ============================================================

class QSML2Engine:
    def __init__(self):
        self.runtime = UnifiedRuntime()
        self.optimizer = AdaptiveOptimizer()
        self.decision_engine = DecisionEngine()
        self.resources = ResourceManager()
        self.parser = QSMLParser()

    def execute(self, source: str) -> dict[str, Any]:
        program = self.parser.parse(source)

        backend = self.resources.choose_backend(
            program.engine,
            qubits=program.qubits,
        )

        started = time.perf_counter()

        result = self.runtime.grover(
    num_qubits=program.qubits,
    marked_states=[program.marked_state],
    iterations=program.iterations,
    shots=program.shots,
    backend=backend,
)


        elapsed = time.perf_counter() - started

        decision = None
        if program.decide:
            decision = self._make_decision(result)

        return {
            "version": "2.0",
            "backend": backend,
            "program": asdict(program),
            "result": result,
            "decision": decision,
            "execution_seconds": round(elapsed, 6),
            "resources": asdict(self.resources.snapshot()),
        }

    def _make_decision(self, result: Any) -> dict[str, Any]:
        """Support both the original result object and runtime result dictionary."""
        try:
            report = self.decision_engine.analyse(result)
            if hasattr(report, "__dict__"):
                return dict(report.__dict__)
            return {"report": str(report)}
        except Exception:
            if isinstance(result, dict):
                measurements = result.get("measurements", {})
                candidates = result.get("top_candidates", [])

                if not candidates and measurements:
                    candidates = sorted(
                        measurements.items(),
                        key=lambda item: item[1],
                        reverse=True,
                    )

                if candidates:
                    best = candidates[0]
                    if isinstance(best, (tuple, list)):
                        state = str(best[0])
                        confidence = float(best[1]) if len(best) > 1 else 0.0
                    else:
                        state = str(best)
                        total = sum(measurements.values()) or 1
                        confidence = measurements.get(state, 0) / total

                    return {
                        "selected_state": state,
                        "confidence": confidence,
                        "method": "highest measured frequency",
                    }

            return {"method": "decision unavailable", "selected_state": None}


# ============================================================
# QSML 1.6 — OPTIMIZATION ADAPTER
# ============================================================

def _call_with_supported_arguments(method, values: dict[str, Any]):
    """Pass only keyword arguments accepted by the installed implementation."""
    signature = inspect.signature(method)
    accepts_kwargs = any(
        p.kind == inspect.Parameter.VAR_KEYWORD
        for p in signature.parameters.values()
    )

    kwargs = values if accepts_kwargs else {
        key: value for key, value in values.items()
        if key in signature.parameters
    }
    return method(**kwargs)


class OptimizationService:
    """Connects matrix-based QUBO input to the existing adaptive optimizer."""

    def __init__(self):
        self.optimizer = AdaptiveOptimizer()

    @staticmethod
    def matrix_to_terms(matrix):
        """
        Convert QUBO matrix into the dictionary format expected by
        QUBOEngine and GPUQUBOEngine.

        E(x) = sum(Q[i,j] * x[i] * x[j])

        The existing QUBOEngine normalizes (i,j) and (j,i)
        by adding their coefficients. Therefore, retain only
        the upper triangle when converting a conventional matrix.
        """
        matrix = np.asarray(matrix, dtype=float)

        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            raise ValueError("QUBO matrix must be square.")

        if not np.all(np.isfinite(matrix)):
            raise ValueError("QUBO matrix contains non-finite values.")

        # A conventional QUBO matrix may be asymmetric.
        # Convert it to its equivalent upper-triangle representation.
        symmetric = matrix + matrix.T
        np.fill_diagonal(symmetric, np.diag(matrix))

        terms = {}
        n = matrix.shape[0]

        for i in range(n):
            for j in range(i, n):
                coefficient = float(symmetric[i, j])
                if coefficient != 0.0:
                    terms[(i, j)] = coefficient

        if not terms:
            raise ValueError("At least one nonzero QUBO coefficient is required.")

        return terms

    def solve_qubo(
        self,
        matrix,
        backend="auto",
        candidates=8192,
        iterations=15000,
        restarts=10,
        seed=42,
    ):
        matrix = np.asarray(matrix, dtype=float)

        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            raise ValueError("QUBO matrix must be square.")

        terms = self.matrix_to_terms(matrix)
        num_variables = matrix.shape[0]

        started = time.perf_counter()

        result = self.optimizer.solve(
            num_variables=num_variables,
            terms=terms,
            backend=backend,
            iterations=iterations,
            restarts=restarts,
            candidates=candidates,
            seed=seed,
        )

        elapsed = time.perf_counter() - started

        output = dict(result) if isinstance(result, dict) else (
            dict(result.__dict__) if hasattr(result, "__dict__")
            else {"result": str(result)}
        )

        output["elapsed_seconds"] = round(elapsed, 6)
        output["requested_backend"] = backend
        output["num_variables"] = num_variables
        output["terms"] = {
    f"{i},{j}": coefficient
    for (i, j), coefficient in terms.items()
}

        return output

# ============================================================
# QSML 1.8 + 1.9 — BENCHMARKING
# ============================================================

class BenchmarkSuite:
    """Runs repeatable timing tests without claiming unlike workloads are equal."""

    @staticmethod
    def run(callable_object, repeats: int = 3) -> dict[str, Any]:
        if repeats < 1:
            raise ValueError("Benchmark repeats must be at least 1.")

        durations = []
        last_result = None

        for _ in range(repeats):
            start = time.perf_counter()
            last_result = callable_object()
            durations.append(time.perf_counter() - start)

        return {
            "repeats": repeats,
            "times_seconds": [round(x, 6) for x in durations],
            "mean_seconds": round(float(np.mean(durations)), 6),
            "min_seconds": round(float(np.min(durations)), 6),
            "max_seconds": round(float(np.max(durations)), 6),
            "result": last_result,
        }

    @staticmethod
    def compare_cpu_gpu(
        engine: QSML2Engine,
        source: str,
        repeats: int = 3,
    ) -> dict[str, Any]:
        """Run identical QSML source on both backends, if GPU is available."""
        program = engine.parser.parse(source)
        output = {}

        for backend in ("CPU", "GPU"):
            try:
                forced_source = source.replace(
                    f"ENGINE {program.engine}",
                    f"ENGINE {backend}",
                )
                if forced_source == source and "ENGINE" not in source.upper():
                    forced_source = f"ENGINE {backend}\n{source}"

                output[backend] = BenchmarkSuite.run(
                    lambda: engine.execute(forced_source),
                    repeats=repeats,
                )
            except Exception as exc:
                output[backend] = {"available": False, "error": str(exc)}

        return output


# ============================================================
# QSML 2.0 — REPORTING
# ============================================================

def print_report(output: dict[str, Any]) -> None:
    print("\n" + "=" * 60)
    print("                 QSML 2.0 EXECUTION REPORT")
    print("=" * 60)

    print(f"Version:          {output.get('version')}")
    print(f"Backend:          {output.get('backend')}")
    print(f"Execution time:   {output.get('execution_seconds')} seconds")

    program = output.get("program", {})
    print(f"Qubits:           {program.get('qubits')}")
    print(f"Marked state:     {program.get('marked_state')}")
    print(f"Measurement shots:{program.get('shots')}")

    decision = output.get("decision")
    if decision:
        print("\nDecision:")
        for key, value in decision.items():
            print(f"  {key}: {value}")

    resources = output.get("resources", {})
    print("\nHardware:")
    print(f"  CPU cores:      {resources.get('cpu_count')}")
    print(f"  RAM available:  {resources.get('ram_available_gb')} GB")
    print(f"  GPU available:  {resources.get('gpu_available')}")
    print(f"  GPU:            {resources.get('gpu_name')}")
    print(f"  VRAM available: {resources.get('gpu_free_gb')} GB")

    print("=" * 60)

# ============================================================
# QSML 1.8 — PARALLEL TASK EXECUTION
# ============================================================

from concurrent.futures import ThreadPoolExecutor, as_completed


class ParallelExecutor:
    """Runs independent classical tasks concurrently on CPU threads."""

    def __init__(self, workers: int | None = None):
        available = os.cpu_count() or 1
        self.workers = max(1, min(workers or available, available))

    def map(self, function, items):
        items = list(items)
        results = [None] * len(items)

        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            futures = {
                pool.submit(function, item): index
                for index, item in enumerate(items)
            }

            for future in as_completed(futures):
                index = futures[future]
                results[index] = future.result()

        return results
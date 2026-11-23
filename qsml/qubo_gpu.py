"""QSML 3.1.1 - GPU candidate-search QUBO engine."""
from dataclasses import dataclass
import math

import cupy as cp

from .qubo_contract import (
    normalize_qubo_terms,
    validate_positive_int,
    validate_seed,
)


@dataclass
class GPUQUBOResult:
    solution: list[int]
    energy: float
    evaluations: int
    device: str


class GPUQUBOEngine:
    def __init__(self):
        if cp.cuda.runtime.getDeviceCount() < 1:
            raise RuntimeError("No CUDA GPU detected")
        self.device = cp.cuda.Device()

    def solve(
        self,
        num_variables: int,
        terms: dict[tuple[int, int], float],
        candidates: int = 4096,
        seed: int = 42,
    ) -> GPUQUBOResult:
        validate_positive_int(num_variables, "num_variables")
        validate_positive_int(candidates, "candidates")
        validate_seed(seed)
        normalized = normalize_qubo_terms(num_variables, terms)

        rng = cp.random.RandomState(seed)
        solutions = rng.randint(
            0, 2, size=(candidates, num_variables), dtype=cp.int32
        )
        energies = cp.zeros(candidates, dtype=cp.float64)

        for (i, j), coefficient in normalized.items():
            energies += coefficient * solutions[:, i] * solutions[:, j]

        cp.cuda.Stream.null.synchronize()
        if not bool(cp.all(cp.isfinite(energies)).get()):
            raise ValueError("GPU QUBO energies contain non-finite values")

        best_index = int(cp.argmin(energies).get())
        best_solution = solutions[best_index].get().astype(int).tolist()
        best_energy = float(energies[best_index].get())

        if not math.isfinite(best_energy):
            raise ValueError("GPU QUBO best energy is not finite")

        return GPUQUBOResult(
            solution=best_solution,
            energy=best_energy,
            evaluations=candidates,
            device=str(self.device),
        )

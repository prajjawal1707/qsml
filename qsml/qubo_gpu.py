"""QSML 0.9 - GPU-accelerated QUBO engine.

Uses CuPy to evaluate binary QUBO candidate solutions on CUDA.
"""

from dataclasses import dataclass
import math

import cupy as cp


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

        if not isinstance(num_variables, int) or num_variables < 1:
            raise ValueError("num_variables must be positive")

        if not isinstance(candidates, int) or candidates < 1:
            raise ValueError("candidates must be positive")

        if not terms:
            raise ValueError("QUBO terms cannot be empty")

        normalized = {}

        for (i, j), coefficient in terms.items():
            if not isinstance(i, int) or not isinstance(j, int):
                raise ValueError("QUBO indices must be integers")

            if not (0 <= i < num_variables and 0 <= j < num_variables):
                raise ValueError("QUBO index out of range")

            if not math.isfinite(coefficient):
                raise ValueError("QUBO coefficients must be finite")

            key = (min(i, j), max(i, j))
            normalized[key] = normalized.get(key, 0.0) + coefficient

        rng = cp.random.RandomState(seed)

        # Generate binary candidate solutions on the GPU.
        solutions = rng.randint(
            0,
            2,
            size=(candidates, num_variables),
            dtype=cp.int32,
        )

        energies = cp.zeros(candidates, dtype=cp.float64)

        # Evaluate the QUBO objective on the GPU.
        for (i, j), coefficient in normalized.items():
            energies += coefficient * solutions[:, i] * solutions[:, j]

        # Find the candidate with minimum energy.
        best_index = int(cp.argmin(energies).get())

        best_solution = solutions[best_index].get().astype(int).tolist()

        best_energy = float(energies[best_index].get())

        cp.cuda.Stream.null.synchronize()

        return GPUQUBOResult(
            solution=best_solution,
            energy=best_energy,
            evaluations=candidates,
            device=str(self.device),
        )

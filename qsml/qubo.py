"""QSML 3.1.1 - General QUBO optimization engine."""
from dataclasses import dataclass
import math
import random

from .qubo_contract import (
    normalize_qubo_terms,
    validate_positive_int,
    validate_seed,
)


@dataclass
class QUBOResult:
    solution: list[int]
    energy: float
    evaluations: int


class QUBOEngine:
    """Classical simulated-annealing solver for binary QUBO objectives."""

    def solve(
        self,
        num_variables: int,
        terms: dict[tuple[int, int], float],
        iterations: int = 15000,
        restarts: int = 10,
        seed: int | None = 42,
    ) -> QUBOResult:
        validate_positive_int(num_variables, "num_variables")
        validate_positive_int(iterations, "iterations")
        validate_positive_int(restarts, "restarts")
        validate_seed(seed)
        normalized_terms = normalize_qubo_terms(num_variables, terms)

        def energy(solution):
            try:
                value = math.fsum(
                    coefficient * solution[i] * solution[j]
                    for (i, j), coefficient in normalized_terms.items()
                )
            except OverflowError as exc:
                raise ValueError("QUBO energy overflowed") from exc
            if not math.isfinite(value):
                raise ValueError("QUBO energy is not finite")
            return value

        rng = random.Random(seed)
        largest_coefficient = max(
            abs(value) for value in normalized_terms.values()
        )
        start_temperature = max(1.0, largest_coefficient * num_variables)
        if not math.isfinite(start_temperature):
            raise ValueError("QUBO temperature scale is not finite")
        end_temperature = 0.001

        best_solution = [0] * num_variables
        best_energy = energy(best_solution)
        evaluations = 0

        for _ in range(restarts):
            current = [rng.randint(0, 1) for _ in range(num_variables)]
            current_energy = energy(current)
            evaluations += 1

            if current_energy < best_energy:
                best_solution = current.copy()
                best_energy = current_energy

            for step in range(iterations):
                fraction = step / max(1, iterations - 1)
                temperature = (
                    start_temperature
                    * (end_temperature / start_temperature) ** fraction
                )

                candidate = current.copy()
                index = rng.randrange(num_variables)
                candidate[index] = 1 - candidate[index]
                candidate_energy = energy(candidate)
                evaluations += 1

                delta = candidate_energy - current_energy
                if delta <= 0 or rng.random() < math.exp(
                    max(-700.0, -delta / temperature)
                ):
                    current = candidate
                    current_energy = candidate_energy

                if current_energy < best_energy:
                    best_solution = current.copy()
                    best_energy = current_energy

        return QUBOResult(
            solution=best_solution,
            energy=best_energy,
            evaluations=evaluations,
        )

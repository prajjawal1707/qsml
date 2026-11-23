"""QSML 0.8 - General QUBO optimization engine.

Minimizes a quadratic binary objective:

    E(x) = sum(Q[i, j] * x[i] * x[j])

Each variable x[i] can be either 0 or 1.
Uses classical simulated annealing to search for low-energy solutions.
"""

from dataclasses import dataclass
import math
import random


@dataclass
class QUBOResult:
    solution: list[int]
    energy: float
    evaluations: int


class QUBOEngine:
    """General-purpose binary quadratic optimization engine."""

    def solve(
        self,
        num_variables: int,
        terms: dict[tuple[int, int], float],
        iterations: int = 15000,
        restarts: int = 10,
        seed: int | None = 42,
    ) -> QUBOResult:

        if not isinstance(num_variables, int) or num_variables < 1:
            raise ValueError("num_variables must be a positive integer")

        if iterations < 1 or restarts < 1:
            raise ValueError("iterations and restarts must be positive")

        # Normalize terms so each pair uses i <= j.
        normalized_terms = {}

        for pair, coefficient in terms.items():
            if len(pair) != 2:
                raise ValueError("Every QUBO term must have two indices")

            i, j = pair

            if not isinstance(i, int) or not isinstance(j, int):
                raise ValueError("QUBO indices must be integers")

            if not (0 <= i < num_variables and 0 <= j < num_variables):
                raise ValueError("QUBO index is outside the variable range")

            if not math.isfinite(coefficient):
                raise ValueError("QUBO coefficients must be finite")

            key = (min(i, j), max(i, j))
            normalized_terms[key] = normalized_terms.get(key, 0.0) + coefficient

        if not normalized_terms:
            raise ValueError("At least one QUBO term is required")

        def energy(solution):
            return sum(
                coefficient * solution[i] * solution[j]
                for (i, j), coefficient in normalized_terms.items()
            )

        rng = random.Random(seed)

        largest_coefficient = max(abs(value) for value in normalized_terms.values())

        start_temperature = max(1.0, largest_coefficient * num_variables)
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

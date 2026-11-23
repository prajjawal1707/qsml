"""QSML 0.7 - Quantum-inspired optimization.

Uses simulated annealing to search for high-value feasible
solutions to binary selection problems such as knapsack.
"""

from dataclasses import dataclass
import math
import random


@dataclass
class OptimizationResult:
    selected_items: list[int]
    selection: list[int]
    total_weight: float
    total_value: float
    capacity: float
    evaluations: int


class OptimizationEngine:
    """Binary selection optimizer using simulated annealing."""

    def solve_knapsack(
        self,
        weights: list[float],
        values: list[float],
        capacity: float,
        iterations: int = 12000,
        restarts: int = 8,
        seed: int | None = 42,
    ) -> OptimizationResult:

        if not weights or len(weights) != len(values):
            raise ValueError("Weights and values must have equal nonzero lengths")

        if any(not math.isfinite(w) or w <= 0 for w in weights):
            raise ValueError("Every weight must be positive and finite")

        if any(not math.isfinite(v) or v < 0 for v in values):
            raise ValueError("Values must be nonnegative and finite")

        if not math.isfinite(capacity) or capacity <= 0:
            raise ValueError("Capacity must be positive and finite")

        if iterations < 1 or restarts < 1:
            raise ValueError("Iterations and restarts must be positive")

        rng = random.Random(seed)
        count = len(weights)

        # Any overweight solution receives a score below zero.
        penalty = sum(values) + 1.0

        def evaluate(selection):
            weight = sum(weights[i] for i in range(count) if selection[i])
            value = sum(values[i] for i in range(count) if selection[i])

            excess = max(0.0, weight - capacity)
            score = value - penalty * excess

            return score, weight, value

        best_selection = [0] * count
        best_weight = 0.0
        best_value = 0.0
        evaluations = 0

        for _ in range(restarts):

            # Begin with a random feasible selection.
            order = list(range(count))
            rng.shuffle(order)

            current = [0] * count
            current_weight = 0.0

            for index in order:
                if current_weight + weights[index] <= capacity:
                    if rng.random() < 0.5:
                        current[index] = 1
                        current_weight += weights[index]

            current_score, _, _ = evaluate(current)
            evaluations += 1

            local_best = current.copy()
            local_score = current_score

            # Gradually cool the system.
            start_temperature = max(1.0, sum(values) / 5.0)
            end_temperature = 0.01

            for step in range(iterations):
                fraction = step / max(1, iterations - 1)

                temperature = (
                    start_temperature
                    * (end_temperature / start_temperature) ** fraction
                )

                candidate = current.copy()

                # Flip one randomly selected binary decision.
                index = rng.randrange(count)
                candidate[index] = 1 - candidate[index]

                candidate_score, _, _ = evaluate(candidate)
                evaluations += 1

                delta = candidate_score - current_score

                # Accept improvements; sometimes accept worse moves.
                if delta >= 0 or rng.random() < math.exp(
                    max(-700.0, delta / temperature)
                ):
                    current = candidate
                    current_score = candidate_score

                if current_score > local_score:
                    local_best = current.copy()
                    local_score = current_score

            _, local_weight, local_value = evaluate(local_best)

            if local_weight <= capacity and local_value > best_value:
                best_selection = local_best
                best_weight = local_weight
                best_value = local_value

        selected_items = [i for i, selected in enumerate(best_selection) if selected]

        return OptimizationResult(
            selected_items=selected_items,
            selection=best_selection,
            total_weight=best_weight,
            total_value=best_value,
            capacity=capacity,
            evaluations=evaluations,
        )

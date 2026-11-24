
"""QSML 4.2 — Guided Decision Search."""

import math
import random

from .decision_space import DecisionSpaceExplorer


def _positive_integer(value, name, maximum=None):
    if (
        not isinstance(value, int)
        or isinstance(value, bool)
        or value < 1
    ):
        raise ValueError(f"{name} must be a positive integer")

    if maximum is not None and value > maximum:
        raise ValueError(f"{name} cannot exceed {maximum}")

    return value


def _finite_number(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")

    value = float(value)

    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")

    return value


class GuidedDecisionSearch:
    """Improve supplied candidates through bounded one-bit neighborhood search.

    All objectives are minimized.

    The weighted score is a search heuristic, not a proof of Pareto
    optimality. The final candidates are independently passed through
    DecisionSpaceExplorer.
    """

    def __init__(self, tolerance=1e-9):
        self.explorer = DecisionSpaceExplorer(tolerance=tolerance)
        self.tolerance = float(tolerance)

    def search(
        self,
        num_variables,
        objectives,
        candidates,
        constraints=None,
        steps_per_seed=20,
        neighbor_budget=10000,
        weights=None,
        seed=42,
    ):
        steps_per_seed = _positive_integer(
            steps_per_seed, "steps_per_seed", maximum=10000
        )
        neighbor_budget = _positive_integer(
            neighbor_budget, "neighbor_budget", maximum=1_000_000
        )

        if not isinstance(seed, int) or isinstance(seed, bool):
            raise ValueError("seed must be an integer")

        # First validate and analyze the starting candidate pool.
        initial_report = self.explorer.analyze(
            num_variables,
            objectives,
            candidates,
            constraints,
        )

        initial_records = initial_report["candidate_results"]

        if not initial_records:
            raise ValueError("At least one starting candidate is required")

        names = initial_report["objective_names"]

        if weights is None:
            weights = {name: 1.0 for name in names}

        if not isinstance(weights, dict) or set(weights) != set(names):
            raise ValueError(
                "weights must contain exactly one value per objective"
            )

        checked_weights = {}

        for name in names:
            value = _finite_number(weights[name], f"weight {name}")

            if value < 0:
                raise ValueError("Objective weights must be non-negative")

            checked_weights[name] = value

        if math.fsum(checked_weights.values()) <= 0:
            raise ValueError("At least one objective weight must be positive")

        weight_total = math.fsum(checked_weights.values())

        checked_weights = {
            name: value / weight_total
            for name, value in checked_weights.items()
        }

        # Establish fixed normalization scales from the supplied pool.
        bounds = {}

        for name in names:
            values = [
                record["objectives"][name]
                for record in initial_records
            ]

            low = min(values)
            high = max(values)
            scale = high - low

            if scale == 0:
                scale = max(abs(low), 1.0)

            bounds[name] = (low, scale)

        def analyze_one(solution):
            report = self.explorer.analyze(
                num_variables,
                objectives,
                [solution],
                constraints,
            )

            return report["candidate_results"][0]

        def score(record):
            objective_score = math.fsum(
                checked_weights[name]
                * (
                    record["objectives"][name] - bounds[name][0]
                )
                / bounds[name][1]
                for name in names
            )

            total_violation = math.fsum(
                item["violation"]
                for item in record["constraints"]
            )

            # Feasibility takes priority over objective score.
            return (
                0 if record["feasible"] else 1,
                total_violation,
                objective_score,
            )

        rng = random.Random(seed)

        visited = {
            tuple(item["solution"])
            for item in initial_records
        }

        all_records = {
            tuple(item["solution"]): item
            for item in initial_records
        }

        evaluations = 0
        accepted_moves = 0
        budget_exhausted = False

        for starting_record in initial_records:
            current = list(starting_record["solution"])
            current_record = all_records[tuple(current)]

            for _ in range(steps_per_seed):
                if evaluations + num_variables > neighbor_budget:
                    budget_exhausted = True
                    break

                neighbors = []

                # Evaluate each one-bit neighbor.
                for bit_index in range(num_variables):
                    candidate = list(current)
                    candidate[bit_index] = 1 - candidate[bit_index]
                    key = tuple(candidate)

                    if key in visited:
                        continue

                    visited.add(key)

                    record = analyze_one(candidate)
                    evaluations += 1
                    all_records[key] = record

                    neighbors.append(record)

                if not neighbors:
                    break

                best_neighbor = min(neighbors, key=score)

                if score(best_neighbor) < score(current_record):
                    current_record = best_neighbor
                    current = list(best_neighbor["solution"])
                    accepted_moves += 1
                else:
                    # Randomly restart from the best starting pool
                    # candidate on the next seed.
                    break

            if budget_exhausted:
                break

            if evaluations >= neighbor_budget:
                budget_exhausted = True
                break

        final_records = list(all_records.values())

        final_candidates = [
            item["solution"]
            for item in final_records
        ]

        final_report = self.explorer.analyze(
            num_variables,
            objectives,
            final_candidates,
            constraints,
        )

        final_report["guided_search"] = {
            "method": "WEIGHTED_ONE_BIT_LOCAL_SEARCH",
            "seed": seed,
            "steps_per_seed": steps_per_seed,
            "neighbor_budget": neighbor_budget,
            "neighbor_evaluations": evaluations,
            "accepted_moves": accepted_moves,
            "visited_candidates": len(visited),
            "budget_exhausted": budget_exhausted,
            "weights": checked_weights,
            "objective_normalization": bounds,
        }

        final_report["global_optimum_proven"] = False

        return final_report

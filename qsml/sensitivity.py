
"""QSML 4.5 — Objective coefficient sensitivity analysis."""

import math
import random

from .decision_space import DecisionSpaceExplorer
from .qubo_contract import normalize_qubo_terms


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


def analyze_coefficient_sensitivity(
    num_variables,
    objectives,
    candidates,
    constraints=None,
    trials=20,
    perturbation=0.05,
    seed=42,
    tolerance=1e-9,
):
    """Perturb objective coefficients and track frontier membership.

    Each nonzero coefficient is multiplied by a random factor from
    [1-perturbation, 1+perturbation].

    Constraints are kept unchanged. This is a coefficient sensitivity
    experiment, not a statistical confidence interval.
    """

    trials = _positive_integer(trials, "trials", maximum=1000)

    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ValueError("seed must be an integer")

    if isinstance(perturbation, bool) or not isinstance(
        perturbation, (int, float)
    ):
        raise ValueError("perturbation must be numeric")

    perturbation = float(perturbation)

    if not math.isfinite(perturbation) or not 0 <= perturbation <= 1:
        raise ValueError("perturbation must be between 0 and 1")

    if not isinstance(objectives, dict) or not objectives:
        raise ValueError("objectives must be a non-empty dictionary")

    try:
        candidate_pool = list(candidates)
    except TypeError as exc:
        raise ValueError("candidates must be iterable") from exc

    if not candidate_pool:
        raise ValueError("At least one candidate is required")

    normalized = {
        name: normalize_qubo_terms(num_variables, terms)
        for name, terms in objectives.items()
    }

    explorer = DecisionSpaceExplorer(tolerance=tolerance)

    baseline = explorer.analyze(
        num_variables,
        normalized,
        candidate_pool,
        constraints,
    )

    def frontier_keys(report):
        return {
            tuple(item["solution"])
            for item in report["frontier"]
        }

    baseline_frontier = frontier_keys(baseline)

    frequency = {
        tuple(item["solution"]): 0
        for item in baseline["candidate_results"]
    }

    for solution in baseline_frontier:
        frequency[solution] += 1

    rng = random.Random(seed)
    scenario_frontiers = []
    errors = []

    for trial_index in range(trials):
        perturbed = {}

        try:
            for name, terms in normalized.items():
                changed_terms = {}

                for pair, coefficient in terms.items():
                    factor = rng.uniform(
                        1.0 - perturbation,
                        1.0 + perturbation,
                    )

                    changed = coefficient * factor

                    if not math.isfinite(changed):
                        raise ValueError(
                            "Perturbed coefficient became non-finite"
                        )

                    changed_terms[pair] = changed

                perturbed[name] = changed_terms

            report = explorer.analyze(
                num_variables,
                perturbed,
                candidate_pool,
                constraints,
            )

            current_frontier = frontier_keys(report)
            scenario_frontiers.append(current_frontier)

            for solution in current_frontier:
                frequency[solution] = frequency.get(solution, 0) + 1

        except (ValueError, OverflowError) as exc:
            errors.append({
                "trial": trial_index + 1,
                "error": str(exc),
            })

    completed = len(scenario_frontiers)

    candidate_frequency = []

    for solution, count in frequency.items():
        candidate_frequency.append({
            "solution": list(solution),
            "frontier_appearances": count,
            "frontier_frequency": (
                count / completed if completed else 0.0
            ),
        })

    candidate_frequency.sort(
        key=lambda item: (
            -item["frontier_frequency"],
            item["solution"],
        )
    )

    stable_candidates = [
        item for item in candidate_frequency
        if completed > 0 and item["frontier_appearances"] == completed
    ]

    return {
        "seed": seed,
        "requested_trials": trials,
        "completed_trials": completed,
        "perturbation": perturbation,
        "baseline_frontier": [
            list(solution)
            for solution in sorted(baseline_frontier)
        ],
        "candidate_frontier_frequency": candidate_frequency,
        "stable_frontier_candidates": stable_candidates,
        "errors": errors,
        "interpretation": (
            "Frontier frequency measures membership under the tested "
            "random coefficient perturbations. It is not a probability "
            "that a decision is optimal in the real world."
        ),
    }

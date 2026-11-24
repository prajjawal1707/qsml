
"""QSML 3.3 — Seeded randomized QUBO regression testing."""

import math
import random

from .qubo import QUBOEngine
from .validation import (
    solve_qubo_exact,
    validate_qubo_solution,
)


def _positive_integer(value, name, maximum=None):
    if (
        not isinstance(value, int)
        or isinstance(value, bool)
        or value < 1
    ):
        raise ValueError(f"{name} must be a positive integer")

    if maximum is not None and value > maximum:
        raise ValueError(f"{name} must not exceed {maximum}")

    return value


def make_random_qubo(num_variables, rng, density=0.35, scale=10.0):
    """Create a reproducible QUBO using a supplied random generator."""

    num_variables = _positive_integer(
        num_variables, "num_variables", maximum=16
    )

    if not 0.0 <= density <= 1.0:
        raise ValueError("density must be between 0 and 1")

    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("scale must be finite and positive")

    terms = {}

    for i in range(num_variables):
        terms[(i, i)] = rng.uniform(-scale, scale)

        for j in range(i + 1, num_variables):
            if rng.random() < density:
                terms[(i, j)] = rng.uniform(-scale, scale)

    return terms


def run_qubo_regression(
    cases=8,
    min_variables=2,
    max_variables=10,
    seed=2026,
    iterations=2000,
    restarts=3,
):
    """Run deterministic CPU-solver checks on small exact-reference QUBOs.

    The report validates every solver result and calculates the exact
    optimality gap. A non-zero gap is reported, not automatically treated
    as a failure: simulated annealing is a heuristic.
    """

    cases = _positive_integer(cases, "cases", maximum=100)
    min_variables = _positive_integer(
        min_variables, "min_variables", maximum=16
    )
    max_variables = _positive_integer(
        max_variables, "max_variables", maximum=16
    )
    iterations = _positive_integer(iterations, "iterations")
    restarts = _positive_integer(restarts, "restarts")

    if min_variables > max_variables:
        raise ValueError("min_variables cannot exceed max_variables")

    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ValueError("seed must be an integer")

    rng = random.Random(seed)
    engine = QUBOEngine()

    results = []
    errors = []

    for case_index in range(cases):
        num_variables = rng.randint(
            min_variables,
            max_variables,
        )

        terms = make_random_qubo(
            num_variables,
            rng,
        )

        exact = solve_qubo_exact(
            num_variables,
            terms,
            max_variables=16,
        )

        solver_result = engine.solve(
            num_variables,
            terms,
            iterations=iterations,
            restarts=restarts,
            seed=seed + case_index,
        )

        validation = validate_qubo_solution(
            num_variables,
            terms,
            solver_result.solution,
            reported_energy=solver_result.energy,
        )

        if not validation.valid:
            errors.append({
                "case": case_index + 1,
                "errors": validation.errors,
            })
            continue

        gap = float(validation.energy) - exact.energy

        if gap < -1e-8:
            errors.append({
                "case": case_index + 1,
                "errors": [
                    "Solver energy is below the exact reference energy"
                ],
            })
            continue

        gap = max(0.0, gap)

        results.append({
            "case": case_index + 1,
            "num_variables": num_variables,
            "num_terms": len(terms),
            "solution": list(solver_result.solution),
            "energy": float(validation.energy),
            "exact_energy": float(exact.energy),
            "optimality_gap": gap,
            "optimal": math.isclose(
                validation.energy,
                exact.energy,
                rel_tol=1e-8,
                abs_tol=1e-8,
            ),
            "evaluations": solver_result.evaluations,
        })

    gaps = [item["optimality_gap"] for item in results]

    return {
        "seed": seed,
        "requested_cases": cases,
        "completed_cases": len(results),
        "valid": len(errors) == 0 and len(results) == cases,
        "optimal_cases": sum(item["optimal"] for item in results),
        "mean_optimality_gap": (
            sum(gaps) / len(gaps) if gaps else None
        ),
        "maximum_optimality_gap": max(gaps) if gaps else None,
        "results": results,
        "errors": errors,
    }

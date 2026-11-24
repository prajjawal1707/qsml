
"""QSML 3.2 — Equal-objective-evaluation CPU/GPU benchmarking."""

import statistics
import time

from .adaptive import AdaptiveOptimizer
from .validation import validate_qubo_solution


def _positive_integer(value, name, minimum=1):
    if (
        not isinstance(value, int)
        or isinstance(value, bool)
        or value < minimum
    ):
        raise ValueError(
            f"{name} must be an integer >= {minimum}"
        )
    return value


def _summary(values):
    return {
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "minimum": min(values),
        "maximum": max(values),
        "standard_deviation": (
            statistics.stdev(values) if len(values) > 1 else 0.0
        ),
    }


def run_equal_budget_comparison(
    num_variables,
    terms,
    evaluation_budget=5000,
    repeats=3,
    seed=42,
    backends=("CPU", "GPU"),
):
    """Compare backends with the same reported objective-evaluation budget.

    For the CPU solver, one baseline evaluation and one initial random
    evaluation occur before the annealing moves. Thus:
        iterations = evaluation_budget - 2
        restarts = 1

    For the GPU random-candidate solver:
        candidates = evaluation_budget

    Equal objective-evaluation counts do not imply equal hardware work
    or equal optimization algorithms.
    """

    evaluation_budget = _positive_integer(
        evaluation_budget, "evaluation_budget", minimum=3
    )
    repeats = _positive_integer(repeats, "repeats")
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ValueError("seed must be an integer")

    if not isinstance(backends, (tuple, list)) or not backends:
        raise ValueError("backends must be a non-empty tuple or list")

    optimizer = AdaptiveOptimizer()
    available = optimizer.available_backends()

    output = {
        "evaluation_budget_per_trial": evaluation_budget,
        "repeats": repeats,
        "seed": seed,
        "equal_evaluation_budget": True,
        "comparison_note": (
            "Equal QUBO objective-evaluation counts; algorithms and "
            "hardware execution costs remain different."
        ),
        "results": {},
    }

    for requested_backend in backends:
        if not isinstance(requested_backend, str):
            raise ValueError("Backend names must be strings")

        backend = requested_backend.upper()

        if backend not in ("CPU", "GPU"):
            raise ValueError(f"Unsupported backend: {backend}")

        if backend not in available:
            output["results"][backend] = {
                "available": False,
                "reason": f"{backend} backend is unavailable",
            }
            continue

        energies = []
        runtimes = []
        evaluations = []
        methods = []
        trials = []

        for trial_index in range(repeats):
            trial_seed = seed + trial_index
            start = time.perf_counter()

            if backend == "CPU":
                result = optimizer.solve(
                    num_variables=num_variables,
                    terms=terms,
                    backend="CPU",
                    iterations=evaluation_budget - 2,
                    restarts=1,
                    seed=trial_seed,
                )
            else:
                result = optimizer.solve(
                    num_variables=num_variables,
                    terms=terms,
                    backend="GPU",
                    candidates=evaluation_budget,
                    seed=trial_seed,
                )

            wall_seconds = time.perf_counter() - start
            reported_evaluations = result["evaluations"]

            if reported_evaluations != evaluation_budget:
                raise RuntimeError(
                    f"{backend} reported {reported_evaluations} evaluations; "
                    f"expected {evaluation_budget}"
                )

            validation = validate_qubo_solution(
                num_variables,
                terms,
                result["solution"],
                reported_energy=result["energy"],
            )

            if not validation.valid:
                raise ValueError(
                    f"{backend} returned an invalid result: "
                    f"{validation.errors}"
                )

            energies.append(float(validation.energy))
            runtimes.append(float(wall_seconds))
            evaluations.append(int(reported_evaluations))
            methods.append(result.get("method"))

            trials.append({
                "trial": trial_index + 1,
                "seed": trial_seed,
                "energy": float(validation.energy),
                "elapsed_seconds": float(wall_seconds),
                "evaluations": int(reported_evaluations),
                "solution": list(result["solution"]),
            })

        output["results"][backend] = {
            "available": True,
            "method": methods[0],
            "trials": trials,
            "energy_statistics": _summary(energies),
            "runtime_statistics": _summary(runtimes),
            "best_energy": min(energies),
            "evaluations_per_trial": evaluations,
        }

    return output

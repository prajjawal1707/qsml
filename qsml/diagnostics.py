
"""QSML 3.4 — Solver diagnostics and quality reporting."""

import math

from .validation import (
    solve_qubo_exact,
    validate_qubo_solution,
)


def diagnose_solver_result(
    num_variables,
    terms,
    solver_result,
    exact_max_variables=16,
    tolerance=1e-8,
):
    """Validate a solver result and measure quality against an exact result.

    Exact analysis is deliberately limited to small QUBO instances.
    Larger problems receive a valid/invalid report without an exact gap.
    """

    if not isinstance(exact_max_variables, int) or isinstance(
        exact_max_variables, bool
    ):
        raise ValueError("exact_max_variables must be an integer")

    if exact_max_variables < 1 or exact_max_variables > 20:
        raise ValueError("exact_max_variables must be between 1 and 20")

    if isinstance(tolerance, bool) or not isinstance(
        tolerance, (int, float)
    ):
        raise ValueError("tolerance must be numeric")

    tolerance = float(tolerance)

    if not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError("tolerance must be finite and non-negative")

    if not isinstance(solver_result, dict):
        if hasattr(solver_result, "__dict__"):
            solver_result = vars(solver_result)
        else:
            raise ValueError("solver_result must be a dictionary or result object")

    solution = solver_result.get("solution")
    reported_energy = solver_result.get("energy")

    validation = validate_qubo_solution(
        num_variables,
        terms,
        solution,
        reported_energy=reported_energy,
        tolerance=tolerance,
    )

    report = {
        "valid": validation.valid,
        "status": "VALID" if validation.valid else "INVALID",
        "solution": list(solution) if solution is not None else None,
        "energy": validation.energy,
        "reported_energy": reported_energy,
        "errors": list(validation.errors),
        "backend": solver_result.get("backend"),
        "method": solver_result.get("method"),
        "evaluations": solver_result.get("evaluations"),
        "elapsed_seconds": solver_result.get("elapsed_seconds"),
        "exact_checked": False,
        "exact_energy": None,
        "optimality_gap": None,
        "optimal": None,
    }

    if not validation.valid:
        return report

    if num_variables > exact_max_variables:
        report["status"] = "VALID_EXACT_CHECK_SKIPPED"
        return report

    exact = solve_qubo_exact(
        num_variables,
        terms,
        max_variables=exact_max_variables,
    )

    gap = validation.energy - exact.energy

    if gap < -tolerance:
        report["valid"] = False
        report["status"] = "INVALID_EXACT_REFERENCE_MISMATCH"
        report["errors"].append(
            "Reported solution energy is below the exact reference energy"
        )
        return report

    gap = max(0.0, gap)

    report["exact_checked"] = True
    report["exact_energy"] = exact.energy
    report["optimality_gap"] = gap
    report["optimal"] = math.isclose(
        validation.energy,
        exact.energy,
        rel_tol=tolerance,
        abs_tol=tolerance,
    )
    report["status"] = (
        "VALID_OPTIMAL" if report["optimal"] else "VALID_FEASIBLE"
    )

    return report


def print_diagnostics(report):
    """Print a readable solver diagnostic report."""

    print("\nQSML SOLVER DIAGNOSTICS")
    print("=" * 60)

    for key in (
        "status",
        "valid",
        "backend",
        "method",
        "energy",
        "exact_checked",
        "exact_energy",
        "optimality_gap",
        "optimal",
        "evaluations",
        "elapsed_seconds",
    ):
        print(f"{key}: {report.get(key)}")

    for error in report.get("errors", []):
        print("ERROR:", error)

    print("=" * 60)

"""QSML independent QUBO validation and exact reference solver."""
from dataclasses import dataclass
from itertools import product
import math

from .qubo_contract import normalize_qubo_terms, validate_positive_int


@dataclass
class ValidationReport:
    valid: bool
    energy: float | None
    errors: list[str]


@dataclass
class ExactQUBOResult:
    solution: list[int]
    energy: float
    evaluations: int
    method: str = "EXACT_BRUTE_FORCE"


def normalize_terms(num_variables, terms):
    """Compatibility wrapper for the shared QSML QUBO contract."""
    return normalize_qubo_terms(num_variables, terms, allow_empty=True)


def evaluate_qubo(num_variables, terms, solution):
    """Evaluate E(x) = sum(Q[i,j] * x[i] * x[j])."""
    normalized = normalize_terms(num_variables, terms)

    try:
        if len(solution) != num_variables:
            raise ValueError("Solution length does not match num_variables")
    except TypeError as exc:
        raise ValueError("Solution must be a sized sequence") from exc

    for value in solution:
        if value not in (0, 1):
            raise ValueError("Solution must contain only binary values")

    try:
        energy = math.fsum(
            coefficient * solution[i] * solution[j]
            for (i, j), coefficient in normalized.items()
        )
    except OverflowError as exc:
        raise ValueError("QUBO energy overflowed") from exc

    if not math.isfinite(energy):
        raise ValueError("QUBO energy must be finite")
    return float(energy)


def validate_qubo_solution(
    num_variables,
    terms,
    solution,
    reported_energy=None,
    tolerance=1e-8,
):
    """Independently validate a solver's returned solution."""
    errors = []

    if (
        isinstance(tolerance, bool)
        or not isinstance(tolerance, (int, float))
        or not math.isfinite(tolerance)
        or tolerance < 0
    ):
        return ValidationReport(
            valid=False,
            energy=None,
            errors=["tolerance must be finite and non-negative"],
        )

    try:
        actual_energy = evaluate_qubo(num_variables, terms, solution)
    except (ValueError, TypeError) as exc:
        return ValidationReport(valid=False, energy=None, errors=[str(exc)])

    if reported_energy is not None:
        if isinstance(reported_energy, bool) or not isinstance(
            reported_energy, (int, float)
        ):
            errors.append("Reported energy must be numeric")
        elif not math.isfinite(reported_energy):
            errors.append("Reported energy must be finite")
        elif not math.isclose(
            actual_energy,
            reported_energy,
            rel_tol=tolerance,
            abs_tol=tolerance,
        ):
            errors.append(
                f"Energy mismatch: calculated={actual_energy}, "
                f"reported={reported_energy}"
            )

    return ValidationReport(
        valid=len(errors) == 0,
        energy=actual_energy,
        errors=errors,
    )


def solve_qubo_exact(
    num_variables,
    terms,
    max_variables=20,
):
    """Find the exact QUBO minimum by exhaustive enumeration."""
    validate_positive_int(num_variables, "num_variables")
    validate_positive_int(max_variables, "max_variables")

    if num_variables > max_variables:
        raise ValueError(
            f"Exact solver limited to {max_variables} variables; "
            f"received {num_variables}"
        )

    normalized = normalize_terms(num_variables, terms)
    best_solution = None
    best_energy = float("inf")
    evaluations = 0

    for candidate in product((0, 1), repeat=num_variables):
        try:
            energy = math.fsum(
                coefficient * candidate[i] * candidate[j]
                for (i, j), coefficient in normalized.items()
            )
        except OverflowError as exc:
            raise ValueError("QUBO energy overflowed") from exc

        if not math.isfinite(energy):
            raise ValueError("QUBO energy must be finite")

        evaluations += 1
        if energy < best_energy:
            best_energy = energy
            best_solution = list(candidate)

    return ExactQUBOResult(
        solution=best_solution,
        energy=float(best_energy),
        evaluations=evaluations,
    )

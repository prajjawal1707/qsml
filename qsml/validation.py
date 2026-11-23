from dataclasses import dataclass
from itertools import product
import math


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
    """Validate and normalize QUBO terms."""

    if not isinstance(num_variables, int) or isinstance(num_variables, bool):
        raise ValueError("num_variables must be an integer")

    if num_variables < 1:
        raise ValueError("num_variables must be positive")

    if not isinstance(terms, dict):
        raise ValueError("terms must be a dictionary")

    normalized = {}

    for key, coefficient in terms.items():
        if not isinstance(key, tuple) or len(key) != 2:
            raise ValueError(f"Invalid QUBO term key: {key}")

        i, j = key

        if (
            not isinstance(i, int)
            or isinstance(i, bool)
            or not isinstance(j, int)
            or isinstance(j, bool)
        ):
            raise ValueError("QUBO indices must be integers")

        if not (0 <= i < num_variables):
            raise ValueError(f"Invalid QUBO index: {i}")

        if not (0 <= j < num_variables):
            raise ValueError(f"Invalid QUBO index: {j}")

        if not isinstance(coefficient, (int, float)):
            raise ValueError("QUBO coefficients must be numeric")

        if not math.isfinite(coefficient):
            raise ValueError("QUBO coefficients must be finite")

        pair = (min(i, j), max(i, j))
        normalized[pair] = normalized.get(pair, 0.0) + float(coefficient)

    return normalized


def evaluate_qubo(num_variables, terms, solution):
    """Evaluate E(x) = sum(Q[i,j] * x[i] * x[j])."""

    normalized = normalize_terms(num_variables, terms)

    if len(solution) != num_variables:
        raise ValueError("Solution length does not match num_variables")

    for value in solution:
        if value not in (0, 1):
            raise ValueError("Solution must contain only binary values")

    energy = 0.0

    for (i, j), coefficient in normalized.items():
        energy += coefficient * solution[i] * solution[j]

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

    try:
        actual_energy = evaluate_qubo(num_variables, terms, solution)
    except (ValueError, TypeError) as exc:
        return ValidationReport(
            valid=False,
            energy=None,
            errors=[str(exc)],
        )

    if reported_energy is not None:
        if not isinstance(reported_energy, (int, float)):
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
    """
    Find the exact QUBO minimum by exhaustive enumeration.

    Warning: complexity grows exponentially with variable count.
    This is intended for small reference problems and testing.
    """

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
        energy = 0.0

        for (i, j), coefficient in normalized.items():
            energy += coefficient * candidate[i] * candidate[j]

        evaluations += 1

        if energy < best_energy:
            best_energy = energy
            best_solution = list(candidate)

    return ExactQUBOResult(
        solution=best_solution,
        energy=float(best_energy),
        evaluations=evaluations,
    )

"""Independent exhaustive validation of linear-constraint QUBO encodings."""
from dataclasses import dataclass
from itertools import product
import math


@dataclass
class ConstraintValidationReport:
    constraint: str
    valid: bool
    checked_assignments: int
    original_assignments: int
    feasible_assignments: int
    infeasible_assignments: int
    baseline_energy: float
    max_feasible_min_energy: float | None
    min_infeasible_min_energy: float | None
    errors: list[str]


def _number(value, label, *, positive=False, non_negative=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{label} must be finite")
    if positive and value <= 0:
        raise ValueError(f"{label} must be positive")
    if non_negative and value < 0:
        raise ValueError(f"{label} must be non-negative")
    return value


def _validate_coefficients(coefficients, num_variables):
    if not isinstance(coefficients, dict) or not coefficients:
        raise ValueError("coefficients must be a non-empty dictionary")
    checked = {}
    for index, value in coefficients.items():
        if not isinstance(index, int) or isinstance(index, bool):
            raise ValueError("coefficient indices must be integers")
        if not 0 <= index < num_variables:
            raise ValueError(f"coefficient index {index} is out of range")
        checked[index] = _number(value, "coefficient")
    return checked


def _validated_encoding(encoding, expected_kind, max_variables):
    if not isinstance(encoding, dict):
        raise ValueError("encoding must be a dictionary")
    if encoding.get("constraint") != expected_kind:
        raise ValueError(f"encoding must be a {expected_kind} constraint")
    total = encoding.get("num_variables")
    if (
        not isinstance(total, int)
        or isinstance(total, bool)
        or total < 1
    ):
        raise ValueError("encoding num_variables must be a positive integer")
    if (
        not isinstance(max_variables, int)
        or isinstance(max_variables, bool)
        or max_variables < 1
    ):
        raise ValueError("max_variables must be a positive integer")
    if total > max_variables:
        raise ValueError(
            f"Exhaustive validation limited to {max_variables} variables; "
            f"received {total}"
        )

    terms = encoding.get("terms")
    if not isinstance(terms, dict):
        raise ValueError("encoding terms must be a dictionary")
    checked_terms = {}
    for key, coefficient in terms.items():
        if not isinstance(key, tuple) or len(key) != 2:
            raise ValueError(f"Invalid QUBO term key: {key!r}")
        i, j = key
        if (
            not isinstance(i, int) or isinstance(i, bool)
            or not isinstance(j, int) or isinstance(j, bool)
            or not (0 <= i < total and 0 <= j < total)
        ):
            raise ValueError(f"Invalid QUBO indices: {key!r}")
        checked_terms[key] = _number(coefficient, f"QUBO coefficient {key}")
    penalty = _number(encoding.get("penalty"), "penalty", positive=True)
    return total, checked_terms, penalty


def _direct_energy(num_variables, terms, assignment):
    if len(assignment) != num_variables or any(bit not in (0, 1) for bit in assignment):
        raise ValueError("Assignment must contain one binary value per variable")
    try:
        result = math.fsum(
            coefficient * assignment[i] * assignment[j]
            for (i, j), coefficient in terms.items()
        )
    except OverflowError as exc:
        raise ValueError("QUBO energy overflowed during validation") from exc
    if not math.isfinite(result):
        raise ValueError("QUBO energy is not finite during validation")
    return result


def _close(actual, expected, tolerance):
    return math.isclose(
        actual, expected, rel_tol=tolerance, abs_tol=tolerance
    )


def validate_equality_encoding(
    encoding,
    coefficients,
    rhs,
    tolerance=1e-9,
    max_variables=16,
):
    """Exhaustively verify the generated equality penalty polynomial.

    For L(x)=sum(a_i*x_i), the generated QUBO should equal:
        penalty * ((L(x)-rhs)**2 - rhs**2)
    because the constant penalty*rhs**2 is omitted from the QUBO.
    """
    total, terms, penalty = _validated_encoding(
        encoding, "equality", max_variables
    )
    checked = _validate_coefficients(coefficients, total)
    rhs = _number(rhs, "rhs")
    tolerance = _number(tolerance, "tolerance", non_negative=True)
    baseline = -penalty * rhs * rhs
    errors = []
    feasible = 0
    infeasible = 0
    feasible_energies = []
    infeasible_energies = []
    checked_assignments = 0

    for assignment in product((0, 1), repeat=total):
        lhs = math.fsum(value * assignment[index] for index, value in checked.items())
        actual = _direct_energy(total, terms, assignment)
        expected = penalty * ((lhs - rhs) ** 2 - rhs ** 2)
        checked_assignments += 1
        if not _close(actual, expected, tolerance):
            errors.append(
                f"Assignment {assignment}: QUBO={actual}, expected={expected}"
            )
        if abs(lhs - rhs) <= tolerance:
            feasible += 1
            feasible_energies.append(actual)
        else:
            infeasible += 1
            infeasible_energies.append(actual)

    if feasible and any(
        not _close(value, baseline, tolerance) for value in feasible_energies
    ):
        errors.append("A feasible equality assignment does not reach baseline energy")
    if infeasible and any(
        value <= baseline + tolerance for value in infeasible_energies
    ):
        errors.append("An infeasible equality assignment is not penalized above baseline")

    return ConstraintValidationReport(
        constraint="equality",
        valid=not errors,
        checked_assignments=checked_assignments,
        original_assignments=checked_assignments,
        feasible_assignments=feasible,
        infeasible_assignments=infeasible,
        baseline_energy=baseline,
        max_feasible_min_energy=max(feasible_energies) if feasible_energies else None,
        min_infeasible_min_energy=min(infeasible_energies) if infeasible_energies else None,
        errors=errors,
    )


def validate_at_most_encoding(
    encoding,
    coefficients,
    limit,
    tolerance=1e-9,
    max_variables=16,
):
    """Exhaustively verify the at-most penalty and its binary slack encoding.

    For A(x)=sum(a_i*x_i), slack S(s), and limit b, the QUBO should equal:
        penalty * ((A(x)+S(s)-b)**2 - b**2)

    For each original assignment, minimizing over slack bits must reach the
    baseline exactly when A(x) <= b.
    """
    total, terms, penalty = _validated_encoding(
        encoding, "at_most", max_variables
    )
    original = encoding.get("original_variables")
    slack_count = encoding.get("slack_variables")
    weights = encoding.get("slack_weights")

    if (
        not isinstance(original, int) or isinstance(original, bool)
        or original < 1
    ):
        raise ValueError("original_variables must be a positive integer")
    if (
        not isinstance(slack_count, int) or isinstance(slack_count, bool)
        or slack_count < 0
    ):
        raise ValueError("slack_variables must be a non-negative integer")
    if total != original + slack_count:
        raise ValueError("total variables do not match original plus slack variables")
    if not isinstance(weights, list) or len(weights) != slack_count:
        raise ValueError("slack_weights do not match slack_variables")
    checked = _validate_coefficients(coefficients, original)
    for value in checked.values():
        if value < 0 or not float(value).is_integer():
            raise ValueError("at_most coefficients must be non-negative integers")
    limit_number = _number(limit, "limit", non_negative=True)
    if not limit_number.is_integer():
        raise ValueError("limit must be a non-negative integer")
    limit = int(limit_number)
    if encoding.get("limit") != limit:
        raise ValueError("encoding limit does not match requested limit")
    checked_weights = []
    for weight in weights:
        if not isinstance(weight, int) or isinstance(weight, bool) or weight < 1:
            raise ValueError("slack weights must be positive integers")
        checked_weights.append(weight)
    if sum(checked_weights) < limit:
        raise ValueError("slack weights cannot represent every residual up to limit")

    tolerance = _number(tolerance, "tolerance", non_negative=True)
    baseline = -penalty * limit * limit
    errors = []
    checked_assignments = 0
    feasible_count = 0
    infeasible_count = 0
    feasible_minima = []
    infeasible_minima = []

    for original_bits in product((0, 1), repeat=original):
        activity = math.fsum(
            value * original_bits[index] for index, value in checked.items()
        )
        best_energy = float("inf")

        for slack_bits in product((0, 1), repeat=slack_count):
            assignment = original_bits + slack_bits
            slack_value = sum(
                weight * bit for weight, bit in zip(checked_weights, slack_bits)
            )
            actual = _direct_energy(total, terms, assignment)
            expected = penalty * (
                (activity + slack_value - limit) ** 2 - limit ** 2
            )
            checked_assignments += 1
            if not _close(actual, expected, tolerance):
                errors.append(
                    f"Assignment {assignment}: QUBO={actual}, expected={expected}"
                )
            best_energy = min(best_energy, actual)

        expected_minimum = (
            baseline if activity <= limit
            else baseline + penalty * (activity - limit) ** 2
        )
        if not _close(best_energy, expected_minimum, tolerance):
            errors.append(
                f"Original assignment {original_bits}: minimum slack energy "
                f"{best_energy}, expected {expected_minimum}"
            )

        if activity <= limit:
            feasible_count += 1
            feasible_minima.append(best_energy)
        else:
            infeasible_count += 1
            infeasible_minima.append(best_energy)

    return ConstraintValidationReport(
        constraint="at_most",
        valid=not errors,
        checked_assignments=checked_assignments,
        original_assignments=2 ** original,
        feasible_assignments=feasible_count,
        infeasible_assignments=infeasible_count,
        baseline_energy=baseline,
        max_feasible_min_energy=max(feasible_minima) if feasible_minima else None,
        min_infeasible_min_energy=min(infeasible_minima) if infeasible_minima else None,
        errors=errors,
    )

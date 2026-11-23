"""QSML constraint-to-QUBO conversion with finite-arithmetic safeguards."""
import math


def _finite_number(value, label):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    try:
        value = float(value)
    except (OverflowError, ValueError) as exc:
        raise ValueError(f"{label} must be representable as a finite float") from exc
    if not math.isfinite(value):
        raise ValueError(f"{label} must be finite")
    return value


def _accumulate_term(terms, key, increment):
    if not math.isfinite(increment):
        raise ValueError(f"Constraint penalty term {key} is not finite")
    combined = terms.get(key, 0.0) + increment
    if not math.isfinite(combined):
        raise ValueError(f"Combined constraint penalty term {key} is not finite")
    if combined == 0.0:
        terms.pop(key, None)
    else:
        terms[key] = combined


def add_terms(first, second):
    """Validate and add two QUBO dictionaries, combining duplicate terms."""
    if not isinstance(first, dict) or not isinstance(second, dict):
        raise ValueError("Both term collections must be dictionaries")

    merged = {}
    for collection in (first, second):
        for key, raw_value in collection.items():
            if (
                not isinstance(key, tuple)
                or len(key) != 2
                or any(
                    not isinstance(index, int) or isinstance(index, bool)
                    or index < 0
                    for index in key
                )
            ):
                raise ValueError(f"Invalid QUBO term key: {key!r}")
            value = _finite_number(raw_value, f"coefficient for {key}")
            _accumulate_term(merged, key, value)
    return merged


class ConstraintBuilder:
    """Build penalties for binary linear constraints.

    QUBO convention:
        E(x) = sum(c[i,j] * x[i] * x[j])

    The constant term of a squared penalty is intentionally omitted.
    Omitting it shifts every assignment by the same constant and does not
    change the minimizing assignments.
    """

    @staticmethod
    def _square_penalty(num_variables, coefficients, rhs, penalty):
        penalty = _finite_number(penalty, "penalty")
        rhs = _finite_number(rhs, "rhs")
        if penalty <= 0:
            raise ValueError("penalty must be positive")

        terms = {}
        entries = sorted(coefficients.items())

        for index, coefficient in entries:
            coefficient = _finite_number(coefficient, "coefficient")
            diagonal = penalty * (
                coefficient * coefficient - 2.0 * rhs * coefficient
            )
            _accumulate_term(terms, (index, index), diagonal)

        for position, (i, ci) in enumerate(entries):
            for j, cj in entries[position + 1:]:
                cross = 2.0 * penalty * ci * cj
                _accumulate_term(terms, (i, j), cross)

        return terms

    @staticmethod
    def _validate_coefficients(coefficients, num_variables):
        if (
            not isinstance(num_variables, int)
            or isinstance(num_variables, bool)
        ):
            raise ValueError("num_variables must be an integer")
        if num_variables < 1:
            raise ValueError("num_variables must be positive")
        if not isinstance(coefficients, dict) or not coefficients:
            raise ValueError("coefficients must be a non-empty dictionary")

        checked = {}
        for index, coefficient in coefficients.items():
            if not isinstance(index, int) or isinstance(index, bool):
                raise ValueError("variable indices must be integers")
            if not 0 <= index < num_variables:
                raise ValueError(f"variable index {index} is out of range")
            checked[index] = _finite_number(coefficient, "coefficient")
        return checked

    def equality(self, num_variables, coefficients, rhs, penalty=10.0):
        checked = self._validate_coefficients(coefficients, num_variables)
        checked_rhs = _finite_number(rhs, "rhs")
        checked_penalty = _finite_number(penalty, "penalty")
        if checked_penalty <= 0:
            raise ValueError("penalty must be positive")
        return {
            "num_variables": num_variables,
            "terms": self._square_penalty(
                num_variables, checked, checked_rhs, checked_penalty
            ),
            "constraint": "equality",
            "penalty": checked_penalty,
        }

    def at_most(self, num_variables, coefficients, limit, penalty=10.0):
        """Encode sum(a_i*x_i) <= limit with non-negative binary slack bits.

        Coefficients and limit must be non-negative integers. Slack bits
        represent every integer residual from zero through limit.
        """
        checked = self._validate_coefficients(coefficients, num_variables)
        limit_value = _finite_number(limit, "limit")
        if limit_value < 0 or not limit_value.is_integer():
            raise ValueError("limit must be a non-negative integer")
        for value in checked.values():
            if value < 0 or not value.is_integer():
                raise ValueError(
                    "at_most coefficients must be non-negative integers"
                )

        checked_penalty = _finite_number(penalty, "penalty")
        if checked_penalty <= 0:
            raise ValueError("penalty must be positive")

        limit_int = int(limit_value)
        slack_weights = []
        power = 1
        while sum(slack_weights) < limit_int:
            slack_weights.append(power)
            power *= 2

        total_variables = num_variables + len(slack_weights)
        expression = dict(checked)
        for offset, weight in enumerate(slack_weights):
            try:
                expression[num_variables + offset] = float(weight)
            except OverflowError as exc:
                raise ValueError("Slack weight is too large for QUBO arithmetic") from exc

        terms = self._square_penalty(
            total_variables, expression, float(limit_int), checked_penalty
        )

        return {
            "num_variables": total_variables,
            "original_variables": num_variables,
            "slack_variables": len(slack_weights),
            "slack_weights": slack_weights,
            "terms": terms,
            "constraint": "at_most",
            "limit": limit_int,
            "penalty": checked_penalty,
        }

    @staticmethod
    def combine(objective_terms, constraint_terms):
        return add_terms(objective_terms, constraint_terms)

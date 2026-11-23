"""QSML 3.1.1 shared QUBO input contract."""
import math


def normalize_qubo_terms(num_variables, terms, *, allow_empty=False):
    """Validate and canonicalize QUBO terms using the QSML convention.

    E(x) = sum(c[i,j] * x[i] * x[j]), with canonical keys i <= j.
    """
    if not isinstance(num_variables, int) or isinstance(num_variables, bool):
        raise ValueError("num_variables must be an integer")
    if num_variables < 1:
        raise ValueError("num_variables must be positive")
    if not isinstance(terms, dict):
        raise ValueError("terms must be a dictionary")
    if not terms and not allow_empty:
        raise ValueError("At least one QUBO term is required")

    normalized = {}
    for key, coefficient in terms.items():
        if not isinstance(key, tuple) or len(key) != 2:
            raise ValueError(f"Invalid QUBO term key: {key!r}")

        i, j = key
        if (
            not isinstance(i, int) or isinstance(i, bool)
            or not isinstance(j, int) or isinstance(j, bool)
        ):
            raise ValueError("QUBO indices must be integers")

        if not (0 <= i < num_variables and 0 <= j < num_variables):
            raise ValueError(f"QUBO index out of range: {key!r}")

        if isinstance(coefficient, bool) or not isinstance(coefficient, (int, float)):
            raise ValueError("QUBO coefficients must be numeric and not Boolean")

        coefficient = float(coefficient)
        if not math.isfinite(coefficient):
            raise ValueError("QUBO coefficients must be finite")

        pair = (min(i, j), max(i, j))
        combined = normalized.get(pair, 0.0) + coefficient
        if not math.isfinite(combined):
            raise ValueError(f"Combined coefficient for {pair} is not finite")
        normalized[pair] = combined

    if not normalized and not allow_empty:
        raise ValueError("At least one QUBO term is required")
    return normalized


def validate_positive_int(value, label):
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError(f"{label} must be a positive integer")
    return value


def validate_seed(seed):
    if seed is not None and (
        not isinstance(seed, int) or isinstance(seed, bool)
    ):
        raise ValueError("seed must be an integer or None")
    return seed

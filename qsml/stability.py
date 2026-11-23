"""QSML 3.1.4 — Numerical stability and stress-testing utilities."""

import math
import random

from .qubo_contract import normalize_qubo_terms


def _positive_integer(value, name):
    if (
        not isinstance(value, int)
        or isinstance(value, bool)
        or value < 1
    ):
        raise ValueError(f"{name} must be a positive integer")

    return value


def _finite_number(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")

    try:
        value = float(value)
    except (OverflowError, ValueError) as exc:
        raise ValueError(f"{name} must be finite") from exc

    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")

    return value


def stable_qubo_energy(num_variables, terms, solution):
    """Calculate a QUBO energy using accurate floating-point summation."""

    normalized = normalize_qubo_terms(num_variables, terms)

    if isinstance(solution, (str, bytes)):
        raise ValueError("Solution must be a binary sequence")

    try:
        bits = list(solution)
    except TypeError as exc:
        raise ValueError("Solution must be iterable") from exc

    if len(bits) != num_variables:
        raise ValueError("Solution length does not match num_variables")

    for bit in bits:
        if bit not in (0, 1):
            raise ValueError("Solution must contain only binary values")

    try:
        energy = math.fsum(
            coefficient * bits[i] * bits[j]
            for (i, j), coefficient in normalized.items()
        )
    except (OverflowError, ValueError) as exc:
        raise ValueError("QUBO energy overflowed or became invalid") from exc

    if not math.isfinite(energy):
        raise ValueError("QUBO energy must be finite")

    return float(energy)


def energies_close(
    actual,
    expected,
    relative_tolerance=1e-10,
    absolute_tolerance=1e-12,
):
    """Compare two energies using absolute and relative tolerances."""

    actual = _finite_number(actual, "actual")
    expected = _finite_number(expected, "expected")

    relative_tolerance = _finite_number(
        relative_tolerance,
        "relative_tolerance",
    )

    absolute_tolerance = _finite_number(
        absolute_tolerance,
        "absolute_tolerance",
    )

    if relative_tolerance < 0 or absolute_tolerance < 0:
        raise ValueError("Tolerances must be non-negative")

    return math.isclose(
        actual,
        expected,
        rel_tol=relative_tolerance,
        abs_tol=absolute_tolerance,
    )


def compare_energy_vectors(
    reference,
    measured,
    relative_tolerance=1e-10,
    absolute_tolerance=1e-12,
):
    """Compare two vectors of energies and summarize numerical error."""

    relative_tolerance = _finite_number(
        relative_tolerance,
        "relative_tolerance",
    )

    absolute_tolerance = _finite_number(
        absolute_tolerance,
        "absolute_tolerance",
    )

    if relative_tolerance < 0 or absolute_tolerance < 0:
        raise ValueError("Tolerances must be non-negative")

    if isinstance(reference, (str, bytes)):
        raise ValueError("Reference energies must be iterable")

    if isinstance(measured, (str, bytes)):
        raise ValueError("Measured energies must be iterable")

    try:
        reference_values = [
            _finite_number(value, "reference energy")
            for value in reference
        ]

        measured_values = [
            _finite_number(value, "measured energy")
            for value in measured
        ]
    except TypeError as exc:
        raise ValueError("Energy inputs must be iterable") from exc

    if not reference_values:
        raise ValueError("At least one energy is required")

    if len(reference_values) != len(measured_values):
        raise ValueError("Energy vectors must have equal lengths")

    absolute_errors = []
    relative_errors = []
    errors = []

    for index, (expected, actual) in enumerate(
        zip(reference_values, measured_values)
    ):
        absolute_error = abs(actual - expected)

        scale = max(abs(actual), abs(expected))

        relative_error = (
            absolute_error / scale if scale > 0 else 0.0
        )

        absolute_errors.append(absolute_error)
        relative_errors.append(relative_error)

        if not math.isclose(
            actual,
            expected,
            rel_tol=relative_tolerance,
            abs_tol=absolute_tolerance,
        ):
            errors.append(
                f"Index {index}: reference={expected}, "
                f"measured={actual}, absolute error={absolute_error}"
            )

    rms_error = math.hypot(*absolute_errors) / math.sqrt(
        len(absolute_errors)
    )

    return {
        "valid": len(errors) == 0,
        "sample_count": len(reference_values),
        "max_absolute_error": max(absolute_errors),
        "max_relative_error": max(relative_errors),
        "rms_absolute_error": rms_error,
        "errors": errors,
    }


def generate_stress_qubo(
    num_variables,
    seed=42,
    density=0.10,
    coefficient_scale=1000.0,
):
    """Generate a deterministic random QUBO for stress testing.

    Every variable receives a diagonal term. Off-diagonal terms are
    included according to density.

    This generates test data; it does not run an optimizer.
    """

    num_variables = _positive_integer(
        num_variables,
        "num_variables",
    )

    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ValueError("seed must be an integer")

    density = _finite_number(density, "density")
    coefficient_scale = _finite_number(
        coefficient_scale,
        "coefficient_scale",
    )

    if not 0.0 <= density <= 1.0:
        raise ValueError("density must be between 0 and 1")

    if coefficient_scale <= 0:
        raise ValueError("coefficient_scale must be positive")

    rng = random.Random(seed)
    terms = {}

    for i in range(num_variables):
        terms[(i, i)] = rng.uniform(
            -coefficient_scale,
            coefficient_scale,
        )

        for j in range(i + 1, num_variables):
            if rng.random() < density:
                terms[(i, j)] = rng.uniform(
                    -coefficient_scale,
                    coefficient_scale,
                )

    return terms


def estimate_gpu_candidate_memory_bytes(
    num_candidates,
    num_variables,
    bytes_per_variable=4,
    bytes_per_energy=8,
):
    """Estimate core GPU candidate and energy array payload.

    Default representation:
        Candidate matrix: int32, 4 bytes per variable.
        Energy vector: float64, 8 bytes per candidate.

    This excludes CUDA context, CuPy allocator overhead, temporary
    allocations, other arrays, and memory used by other processes.
    """

    num_candidates = _positive_integer(
        num_candidates,
        "num_candidates",
    )

    num_variables = _positive_integer(
        num_variables,
        "num_variables",
    )

    bytes_per_variable = _positive_integer(
        bytes_per_variable,
        "bytes_per_variable",
    )

    bytes_per_energy = _positive_integer(
        bytes_per_energy,
        "bytes_per_energy",
    )

    candidate_bytes = (
        num_candidates
        * num_variables
        * bytes_per_variable
    )

    energy_bytes = (
        num_candidates
        * bytes_per_energy
    )

    return candidate_bytes + energy_bytes


def check_gpu_memory_budget(
    num_candidates,
    num_variables,
    max_bytes,
    bytes_per_variable=4,
    bytes_per_energy=8,
):
    """Check whether core GPU array payload fits a supplied budget."""

    max_bytes = _positive_integer(max_bytes, "max_bytes")

    estimated_bytes = estimate_gpu_candidate_memory_bytes(
        num_candidates=num_candidates,
        num_variables=num_variables,
        bytes_per_variable=bytes_per_variable,
        bytes_per_energy=bytes_per_energy,
    )

    return {
        "fits": estimated_bytes <= max_bytes,
        "estimated_bytes": estimated_bytes,
        "budget_bytes": max_bytes,
        "headroom_bytes": max_bytes - estimated_bytes,
    }
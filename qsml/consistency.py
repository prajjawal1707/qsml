"""QSML 3.1.3 — CPU/GPU QUBO numerical consistency."""

from dataclasses import dataclass
import math

from .qubo_contract import normalize_qubo_terms


@dataclass
class NumericalConsistencyReport:
    valid: bool
    sample_count: int
    absolute_tolerance: float
    relative_tolerance: float
    max_absolute_error: float
    max_relative_error: float
    cpu_energies: list[float]
    gpu_energies: list[float]
    errors: list[str]


def _validate_candidates(num_variables, candidates):
    """Validate and normalize a collection of binary assignments."""

    if isinstance(candidates, (str, bytes)):
        raise ValueError("Candidates must be a collection of binary assignments")

    try:
        rows = list(candidates)
    except TypeError as exc:
        raise ValueError("Candidates must be iterable") from exc

    if not rows:
        raise ValueError("At least one candidate is required")

    checked = []

    for row_number, row in enumerate(rows):
        if isinstance(row, (str, bytes)):
            raise ValueError(
                f"Candidate {row_number} must be a binary sequence"
            )

        try:
            bits = list(row)
        except TypeError as exc:
            raise ValueError(
                f"Candidate {row_number} must be iterable"
            ) from exc

        if len(bits) != num_variables:
            raise ValueError(
                f"Candidate {row_number} has {len(bits)} bits; "
                f"expected {num_variables}"
            )

        normalized_bits = []

        for bit in bits:
            if bit is True:
                normalized_bits.append(1)
            elif bit is False:
                normalized_bits.append(0)
            elif bit in (0, 1):
                normalized_bits.append(int(bit))
            else:
                raise ValueError(
                    f"Candidate {row_number} contains a non-binary value"
                )

        checked.append(normalized_bits)

    return checked


def _validate_tolerance(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")

    try:
        value = float(value)
    except (OverflowError, ValueError) as exc:
        raise ValueError(f"{name} must be finite") from exc

    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and non-negative")

    return value


def _calculate_energy(normalized_terms, solution):
    """Calculate one QUBO energy using accurate CPU summation."""

    try:
        energy = math.fsum(
            coefficient * solution[i] * solution[j]
            for (i, j), coefficient in normalized_terms.items()
        )
    except OverflowError as exc:
        raise ValueError("CPU QUBO energy overflowed") from exc

    if not math.isfinite(energy):
        raise ValueError("CPU QUBO energy is not finite")

    return float(energy)


def evaluate_qubo_candidates_cpu(
    num_variables,
    terms,
    candidates,
):
    """Evaluate the same supplied candidate assignments on the CPU."""

    normalized_terms = normalize_qubo_terms(num_variables, terms)

    rows = _validate_candidates(
        num_variables,
        candidates,
    )

    return [
        _calculate_energy(normalized_terms, row)
        for row in rows
    ]


def evaluate_qubo_candidates_gpu(
    num_variables,
    terms,
    candidates,
):
    """Evaluate the same supplied candidate assignments using CuPy."""

    normalized_terms = normalize_qubo_terms(num_variables, terms)

    rows = _validate_candidates(
        num_variables,
        candidates,
    )

    try:
        import cupy as cp
    except ImportError as exc:
        raise RuntimeError("CuPy is not installed") from exc

    try:
        if cp.cuda.runtime.getDeviceCount() < 1:
            raise RuntimeError("No CUDA GPU detected")
    except RuntimeError:
        raise
    except Exception as exc:
        raise RuntimeError("CUDA GPU is unavailable") from exc

    gpu_candidates = cp.asarray(rows, dtype=cp.float64)

    energies = cp.zeros(
        len(rows),
        dtype=cp.float64,
    )

    for (i, j), coefficient in normalized_terms.items():
        energies += (
            coefficient
            * gpu_candidates[:, i]
            * gpu_candidates[:, j]
        )

    cp.cuda.get_current_stream().synchronize()

    if not bool(cp.all(cp.isfinite(energies)).get()):
        raise ValueError("GPU QUBO energies contain non-finite values")

    results = cp.asnumpy(energies).tolist()

    return [float(value) for value in results]


def compare_cpu_gpu_energies(
    num_variables,
    terms,
    candidates,
    absolute_tolerance=1e-10,
    relative_tolerance=1e-10,
):
    """Compare CPU and GPU energies for identical binary assignments."""

    absolute_tolerance = _validate_tolerance(
        absolute_tolerance,
        "absolute_tolerance",
    )

    relative_tolerance = _validate_tolerance(
        relative_tolerance,
        "relative_tolerance",
    )

    rows = _validate_candidates(
        num_variables,
        candidates,
    )

    cpu_energies = evaluate_qubo_candidates_cpu(
        num_variables,
        terms,
        rows,
    )

    gpu_energies = evaluate_qubo_candidates_gpu(
        num_variables,
        terms,
        rows,
    )

    if len(cpu_energies) != len(gpu_energies):
        raise RuntimeError("CPU and GPU returned different result counts")

    errors = []
    max_absolute_error = 0.0
    max_relative_error = 0.0

    for index, (cpu_value, gpu_value) in enumerate(
        zip(cpu_energies, gpu_energies)
    ):
        absolute_error = abs(cpu_value - gpu_value)

        scale = max(abs(cpu_value), abs(gpu_value))

        relative_error = (
            absolute_error / scale if scale != 0 else 0.0
        )

        max_absolute_error = max(
            max_absolute_error,
            absolute_error,
        )

        max_relative_error = max(
            max_relative_error,
            relative_error,
        )

        if not math.isclose(
            cpu_value,
            gpu_value,
            rel_tol=relative_tolerance,
            abs_tol=absolute_tolerance,
        ):
            errors.append(
                f"Candidate {index}: CPU={cpu_value}, "
                f"GPU={gpu_value}, absolute error={absolute_error}"
            )

    return NumericalConsistencyReport(
        valid=len(errors) == 0,
        sample_count=len(rows),
        absolute_tolerance=absolute_tolerance,
        relative_tolerance=relative_tolerance,
        max_absolute_error=max_absolute_error,
        max_relative_error=max_relative_error,
        cpu_energies=cpu_energies,
        gpu_energies=gpu_energies,
        errors=errors,
    )
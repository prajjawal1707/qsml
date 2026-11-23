"""Memory estimates and conservative state-vector planning."""
from __future__ import annotations
import math

BYTES_PER_COMPLEX128 = 16


def state_vector_bytes(num_qubits: int) -> int:
    if not isinstance(num_qubits, int) or num_qubits < 1:
        raise ValueError("num_qubits must be a positive integer")
    return (1 << num_qubits) * BYTES_PER_COMPLEX128


def format_bytes(size: int) -> str:
    value = float(size)
    for unit in ("B", "KB", "MB", "GB", "TB", "PB"):
        if value < 1024 or unit == "PB":
            return f"{value:.2f} {unit}"
        value /= 1024
    return f"{value:.2f} PB"


def recommend_qubit_limit(
    available_ram_bytes: int,
    budget_fraction: float = 0.20,
    working_copies: int = 3,
    hard_limit: int = 24,
) -> int:
    """Conservative upper bound accounting for temporary working arrays."""
    if available_ram_bytes < 0:
        raise ValueError("available_ram_bytes cannot be negative")
    if not 0 < budget_fraction <= 1:
        raise ValueError("budget_fraction must be in (0, 1]")
    if working_copies < 1:
        raise ValueError("working_copies must be >= 1")
    budget = int(available_ram_bytes * budget_fraction)
    bytes_per_basis = BYTES_PER_COMPLEX128 * working_copies
    if budget < bytes_per_basis:
        return 0
    estimate = int(math.floor(math.log2(budget / bytes_per_basis)))
    return max(0, min(estimate, hard_limit))

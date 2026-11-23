"""State-vector quantum simulator using ordinary classical memory.

This is mathematical simulation, not physical quantum hardware.
Qubit indexing is little-endian: q0 is the least-significant bit.
"""
from __future__ import annotations
import numpy as np
import psutil
from .resources import state_vector_bytes


class QuantumRegister:
    """An n-qubit pure state stored as 2**n complex amplitudes."""

    def __init__(self, num_qubits: int, memory_fraction: float = 0.20):
        if not isinstance(num_qubits, int) or num_qubits < 1:
            raise ValueError("num_qubits must be a positive integer")
        if not 0 < memory_fraction <= 1:
            raise ValueError("memory_fraction must be in (0, 1]")
        required = state_vector_bytes(num_qubits)
        # Reserve room for the state plus two similarly sized working arrays.
        estimated_working = required * 3
        available = psutil.virtual_memory().available
        if estimated_working > int(available * memory_fraction):
            raise MemoryError(
                f"{num_qubits} qubits need about {estimated_working / (1024**3):.2f} "
                f"GiB for state + working space; current safe budget is "
                f"{available * memory_fraction / (1024**3):.2f} GiB. "
                "Close memory-heavy apps or use fewer qubits."
            )
        self.num_qubits = num_qubits
        self.state = np.zeros(1 << num_qubits, dtype=np.complex128)
        self.state[0] = 1.0 + 0.0j

    def _check_qubit(self, qubit: int) -> None:
        if not isinstance(qubit, int) or not 0 <= qubit < self.num_qubits:
            raise ValueError(f"qubit must be in [0, {self.num_qubits - 1}]")

    def _apply_single_qubit(self, qubit: int, matrix: np.ndarray) -> None:
        self._check_qubit(qubit)
        if matrix.shape != (2, 2):
            raise ValueError("gate matrix must be 2x2")
        stride, block = 1 << qubit, 2 << qubit
        for base in range(0, self.state.size, block):
            for offset in range(stride):
                i0, i1 = base + offset, base + offset + stride
                a0, a1 = self.state[i0], self.state[i1]
                self.state[i0] = matrix[0, 0] * a0 + matrix[0, 1] * a1
                self.state[i1] = matrix[1, 0] * a0 + matrix[1, 1] * a1

    def h(self, qubit: int):
        s = 1 / np.sqrt(2)
        self._apply_single_qubit(qubit, np.array([[s, s], [s, -s]], dtype=complex))
        return self

    def x(self, qubit: int):
        self._apply_single_qubit(qubit, np.array([[0, 1], [1, 0]], dtype=complex))
        return self

    def z(self, qubit: int):
        self._apply_single_qubit(qubit, np.array([[1, 0], [0, -1]], dtype=complex))
        return self

    def y(self, qubit: int):
        self._apply_single_qubit(qubit, np.array([[0, -1j], [1j, 0]], dtype=complex))
        return self

    def cnot(self, control: int, target: int):
        self._check_qubit(control)
        self._check_qubit(target)
        if control == target:
            raise ValueError("control and target must be different")
        cm, tm = 1 << control, 1 << target
        for i in range(self.state.size):
            if (i & cm) and not (i & tm):
                j = i | tm
                self.state[i], self.state[j] = self.state[j], self.state[i]
        return self

    def probabilities(self) -> np.ndarray:
        return np.abs(self.state) ** 2

    def measure(self, shots: int = 1, seed: int | None = None) -> dict[str, int]:
        if not isinstance(shots, int) or shots < 1:
            raise ValueError("shots must be a positive integer")
        probs = self.probabilities()
        if not np.isclose(float(probs.sum()), 1.0, atol=1e-10):
            raise RuntimeError("State is not normalized")
        samples = np.random.default_rng(seed).choice(self.state.size, size=shots, p=probs)
        counts: dict[str, int] = {}
        for value in samples:
            key = format(int(value), f"0{self.num_qubits}b")
            counts[key] = counts.get(key, 0) + 1
        return dict(sorted(counts.items()))

    def state_description(self, tolerance: float = 1e-10):
        return [
            (format(i, f"0{self.num_qubits}b"), complex(a))
            for i, a in enumerate(self.state) if abs(a) > tolerance
        ]

    def validate(self, atol: float = 1e-10) -> bool:
        return bool(np.isclose(float(self.probabilities().sum()), 1.0, atol=atol))

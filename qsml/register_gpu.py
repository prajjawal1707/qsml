
"""QSML 1.0 - GPU state-vector simulator."""

import cupy as cp
import numpy as np


class GPUQuantumRegister:

    def __init__(self, num_qubits: int):
        if not isinstance(num_qubits, int) or num_qubits < 1:
            raise ValueError("Qubit count must be positive")

        if num_qubits > 20:
            raise MemoryError(
                "GPU simulator currently limits registers to 20 qubits"
            )

        self.num_qubits = num_qubits
        self.size = 2 ** num_qubits

        free_bytes, total_bytes = cp.cuda.runtime.memGetInfo()

        # Reserve room for three state-vector-sized allocations.
        required_bytes = self.size * 16 * 3

        if required_bytes > free_bytes * 0.5:
            raise MemoryError(
                "Insufficient free GPU memory for safe allocation"
            )

        self.state = cp.zeros(self.size, dtype=cp.complex128)
        self.state[0] = 1.0

    def x(self, qubit: int):
        """Apply Pauli-X gate."""
        self._validate_qubit(qubit)

        mask = 1 << qubit
        zero = cp.arange(self.size)[
            (cp.arange(self.size) & mask) == 0
        ]
        one = zero | mask

        a = self.state[zero].copy()
        b = self.state[one].copy()

        self.state[zero] = b
        self.state[one] = a

        return self

    def h(self, qubit: int):
        """Apply Hadamard gate."""
        self._validate_qubit(qubit)

        mask = 1 << qubit
        indices = cp.arange(self.size)
        zero = indices[(indices & mask) == 0]
        one = zero | mask

        a = self.state[zero].copy()
        b = self.state[one].copy()

        scale = 1.0 / cp.sqrt(2.0)

        self.state[zero] = (a + b) * scale
        self.state[one] = (a - b) * scale

        return self

    def cnot(self, control: int, target: int):
        """Apply controlled-X gate."""
        self._validate_qubit(control)
        self._validate_qubit(target)

        if control == target:
            raise ValueError("Control and target must differ")

        control_mask = 1 << control
        target_mask = 1 << target

        indices = cp.arange(self.size)

        swap_indices = indices[
            ((indices & control_mask) != 0)
            & ((indices & target_mask) == 0)
        ]

        partner_indices = swap_indices | target_mask

        a = self.state[swap_indices].copy()
        b = self.state[partner_indices].copy()

        self.state[swap_indices] = b
        self.state[partner_indices] = a

        return self

    def probabilities(self):
        """Return computational-basis probabilities."""
        return cp.abs(self.state) ** 2

    def amplitudes(self):
        """Copy amplitudes back to CPU."""
        cp.cuda.Stream.null.synchronize()
        return cp.asnumpy(self.state)

    def measure(self, shots: int = 1000, seed: int | None = None):
        """Sample computational-basis measurements."""
        if shots < 1:
            raise ValueError("Shots must be positive")

        probabilities = self.probabilities()
        probabilities = probabilities / cp.sum(probabilities)

        # Copy the small probability vector to CPU for sampling.
        cpu_probabilities = cp.asnumpy(probabilities)

        rng = np.random.default_rng(seed)

        samples = rng.choice(
            self.size,
            size=shots,
            p=cpu_probabilities,
        )

        results = {}

        for sample in samples:
            bitstring = format(
                int(sample),
                f"0{self.num_qubits}b"
            )
            results[bitstring] = results.get(bitstring, 0) + 1

        return results

    def _validate_qubit(self, qubit: int):
        if not isinstance(qubit, int):
            raise ValueError("Qubit index must be an integer")

        if not 0 <= qubit < self.num_qubits:
            raise ValueError("Qubit index out of range")

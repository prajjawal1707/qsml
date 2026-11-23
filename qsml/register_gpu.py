"""QSML 1.1 - Extended GPU state-vector simulator."""

import math
import cupy as cp
import numpy as np


class GPUQuantumRegister:

    def __init__(self, num_qubits: int):
        if not isinstance(num_qubits, int) or num_qubits < 1:
            raise ValueError("Qubit count must be positive")

        if num_qubits > 20:
            raise MemoryError("Maximum supported register is 20 qubits")

        self.num_qubits = num_qubits
        self.size = 2**num_qubits

        free_bytes, _ = cp.cuda.runtime.memGetInfo()
        required_bytes = self.size * 16 * 3

        if required_bytes > free_bytes * 0.5:
            raise MemoryError("Insufficient free GPU memory")

        self.state = cp.zeros(self.size, dtype=cp.complex128)
        self.state[0] = 1.0

    def _validate_qubit(self, qubit):
        if not isinstance(qubit, int):
            raise ValueError("Qubit index must be an integer")

        if not 0 <= qubit < self.num_qubits:
            raise ValueError("Qubit index out of range")

    def _validate_angle(self, theta):
        if not isinstance(theta, (int, float)):
            raise ValueError("Angle must be numeric")

        if not math.isfinite(theta):
            raise ValueError("Angle must be finite")

    def _pair_indices(self, qubit):
        """Return indices where the selected bit is zero and one."""
        mask = 1 << qubit
        indices = cp.arange(self.size)
        zero = indices[(indices & mask) == 0]
        one = zero | mask
        return zero, one

    def x(self, qubit):
        """Pauli-X gate."""
        self._validate_qubit(qubit)
        zero, one = self._pair_indices(qubit)

        a = self.state[zero].copy()
        b = self.state[one].copy()

        self.state[zero] = b
        self.state[one] = a
        return self

    def y(self, qubit):
        """Pauli-Y gate."""
        self._validate_qubit(qubit)
        zero, one = self._pair_indices(qubit)

        a = self.state[zero].copy()
        b = self.state[one].copy()

        self.state[zero] = -1j * b
        self.state[one] = 1j * a
        return self

    def z(self, qubit):
        """Pauli-Z gate."""
        self._validate_qubit(qubit)
        mask = 1 << qubit
        indices = cp.arange(self.size)
        selected = indices[(indices & mask) != 0]

        self.state[selected] *= -1
        return self

    def h(self, qubit):
        """Hadamard gate."""
        self._validate_qubit(qubit)
        zero, one = self._pair_indices(qubit)

        a = self.state[zero].copy()
        b = self.state[one].copy()

        scale = 1.0 / cp.sqrt(2.0)

        self.state[zero] = (a + b) * scale
        self.state[one] = (a - b) * scale
        return self

    def s(self, qubit):
        """S phase gate."""
        self._validate_qubit(qubit)
        mask = 1 << qubit
        indices = cp.arange(self.size)
        selected = indices[(indices & mask) != 0]

        self.state[selected] *= 1j
        return self

    def t(self, qubit):
        """T phase gate."""
        self._validate_qubit(qubit)
        mask = 1 << qubit
        indices = cp.arange(self.size)
        selected = indices[(indices & mask) != 0]

        self.state[selected] *= cp.exp(1j * cp.pi / 4)
        return self

    def rx(self, qubit, theta):
        """Rotation around the X axis."""
        self._validate_qubit(qubit)
        self._validate_angle(theta)

        zero, one = self._pair_indices(qubit)

        a = self.state[zero].copy()
        b = self.state[one].copy()

        c = math.cos(theta / 2)
        s = math.sin(theta / 2)

        self.state[zero] = c * a - 1j * s * b
        self.state[one] = -1j * s * a + c * b
        return self

    def ry(self, qubit, theta):
        """Rotation around the Y axis."""
        self._validate_qubit(qubit)
        self._validate_angle(theta)

        zero, one = self._pair_indices(qubit)

        a = self.state[zero].copy()
        b = self.state[one].copy()

        c = math.cos(theta / 2)
        s = math.sin(theta / 2)

        self.state[zero] = c * a - s * b
        self.state[one] = s * a + c * b
        return self

    def rz(self, qubit, theta):
        """Rotation around the Z axis."""
        self._validate_qubit(qubit)
        self._validate_angle(theta)

        mask = 1 << qubit
        indices = cp.arange(self.size)

        zero = indices[(indices & mask) == 0]
        one = indices[(indices & mask) != 0]

        self.state[zero] *= cp.exp(-0.5j * theta)
        self.state[one] *= cp.exp(0.5j * theta)
        return self

    def cnot(self, control, target):
        """Controlled-X gate."""
        self._validate_qubit(control)
        self._validate_qubit(target)

        if control == target:
            raise ValueError("Control and target must differ")

        control_mask = 1 << control
        target_mask = 1 << target

        indices = cp.arange(self.size)

        swap = indices[((indices & control_mask) != 0) & ((indices & target_mask) == 0)]

        partner = swap | target_mask

        a = self.state[swap].copy()
        b = self.state[partner].copy()

        self.state[swap] = b
        self.state[partner] = a
        return self

    def probabilities(self):
        return cp.abs(self.state) ** 2

    def amplitudes(self):
        cp.cuda.Stream.null.synchronize()
        return cp.asnumpy(self.state)

    def norm(self):
        """Return total state probability."""
        return float(cp.sum(self.probabilities()).get())

    def validate_state(self, tolerance=1e-8):
        """Check normalization and finite amplitudes."""
        finite = bool(cp.all(cp.isfinite(self.state)).get())
        total = self.norm()

        return {
            "finite": finite,
            "norm": total,
            "normalized": finite and abs(total - 1.0) <= tolerance,
        }

    def measure(self, shots=1000, seed=None):
        if not isinstance(shots, int) or shots < 1:
            raise ValueError("Shots must be a positive integer")

        probabilities = self.probabilities()
        probabilities /= cp.sum(probabilities)

        cpu_probabilities = cp.asnumpy(probabilities)
        rng = np.random.default_rng(seed)

        samples = rng.choice(
            self.size,
            size=shots,
            p=cpu_probabilities,
        )

        results = {}

        for sample in samples:
            state = format(int(sample), f"0{self.num_qubits}b")
            results[state] = results.get(state, 0) + 1

        return results

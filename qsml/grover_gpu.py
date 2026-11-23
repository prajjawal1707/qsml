
"""QSML 1.2 - GPU Grover search."""

import math
import cupy as cp

from .register_gpu import GPUQuantumRegister


class GPUGroverSearch:
    """Grover amplitude amplification on a GPU state vector."""

    def __init__(self, num_qubits: int):
        if not isinstance(num_qubits, int) or num_qubits < 1:
            raise ValueError("Qubit count must be positive")

        self.num_qubits = num_qubits
        self.register = GPUQuantumRegister(num_qubits)
        self.size = 2 ** num_qubits

    def _phase_flip(self, indices_to_flip):
        """Multiply selected state amplitudes by -1."""
        self.register.state[indices_to_flip] *= -1

    def _oracle(self, marked_indices):
        """Flip the phase of marked states."""
        indices = cp.asarray(marked_indices, dtype=cp.int64)
        self._phase_flip(indices)

    def _diffuser(self):
        """Apply Grover's diffusion operator."""

        # H on every qubit.
        for qubit in range(self.num_qubits):
            self.register.h(qubit)

        # X on every qubit.
        for qubit in range(self.num_qubits):
            self.register.x(qubit)

        # Phase flip of |11...1>.
        self._phase_flip([self.size - 1])

        # Reverse X and H operations.
        for qubit in range(self.num_qubits):
            self.register.x(qubit)

        for qubit in range(self.num_qubits):
            self.register.h(qubit)

    def search(
        self,
        marked_states: list[str],
        iterations: int | None = None,
        shots: int = 1000,
        seed: int | None = 42,
    ):
        if not marked_states:
            raise ValueError("At least one marked state is required")

        if shots < 1:
            raise ValueError("Shots must be positive")

        marked_indices = set()

        for state in marked_states:
            if (
                not isinstance(state, str)
                or len(state) != self.num_qubits
                or any(bit not in "01" for bit in state)
            ):
                raise ValueError(
                    "Each marked state must be a valid binary string "
                    "matching the register size"
                )

            marked_indices.add(int(state, 2))

        if len(marked_indices) >= self.size:
            raise ValueError("At least one unmarked state is required")

        marked_count = len(marked_indices)

        if iterations is None:
            angle = math.asin(math.sqrt(marked_count / self.size))
            iterations = max(
                0,
                round(math.pi / (4 * angle) - 0.5),
            )

        if not isinstance(iterations, int) or iterations < 0:
            raise ValueError("Iterations must be a nonnegative integer")

        # Prepare uniform superposition.
        for qubit in range(self.num_qubits):
            self.register.h(qubit)

        # Grover iterations.
        for _ in range(iterations):
            self._oracle(sorted(marked_indices))
            self._diffuser()

        probabilities = self.register.probabilities()
        probabilities /= cp.sum(probabilities)

        # Return probabilities and measurements.
        cpu_probabilities = cp.asnumpy(probabilities)

        measurements = self.register.measure(
            shots=shots,
            seed=seed,
        )

        candidates = [
            (
                format(index, f"0{self.num_qubits}b"),
                float(probability),
            )
            for index, probability in enumerate(cpu_probabilities)
        ]

        candidates.sort(key=lambda item: item[1], reverse=True)

        cp.cuda.Stream.null.synchronize()

        return {
            "qubits": self.num_qubits,
            "marked_states": sorted(
                format(index, f"0{self.num_qubits}b")
                for index in marked_indices
            ),
            "iterations": iterations,
            "top_candidates": candidates[:min(8, self.size)],
            "measurements": measurements,
            "state_validation": self.register.validate_state(),
        }

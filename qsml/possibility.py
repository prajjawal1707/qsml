"""Possibility Engine: Grover amplitude amplification on a classical simulator."""
from __future__ import annotations
from dataclasses import dataclass
import math
import numpy as np
from .register import QuantumRegister


@dataclass
class PossibilityResult:
    """Results from a simulated Grover search."""
    num_qubits: int
    marked_states: tuple[str, ...]
    iterations: int
    measurements: dict[str, int]
    probabilities: dict[str, float]
    top_candidates: list[tuple[str, float]]

    @property
    def most_measured(self) -> str | None:
        if not self.measurements:
            return None
        return max(self.measurements, key=self.measurements.get)


def _validate_bitstring(value: str, num_qubits: int) -> str:
    if not isinstance(value, str) or len(value) != num_qubits or any(c not in "01" for c in value):
        raise ValueError(f"marked states must be {num_qubits}-character binary strings")
    return value


def grover_search(
    num_qubits: int,
    marked_states: list[str] | tuple[str, ...],
    shots: int = 1000,
    iterations: int | None = None,
    seed: int | None = None,
) -> PossibilityResult:
    """Run Grover search for marked bitstrings using a state-vector simulation.

    This is a classical simulation of the algorithm. It is useful for learning
    amplitude amplification, not a quantum speedup on a conventional PC.
    """
    if not marked_states:
        raise ValueError("provide at least one marked state")
    if len(set(marked_states)) != len(marked_states):
        raise ValueError("marked_states must not contain duplicates")
    marked = tuple(_validate_bitstring(s, num_qubits) for s in marked_states)
    space_size = 1 << num_qubits
    if len(marked) >= space_size:
        raise ValueError("at least one unmarked state is required")
    if shots < 1:
        raise ValueError("shots must be positive")

    count = len(marked)
    if iterations is None:
        iterations = max(1, int(math.floor((math.pi / 4) * math.sqrt(space_size / count))))
    if iterations < 0:
        raise ValueError("iterations cannot be negative")

    register = QuantumRegister(num_qubits)
    for q in range(num_qubits):
        register.h(q)

    marked_indices = [int(bits, 2) for bits in marked]
    for _ in range(iterations):
        # Oracle: phase-flip the marked candidates.
        register.state[marked_indices] *= -1
        # Diffuser: reflect all amplitudes around their mean.
        mean = register.state.mean()
        register.state[:] = (2 * mean) - register.state

    if not register.validate():
        raise RuntimeError("amplitude amplification produced an invalid state")

    probs_array = register.probabilities()
    probs = {
        format(i, f"0{num_qubits}b"): float(p)
        for i, p in enumerate(probs_array) if p > 1e-12
    }
    ranked = sorted(probs.items(), key=lambda item: item[1], reverse=True)
    measurements = register.measure(shots=shots, seed=seed)
    return PossibilityResult(
        num_qubits=num_qubits,
        marked_states=marked,
        iterations=iterations,
        measurements=measurements,
        probabilities=probs,
        top_candidates=ranked[:min(8, len(ranked))],
    )

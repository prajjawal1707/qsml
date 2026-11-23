"""QSML: quantum-inspired computing on classical hardware."""

from .register import QuantumRegister
from .possibility import PossibilityResult, grover_search

__all__ = ["QuantumRegister", "PossibilityResult", "grover_search"]

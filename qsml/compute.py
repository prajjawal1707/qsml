"""QSML adaptive hardware planner.

Inspects available system resources and recommends a safe
execution device. This version plans execution; it does not
move simulator calculations onto the GPU yet.
"""

from dataclasses import dataclass
import importlib.util

import psutil


@dataclass
class ComputePlan:
    recommended_device: str
    estimated_memory_mb: float
    cpu_available_mb: float
    gpu_available_mb: float | None
    reason: str
    can_run: bool


def estimate_memory_bytes(num_qubits: int) -> int:
    """Estimate memory for a state vector and working copies."""

    if not isinstance(num_qubits, int) or num_qubits < 1:
        raise ValueError("num_qubits must be a positive integer")

    # complex128 = 16 bytes per amplitude.
    # Three copies provide a conservative working estimate.
    return 3 * 16 * (2**num_qubits)


def detect_gpu():
    """Return free GPU memory in MB, or None if CUDA is unavailable."""

    if importlib.util.find_spec("cupy") is None:
        return None

    try:
        import cupy as cp

        device_count = cp.cuda.runtime.getDeviceCount()

        if device_count < 1:
            return None

        free_bytes, total_bytes = cp.cuda.runtime.memGetInfo()
        return free_bytes / (1024**2)

    except Exception:
        # Missing CUDA drivers or incompatible GPU libraries.
        return None


def plan_execution(
    num_qubits: int,
    gpu_min_qubits: int = 16,
) -> ComputePlan:
    """Choose a safe device recommendation from available resources."""

    required_bytes = estimate_memory_bytes(num_qubits)
    required_mb = required_bytes / (1024**2)

    available_ram = psutil.virtual_memory().available
    cpu_available_mb = available_ram / (1024**2)

    # Keep most available system RAM free for Windows and other apps.
    cpu_budget_mb = cpu_available_mb * 0.20

    gpu_available_mb = detect_gpu()

    if gpu_available_mb is not None:
        # Keep a reserve of GPU memory for graphics and other processes.
        gpu_budget_mb = gpu_available_mb * 0.70
    else:
        gpu_budget_mb = 0.0

    cpu_fits = required_mb <= cpu_budget_mb
    gpu_fits = gpu_available_mb is not None and required_mb <= gpu_budget_mb

    if not cpu_fits and not gpu_fits:
        return ComputePlan(
            recommended_device="NONE",
            estimated_memory_mb=required_mb,
            cpu_available_mb=cpu_available_mb,
            gpu_available_mb=gpu_available_mb,
            reason="Estimated memory exceeds the safe CPU and GPU budgets.",
            can_run=False,
        )

    # Prefer CPU for small jobs. Use GPU only when it is available,
    # the workload is large enough, and the estimated memory fits.
    if gpu_fits and (not cpu_fits or num_qubits >= gpu_min_qubits):
        return ComputePlan(
            recommended_device="GPU",
            estimated_memory_mb=required_mb,
            cpu_available_mb=cpu_available_mb,
            gpu_available_mb=gpu_available_mb,
            reason="GPU memory is sufficient for this workload.",
            can_run=True,
        )

    return ComputePlan(
        recommended_device="CPU",
        estimated_memory_mb=required_mb,
        cpu_available_mb=cpu_available_mb,
        gpu_available_mb=gpu_available_mb,
        reason="CPU has sufficient safe memory; this workload does not need GPU selection.",
        can_run=True,
    )

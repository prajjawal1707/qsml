
"""QSML 3.1.5 — Resource preflight and memory safety."""

import math


def _positive_integer(value, name):
    if (
        not isinstance(value, int)
        or isinstance(value, bool)
        or value < 1
    ):
        raise ValueError(f"{name} must be a positive integer")
    return value


def estimate_gpu_payload_bytes(
    num_candidates,
    num_variables,
    bytes_per_variable=4,
    bytes_per_energy=8,
):
    """Estimate candidate matrix plus energy-vector payload."""

    num_candidates = _positive_integer(num_candidates, "num_candidates")
    num_variables = _positive_integer(num_variables, "num_variables")
    bytes_per_variable = _positive_integer(
        bytes_per_variable, "bytes_per_variable"
    )
    bytes_per_energy = _positive_integer(
        bytes_per_energy, "bytes_per_energy"
    )

    return num_candidates * (
        num_variables * bytes_per_variable + bytes_per_energy
    )


def check_gpu_candidate_budget(
    num_candidates,
    num_variables,
    available_bytes,
    safety_fraction=0.20,
):
    """Check payload against a fraction of currently available VRAM.

    available_bytes should be supplied by the caller or obtained from
    gpu_memory_snapshot(). The safety fraction reserves the remainder
    for CUDA context, allocator overhead, temporary arrays and other work.
    """

    available_bytes = _positive_integer(
        available_bytes, "available_bytes"
    )

    if isinstance(safety_fraction, bool) or not isinstance(
        safety_fraction, (int, float)
    ):
        raise ValueError("safety_fraction must be numeric")

    safety_fraction = float(safety_fraction)

    if not math.isfinite(safety_fraction) or not (
        0.0 < safety_fraction <= 1.0
    ):
        raise ValueError("safety_fraction must be in the range (0, 1]")

    estimated_bytes = estimate_gpu_payload_bytes(
        num_candidates,
        num_variables,
    )

    allowed_bytes = int(available_bytes * safety_fraction)
    bytes_per_candidate = num_variables * 4 + 8

    return {
        "fits": estimated_bytes <= allowed_bytes,
        "estimated_payload_bytes": estimated_bytes,
        "available_bytes": available_bytes,
        "allowed_payload_bytes": allowed_bytes,
        "headroom_bytes": allowed_bytes - estimated_bytes,
        "recommended_max_candidates": (
            allowed_bytes // bytes_per_candidate
        ),
        "safety_fraction": safety_fraction,
    }


def gpu_memory_snapshot():
    """Return current CUDA free/total memory, without allocating arrays."""

    try:
        import cupy as cp
    except ImportError:
        return {
            "available": False,
            "reason": "CuPy is not installed",
        }

    try:
        if cp.cuda.runtime.getDeviceCount() < 1:
            return {
                "available": False,
                "reason": "No CUDA device detected",
            }

        free_bytes, total_bytes = cp.cuda.runtime.memGetInfo()

        return {
            "available": True,
            "free_bytes": int(free_bytes),
            "total_bytes": int(total_bytes),
            "device": str(cp.cuda.Device()),
        }

    except Exception as exc:
        return {
            "available": False,
            "reason": str(exc),
        }


def preflight_gpu_candidates(
    num_candidates,
    num_variables,
    safety_fraction=0.20,
):
    """Inspect real GPU memory and check the proposed candidate workload."""

    snapshot = gpu_memory_snapshot()

    if not snapshot["available"]:
        raise RuntimeError(
            f"GPU memory inspection unavailable: {snapshot['reason']}"
        )

    report = check_gpu_candidate_budget(
        num_candidates=num_candidates,
        num_variables=num_variables,
        available_bytes=snapshot["free_bytes"],
        safety_fraction=safety_fraction,
    )

    report["device"] = snapshot["device"]
    report["total_vram_bytes"] = snapshot["total_bytes"]

    if not report["fits"]:
        raise MemoryError(
            "GPU candidate workload exceeds the configured memory budget. "
            f"Estimated payload: {report['estimated_payload_bytes']} bytes; "
            f"allowed payload: {report['allowed_payload_bytes']} bytes."
        )

    return report

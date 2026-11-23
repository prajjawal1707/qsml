"""Best-effort, read-only hardware inspection."""

from __future__ import annotations
import platform
import shutil
import subprocess
import psutil


def _nvidia_gpus() -> list[str]:
    exe = shutil.which("nvidia-smi")
    if not exe:
        return []
    try:
        result = subprocess.run(
            [exe, "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=3,
            check=False,
        )
        if result.returncode == 0:
            return [line.strip() for line in result.stdout.splitlines() if line.strip()]
    except (OSError, subprocess.TimeoutExpired):
        pass
    return []


def _windows_display_adapters() -> list[str]:
    if platform.system() != "Windows":
        return []
    command = (
        "Get-CimInstance Win32_VideoController | Select-Object -ExpandProperty Name"
    )
    try:
        result = subprocess.run(
            ["powershell", "-NoProfile", "-Command", command],
            capture_output=True,
            text=True,
            timeout=6,
            check=False,
        )
        if result.returncode == 0:
            return list(
                dict.fromkeys(
                    line.strip() for line in result.stdout.splitlines() if line.strip()
                )
            )
    except (OSError, subprocess.TimeoutExpired):
        pass
    return []


def inspect_hardware() -> dict:
    mem, disk = psutil.virtual_memory(), psutil.disk_usage("/")
    return {
        "system": platform.system(),
        "release": platform.release(),
        "processor": platform.processor() or "Not reported",
        "logical_cores": psutil.cpu_count(logical=True) or 1,
        "physical_cores": psutil.cpu_count(logical=False) or 1,
        "ram_total_gib": round(mem.total / (1024**3), 2),
        "ram_available_gib": round(mem.available / (1024**3), 2),
        "disk_total_gib": round(disk.total / (1024**3), 2),
        "disk_free_gib": round(disk.free / (1024**3), 2),
        "display_adapters": _windows_display_adapters() or ["Not detected"],
        "nvidia_smi_gpus": _nvidia_gpus()
        or ["nvidia-smi unavailable / GPU not confirmed"],
    }

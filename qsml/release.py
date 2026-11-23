"""QSML 3.0 release readiness and environment checks."""
import importlib
import platform
import sys


VERSION = "3.0.0"

REQUIRED_MODULES = (
    "qsml.register",
    "qsml.possibility",
    "qsml.runtime",
    "qsml.adaptive",
    "qsml.batching",
    "qsml.validation",
    "qsml.quality",
    "qsml.benchmarking",
    "qsml.experiment_store",
    "qsml.constraints",
    "qsml.problems",
    "qsml.api",
)


def environment_info():
    return {
        "qsml_version": VERSION,
        "python_version": platform.python_version(),
        "implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "executable": sys.executable,
    }


def run_release_checks():
    checks = []

    for module_name in REQUIRED_MODULES:
        try:
            importlib.import_module(module_name)
            checks.append({
                "name": f"Import {module_name}",
                "passed": True,
                "error": None,
            })
        except Exception as exc:
            checks.append({
                "name": f"Import {module_name}",
                "passed": False,
                "error": f"{type(exc).__name__}: {exc}",
            })

    return {
        "version": VERSION,
        "environment": environment_info(),
        "checks": checks,
        "passed": sum(item["passed"] for item in checks),
        "failed": sum(not item["passed"] for item in checks),
        "ready": all(item["passed"] for item in checks),
    }

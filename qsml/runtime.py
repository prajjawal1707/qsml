
"""QSML 1.3 - Unified CPU/GPU runtime."""

from .possibility import grover_search


class UnifiedRuntime:

    def available_backends(self):
        backends = ["CPU"]

        try:
            import cupy as cp

            if cp.cuda.runtime.getDeviceCount() > 0:
                backends.append("GPU")
        except Exception:
            pass

        return backends

    def grover(
        self,
        num_qubits,
        marked_states,
        backend="auto",
        iterations=None,
        shots=1000,
        seed=42,
    ):
        backend = backend.upper()

        available = self.available_backends()

        if backend == "AUTO":
            backend = "GPU" if "GPU" in available and num_qubits >= 5 else "CPU"

        if backend not in ("CPU", "GPU"):
            raise ValueError("Backend must be auto, CPU, or GPU")

        if backend not in available:
            raise RuntimeError(f"{backend} backend is unavailable")

        if backend == "GPU":
            from .grover_gpu import GPUGroverSearch

            engine = GPUGroverSearch(num_qubits)
            result = engine.search(
                marked_states=marked_states,
                iterations=iterations,
                shots=shots,
                seed=seed,
            )

            return {
                "backend": "GPU",
                "qubits": result["qubits"],
                "iterations": result["iterations"],
                "top_candidates": result["top_candidates"],
                "measurements": result["measurements"],
                "validation": result["state_validation"],
            }

        result = grover_search(
            num_qubits=num_qubits,
            marked_states=marked_states,
            shots=shots,
            iterations=iterations,
            seed=seed,
        )

        return {
            "backend": "CPU",
            "qubits": num_qubits,
            "iterations": iterations,
            "top_candidates": result.top_candidates,
            "measurements": result.measurements,
            "validation": None,
        }

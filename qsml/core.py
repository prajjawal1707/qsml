
"""QSML 3.5 — Unified core optimization interface."""

from .adaptive import AdaptiveOptimizer
from .qubo_contract import normalize_qubo_terms
from .resource_guard import preflight_gpu_candidates
from .diagnostics import diagnose_solver_result


class QSMLCore:
    """Unified interface for validated CPU/GPU QUBO optimization."""

    def __init__(
        self,
        gpu_safety_fraction=0.20,
        exact_diagnostics_limit=16,
    ):
        if (
            isinstance(gpu_safety_fraction, bool)
            or not isinstance(gpu_safety_fraction, (int, float))
            or not 0.0 < gpu_safety_fraction <= 1.0
        ):
            raise ValueError(
                "gpu_safety_fraction must be in the range (0, 1]"
            )

        if (
            not isinstance(exact_diagnostics_limit, int)
            or isinstance(exact_diagnostics_limit, bool)
            or not 1 <= exact_diagnostics_limit <= 20
        ):
            raise ValueError(
                "exact_diagnostics_limit must be between 1 and 20"
            )

        self.gpu_safety_fraction = float(gpu_safety_fraction)
        self.exact_diagnostics_limit = exact_diagnostics_limit
        self.optimizer = AdaptiveOptimizer()

    def available_backends(self):
        """Return installed and usable optimization backends."""
        return self.optimizer.available_backends()

    def solve(
        self,
        num_variables,
        terms,
        backend="auto",
        evaluation_budget=8192,
        seed=42,
        exact_diagnostics=True,
    ):
        """Solve a QUBO with a controlled objective-evaluation budget."""

        if (
            not isinstance(evaluation_budget, int)
            or isinstance(evaluation_budget, bool)
            or evaluation_budget < 3
        ):
            raise ValueError(
                "evaluation_budget must be an integer of at least 3"
            )

        if not isinstance(seed, int) or isinstance(seed, bool):
            raise ValueError("seed must be an integer")

        if not isinstance(backend, str):
            raise ValueError("backend must be a string")

        if not isinstance(exact_diagnostics, bool):
            raise ValueError("exact_diagnostics must be Boolean")

        normalized_terms = normalize_qubo_terms(
            num_variables,
            terms,
        )

        requested_backend = backend.upper()

        if requested_backend == "AUTO":
            selected_backend = self.optimizer.choose_backend(
                num_variables,
                candidates=evaluation_budget,
            )
        elif requested_backend in ("CPU", "GPU"):
            selected_backend = requested_backend
        else:
            raise ValueError("backend must be auto, CPU, or GPU")

        available = self.available_backends()

        if selected_backend not in available:
            raise RuntimeError(
                f"{selected_backend} backend is unavailable"
            )

        resource_report = None

        if selected_backend == "GPU":
            resource_report = preflight_gpu_candidates(
                num_candidates=evaluation_budget,
                num_variables=num_variables,
                safety_fraction=self.gpu_safety_fraction,
            )

            result = self.optimizer.solve(
                num_variables=num_variables,
                terms=normalized_terms,
                backend="GPU",
                candidates=evaluation_budget,
                seed=seed,
            )

        else:
            # The CPU engine counts:
            # 1 baseline solution + 1 initial random solution
            # + (evaluation_budget - 2) annealing moves.
            result = self.optimizer.solve(
                num_variables=num_variables,
                terms=normalized_terms,
                backend="CPU",
                iterations=evaluation_budget - 2,
                restarts=1,
                candidates=evaluation_budget,
                seed=seed,
            )

        if result["evaluations"] != evaluation_budget:
            raise RuntimeError(
                "Solver evaluation count does not match requested budget: "
                f"expected {evaluation_budget}, "
                f"received {result['evaluations']}"
            )

        diagnostics = diagnose_solver_result(
            num_variables=num_variables,
            terms=normalized_terms,
            solver_result=result,
            exact_max_variables=self.exact_diagnostics_limit,
        ) if exact_diagnostics else None

        if diagnostics is not None and not diagnostics["valid"]:
            raise RuntimeError(
                f"Solver result failed validation: {diagnostics['errors']}"
            )

        output = dict(result)
        output.update({
            "requested_backend": requested_backend,
            "evaluation_budget": evaluation_budget,
            "seed": seed,
            "resource_preflight": resource_report,
            "diagnostics": diagnostics,
            "qsml_core_version": "3.5",
        })

        return output

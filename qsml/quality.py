from dataclasses import dataclass, asdict
import math

from qsml.validation import (
    solve_qubo_exact,
    validate_qubo_solution,
)


@dataclass
class SolverQualityReport:
    solver: str
    backend: str | None
    method: str | None
    solution: list[int]
    energy: float
    reported_energy: float | None
    optimality_gap: float
    relative_gap: float | None
    optimal: bool
    valid: bool
    evaluations: int | None
    elapsed_seconds: float | None
    errors: list[str]


class SolverQualityAnalyzer:
    """Compare solver results with an exact small-problem reference."""

    def __init__(self, tolerance=1e-8, max_exact_variables=20):
        if tolerance < 0 or not math.isfinite(tolerance):
            raise ValueError("tolerance must be finite and non-negative")

        if max_exact_variables < 1:
            raise ValueError("max_exact_variables must be positive")

        self.tolerance = tolerance
        self.max_exact_variables = max_exact_variables

    @staticmethod
    def _get(result, key, default=None):
        if isinstance(result, dict):
            return result.get(key, default)
        return getattr(result, key, default)

    def compare(self, num_variables, terms, solver_results):
        if not isinstance(solver_results, dict) or not solver_results:
            raise ValueError("solver_results must be a non-empty dictionary")

        exact = solve_qubo_exact(
            num_variables,
            terms,
            max_variables=self.max_exact_variables,
        )

        reports = []

        for solver_name, result in solver_results.items():
            solution = self._get(result, "solution")
            reported_energy = self._get(result, "energy")

            if solution is None:
                raise ValueError(f"{solver_name} did not provide a solution")

            validation = validate_qubo_solution(
                num_variables,
                terms,
                solution,
                reported_energy=reported_energy,
                tolerance=self.tolerance,
            )

            if not validation.valid:
                reports.append(
                    SolverQualityReport(
                        solver=str(solver_name),
                        backend=self._get(result, "backend"),
                        method=self._get(result, "method"),
                        solution=list(solution),
                        energy=(
                            validation.energy
                            if validation.energy is not None
                            else float("nan")
                        ),
                        reported_energy=reported_energy,
                        optimality_gap=float("nan"),
                        relative_gap=None,
                        optimal=False,
                        valid=False,
                        evaluations=self._get(result, "evaluations"),
                        elapsed_seconds=self._get(result, "elapsed_seconds"),
                        errors=validation.errors,
                    )
                )
                continue

            energy = validation.energy
            gap = max(0.0, energy - exact.energy)

            relative_gap = (
                gap / abs(exact.energy) if abs(exact.energy) > self.tolerance else None
            )

            reports.append(
                SolverQualityReport(
                    solver=str(solver_name),
                    backend=self._get(result, "backend"),
                    method=self._get(result, "method"),
                    solution=list(solution),
                    energy=energy,
                    reported_energy=reported_energy,
                    optimality_gap=gap,
                    relative_gap=relative_gap,
                    optimal=gap <= self.tolerance,
                    valid=True,
                    evaluations=self._get(result, "evaluations"),
                    elapsed_seconds=self._get(result, "elapsed_seconds"),
                    errors=[],
                )
            )

        return {
            "exact_solution": exact.solution,
            "exact_energy": exact.energy,
            "exact_evaluations": exact.evaluations,
            "reports": [asdict(report) for report in reports],
        }

    @staticmethod
    def print_report(comparison):
        print("\nQSML SOLVER QUALITY REPORT")
        print("=" * 72)
        print(f"Exact energy: {comparison['exact_energy']:.8f}")
        print(f"Exact solution: {comparison['exact_solution']}")
        print(f"Exact evaluations: {comparison['exact_evaluations']}")
        print("-" * 72)

        for report in comparison["reports"]:
            print(f"\nSolver: {report['solver']}")
            print(f"  Backend: {report['backend']}")
            print(f"  Method: {report['method']}")
            print(f"  Energy: {report['energy']:.8f}")
            print(f"  Optimality gap: {report['optimality_gap']:.8f}")
            print(f"  Optimal: {report['optimal']}")
            print(f"  Valid: {report['valid']}")
            print(f"  Evaluations: {report['evaluations']}")
            print(f"  Elapsed seconds: {report['elapsed_seconds']}")

            for error in report["errors"]:
                print(f"  ERROR: {error}")

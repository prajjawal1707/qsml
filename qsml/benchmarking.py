
import math
import random
import statistics
import time


class BenchmarkRunner:
    """Repeatable benchmarking for QSML solver functions."""

    def __init__(self, base_seed=42):
        if not isinstance(base_seed, int) or isinstance(base_seed, bool):
            raise ValueError("base_seed must be an integer")

        self.base_seed = base_seed

    @staticmethod
    def _validate_result(result):
        if not isinstance(result, dict):
            raise ValueError("Solver must return a dictionary")

        if "energy" not in result:
            raise ValueError("Solver result must contain energy")

        energy = result["energy"]

        if not isinstance(energy, (int, float)):
            raise ValueError("Energy must be numeric")

        if not math.isfinite(energy):
            raise ValueError("Energy must be finite")

        return float(energy)

    @staticmethod
    def _statistics(values):
        return {
            "mean": statistics.mean(values),
            "median": statistics.median(values),
            "minimum": min(values),
            "maximum": max(values),
            "standard_deviation": (
                statistics.stdev(values) if len(values) > 1 else 0.0
            ),
        }

    def run(self, solver, repeats=5, name="QSML Solver"):
        """
        Run a solver repeatedly.

        The solver must accept seed=integer and return a dictionary
        containing at least an energy field.
        """

        if not callable(solver):
            raise ValueError("solver must be callable")

        if not isinstance(repeats, int) or isinstance(repeats, bool):
            raise ValueError("repeats must be an integer")

        if repeats < 1 or repeats > 1000:
            raise ValueError("repeats must be between 1 and 1000")

        trials = []

        for index in range(repeats):
            seed = self.base_seed + index

            random.seed(seed)

            start = time.perf_counter()
            result = solver(seed=seed)
            elapsed = time.perf_counter() - start

            energy = self._validate_result(result)

            trials.append({
                "trial": index + 1,
                "seed": seed,
                "energy": energy,
                "elapsed_seconds": elapsed,
                "evaluations": result.get("evaluations"),
                "backend": result.get("backend"),
                "method": result.get("method"),
            })

        energies = [trial["energy"] for trial in trials]
        times = [trial["elapsed_seconds"] for trial in trials]

        return {
            "benchmark": name,
            "repeats": repeats,
            "base_seed": self.base_seed,
            "energy_statistics": self._statistics(energies),
            "runtime_statistics": self._statistics(times),
            "best_energy": min(energies),
            "worst_energy": max(energies),
            "trials": trials,
        }

    def compare(self, solvers, repeats=5):
        """Benchmark multiple named solvers."""

        if not isinstance(solvers, dict) or not solvers:
            raise ValueError("solvers must be a non-empty dictionary")

        results = {}

        for name, solver in solvers.items():
            results[name] = self.run(
                solver=solver,
                repeats=repeats,
                name=name,
            )

        return {
            "repeats": repeats,
            "base_seed": self.base_seed,
            "results": results,
        }

    @staticmethod
    def print_report(report):
        print("\nQSML REPRODUCIBLE BENCHMARK")
        print("=" * 65)

        for name, result in report["results"].items():
            energy = result["energy_statistics"]
            runtime = result["runtime_statistics"]

            print(f"\nSolver: {name}")
            print(f"  Trials: {result['repeats']}")
            print(f"  Mean energy: {energy['mean']:.8f}")
            print(f"  Best energy: {result['best_energy']:.8f}")
            print(f"  Worst energy: {result['worst_energy']:.8f}")
            print(f"  Energy std dev: {energy['standard_deviation']:.8f}")
            print(f"  Mean runtime: {runtime['mean']:.8f} seconds")
            print(f"  Median runtime: {runtime['median']:.8f} seconds")
            print(f"  Runtime std dev: {runtime['standard_deviation']:.8f}")
            print(f"  Minimum runtime: {runtime['minimum']:.8f}")
            print(f"  Maximum runtime: {runtime['maximum']:.8f}")

        print("\n" + "=" * 65)

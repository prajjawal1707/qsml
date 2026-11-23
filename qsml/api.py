"""Unified public QSML API."""
from qsml.adaptive import AdaptiveOptimizer
from qsml.benchmarking import BenchmarkRunner
from qsml.problems import ProblemLibrary
from qsml.experiment_store import ExperimentStore


class QSML:
    """Convenience interface over the existing QSML engines."""

    VERSION = "3.0.0"

    def __init__(self, experiment_database="qsml_experiments.sqlite3"):
        self.optimizer = AdaptiveOptimizer()
        self.benchmarker = BenchmarkRunner()
        self.experiments = ExperimentStore(experiment_database)
        self.problems = ProblemLibrary()

    def solve_qubo(
        self,
        num_variables,
        terms,
        backend="auto",
        iterations=12000,
        restarts=8,
        candidates=4096,
        seed=42,
    ):
        return self.optimizer.solve(
            num_variables=num_variables,
            terms=terms,
            backend=backend,
            iterations=iterations,
            restarts=restarts,
            candidates=candidates,
            seed=seed,
        )

    def solve_problem(self, problem, **solver_options):
        if not isinstance(problem, dict):
            raise ValueError("problem must be a problem-builder dictionary")
        return self.solve_qubo(
            num_variables=problem["num_variables"],
            terms=problem["terms"],
            **solver_options,
        )

    def grover(
        self,
        num_qubits,
        marked_states,
        iterations=1,
        shots=1000,
        backend="auto",
    ):
        from qsml.runtime import UnifiedRuntime

        runtime = UnifiedRuntime()
        return runtime.grover(
            num_qubits=num_qubits,
            marked_states=marked_states,
            iterations=iterations,
            shots=shots,
            backend=backend,
        )

    def benchmark(self, solvers, repeats=5, base_seed=42):
        return BenchmarkRunner(base_seed=base_seed).compare(
            solvers=solvers,
            repeats=repeats,
        )

    def save_experiment(self, name, config, result, tags=None):
        return self.experiments.save(name, config, result, tags)

    def build_max_cut(self, num_vertices, edges):
        return self.problems.max_cut(num_vertices, edges)

    def build_knapsack(self, weights, values, capacity, penalty=10.0):
        return self.problems.knapsack(
            weights, values, capacity, penalty
        )

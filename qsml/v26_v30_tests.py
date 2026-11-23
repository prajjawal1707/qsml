"""Focused tests for QSML 2.6 through 3.0."""
import itertools
import tempfile
from pathlib import Path

from qsml.experiment_store import ExperimentStore
from qsml.constraints import ConstraintBuilder
from qsml.problems import ProblemLibrary
from qsml.api import QSML
from qsml.release import run_release_checks


def _energy(terms, solution):
    return sum(
        coefficient * solution[i] * solution[j]
        for (i, j), coefficient in terms.items()
    )


def run_v26_v30_tests():
    passed = 0
    failed = 0

    def check(name, function):
        nonlocal passed, failed
        try:
            function()
            print(f"[PASS] {name}")
            passed += 1
        except Exception as exc:
            print(f"[FAIL] {name}: {exc}")
            failed += 1

    def test_store_roundtrip():
        with tempfile.TemporaryDirectory() as directory:
            store = ExperimentStore(Path(directory) / "test.sqlite3")
            identifier = store.save(
                "unit-test",
                {"backend": "CPU"},
                {"energy": -4.0},
                tags=["test"],
            )
            record = store.get(identifier)
            assert record["name"] == "unit-test"
            assert record["config"]["backend"] == "CPU"
            assert record["result"]["energy"] == -4.0
            assert record["tags"] == ["test"]

    def test_store_list_compare_delete():
        with tempfile.TemporaryDirectory() as directory:
            store = ExperimentStore(Path(directory) / "test.sqlite3")
            first = store.save("run", {}, {"energy": -2.0})
            second = store.save("run", {}, {"energy": -3.0})
            assert len(store.list(name="run")) == 2
            comparison = store.compare([first, second])
            assert [row["value"] for row in comparison] == [-2.0, -3.0]
            assert store.delete(first) is True
            assert store.get(first) is None

    def test_store_rejects_bad_data():
        with tempfile.TemporaryDirectory() as directory:
            store = ExperimentStore(Path(directory) / "test.sqlite3")
            rejected = False
            try:
                store.save("", {}, {})
            except ValueError:
                rejected = True
            assert rejected

    def test_equality_penalty():
        builder = ConstraintBuilder()
        result = builder.equality(2, {0: 1, 1: 1}, 1, penalty=5)
        # The penalty is zero for exactly one selected variable.
        # The constant penalty offset is omitted from QUBO terms.
        feasible_energy = -5.0
        assert abs(_energy(result["terms"], [1, 0]) - feasible_energy) < 1e-10
        assert abs(_energy(result["terms"], [0, 1]) - feasible_energy) < 1e-10
        assert _energy(result["terms"], [1, 1]) > feasible_energy

    def test_at_most_penalty():
        builder = ConstraintBuilder()
        result = builder.at_most(2, {0: 2, 1: 3}, 3, penalty=5)
        # For x=[1,0], slack=1 makes the encoded equality exact.
        best = min(
            _energy(result["terms"], bits)
            for bits in itertools.product(
                (0, 1), repeat=result["num_variables"]
            )
            if list(bits[:2]) == [1, 0]
        )
        # The omitted constant is -penalty * limit**2.
        assert abs(best - (-45.0)) < 1e-10

    def test_constraint_rejects_invalid_coefficients():
        rejected = False
        try:
            ConstraintBuilder().at_most(2, {0: -1}, 3)
        except ValueError:
            rejected = True
        assert rejected

    def test_max_cut_builder():
        problem = ProblemLibrary.max_cut(2, [(0, 1)])
        assert problem["num_variables"] == 2
        assert _energy(problem["terms"], [0, 1]) == -1.0
        assert _energy(problem["terms"], [0, 0]) == 0.0

    def test_max_cut_rejects_bad_edge():
        rejected = False
        try:
            ProblemLibrary.max_cut(2, [(0, 2)])
        except ValueError:
            rejected = True
        assert rejected

    def test_knapsack_builder():
        problem = ProblemLibrary.knapsack(
            weights=[2, 3],
            values=[3, 4],
            capacity=3,
            penalty=10,
        )
        assert problem["decision_variables"] == 2
        assert problem["num_variables"] >= 2
        # Selecting only item 2 is feasible; there is a slack assignment
        # making its constraint penalty zero.
        minimum = min(
            _energy(problem["terms"], bits)
            for bits in itertools.product(
                (0, 1), repeat=problem["num_variables"]
            )
            if list(bits[:2]) == [0, 1]
        )
        # Constraint offset is -10 * 3**2; selected item's objective is -4.
        assert abs(minimum - (-94.0)) < 1e-10

    def test_knapsack_rejects_fractional_weight():
        rejected = False
        try:
            ProblemLibrary.knapsack([1.5], [2], 3)
        except ValueError:
            rejected = True
        assert rejected

    def test_unified_api_initializes():
        with tempfile.TemporaryDirectory() as directory:
            api = QSML(Path(directory) / "api.sqlite3")
            assert api.VERSION == "3.0.0"
            assert api.build_max_cut(2, [(0, 1)])["num_variables"] == 2

    def test_unified_api_cpu_qubo():
        with tempfile.TemporaryDirectory() as directory:
            api = QSML(Path(directory) / "api.sqlite3")
            result = api.solve_qubo(
                2,
                {(0, 0): -2.0, (1, 1): -2.0, (0, 1): 1.0},
                backend="CPU",
                iterations=1000,
                restarts=2,
                seed=42,
            )
            assert result["solution"] in ([1, 1], [1, 0], [0, 1], [0, 0])
            assert isinstance(result["energy"], (int, float))

    def test_api_save_experiment():
        with tempfile.TemporaryDirectory() as directory:
            api = QSML(Path(directory) / "api.sqlite3")
            identifier = api.save_experiment(
                "api-test", {"backend": "CPU"}, {"energy": -1.0}
            )
            assert api.experiments.get(identifier)["result"]["energy"] == -1.0

    def test_release_checks():
        report = run_release_checks()
        assert report["passed"] > 0
        assert report["failed"] == 0
        assert report["ready"] is True

    def test_release_environment():
        from qsml.release import environment_info
        info = environment_info()
        assert info["qsml_version"] == "3.0.0"
        assert info["python_version"]

    checks = [
        ("Experiment storage round trip", test_store_roundtrip),
        ("Experiment listing, comparison and deletion", test_store_list_compare_delete),
        ("Experiment input validation", test_store_rejects_bad_data),
        ("Equality constraint penalty", test_equality_penalty),
        ("At-most constraint penalty", test_at_most_penalty),
        ("Constraint input validation", test_constraint_rejects_invalid_coefficients),
        ("Max-Cut problem builder", test_max_cut_builder),
        ("Max-Cut invalid edge rejection", test_max_cut_rejects_bad_edge),
        ("Knapsack problem builder", test_knapsack_builder),
        ("Knapsack invalid weight rejection", test_knapsack_rejects_fractional_weight),
        ("Unified API initialization", test_unified_api_initializes),
        ("Unified API CPU QUBO", test_unified_api_cpu_qubo),
        ("Unified API experiment storage", test_api_save_experiment),
        ("QSML 3.0 release checks", test_release_checks),
        ("QSML 3.0 environment metadata", test_release_environment),
    ]

    for name, function in checks:
        check(name, function)

    print("\nQSML 2.6–3.0 TEST SUMMARY")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Total: {passed + failed}")

    return passed, failed

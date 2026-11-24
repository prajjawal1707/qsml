
"""QSML 3.1.5–3.5 — Resource, benchmark, regression and core tests."""

from qsml.resource_guard import (
    estimate_gpu_payload_bytes,
    check_gpu_candidate_budget,
)
from qsml.fair_benchmark import run_equal_budget_comparison
from qsml.regression import run_qubo_regression
from qsml.diagnostics import diagnose_solver_result
from qsml.core import QSMLCore


def run_v315_v35_tests():
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

    def must_raise(function):
        try:
            function()
        except (ValueError, RuntimeError, MemoryError):
            return

        raise AssertionError("Expected a controlled exception")

    terms = {
        (0, 0): -2.0,
        (1, 1): -2.0,
        (0, 1): 1.0,
    }

    check(
        "GPU candidate payload estimation",
        lambda: (
            estimate_gpu_payload_bytes(100, 10) == 4800
            or (_ for _ in ()).throw(AssertionError("Wrong estimate"))
        ),
    )

    check(
        "GPU memory budget accepts fitting workload",
        lambda: (
            check_gpu_candidate_budget(
                100, 10, 10000, safety_fraction=0.8
            )["fits"] is True
            or (_ for _ in ()).throw(AssertionError("Expected fit"))
        ),
    )

    check(
        "GPU memory budget rejects oversized workload",
        lambda: (
            check_gpu_candidate_budget(
                100, 10, 1000, safety_fraction=0.5
            )["fits"] is False
            or (_ for _ in ()).throw(AssertionError("Expected rejection"))
        ),
    )

    check(
        "GPU memory budget rejects invalid safety fraction",
        lambda: must_raise(
            lambda: check_gpu_candidate_budget(
                100, 10, 10000, safety_fraction=1.5
            )
        ),
    )

    def cpu_budget_test():
        report = run_equal_budget_comparison(
            2,
            terms,
            evaluation_budget=40,
            repeats=2,
            seed=11,
            backends=("CPU",),
        )

        cpu = report["results"]["CPU"]

        assert cpu["available"] is True
        assert cpu["evaluations_per_trial"] == [40, 40]
        assert len(cpu["trials"]) == 2

    check(
        "CPU equal-evaluation benchmark budget",
        cpu_budget_test,
    )

    def backend_budget_test():
        report = run_equal_budget_comparison(
            2,
            terms,
            evaluation_budget=40,
            repeats=1,
            seed=12,
            backends=("CPU", "GPU"),
        )

        cpu = report["results"]["CPU"]
        assert cpu["available"] is True
        assert cpu["evaluations_per_trial"] == [40]

        gpu = report["results"]["GPU"]

        if gpu["available"]:
            assert gpu["evaluations_per_trial"] == [40]

    check(
        "CPU/GPU equal-budget comparison protocol",
        backend_budget_test,
    )

    check(
        "Benchmark rejects insufficient evaluation budget",
        lambda: must_raise(
            lambda: run_equal_budget_comparison(
                2,
                terms,
                evaluation_budget=2,
                backends=("CPU",),
            )
        ),
    )

    def regression_validity_test():
        report = run_qubo_regression(
            cases=3,
            min_variables=2,
            max_variables=6,
            seed=100,
            iterations=500,
            restarts=2,
        )

        assert report["valid"] is True
        assert report["completed_cases"] == 3
        assert report["requested_cases"] == 3
        assert len(report["results"]) == 3

    check(
        "Randomized QUBO regression validity",
        regression_validity_test,
    )

    def regression_repeatability_test():
        first = run_qubo_regression(
            cases=3,
            min_variables=2,
            max_variables=6,
            seed=200,
            iterations=400,
            restarts=2,
        )

        second = run_qubo_regression(
            cases=3,
            min_variables=2,
            max_variables=6,
            seed=200,
            iterations=400,
            restarts=2,
        )

        first_energies = [
            item["energy"] for item in first["results"]
        ]

        second_energies = [
            item["energy"] for item in second["results"]
        ]

        assert first_energies == second_energies

    check(
        "Randomized regression fixed-seed repeatability",
        regression_repeatability_test,
    )

    check(
        "Regression rejects oversized exact-reference problems",
        lambda: must_raise(
            lambda: run_qubo_regression(
                cases=1,
                min_variables=2,
                max_variables=17,
            )
        ),
    )

    def exact_diagnostic_test():
        report = diagnose_solver_result(
            2,
            terms,
            {
                "solution": [1, 1],
                "energy": -3.0,
                "backend": "CPU",
                "method": "TEST",
                "evaluations": 10,
            },
        )

        assert report["valid"] is True
        assert report["exact_checked"] is True
        assert report["optimal"] is True
        assert report["optimality_gap"] == 0.0

    check(
        "Diagnostics identify exact optimum",
        exact_diagnostic_test,
    )

    def suboptimal_diagnostic_test():
        report = diagnose_solver_result(
            2,
            terms,
            {
                "solution": [1, 0],
                "energy": -2.0,
            },
        )

        assert report["valid"] is True
        assert report["exact_checked"] is True
        assert report["optimal"] is False
        assert report["optimality_gap"] == 1.0

    check(
        "Diagnostics identify valid suboptimal solution",
        suboptimal_diagnostic_test,
    )

    def invalid_diagnostic_test():
        report = diagnose_solver_result(
            2,
            terms,
            {
                "solution": [1, 1],
                "energy": -2.0,
            },
        )

        assert report["valid"] is False
        assert report["status"] == "INVALID"
        assert len(report["errors"]) > 0

    check(
        "Diagnostics reject incorrect reported energy",
        invalid_diagnostic_test,
    )

    def large_problem_diagnostic_test():
        large_terms = {
            (i, i): -1.0
            for i in range(5)
        }

        report = diagnose_solver_result(
            5,
            large_terms,
            {
                "solution": [1, 1, 1, 1, 1],
                "energy": -5.0,
            },
            exact_max_variables=3,
        )

        assert report["valid"] is True
        assert report["exact_checked"] is False
        assert report["status"] == "VALID_EXACT_CHECK_SKIPPED"

    check(
        "Diagnostics safely skip oversized exact checks",
        large_problem_diagnostic_test,
    )

    def core_cpu_test():
        core = QSMLCore()

        result = core.solve(
            2,
            terms,
            backend="CPU",
            evaluation_budget=50,
            seed=42,
        )

        assert result["backend"] == "CPU"
        assert result["evaluations"] == 50
        assert result["diagnostics"]["valid"] is True
        assert result["qsml_core_version"] == "3.5"

    check(
        "Unified QSML Core CPU execution",
        core_cpu_test,
    )

    def core_budget_test():
        core = QSMLCore()

        result = core.solve(
            2,
            terms,
            backend="CPU",
            evaluation_budget=60,
            seed=43,
        )

        assert result["evaluation_budget"] == 60
        assert result["evaluations"] == 60

    check(
        "Unified Core evaluation-budget enforcement",
        core_budget_test,
    )

    check(
        "Unified Core rejects too-small evaluation budget",
        lambda: must_raise(
            lambda: QSMLCore().solve(
                2,
                terms,
                backend="CPU",
                evaluation_budget=2,
            )
        ),
    )

    check(
        "Unified Core rejects unsupported backend",
        lambda: must_raise(
            lambda: QSMLCore().solve(
                2,
                terms,
                backend="QUANTUM",
                evaluation_budget=50,
            )
        ),
    )

    check(
        "Unified Core rejects invalid QUBO terms",
        lambda: must_raise(
            lambda: QSMLCore().solve(
                2,
                {(0, 2): 1.0},
                backend="CPU",
                evaluation_budget=50,
            )
        ),
    )

    def core_auto_backend_test():
        core = QSMLCore()

        result = core.solve(
            2,
            terms,
            backend="auto",
            evaluation_budget=50,
            seed=44,
        )

        assert result["backend"] in core.available_backends()
        assert result["diagnostics"]["valid"] is True

    check(
        "Unified Core automatic backend selection",
        core_auto_backend_test,
    )

    print("\nQSML 3.1.5–3.5 TEST SUMMARY")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Total:  {passed + failed}")

    return passed, failed

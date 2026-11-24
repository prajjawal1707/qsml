
"""QSML 4.0 — Decision Space Intelligence tests."""

from qsml.decision_space import DecisionSpaceExplorer


def run_v40_tests():
    passed = 0
    failed = 0

    explorer = DecisionSpaceExplorer()

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
        except ValueError:
            return

        raise AssertionError("Expected ValueError")

    objectives = {
        "cost": {
            (0, 0): 1.0,
            (1, 1): 2.0,
        },
        "risk": {
            (0, 0): 2.0,
            (1, 1): 1.0,
        },
    }

    candidates = [
        [1, 0],
        [0, 1],
        [1, 1],
        [0, 0],
    ]

    constraints = [
        {
            "name": "select_at_least_one",
            "coefficients": {0: 1, 1: 1},
            "sense": ">=",
            "rhs": 1,
        }
    ]

    def test_pareto_frontier():
        report = explorer.analyze(
            2,
            objectives,
            candidates,
            constraints,
        )

        frontier = {
            tuple(item["solution"])
            for item in report["frontier"]
        }

        assert frontier == {(1, 0), (0, 1)}
        assert report["frontier_count"] == 2
        assert report["global_optimum_proven"] is False

    check(
        "Identify two-objective Pareto frontier",
        test_pareto_frontier,
    )

    def test_objective_values():
        report = explorer.analyze(
            2,
            objectives,
            candidates,
            constraints,
        )

        result = next(
            item for item in report["candidate_results"]
            if item["solution"] == [1, 0]
        )

        assert result["objectives"] == {
            "cost": 1.0,
            "risk": 2.0,
        }

    check(
        "Calculate multiple objective values",
        test_objective_values,
    )

    def test_infeasible_filter():
        report = explorer.analyze(
            2,
            objectives,
            candidates,
            constraints,
        )

        assert report["feasible_count"] == 3
        assert report["infeasible_count"] == 1

        infeasible = [
            item for item in report["candidate_results"]
            if not item["feasible"]
        ]

        assert len(infeasible) == 1
        assert infeasible[0]["solution"] == [0, 0]

    check(
        "Exclude infeasible candidates from frontier",
        test_infeasible_filter,
    )

    def test_less_equal_constraint():
        report = explorer.analyze(
            2,
            {"cost": objectives["cost"]},
            [[0, 0], [1, 0], [1, 1]],
            [{
                "name": "capacity",
                "coefficients": {0: 1, 1: 1},
                "sense": "<=",
                "rhs": 1,
            }],
        )

        assert report["feasible_count"] == 2
        assert report["infeasible_count"] == 1

    check(
        "Validate less-than-or-equal constraint",
        test_less_equal_constraint,
    )

    def test_equality_constraint():
        report = explorer.analyze(
            2,
            {"cost": objectives["cost"]},
            [[1, 0], [0, 1], [1, 1]],
            [{
                "name": "exact_selection",
                "coefficients": {0: 1, 1: 1},
                "sense": "==",
                "rhs": 1,
            }],
        )

        assert report["feasible_count"] == 2
        assert report["infeasible_count"] == 1

    check(
        "Validate equality hard constraint",
        test_equality_constraint,
    )

    def test_constraint_tolerance():
        tolerant = DecisionSpaceExplorer(tolerance=1e-6)

        report = tolerant.analyze(
            1,
            {"cost": {(0, 0): 1.0}},
            [[1]],
            [{
                "name": "upper_limit",
                "coefficients": {0: 1.0},
                "sense": "<=",
                "rhs": 1.0000005,
            }],
        )

        assert report["feasible_count"] == 1

    check(
        "Apply constraint tolerance",
        test_constraint_tolerance,
    )

    def test_duplicate_candidates():
        report = explorer.analyze(
            2,
            objectives,
            [[1, 0], [1, 0], [0, 1]],
        )

        assert report["candidate_count"] == 3
        assert report["unique_candidate_count"] == 2
        assert report["duplicate_count"] == 1

    check(
        "Remove duplicate candidate solutions",
        test_duplicate_candidates,
    )

    def test_equal_objectives():
        report = explorer.analyze(
            2,
            {
                "objective_a": {
                    (0, 0): 0.0,
                },
            },
            [[1, 0], [0, 1]],
        )

        assert report["frontier_count"] == 2

    check(
        "Preserve distinct candidates with equal objective values",
        test_equal_objectives,
    )

    def test_scenario_analysis():
        scenarios = {
            "normal": objectives,
            "risk_priority": {
                "cost": objectives["risk"],
                "risk": objectives["cost"],
            },
        }

        report = explorer.analyze_scenarios(
            2,
            scenarios,
            candidates[:2],
        )

        assert report["scenario_count"] == 2
        assert len(report["frontier_in_every_scenario"]) == 2

    check(
        "Compare multiple decision scenarios",
        test_scenario_analysis,
    )

    def test_empty_objectives():
        must_raise(
            lambda: explorer.analyze(
                2,
                {},
                [[1, 0]],
            )
        )

    check(
        "Reject empty objective collection",
        test_empty_objectives,
    )

    def test_invalid_candidate():
        must_raise(
            lambda: explorer.analyze(
                2,
                objectives,
                [[1, 2]],
            )
        )

    check(
        "Reject non-binary candidate",
        test_invalid_candidate,
    )

    def test_invalid_constraint_index():
        must_raise(
            lambda: explorer.analyze(
                2,
                objectives,
                [[1, 0]],
                [{
                    "name": "invalid",
                    "coefficients": {2: 1},
                    "sense": "<=",
                    "rhs": 1,
                }],
            )
        )

    check(
        "Reject out-of-range constraint variable",
        test_invalid_constraint_index,
    )

    def test_invalid_constraint_sense():
        must_raise(
            lambda: explorer.analyze(
                2,
                objectives,
                [[1, 0]],
                [{
                    "name": "invalid",
                    "coefficients": {0: 1},
                    "sense": "!=",
                    "rhs": 1,
                }],
            )
        )

    check(
        "Reject unsupported constraint operator",
        test_invalid_constraint_sense,
    )

    def test_invalid_rhs():
        must_raise(
            lambda: explorer.analyze(
                2,
                objectives,
                [[1, 0]],
                [{
                    "name": "invalid",
                    "coefficients": {0: 1},
                    "sense": "<=",
                    "rhs": float("nan"),
                }],
            )
        )

    check(
        "Reject non-finite constraint right-hand side",
        test_invalid_rhs,
    )

    print("\nQSML 4.0 TEST SUMMARY")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Total:  {passed + failed}")

    return passed, failed

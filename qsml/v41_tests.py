
"""QSML 4.1 — Active Decision Space Exploration tests."""

from qsml.decision_search import (
    generate_candidate_pool,
    ActiveDecisionExplorer,
)


def run_v41_tests():
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

    def test_reproducibility():
        first = generate_candidate_pool(
            8,
            candidate_budget=40,
            seed=100,
        )

        second = generate_candidate_pool(
            8,
            candidate_budget=40,
            seed=100,
        )

        assert first == second

    check(
        "Candidate generation reproducibility",
        test_reproducibility,
    )

    def test_candidate_budget():
        result = generate_candidate_pool(
            10,
            candidate_budget=50,
            seed=42,
        )

        assert len(result) == 50

    check(
        "Candidate budget enforcement",
        test_candidate_budget,
    )

    def test_candidate_uniqueness():
        result = generate_candidate_pool(
            10,
            candidate_budget=100,
            seed=42,
        )

        assert len(result) == len({
            tuple(candidate) for candidate in result
        })

    check(
        "Candidate uniqueness",
        test_candidate_uniqueness,
    )

    def test_boundary_candidates():
        result = generate_candidate_pool(
            5,
            candidate_budget=12,
            seed=42,
        )

        assert [0] * 5 in result
        assert [1] * 5 in result

    check(
        "Include all-zero and all-one candidates",
        test_boundary_candidates,
    )

    def test_initial_candidates():
        initial = [[1, 0, 1], [0, 1, 0]]

        result = generate_candidate_pool(
            3,
            candidate_budget=8,
            seed=42,
            initial_candidates=initial,
        )

        assert [1, 0, 1] in result
        assert [0, 1, 0] in result

    check(
        "Include initial candidate solutions",
        test_initial_candidates,
    )

    def test_small_space_exhaustion():
        result = generate_candidate_pool(
            3,
            candidate_budget=8,
            seed=42,
        )

        assert len(result) == 8
        assert len({tuple(item) for item in result}) == 8

    check(
        "Exhaust small binary search space when budget permits",
        test_small_space_exhaustion,
    )

    def test_mutation_neighbors():
        result = generate_candidate_pool(
            4,
            candidate_budget=12,
            seed=42,
            initial_candidates=[[0, 0, 0, 0]],
            mutation_rounds=5,
            mutation_probability=0.0,
        )

        base = [0, 0, 0, 0]

        neighbors = [
            candidate for candidate in result
            if sum(
                a != b for a, b in zip(base, candidate)
            ) == 1
        ]

        assert len(neighbors) > 0

    check(
        "Explore mutated neighboring candidates",
        test_mutation_neighbors,
    )

    check(
        "Reject zero candidate budget",
        lambda: must_raise(
            lambda: generate_candidate_pool(
                5,
                candidate_budget=0,
            )
        ),
    )

    check(
        "Reject invalid mutation probability",
        lambda: must_raise(
            lambda: generate_candidate_pool(
                5,
                mutation_probability=1.5,
            )
        ),
    )

    check(
        "Reject non-binary initial candidate",
        lambda: must_raise(
            lambda: generate_candidate_pool(
                3,
                initial_candidates=[[1, 2, 0]],
            )
        ),
    )

    def test_full_exploration():
        explorer = ActiveDecisionExplorer()

        report = explorer.explore(
            num_variables=2,
            objectives=objectives,
            candidate_budget=4,
            seed=42,
        )

        assert report["unique_candidate_count"] == 4
        assert report["frontier_count"] >= 1
        assert report["global_optimum_proven"] is False

    check(
        "Generate and analyze decision space",
        test_full_exploration,
    )

    def test_scenario_exploration():
        explorer = ActiveDecisionExplorer()

        scenarios = {
            "normal": objectives,
            "risk_priority": {
                "cost": objectives["risk"],
                "risk": objectives["cost"],
            },
        }

        report = explorer.explore_scenarios(
            num_variables=2,
            scenarios=scenarios,
            candidate_budget=4,
            seed=42,
        )

        assert report["scenario_count"] == 2
        assert report["generation"]["generated_candidates"] == 4

    check(
        "Generate one shared pool for multiple scenarios",
        test_scenario_exploration,
    )

    print("\nQSML 4.1 TEST SUMMARY")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Total:  {passed + failed}")

    return passed, failed


"""QSML 4.2–4.9 — Decision Intelligence tests."""

import tempfile
from pathlib import Path

from qsml.guided_search import GuidedDecisionSearch
from qsml.decision_space import DecisionSpaceExplorer
from qsml.diversity import (
    hamming_distance,
    select_diverse_frontier,
)
from qsml.robustness import analyze_scenario_robustness
from qsml.sensitivity import analyze_coefficient_sensitivity
from qsml.explainability import explain_decision
from qsml.solver_pool import merge_solver_candidates
from qsml.decision_record import (
    create_decision_record,
    verify_decision_record,
    save_decision_record,
    load_decision_record,
)
from qsml.decision_intelligence import QSMLDecisionIntelligence


def run_v42_v49_tests():
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
        except (ValueError, TypeError):
            return

        raise AssertionError("Expected controlled validation exception")

    objectives = {
        "cost": {
            (0, 0): -1.0,
            (1, 1): -2.0,
        },
        "risk": {
            (0, 0): 2.0,
            (1, 1): 1.0,
        },
    }

    candidates = [
        [0, 0],
        [1, 0],
        [0, 1],
        [1, 1],
    ]

    # ---------------------------------------------------------
    # QSML 4.2 — GUIDED SEARCH
    # ---------------------------------------------------------

    def guided_improvement():
        search = GuidedDecisionSearch()

        report = search.search(
            2,
            {"cost": objectives["cost"]},
            [[0, 0]],
            steps_per_seed=4,
            neighbor_budget=50,
            seed=42,
        )

        best_energy = min(
            item["objectives"]["cost"]
            for item in report["candidate_results"]
        )

        assert best_energy <= 0.0
        assert report["guided_search"]["neighbor_evaluations"] > 0

    check(
        "Guided search explores and improves candidates",
        guided_improvement,
    )

    def guided_repeatability():
        search = GuidedDecisionSearch()

        first = search.search(
            2,
            objectives,
            [[0, 0], [1, 1]],
            steps_per_seed=3,
            neighbor_budget=50,
            seed=123,
        )

        second = search.search(
            2,
            objectives,
            [[0, 0], [1, 1]],
            steps_per_seed=3,
            neighbor_budget=50,
            seed=123,
        )

        first_solutions = [
            item["solution"]
            for item in first["candidate_results"]
        ]

        second_solutions = [
            item["solution"]
            for item in second["candidate_results"]
        ]

        assert first_solutions == second_solutions

    check(
        "Guided search fixed-seed repeatability",
        guided_repeatability,
    )

    check(
        "Guided search rejects invalid evaluation budget",
        lambda: must_raise(
            lambda: GuidedDecisionSearch().search(
                2,
                objectives,
                [[0, 0]],
                neighbor_budget=0,
            )
        ),
    )

    # ---------------------------------------------------------
    # QSML 4.3 — DIVERSITY ARCHIVE
    # ---------------------------------------------------------

    def diversity_selection():
        report = DecisionSpaceExplorer().analyze(
            2,
            objectives,
            candidates,
        )

        selected = select_diverse_frontier(
            report,
            max_candidates=2,
        )

        assert selected["selected_count"] == 2
        assert len(selected["selected"]) == 2

    check(
        "Select diverse Pareto candidates",
        diversity_selection,
    )

    check(
        "Hamming distance calculation",
        lambda: (
            hamming_distance([0, 0, 1], [1, 0, 0]) == 2
            or (_ for _ in ()).throw(AssertionError("Wrong distance"))
        ),
    )

    def empty_frontier():
        result = select_diverse_frontier(
            {
                "frontier": [],
                "objective_names": ["cost"],
            },
            max_candidates=3,
        )

        assert result["selected_count"] == 0

    check(
        "Empty Pareto frontier handling",
        empty_frontier,
    )

    # ---------------------------------------------------------
    # QSML 4.4 — ROBUSTNESS
    # ---------------------------------------------------------

    scenarios = {
        "normal": objectives,
        "stress": {
            "cost": {
                (0, 0): -0.5,
                (1, 1): -1.0,
            },
            "risk": {
                (0, 0): 3.0,
                (1, 1): 2.0,
            },
        },
    }

    def robustness_analysis():
        report = analyze_scenario_robustness(
            2,
            scenarios,
            candidates,
        )

        assert report["scenario_count"] == 2
        assert report["candidate_count"] == 4
        assert len(report["candidate_results"]) == 4

    check(
        "Analyze multiple operating scenarios",
        robustness_analysis,
    )

    def robustness_feasibility():
        report = analyze_scenario_robustness(
            2,
            scenarios,
            candidates,
            constraints=[{
                "name": "select_one",
                "coefficients": {0: 1, 1: 1},
                "sense": ">=",
                "rhs": 1,
            }],
        )

        for item in report["robust_candidates"]:
            assert item["feasible_in_all_scenarios"] is True

    check(
        "Identify candidates feasible across scenarios",
        robustness_feasibility,
    )

    check(
        "Reject scenario objective-name mismatch",
        lambda: must_raise(
            lambda: analyze_scenario_robustness(
                2,
                {
                    "one": {"cost": objectives["cost"]},
                    "two": {"risk": objectives["risk"]},
                },
                candidates,
            )
        ),
    )

    # ---------------------------------------------------------
    # QSML 4.5 — SENSITIVITY
    # ---------------------------------------------------------

    def sensitivity_zero_perturbation():
        report = analyze_coefficient_sensitivity(
            2,
            objectives,
            candidates,
            trials=3,
            perturbation=0.0,
            seed=42,
        )

        assert report["completed_trials"] == 3
        assert len(report["baseline_frontier"]) > 0

    check(
        "Sensitivity analysis with zero perturbation",
        sensitivity_zero_perturbation,
    )

    def sensitivity_repeatability():
        first = analyze_coefficient_sensitivity(
            2,
            objectives,
            candidates,
            trials=4,
            perturbation=0.1,
            seed=42,
        )

        second = analyze_coefficient_sensitivity(
            2,
            objectives,
            candidates,
            trials=4,
            perturbation=0.1,
            seed=42,
        )

        assert first["candidate_frontier_frequency"] == (
            second["candidate_frontier_frequency"]
        )

    check(
        "Sensitivity analysis repeatability",
        sensitivity_repeatability,
    )

    check(
        "Sensitivity rejects invalid perturbation",
        lambda: must_raise(
            lambda: analyze_coefficient_sensitivity(
                2,
                objectives,
                candidates,
                perturbation=2.0,
            )
        ),
    )

    # ---------------------------------------------------------
    # QSML 4.6 — EXPLAINABILITY
    # ---------------------------------------------------------

    def explanation_energy():
        report = explain_decision(
            2,
            [1, 1],
            {"cost": objectives["cost"]},
        )

        assert report["objectives"]["cost"]["energy"] == -3.0
        assert report["objectives"]["cost"]["contribution_sum"] == -3.0

    check(
        "Explain objective contributions",
        explanation_energy,
    )

    def explanation_constraints():
        report = explain_decision(
            2,
            [1, 0],
            {"cost": objectives["cost"]},
            constraints=[{
                "name": "select_both",
                "coefficients": {0: 1, 1: 1},
                "sense": "==",
                "rhs": 2,
            }],
        )

        assert report["feasible"] is False
        assert report["constraints"][0]["violation"] == 1.0

    check(
        "Explain violated hard constraint",
        explanation_constraints,
    )

    check(
        "Reject invalid explanation solution",
        lambda: must_raise(
            lambda: explain_decision(
                2,
                [1, 2],
                {"cost": objectives["cost"]},
            )
        ),
    )

    # ---------------------------------------------------------
    # QSML 4.7 — SOLVER COLLABORATION
    # ---------------------------------------------------------

    objective_terms = objectives["cost"]

    def solver_merge():
        report = merge_solver_candidates(
            2,
            objective_terms,
            {
                "CPU": {
                    "solution": [1, 1],
                    "energy": -3.0,
                },
                "GPU": {
                    "solution": [1, 0],
                    "energy": -1.0,
                },
            },
            supplied_candidates=[[0, 0]],
        )

        assert report["candidate_count"] == 3
        assert report["all_solver_results_valid"] is True

    check(
        "Merge candidates from multiple solvers",
        solver_merge,
    )

    def solver_invalid_energy():
        report = merge_solver_candidates(
            2,
            objective_terms,
            {
                "BAD_SOLVER": {
                    "solution": [1, 1],
                    "energy": -1.0,
                },
            },
        )

        assert report["candidate_count"] == 0
        assert len(report["errors"]) == 1

    check(
        "Reject solver with incorrect reported energy",
        solver_invalid_energy,
    )

    def solver_duplicate_merge():
        report = merge_solver_candidates(
            2,
            objective_terms,
            {
                "CPU": {"solution": [1, 1], "energy": -3.0},
                "GPU": {"solution": [1, 1], "energy": -3.0},
            },
        )

        assert report["candidate_count"] == 1
        assert len(report["provenance"][0]["sources"]) == 2

    check(
        "Deduplicate solver outputs and retain provenance",
        solver_duplicate_merge,
    )

    # ---------------------------------------------------------
    # QSML 4.8 — DECISION RECORDS
    # ---------------------------------------------------------

    def decision_record_integrity():
        record = create_decision_record(
            configuration={"seed": 42},
            result={"solution": [1, 1], "energy": -3.0},
        )

        verification = verify_decision_record(record)

        assert verification["valid"] is True

    check(
        "Decision record hash verification",
        decision_record_integrity,
    )

    def tampered_record():
        record = create_decision_record(
            configuration={"seed": 42},
            result={"energy": -3.0},
        )

        record["result"]["energy"] = -2.0

        assert verify_decision_record(record)["valid"] is False

    check(
        "Detect changed decision record",
        tampered_record,
    )

    def decision_record_round_trip():
        record = create_decision_record(
            configuration={"seed": 123},
            result={"solution": [1, 0], "energy": -1.0},
        )

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "decision.json"

            save_decision_record(path, record)

            loaded = load_decision_record(path)

            assert loaded["integrity"]["valid"] is True
            assert loaded["record"]["result"]["energy"] == -1.0

    check(
        "Save and load decision record",
        decision_record_round_trip,
    )

    # ---------------------------------------------------------
    # QSML 4.9 — UNIFIED DECISION INTELLIGENCE
    # ---------------------------------------------------------

    def unified_workflow():
        engine = QSMLDecisionIntelligence()

        result = engine.run(
            num_variables=2,
            objectives=objectives,
            candidate_budget=4,
            seed=42,
            guided_steps=2,
            neighbor_budget=20,
            diversity_limit=4,
        )

        assert result["qsml_version"] == "4.9"
        assert result["decision_report"]["unique_candidate_count"] >= 1
        assert result["global_optimum_proven"] is False
        assert len(result["explanations"]) > 0

    check(
        "Unified decision intelligence workflow",
        unified_workflow,
    )

    def unified_scenarios():
        engine = QSMLDecisionIntelligence()

        result = engine.run(
            num_variables=2,
            objectives=objectives,
            candidate_budget=4,
            seed=43,
            guided_steps=1,
            neighbor_budget=10,
            scenarios=scenarios,
        )

        assert result["robustness"] is not None
        assert result["robustness"]["scenario_count"] == 2

    check(
        "Unified workflow with scenario analysis",
        unified_scenarios,
    )

    def unified_record():
        engine = QSMLDecisionIntelligence()

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "qsml_decision.json"

            result = engine.run(
                num_variables=2,
                objectives=objectives,
                candidate_budget=4,
                seed=44,
                guided_steps=1,
                neighbor_budget=10,
                record_path=str(path),
            )

            loaded = load_decision_record(path)

            assert path.exists()
            assert loaded["integrity"]["valid"] is True
            assert result["saved_record_path"] == str(path)

    check(
        "Unified workflow writes verifiable audit record",
        unified_record,
    )

    print("\nQSML 4.2–4.9 TEST SUMMARY")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Total:  {passed + failed}")

    return passed, failed

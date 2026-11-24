
"""QSML 4.4 — Robustness analysis across decision scenarios."""

import math

from .decision_space import DecisionSpaceExplorer


def _positive_integer(value, name):
    if (
        not isinstance(value, int)
        or isinstance(value, bool)
        or value < 1
    ):
        raise ValueError(f"{name} must be a positive integer")
    return value


def analyze_scenario_robustness(
    num_variables,
    scenarios,
    candidates,
    constraints=None,
    tolerance=1e-9,
):
    """Measure feasibility and objective variation across scenarios.

    Every scenario must use the same set of objective names.
    All objectives are minimized.

    A robust candidate here means it is feasible in every supplied
    scenario. It does not imply robustness to unmodeled conditions.
    """

    if not isinstance(scenarios, dict) or not scenarios:
        raise ValueError("scenarios must be a non-empty dictionary")

    if isinstance(candidates, (str, bytes)):
        raise ValueError("candidates must be iterable")

    try:
        candidate_pool = list(candidates)
    except TypeError as exc:
        raise ValueError("candidates must be iterable") from exc

    if not candidate_pool:
        raise ValueError("At least one candidate is required")

    scenario_reports = {}
    expected_names = None

    for scenario_name, objectives in scenarios.items():
        if not isinstance(scenario_name, str) or not scenario_name.strip():
            raise ValueError("Scenario names must be non-empty strings")

        explorer = DecisionSpaceExplorer(tolerance=tolerance)

        report = explorer.analyze(
            num_variables,
            objectives,
            candidate_pool,
            constraints,
        )

        names = set(report["objective_names"])

        if expected_names is None:
            expected_names = names
        elif names != expected_names:
            raise ValueError(
                "All scenarios must have identical objective names"
            )

        scenario_reports[scenario_name] = report

    candidate_map = {}

    for scenario_name, report in scenario_reports.items():
        for item in report["candidate_results"]:
            key = tuple(item["solution"])

            candidate_map.setdefault(key, {
                "solution": list(item["solution"]),
                "scenarios": {},
            })

            candidate_map[key]["scenarios"][scenario_name] = item

    results = []

    for key, record in candidate_map.items():
        scenario_items = record["scenarios"]

        feasibility = {
            name: item["feasible"]
            for name, item in scenario_items.items()
        }

        feasible_count = sum(feasibility.values())
        all_feasible = feasible_count == len(scenario_reports)

        objective_summary = {}

        for objective_name in sorted(expected_names):
            values = [
                item["objectives"][objective_name]
                for item in scenario_items.values()
            ]

            objective_summary[objective_name] = {
                "best_case": min(values),
                "worst_case": max(values),
                "mean": math.fsum(values) / len(values),
                "range": max(values) - min(values),
                "scenario_values": {
                    name: item["objectives"][objective_name]
                    for name, item in scenario_items.items()
                },
            }

        results.append({
            "solution": list(record["solution"]),
            "feasible_in_all_scenarios": all_feasible,
            "feasible_scenario_count": feasible_count,
            "scenario_count": len(scenario_reports),
            "feasibility_by_scenario": feasibility,
            "objectives": objective_summary,
        })

    robust_candidates = [
        item for item in results
        if item["feasible_in_all_scenarios"]
    ]

    return {
        "scenario_count": len(scenario_reports),
        "candidate_count": len(results),
        "robust_candidate_count": len(robust_candidates),
        "robust_candidates": robust_candidates,
        "candidate_results": results,
        "scenario_reports": scenario_reports,
        "definition": (
            "Robust candidates are feasible in every supplied scenario. "
            "Objective ranges describe variation across those scenarios."
        ),
        "global_robust_optimum_proven": False,
    }

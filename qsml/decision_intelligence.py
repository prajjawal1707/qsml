
"""QSML 4.9 — Unified Decision Intelligence Engine."""

from .decision_search import ActiveDecisionExplorer
from .guided_search import GuidedDecisionSearch
from .diversity import select_diverse_frontier
from .robustness import analyze_scenario_robustness
from .sensitivity import analyze_coefficient_sensitivity
from .explainability import explain_decision
from .solver_pool import merge_solver_candidates
from .decision_record import (
    create_decision_record,
    save_decision_record,
)


class QSMLDecisionIntelligence:
    """Unified bounded candidate exploration and decision-analysis workflow."""

    def __init__(self, tolerance=1e-9):
        self.candidate_engine = ActiveDecisionExplorer(
            tolerance=tolerance
        )

        self.guided_engine = GuidedDecisionSearch(
            tolerance=tolerance
        )

        self.tolerance = tolerance

    def run(
        self,
        num_variables,
        objectives,
        constraints=None,
        candidate_budget=256,
        seed=42,
        initial_candidates=None,
        solver_results=None,
        guided_steps=5,
        neighbor_budget=1000,
        diversity_limit=20,
        scenarios=None,
        sensitivity_trials=0,
        sensitivity_perturbation=0.05,
        record_path=None,
    ):
        """Run the integrated QSML decision intelligence workflow.

        The workflow operates on a bounded generated candidate pool.
        It does not prove a global optimum over the full binary space.
        """

        # Stage 1: Generate a reproducible candidate pool.
        candidates = self.candidate_engine.generate(
            num_variables=num_variables,
            candidate_budget=candidate_budget,
            seed=seed,
            initial_candidates=initial_candidates,
        )

        solver_merge = None

        # Stage 2: Incorporate candidate solutions from other solvers.
        if solver_results is not None:
            if not objectives:
                raise ValueError("At least one objective is required")

            first_objective_name = next(iter(objectives))

            solver_merge = merge_solver_candidates(
                num_variables=num_variables,
                objective_terms=objectives[first_objective_name],
                solver_results=solver_results,
                supplied_candidates=candidates,
            )

            candidates = solver_merge["candidates"]

        # Stage 3: Improve candidates through bounded guided search.
        if guided_steps > 0:
            guided_report = self.guided_engine.search(
                num_variables=num_variables,
                objectives=objectives,
                candidates=candidates,
                constraints=constraints,
                steps_per_seed=guided_steps,
                neighbor_budget=neighbor_budget,
                seed=seed,
            )

            guided_candidates = [
                item["solution"]
                for item in guided_report["candidate_results"]
            ]

            candidates.extend(guided_candidates)

        else:
            guided_report = None

        # Stage 4: Analyze the combined decision pool.
        from .decision_space import DecisionSpaceExplorer

        explorer = DecisionSpaceExplorer(
            tolerance=self.tolerance
        )

        decision_report = explorer.analyze(
            num_variables=num_variables,
            objectives=objectives,
            candidates=candidates,
            constraints=constraints,
        )

        # Stage 5: Preserve diverse Pareto alternatives.
        diversity_report = select_diverse_frontier(
            decision_report,
            max_candidates=diversity_limit,
        )

        # Stage 6: Evaluate across supplied scenarios, when requested.
        robustness_report = None

        if scenarios is not None:
            robustness_report = analyze_scenario_robustness(
                num_variables=num_variables,
                scenarios=scenarios,
                candidates=candidates,
                constraints=constraints,
                tolerance=self.tolerance,
            )

        # Stage 7: Measure coefficient sensitivity, when requested.
        sensitivity_report = None

        if sensitivity_trials > 0:
            sensitivity_report = analyze_coefficient_sensitivity(
                num_variables=num_variables,
                objectives=objectives,
                candidates=candidates,
                constraints=constraints,
                trials=sensitivity_trials,
                perturbation=sensitivity_perturbation,
                seed=seed,
                tolerance=self.tolerance,
            )

        # Stage 8: Explain the selected Pareto alternatives.
        explanations = []

        for item in diversity_report["selected"]:
            explanations.append(
                explain_decision(
                    num_variables=num_variables,
                    solution=item["solution"],
                    objectives=objectives,
                    constraints=constraints,
                    tolerance=self.tolerance,
                )
            )

        result = {
            "qsml_version": "4.9",
            "method": "UNIFIED_DECISION_INTELLIGENCE",
            "seed": seed,
            "num_variables": num_variables,
            "candidate_budget": candidate_budget,
            "generated_and_merged_candidate_count": len(candidates),
            "decision_report": decision_report,
            "guided_search": guided_report,
            "diversity": diversity_report,
            "robustness": robustness_report,
            "sensitivity": sensitivity_report,
            "explanations": explanations,
            "solver_merge": solver_merge,
            "global_optimum_proven": False,
        }

        # Stage 9: Optionally produce a reproducible JSON audit record.
        record = create_decision_record(
            configuration={
                "num_variables": num_variables,
                "candidate_budget": candidate_budget,
                "seed": seed,
                "guided_steps": guided_steps,
                "neighbor_budget": neighbor_budget,
                "diversity_limit": diversity_limit,
                "sensitivity_trials": sensitivity_trials,
            },
            result=result,
            project_version="4.9",
        )

        saved_record_path = None

        if record_path is not None:
            saved_record_path = save_decision_record(
                record_path,
                record,
            )

        result["decision_record"] = record
        result["saved_record_path"] = saved_record_path

        return result

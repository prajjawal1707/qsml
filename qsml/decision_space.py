
"""QSML 4.0 — Decision Space Intelligence.

Analyzes supplied binary candidate solutions against multiple
minimization objectives and hard linear constraints.

The Pareto frontier is calculated only over the supplied candidates.
It is not a claim of global optimality over every possible bitstring.
"""

import math

from .stability import stable_qubo_energy
from .qubo_contract import normalize_qubo_terms


def _finite_number(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")

    try:
        value = float(value)
    except (OverflowError, ValueError) as exc:
        raise ValueError(f"{name} must be finite") from exc

    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")

    return value


def _validate_tolerance(tolerance):
    tolerance = _finite_number(tolerance, "tolerance")

    if tolerance < 0:
        raise ValueError("tolerance must be non-negative")

    return tolerance


def _validate_candidates(num_variables, candidates):
    if isinstance(candidates, (str, bytes)):
        raise ValueError("Candidates must be binary sequences")

    try:
        rows = list(candidates)
    except TypeError as exc:
        raise ValueError("Candidates must be iterable") from exc

    if not rows:
        raise ValueError("At least one candidate is required")

    unique_rows = []
    seen = set()

    for row_number, row in enumerate(rows):
        if isinstance(row, (str, bytes)):
            raise ValueError(
                f"Candidate {row_number} must be a binary sequence"
            )

        try:
            bits = list(row)
        except TypeError as exc:
            raise ValueError(
                f"Candidate {row_number} must be iterable"
            ) from exc

        if len(bits) != num_variables:
            raise ValueError(
                f"Candidate {row_number} has incorrect length"
            )

        normalized = []

        for bit in bits:
            if isinstance(bit, bool):
                normalized.append(int(bit))
            elif bit in (0, 1):
                normalized.append(int(bit))
            else:
                raise ValueError(
                    f"Candidate {row_number} contains a non-binary value"
                )

        key = tuple(normalized)

        if key not in seen:
            seen.add(key)
            unique_rows.append(normalized)

    return unique_rows, len(rows) - len(unique_rows)


def _normalize_objectives(num_variables, objectives):
    if not isinstance(objectives, dict) or not objectives:
        raise ValueError("objectives must be a non-empty dictionary")

    normalized = {}

    for name, terms in objectives.items():
        if not isinstance(name, str) or not name.strip():
            raise ValueError("Objective names must be non-empty strings")

        if name in normalized:
            raise ValueError(f"Duplicate objective name: {name}")

        normalized[name] = normalize_qubo_terms(
            num_variables,
            terms,
        )

    return normalized


def _normalize_constraints(num_variables, constraints):
    if constraints is None:
        return []

    if not isinstance(constraints, (list, tuple)):
        raise ValueError("constraints must be a list or tuple")

    normalized = []

    for index, constraint in enumerate(constraints):
        if not isinstance(constraint, dict):
            raise ValueError(
                f"Constraint {index} must be a dictionary"
            )

        name = constraint.get("name", f"constraint_{index}")

        if not isinstance(name, str) or not name.strip():
            raise ValueError("Constraint names must be non-empty strings")

        coefficients = constraint.get("coefficients")

        if not isinstance(coefficients, dict) or not coefficients:
            raise ValueError(
                f"Constraint {name} needs non-empty coefficients"
            )

        checked_coefficients = {}

        for variable, coefficient in coefficients.items():
            if (
                not isinstance(variable, int)
                or isinstance(variable, bool)
            ):
                raise ValueError(
                    f"Constraint {name} has a non-integer variable index"
                )

            if not 0 <= variable < num_variables:
                raise ValueError(
                    f"Constraint {name} has an out-of-range variable index"
                )

            checked_coefficients[variable] = _finite_number(
                coefficient,
                f"coefficient in {name}",
            )

        sense = constraint.get("sense")

        if sense not in ("<=", ">=", "=="):
            raise ValueError(
                f"Constraint {name} sense must be <=, >= or =="
            )

        rhs = _finite_number(
            constraint.get("rhs"),
            f"right-hand side in {name}",
        )

        normalized.append({
            "name": name,
            "coefficients": checked_coefficients,
            "sense": sense,
            "rhs": rhs,
        })

    return normalized


def _evaluate_constraints(constraints, solution, tolerance):
    violations = []

    for constraint in constraints:
        activity = math.fsum(
            coefficient * solution[index]
            for index, coefficient
            in constraint["coefficients"].items()
        )

        if not math.isfinite(activity):
            raise ValueError(
                f"Constraint activity is not finite: {constraint['name']}"
            )

        sense = constraint["sense"]
        rhs = constraint["rhs"]

        if sense == "<=":
            violation = max(0.0, activity - rhs)
        elif sense == ">=":
            violation = max(0.0, rhs - activity)
        else:
            violation = abs(activity - rhs)

        satisfied = violation <= tolerance

        violations.append({
            "name": constraint["name"],
            "activity": activity,
            "sense": sense,
            "rhs": rhs,
            "violation": violation,
            "satisfied": satisfied,
        })

    return violations


def _dominates(first, second, tolerance):
    """Return True when first dominates second for minimization."""

    no_worse = all(
        first[name] <= second[name] + tolerance
        for name in first
    )

    strictly_better = any(
        first[name] < second[name] - tolerance
        for name in first
    )

    return no_worse and strictly_better


class DecisionSpaceExplorer:
    """Explore Pareto trade-offs over a supplied candidate pool."""

    def __init__(self, tolerance=1e-9):
        self.tolerance = _validate_tolerance(tolerance)

    def analyze(
        self,
        num_variables,
        objectives,
        candidates,
        constraints=None,
    ):
        """Evaluate candidates and identify the feasible Pareto frontier.

        All objectives are minimized.
        """

        normalized_objectives = _normalize_objectives(
            num_variables,
            objectives,
        )

        normalized_constraints = _normalize_constraints(
            num_variables,
            constraints,
        )

        rows, duplicate_count = _validate_candidates(
            num_variables,
            candidates,
        )

        candidate_results = []
        feasible_results = []

        for candidate_index, solution in enumerate(rows):
            objective_values = {}

            for name, terms in normalized_objectives.items():
                objective_values[name] = stable_qubo_energy(
                    num_variables,
                    terms,
                    solution,
                )

            constraint_results = _evaluate_constraints(
                normalized_constraints,
                solution,
                self.tolerance,
            )

            feasible = all(
                item["satisfied"]
                for item in constraint_results
            )

            record = {
                "candidate_index": candidate_index,
                "solution": list(solution),
                "objectives": objective_values,
                "feasible": feasible,
                "constraints": constraint_results,
            }

            candidate_results.append(record)

            if feasible:
                feasible_results.append(record)

        frontier = []

        for candidate in feasible_results:
            candidate_objectives = candidate["objectives"]

            dominated = any(
                _dominates(
                    other["objectives"],
                    candidate_objectives,
                    self.tolerance,
                )
                for other in feasible_results
                if other is not candidate
            )

            if not dominated:
                frontier.append(candidate)

        return {
            "method": "PARETO_FILTER_ON_SUPPLIED_CANDIDATES",
            "objective_direction": "MINIMIZE",
            "candidate_count": len(rows) + duplicate_count,
            "unique_candidate_count": len(rows),
            "duplicate_count": duplicate_count,
            "feasible_count": len(feasible_results),
            "infeasible_count": len(rows) - len(feasible_results),
            "frontier_count": len(frontier),
            "frontier": frontier,
            "candidate_results": candidate_results,
            "objective_names": list(normalized_objectives.keys()),
            "tolerance": self.tolerance,
            "global_optimum_proven": False,
        }

    def analyze_scenarios(
        self,
        num_variables,
        scenarios,
        candidates,
        constraints=None,
    ):
        """Compare the same candidate pool across named scenarios."""

        if not isinstance(scenarios, dict) or not scenarios:
            raise ValueError("scenarios must be a non-empty dictionary")

        if isinstance(candidates, (str, bytes)):
            raise ValueError("Candidates must be iterable binary sequences")

        try:
            candidate_pool = list(candidates)
        except TypeError as exc:
            raise ValueError("Candidates must be iterable") from exc

        if not candidate_pool:
            raise ValueError("At least one candidate is required")

        scenario_results = {}
        frontier_sets = []

        for scenario_name, objectives in scenarios.items():
            if (
                not isinstance(scenario_name, str)
                or not scenario_name.strip()
            ):
                raise ValueError("Scenario names must be non-empty strings")

            report = self.analyze(
                num_variables=num_variables,
                objectives=objectives,
                candidates=candidate_pool,
                constraints=constraints,
            )

            scenario_results[scenario_name] = report

            frontier_sets.append({
                tuple(item["solution"])
                for item in report["frontier"]
            })

        common_frontier = set.intersection(*frontier_sets)

        return {
            "scenario_count": len(scenario_results),
            "scenarios": scenario_results,
            "frontier_in_every_scenario": [
                list(solution)
                for solution in sorted(common_frontier)
            ],
            "interpretation": (
                "Candidates appearing in every supplied scenario's "
                "Pareto frontier. This is not a robustness guarantee "
                "outside the supplied scenarios."
            ),
        }
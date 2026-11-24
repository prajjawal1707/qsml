
"""QSML 4.6 — Decision explanation and contribution analysis."""

import math

from .qubo_contract import normalize_qubo_terms
from .stability import stable_qubo_energy


def explain_decision(
    num_variables,
    solution,
    objectives,
    constraints=None,
    tolerance=1e-9,
):
    """Explain objective term contributions and hard-constraint status."""

    if (
        isinstance(tolerance, bool)
        or not isinstance(tolerance, (int, float))
        or not math.isfinite(tolerance)
        or tolerance < 0
    ):
        raise ValueError("tolerance must be finite and non-negative")

    if isinstance(solution, (str, bytes)):
        raise ValueError("solution must be a binary sequence")

    try:
        bits = list(solution)
    except TypeError as exc:
        raise ValueError("solution must be iterable") from exc

    if len(bits) != num_variables:
        raise ValueError("Solution length does not match num_variables")

    for bit in bits:
        if bit not in (0, 1, False, True):
            raise ValueError("Solution must be binary")

    bits = [int(bit) for bit in bits]

    if not isinstance(objectives, dict) or not objectives:
        raise ValueError("objectives must be a non-empty dictionary")

    objective_explanations = {}

    for name, terms in objectives.items():
        normalized = normalize_qubo_terms(
            num_variables,
            terms,
        )

        contributions = []

        for (i, j), coefficient in normalized.items():
            contribution = coefficient * bits[i] * bits[j]

            if not math.isfinite(contribution):
                raise ValueError(
                    f"Non-finite contribution in objective {name}"
                )

            contributions.append({
                "term": [i, j],
                "coefficient": coefficient,
                "variables": [bits[i], bits[j]],
                "contribution": contribution,
            })

        contributions.sort(
            key=lambda item: abs(item["contribution"]),
            reverse=True,
        )

        energy = stable_qubo_energy(
            num_variables,
            normalized,
            bits,
        )

        objective_explanations[name] = {
            "energy": energy,
            "term_contributions": contributions,
            "contribution_sum": math.fsum(
                item["contribution"]
                for item in contributions
            ),
        }

    constraint_explanations = []

    if constraints is not None:
        if not isinstance(constraints, (list, tuple)):
            raise ValueError("constraints must be a list or tuple")

        for index, constraint in enumerate(constraints):
            if not isinstance(constraint, dict):
                raise ValueError("Each constraint must be a dictionary")

            name = constraint.get("name", f"constraint_{index}")
            coefficients = constraint.get("coefficients")
            sense = constraint.get("sense")
            rhs = constraint.get("rhs")

            if not isinstance(coefficients, dict) or not coefficients:
                raise ValueError(
                    f"Constraint {name} requires coefficients"
                )

            checked = {}

            for variable, coefficient in coefficients.items():
                if (
                    not isinstance(variable, int)
                    or isinstance(variable, bool)
                    or not 0 <= variable < num_variables
                ):
                    raise ValueError(
                        f"Invalid variable index in constraint {name}"
                    )

                if isinstance(coefficient, bool) or not isinstance(
                    coefficient, (int, float)
                ):
                    raise ValueError(
                        f"Invalid coefficient in constraint {name}"
                    )

                coefficient = float(coefficient)

                if not math.isfinite(coefficient):
                    raise ValueError(
                        f"Non-finite coefficient in constraint {name}"
                    )

                checked[variable] = coefficient

            if sense not in ("<=", ">=", "=="):
                raise ValueError(
                    f"Unsupported constraint operator in {name}"
                )

            if isinstance(rhs, bool) or not isinstance(rhs, (int, float)):
                raise ValueError(f"Invalid right-hand side in {name}")

            rhs = float(rhs)

            if not math.isfinite(rhs):
                raise ValueError(f"Non-finite right-hand side in {name}")

            activity = math.fsum(
                coefficient * bits[variable]
                for variable, coefficient in checked.items()
            )

            if sense == "<=":
                violation = max(0.0, activity - rhs)
            elif sense == ">=":
                violation = max(0.0, rhs - activity)
            else:
                violation = abs(activity - rhs)

            constraint_explanations.append({
                "name": name,
                "activity": activity,
                "operator": sense,
                "rhs": rhs,
                "violation": violation,
                "satisfied": violation <= tolerance,
            })

    feasible = all(
        item["satisfied"]
        for item in constraint_explanations
    )

    return {
        "solution": bits,
        "objectives": objective_explanations,
        "constraints": constraint_explanations,
        "feasible": feasible,
        "constraint_violation_total": math.fsum(
            item["violation"]
            for item in constraint_explanations
        ),
        "explanation_method": "TERM_CONTRIBUTION_AND_CONSTRAINT_AUDIT",
    }

"""Reusable QUBO problem builders."""
import math

from qsml.constraints import ConstraintBuilder, add_terms


def _number(value, label, nonnegative=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{label} must be finite")
    if nonnegative and value < 0:
        raise ValueError(f"{label} must be non-negative")
    return value


class ProblemLibrary:
    @staticmethod
    def max_cut(num_vertices, edges):
        """
        Build a QUBO minimizing the negative weighted cut value.

        Each edge is (u, v) or (u, v, weight).
        """
        if not isinstance(num_vertices, int) or isinstance(num_vertices, bool):
            raise ValueError("num_vertices must be an integer")
        if num_vertices < 2:
            raise ValueError("Max-Cut requires at least two vertices")
        if not isinstance(edges, (list, tuple)) or not edges:
            raise ValueError("edges must be a non-empty list or tuple")

        terms = {}
        for edge in edges:
            if not isinstance(edge, (list, tuple)) or len(edge) not in (2, 3):
                raise ValueError("Each edge must contain 2 or 3 values")
            u, v = edge[0], edge[1]
            weight = 1.0 if len(edge) == 2 else _number(
                edge[2], "edge weight", nonnegative=True
            )
            if (
                not isinstance(u, int) or isinstance(u, bool)
                or not isinstance(v, int) or isinstance(v, bool)
                or not 0 <= u < num_vertices
                or not 0 <= v < num_vertices
                or u == v
            ):
                raise ValueError("Invalid edge endpoints")
            terms[(u, u)] = terms.get((u, u), 0.0) - weight
            terms[(v, v)] = terms.get((v, v), 0.0) - weight
            pair = (min(u, v), max(u, v))
            terms[pair] = terms.get(pair, 0.0) + 2.0 * weight

        return {
            "name": "max_cut",
            "num_variables": num_vertices,
            "terms": terms,
            "metadata": {"edges": [list(edge) for edge in edges]},
        }

    @staticmethod
    def knapsack(weights, values, capacity, penalty=10.0):
        """
        Build a QUBO for maximizing item values under an integer capacity.

        Returns a QUBO with original decision bits followed by slack bits.
        """
        if not isinstance(weights, (list, tuple)) or not weights:
            raise ValueError("weights must be a non-empty list")
        if not isinstance(values, (list, tuple)) or len(values) != len(weights):
            raise ValueError("values must match weights in length")

        checked_weights = []
        checked_values = []
        for weight in weights:
            number = _number(weight, "weight", nonnegative=True)
            if not number.is_integer():
                raise ValueError("weights must be integers")
            checked_weights.append(int(number))
        for value in values:
            checked_values.append(_number(value, "value"))

        capacity_value = _number(capacity, "capacity", nonnegative=True)
        if not capacity_value.is_integer():
            raise ValueError("capacity must be an integer")
        capacity_int = int(capacity_value)

        constraint = ConstraintBuilder().at_most(
            len(checked_weights),
            {i: weight for i, weight in enumerate(checked_weights)},
            capacity_int,
            penalty=penalty,
        )

        terms = dict(constraint["terms"])
        for index, value in enumerate(checked_values):
            terms[(index, index)] = terms.get((index, index), 0.0) - value

        return {
            "name": "knapsack",
            "num_variables": constraint["num_variables"],
            "decision_variables": len(checked_weights),
            "slack_variables": constraint["slack_variables"],
            "terms": terms,
            "metadata": {
                "weights": checked_weights,
                "values": checked_values,
                "capacity": capacity_int,
                "penalty": float(penalty),
            },
        }


"""QSML 4.3 — Pareto diversity archive."""

import math


def hamming_distance(first, second):
    """Count binary positions at which two solutions differ."""

    if len(first) != len(second):
        raise ValueError("Solutions must have equal lengths")

    return sum(a != b for a, b in zip(first, second))


def select_diverse_frontier(
    decision_report,
    max_candidates=20,
    min_hamming_distance=1,
):
    """Select a diverse subset of a supplied Pareto frontier.

    Uses a greedy max-min Hamming-distance strategy. This is a diversity
    heuristic, not a guarantee of globally maximum diversity.
    """

    if not isinstance(decision_report, dict):
        raise ValueError("decision_report must be a dictionary")

    if (
        not isinstance(max_candidates, int)
        or isinstance(max_candidates, bool)
        or max_candidates < 1
    ):
        raise ValueError("max_candidates must be positive")

    if (
        not isinstance(min_hamming_distance, int)
        or isinstance(min_hamming_distance, bool)
        or min_hamming_distance < 0
    ):
        raise ValueError("min_hamming_distance must be non-negative")

    frontier = decision_report.get("frontier")

    if not isinstance(frontier, list):
        raise ValueError("decision_report frontier must be a list")

    if not frontier:
        return {
            "selected": [],
            "selected_count": 0,
            "frontier_count": 0,
            "minimum_pairwise_distance": None,
        }

    objective_names = decision_report.get("objective_names", [])

    if not objective_names:
        raise ValueError("No objective names found in decision report")

    # Score is used only to select the initial candidate.
    values = {
        name: [
            item["objectives"][name]
            for item in frontier
        ]
        for name in objective_names
    }

    bounds = {}

    for name in objective_names:
        low = min(values[name])
        high = max(values[name])
        scale = high - low

        if scale == 0:
            scale = max(abs(low), 1.0)

        bounds[name] = (low, scale)

    def scalar_score(item):
        return math.fsum(
            (
                item["objectives"][name] - bounds[name][0]
            ) / bounds[name][1]
            for name in objective_names
        )

    ordered = sorted(frontier, key=scalar_score)
    selected = [ordered[0]]
    remaining = ordered[1:]
    relaxed = False

    while remaining and len(selected) < max_candidates:
        ranked = []

        for candidate in remaining:
            distance = min(
                hamming_distance(
                    candidate["solution"],
                    chosen["solution"],
                )
                for chosen in selected
            )

            ranked.append((distance, scalar_score(candidate), candidate))

        ranked.sort(key=lambda item: (-item[0], item[1]))

        best_distance, _, best_candidate = ranked[0]

        if best_distance < min_hamming_distance:
            relaxed = True

        selected.append(best_candidate)
        remaining.remove(best_candidate)

    pairwise_distances = []

    for i, first in enumerate(selected):
        for second in selected[i + 1:]:
            pairwise_distances.append(
                hamming_distance(first["solution"], second["solution"])
            )

    return {
        "selected": selected,
        "selected_count": len(selected),
        "frontier_count": len(frontier),
        "minimum_pairwise_distance": (
            min(pairwise_distances)
            if pairwise_distances else None
        ),
        "distance_threshold_relaxed": relaxed,
        "method": "GREEDY_MAX_MIN_HAMMING_DISTANCE",
    }

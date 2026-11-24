
"""QSML 4.1 — Active Decision Space Exploration."""

import itertools
import math
import random

from .decision_space import DecisionSpaceExplorer


MAX_CANDIDATE_BUDGET = 1_000_000
MAX_POOL_BITS = 5_000_000
MAX_MUTATION_ROUNDS = 1_000


def _positive_integer(value, name):
    if (
        not isinstance(value, int)
        or isinstance(value, bool)
        or value < 1
    ):
        raise ValueError(f"{name} must be a positive integer")

    return value


def _normalize_candidate(num_variables, candidate, index):
    if isinstance(candidate, (str, bytes)):
        raise ValueError(f"Candidate {index} must be a binary sequence")

    try:
        bits = list(candidate)
    except TypeError as exc:
        raise ValueError(f"Candidate {index} must be iterable") from exc

    if len(bits) != num_variables:
        raise ValueError(f"Candidate {index} has an incorrect length")

    result = []

    for bit in bits:
        if isinstance(bit, bool):
            result.append(int(bit))
        elif bit in (0, 1):
            result.append(int(bit))
        else:
            raise ValueError(
                f"Candidate {index} contains a non-binary value"
            )

    return result


def generate_candidate_pool(
    num_variables,
    candidate_budget=4096,
    seed=42,
    initial_candidates=None,
    mutation_rounds=2,
    mutation_probability=0.15,
):
    """Generate a bounded, reproducible pool of distinct binary solutions.

    Strategies:
    1. Include all-zero and all-one boundary candidates.
    2. Include supplied initial candidate solutions.
    3. Generate single-bit and additional random mutations.
    4. Fill remaining capacity through random sampling or bounded
       enumeration of a small binary search space.

    This function generates candidates; it does not prove optimality.
    """

    num_variables = _positive_integer(
        num_variables,
        "num_variables",
    )

    candidate_budget = _positive_integer(
        candidate_budget,
        "candidate_budget",
    )

    if candidate_budget > MAX_CANDIDATE_BUDGET:
        raise ValueError(
            f"candidate_budget cannot exceed {MAX_CANDIDATE_BUDGET}"
        )

    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ValueError("seed must be an integer")

    if (
        not isinstance(mutation_rounds, int)
        or isinstance(mutation_rounds, bool)
        or mutation_rounds < 0
        or mutation_rounds > MAX_MUTATION_ROUNDS
    ):
        raise ValueError(
            f"mutation_rounds must be between 0 and "
            f"{MAX_MUTATION_ROUNDS}"
        )

    if (
        isinstance(mutation_probability, bool)
        or not isinstance(mutation_probability, (int, float))
    ):
        raise ValueError("mutation_probability must be numeric")

    mutation_probability = float(mutation_probability)

    if (
        not math.isfinite(mutation_probability)
        or not 0.0 <= mutation_probability <= 1.0
    ):
        raise ValueError(
            "mutation_probability must be between 0 and 1"
        )

    total_possible = 1 << num_variables
    effective_budget = min(candidate_budget, total_possible)

    if effective_budget * num_variables > MAX_POOL_BITS:
        raise ValueError(
            "Requested candidate pool exceeds the configured "
            "memory-safety limit. Reduce candidate_budget or "
            "num_variables."
        )

    # Validate supplied initial candidates before any early return.
    initial_rows = []

    if initial_candidates is not None:
        if isinstance(initial_candidates, (str, bytes)):
            raise ValueError(
                "initial_candidates must be a collection of binary sequences"
            )

        try:
            raw_initial_rows = list(initial_candidates)
        except TypeError as exc:
            raise ValueError(
                "initial_candidates must be iterable"
            ) from exc

        for index, candidate in enumerate(raw_initial_rows):
            initial_rows.append(
                _normalize_candidate(
                    num_variables,
                    candidate,
                    index,
                )
            )

    # For a small enough space, enumerate every possible assignment.
    if total_possible <= candidate_budget:
        return [
            list(bits)
            for bits in itertools.product(
                (0, 1),
                repeat=num_variables,
            )
        ]

    rng = random.Random(seed)
    candidates = []
    seen = set()

    def add_candidate(candidate):
        if len(candidates) >= effective_budget:
            return False

        key = tuple(candidate)

        if key in seen:
            return False

        seen.add(key)
        candidates.append(list(candidate))
        return True

    zero_candidate = [0] * num_variables
    one_candidate = [1] * num_variables

    add_candidate(zero_candidate)
    add_candidate(one_candidate)

    for candidate in initial_rows:
        add_candidate(candidate)

    mutation_seeds = [
        list(candidate)
        for candidate in candidates
    ]

    # Generate local mutations.
    for base in mutation_seeds:
        for _ in range(mutation_rounds):
            if len(candidates) >= effective_budget:
                break

            mutated = list(base)

            # Always flip one randomly selected bit.
            first_index = rng.randrange(num_variables)
            mutated[first_index] = 1 - mutated[first_index]

            # Other bits may also be flipped.
            for index in range(num_variables):
                if index == first_index:
                    continue

                if rng.random() < mutation_probability:
                    mutated[index] = 1 - mutated[index]

            add_candidate(mutated)

        if len(candidates) >= effective_budget:
            break

    # Fill the remaining budget.
    if total_possible <= 1_000_000:
        # Avoid repeated random collisions near the edge of a small space.
        for bits in itertools.product(
            (0, 1),
            repeat=num_variables,
        ):
            if len(candidates) >= effective_budget:
                break

            add_candidate(bits)

    else:
        while len(candidates) < effective_budget:
            random_candidate = [
                rng.randint(0, 1)
                for _ in range(num_variables)
            ]

            add_candidate(random_candidate)

    return candidates


class ActiveDecisionExplorer:
    """Generate candidate decisions and explore their Pareto frontier."""

    def __init__(self, tolerance=1e-9):
        self.explorer = DecisionSpaceExplorer(
            tolerance=tolerance
        )
        self.tolerance = tolerance

    def generate(
        self,
        num_variables,
        candidate_budget=4096,
        seed=42,
        initial_candidates=None,
        mutation_rounds=2,
        mutation_probability=0.15,
    ):
        """Generate a candidate pool without objective analysis."""

        return generate_candidate_pool(
            num_variables=num_variables,
            candidate_budget=candidate_budget,
            seed=seed,
            initial_candidates=initial_candidates,
            mutation_rounds=mutation_rounds,
            mutation_probability=mutation_probability,
        )

    def explore(
        self,
        num_variables,
        objectives,
        candidate_budget=4096,
        seed=42,
        initial_candidates=None,
        mutation_rounds=2,
        mutation_probability=0.15,
        constraints=None,
    ):
        """Generate candidates and analyze their feasible Pareto frontier."""

        candidates = self.generate(
            num_variables=num_variables,
            candidate_budget=candidate_budget,
            seed=seed,
            initial_candidates=initial_candidates,
            mutation_rounds=mutation_rounds,
            mutation_probability=mutation_probability,
        )

        report = self.explorer.analyze(
            num_variables=num_variables,
            objectives=objectives,
            candidates=candidates,
            constraints=constraints,
        )

        report["generation"] = {
            "method": "BOUNDARY_MUTATION_AND_RANDOM_SAMPLING",
            "seed": seed,
            "requested_budget": candidate_budget,
            "generated_candidates": len(candidates),
            "mutation_rounds": mutation_rounds,
            "mutation_probability": mutation_probability,
            "initial_candidates_provided": (
                initial_candidates is not None
            ),
        }

        report["global_optimum_proven"] = False

        return report

    def explore_scenarios(
        self,
        num_variables,
        scenarios,
        candidate_budget=4096,
        seed=42,
        initial_candidates=None,
        mutation_rounds=2,
        mutation_probability=0.15,
        constraints=None,
    ):
        """Generate one shared candidate pool and compare scenarios."""

        candidates = self.generate(
            num_variables=num_variables,
            candidate_budget=candidate_budget,
            seed=seed,
            initial_candidates=initial_candidates,
            mutation_rounds=mutation_rounds,
            mutation_probability=mutation_probability,
        )

        report = self.explorer.analyze_scenarios(
            num_variables=num_variables,
            scenarios=scenarios,
            candidates=candidates,
            constraints=constraints,
        )

        report["generation"] = {
            "method": "SHARED_BOUNDARY_MUTATION_RANDOM_POOL",
            "seed": seed,
            "requested_budget": candidate_budget,
            "generated_candidates": len(candidates),
        }

        return report


"""QSML 4.7 — Multi-solver candidate integration."""

from .stability import stable_qubo_energy


def _get(result, key, default=None):
    if isinstance(result, dict):
        return result.get(key, default)

    return getattr(result, key, default)


def _validate_bits(num_variables, solution):
    if isinstance(solution, (str, bytes)):
        raise ValueError("Solver solution must be a binary sequence")

    try:
        bits = list(solution)
    except TypeError as exc:
        raise ValueError("Solver solution must be iterable") from exc

    if len(bits) != num_variables:
        raise ValueError("Solver solution has an incorrect length")

    normalized = []

    for bit in bits:
        if isinstance(bit, bool):
            normalized.append(int(bit))
        elif bit in (0, 1):
            normalized.append(int(bit))
        else:
            raise ValueError("Solver solution contains a non-binary value")

    return normalized


def merge_solver_candidates(
    num_variables,
    objective_terms,
    solver_results,
    supplied_candidates=None,
):
    """Merge candidate solutions and retain source provenance.

    objective_terms is the QUBO objective used to verify reported solver
    energies. This function does not compare unlike objective functions.
    """

    if not isinstance(solver_results, dict):
        raise ValueError("solver_results must be a dictionary")

    candidates = []
    sources = {}
    errors = []

    def add(candidate, source):
        key = tuple(candidate)

        if key not in sources:
            sources[key] = []

        if source not in sources[key]:
            sources[key].append(source)

        if not any(tuple(item) == key for item in candidates):
            candidates.append(list(candidate))

    if supplied_candidates is not None:
        if isinstance(supplied_candidates, (str, bytes)):
            raise ValueError("supplied_candidates must be iterable")

        try:
            rows = list(supplied_candidates)
        except TypeError as exc:
            raise ValueError(
                "supplied_candidates must be iterable"
            ) from exc

        for index, solution in enumerate(rows):
            try:
                bits = _validate_bits(num_variables, solution)
                add(bits, "SUPPLIED_CANDIDATE")
            except ValueError as exc:
                errors.append({
                    "source": f"SUPPLIED_{index}",
                    "error": str(exc),
                })

    for solver_name, result in solver_results.items():
        try:
            if not isinstance(solver_name, str) or not solver_name.strip():
                raise ValueError("Solver name must be a non-empty string")

            solution = _get(result, "solution")

            if solution is None:
                raise ValueError("Solver did not provide a solution")

            bits = _validate_bits(num_variables, solution)

            actual_energy = stable_qubo_energy(
                num_variables,
                objective_terms,
                bits,
            )

            reported_energy = _get(result, "energy")

            if reported_energy is not None:
                if isinstance(reported_energy, bool) or not isinstance(
                    reported_energy, (int, float)
                ):
                    raise ValueError("Reported energy must be numeric")

                if abs(float(reported_energy) - actual_energy) > (
                    1e-8 * max(1.0, abs(actual_energy))
                ):
                    raise ValueError(
                        "Reported energy does not match independent energy"
                    )

            add(bits, solver_name)

        except (ValueError, TypeError, OverflowError) as exc:
            errors.append({
                "source": str(solver_name),
                "error": str(exc),
            })

    return {
        "candidates": candidates,
        "candidate_count": len(candidates),
        "source_count": len(solver_results),
        "provenance": [
            {
                "solution": list(solution),
                "sources": source_names,
            }
            for solution, source_names in sources.items()
        ],
        "errors": errors,
        "all_solver_results_valid": len(errors) == 0,
    }

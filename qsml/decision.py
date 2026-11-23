"""QSML Decision Engine.

Analyses possibility results and selects the strongest candidate.
This is classical statistical decision logic operating on simulator output.
"""

from dataclasses import dataclass
import math


@dataclass
class DecisionReport:
    selected_state: str
    confidence: float
    measured_frequency: float
    entropy_bits: float
    normalized_uncertainty: float
    ranked_candidates: list


class DecisionEngine:
    """Analyse candidate probabilities and measurement results."""

    def analyse(self, result) -> DecisionReport:
        probabilities = dict(result.top_candidates)
        measurements = result.measurements

        if not probabilities:
            raise ValueError("No candidate probabilities available")

        total_probability = sum(probabilities.values())

        if total_probability <= 0:
            raise ValueError("Candidate probabilities must have a positive sum")

        # Normalize probabilities to account for rounding.
        probabilities = {
            state: probability / total_probability
            for state, probability in probabilities.items()
        }

        # Rank candidates from strongest to weakest.
        ranked = sorted(
            probabilities.items(),
            key=lambda item: item[1],
            reverse=True,
        )

        selected_state, confidence = ranked[0]

        total_shots = sum(measurements.values())

        if total_shots > 0:
            measured_frequency = measurements.get(selected_state, 0) / total_shots
        else:
            measured_frequency = 0.0

        # Shannon entropy: measures uncertainty in the distribution.
        entropy = -sum(p * math.log2(p) for p in probabilities.values() if p > 0)

        candidate_count = len(probabilities)

        if candidate_count > 1:
            maximum_entropy = math.log2(candidate_count)
            normalized_uncertainty = entropy / maximum_entropy
        else:
            normalized_uncertainty = 0.0

        return DecisionReport(
            selected_state=selected_state,
            confidence=confidence,
            measured_frequency=measured_frequency,
            entropy_bits=entropy,
            normalized_uncertainty=normalized_uncertainty,
            ranked_candidates=ranked,
        )

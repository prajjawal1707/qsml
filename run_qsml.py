
from pathlib import Path

from qsml.interpreter import execute_program
from qsml.decision import DecisionEngine


program_path = Path("programs/first_search.qsml")
source = program_path.read_text(encoding="utf-8")

print("=" * 55)
print("QSML 0.5 — QUANTUM-INSPIRED DECISION ENGINE")
print("=" * 55)

# Execute the QSML program.
output = execute_program(source, seed=42)

print("\n[1] PROGRAM INFORMATION")
print("Registered qubits:", output["qubits"])
print("Marked states:", output["marked_states"])

result = output["result"]

# Analyse the search results.
engine = DecisionEngine()
report = engine.analyse(result)

print("\n[2] CANDIDATE ANALYSIS")

for state, probability in report.ranked_candidates:
    print(f"{state} : {probability:.3%}")

print("\n[3] DECISION ENGINE")
print("Selected answer:", report.selected_state)
print(f"Probability confidence: {report.confidence:.3%}")
print(f"Measured frequency: {report.measured_frequency:.3%}")

print("\n[4] UNCERTAINTY ANALYSIS")
print(f"Entropy: {report.entropy_bits:.4f} bits")
print(
    f"Normalized uncertainty: "
    f"{report.normalized_uncertainty:.3%}"
)

print("\n[5] FINAL DECISION")
print("QSML selected:", report.selected_state)

print("\nQSML 0.5 EXECUTION COMPLETE")

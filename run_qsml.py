
from pathlib import Path

from qsml.interpreter import execute_program


program_path = Path("programs/first_search.qsml")

source = program_path.read_text(encoding="utf-8")

print("=" * 50)
print("QSML PROGRAM EXECUTION")
print("=" * 50)

output = execute_program(source, seed=42)

print("\nREGISTER:", output["qubits"], "qubits")
print("MARKED STATES:", output["marked_states"])

result = output["result"]

print("\nTOP CANDIDATES")

for state, probability in result.top_candidates:
    print(f"{state} : {probability:.3%}")

print("\nMEASUREMENT RESULTS")
print(result.measurements)

print("\nQSML PROGRAM FINISHED")

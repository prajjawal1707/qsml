
from qsml.grover_gpu import GPUGroverSearch

print("=" * 55)
print("QSML 1.2 — GPU GROVER SEARCH")
print("=" * 55)

num_qubits = 3
marked_states = ["101"]

engine = GPUGroverSearch(num_qubits)

print("\n[1] SEARCH CONFIGURATION")
print("Qubits:", num_qubits)
print("Search space:", 2 ** num_qubits)
print("Marked states:", marked_states)

print("\n[2] GPU SEARCH")

result = engine.search(
    marked_states=marked_states,
    iterations=2,
    shots=1000,
    seed=42,
)

print("Iterations:", result["iterations"])

print("\n[3] TOP CANDIDATES")

for state, probability in result["top_candidates"]:
    print(f"{state}: {probability:.3%}")

print("\n[4] MEASUREMENTS")
print(result["measurements"])

print("\n[5] STATE VALIDATION")
print(result["state_validation"])

print("\n[6] FINAL RESULT")

best_state = result["top_candidates"][0][0]

print("Selected candidate:", best_state)
print("Expected answer:", marked_states[0])

if best_state in marked_states:
    print("GPU GROVER SEARCH: PASSED")
else:
    print("GPU GROVER SEARCH: CHECK RESULT")

print("\nQSML 1.2 EXECUTION COMPLETE")


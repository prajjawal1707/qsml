
from pathlib import Path

from qsml.interpreter import execute_program
from qsml.decision import DecisionEngine
from qsml.compute import plan_execution


program_path = Path("programs/first_search.qsml")
source = program_path.read_text(encoding="utf-8")

print("=" * 55)
print("QSML 0.6 — ADAPTIVE HARDWARE ENGINE")
print("=" * 55)

from qsml.language import parse_program

# 1. Read the workload size before allocating simulator memory.
instructions = parse_program(source)

register_instruction = next(
    (item for item in instructions if item["command"] == "REGISTER"),
    None,
)

if register_instruction is None:
    raise ValueError("Program must contain a REGISTER instruction")

num_qubits = register_instruction["qubits"]

# 2. Check resources before running the simulator.
print("\n[1] ADAPTIVE HARDWARE PLANNER")

plan = plan_execution(num_qubits)

print("Recommended device:", plan.recommended_device)
print(f"Estimated memory: {plan.estimated_memory_mb:.4f} MB")
print(f"Available system RAM: {plan.cpu_available_mb:.2f} MB")

if plan.gpu_available_mb is not None:
    print(f"Available GPU memory: {plan.gpu_available_mb:.2f} MB")
else:
    print("CUDA GPU backend: Not available to QSML")

print("Reason:", plan.reason)

if not plan.can_run:
    raise MemoryError("QSML stopped: insufficient safe memory.")

# 3. Execute only after the resource check passes.
output = execute_program(source, seed=42)

print("\n[2] PROGRAM INFORMATION")
print("Registered qubits:", output["qubits"])
print("Marked states:", output["marked_states"])

# 4. Analyse the result.
result = output["result"]
engine = DecisionEngine()
report = engine.analyse(result)

print("\n[3] DECISION ENGINE")
print("Selected answer:", report.selected_state)
print(f"Probability confidence: {report.confidence:.3%}")
print(f"Measured frequency: {report.measured_frequency:.3%}")
print(f"Normalized uncertainty: {report.normalized_uncertainty:.3%}")

print("\n[4] EXECUTION STATUS")
print("Resource check: PASSED")
print("Search result:", report.selected_state)

print("\nQSML 0.6 HARDWARE PLANNING COMPLETE")
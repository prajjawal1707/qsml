
import math

from qsml.register_gpu import GPUQuantumRegister


print("=" * 55)
print("QSML 1.1 — GPU QUANTUM GATE TESTS")
print("=" * 55)

tests = [
    ("X", lambda q: q.x(0)),
    ("Y", lambda q: q.y(0)),
    ("Z", lambda q: q.z(0)),
    ("H", lambda q: q.h(0)),
    ("S", lambda q: q.s(0)),
    ("T", lambda q: q.t(0)),
    ("RX", lambda q: q.rx(0, math.pi / 3)),
    ("RY", lambda q: q.ry(0, math.pi / 3)),
    ("RZ", lambda q: q.rz(0, math.pi / 3)),
]

print("\n[1] SINGLE-QUBIT GATES")

for name, operation in tests:
    register = GPUQuantumRegister(1)
    operation(register)

    check = register.validate_state()

    status = "PASSED" if check["normalized"] else "FAILED"

    print(
        f"{name:4} | "
        f"Norm: {check['norm']:.10f} | "
        f"{status}"
    )

print("\n[2] CNOT TEST")

register = GPUQuantumRegister(2)
register.x(0)
register.cnot(0, 1)

print("State:", register.amplitudes())
print("Validation:", register.validate_state())

print("\n[3] BELL STATE REGRESSION TEST")

bell = GPUQuantumRegister(2)
bell.h(0)
bell.cnot(0, 1)

print("Probabilities:", bell.probabilities().get())
print("Measurements:", bell.measure(1000, seed=42))
print("Validation:", bell.validate_state())

print("\nQSML 1.1 GATE TESTS COMPLETE")

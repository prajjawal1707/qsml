
from qsml.register_gpu import GPUQuantumRegister

print("=" * 55)
print("QSML 1.0 — GPU STATE-VECTOR SIMULATOR")
print("=" * 55)

# Create two qubits.
register = GPUQuantumRegister(2)

print("\n[1] GPU REGISTER")
print("Qubits:", register.num_qubits)
print("State-vector size:", register.size)

# Create a Bell state.
register.h(0)
register.cnot(0, 1)

print("\n[2] GPU STATE AMPLITUDES")

for index, amplitude in enumerate(register.amplitudes()):
    state = format(index, "02b")
    print(f"|{state}> : {amplitude.real:.4f} "
          f"+ {amplitude.imag:.4f}i")

print("\n[3] STATE PROBABILITIES")

probabilities = register.probabilities().get()

for index, probability in enumerate(probabilities):
    state = format(index, "02b")
    print(f"{state}: {probability:.3%}")

print("\n[4] GPU MEASUREMENTS")
print(register.measure(shots=1000, seed=42))

print("\nQSML 1.0 GPU SIMULATOR COMPLETE")

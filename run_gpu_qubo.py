import time

from qsml.qubo_gpu import GPUQUBOEngine

print("=" * 55)
print("QSML 0.9 — GPU QUBO ENGINE")
print("=" * 55)

# Triangle graph Max-Cut problem.
num_vertices = 3

edges = [
    (0, 1, 1.0),
    (1, 2, 1.0),
    (0, 2, 1.0),
]

qubo_terms = {}

for i, j, weight in edges:
    qubo_terms[(i, i)] = qubo_terms.get((i, i), 0.0) - weight
    qubo_terms[(j, j)] = qubo_terms.get((j, j), 0.0) - weight
    qubo_terms[(i, j)] = qubo_terms.get((i, j), 0.0) + 2 * weight

print("\n[1] GPU INITIALIZATION")

engine = GPUQUBOEngine()

print("CUDA device:", engine.device)

print("\n[2] GPU OPTIMIZATION")

start = time.perf_counter()

result = engine.solve(
    num_variables=num_vertices,
    terms=qubo_terms,
    candidates=4096,
    seed=42,
)

elapsed = time.perf_counter() - start

cut_weight = sum(
    weight for i, j, weight in edges if result.solution[i] != result.solution[j]
)

print("Solution:", result.solution)
print("QUBO energy:", result.energy)
print("Cut weight:", cut_weight)
print("Candidates evaluated:", result.evaluations)

print(f"\nGPU execution time: {elapsed * 1000:.3f} ms")

print("\n[3] VALIDATION")

if cut_weight == 2 and result.energy == -2:
    print("GPU Max-Cut: OPTIMAL")
else:
    print("GPU search completed; optimal solution not found in this batch.")

print("\nQSML 0.9 GPU EXECUTION COMPLETE")


from qsml.qubo import QUBOEngine


print("=" * 55)
print("QSML 0.8 — QUBO OPTIMIZATION ENGINE")
print("=" * 55)

# Triangle graph: each edge has weight 1.
num_vertices = 3

edges = [
    (0, 1, 1.0),
    (1, 2, 1.0),
    (0, 2, 1.0),
]

# Convert Max-Cut into a QUBO minimization problem.
#
# Max-Cut objective:
#   maximize sum(w * (x_i + x_j - 2*x_i*x_j))
#
# Equivalent QUBO energy:
#   minimize sum(w * (-x_i - x_j + 2*x_i*x_j))

qubo_terms = {}

for i, j, weight in edges:
    qubo_terms[(i, i)] = qubo_terms.get((i, i), 0.0) - weight
    qubo_terms[(j, j)] = qubo_terms.get((j, j), 0.0) - weight
    qubo_terms[(i, j)] = qubo_terms.get((i, j), 0.0) + 2 * weight

print("\n[1] PROBLEM")
print("Vertices:", num_vertices)
print("Edges:", len(edges))
print("Problem: Maximum Cut")

# Solve the QUBO.
engine = QUBOEngine()

result = engine.solve(
    num_variables=num_vertices,
    terms=qubo_terms,
    iterations=15000,
    restarts=10,
    seed=42,
)

# Calculate the cut weight from the selected binary assignment.
cut_weight = sum(
    weight
    for i, j, weight in edges
    if result.solution[i] != result.solution[j]
)

group_a = [
    i for i, value in enumerate(result.solution) if value == 0
]

group_b = [
    i for i, value in enumerate(result.solution) if value == 1
]

print("\n[2] QUBO RESULT")
print("Binary solution:", result.solution)
print(f"QUBO energy: {result.energy:.4f}")

print("\n[3] GRAPH PARTITION")
print("Group A:", group_a)
print("Group B:", group_b)
print(f"Cut weight: {cut_weight:.2f}")

print("\n[4] VALIDATION")

if result.energy <= 0 and cut_weight == 2:
    print("Max-Cut result: OPTIMAL for this triangle")
else:
    print("A valid solution was found; compare with the known optimum of 2.")

print("Evaluations:", result.evaluations)
print("\nQSML 0.8 QUBO EXECUTION COMPLETE")

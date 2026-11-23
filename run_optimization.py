from qsml.optimization import OptimizationEngine

print("=" * 55)
print("QSML 0.7 — QUANTUM-INSPIRED OPTIMIZATION")
print("=" * 55)

# Example: select items without exceeding the capacity.
items = [
    {"name": "Item A", "weight": 2, "value": 3},
    {"name": "Item B", "weight": 3, "value": 4},
    {"name": "Item C", "weight": 4, "value": 5},
    {"name": "Item D", "weight": 5, "value": 8},
    {"name": "Item E", "weight": 9, "value": 10},
    {"name": "Item F", "weight": 7, "value": 7},
]

capacity = 12

weights = [item["weight"] for item in items]
values = [item["value"] for item in items]

engine = OptimizationEngine()

result = engine.solve_knapsack(
    weights=weights,
    values=values,
    capacity=capacity,
    iterations=12000,
    restarts=8,
    seed=42,
)

print("\n[1] OPTIMIZATION PROBLEM")
print("Available items:", len(items))
print("Maximum capacity:", capacity)

print("\n[2] SELECTED ITEMS")

for index in result.selected_items:
    item = items[index]
    print(f"{item['name']} | " f"Weight: {item['weight']} | " f"Value: {item['value']}")

print("\n[3] OPTIMIZATION RESULT")
print("Selection vector:", result.selection)
print("Total weight:", result.total_weight)
print("Total value:", result.total_value)
print("Capacity:", result.capacity)
print("Evaluations:", result.evaluations)

print("\n[4] VALIDATION")

if result.total_weight <= capacity:
    print("Capacity constraint: PASSED")
else:
    print("Capacity constraint: FAILED")

print("\nQSML 0.7 OPTIMIZATION COMPLETE")

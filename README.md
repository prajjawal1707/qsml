
# QSML — Quantum-inspired Search and Machine Learning

**Current development milestone: QSML 3.1.3**

QSML is a classical quantum-inspired optimization and simulation project. It combines classical state-vector simulation, binary optimization, QUBO modeling, constraint penalties, CPU/GPU execution, and independent mathematical validation.

QSML does not create physical qubits or execute quantum circuits on a quantum computer. Its GPU operations use classical CUDA computation.

---

## 1. Project objectives

The main objectives of QSML are:

- Provide a classical state-vector simulation environment.
- Support binary quadratic optimization problems.
- Convert supported binary constraints into QUBO penalties.
- Execute optimization workloads using CPU and GPU backends.
- Validate optimization results independently.
- Compare solution quality and numerical consistency.
- Record and benchmark optimization experiments.

---

## 2. Current development milestones

| Version | Feature |
|---|---|
| 0.1 | CPU state-vector simulator |
| 0.2 | Hardware and resource awareness |
| 0.3 | Grover search simulation |
| 0.4 | QSML instruction language |
| 0.5 | Decision engine |
| 0.6 | Compute and memory estimation |
| 0.7 | Classical simulated annealing |
| 0.8 | CPU QUBO optimizer |
| 0.9 | GPU QUBO candidate search |
| 1.0–1.3 | GPU state-vector simulator and unified runtime |
| 1.4 | Adaptive optimizer |
| 2.0–2.5 | Unified engine, batching, validation, quality analysis and benchmarking |
| 2.6–3.0 | Experiment storage, constraints, problem library and public API |
| 3.1.1 | Shared QUBO input validation |
| 3.1.2 | Constraint mathematical validation |
| 3.1.3 | CPU/GPU numerical consistency |

---

## 3. QUBO mathematical model

QSML uses the binary quadratic objective:

\[
E(x)=\sum_{(i,j)}Q_{ij}x_ix_j
\]

where:

- \(x_i \in \{0,1\}\) is a binary variable.
- \(Q_{ij}\) is a finite numerical coefficient.
- \(E(x)\) is the energy to minimize.

The implementation stores each pair in canonical form, with \(i \leq j\).

The identity \(x_i^2=x_i\) applies because the variables are binary.

---

## 4. Equality constraint penalty

QSML supports equality constraints of the form:

\[
\sum_i a_ix_i=b
\]

The squared penalty is:

\[
P\left(\sum_i a_ix_i-b\right)^2
\]

For QUBO construction, QSML omits the constant term \(Pb^2\), giving:

\[
\boxed{
E(x)=P\left[
\left(\sum_i a_ix_i-b\right)^2-b^2
\right]
}
\]

Definitions:

- \(a_i\): coefficient of binary variable \(x_i\).
- \(b\): required equality value.
- \(P\): positive penalty strength.
- \(x_i\): binary decision variable.

The omitted constant does not change which assignments minimize the penalty.

### Expanded form

Using \(x_i^2=x_i\):

\[
E(x)=
P\left[
\sum_i(a_i^2-2ba_i)x_i
+
2\sum_{i<j}a_ia_jx_ix_j
\right]
\]

The equality penalty reaches its minimum when the original equality is satisfied.

### Python example

```python
from qsml.constraints import ConstraintBuilder

builder = ConstraintBuilder()

equality = builder.equality(
    num_variables=3,
    coefficients={
        0: 1,
        1: 2,
        2: 1,
    },
    rhs=2,
    penalty=10,
)

print(equality["terms"])
```

---

## 5. At-most constraint penalty

QSML supports constraints of the form:

\[
\sum_i a_ix_i\leq b
\]

A binary slack expression is introduced:

\[
S(s)=\sum_j w_js_j
\]

where:

- \(s_j\in\{0,1\}\) are slack variables.
- \(w_j\) are positive binary weights.
- \(b\) is the upper limit.

The at-most constraint is converted into:

\[
\sum_i a_ix_i+S(s)=b
\]

The corresponding squared penalty is:

\[
P\left(
\sum_i a_ix_i+\sum_jw_js_j-b
\right)^2
\]

QSML omits the constant term \(Pb^2\), so the QUBO energy is:

\[
\boxed{
E(x,s)=P\left[
\left(
\sum_i a_ix_i+\sum_jw_js_j-b
\right)^2-b^2
\right]
}
\]

For non-negative integer coefficients and limits, the slack weights are generated using binary powers:

\[
w_j=1,2,4,8,\ldots
\]

The weights continue until their sum is at least the limit. This allows the required integer residuals from zero through the limit to be represented.

### Python example

```python
from qsml.constraints import ConstraintBuilder

builder = ConstraintBuilder()

constraint = builder.at_most(
    num_variables=2,
    coefficients={
        0: 2,
        1: 1,
    },
    limit=3,
    penalty=10,
)

print("Total variables:", constraint["num_variables"])
print("Slack variables:", constraint["slack_variables"])
print("Slack weights:", constraint["slack_weights"])
print("QUBO terms:", constraint["terms"])
```

The returned variable count includes the additional slack variables.

### Combining objective and constraint terms

```python
from qsml.constraints import ConstraintBuilder

builder = ConstraintBuilder()

objective_terms = {
    (0, 0): -2.0,
    (1, 1): -1.0,
    (0, 1): 0.5,
}

constraint = builder.equality(
    num_variables=2,
    coefficients={
        0: 1,
        1: 1,
    },
    rhs=1,
    penalty=10,
)

combined_terms = builder.combine(
    objective_terms,
    constraint["terms"],
)

print(combined_terms)
```

**Penalty selection:** The penalty must be sufficiently large to discourage constraint violations relative to improvements in the original objective. An excessively large penalty can also make numerical optimization harder. Penalty strength should therefore be tested rather than assumed to be universally correct.

---

## 6. Independent constraint validation

QSML 3.1.2 introduces exhaustive mathematical checks for small constraint problems.

Equality validation:

```python
from qsml.constraints import ConstraintBuilder
from qsml.constraint_validation import (
    validate_equality_encoding,
)

builder = ConstraintBuilder()

coefficients = {
    0: 1,
    1: 2,
    2: 1,
}

encoding = builder.equality(
    3,
    coefficients,
    rhs=2,
    penalty=10,
)

report = validate_equality_encoding(
    encoding,
    coefficients,
    rhs=2,
)

print(report)
```

At-most validation:

```python
from qsml.constraints import ConstraintBuilder
from qsml.constraint_validation import (
    validate_at_most_encoding,
)

builder = ConstraintBuilder()

coefficients = {
    0: 2,
    1: 1,
}

encoding = builder.at_most(
    2,
    coefficients,
    limit=3,
    penalty=10,
)

report = validate_at_most_encoding(
    encoding,
    coefficients,
    limit=3,
)

print(report)
```

These validators enumerate small binary assignments and check that the generated QUBO matches the original penalty expressions.

Exhaustive validation is intended for small test cases because its computational cost grows exponentially with the number of variables.

---

## 7. CPU and GPU optimization

QSML provides different classical search methods.

### CPU

The CPU QUBO engine uses simulated annealing.

```python
from qsml.qubo import QUBOEngine

engine = QUBOEngine()

result = engine.solve(
    num_variables=3,
    terms={
        (0, 0): -2.0,
        (1, 1): -2.0,
        (0, 1): 1.0,
    },
    iterations=5000,
    restarts=4,
    seed=42,
)

print(result)
```

### GPU

The GPU QUBO engine uses parallel random candidate generation and evaluation through CuPy.

```python
from qsml.qubo_gpu import GPUQUBOEngine

engine = GPUQUBOEngine()

result = engine.solve(
    num_variables=3,
    terms={
        (0, 0): -2.0,
        (1, 1): -2.0,
        (0, 1): 1.0,
    },
    candidates=8192,
    seed=42,
)

print(result)
```

These search methods are different. Their solution quality and runtime must be compared using clearly stated workloads and evaluation budgets.

---

## 8. QSML 3.1.3 — Numerical consistency

QSML 3.1.3 evaluates the same candidate assignments using both CPU and GPU arithmetic.

Example:

```python
from qsml.consistency import (
    compare_cpu_gpu_energies,
)

terms = {
    (0, 0): -2.0,
    (1, 1): -2.0,
    (0, 1): 1.0,
}

candidates = [
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1],
]

report = compare_cpu_gpu_energies(
    num_variables=2,
    terms=terms,
    candidates=candidates,
)

print("Valid:", report.valid)
print("Samples:", report.sample_count)
print("Maximum absolute error:", report.max_absolute_error)
print("Maximum relative error:", report.max_relative_error)
print("CPU energies:", report.cpu_energies)
print("GPU energies:", report.gpu_energies)
print("Errors:", report.errors)
```

The default comparison tolerances are:

- Absolute tolerance: \(10^{-10}\)
- Relative tolerance: \(10^{-10}\)

Small floating-point differences are possible because CPU and GPU arithmetic may sum terms in different orders.

This check validates numerical agreement for the supplied assignments. It does not establish that the optimization algorithms have identical search behavior.

---

## 9. Validation and reliability principles

QSML uses independent validation to check:

- QUBO input structure.
- Binary solution validity.
- Reported energy against independently calculated energy.
- Exact small-problem reference solutions.
- Constraint penalty correctness.
- CPU/GPU energy agreement.
- Numerical overflow and non-finite results.

The exact solver is intended for small reference problems and testing, not large-scale optimization.

---

## 10. Running the complete test suite

From Windows PowerShell in the project root:

```powershell
.\.venv-gpu\Scripts\Activate.ps1
python run_qsml.py
```

The complete test suite should include all earlier tests plus the new milestone tests.

| Milestone | Tests |
|---|---:|
| Existing QSML tests through 3.0 | 50 |
| QSML 3.1.1 | 16 |
| QSML 3.1.2 | 21 |
| QSML 3.1.3 | 12 |
| **Expected total** | **99** |

The final summary must calculate its total from the actual pass and failure counters.

---

## 11. Current limitations

- QSML is a classical quantum-inspired project, not a physical quantum computer.
- CPU simulated annealing and GPU random candidate search are different optimization methods.
- GPU acceleration is not guaranteed to be faster for every workload.
- Exhaustive mathematical validation is limited to small problems.
- Penalty strength affects the quality and feasibility of constrained solutions.
- CPU/GPU numerical agreement does not prove global optimization quality.

---

## 12. Future development

The next reliability milestones should focus on:

1. Numerical stability under larger coefficient ranges.
2. Resource and memory safety under larger workloads.
3. Controlled CPU/GPU benchmarking.
4. Larger randomized and structured QUBO test cases.
5. Broader constraint and problem-library validation.

The application interface will be considered after the core engine is sufficiently reliable.


## 13. QSML 3.1.4 — Numerical Stability and Stress Testing

QSML 3.1.4 adds numerical reliability utilities for QUBO energy evaluation and controlled stress testing.

### 13.1 Accurate QUBO energy calculation

The objective remains:

\[
E(x)=\sum_{(i,j)}Q_{ij}x_ix_j
\]

The `stable_qubo_energy()` function uses Python's `math.fsum()` to reduce floating-point summation error.

Example:

```python
from qsml.stability import stable_qubo_energy

terms = {
    (0, 0): 1e16,
    (1, 1): 1.0,
    (2, 2): -1e16,
}

energy = stable_qubo_energy(
    num_variables=3,
    terms=terms,
    solution=[1, 1, 1],
)

print("Energy:", energy)
```

Expected result:

```text
Energy: 1.0
```

This example demonstrates why accurate summation matters when large positive and negative terms nearly cancel.

Non-finite energy results and arithmetic overflow are rejected.

### 13.2 Scale-aware energy comparisons

Use `energies_close()` to compare two energy values:

```python
from qsml.stability import energies_close

result = energies_close(
    actual=1000000000000.01,
    expected=1000000000000.0,
    relative_tolerance=1e-12,
    absolute_tolerance=0.0,
)

print("Numerically close:", result)
```

Both relative and absolute tolerances are considered.

### 13.3 Energy-vector error analysis

Use `compare_energy_vectors()` to compare reference and measured energies:

```python
from qsml.stability import compare_energy_vectors

report = compare_energy_vectors(
    reference=[1.0, 2.0, 3.0],
    measured=[1.0, 2.000000001, 3.0],
    relative_tolerance=1e-8,
    absolute_tolerance=1e-10,
)

print("Valid:", report["valid"])
print("Samples:", report["sample_count"])
print("Maximum absolute error:", report["max_absolute_error"])
print("Maximum relative error:", report["max_relative_error"])
print("RMS absolute error:", report["rms_absolute_error"])
```

The report includes:

- Number of compared energies.
- Maximum absolute error.
- Maximum relative error.
- Root mean square absolute error.
- Individual comparison failures.

### 13.4 Deterministic stress QUBO generation

QSML can generate repeatable random QUBO test problems:

```python
from qsml.stability import generate_stress_qubo

terms = generate_stress_qubo(
    num_variables=128,
    seed=2026,
    density=0.05,
    coefficient_scale=1000000.0,
)

print("Generated QUBO terms:", len(terms))
```

Parameters:

- `num_variables`: Number of binary variables.
- `seed`: Random seed for reproducibility.
- `density`: Probability of including an off-diagonal term.
- `coefficient_scale`: Maximum coefficient magnitude.

Every variable receives a diagonal term. Off-diagonal terms are generated according to the requested density.

This function generates test data; it does not solve the optimization problem.

### 13.5 GPU candidate memory estimation

The GPU QUBO candidate engine uses a binary candidate matrix and an energy array.

The default memory estimate models:

- Candidate matrix: 4 bytes per variable (int32).
- Energy array: 8 bytes per candidate (float64).

```python
from qsml.stability import (
    estimate_gpu_candidate_memory_bytes,
    check_gpu_memory_budget,
)

estimated = estimate_gpu_candidate_memory_bytes(
    num_candidates=4096,
    num_variables=64,
)

print("Estimated array payload:", estimated, "bytes")

budget = check_gpu_memory_budget(
    num_candidates=4096,
    num_variables=64,
    max_bytes=2_000_000,
)

print("Fits estimated budget:", budget["fits"])
print("Headroom:", budget["headroom_bytes"], "bytes")
```

The estimate covers the core candidate and energy array payload only. It is not a guarantee that a GPU allocation will succeed. CUDA context memory, CuPy allocator overhead, temporary arrays, other workloads, and memory fragmentation are not included.

### 13.6 QSML 3.1.4 tests

The new tests cover:

- Accurate summation under cancellation.
- Near-zero energy preservation.
- Subnormal coefficient handling.
- Energy overflow rejection.
- Scale-aware comparisons.
- Energy-vector error statistics.
- Invalid and non-finite inputs.
- GPU candidate memory estimation.
- Memory-budget boundaries.
- Deterministic random QUBO generation.
- Larger 128-variable stress evaluation.

The stress tests use controlled problem sizes and do not intentionally exhaust system RAM or GPU VRAM.

### 13.7 Limitations

Numerical stability checks do not prove that an optimization algorithm finds a global minimum.

Memory estimates are approximate payload estimates, not complete GPU allocation forecasts.

The CPU and GPU search algorithms remain different. Their optimization quality must be evaluated separately from numerical agreement on identical candidate assignments.



## QSML 3.1.5–3.5 — Resource Safety, Benchmarking and Unified Core

### QSML 3.1.5 — Resource preflight

The resource guard estimates the candidate matrix and energy-vector payload before GPU candidate generation.

```python
from qsml.resource_guard import (
    gpu_memory_snapshot,
    preflight_gpu_candidates,
)

snapshot = gpu_memory_snapshot()
print(snapshot)

if snapshot["available"]:
    report = preflight_gpu_candidates(
        num_candidates=8192,
        num_variables=64,
        safety_fraction=0.20,
    )
    print(report)
```

The estimate covers core array payload only. It does not guarantee that every allocation will succeed.

### QSML 3.2 — Fair CPU/GPU evaluation-budget comparison

The benchmark protocol gives both solvers the same number of QUBO objective evaluations.

```python
from qsml.fair_benchmark import run_equal_budget_comparison

report = run_equal_budget_comparison(
    num_variables=3,
    terms={
        (0, 0): -2.0,
        (1, 1): -2.0,
        (0, 1): 1.0,
    },
    evaluation_budget=5000,
    repeats=3,
    seed=42,
)

print(report)
```

Equal objective-evaluation counts do not mean identical algorithms or identical hardware work. CPU simulated annealing and GPU random candidate search remain different methods.

### QSML 3.3 — Randomized QUBO regression

Generate seeded small QUBO problems, compare solver results with exact references, and check reported energy independently.

```python
from qsml.regression import run_qubo_regression

report = run_qubo_regression(
    cases=8,
    min_variables=2,
    max_variables=10,
    seed=2026,
    iterations=2000,
    restarts=3,
)

print("Valid:", report["valid"])
print("Completed:", report["completed_cases"])
print("Optimal cases:", report["optimal_cases"])
print("Mean optimality gap:", report["mean_optimality_gap"])
print("Maximum gap:", report["maximum_optimality_gap"])
```

A valid result is not necessarily optimal. The optimality gap is reported rather than hidden.

### QSML 3.4 — Solver diagnostics

Independently validate solver output and compare it against an exact solution when the problem is small enough.

```python
from qsml.diagnostics import (
    diagnose_solver_result,
    print_diagnostics,
)

result = {
    "solution": [1, 1],
    "energy": -3.0,
    "backend": "CPU",
    "method": "TEST",
    "evaluations": 100,
}

report = diagnose_solver_result(
    num_variables=2,
    terms={
        (0, 0): -2.0,
        (1, 1): -2.0,
        (0, 1): 1.0,
    },
    solver_result=result,
)

print_diagnostics(report)
```

The diagnostic statuses distinguish invalid results, exact optimal solutions, valid feasible solutions, and results whose exact comparison was skipped.

### QSML 3.5 — Unified Core

The unified Core combines backend selection, evaluation budgets, GPU memory preflight and diagnostics.

```python
from qsml.core import QSMLCore

core = QSMLCore(
    gpu_safety_fraction=0.20,
    exact_diagnostics_limit=16,
)

result = core.solve(
    num_variables=3,
    terms={
        (0, 0): -2.0,
        (1, 1): -2.0,
        (0, 1): 1.0,
    },
    backend="auto",
    evaluation_budget=5000,
    seed=42,
)

print("Backend:", result["backend"])
print("Method:", result["method"])
print("Solution:", result["solution"])
print("Energy:", result["energy"])
print("Evaluations:", result["evaluations"])
print("Diagnostics:", result["diagnostics"])
```

### Current limitations

- QSML remains a classical quantum-inspired optimization and simulation project.
- CPU and GPU use different search algorithms.
- Equal evaluation budgets do not guarantee equal computational cost.
- Exact reference comparisons are limited to small problems.
- Memory preflight is an estimate, not a guarantee against all allocation failures.
- A valid feasible solution may still be suboptimal.

The application interface remains a later milestone. The immediate priority is to validate the engine and make its performance and limitations measurable.



# QSML 4.0 — Decision Space Intelligence

QSML 4.0 introduces multi-objective decision analysis.

The objective is to explore trade-offs between several mathematical objectives while checking whether each candidate satisfies the supplied hard constraints.

## 4.0.1 Multi-objective decision analysis

QSML supports multiple QUBO objectives, all treated as minimization objectives.

For candidate \(x\), the objective vector is:

\[
F(x)=
\begin{bmatrix}
E_1(x)\\
E_2(x)\\
\vdots\\
E_m(x)
\end{bmatrix}
\]

A candidate is Pareto-dominated if another feasible candidate is no worse in every objective and strictly better in at least one objective.

The Pareto frontier contains the non-dominated feasible candidates from the supplied candidate pool.

## 4.0.2 Example

```python
from qsml.decision_space import DecisionSpaceExplorer

objectives = {
    "cost": {
        (0, 0): 1.0,
        (1, 1): 2.0,
    },
    "risk": {
        (0, 0): 2.0,
        (1, 1): 1.0,
    },
}

candidates = [
    [1, 0],
    [0, 1],
    [1, 1],
    [0, 0],
]

constraints = [
    {
        "name": "select_at_least_one",
        "coefficients": {
            0: 1,
            1: 1,
        },
        "sense": ">=",
        "rhs": 1,
    }
]

explorer = DecisionSpaceExplorer()

report = explorer.analyze(
    num_variables=2,
    objectives=objectives,
    candidates=candidates,
    constraints=constraints,
)

print("Feasible candidates:", report["feasible_count"])
print("Infeasible candidates:", report["infeasible_count"])
print("Pareto frontier:", report["frontier"])
```

## 4.0.3 Hard constraints

QSML supports three linear constraint operators:

- `<=`: activity must not exceed the right-hand side.
- `>=`: activity must not be below the right-hand side.
- `==`: activity must equal the right-hand side within the configured numerical tolerance.

Example:

```python
constraints = [
    {
        "name": "budget",
        "coefficients": {
            0: 100,
            1: 200,
        },
        "sense": "<=",
        "rhs": 300,
    },
    {
        "name": "minimum_selection",
        "coefficients": {
            0: 1,
            1: 1,
        },
        "sense": ">=",
        "rhs": 1,
    },
]
```

In this decision-analysis layer, constraint violations are evaluated directly. Infeasible candidates are excluded from the Pareto frontier.

## 4.0.4 Scenario comparisons

The same candidate pool can be evaluated under different objective scenarios.

```python
scenarios = {
    "normal": {
        "cost": objectives["cost"],
        "risk": objectives["risk"],
    },
    "risk_priority": {
        "cost": objectives["risk"],
        "risk": objectives["cost"],
    },
}

scenario_report = explorer.analyze_scenarios(
    num_variables=2,
    scenarios=scenarios,
    candidates=candidates,
    constraints=constraints,
)

print(scenario_report["frontier_in_every_scenario"])
```

Candidates appearing in every supplied scenario's Pareto frontier may be interesting alternatives for further investigation.

This is not a formal robustness guarantee for conditions outside the scenarios that were tested.

## 4.0.5 Important limitations

- The frontier is calculated over supplied candidates, not the entire binary decision space.
- All objectives are minimized in this release.
- The current implementation evaluates linear hard constraints.
- The system does not prove global optimality.
- Scenario analysis only compares the scenarios explicitly supplied.
- Candidate generation and more advanced robust multi-objective optimization are future development areas.

QSML 4.0 moves the project from single-result optimization toward explicit exploration of trade-offs between feasible decisions.



## QSML 4.1 — Active Decision Space Exploration

QSML 4.1 introduces bounded generation of binary candidate solutions.

Instead of accepting only a supplied candidate pool, QSML can now generate candidate assignments through:

1. Boundary candidates.
2. User-supplied initial candidates.
3. Single-bit and random mutations.
4. Seeded random sampling.

### Generate candidate solutions

```python
from qsml.decision_search import generate_candidate_pool

candidates = generate_candidate_pool(
    num_variables=12,
    candidate_budget=1000,
    seed=42,
    mutation_rounds=3,
    mutation_probability=0.15,
)

print("Generated candidates:", len(candidates))
```

### Explore a decision space

```python
from qsml.decision_search import ActiveDecisionExplorer

objectives = {
    "cost": {
        (0, 0): 1.0,
        (1, 1): 2.0,
    },
    "risk": {
        (0, 0): 2.0,
        (1, 1): 1.0,
    },
}

explorer = ActiveDecisionExplorer()

report = explorer.explore(
    num_variables=2,
    objectives=objectives,
    candidate_budget=4,
    seed=42,
)

print("Candidates:", report["unique_candidate_count"])
print("Feasible:", report["feasible_count"])
print("Pareto frontier:", report["frontier"])
```

### Explore multiple scenarios

```python
scenarios = {
    "normal": objectives,
    "risk_priority": {
        "cost": objectives["risk"],
        "risk": objectives["cost"],
    },
}

report = explorer.explore_scenarios(
    num_variables=2,
    scenarios=scenarios,
    candidate_budget=4,
    seed=42,
)

print(report["frontier_in_every_scenario"])
```

### Limitations

The generated candidate pool is bounded by the candidate budget. For large binary spaces, it does not cover every possible assignment.

The Pareto frontier is therefore the frontier of the generated candidate pool, not a proven global frontier.

Randomized candidate generation is reproducible when the seed and other parameters are unchanged.

The configured candidate budget and pool-size safety limits prevent accidentally requesting excessively large candidate collections.


| 4.1 | Active Decision Space Exploration |
| 4.2 | Guided Decision Search |
| 4.3 | Pareto Diversity Archive |
| 4.4 | Scenario Robustness Analysis |
| 4.5 | Coefficient Sensitivity Analysis |
| 4.6 | Decision Explainability |
| 4.7 | Multi-Solver Collaboration |
| 4.8 | Reproducible Decision Audit Records |
| 4.9 | Unified Decision Intelligence Engine |


# QSML 4.1–4.9 — Decision Intelligence

QSML 4.x expands the optimizer into a bounded decision exploration and analysis framework.

## QSML 4.1 — Active Exploration

Generates binary candidates using boundary solutions, supplied starting points, mutations and random sampling.

## QSML 4.2 — Guided Search

Improves candidates by exploring one-bit neighboring solutions. Feasibility takes priority over objective scores.

The weighted objective score is a search heuristic. It does not prove Pareto optimality.

## QSML 4.3 — Diversity Archive

Selects a diverse subset of Pareto candidates using Hamming distance.

This helps retain alternatives that differ in their binary decisions.

## QSML 4.4 — Scenario Robustness

Tests the same candidates across different named objective scenarios.

QSML reports:
- Feasibility in each scenario.
- Feasibility in all supplied scenarios.
- Best, mean and worst objective values.
- Objective variation between scenarios.

Robustness is limited to the supplied scenarios.

## QSML 4.5 — Sensitivity Analysis

Perturbs objective coefficients using reproducible random factors and measures how often candidates appear on the Pareto frontier.

Frontier frequency is an experimental sensitivity metric, not a probability of real-world success.

## QSML 4.6 — Decision Explainability

Breaks each QUBO objective into its individual term contributions and reports constraint activities and violations.

## QSML 4.7 — Multi-Solver Collaboration

Combines candidate solutions from different solvers and independently checks reported energies against the selected QUBO objective.

Candidate provenance is retained.

Only results calculated against the same objective are compared.

## QSML 4.8 — Decision Audit Records

Stores structured decision records as JSON.

Records contain:
- Configuration
- Seed and search parameters
- Decision results
- Project version
- Creation timestamp
- SHA-256 integrity hash

A matching hash detects changes to the stored content. It does not authenticate the source or prove mathematical correctness.

## QSML 4.9 — Unified Decision Intelligence

Connects candidate generation, guided search, Pareto diversity, scenario analysis, sensitivity analysis, explanations, optional solver integration and audit records.

Example:

```python
from qsml.decision_intelligence import QSMLDecisionIntelligence

objectives = {
    "cost": {
        (0, 0): 1.0,
        (1, 1): 2.0,
    },
    "risk": {
        (0, 0): 2.0,
        (1, 1): 1.0,
    },
}

engine = QSMLDecisionIntelligence()

result = engine.run(
    num_variables=2,
    objectives=objectives,
    candidate_budget=4,
    seed=42,
    guided_steps=2,
    neighbor_budget=20,
    diversity_limit=4,
    sensitivity_trials=5,
    record_path="experiments/decision_4_9.json",
)

print("Pareto frontier:")
print(result["decision_report"]["frontier"])

print("Diverse alternatives:")
print(result["diversity"]["selected"])

print("Sensitivity:")
print(result["sensitivity"])

print("Audit record:")
print(result["saved_record_path"])
```

## Limitations

- Search is bounded by the candidate and neighbor budgets.
- Pareto results refer to explored candidates, not the complete binary search space unless exhaustive enumeration actually covers it.
- Guided search uses a weighted heuristic and can miss good alternatives.
- Scenario robustness is conditional on the specified scenarios.
- Sensitivity experiments depend on the chosen perturbation distribution.
- Solver results must be based on the same QUBO objective to be meaningfully compared.
- Audit hashes are integrity checks, not digital signatures.
- QSML is classical software. It does not create or execute physical qubits by itself.

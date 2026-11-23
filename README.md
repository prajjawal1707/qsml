
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

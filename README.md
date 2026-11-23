# QSML

**Quantum-inspired Search and Machine Learning** is an experimental
classical computing project for quantum-style state-vector simulation
and optimization on CPUs and NVIDIA GPUs.

> QSML does not create physical qubits, is not a quantum computer, and
> does not claim a general quantum speedup.

## Status

-   Development baseline: QSML 2.1
-   Reported integration/reliability suite: 15 passed, 0 failed
-   Tested environment: Windows 11, Python 3.14 GPU environment, NVIDIA
    GeForce RTX 2050 with 4 GB VRAM
-   CPU fallback available

These results describe the reported tests and hardware only. They do not
establish production readiness or performance on arbitrary workloads.

## Current capabilities

### State-vector simulation

-   Classical complex-amplitude state vectors
-   X, Y, Z, H, S, T, RX, RY, RZ and CNOT gates
-   Probability calculation, measurement and state validation
-   CPU and CuPy-based GPU paths

### Search

-   Grover-style amplitude amplification
-   Unified CPU/GPU runtime
-   Measurement sampling and candidate ranking

### Optimization

-   Binary Quadratic Optimization (QUBO)
-   CPU simulated annealing
-   GPU random-candidate search
-   Adaptive backend selection
-   Candidate decision summaries

### Runtime

-   Small QSML search instruction language
-   RAM and VRAM inspection
-   State-vector memory estimates
-   Integration, correctness and timing tests

## Run

In PowerShell, activate the GPU environment:

``` powershell
.\.venv-gpu\Scripts\Activate.ps1
```

Run the integrated test launcher:

``` powershell
python run_qsml.py
```

The CPU path can be used when a compatible CUDA GPU is unavailable.

## Example program

`programs/first_search.qsml`:

``` text
ENGINE AUTO
REGISTER 3
SUPERPOSITION ALL
MARK 101
AMPLIFY 2
MEASURE 1000
DECIDE BEST
SHOW RESULTS
```

The language currently supports a limited search workflow; it is not yet
a general-purpose programming language.

## Limitations

-   State-vector simulation is classical and scales exponentially with
    qubit count.
-   GPU execution may be slower for small workloads due to launch and
    transfer overhead.
-   CPU QUBO uses simulated annealing; GPU QUBO uses random candidate
    search. Their timings are not directly comparable without controlled
    evaluation budgets.
-   Current tests cover selected small problems, not arbitrary large
    real-world workloads.
-   No claim is made that QSML outperforms established optimization
    libraries or commercial AI systems.

## Roadmap

-   **2.2:** VRAM-aware chunked GPU candidate batches and hybrid CPU/GPU
    search orchestration.
-   **2.3:** Larger optimization workloads, stronger algorithms and
    scaling tests.
-   **2.4:** Advanced resource scheduling and memory management.
-   **2.5:** Richer QSML language and reusable problem models.
-   **3.0:** Stable interfaces, broader validation, benchmarks and
    practical workflows.

Roadmap items are plans, not completed features.

## Validation principles

For each new algorithm: 1. Preserve existing passing tests. 2. Add
known-answer correctness tests. 3. Test invalid inputs and resource
limits. 4. Record hardware, workload sizes and random seeds. 5. Compare
CPU and GPU using equivalent objectives and clearly reported evaluation
budgets.

## License

No license has been declared. Until a license file is added, all rights
are reserved by default. Add an appropriate license before inviting
reuse or redistribution.

## Vision

Build a dependable quantum-inspired classical computing and optimization
platform that models problems, explores candidate solutions, uses
available hardware intelligently and reports verifiable results.
Prioritize correctness, reproducibility and usefulness over unsupported
claims of quantum advantage.

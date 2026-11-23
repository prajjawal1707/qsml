import json
import traceback
from pathlib import Path

import numpy as np

from qsml.v2_engine import (
    QSML2Engine,
    OptimizationService,
    BenchmarkSuite,
    print_report,
)
from qsml.batching import (
    BatchedGPUQUBOEngine,
    HybridQUBOEngine,
)

ROOT = Path(__file__).resolve().parent
PROGRAM_FILE = ROOT / "programs" / "first_search.qsml"


def passed(name):
    print(f"[PASS] {name}")


def failed(name, error):
    print(f"[FAIL] {name}: {error}")


def main():
    print("\n" + "=" * 60)
    print("             QSML 2.0 — FULL SYSTEM TEST")
    print("=" * 60)

    engine = QSML2Engine()
    optimization = OptimizationService()

    tests_passed = 0
    tests_failed = 0

    # --------------------------------------------------------
    # TEST 1 — RESOURCE MANAGER
    # --------------------------------------------------------
    print("\n[1] RESOURCE MANAGER")

    try:
        resources = engine.resources.snapshot()
        print(json.dumps(resources.__dict__, indent=2))

        assert resources.cpu_count >= 1
        assert resources.ram_total_gb > 0

        passed("Hardware and memory inspection")
        tests_passed += 1
    except Exception as exc:
        failed("Hardware and memory inspection", exc)
        tests_failed += 1

    # --------------------------------------------------------
    # TEST 2 — QSML PARSER
    # --------------------------------------------------------
    print("\n[2] QSML LANGUAGE PARSER")

    try:
        source = PROGRAM_FILE.read_text(encoding="utf-8")
        program = engine.parser.parse(source)

        assert program.qubits == 3
        assert program.marked_state == "101"
        assert program.shots > 0

        print("Parsed program:", program)
        passed("QSML parser")
        tests_passed += 1
    except Exception as exc:
        failed("QSML parser", exc)
        tests_failed += 1
        print("\nCannot continue without a valid QSML program.")
        print(f"Expected program file: {PROGRAM_FILE}")
        return

    # --------------------------------------------------------
    # TEST 3 — CPU GROVER EXECUTION
    # --------------------------------------------------------
    print("\n[3] CPU EXECUTION")

    try:
        cpu_source = source.replace(
            f"ENGINE {program.engine}",
            "ENGINE CPU",
        )
        if "ENGINE" not in source.upper():
            cpu_source = "ENGINE CPU\n" + source

        cpu_output = engine.execute(cpu_source)

        assert cpu_output["backend"] == "CPU"
        assert cpu_output["execution_seconds"] >= 0

        print_report(cpu_output)
        passed("CPU unified runtime")
        tests_passed += 1
    except Exception as exc:
        failed("CPU unified runtime", exc)
        traceback.print_exc()
        cpu_output = None
        tests_failed += 1

    # --------------------------------------------------------
    # TEST 4 — GPU GROVER EXECUTION
    # --------------------------------------------------------
    print("\n[4] GPU EXECUTION")

    gpu_output = None

    try:
        gpu_source = source.replace(
            f"ENGINE {program.engine}",
            "ENGINE GPU",
        )
        if "ENGINE" not in source.upper():
            gpu_source = "ENGINE GPU\n" + source

        gpu_output = engine.execute(gpu_source)

        assert gpu_output["backend"] == "GPU"

        print_report(gpu_output)
        passed("GPU unified runtime")
        tests_passed += 1
    except Exception as exc:
        failed("GPU unified runtime", exc)
        print("GPU test may fail if CUDA or VRAM is unavailable.")
        tests_failed += 1

    # --------------------------------------------------------
    # TEST 5 — DECISION ENGINE
    # --------------------------------------------------------
    print("\n[5] DECISION ENGINE")

    try:
        selected_output = cpu_output or gpu_output
        assert selected_output is not None

        decision = selected_output["decision"]
        assert decision is not None
        assert decision.get("selected_state") is not None

        print("Decision:", decision)
        passed("Decision selection")
        tests_passed += 1
    except Exception as exc:
        failed("Decision selection", exc)
        tests_failed += 1

    # --------------------------------------------------------
    # TEST 6 — QUBO OPTIMIZATION
    # --------------------------------------------------------
    # --------------------------------------------------------
    # TEST 6 — QUBO OPTIMIZATION
    # --------------------------------------------------------
    print("\n[6] QUBO OPTIMIZATION")

    try:
        Q = np.array([
        [-2,  1,  1],
        [ 1, -2,  1],
        [ 1,  1, -2],
        ], dtype=float)

        result = optimization.solve_qubo(
            Q,
            backend="CPU",
            iterations=12000,
            restarts=8,
        )

        print("Optimizer output:")
        print(json.dumps(result, indent=2, default=str))

        solution = result["solution"]
        energy = result["energy"]

        assert len(solution) == 3
        assert all(bit in (0, 1) for bit in solution)
        assert np.isfinite(energy)
        assert abs(energy - (-2.0)) < 1e-9

        print("Solution:", solution)
        print("Energy:", energy)

        passed("QUBO optimizer and Max-Cut correctness")
        tests_passed += 1

    except Exception as exc:
        failed("QUBO optimizer and Max-Cut correctness", exc)
        traceback.print_exc()
        tests_failed += 1
    # --------------------------------------------------------
    # TEST 7 — BENCHMARK SUITE
    # --------------------------------------------------------
    print("\n[7] BENCHMARK SUITE")

    try:
        benchmark = BenchmarkSuite.run(
            lambda: sum(i * i for i in range(10000)),
            repeats=3,
        )

        print(json.dumps(benchmark, indent=2, default=str))
        assert benchmark["mean_seconds"] >= 0

        passed("Benchmark timing")
        tests_passed += 1
    except Exception as exc:
        failed("Benchmark timing", exc)
        tests_failed += 1

    # --------------------------------------------------------
    # TEST 8 — CPU/GPU COMPARISON
    # --------------------------------------------------------
    print("\n[8] CPU/GPU COMPARISON")

    try:
        comparison = BenchmarkSuite.compare_cpu_gpu(
            engine,
            source,
            repeats=2,
        )

        for backend, details in comparison.items():
            print(f"\n{backend}:")
            print(json.dumps(details, indent=2, default=str))

        passed("CPU/GPU comparison procedure")
        tests_passed += 1
    except Exception as exc:
        failed("CPU/GPU comparison procedure", exc)
        tests_failed += 1


    # ========================================================
    # QSML 2.1 — RELIABILITY TESTS
    # ========================================================

    print("\n" + "=" * 60)
    print("             QSML 2.1 — RELIABILITY TESTS")
    print("=" * 60)

    # --------------------------------------------------------
    # Helper: construct a cycle Max-Cut QUBO
    # --------------------------------------------------------
    def make_cycle_qubo(n):
        """
        Even cycle Max-Cut.
        Known optimal energy = -n.
        """
        Q = np.zeros((n, n), dtype=float)

        for i in range(n):
            j = (i + 1) % n

            Q[i, i] -= 1.0
            Q[j, j] -= 1.0

            Q[i, j] += 0.5
            Q[j, i] += 0.5

        return Q

    def exact_qubo_energy(matrix, solution):
        terms = OptimizationService.matrix_to_terms(matrix)

        return sum(
            coefficient * solution[i] * solution[j]
            for (i, j), coefficient in terms.items()
        )

    def brute_force_optimum(matrix):
        n = matrix.shape[0]
        best_energy = float("inf")

        for number in range(2 ** n):
            solution = [
                (number >> bit) & 1
                for bit in range(n)
            ]

            energy = exact_qubo_energy(matrix, solution)
            best_energy = min(best_energy, energy)

        return best_energy

    # --------------------------------------------------------
    # TEST 9 — GPU QUBO CORRECTNESS
    # --------------------------------------------------------
    print("\n[9] GPU QUBO CORRECTNESS")

    try:
        available = optimization.optimizer.available_backends()

        if "GPU" not in available:
            print("[SKIP] CUDA GPU optimizer unavailable")
        else:
            Q = make_cycle_qubo(3)
            expected = brute_force_optimum(Q)

            gpu_result = optimization.solve_qubo(
                Q,
                backend="GPU",
                candidates=32768,
                seed=42,
            )

            actual = gpu_result["energy"]
            solution = gpu_result["solution"]

            print("GPU solution:", solution)
            print("GPU energy:", actual)
            print("Expected energy:", expected)

            assert len(solution) == 3
            assert all(bit in (0, 1) for bit in solution)
            assert abs(actual - expected) < 1e-8

            passed("GPU QUBO correctness")
            tests_passed += 1

    except Exception as exc:
        failed("GPU QUBO correctness", exc)
        traceback.print_exc()
        tests_failed += 1

    # --------------------------------------------------------
    # TEST 10 — MULTIPLE QUBO PROBLEM SIZES
    # --------------------------------------------------------
    print("\n[10] MULTIPLE QUBO PROBLEM SIZES")

    try:
        size_results = []

        for n in (3, 6, 8):
            Q = make_cycle_qubo(n)
            expected = brute_force_optimum(Q)

            cpu_result = optimization.solve_qubo(
                Q,
                backend="CPU",
                iterations=12000,
                restarts=8,
                seed=42,
            )

            assert abs(cpu_result["energy"] - expected) < 1e-8

            print(
                f"{n} variables | "
                f"CPU energy: {cpu_result['energy']} | "
                f"Expected: {expected}"
            )

            size_results.append(n)

        passed("QUBO sizes 3, 6 and 8")
        tests_passed += 1

    except Exception as exc:
        failed("Multiple QUBO problem sizes", exc)
        traceback.print_exc()
        tests_failed += 1

    # --------------------------------------------------------
    # TEST 11 — CPU/GPU SOLUTION QUALITY
    # --------------------------------------------------------
    print("\n[11] CPU/GPU SOLUTION QUALITY")

    try:
        Q = make_cycle_qubo(8)
        expected = brute_force_optimum(Q)

        cpu_result = optimization.solve_qubo(
            Q,
            backend="CPU",
            iterations=12000,
            restarts=8,
            seed=42,
        )

        print("CPU energy:", cpu_result["energy"])
        print("Exact optimum:", expected)

        assert abs(cpu_result["energy"] - expected) < 1e-8

        if "GPU" in optimization.optimizer.available_backends():
            gpu_result = optimization.solve_qubo(
                Q,
                backend="GPU",
                candidates=32768,
                seed=42,
            )

            print("GPU energy:", gpu_result["energy"])

            assert abs(gpu_result["energy"] - expected) < 1e-8

            print("Both backends reached the known optimum.")
        else:
            print("[SKIP] GPU comparison unavailable.")

        passed("CPU/GPU solution quality")
        tests_passed += 1

    except Exception as exc:
        failed("CPU/GPU solution quality", exc)
        traceback.print_exc()
        tests_failed += 1

    # --------------------------------------------------------
    # TEST 12 — RESOURCE AND MEMORY SAFETY
    # --------------------------------------------------------
    print("\n[12] RESOURCE AND MEMORY SAFETY")

    try:
        estimate = engine.resources.estimate_statevector_gb(30)

        print("30-qubit state-vector estimate:", estimate, "GB")

        assert estimate > 0

        # Invalid qubit counts must be rejected.
        invalid_qubits_rejected = False

        try:
            engine.resources.estimate_statevector_gb(0)
        except ValueError:
            invalid_qubits_rejected = True

        assert invalid_qubits_rejected

        # A 30-qubit state vector requires about 16 GB.
        # It must be rejected before allocation on this machine.
        for backend in ("CPU", "GPU", "AUTO"):
            rejected = False

            try:
                engine.resources.choose_backend(
                    backend,
                    qubits=30,
                )
            except (MemoryError, RuntimeError):
                rejected = True

            assert rejected, (
                f"{backend} did not reject the oversized workload"
            )

        print("Oversized state-vector request safely rejected.")
        passed("Memory estimation and safety checks")
        tests_passed += 1

    except Exception as exc:
        failed("Memory estimation and safety checks", exc)
        traceback.print_exc()
        tests_failed += 1

    # --------------------------------------------------------
    # TEST 13 — INVALID INPUTS
    # --------------------------------------------------------
    print("\n[13] INVALID INPUT VALIDATION")

    try:
        invalid_cases = [
            np.array([[1, 2, 3], [4, 5, 6]], dtype=float),
            np.array([[1, np.nan], [0, 1]], dtype=float),
            np.zeros((3, 3), dtype=float),
        ]

        for invalid_matrix in invalid_cases:
            rejected = False

            try:
                optimization.solve_qubo(
                    invalid_matrix,
                    backend="CPU",
                )
            except ValueError:
                rejected = True

            assert rejected, "Invalid QUBO matrix was not rejected."

        print("Non-square, NaN and empty QUBO inputs rejected.")
        passed("Invalid input validation")
        tests_passed += 1

    except Exception as exc:
        failed("Invalid input validation", exc)
        traceback.print_exc()
        tests_failed += 1

    # --------------------------------------------------------
    # TEST 14 — CPU REPEATABILITY
    # --------------------------------------------------------
    print("\n[14] CPU REPEATABILITY")

    try:
        Q = make_cycle_qubo(6)

        first = optimization.solve_qubo(
            Q,
            backend="CPU",
            iterations=5000,
            restarts=4,
            seed=123,
        )

        second = optimization.solve_qubo(
            Q,
            backend="CPU",
            iterations=5000,
            restarts=4,
            seed=123,
        )

        assert first["solution"] == second["solution"]
        assert abs(first["energy"] - second["energy"]) < 1e-10

        print("First solution:", first["solution"])
        print("Second solution:", second["solution"])
        print("Both energies:", first["energy"])

        passed("CPU fixed-seed repeatability")
        tests_passed += 1

    except Exception as exc:
        failed("CPU repeatability", exc)
        traceback.print_exc()
        tests_failed += 1

    # --------------------------------------------------------
    # TEST 15 — GPU REPEATABILITY
    # --------------------------------------------------------
    print("\n[15] GPU REPEATABILITY")

    try:
        if "GPU" not in optimization.optimizer.available_backends():
            print("[SKIP] GPU unavailable")
        else:
            Q = make_cycle_qubo(6)

            first = optimization.solve_qubo(
                Q,
                backend="GPU",
                candidates=16384,
                seed=123,
            )

            second = optimization.solve_qubo(
                Q,
                backend="GPU",
                candidates=16384,
                seed=123,
            )

            assert first["solution"] == second["solution"]
            assert abs(first["energy"] - second["energy"]) < 1e-10

            print("First solution:", first["solution"])
            print("Second solution:", second["solution"])
            print("Both energies:", first["energy"])

            passed("GPU fixed-seed repeatability")
            tests_passed += 1

    except Exception as exc:
        failed("GPU repeatability", exc)
        traceback.print_exc()
        tests_failed += 1

    # ========================================================
    # QSML 2.2 — BATCHING AND HYBRID TESTS
    # ========================================================

    print("\n" + "=" * 60)
    print("             QSML 2.2 — BATCHING TESTS")
    print("=" * 60)

    # --------------------------------------------------------
    # TEST 16 — BATCH PLANNER
    # --------------------------------------------------------
    print("\n[16] GPU BATCH PLANNER")

    try:
        planner = BatchedGPUQUBOEngine(
            max_batch_size=4096
        )

        batches = planner.plan_batches(
            total_candidates=10000,
            batch_size=4096,
        )

        print("Planned batches:", batches)

        assert batches == [4096, 4096, 1808]
        assert sum(batches) == 10000

        passed("GPU batch planning")
        tests_passed += 1

    except Exception as exc:
        failed("GPU batch planning", exc)
        traceback.print_exc()
        tests_failed += 1

    # --------------------------------------------------------
    # TEST 17 — VRAM-AWARE BATCH SIZE
    # --------------------------------------------------------
    print("\n[17] VRAM-AWARE BATCH SIZING")

    try:
        planner = BatchedGPUQUBOEngine(
            memory_fraction=0.20,
            max_batch_size=8192,
        )

        batch_size = planner.choose_batch_size(
            num_variables=8,
            free_vram_bytes=1024 * 1024 * 1024,
        )

        print("Selected batch size:", batch_size)

        assert 1 <= batch_size <= 8192

        small_batch = planner.choose_batch_size(
            num_variables=8,
            free_vram_bytes=64 * 1024 * 1024,
        )

        print("Batch with lower free VRAM:", small_batch)

        assert small_batch <= batch_size

        passed("VRAM-aware batch sizing")
        tests_passed += 1

    except Exception as exc:
        failed("VRAM-aware batch sizing", exc)
        traceback.print_exc()
        tests_failed += 1

    # --------------------------------------------------------
    # TEST 18 — BATCHED GPU QUBO
    # --------------------------------------------------------
    print("\n[18] BATCHED GPU QUBO")

    try:
        available = optimization.optimizer.available_backends()

        if "GPU" not in available:
            print("[SKIP] GPU unavailable")
        else:
            Q = make_cycle_qubo(8)
            terms = OptimizationService.matrix_to_terms(Q)
            expected = brute_force_optimum(Q)

            batched_gpu = BatchedGPUQUBOEngine(
                max_batch_size=4096
            )

            result = batched_gpu.solve(
                num_variables=8,
                terms=terms,
                candidates=32768,
                seed=42,
                requested_batch_size=4096,
            )

            print("Solution:", result["solution"])
            print("Energy:", result["energy"])
            print("Batches:", result["batch_count"])
            print("Evaluations:", result["evaluations"])

            assert result["batch_count"] == 8
            assert result["evaluations"] > 0
            assert abs(result["energy"] - expected) < 1e-8

            passed("Batched GPU QUBO correctness")
            tests_passed += 1

    except Exception as exc:
        failed("Batched GPU QUBO correctness", exc)
        traceback.print_exc()
        tests_failed += 1

    # --------------------------------------------------------
    # TEST 19 — HYBRID CPU/GPU SEARCH
    # --------------------------------------------------------
    print("\n[19] HYBRID CPU/GPU SEARCH")

    try:
        available = optimization.optimizer.available_backends()

        if "GPU" not in available:
            print("[SKIP] GPU unavailable")
        else:
            Q = make_cycle_qubo(8)
            terms = OptimizationService.matrix_to_terms(Q)
            expected = brute_force_optimum(Q)

            hybrid = HybridQUBOEngine(
                gpu_memory_fraction=0.20,
                max_gpu_batch_size=4096,
            )

            result = hybrid.solve(
                num_variables=8,
                terms=terms,
                gpu_candidates=32768,
                cpu_iterations=12000,
                cpu_restarts=8,
                seed=42,
                gpu_batch_size=4096,
            )

            print("Selected backend:", result["selected_backend"])
            print("Selected solution:", result["solution"])
            print("Selected energy:", result["energy"])
            print("CPU energy:", result["cpu_result"]["energy"])
            print("GPU energy:", result["gpu_result"]["energy"])

            assert result["backend"] == "HYBRID"
            assert abs(result["energy"] - expected) < 1e-8

            passed("Hybrid CPU/GPU correctness")
            tests_passed += 1

    except Exception as exc:
        failed("Hybrid CPU/GPU correctness", exc)
        traceback.print_exc()
        tests_failed += 1

    # --------------------------------------------------------
    # FINAL SUMMARY
    # --------------------------------------------------------
    total = tests_passed + tests_failed

    print("\n" + "=" * 60)
    print("                 QSML 2.1 RELIABILITY SUMMARY")
    print("=" * 60)
    print(f"Tests passed: {tests_passed}")
    print(f"Tests failed: {tests_failed}")
    print(f"Total tests:  {total}")

    if tests_failed == 0:
        print("\nALL TESTS PASSED")
    else:
        print("\nSOME TESTS FAILED — inspect the errors above.")

    print("=" * 60)


if __name__ == "__main__":
    main()
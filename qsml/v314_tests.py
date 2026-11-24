"""QSML 3.1.4 — Numerical stability and stress tests."""

import math
import random

from qsml.stability import (
    stable_qubo_energy,
    energies_close,
    compare_energy_vectors,
    generate_stress_qubo,
    estimate_gpu_candidate_memory_bytes,
    check_gpu_memory_budget,
)


def run_v314_tests():
    passed = 0
    failed = 0

    def check(name, function):
        nonlocal passed, failed

        try:
            function()
            print(f"[PASS] {name}")
            passed += 1
        except Exception as exc:
            print(f"[FAIL] {name}: {exc}")
            failed += 1

    def must_raise(function):
        try:
            function()
        except ValueError:
            return

        raise AssertionError("Expected ValueError")

    check(
        "Accurate summation under cancellation",
        lambda: (
            lambda value: (
                value == 1.0
                or (_ for _ in ()).throw(AssertionError(value))
            )
        )(
            stable_qubo_energy(
                3,
                {
                    (0, 0): 1e16,
                    (1, 1): 1.0,
                    (2, 2): -1e16,
                },
                [1, 1, 1],
            )
        ),
    )

    check(
        "Near-zero energy preservation",
        lambda: (
            lambda value: (
                0.0 < value < 1e-14
                or (_ for _ in ()).throw(AssertionError(value))
            )
        )(
            stable_qubo_energy(
                2,
                {
                    (0, 0): 1.0,
                    (1, 1): -1.0 + 1e-15,
                },
                [1, 1],
            )
        ),
    )

    check(
        "Subnormal coefficient handling",
        lambda: (
            lambda value: (
                value == 5e-324
                or (_ for _ in ()).throw(AssertionError(value))
            )
        )(
            stable_qubo_energy(
                1,
                {(0, 0): 5e-324},
                [1],
            )
        ),
    )

    check(
        "Reject energy overflow",
        lambda: must_raise(
            lambda: stable_qubo_energy(
                2,
                {
                    (0, 0): 1e308,
                    (1, 1): 1e308,
                },
                [1, 1],
            )
        ),
    )

    check(
        "Scale-aware energy comparison",
        lambda: (
            energies_close(
                actual=1e12 + 0.01,
                expected=1e12,
                relative_tolerance=1e-12,
                absolute_tolerance=0.0,
            )
            or (_ for _ in ()).throw(
                AssertionError("Expected values to be close")
            )
        ),
    )

    check(
        "Energy vector error statistics",
        lambda: (
            lambda report: (
                report["valid"]
                and report["sample_count"] == 3
                and report["max_absolute_error"] > 0
                and report["rms_absolute_error"] > 0
                or (_ for _ in ()).throw(AssertionError(report))
            )
        )(
            compare_energy_vectors(
                [1.0, 2.0, 3.0],
                [1.0 + 1e-9, 2.0 - 1e-9, 3.0],
                relative_tolerance=1e-8,
                absolute_tolerance=1e-10,
            )
        ),
    )

    check(
        "Reject mismatched energy vector lengths",
        lambda: must_raise(
            lambda: compare_energy_vectors(
                [1.0, 2.0],
                [1.0],
            )
        ),
    )

    check(
        "Reject non-finite measured energies",
        lambda: must_raise(
            lambda: compare_energy_vectors(
                [1.0, 2.0],
                [1.0, float("nan")],
            )
        ),
    )

    check(
        "GPU candidate memory estimation",
        lambda: (
            lambda value: (
                value == 1081344
                or (_ for _ in ()).throw(AssertionError(value))
            )
        )(
            estimate_gpu_candidate_memory_bytes(
                num_candidates=4096,
                num_variables=64,
            )
        ),
    )

    def memory_budget_test():
        fits = check_gpu_memory_budget(
            num_candidates=4096,
            num_variables=64,
            max_bytes=2_000_000,
        )

        does_not_fit = check_gpu_memory_budget(
            num_candidates=4096,
            num_variables=64,
            max_bytes=1_000_000,
        )

        assert fits["fits"] is True
        assert does_not_fit["fits"] is False

    check(
        "GPU memory budget boundary",
        memory_budget_test,
    )

    check(
        "Deterministic stress QUBO generation",
        lambda: (
            lambda first, second: (
                first == second
                or (_ for _ in ()).throw(
                    AssertionError("Same seed produced different QUBOs")
                )
            )
        )(
            generate_stress_qubo(
                num_variables=32,
                seed=1234,
                density=0.15,
                coefficient_scale=10000.0,
            ),
            generate_stress_qubo(
                num_variables=32,
                seed=1234,
                density=0.15,
                coefficient_scale=10000.0,
            ),
        ),
    )

    def large_stress_test():
        num_variables = 128

        terms = generate_stress_qubo(
            num_variables=num_variables,
            seed=2026,
            density=0.05,
            coefficient_scale=1e6,
        )

        rng = random.Random(2026)

        solution = [
            rng.randint(0, 1)
            for _ in range(num_variables)
        ]

        first_energy = stable_qubo_energy(
            num_variables,
            terms,
            solution,
        )

        second_energy = stable_qubo_energy(
            num_variables,
            terms,
            solution,
        )

        assert math.isfinite(first_energy)
        assert first_energy == second_energy
        assert len(terms) >= num_variables

        print("    Variables:", num_variables)
        print("    QUBO terms:", len(terms))
        print("    Energy:", first_energy)

    check(
        "128-variable deterministic stress evaluation",
        large_stress_test,
    )

    print("\nQSML 3.1.4 TEST SUMMARY")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Total:  {passed + failed}")

    return passed, failed
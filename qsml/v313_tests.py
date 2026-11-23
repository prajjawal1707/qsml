"""QSML 3.1.3 — CPU/GPU numerical consistency tests."""

import math

from qsml.consistency import (
    evaluate_qubo_candidates_cpu,
    evaluate_qubo_candidates_gpu,
    compare_cpu_gpu_energies,
)


def run_v313_tests():
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
        except (ValueError, RuntimeError):
            return

        raise AssertionError("Expected ValueError or RuntimeError")

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

    check(
        "CPU QUBO candidate energy evaluation",
        lambda: (
            lambda result: (
                result == [0.0, -2.0, -2.0, -3.0]
                or (_ for _ in ()).throw(AssertionError(result))
            )
        )(
            evaluate_qubo_candidates_cpu(
                2,
                terms,
                candidates,
            )
        ),
    )

    check(
        "Reversed QUBO pair consistency",
        lambda: (
            lambda first, second: (
                first == second
                or (_ for _ in ()).throw(
                    AssertionError((first, second))
                )
            )
        )(
            evaluate_qubo_candidates_cpu(
                2,
                {(0, 1): 3.0},
                [[1, 1], [1, 0]],
            ),
            evaluate_qubo_candidates_cpu(
                2,
                {(1, 0): 3.0},
                [[1, 1], [1, 0]],
            ),
        ),
    )

    check(
        "Reject empty candidate collection",
        lambda: must_raise(
            lambda: evaluate_qubo_candidates_cpu(
                2,
                terms,
                [],
            )
        ),
    )

    check(
        "Reject wrong candidate length",
        lambda: must_raise(
            lambda: evaluate_qubo_candidates_cpu(
                2,
                terms,
                [[1]],
            )
        ),
    )

    check(
        "Reject non-binary candidate values",
        lambda: must_raise(
            lambda: evaluate_qubo_candidates_cpu(
                2,
                terms,
                [[1, 2]],
            )
        ),
    )

    check(
        "Reject malformed candidate row",
        lambda: must_raise(
            lambda: evaluate_qubo_candidates_cpu(
                2,
                terms,
                [1],
            )
        ),
    )

    check(
        "Reject NaN QUBO coefficients",
        lambda: must_raise(
            lambda: evaluate_qubo_candidates_cpu(
                1,
                {(0, 0): float("nan")},
                [[1]],
            )
        ),
    )

    check(
        "Reject CPU energy overflow",
        lambda: must_raise(
            lambda: evaluate_qubo_candidates_cpu(
                2,
                {
                    (0, 0): 1e308,
                    (1, 1): 1e308,
                },
                [[1, 1]],
            )
        ),
    )

    check(
        "Reject negative absolute tolerance",
        lambda: must_raise(
            lambda: compare_cpu_gpu_energies(
                2,
                terms,
                candidates,
                absolute_tolerance=-1.0,
            )
        ),
    )

    check(
        "Reject NaN relative tolerance",
        lambda: must_raise(
            lambda: compare_cpu_gpu_energies(
                2,
                terms,
                candidates,
                relative_tolerance=float("nan"),
            )
        ),
    )

    check(
        "CPU candidate evaluation repeatability",
        lambda: (
            lambda first, second: (
                first == second
                or (_ for _ in ()).throw(
                    AssertionError((first, second))
                )
            )
        )(
            evaluate_qubo_candidates_cpu(2, terms, candidates),
            evaluate_qubo_candidates_cpu(2, terms, candidates),
        ),
    )

    def gpu_consistency_test():
        try:
            import cupy as cp

            gpu_available = cp.cuda.runtime.getDeviceCount() > 0

        except Exception:
            gpu_available = False

        if not gpu_available:
            must_raise(
                lambda: evaluate_qubo_candidates_gpu(
                    2,
                    terms,
                    candidates,
                )
            )
            return

        gpu_energies = evaluate_qubo_candidates_gpu(
            2,
            terms,
            candidates,
        )

        cpu_energies = evaluate_qubo_candidates_cpu(
            2,
            terms,
            candidates,
        )

        assert len(gpu_energies) == len(cpu_energies)

        for cpu_value, gpu_value in zip(
            cpu_energies,
            gpu_energies,
        ):
            assert math.isclose(
                cpu_value,
                gpu_value,
                rel_tol=1e-10,
                abs_tol=1e-10,
            )

        report = compare_cpu_gpu_energies(
            2,
            terms,
            candidates,
        )

        assert report.valid is True
        assert report.sample_count == len(candidates)
        assert report.max_absolute_error <= 1e-10
        assert len(report.errors) == 0

        print(
            "    Maximum CPU/GPU absolute error:",
            report.max_absolute_error,
        )

    check(
        "GPU numerical consistency and availability handling",
        gpu_consistency_test,
    )

    print("\nQSML 3.1.3 TEST SUMMARY")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Total:  {passed + failed}")

    return passed, failed
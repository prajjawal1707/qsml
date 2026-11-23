
from qsml.benchmarking import BenchmarkRunner


def run_benchmarking_tests():
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

    def test_repeat_count():
        runner = BenchmarkRunner(base_seed=42)

        result = runner.run(
            lambda seed: {
                "energy": float(seed % 5),
                "evaluations": 100,
            },
            repeats=5,
        )

        assert result["repeats"] == 5
        assert len(result["trials"]) == 5

    def test_seed_repeatability():
        runner = BenchmarkRunner(base_seed=42)

        def solver(seed):
            return {"energy": float(seed % 7)}

        first = runner.run(solver, repeats=5)
        second = runner.run(solver, repeats=5)

        first_energies = [
            item["energy"] for item in first["trials"]
        ]
        second_energies = [
            item["energy"] for item in second["trials"]
        ]

        assert first_energies == second_energies

    def test_statistics():
        runner = BenchmarkRunner()

        result = runner.run(
            lambda seed: {"energy": 5.0},
            repeats=3,
        )

        stats = result["energy_statistics"]

        assert stats["mean"] == 5.0
        assert stats["median"] == 5.0
        assert stats["minimum"] == 5.0
        assert stats["maximum"] == 5.0
        assert stats["standard_deviation"] == 0.0

    def test_multiple_solvers():
        runner = BenchmarkRunner()

        result = runner.compare(
            {
                "CPU": lambda seed: {
                    "energy": -2.0,
                    "backend": "CPU",
                },
                "GPU": lambda seed: {
                    "energy": -3.0,
                    "backend": "GPU",
                },
            },
            repeats=2,
        )

        assert len(result["results"]) == 2
        assert result["results"]["CPU"]["best_energy"] == -2.0
        assert result["results"]["GPU"]["best_energy"] == -3.0

    def test_invalid_repeats():
        runner = BenchmarkRunner()
        rejected = False

        try:
            runner.run(
                lambda seed: {"energy": 0.0},
                repeats=0,
            )
        except ValueError:
            rejected = True

        assert rejected

    def test_invalid_energy():
        runner = BenchmarkRunner()
        rejected = False

        try:
            runner.run(
                lambda seed: {"energy": float("nan")},
                repeats=1,
            )
        except ValueError:
            rejected = True

        assert rejected

    check("Repeated benchmark execution", test_repeat_count)
    check("Fixed-seed repeatability", test_seed_repeatability)
    check("Statistical calculations", test_statistics)
    check("Multiple solver comparison", test_multiple_solvers)
    check("Invalid repeat count rejection", test_invalid_repeats)
    check("Invalid energy rejection", test_invalid_energy)

    print("\nQSML 2.5 BENCHMARK TESTS")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Total: {passed + failed}")

    return passed, failed

"""QSML 3.1.1 shared QUBO contract tests."""
import math

from qsml.qubo import QUBOEngine
from qsml.qubo_contract import normalize_qubo_terms
from qsml.validation import evaluate_qubo, solve_qubo_exact, validate_qubo_solution


def run_v31_tests():
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

    def must_raise(function, error=ValueError):
        try:
            function()
        except error:
            return
        raise AssertionError(f"Expected {error.__name__}")

    engine = QUBOEngine()

    check("Reject Boolean variable count", lambda: must_raise(
        lambda: engine.solve(True, {(0, 0): 1.0})
    ))
    check("Reject Boolean variable index", lambda: must_raise(
        lambda: normalize_qubo_terms(2, {(True, 1): 1.0})
    ))
    check("Reject Boolean coefficient", lambda: must_raise(
        lambda: normalize_qubo_terms(1, {(0, 0): True})
    ))
    check("Reject non-dictionary terms", lambda: must_raise(
        lambda: normalize_qubo_terms(2, [((0, 1), 1.0)])
    ))
    check("Reject malformed term key", lambda: must_raise(
        lambda: normalize_qubo_terms(2, {(0, 1, 1): 1.0})
    ))
    check("Reject out-of-range index", lambda: must_raise(
        lambda: normalize_qubo_terms(2, {(0, 2): 1.0})
    ))
    check("Reject NaN coefficient", lambda: must_raise(
        lambda: normalize_qubo_terms(1, {(0, 0): float("nan")})
    ))
    check("Reject infinite coefficient", lambda: must_raise(
        lambda: normalize_qubo_terms(1, {(0, 0): float("inf")})
    ))
    check("Reject overflow during pair aggregation", lambda: must_raise(
        lambda: normalize_qubo_terms(
            2, {(0, 1): 1e308, (1, 0): 1e308}
        )
    ))
    check("Normalize reversed pairs", lambda: (
        (lambda result: (
            result == {(0, 1): 3.0}
            or (_ for _ in ()).throw(AssertionError(result))
        ))(normalize_qubo_terms(2, {(0, 1): 1.0, (1, 0): 2.0}))
    ))
    check("Reject Boolean iteration count", lambda: must_raise(
        lambda: engine.solve(1, {(0, 0): 1.0}, iterations=True)
    ))
    check("Reject Boolean seed", lambda: must_raise(
        lambda: engine.solve(1, {(0, 0): 1.0}, seed=True)
    ))
    check("Reject invalid exact-solver variable limit", lambda: must_raise(
        lambda: solve_qubo_exact(1, {(0, 0): 1.0}, max_variables=True)
    ))
    check("Preserve independent QUBO energy", lambda: (
        (lambda value: (
            math.isclose(value, -3.0)
            or (_ for _ in ()).throw(AssertionError(value))
        ))(evaluate_qubo(2, {(0, 0): -2.0, (1, 1): -2.0, (0, 1): 1.0}, [1, 1]))
    ))
    check("Preserve exact optimum", lambda: (
        (lambda result: (
            result.energy == -3.0 and result.solution == [1, 1]
            or (_ for _ in ()).throw(AssertionError(result))
        ))(solve_qubo_exact(2, {(0, 0): -2.0, (1, 1): -2.0, (0, 1): 1.0}))
    ))
    check("Reject incorrect reported energy", lambda: (
        (lambda report: (
            not report.valid
            or (_ for _ in ()).throw(AssertionError(report))
        ))(validate_qubo_solution(2, {(0, 0): -2.0, (1, 1): -2.0, (0, 1): 1.0}, [1, 1], -2.0))
    ))

    print(f"\nQSML 3.1.1: {passed} passed, {failed} failed")
    return passed, failed

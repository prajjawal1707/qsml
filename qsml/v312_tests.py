"""QSML 3.1.2 exhaustive constraint mathematics and safety tests."""
import math

from qsml.constraints import ConstraintBuilder, add_terms
from qsml.constraint_validation import (
    validate_at_most_encoding,
    validate_equality_encoding,
)


def run_v312_tests():
    passed = 0
    failed = 0
    builder = ConstraintBuilder()

    def check(name, function):
        nonlocal passed, failed
        try:
            function()
            print(f"[PASS] {name}")
            passed += 1
        except Exception as exc:
            print(f"[FAIL] {name}: {exc}")
            failed += 1

    def assert_valid(report):
        assert report.valid, report.errors

    def must_raise(function):
        try:
            function()
        except ValueError:
            return
        raise AssertionError("Expected ValueError")

    check("Equality polynomial exhaustive verification", lambda: assert_valid(
        validate_equality_encoding(
            builder.equality(3, {0: 2, 1: -1, 2: 3}, 2, penalty=4),
            {0: 2, 1: -1, 2: 3}, 2,
        )
    ))
    check("Equality encoding with negative right-hand side", lambda: assert_valid(
        validate_equality_encoding(
            builder.equality(2, {0: 1, 1: 2}, -1, penalty=2),
            {0: 1, 1: 2}, -1,
        )
    ))
    check("Equality encoding with zero right-hand side", lambda: assert_valid(
        validate_equality_encoding(
            builder.equality(2, {0: 1, 1: 2}, 0, penalty=3),
            {0: 1, 1: 2}, 0,
        )
    ))
    check("Equality feasible and infeasible energy separation", lambda: (
        (lambda report: (
            report.feasible_assignments > 0
            and report.infeasible_assignments > 0
            and report.min_infeasible_min_energy > report.baseline_energy
            or (_ for _ in ()).throw(AssertionError(report))
        ))(validate_equality_encoding(
            builder.equality(2, {0: 1, 1: 1}, 1, penalty=5),
            {0: 1, 1: 1}, 1,
        ))
    ))
    check("At-most polynomial exhaustive verification", lambda: assert_valid(
        validate_at_most_encoding(
            builder.at_most(2, {0: 2, 1: 1}, 3, penalty=4),
            {0: 2, 1: 1}, 3,
        )
    ))
    check("At-most encoding with zero limit", lambda: assert_valid(
        validate_at_most_encoding(
            builder.at_most(2, {0: 1, 1: 2}, 0, penalty=3),
            {0: 1, 1: 2}, 0,
        )
    ))
    check("At-most exact-limit assignment is feasible", lambda: (
        (lambda report: (
            report.feasible_assignments > 0
            and math.isclose(report.max_feasible_min_energy, report.baseline_energy)
            or (_ for _ in ()).throw(AssertionError(report))
        ))(validate_at_most_encoding(
            builder.at_most(2, {0: 2, 1: 1}, 3, penalty=4),
            {0: 2, 1: 1}, 3,
        ))
    ))
    check("Slack weights cover every residual through limit", lambda: (
        (lambda encoding: (
            all(any(sum(w * b for w, b in zip(
                encoding["slack_weights"], bits
            )) == residual for bits in __import__("itertools").product(
                (0, 1), repeat=encoding["slack_variables"]
            )) for residual in range(6))
            or (_ for _ in ()).throw(AssertionError(encoding))
        ))(builder.at_most(2, {0: 2, 1: 1}, 5))
    ))
    check("Infeasible at-most assignments receive positive excess penalty", lambda: (
        (lambda report: (
            report.infeasible_assignments > 0
            and report.min_infeasible_min_energy > report.baseline_energy
            or (_ for _ in ()).throw(AssertionError(report))
        ))(validate_at_most_encoding(
            builder.at_most(2, {0: 2, 1: 2}, 1, penalty=7),
            {0: 2, 1: 2}, 1,
        ))
    ))
    check("At-most encoding with sparse original indices", lambda: assert_valid(
        validate_at_most_encoding(
            builder.at_most(4, {1: 2, 3: 1}, 2),
            {1: 2, 3: 1}, 2,
        )
    ))
    check("At-most encoding with no slack bits", lambda: assert_valid(
        validate_at_most_encoding(
            builder.at_most(2, {0: 1, 1: 1}, 0),
            {0: 1, 1: 1}, 0,
        )
    ))
    check("Reject fractional at-most limit", lambda: must_raise(
        lambda: builder.at_most(2, {0: 1}, 1.5)
    ))
    check("Reject fractional at-most coefficient", lambda: must_raise(
        lambda: builder.at_most(2, {0: 1.5}, 2)
    ))
    check("Reject negative at-most coefficient", lambda: must_raise(
        lambda: builder.at_most(2, {0: -1}, 2)
    ))
    check("Reject negative at-most limit", lambda: must_raise(
        lambda: builder.at_most(2, {0: 1}, -1)
    ))
    check("Reject zero equality penalty", lambda: must_raise(
        lambda: builder.equality(1, {0: 1}, 1, penalty=0)
    ))
    check("Reject non-finite equality right-hand side", lambda: must_raise(
        lambda: builder.equality(1, {0: 1}, float("inf"))
    ))
    check("Reject overflow in squared equality coefficients", lambda: must_raise(
        lambda: builder.equality(1, {0: 1e308}, 0, penalty=10)
    ))
    check("Reject overflow while merging QUBO terms", lambda: must_raise(
        lambda: add_terms({(0, 0): 1e308}, {(0, 0): 1e308})
    ))
    check("Reject malformed term keys while combining", lambda: must_raise(
        lambda: add_terms({(0, 1, 2): 1.0}, {})
    ))
    check("Combine valid objective and penalty terms", lambda: (
        (lambda terms: (
            terms == {(0, 0): -1.0, (0, 1): 3.0}
            or (_ for _ in ()).throw(AssertionError(terms))
        ))(add_terms(
            {(0, 0): -2.0, (0, 1): 1.0},
            {(0, 0): 1.0, (0, 1): 2.0},
        ))
    ))

    print(f"\nQSML 3.1.2: {passed} passed, {failed} failed")
    return passed, failed

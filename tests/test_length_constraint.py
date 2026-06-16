import pytest

from llmcalibre import LengthConstraint


def test_length_constraint_passes_within_bounds() -> None:
    result = LengthConstraint(min_chars=2, max_chars=5).evaluate("hey")

    assert result.score == 1.0
    assert result.passed is True
    assert result.rationale == "Output length is valid: 3 characters."
    assert result.metadata == {"length": 3, "min_chars": 2, "max_chars": 5}


def test_length_constraint_fails_below_minimum() -> None:
    result = LengthConstraint(min_chars=5).evaluate("hey")

    assert result.score == 0.0
    assert result.passed is False
    assert result.rationale == "Output is too short: 3 characters (minimum 5)."


def test_length_constraint_fails_above_maximum() -> None:
    result = LengthConstraint(max_chars=2).evaluate("hey")

    assert result.score == 0.0
    assert result.passed is False
    assert result.rationale == "Output is too long: 3 characters (maximum 2)."


def test_length_constraint_rejects_impossible_bounds() -> None:
    with pytest.raises(ValueError, match="min_chars cannot be greater"):
        LengthConstraint(min_chars=10, max_chars=5)

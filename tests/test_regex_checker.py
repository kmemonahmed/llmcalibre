import re

import pytest

from llmcalibre import EvalResult, RegexChecker


def test_regex_checker_passes_when_required_pattern_matches() -> None:
    result = RegexChecker(required_patterns=[r"\bParis\b"]).evaluate(
        "Paris is capital"
    )

    assert result == EvalResult(
        score=1.0,
        passed=True,
        rationale=r"Regex check passed. Matched required patterns: \bParis\b.",
        metadata={
            "matched_required_patterns": [r"\bParis\b"],
            "missing_required_patterns": [],
            "found_forbidden_patterns": [],
            "allowed_forbidden_patterns": [],
            "flags": 0,
        },
    )


def test_regex_checker_matches_date_pattern() -> None:
    result = RegexChecker(required_patterns=[r"\d{4}-\d{2}-\d{2}"]).evaluate(
        "Date: 2026-06-16"
    )

    assert result.score == 1.0
    assert result.passed is True
    assert result.metadata["matched_required_patterns"] == [r"\d{4}-\d{2}-\d{2}"]


def test_regex_checker_scores_partial_required_matches() -> None:
    result = RegexChecker(
        required_patterns=[r"\bParis\b", r"\bFrance\b"],
    ).evaluate("Paris")

    assert result.score == 0.5
    assert result.passed is False
    assert result.metadata["matched_required_patterns"] == [r"\bParis\b"]
    assert result.metadata["missing_required_patterns"] == [r"\bFrance\b"]


def test_regex_checker_fails_when_forbidden_pattern_matches() -> None:
    result = RegexChecker(forbidden_patterns=[r"(?i)password"]).evaluate(
        "Do not print this password."
    )

    assert result.score == 0.0
    assert result.passed is False
    assert result.metadata["found_forbidden_patterns"] == [r"(?i)password"]
    assert result.metadata["allowed_forbidden_patterns"] == []


def test_regex_checker_passes_when_forbidden_patterns_are_absent() -> None:
    result = RegexChecker(forbidden_patterns=[r"TODO", r"FIXME"]).evaluate(
        "Ready to ship."
    )

    assert result.score == 1.0
    assert result.passed is True
    assert result.metadata["found_forbidden_patterns"] == []
    assert result.metadata["allowed_forbidden_patterns"] == [r"TODO", r"FIXME"]
    assert result.rationale == (
        "Regex check passed. All forbidden patterns were avoided."
    )


def test_regex_checker_combines_required_and_forbidden_scores() -> None:
    result = RegexChecker(
        required_patterns=[r"```json"],
        forbidden_patterns=[r"TODO"],
    ).evaluate("```json\n{\"answer\": true}\n```\nTODO")

    assert result.score == 0.5
    assert result.passed is False
    assert result.metadata["matched_required_patterns"] == [r"```json"]
    assert result.metadata["missing_required_patterns"] == []
    assert result.metadata["found_forbidden_patterns"] == [r"TODO"]
    assert result.metadata["allowed_forbidden_patterns"] == []


def test_regex_checker_honors_flags() -> None:
    result = RegexChecker(required_patterns=[r"^answer"], flags=re.IGNORECASE).evaluate(
        "Answer: yes"
    )

    assert result.score == 1.0
    assert result.passed is True
    assert result.metadata["flags"] == re.IGNORECASE


def test_regex_checker_callable_behavior() -> None:
    checker = RegexChecker(required_patterns=[r"ok"])

    assert checker("ok").score == 1.0


def test_regex_checker_raises_clear_error_for_invalid_required_regex() -> None:
    with pytest.raises(ValueError, match="Invalid regex in required_patterns"):
        RegexChecker(required_patterns=["["])


def test_regex_checker_raises_clear_error_for_invalid_forbidden_regex() -> None:
    with pytest.raises(ValueError, match="Invalid regex in forbidden_patterns"):
        RegexChecker(forbidden_patterns=["["])


@pytest.mark.parametrize(
    ("required_patterns", "forbidden_patterns"),
    [
        (None, None),
        ([], []),
        ([], None),
        (None, []),
    ],
)
def test_regex_checker_requires_at_least_one_pattern(
    required_patterns: list[str] | None,
    forbidden_patterns: list[str] | None,
) -> None:
    with pytest.raises(ValueError, match="required_patterns or forbidden_patterns"):
        RegexChecker(
            required_patterns=required_patterns,
            forbidden_patterns=forbidden_patterns,
        )

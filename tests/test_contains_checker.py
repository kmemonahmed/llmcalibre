import pytest

from llmcalibre import ContainsChecker, EvalResult


def test_contains_checker_passes_when_required_term_is_present() -> None:
    result = ContainsChecker(required=["Paris"]).evaluate("Paris is capital")

    assert result == EvalResult(
        score=1.0,
        passed=True,
        rationale="Contains check passed. Matched required terms: Paris.",
        metadata={
            "matched_required": ["Paris"],
            "missing_required": [],
            "found_forbidden": [],
            "allowed_forbidden": [],
            "case_sensitive": False,
        },
    )


def test_contains_checker_scores_partial_required_matches() -> None:
    result = ContainsChecker(required=["Paris", "France"]).evaluate("Paris")

    assert result.score == 0.5
    assert result.passed is False
    assert result.metadata["matched_required"] == ["Paris"]
    assert result.metadata["missing_required"] == ["France"]
    assert result.rationale == (
        "Contains check failed. Matched required terms: Paris. "
        "Missing required terms: France."
    )


def test_contains_checker_fails_when_forbidden_term_is_found() -> None:
    result = ContainsChecker(forbidden=["violence"]).evaluate(
        "The answer contains violence."
    )

    assert result.score == 0.0
    assert result.passed is False
    assert result.metadata["found_forbidden"] == ["violence"]
    assert result.metadata["allowed_forbidden"] == []


def test_contains_checker_combines_required_and_forbidden_scores() -> None:
    result = ContainsChecker(
        required=["json"],
        forbidden=["markdown"],
    ).evaluate("Return JSON, not markdown.")

    assert result.score == 0.5
    assert result.passed is False
    assert result.metadata["matched_required"] == ["json"]
    assert result.metadata["missing_required"] == []
    assert result.metadata["found_forbidden"] == ["markdown"]
    assert result.metadata["allowed_forbidden"] == []


def test_contains_checker_is_case_insensitive_by_default() -> None:
    result = ContainsChecker(required=["json"]).evaluate("Return JSON.")

    assert result.score == 1.0
    assert result.passed is True
    assert result.metadata["matched_required"] == ["json"]
    assert result.metadata["case_sensitive"] is False


def test_contains_checker_can_be_case_sensitive() -> None:
    result = ContainsChecker(required=["json"], case_sensitive=True).evaluate(
        "Return JSON."
    )

    assert result.score == 0.0
    assert result.passed is False
    assert result.metadata["missing_required"] == ["json"]
    assert result.metadata["case_sensitive"] is True


def test_contains_checker_passes_when_all_forbidden_terms_are_absent() -> None:
    result = ContainsChecker(forbidden=["markdown", "xml"]).evaluate("Return JSON.")

    assert result.score == 1.0
    assert result.passed is True
    assert result.metadata["found_forbidden"] == []
    assert result.metadata["allowed_forbidden"] == ["markdown", "xml"]
    assert result.rationale == (
        "Contains check passed. All forbidden terms were avoided."
    )


def test_contains_checker_honors_custom_threshold() -> None:
    result = ContainsChecker(required=["Paris", "France"], threshold=0.5).evaluate(
        "Paris"
    )

    assert result.score == 0.5
    assert result.passed is True


@pytest.mark.parametrize(
    ("required", "forbidden"),
    [
        (None, None),
        ([], []),
        ([], None),
        (None, []),
    ],
)
def test_contains_checker_requires_at_least_one_term(
    required: list[str] | None,
    forbidden: list[str] | None,
) -> None:
    with pytest.raises(ValueError, match="required or forbidden"):
        ContainsChecker(required=required, forbidden=forbidden)

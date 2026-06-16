from llmcalibre import EvalResult, FormatChecker


def test_format_checker_passes_valid_json() -> None:
    result = FormatChecker(format="json").evaluate('{"answer": 42}')

    assert result == EvalResult(
        score=1.0,
        passed=True,
        rationale="Output is valid JSON.",
        metadata={"format": "json"},
    )


def test_format_checker_fails_invalid_json_without_raising() -> None:
    result = FormatChecker(format="json").evaluate("{bad json")

    assert result.score == 0.0
    assert result.passed is False
    assert result.rationale == "Output is not valid JSON."
    assert result.metadata["format"] == "json"
    assert "error" in result.metadata


def test_format_checker_handles_non_string_input_without_raising() -> None:
    result = FormatChecker(format="json").evaluate(None)  # type: ignore[arg-type]

    assert result.score == 0.0
    assert result.passed is False


def test_format_checker_reports_unsupported_format() -> None:
    result = FormatChecker(format="xml").evaluate("<answer />")

    assert result.score == 0.0
    assert result.passed is False
    assert result.rationale == "Unsupported format: xml"

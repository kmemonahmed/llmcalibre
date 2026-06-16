from llmcalibre import EvalResult


def test_eval_result_defaults_metadata() -> None:
    result = EvalResult(score=0.75, passed=True, rationale="Looks good.")

    assert result.score == 0.75
    assert result.passed is True
    assert result.rationale == "Looks good."
    assert result.metadata == {}


def test_eval_result_metadata_instances_are_independent() -> None:
    first = EvalResult(score=1.0, passed=True, rationale="First.")
    second = EvalResult(score=1.0, passed=True, rationale="Second.")

    first.metadata["key"] = "value"

    assert second.metadata == {}

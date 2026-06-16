import pytest

from llmcalibre import BaseEvaluator, EvalResult, assert_eval


class StaticEvaluator(BaseEvaluator):
    def __init__(self, result: EvalResult) -> None:
        super().__init__()
        self.result = result

    def evaluate(self, output: str) -> EvalResult:
        del output
        return self.result


def test_assert_eval_passing_evaluators_do_not_raise() -> None:
    assert_eval(
        "output",
        evaluators=[
            StaticEvaluator(EvalResult(score=1.0, passed=True, rationale="Passed.")),
        ],
    )


def test_assert_eval_failing_evaluator_raises_assertion_error() -> None:
    evaluator = StaticEvaluator(
        EvalResult(score=0.25, passed=False, rationale="Missing required term.")
    )

    with pytest.raises(AssertionError) as exc_info:
        assert_eval("output", evaluators=[evaluator])

    message = str(exc_info.value)
    assert "StaticEvaluator" in message
    assert "score: 0.250" in message
    assert "Missing required term." in message


def test_assert_eval_empty_evaluator_list_raises_value_error() -> None:
    with pytest.raises(ValueError, match="evaluators must not be empty"):
        assert_eval("output", evaluators=[])


def test_assert_eval_includes_multiple_failures() -> None:
    first = StaticEvaluator(
        EvalResult(score=0.0, passed=False, rationale="First failed.")
    )
    second = StaticEvaluator(
        EvalResult(score=0.5, passed=False, rationale="Second failed.")
    )

    with pytest.raises(AssertionError) as exc_info:
        assert_eval("output", evaluators=[first, second])

    message = str(exc_info.value)
    assert message.count("StaticEvaluator") == 2
    assert "score: 0.000" in message
    assert "score: 0.500" in message
    assert "First failed." in message
    assert "Second failed." in message

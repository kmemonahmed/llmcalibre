from typing import Any

import pytest

from llmcalibre import BaseEvaluator, EvalResult


class ConcreteEvaluator(BaseEvaluator):
    def evaluate(self, output: str, **kwargs: Any) -> EvalResult:
        del output, kwargs
        return EvalResult(score=1.0, passed=True, rationale="Passed.")


def test_base_evaluator_allows_threshold_zero() -> None:
    evaluator = ConcreteEvaluator(threshold=0.0)

    assert evaluator.threshold == 0.0


def test_base_evaluator_allows_threshold_one() -> None:
    evaluator = ConcreteEvaluator(threshold=1.0)

    assert evaluator.threshold == 1.0


def test_base_evaluator_rejects_threshold_below_zero() -> None:
    with pytest.raises(ValueError, match="threshold must be between 0.0 and 1.0"):
        ConcreteEvaluator(threshold=-0.1)


def test_base_evaluator_rejects_threshold_above_one() -> None:
    with pytest.raises(ValueError, match="threshold must be between 0.0 and 1.0"):
        ConcreteEvaluator(threshold=1.1)

"""Pipeline orchestration for running multiple evaluators."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from llmcalibre.core.base import BaseEvaluator
from llmcalibre.core.result import EvalResult


@dataclass(frozen=True)
class PipelineSummary:
    """Aggregate statistics for a set of evaluation results."""

    total_passed: int
    total_count: int
    average_score: float


class EvalPipeline:
    """Run multiple evaluators against the same output."""

    def __init__(self, evaluators: Sequence[BaseEvaluator]) -> None:
        """Create a pipeline from an ordered sequence of evaluators."""
        self.evaluators = list(evaluators)
        self._last_results: list[EvalResult] = []

    def run(self, output: str, **kwargs: Any) -> list[EvalResult]:
        """Run every evaluator against one output."""
        self._last_results = [
            evaluator.evaluate(output, **kwargs) for evaluator in self.evaluators
        ]
        return list(self._last_results)

    def __call__(self, output: str, **kwargs: Any) -> list[EvalResult]:
        """Run the pipeline by calling the pipeline instance."""
        return self.run(output, **kwargs)

    def summary(
        self, results: Sequence[EvalResult] | None = None
    ) -> PipelineSummary:
        """Summarize evaluation results.

        If no results are provided, the most recent pipeline run is summarized.
        """
        result_list = list(self._last_results if results is None else results)
        total_count = len(result_list)
        total_passed = sum(result.passed for result in result_list)
        average_score = (
            sum(result.score for result in result_list) / total_count
            if total_count
            else 0.0
        )
        return PipelineSummary(
            total_passed=total_passed,
            total_count=total_count,
            average_score=average_score,
        )

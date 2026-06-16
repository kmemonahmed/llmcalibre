"""Pytest-oriented assertion helpers."""

from __future__ import annotations

from llmcalibre.core.base import BaseEvaluator
from llmcalibre.core.pipeline import EvalPipeline


def assert_eval(output: str, evaluators: list[BaseEvaluator]) -> None:
    """Assert that all evaluators pass for an output.

    Raises:
        ValueError: If no evaluators are provided.
        AssertionError: If one or more evaluators fail.
    """
    if not evaluators:
        raise ValueError("evaluators must not be empty")

    pipeline = EvalPipeline(evaluators)
    results = pipeline.run(output)
    failures = [
        (evaluator, result)
        for evaluator, result in zip(evaluators, results)
        if not result.passed
    ]

    if failures:
        lines = ["llmcalibre evaluation failed:"]
        for evaluator, result in failures:
            lines.extend(
                [
                    f"- {evaluator.__class__.__name__}",
                    f"  score: {result.score:.3f}",
                    f"  rationale: {result.rationale}",
                ]
            )
        raise AssertionError("\n".join(lines))

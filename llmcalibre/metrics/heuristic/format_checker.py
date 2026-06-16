"""Heuristic format validation evaluators."""

from __future__ import annotations

import json
from typing import Any

from llmcalibre.core.base import BaseEvaluator
from llmcalibre.core.result import EvalResult


class FormatChecker(BaseEvaluator):
    """Check whether an output matches a requested data format."""

    def __init__(self, format: str = "json", threshold: float = 0.5) -> None:
        """Create a format checker.

        Currently only ``format="json"`` is supported.
        """
        super().__init__(threshold=threshold)
        self.format = format.lower()

    def evaluate(self, output: str, **kwargs: Any) -> EvalResult:
        """Return 1.0 for valid JSON and 0.0 otherwise."""
        del kwargs
        if self.format != "json":
            return EvalResult(
                score=0.0,
                passed=False,
                rationale=f"Unsupported format: {self.format}",
                metadata={"format": self.format},
            )

        try:
            json.loads(output)
        except (TypeError, json.JSONDecodeError) as error:
            return EvalResult(
                score=0.0,
                passed=False,
                rationale="Output is not valid JSON.",
                metadata={"format": self.format, "error": str(error)},
            )

        return EvalResult(
            score=1.0,
            passed=self.threshold <= 1.0,
            rationale="Output is valid JSON.",
            metadata={"format": self.format},
        )

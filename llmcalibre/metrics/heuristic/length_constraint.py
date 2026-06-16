"""Heuristic length constraint evaluator."""

from __future__ import annotations

from typing import Any

from llmcalibre.core.base import BaseEvaluator
from llmcalibre.core.result import EvalResult


class LengthConstraint(BaseEvaluator):
    """Check whether an output length falls within character limits."""

    def __init__(
        self,
        min_chars: int | None = None,
        max_chars: int | None = None,
        threshold: float = 0.5,
    ) -> None:
        """Create a length constraint with optional lower and upper bounds."""
        super().__init__(threshold=threshold)
        if min_chars is not None and min_chars < 0:
            raise ValueError("min_chars must be non-negative")
        if max_chars is not None and max_chars < 0:
            raise ValueError("max_chars must be non-negative")
        if (
            min_chars is not None
            and max_chars is not None
            and min_chars > max_chars
        ):
            raise ValueError("min_chars cannot be greater than max_chars")
        self.min_chars = min_chars
        self.max_chars = max_chars

    def evaluate(self, output: str, **kwargs: Any) -> EvalResult:
        """Return 1.0 when output length satisfies configured bounds."""
        del kwargs
        length = len(output)
        metadata = {
            "length": length,
            "min_chars": self.min_chars,
            "max_chars": self.max_chars,
        }

        if self.min_chars is not None and length < self.min_chars:
            return EvalResult(
                score=0.0,
                passed=False,
                rationale=(
                    f"Output is too short: {length} characters "
                    f"(minimum {self.min_chars})."
                ),
                metadata=metadata,
            )

        if self.max_chars is not None and length > self.max_chars:
            return EvalResult(
                score=0.0,
                passed=False,
                rationale=(
                    f"Output is too long: {length} characters "
                    f"(maximum {self.max_chars})."
                ),
                metadata=metadata,
            )

        return EvalResult(
            score=1.0,
            passed=self.threshold <= 1.0,
            rationale=f"Output length is valid: {length} characters.",
            metadata=metadata,
        )

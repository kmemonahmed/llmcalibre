"""Abstract evaluator contract."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from llmcalibre.core.result import EvalResult


class BaseEvaluator(ABC):
    """Base class for all evaluators."""

    threshold: float = 0.5

    def __init__(self, threshold: float = 0.5) -> None:
        """Create an evaluator with a pass threshold."""
        if threshold < 0.0 or threshold > 1.0:
            raise ValueError("threshold must be between 0.0 and 1.0")
        self.threshold = threshold

    @abstractmethod
    def evaluate(self, output: str, **kwargs: Any) -> EvalResult:
        """Evaluate an LLM output and return a normalized result."""
        raise NotImplementedError

    def __call__(self, output: str, **kwargs: Any) -> EvalResult:
        """Evaluate an output by calling the evaluator instance."""
        return self.evaluate(output, **kwargs)

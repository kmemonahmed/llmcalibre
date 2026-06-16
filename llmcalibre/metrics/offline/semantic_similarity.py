"""Semantic similarity evaluator backed by sentence-transformers."""

from __future__ import annotations

import importlib
import math
from collections.abc import Sequence
from typing import Any

from llmcalibre.core.base import BaseEvaluator
from llmcalibre.core.result import EvalResult

MISSING_SENTENCE_TRANSFORMERS_MESSAGE = (
    "SemanticSimilarity requires sentence-transformers. "
    "Install with: pip install llmcalibre[nlp]"
)


class SemanticSimilarity(BaseEvaluator):
    """Measure semantic similarity between output and reference text."""

    def __init__(
        self,
        threshold: float = 0.7,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    ) -> None:
        """Create a semantic similarity evaluator and load the embedding model."""
        super().__init__(threshold=threshold)
        self.model_name = model_name
        self._model = self._load_model(model_name)

    def evaluate(self, output: str, **kwargs: Any) -> EvalResult:
        """Compare output to a reference text using cosine similarity.

        Pass the reference text as ``reference="..."``.
        """
        reference = kwargs.get("reference")
        if not isinstance(reference, str):
            raise ValueError("SemanticSimilarity requires reference text")

        output_embedding, reference_embedding = self._model.encode([output, reference])
        score = self._normalized_cosine_similarity(
            self._to_float_sequence(output_embedding),
            self._to_float_sequence(reference_embedding),
        )
        passed = score >= self.threshold
        status = "passes" if passed else "fails"

        return EvalResult(
            score=score,
            passed=passed,
            rationale=(
                f"Semantic similarity {status} threshold "
                f"{self.threshold:.3f} with score {score:.3f}."
            ),
            metadata={"model_name": self.model_name},
        )

    @staticmethod
    def _load_model(model_name: str) -> Any:
        """Load sentence-transformers lazily."""
        try:
            sentence_transformers = importlib.import_module("sentence_transformers")
        except ImportError as error:
            raise ImportError(MISSING_SENTENCE_TRANSFORMERS_MESSAGE) from error

        return sentence_transformers.SentenceTransformer(model_name)

    @staticmethod
    def _to_float_sequence(embedding: Any) -> list[float]:
        """Convert embedding-like values to plain floats."""
        return [float(value) for value in embedding]

    @staticmethod
    def _normalized_cosine_similarity(
        first: Sequence[float],
        second: Sequence[float],
    ) -> float:
        """Return cosine similarity normalized from [-1, 1] to [0, 1]."""
        dot = sum(left * right for left, right in zip(first, second))
        first_norm = math.sqrt(sum(value * value for value in first))
        second_norm = math.sqrt(sum(value * value for value in second))
        if first_norm == 0.0 or second_norm == 0.0:
            return 0.0

        cosine = dot / (first_norm * second_norm)
        bounded_cosine = max(-1.0, min(1.0, cosine))
        return (bounded_cosine + 1.0) / 2.0

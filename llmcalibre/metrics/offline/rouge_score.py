"""ROUGE overlap evaluator backed by rouge-score."""

from __future__ import annotations

import importlib
from typing import Any

from llmcalibre.core.base import BaseEvaluator
from llmcalibre.core.result import EvalResult

MISSING_ROUGE_SCORE_MESSAGE = (
    "RougeScore requires rouge-score. Install with: pip install llmcalibre[nlp]"
)

SUPPORTED_ROUGE_TYPES = {"rouge1", "rouge2", "rougeL"}


class RougeScore(BaseEvaluator):
    """Measure text overlap between output and reference using ROUGE."""

    def __init__(
        self,
        rouge_type: str = "rougeL",
        threshold: float = 0.5,
        use_stemmer: bool = True,
    ) -> None:
        """Create a ROUGE evaluator."""
        super().__init__(threshold=threshold)
        if rouge_type not in SUPPORTED_ROUGE_TYPES:
            supported = ", ".join(sorted(SUPPORTED_ROUGE_TYPES))
            raise ValueError(
                f"Invalid rouge_type: {rouge_type}. Supported: {supported}"
            )

        self.rouge_type = rouge_type
        self.use_stemmer = use_stemmer
        scorer_cls = self._load_scorer_class()
        self._scorer = scorer_cls([rouge_type], use_stemmer=use_stemmer)

    def evaluate(self, output: str, **kwargs: Any) -> EvalResult:
        """Compare output to reference text and return the ROUGE F1 score.

        Pass the reference text as ``reference="..."``.
        """
        reference = kwargs.get("reference")
        if not isinstance(reference, str):
            raise ValueError("RougeScore requires reference text")

        if not output or not reference:
            return EvalResult(
                score=0.0,
                passed=False,
                rationale="ROUGE score is 0.000 because output or reference is empty.",
                metadata={
                    "rouge_type": self.rouge_type,
                    "precision": 0.0,
                    "recall": 0.0,
                    "fmeasure": 0.0,
                    "use_stemmer": self.use_stemmer,
                },
            )

        scores = self._scorer.score(reference, output)
        score = scores[self.rouge_type]
        fmeasure = float(score.fmeasure)
        precision = float(score.precision)
        recall = float(score.recall)
        passed = fmeasure >= self.threshold
        status = "passes" if passed else "fails"

        return EvalResult(
            score=fmeasure,
            passed=passed,
            rationale=(
                f"{self.rouge_type} F1 score {status} threshold "
                f"{self.threshold:.3f} with score {fmeasure:.3f}."
            ),
            metadata={
                "rouge_type": self.rouge_type,
                "precision": precision,
                "recall": recall,
                "fmeasure": fmeasure,
                "use_stemmer": self.use_stemmer,
            },
        )

    @staticmethod
    def _load_scorer_class() -> Any:
        """Load rouge-score lazily."""
        try:
            rouge_scorer = importlib.import_module("rouge_score.rouge_scorer")
        except ImportError as error:
            raise ImportError(MISSING_ROUGE_SCORE_MESSAGE) from error

        return rouge_scorer.RougeScorer

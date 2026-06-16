from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any

import pytest

from llmcalibre import RougeScore
from llmcalibre.metrics import RougeScore as MetricsRougeScore
from llmcalibre.metrics.offline import RougeScore as OfflineRougeScore
from llmcalibre.metrics.offline.rouge_score import MISSING_ROUGE_SCORE_MESSAGE


@dataclass(frozen=True)
class FakeScore:
    precision: float
    recall: float
    fmeasure: float


class FakeScorer:
    def __init__(self, scores: dict[str, FakeScore]) -> None:
        self.scores = scores
        self.seen_reference = ""
        self.seen_output = ""

    def score(self, reference: str, output: str) -> dict[str, FakeScore]:
        self.seen_reference = reference
        self.seen_output = output
        return self.scores


def _patch_scorer(
    monkeypatch: pytest.MonkeyPatch,
    fake_scorer: FakeScorer,
) -> None:
    monkeypatch.setattr(
        RougeScore,
        "_load_scorer_class",
        staticmethod(lambda: lambda rouge_types, use_stemmer: fake_scorer),
    )


def test_rouge_score_passes_for_valid_score(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_scorer = FakeScorer({"rougeL": FakeScore(0.8, 0.9, 0.85)})
    _patch_scorer(monkeypatch, fake_scorer)

    checker = RougeScore(rouge_type="rougeL", threshold=0.5, use_stemmer=False)
    result = checker.evaluate("Paris is capital", reference="Paris is capital")

    assert result.score == 0.85
    assert result.passed is True
    assert result.metadata == {
        "rouge_type": "rougeL",
        "precision": 0.8,
        "recall": 0.9,
        "fmeasure": 0.85,
        "use_stemmer": False,
    }
    assert fake_scorer.seen_reference == "Paris is capital"
    assert fake_scorer.seen_output == "Paris is capital"


def test_rouge_score_fails_for_low_score(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_scorer(
        monkeypatch,
        FakeScorer({"rouge1": FakeScore(0.2, 0.3, 0.25)}),
    )

    result = RougeScore(rouge_type="rouge1", threshold=0.5).evaluate(
        "short answer",
        reference="different reference",
    )

    assert result.score == 0.25
    assert result.passed is False
    assert "fails threshold" in result.rationale


def test_rouge_score_invalid_rouge_type_raises_value_error() -> None:
    with pytest.raises(ValueError, match="Invalid rouge_type"):
        RougeScore(rouge_type="rouge3")


def test_rouge_score_missing_dependency_raises_import_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_import_module = importlib.import_module

    def fake_import_module(name: str, package: str | None = None) -> Any:
        if name == "rouge_score.rouge_scorer":
            raise ImportError("missing")
        return real_import_module(name, package)

    monkeypatch.setattr(importlib, "import_module", fake_import_module)

    with pytest.raises(ImportError, match="rouge-score"):
        RougeScore()

    with pytest.raises(ImportError, match="llmcalibre\\[nlp\\]"):
        RougeScore()


def test_rouge_score_empty_output_or_reference_returns_zero(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_scorer(
        monkeypatch,
        FakeScorer({"rougeL": FakeScore(1.0, 1.0, 1.0)}),
    )

    result = RougeScore().evaluate("", reference="reference")

    assert result.score == 0.0
    assert result.passed is False
    assert result.metadata["fmeasure"] == 0.0
    assert "empty" in result.rationale


def test_rouge_score_callable_behavior(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_scorer(
        monkeypatch,
        FakeScorer({"rouge2": FakeScore(0.7, 0.8, 0.75)}),
    )

    result = RougeScore(rouge_type="rouge2")("output", reference="reference")

    assert result.score == 0.75
    assert result.passed is True


def test_rouge_score_requires_reference(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_scorer(
        monkeypatch,
        FakeScorer({"rougeL": FakeScore(0.7, 0.8, 0.75)}),
    )

    with pytest.raises(ValueError, match="reference text"):
        RougeScore().evaluate("output")


def test_rouge_score_import_error_message_constant() -> None:
    assert MISSING_ROUGE_SCORE_MESSAGE == (
        "RougeScore requires rouge-score. Install with: pip install llmcalibre[nlp]"
    )


def test_rouge_score_exports_work() -> None:
    assert RougeScore is MetricsRougeScore
    assert RougeScore is OfflineRougeScore

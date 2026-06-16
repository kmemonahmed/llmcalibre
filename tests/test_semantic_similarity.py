from __future__ import annotations

import importlib
from typing import Any

import pytest

from llmcalibre import SemanticSimilarity
from llmcalibre.metrics import SemanticSimilarity as MetricsSemanticSimilarity
from llmcalibre.metrics.offline import SemanticSimilarity as OfflineSemanticSimilarity
from llmcalibre.metrics.offline.semantic_similarity import (
    MISSING_SENTENCE_TRANSFORMERS_MESSAGE,
)


class FakeModel:
    def __init__(self, embeddings: list[list[float]]) -> None:
        self.embeddings = embeddings
        self.seen_inputs: list[str] = []

    def encode(self, inputs: list[str]) -> list[list[float]]:
        self.seen_inputs = inputs
        return self.embeddings


def test_semantic_similarity_passes_for_similar_embeddings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_model = FakeModel([[1.0, 0.0], [1.0, 0.0]])
    monkeypatch.setattr(
        SemanticSimilarity,
        "_load_model",
        staticmethod(lambda model_name: fake_model),
    )

    checker = SemanticSimilarity(threshold=0.9, model_name="fake-model")
    result = checker.evaluate("hello", reference="hi")

    assert result.score == 1.0
    assert result.passed is True
    assert result.metadata == {"model_name": "fake-model"}
    assert fake_model.seen_inputs == ["hello", "hi"]


def test_semantic_similarity_fails_for_dissimilar_embeddings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        SemanticSimilarity,
        "_load_model",
        staticmethod(lambda model_name: FakeModel([[1.0, 0.0], [-1.0, 0.0]])),
    )

    result = SemanticSimilarity(threshold=0.7).evaluate("hello", reference="bye")

    assert result.score == 0.0
    assert result.passed is False
    assert "fails threshold" in result.rationale


def test_semantic_similarity_normalizes_orthogonal_embeddings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        SemanticSimilarity,
        "_load_model",
        staticmethod(lambda model_name: FakeModel([[1.0, 0.0], [0.0, 1.0]])),
    )

    result = SemanticSimilarity(threshold=0.5).evaluate("hello", reference="topic")

    assert result.score == 0.5
    assert result.passed is True


def test_semantic_similarity_requires_reference(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        SemanticSimilarity,
        "_load_model",
        staticmethod(lambda model_name: FakeModel([[1.0], [1.0]])),
    )

    checker = SemanticSimilarity()

    with pytest.raises(ValueError, match="reference text"):
        checker.evaluate("hello")


def test_semantic_similarity_import_error_handling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_import_module = importlib.import_module

    def fake_import_module(name: str, package: str | None = None) -> Any:
        if name == "sentence_transformers":
            raise ImportError("missing")
        return real_import_module(name, package)

    monkeypatch.setattr(importlib, "import_module", fake_import_module)

    with pytest.raises(ImportError, match="sentence-transformers"):
        SemanticSimilarity()

    with pytest.raises(ImportError, match="llmcalibre\\[nlp\\]"):
        SemanticSimilarity()


def test_semantic_similarity_import_error_message_constant() -> None:
    assert MISSING_SENTENCE_TRANSFORMERS_MESSAGE == (
        "SemanticSimilarity requires sentence-transformers. "
        "Install with: pip install llmcalibre[nlp]"
    )


def test_semantic_similarity_exports_work() -> None:
    assert SemanticSimilarity is MetricsSemanticSimilarity
    assert SemanticSimilarity is OfflineSemanticSimilarity

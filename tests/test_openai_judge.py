from __future__ import annotations

import importlib
from types import SimpleNamespace
from typing import Any

import pytest

from llmcalibre import OpenAIJudge
from llmcalibre.metrics import OpenAIJudge as MetricsOpenAIJudge
from llmcalibre.metrics.judge import OpenAIJudge as JudgeOpenAIJudge
from llmcalibre.metrics.judge.openai_judge import MISSING_OPENAI_MESSAGE


class FakeCompletions:
    def __init__(self, raw_response: str | None = None, error: Exception | None = None):
        self.raw_response = raw_response
        self.error = error
        self.seen_kwargs: dict[str, Any] = {}

    def create(self, **kwargs: Any) -> Any:
        self.seen_kwargs = kwargs
        if self.error is not None:
            raise self.error
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content=self.raw_response),
                )
            ]
        )


class FakeClient:
    def __init__(self, completions: FakeCompletions) -> None:
        self.chat = SimpleNamespace(completions=completions)


def _patch_client(
    monkeypatch: pytest.MonkeyPatch,
    completions: FakeCompletions,
) -> None:
    def fake_client_cls(**kwargs: Any) -> FakeClient:
        del kwargs
        return FakeClient(completions)

    monkeypatch.setattr(
        OpenAIJudge,
        "_load_client_class",
        staticmethod(lambda: fake_client_cls),
    )


def test_openai_judge_valid_json_passes(monkeypatch: pytest.MonkeyPatch) -> None:
    completions = FakeCompletions(
        '{"score": 0.9, "passed": false, "rationale": "Looks good."}'
    )
    _patch_client(monkeypatch, completions)

    judge = OpenAIJudge(api_key="test", base_url="https://example.com", threshold=0.7)
    result = judge.evaluate(
        "answer",
        prompt="prompt",
        reference="reference",
        criteria="criteria",
    )

    assert result.score == 0.9
    assert result.passed is True
    assert result.rationale == "Looks good."
    assert result.metadata["model"] == "gpt-4o-mini"
    assert result.metadata["base_url"] == "https://example.com"
    assert result.metadata["raw_response"] == completions.raw_response
    assert result.metadata["threshold"] == 0.7
    assert result.metadata["criteria"] == "criteria"
    assert completions.seen_kwargs["model"] == "gpt-4o-mini"
    assert completions.seen_kwargs["response_format"] == {"type": "json_object"}


def test_openai_judge_valid_json_below_threshold_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_client(
        monkeypatch,
        FakeCompletions('{"score": 0.4, "passed": true, "rationale": "Weak."}'),
    )

    result = OpenAIJudge(threshold=0.7).evaluate("answer")

    assert result.score == 0.4
    assert result.passed is False
    assert result.rationale == "Weak."


def test_openai_judge_invalid_json_returns_graceful_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_client(monkeypatch, FakeCompletions("not json"))

    result = OpenAIJudge().evaluate("answer")

    assert result.score == 0.0
    assert result.passed is False
    assert "Judge failed gracefully" in result.rationale
    assert result.metadata["raw_response"] == "not json"
    assert "error" in result.metadata


def test_openai_judge_missing_score_returns_graceful_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_client(
        monkeypatch,
        FakeCompletions('{"passed": true, "rationale": "Missing score."}'),
    )

    result = OpenAIJudge().evaluate("answer")

    assert result.score == 0.0
    assert result.passed is False
    assert "missing keys: score" in result.metadata["error"]


def test_openai_judge_score_outside_range_is_clamped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_client(
        monkeypatch,
        FakeCompletions('{"score": 1.7, "passed": true, "rationale": "Great."}'),
    )

    result = OpenAIJudge(threshold=1.0).evaluate("answer")

    assert result.score == 1.0
    assert result.passed is True


def test_openai_judge_api_error_returns_graceful_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_client(monkeypatch, FakeCompletions(error=RuntimeError("api failed")))

    result = OpenAIJudge().evaluate("answer", criteria="accuracy")

    assert result.score == 0.0
    assert result.passed is False
    assert "Judge failed gracefully" in result.rationale
    assert result.metadata["raw_response"] is None
    assert result.metadata["criteria"] == "accuracy"
    assert "api failed" in result.metadata["error"]


def test_openai_judge_missing_dependency_raises_import_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_import_module = importlib.import_module

    def fake_import_module(name: str, package: str | None = None) -> Any:
        if name == "openai":
            raise ImportError("missing")
        return real_import_module(name, package)

    monkeypatch.setattr(importlib, "import_module", fake_import_module)

    with pytest.raises(ImportError, match="openai"):
        OpenAIJudge()

    with pytest.raises(ImportError, match="llmcalibre\\[judge\\]"):
        OpenAIJudge()


def test_openai_judge_callable_behavior(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_client(
        monkeypatch,
        FakeCompletions('{"score": 0.8, "passed": true, "rationale": "Good."}'),
    )

    result = OpenAIJudge()("answer")

    assert result.score == 0.8
    assert result.passed is True


def test_openai_judge_import_error_message_constant() -> None:
    assert MISSING_OPENAI_MESSAGE == (
        "OpenAIJudge requires openai. Install with: pip install llmcalibre[judge]"
    )


def test_openai_judge_exports_work() -> None:
    assert OpenAIJudge is MetricsOpenAIJudge
    assert OpenAIJudge is JudgeOpenAIJudge

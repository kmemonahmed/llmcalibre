"""OpenAI-compatible LLM judge evaluator."""

from __future__ import annotations

import importlib
import json
from typing import Any

from llmcalibre.core.base import BaseEvaluator
from llmcalibre.core.result import EvalResult

MISSING_OPENAI_MESSAGE = (
    "OpenAIJudge requires openai. Install with: pip install llmcalibre[judge]"
)

DEFAULT_SYSTEM_PROMPT = (
    "You are a strict evaluator. Return only valid JSON with keys: "
    "score, passed, rationale."
)


class OpenAIJudge(BaseEvaluator):
    """Evaluate outputs with an OpenAI-compatible judge model."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "gpt-4o-mini",
        base_url: str | None = None,
        threshold: float = 0.7,
        system_prompt: str | None = None,
    ) -> None:
        """Create an OpenAI-compatible judge evaluator."""
        super().__init__(threshold=threshold)
        self.model = model
        self.base_url = base_url
        self.system_prompt = system_prompt or DEFAULT_SYSTEM_PROMPT
        client_cls = self._load_client_class()
        client_kwargs = {}
        if api_key is not None:
            client_kwargs["api_key"] = api_key
        if base_url is not None:
            client_kwargs["base_url"] = base_url
        self._client = client_cls(**client_kwargs)

    def evaluate(
        self,
        output: str,
        prompt: str | None = None,
        reference: str | None = None,
        criteria: str | None = None,
        **kwargs: Any,
    ) -> EvalResult:
        """Ask the judge model to evaluate output and parse its JSON response."""
        del kwargs
        raw_response = None
        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {
                        "role": "user",
                        "content": self._build_user_prompt(
                            output=output,
                            prompt=prompt,
                            reference=reference,
                            criteria=criteria,
                        ),
                    },
                ],
                response_format={"type": "json_object"},
            )
            raw_response = self._extract_content(response)
            score, rationale = self._parse_judge_response(raw_response)
        except Exception as error:  # noqa: BLE001 - evaluator must fail gracefully.
            return self._failure_result(
                error=str(error),
                raw_response=raw_response,
                criteria=criteria,
            )

        passed = score >= self.threshold
        return EvalResult(
            score=score,
            passed=passed,
            rationale=rationale,
            metadata={
                "model": self.model,
                "base_url": self.base_url,
                "raw_response": raw_response,
                "threshold": self.threshold,
                "criteria": criteria,
            },
        )

    @staticmethod
    def _load_client_class() -> Any:
        """Load the OpenAI SDK lazily."""
        try:
            openai = importlib.import_module("openai")
        except ImportError as error:
            raise ImportError(MISSING_OPENAI_MESSAGE) from error

        return openai.OpenAI

    @staticmethod
    def _build_user_prompt(
        output: str,
        prompt: str | None,
        reference: str | None,
        criteria: str | None,
    ) -> str:
        """Build the judge prompt."""
        sections = [
            "Evaluate the LLM output.",
            'Return strict JSON: {"score": 0.0, "passed": true, "rationale": "..."}',
            f"Output:\n{output}",
        ]
        if prompt is not None:
            sections.append(f"Original prompt:\n{prompt}")
        if reference is not None:
            sections.append(f"Reference:\n{reference}")
        if criteria is not None:
            sections.append(f"Criteria:\n{criteria}")
        return "\n\n".join(sections)

    @staticmethod
    def _extract_content(response: Any) -> str:
        """Extract message content from an OpenAI chat completion response."""
        content = response.choices[0].message.content
        if not isinstance(content, str):
            raise ValueError("Judge response content is not text")
        return content

    @staticmethod
    def _parse_judge_response(raw_response: str) -> tuple[float, str]:
        """Parse and validate the judge JSON response."""
        parsed = json.loads(raw_response)
        if not isinstance(parsed, dict):
            raise ValueError("Judge response must be a JSON object")

        missing_keys = {"score", "passed", "rationale"} - set(parsed)
        if missing_keys:
            missing = ", ".join(sorted(missing_keys))
            raise ValueError(f"Judge response missing keys: {missing}")

        raw_score = parsed["score"]
        if isinstance(raw_score, bool) or not isinstance(raw_score, (int, float)):
            raise ValueError("Judge response score must be numeric")

        raw_rationale = parsed["rationale"]
        if not isinstance(raw_rationale, str):
            raise ValueError("Judge response rationale must be text")

        return max(0.0, min(1.0, float(raw_score))), raw_rationale

    def _failure_result(
        self,
        error: str,
        raw_response: str | None,
        criteria: str | None,
    ) -> EvalResult:
        """Return a graceful failure result for judge errors."""
        return EvalResult(
            score=0.0,
            passed=False,
            rationale=f"Judge failed gracefully: {error}",
            metadata={
                "model": self.model,
                "base_url": self.base_url,
                "raw_response": raw_response,
                "threshold": self.threshold,
                "criteria": criteria,
                "error": error,
            },
        )

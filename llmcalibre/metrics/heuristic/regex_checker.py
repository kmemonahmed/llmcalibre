"""Heuristic regex pattern evaluator."""

from __future__ import annotations

import re
from re import Pattern
from typing import Any

from llmcalibre.core.base import BaseEvaluator
from llmcalibre.core.result import EvalResult


class RegexChecker(BaseEvaluator):
    """Check required and forbidden regex patterns in an output."""

    def __init__(
        self,
        required_patterns: list[str] | None = None,
        forbidden_patterns: list[str] | None = None,
        flags: int = 0,
        threshold: float = 1.0,
    ) -> None:
        """Create a checker and compile regex patterns immediately."""
        super().__init__(threshold=threshold)
        self.required_patterns = list(required_patterns or [])
        self.forbidden_patterns = list(forbidden_patterns or [])
        self.flags = flags

        if not self.required_patterns and not self.forbidden_patterns:
            raise ValueError("required_patterns or forbidden_patterns must be provided")

        self._required_regexes = self._compile_patterns(
            self.required_patterns,
            label="required_patterns",
        )
        self._forbidden_regexes = self._compile_patterns(
            self.forbidden_patterns,
            label="forbidden_patterns",
        )

    def evaluate(self, output: str, **kwargs: Any) -> EvalResult:
        """Score output by required regex matches and forbidden regex misses."""
        del kwargs
        matched_required_patterns = [
            pattern
            for pattern, regex in zip(
                self.required_patterns,
                self._required_regexes,
            )
            if regex.search(output) is not None
        ]
        missing_required_patterns = [
            pattern
            for pattern, regex in zip(
                self.required_patterns,
                self._required_regexes,
            )
            if regex.search(output) is None
        ]
        found_forbidden_patterns = [
            pattern
            for pattern, regex in zip(
                self.forbidden_patterns,
                self._forbidden_regexes,
            )
            if regex.search(output) is not None
        ]
        allowed_forbidden_patterns = [
            pattern
            for pattern, regex in zip(
                self.forbidden_patterns,
                self._forbidden_regexes,
            )
            if regex.search(output) is None
        ]

        scores = []
        if self.required_patterns:
            scores.append(len(matched_required_patterns) / len(self.required_patterns))
        if self.forbidden_patterns:
            scores.append(
                len(allowed_forbidden_patterns) / len(self.forbidden_patterns)
            )

        final_score = sum(scores) / len(scores)
        passed = final_score >= self.threshold

        return EvalResult(
            score=final_score,
            passed=passed,
            rationale=self._build_rationale(
                matched_required_patterns=matched_required_patterns,
                missing_required_patterns=missing_required_patterns,
                found_forbidden_patterns=found_forbidden_patterns,
                passed=passed,
            ),
            metadata={
                "matched_required_patterns": matched_required_patterns,
                "missing_required_patterns": missing_required_patterns,
                "found_forbidden_patterns": found_forbidden_patterns,
                "allowed_forbidden_patterns": allowed_forbidden_patterns,
                "flags": self.flags,
            },
        )

    def _compile_patterns(
        self,
        patterns: list[str],
        label: str,
    ) -> list[Pattern[str]]:
        """Compile regex patterns and raise a clear ValueError on failure."""
        compiled_patterns = []
        for pattern in patterns:
            try:
                compiled_patterns.append(re.compile(pattern, flags=self.flags))
            except re.error as error:
                raise ValueError(
                    f"Invalid regex in {label}: {pattern!r} ({error})"
                ) from error
        return compiled_patterns

    @staticmethod
    def _build_rationale(
        matched_required_patterns: list[str],
        missing_required_patterns: list[str],
        found_forbidden_patterns: list[str],
        passed: bool,
    ) -> str:
        """Build a concise human-readable rationale."""
        parts = []
        if matched_required_patterns:
            parts.append(
                "Matched required patterns: "
                f"{', '.join(matched_required_patterns)}."
            )
        if missing_required_patterns:
            parts.append(
                "Missing required patterns: "
                f"{', '.join(missing_required_patterns)}."
            )
        if found_forbidden_patterns:
            parts.append(
                "Found forbidden patterns: "
                f"{', '.join(found_forbidden_patterns)}."
            )
        if not parts:
            parts.append("All forbidden patterns were avoided.")

        status = "passed" if passed else "failed"
        return f"Regex check {status}. " + " ".join(parts)

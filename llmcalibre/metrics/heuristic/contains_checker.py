"""Heuristic substring containment evaluator."""

from __future__ import annotations

from typing import Any

from llmcalibre.core.base import BaseEvaluator
from llmcalibre.core.result import EvalResult


class ContainsChecker(BaseEvaluator):
    """Check required and forbidden terms in an output."""

    def __init__(
        self,
        required: list[str] | None = None,
        forbidden: list[str] | None = None,
        case_sensitive: bool = False,
        threshold: float = 1.0,
    ) -> None:
        """Create a checker for required and forbidden terms."""
        super().__init__(threshold=threshold)
        self.required = list(required or [])
        self.forbidden = list(forbidden or [])
        self.case_sensitive = case_sensitive

        if not self.required and not self.forbidden:
            raise ValueError("required or forbidden terms must be provided")

    def evaluate(self, output: str, **kwargs: Any) -> EvalResult:
        """Score output by required terms present and forbidden terms absent."""
        del kwargs
        comparable_output = self._normalize(output)

        matched_required = [
            term
            for term in self.required
            if self._normalize(term) in comparable_output
        ]
        missing_required = [
            term
            for term in self.required
            if self._normalize(term) not in comparable_output
        ]
        found_forbidden = [
            term
            for term in self.forbidden
            if self._normalize(term) in comparable_output
        ]
        allowed_forbidden = [
            term
            for term in self.forbidden
            if self._normalize(term) not in comparable_output
        ]

        scores = []
        if self.required:
            scores.append(len(matched_required) / len(self.required))
        if self.forbidden:
            scores.append(len(allowed_forbidden) / len(self.forbidden))

        final_score = sum(scores) / len(scores)
        passed = final_score >= self.threshold

        return EvalResult(
            score=final_score,
            passed=passed,
            rationale=self._build_rationale(
                matched_required=matched_required,
                missing_required=missing_required,
                found_forbidden=found_forbidden,
                passed=passed,
            ),
            metadata={
                "matched_required": matched_required,
                "missing_required": missing_required,
                "found_forbidden": found_forbidden,
                "allowed_forbidden": allowed_forbidden,
                "case_sensitive": self.case_sensitive,
            },
        )

    def _normalize(self, value: str) -> str:
        """Normalize text according to case sensitivity."""
        if self.case_sensitive:
            return value
        return value.casefold()

    @staticmethod
    def _build_rationale(
        matched_required: list[str],
        missing_required: list[str],
        found_forbidden: list[str],
        passed: bool,
    ) -> str:
        """Build a concise human-readable rationale."""
        parts = []
        if matched_required:
            parts.append(f"Matched required terms: {', '.join(matched_required)}.")
        if missing_required:
            parts.append(f"Missing required terms: {', '.join(missing_required)}.")
        if found_forbidden:
            parts.append(f"Found forbidden terms: {', '.join(found_forbidden)}.")
        if not parts:
            parts.append("All forbidden terms were avoided.")

        status = "passed" if passed else "failed"
        return f"Contains check {status}. " + " ".join(parts)

"""Result types returned by evaluators."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class EvalResult:
    """A normalized result from a single evaluator."""

    score: float
    passed: bool
    rationale: str
    metadata: dict[str, Any] = field(default_factory=dict)

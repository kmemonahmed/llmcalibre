"""Heuristic JSON Schema validation evaluator."""

from __future__ import annotations

import importlib
import json
from typing import Any

from llmcalibre.core.base import BaseEvaluator
from llmcalibre.core.result import EvalResult

MISSING_JSONSCHEMA_MESSAGE = (
    "JsonSchemaChecker requires jsonschema. "
    "Install with: pip install llmcalibre[schema]"
)


class JsonSchemaChecker(BaseEvaluator):
    """Validate JSON output against a JSON Schema."""

    def __init__(
        self,
        schema: dict[str, Any],
        threshold: float = 1.0,
    ) -> None:
        """Create a JSON Schema checker and validate the schema."""
        super().__init__(threshold=threshold)
        self.schema = schema
        validator_for, schema_error = self._load_jsonschema()
        validator_cls = validator_for(schema)

        try:
            validator_cls.check_schema(schema)
        except schema_error as error:
            raise ValueError(f"Invalid JSON Schema: {error}") from error

        self._validator = validator_cls(schema)

    def evaluate(self, output: str, **kwargs: Any) -> EvalResult:
        """Return whether output is valid JSON and conforms to the schema."""
        del kwargs
        try:
            parsed = json.loads(output)
        except (TypeError, json.JSONDecodeError) as error:
            return EvalResult(
                score=0.0,
                passed=False,
                rationale="Output is invalid JSON.",
                metadata={
                    "parsed": None,
                    "schema_valid": False,
                    "validation_error": None,
                    "path": [],
                    "error": str(error),
                },
            )

        validation_error = next(self._validator.iter_errors(parsed), None)
        if validation_error is not None:
            return EvalResult(
                score=0.0,
                passed=False,
                rationale="JSON schema validation failed.",
                metadata={
                    "parsed": parsed,
                    "schema_valid": False,
                    "validation_error": validation_error.message,
                    "path": list(validation_error.path),
                },
            )

        return EvalResult(
            score=1.0,
            passed=self.threshold <= 1.0,
            rationale="JSON conforms to schema.",
            metadata={
                "parsed": parsed,
                "schema_valid": True,
                "validation_error": None,
                "path": [],
            },
        )

    @staticmethod
    def _load_jsonschema() -> tuple[Any, type[Exception]]:
        """Load jsonschema lazily so the base package has no runtime dependency."""
        try:
            jsonschema = importlib.import_module("jsonschema")
            validators = importlib.import_module("jsonschema.validators")
        except ImportError as error:
            raise ImportError(MISSING_JSONSCHEMA_MESSAGE) from error

        return validators.validator_for, jsonschema.SchemaError

import pytest

from llmcalibre import EvalResult, JsonSchemaChecker
from llmcalibre.metrics import JsonSchemaChecker as MetricsJsonSchemaChecker
from llmcalibre.metrics.heuristic import JsonSchemaChecker as HeuristicJsonSchemaChecker


def test_json_schema_checker_passes_valid_json_matching_schema() -> None:
    schema = {
        "type": "object",
        "required": ["answer"],
        "properties": {"answer": {"type": "string"}},
    }

    result = JsonSchemaChecker(schema=schema).evaluate('{"answer": "hello"}')

    assert result == EvalResult(
        score=1.0,
        passed=True,
        rationale="JSON conforms to schema.",
        metadata={
            "parsed": {"answer": "hello"},
            "schema_valid": True,
            "validation_error": None,
            "path": [],
        },
    )


def test_json_schema_checker_fails_invalid_json_gracefully() -> None:
    checker = JsonSchemaChecker(schema={"type": "object"})

    result = checker.evaluate("{bad json")

    assert result.score == 0.0
    assert result.passed is False
    assert result.rationale == "Output is invalid JSON."
    assert result.metadata["parsed"] is None
    assert result.metadata["schema_valid"] is False
    assert result.metadata["validation_error"] is None
    assert result.metadata["path"] == []
    assert "error" in result.metadata


def test_json_schema_checker_fails_schema_mismatch_gracefully() -> None:
    schema = {
        "type": "object",
        "properties": {"age": {"type": "integer"}},
    }

    result = JsonSchemaChecker(schema=schema).evaluate('{"age": "old"}')

    assert result.score == 0.0
    assert result.passed is False
    assert result.rationale == "JSON schema validation failed."
    assert result.metadata["parsed"] == {"age": "old"}
    assert result.metadata["schema_valid"] is False
    assert "not of type" in result.metadata["validation_error"]
    assert result.metadata["path"] == ["age"]


def test_json_schema_checker_raises_value_error_for_invalid_schema() -> None:
    with pytest.raises(ValueError, match="Invalid JSON Schema"):
        JsonSchemaChecker(schema={"type": "not-a-real-json-schema-type"})


def test_json_schema_checker_callable_behavior() -> None:
    checker = JsonSchemaChecker(schema={"type": "object"})

    assert checker("{}").score == 1.0


def test_json_schema_checker_import_exports_work() -> None:
    assert JsonSchemaChecker is MetricsJsonSchemaChecker
    assert JsonSchemaChecker is HeuristicJsonSchemaChecker

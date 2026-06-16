# llmcalibre

llmcalibre is an offline-first, framework-agnostic foundation for evaluating LLM outputs. It starts with small, deterministic evaluators that can run anywhere without runtime dependencies.

## Install

```bash
pip install llmcalibre
```

For local development:

```bash
pip install -e ".[dev]"
```

For JSON Schema validation support:

```bash
pip install "llmcalibre[schema]"
```

For offline semantic similarity support:

```bash
pip install "llmcalibre[nlp]"
```

## Quick Example

```python
from llmcalibre import (
    ContainsChecker,
    EvalPipeline,
    FormatChecker,
    JsonSchemaChecker,
    LengthConstraint,
    RegexChecker,
    SemanticSimilarity,
)

schema = {
    "type": "object",
    "required": ["answer"],
    "properties": {
        "answer": {"type": "string"},
    },
}

pipeline = EvalPipeline(
    [
        FormatChecker(format="json"),
        LengthConstraint(min_chars=2, max_chars=100),
        ContainsChecker(required=["answer"], forbidden=["markdown"]),
        RegexChecker(required_patterns=[r'"answer"\s*:'], forbidden_patterns=[r"TODO"]),
        JsonSchemaChecker(schema=schema),
    ]
)

results = pipeline.run('{"answer": "hello"}')
summary = pipeline.summary(results)

print(results)
print(summary)

similarity = SemanticSimilarity(threshold=0.7)
semantic_result = similarity.evaluate(
    "Paris is the capital of France.",
    reference="France's capital city is Paris.",
)
print(semantic_result)
```

## CLI Usage

```bash
llmcalibre check --output '{"name":"Emon"}' --format json
llmcalibre check --output "Paris is in France" --contains Paris --contains France
llmcalibre check --output-file response.txt --min-chars 50 --max-chars 500
llmcalibre check --output "Date: 2026-06-16" --regex "\\d{4}-\\d{2}-\\d{2}"
```

## Pytest Helper

```python
from llmcalibre import ContainsChecker, FormatChecker
from llmcalibre.pytest import assert_eval


def test_llm_response():
    output = '{"city": "Paris", "country": "France"}'

    assert_eval(
        output,
        evaluators=[
            FormatChecker(format="json"),
            ContainsChecker(required=["Paris", "France"]),
        ],
    )
```

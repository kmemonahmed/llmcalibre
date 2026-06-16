# llmcalibre

llmcalibre is an offline-first, framework-agnostic library for evaluating LLM
outputs. It starts with deterministic local checks, then lets you opt into
heavier NLP metrics or OpenAI-compatible judge models only when you need them.

This is an alpha release: APIs are intentionally small, but may still evolve.

## Installation

Base install, with no runtime dependencies:

```bash
pip install llmcalibre
```

Install from a local checkout for development:

```bash
pip install -e ".[dev]"
```

Optional extras:

```bash
pip install "llmcalibre[schema]"
pip install "llmcalibre[nlp]"
pip install "llmcalibre[judge]"
```

Extras can be combined:

```bash
pip install "llmcalibre[schema,nlp,judge]"
```

## Features

- Normalized `EvalResult` objects with score, pass/fail status, rationale, and metadata.
- `EvalPipeline` for running multiple evaluators against one output.
- Heuristic checks for JSON format, length, required/forbidden terms, regex patterns, and JSON Schema.
- Optional offline NLP metrics for semantic similarity and ROUGE.
- Optional OpenAI-compatible LLM judge evaluator with graceful failure handling.
- Stdlib CLI for simple terminal checks.
- Lightweight pytest assertion helper.
- No base runtime dependencies.

## Python API

```python
from llmcalibre import (
    ContainsChecker,
    EvalPipeline,
    FormatChecker,
    LengthConstraint,
    RegexChecker,
)

pipeline = EvalPipeline(
    [
        FormatChecker(format="json"),
        LengthConstraint(min_chars=2, max_chars=100),
        ContainsChecker(required=["answer"], forbidden=["TODO"]),
        RegexChecker(required_patterns=[r'"answer"\s*:']),
    ]
)

results = pipeline.run('{"answer": "Paris"}')
summary = pipeline.summary(results)

print(results)
print(summary)
```

## Optional Evaluators

JSON Schema validation:

```python
from llmcalibre import JsonSchemaChecker

schema = {
    "type": "object",
    "required": ["answer"],
    "properties": {"answer": {"type": "string"}},
}

result = JsonSchemaChecker(schema=schema).evaluate('{"answer": "Paris"}')
print(result)
```

Offline NLP metrics:

```python
from llmcalibre import RougeScore, SemanticSimilarity

similarity = SemanticSimilarity(threshold=0.7)
semantic_result = similarity.evaluate(
    "Paris is the capital of France.",
    reference="France's capital city is Paris.",
)

rouge = RougeScore(rouge_type="rougeL", threshold=0.5)
rouge_result = rouge.evaluate(
    "Paris is the capital of France.",
    reference="The capital of France is Paris.",
)
```

OpenAI-compatible judge:

```python
from llmcalibre import OpenAIJudge

judge = OpenAIJudge(model="gpt-4o-mini")
result = judge.evaluate(
    "Paris is the capital of France.",
    prompt="What is the capital of France?",
    reference="Paris",
    criteria="Reward factual correctness and concise answers.",
)
print(result)
```

## CLI Usage

```bash
llmcalibre check --output '{"name":"Emon"}' --format json
llmcalibre check --output "Paris is in France" --contains Paris --contains France
llmcalibre check --output-file response.txt --min-chars 50 --max-chars 500
llmcalibre check --output "Date: 2026-06-16" --regex "\\d{4}-\\d{2}-\\d{2}"
```

The CLI exits with:

- `0` when all checks pass.
- `1` when at least one check fails.
- `2` for usage or configuration errors.

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

## License

MIT

## Release Process

1. Update the version in `pyproject.toml`.
2. Update `CHANGELOG.md`.
3. Merge the release changes to `main`.
4. Create a GitHub Release with a tag like `v0.1.0-alpha.1`.
5. Publishing to PyPI runs automatically from the release workflow.

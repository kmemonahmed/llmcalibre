# Why LLMCalibre?

LLMCalibre is a lightweight evaluation toolkit for teams that need confidence in LLM output quality without adopting a full evaluation platform. It is designed to fit into normal Python test suites, CI pipelines, local development workflows, and model migration checks.

## The Problem

LLM-powered features often start as prompt experiments and quickly become production dependencies. A support assistant needs to stay within policy. A data extraction workflow needs to keep returning valid JSON. A RAG answer needs to cite the retrieved context rather than inventing details. A model upgrade that looks better in a demo can still break a downstream parser or silently change answer style.

Manual review helps early on, but it does not scale well. It is slow, inconsistent, hard to automate, and easy to skip when a prompt change looks small. Teams need a way to turn expected behavior into repeatable checks.

## Why LLM Evaluation Matters

LLM evaluation gives teams a feedback loop for changes that are otherwise difficult to reason about. It helps answer practical questions:

- Did a prompt edit preserve the required output format?
- Did a new model keep the same behavior on important examples?
- Does generated JSON still match the schema consumed by the application?
- Does a RAG answer contain required facts from the retrieved context?
- Does a judge model agree that the response is complete, grounded, or safe enough?

These checks do not replace human review. They make human review more targeted by catching obvious regressions early and consistently.

## Introducing LLMCalibre

LLMCalibre provides small, composable evaluators that return normalized `EvalResult` objects:

```python
from llmcalibre import ContainsChecker, EvalPipeline, FormatChecker, LengthConstraint

pipeline = EvalPipeline(
    [
        FormatChecker(format="json"),
        LengthConstraint(min_chars=10, max_chars=500),
        ContainsChecker(required=["answer"], forbidden=["TODO"]),
    ]
)

results = pipeline.run('{"answer": "Paris is the capital of France."}')
summary = pipeline.summary(results)

assert summary.total_passed == summary.total_count
```

The library is framework agnostic. It does not require LangChain, LlamaIndex, a hosted service, or a specific model provider. You can evaluate strings produced by any system.

## Core Use Cases

### Prompt Regression Testing

Prompt changes can introduce regressions even when the diff is small. LLMCalibre lets you encode expected output properties as tests.

```python
from llmcalibre import ContainsChecker, FormatChecker
from llmcalibre.pytest import assert_eval


def test_city_extraction_prompt():
    output = '{"city": "Paris", "country": "France"}'

    assert_eval(
        output,
        evaluators=[
            FormatChecker(format="json"),
            ContainsChecker(required=["Paris", "France"]),
        ],
    )
```

In a real project, `output` might come from a fixture, a recorded model response, or a test call to your application layer. The important part is that expected behavior becomes executable.

### Model Migration Testing

Teams often switch models to reduce latency, lower cost, improve quality, or move to a self-hosted model. A migration can preserve average quality while breaking critical edge cases.

With LLMCalibre, you can run the same dataset against outputs from both models and compare evaluator summaries.

```python
from llmcalibre import ContainsChecker, EvalPipeline, RegexChecker

checks = EvalPipeline(
    [
        ContainsChecker(required=["refund", "policy"]),
        RegexChecker(forbidden_patterns=[r"(?i)internal only|do not share"]),
    ]
)

old_model_results = checks.run("Refunds follow the published policy.")
new_model_results = checks.run("Refund policy details are internal only.")

print(checks.summary(old_model_results))
print(checks.summary(new_model_results))
```

This does not decide every migration question. It gives teams a repeatable signal before they move traffic.

### Structured Output Validation

Many production LLM features depend on structured outputs. If the model returns invalid JSON, changes a field name, or emits prose around the object, downstream code can fail.

LLMCalibre supports basic JSON format checks and optional JSON Schema validation.

```python
from llmcalibre import EvalPipeline, FormatChecker, JsonSchemaChecker

schema = {
    "type": "object",
    "required": ["name", "email"],
    "properties": {
        "name": {"type": "string"},
        "email": {"type": "string"},
    },
}

pipeline = EvalPipeline(
    [
        FormatChecker(format="json"),
        JsonSchemaChecker(schema=schema),
    ]
)

results = pipeline.run('{"name": "Ava", "email": "ava@example.com"}')
```

`JsonSchemaChecker` is optional and only requires `jsonschema` when you install `llmcalibre[schema]`.

### RAG Evaluation

RAG systems need answers that are relevant to the prompt and grounded in retrieved content. Lightweight checks can catch missing facts, format drift, and obvious hallucinations before a response reaches users.

```python
from llmcalibre import ContainsChecker, EvalPipeline, RougeScore

answer = "The SLA is 99.9% uptime with support during business hours."
reference = "The service offers 99.9% uptime and business-hours support."

pipeline = EvalPipeline(
    [
        ContainsChecker(required=["99.9%", "business hours"]),
        RougeScore(rouge_type="rougeL", threshold=0.4),
    ]
)

results = [
    pipeline.evaluators[0].evaluate(answer),
    pipeline.evaluators[1].evaluate(answer, reference=reference),
]
summary = pipeline.summary(results)
```

For semantic checks, `SemanticSimilarity` can compare output and reference text using sentence-transformers embeddings when installed with `llmcalibre[nlp]`.

### LLM-as-a-Judge

Some criteria are difficult to capture with deterministic rules. For example, “answer the question completely but concisely” or “explain the tradeoff without giving legal advice” may benefit from a judge model.

```python
from llmcalibre import OpenAIJudge

judge = OpenAIJudge(model="gpt-4o-mini", threshold=0.8)

result = judge.evaluate(
    "A fixed-rate mortgage keeps the same interest rate for the full term.",
    prompt="Explain a fixed-rate mortgage in one sentence.",
    criteria="The response should be accurate, concise, and understandable.",
)

assert result.passed
```

`OpenAIJudge` is designed to fail gracefully. If the API errors, the judge returns invalid JSON, or the response is malformed, evaluation returns a failed `EvalResult` with error metadata rather than crashing the test process.

## Why LLMCalibre?

### Offline First

The core package has no runtime dependencies. Basic checks such as JSON validity, length constraints, required terms, forbidden terms, and regex patterns run locally and deterministically.

Optional integrations are explicit:

- `llmcalibre[schema]` for JSON Schema validation.
- `llmcalibre[nlp]` for semantic similarity and ROUGE.
- `llmcalibre[judge]` for OpenAI-compatible judge models.

### Framework Agnostic

LLMCalibre evaluates text. It does not care whether the output came from a direct API call, an agent framework, a RAG pipeline, a batch job, or a local model. This keeps the evaluation layer portable.

### Composable Evaluators

Evaluators are small and can be combined in `EvalPipeline`. A structured extraction check might combine JSON validation, schema validation, and forbidden text checks. A RAG check might combine required facts, ROUGE, and a judge model.

```python
from llmcalibre import ContainsChecker, EvalPipeline, FormatChecker, RegexChecker

pipeline = EvalPipeline(
    [
        FormatChecker(format="json"),
        ContainsChecker(required=["answer"], forbidden=["markdown"]),
        RegexChecker(forbidden_patterns=[r"(?i)password|secret"]),
    ]
)

results = pipeline.run('{"answer": "Use the public reset flow."}')
print(pipeline.summary(results))
```

### Reliability Focused

LLM evaluation code should not become another source of instability. LLMCalibre favors clear failure modes:

- Bad JSON returns a failed result instead of raising unexpectedly.
- Optional dependency errors explain which extra to install.
- Judge model failures return failed `EvalResult` objects with error metadata.
- Scores, pass/fail state, rationales, and metadata are normalized.

## Current Features

- Core:
  - `EvalResult`
  - `BaseEvaluator`
  - `EvalPipeline`
- Heuristic evaluators:
  - `FormatChecker`
  - `LengthConstraint`
  - `ContainsChecker`
  - `RegexChecker`
  - `JsonSchemaChecker`
- Offline evaluators:
  - `SemanticSimilarity`
  - `RougeScore`
- Judge evaluators:
  - `OpenAIJudge`
- Developer tools:
  - `llmcalibre check` CLI
  - `assert_eval` pytest helper

## Example CI/CD Workflow

A practical CI workflow might run three layers of checks:

1. Unit tests for prompt templates and parsing behavior.
2. LLMCalibre checks against recorded or generated outputs.
3. Optional judge or NLP checks for representative examples.

```yaml
name: evals

on:
  pull_request:

jobs:
  llm-evals:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
      - run: python -m pip install -e ".[dev,schema]"
      - run: python -m pytest
      - run: llmcalibre check --output '{"answer": "Paris"}' --format json --contains Paris
```

This gives teams a low-friction way to block regressions before merging prompt, model, or retrieval changes.

## Comparison With Manual Testing

Manual testing is useful for exploration and qualitative judgment, but it is not enough for regression prevention.

| Manual testing | LLMCalibre checks |
| --- | --- |
| Ad hoc and reviewer-dependent | Repeatable and automated |
| Hard to run on every pull request | Fits into pytest and CI |
| Easy to miss format regressions | Validates JSON, schema, regex, and terms |
| Difficult to compare model migrations | Runs the same checks across outputs |
| Human time scales poorly | Lightweight checks run quickly |

LLMCalibre is not a replacement for human review. It is a way to make routine quality checks consistent, visible, and cheap.

## Vision and Roadmap

LLMCalibre aims to remain a focused toolkit rather than a hosted platform. The direction is:

- More deterministic evaluators for common production failure modes.
- Better dataset and batch evaluation ergonomics.
- Richer summaries for CI and pull request reporting.
- Additional offline metrics that can run without sending data to external services.
- Careful judge integrations with graceful failure behavior and clear metadata.
- Stable APIs that make it easy to embed evaluation into existing development workflows.

The goal is simple: help developers treat LLM behavior as something they can test, compare, and improve with confidence.

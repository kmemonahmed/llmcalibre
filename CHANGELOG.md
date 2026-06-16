# Changelog

## v0.1.0-alpha.1

Initial alpha release of llmcalibre.

### Features

- Core evaluator contract with normalized `EvalResult` objects.
- `EvalPipeline` for running multiple evaluators and summarizing results.
- Heuristic evaluators:
  - `FormatChecker` for JSON validity.
  - `LengthConstraint` for character bounds.
  - `ContainsChecker` for required and forbidden terms.
  - `RegexChecker` for required and forbidden regex patterns.
  - `JsonSchemaChecker` for optional JSON Schema validation.
- Offline optional NLP evaluators:
  - `SemanticSimilarity` using sentence-transformers embeddings.
  - `RougeScore` using ROUGE overlap scores.
- OpenAI-compatible optional judge evaluator:
  - `OpenAIJudge` with graceful fallback for API or parsing failures.
- Stdlib command-line interface:
  - `llmcalibre check` for basic terminal checks.
- Pytest helper:
  - `assert_eval` for concise test assertions.
- Optional extras:
  - `schema` for JSON Schema support.
  - `nlp` for offline NLP metrics.
  - `judge` for OpenAI-compatible model judging.

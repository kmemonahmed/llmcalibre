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

## Quick Example

```python
from llmcalibre import ContainsChecker, EvalPipeline, FormatChecker, LengthConstraint

pipeline = EvalPipeline(
    [
        FormatChecker(format="json"),
        LengthConstraint(min_chars=2, max_chars=100),
        ContainsChecker(required=["answer"], forbidden=["markdown"]),
    ]
)

results = pipeline.run('{"answer": "hello"}')
summary = pipeline.summary(results)

print(results)
print(summary)
```

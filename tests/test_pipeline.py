from typing import Any, Optional

from llmcalibre import BaseEvaluator, EvalPipeline, EvalResult, PipelineSummary


class StaticEvaluator(BaseEvaluator):
    def __init__(self, result: EvalResult) -> None:
        super().__init__()
        self.result = result
        self.seen_output: Optional[str] = None

    def evaluate(self, output: str, **kwargs: Any) -> EvalResult:
        del kwargs
        self.seen_output = output
        return self.result


def test_pipeline_runs_all_evaluators_against_same_output() -> None:
    first = StaticEvaluator(EvalResult(score=1.0, passed=True, rationale="Passed."))
    second = StaticEvaluator(EvalResult(score=0.0, passed=False, rationale="Failed."))
    pipeline = EvalPipeline([first, second])

    results = pipeline.run("model output")

    assert results == [first.result, second.result]
    assert first.seen_output == "model output"
    assert second.seen_output == "model output"


def test_pipeline_call_delegates_to_run() -> None:
    evaluator = StaticEvaluator(EvalResult(score=1.0, passed=True, rationale="Passed."))
    pipeline = EvalPipeline([evaluator])

    assert pipeline("output") == [evaluator.result]


def test_pipeline_summary_uses_given_results() -> None:
    pipeline = EvalPipeline([])
    results = [
        EvalResult(score=1.0, passed=True, rationale="Passed."),
        EvalResult(score=0.25, passed=False, rationale="Failed."),
    ]

    assert pipeline.summary(results) == PipelineSummary(
        total_passed=1,
        total_count=2,
        average_score=0.625,
    )


def test_pipeline_summary_uses_last_results_by_default() -> None:
    evaluator = StaticEvaluator(EvalResult(score=0.5, passed=True, rationale="Passed."))
    pipeline = EvalPipeline([evaluator])

    pipeline.run("output")

    assert pipeline.summary() == PipelineSummary(
        total_passed=1,
        total_count=1,
        average_score=0.5,
    )


def test_pipeline_summary_handles_empty_results() -> None:
    assert EvalPipeline([]).summary() == PipelineSummary(
        total_passed=0,
        total_count=0,
        average_score=0.0,
    )

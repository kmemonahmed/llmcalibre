"""Command-line interface for llmcalibre."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import NoReturn, TextIO, cast

from llmcalibre.core.base import BaseEvaluator
from llmcalibre.core.pipeline import EvalPipeline
from llmcalibre.core.result import EvalResult
from llmcalibre.metrics.heuristic.contains_checker import ContainsChecker
from llmcalibre.metrics.heuristic.format_checker import FormatChecker
from llmcalibre.metrics.heuristic.length_constraint import LengthConstraint
from llmcalibre.metrics.heuristic.regex_checker import RegexChecker


class CliError(Exception):
    """Raised for usage and configuration errors."""


class LlmcalibreParser(argparse.ArgumentParser):
    """Argument parser that raises instead of exiting."""

    def error(self, message: str) -> NoReturn:
        """Raise a CLI error with argparse's formatted usage message."""
        raise CliError(message)


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser."""
    parser = LlmcalibreParser(prog="llmcalibre")
    subparsers = parser.add_subparsers(dest="command")

    check_parser = subparsers.add_parser("check", help="Run output checks")
    check_parser.add_argument("--output", help="LLM output text to evaluate")
    check_parser.add_argument("--output-file", help="Path to LLM output text")
    check_parser.add_argument("--format", choices=["json"], help="Expected format")
    check_parser.add_argument("--min-chars", type=int, help="Minimum character count")
    check_parser.add_argument("--max-chars", type=int, help="Maximum character count")
    check_parser.add_argument(
        "--contains",
        action="append",
        default=[],
        help="Required text; repeatable",
    )
    check_parser.add_argument(
        "--forbid",
        action="append",
        default=[],
        help="Forbidden text; repeatable",
    )
    check_parser.add_argument(
        "--regex",
        action="append",
        default=[],
        help="Required regex pattern; repeatable",
    )
    check_parser.add_argument(
        "--forbid-regex",
        action="append",
        default=[],
        help="Forbidden regex pattern; repeatable",
    )
    return parser


def read_output(args: argparse.Namespace) -> str:
    """Read output text from CLI arguments."""
    if args.output is not None and args.output_file is not None:
        raise CliError("Provide either --output or --output-file, not both.")
    if args.output is None and args.output_file is None:
        raise CliError("Provide either --output or --output-file.")
    if args.output is not None:
        return cast(str, args.output)

    try:
        return Path(cast(str, args.output_file)).read_text(encoding="utf-8")
    except OSError as error:
        raise CliError(f"Could not read --output-file: {error}") from error


def build_evaluators(args: argparse.Namespace) -> list[BaseEvaluator]:
    """Build evaluators from parsed CLI arguments."""
    evaluators: list[BaseEvaluator] = []

    if args.format is not None:
        evaluators.append(FormatChecker(format=args.format))
    if args.min_chars is not None or args.max_chars is not None:
        evaluators.append(
            LengthConstraint(min_chars=args.min_chars, max_chars=args.max_chars)
        )
    if args.contains or args.forbid:
        evaluators.append(
            ContainsChecker(required=args.contains, forbidden=args.forbid)
        )
    if args.regex or args.forbid_regex:
        try:
            evaluators.append(
                RegexChecker(
                    required_patterns=args.regex,
                    forbidden_patterns=args.forbid_regex,
                )
            )
        except ValueError as error:
            raise CliError(str(error)) from error

    if not evaluators:
        raise CliError("Provide at least one evaluator flag.")
    return evaluators


def format_report(
    evaluators: Sequence[BaseEvaluator],
    results: Sequence[EvalResult],
    summary_text: str,
) -> str:
    """Format a plain-text evaluation report."""
    lines = []
    for evaluator, result in zip(evaluators, results):
        status = "PASS" if result.passed else "FAIL"
        lines.extend(
            [
                evaluator.__class__.__name__,
                f"score: {result.score:.3f}",
                f"status: {status}",
                f"rationale: {result.rationale}",
                "",
            ]
        )
    lines.append(summary_text)
    return "\n".join(lines)


def run_check(args: argparse.Namespace, stdout: TextIO) -> int:
    """Run the check command and return a process exit code."""
    output = read_output(args)
    evaluators = build_evaluators(args)
    pipeline = EvalPipeline(evaluators)
    results = pipeline.run(output)
    summary = pipeline.summary(results)
    report = format_report(
        evaluators=evaluators,
        results=results,
        summary_text=(
            f"overall: {summary.total_passed}/{summary.total_count} passed, "
            f"average score: {summary.average_score:.3f}"
        ),
    )
    print(report, file=stdout)
    return 0 if summary.total_passed == summary.total_count else 1


def main(argv: Sequence[str] | None = None) -> int:
    """Run the llmcalibre command-line interface."""
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
        if args.command != "check":
            raise CliError("Expected command: check")
        return run_check(args, stdout=sys.stdout)
    except CliError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

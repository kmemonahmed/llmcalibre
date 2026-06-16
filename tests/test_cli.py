from pathlib import Path

from llmcalibre.cli import main


def test_cli_direct_output_works(capsys) -> None:  # type: ignore[no-untyped-def]
    exit_code = main(["check", "--output", '{"name":"Emon"}', "--format", "json"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "FormatChecker" in captured.out
    assert "status: PASS" in captured.out
    assert "overall: 1/1 passed" in captured.out


def test_cli_output_file_works(
    tmp_path: Path, capsys
) -> None:  # type: ignore[no-untyped-def]
    output_file = tmp_path / "response.txt"
    output_file.write_text("Paris is in France", encoding="utf-8")

    exit_code = main(
        [
            "check",
            "--output-file",
            str(output_file),
            "--contains",
            "Paris",
            "--contains",
            "France",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "ContainsChecker" in captured.out
    assert "overall: 1/1 passed" in captured.out


def test_cli_both_output_and_output_file_returns_code_2(
    tmp_path: Path, capsys
) -> None:  # type: ignore[no-untyped-def]
    output_file = tmp_path / "response.txt"
    output_file.write_text("hello", encoding="utf-8")

    exit_code = main(
        [
            "check",
            "--output",
            "hello",
            "--output-file",
            str(output_file),
            "--format",
            "json",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 2
    assert "not both" in captured.err


def test_cli_missing_output_returns_code_2(capsys) -> None:  # type: ignore[no-untyped-def]
    exit_code = main(["check", "--format", "json"])

    captured = capsys.readouterr()
    assert exit_code == 2
    assert "Provide either --output or --output-file" in captured.err


def test_cli_no_evaluator_returns_code_2(capsys) -> None:  # type: ignore[no-untyped-def]
    exit_code = main(["check", "--output", "hello"])

    captured = capsys.readouterr()
    assert exit_code == 2
    assert "at least one evaluator" in captured.err


def test_cli_passing_eval_returns_code_0(capsys) -> None:  # type: ignore[no-untyped-def]
    exit_code = main(["check", "--output", "Paris", "--contains", "Paris"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "status: PASS" in captured.out


def test_cli_failing_eval_returns_code_1(capsys) -> None:  # type: ignore[no-untyped-def]
    exit_code = main(["check", "--output", "Paris", "--contains", "France"])

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "status: FAIL" in captured.out
    assert "overall: 0/1 passed" in captured.out


def test_cli_repeated_contains_works(capsys) -> None:  # type: ignore[no-untyped-def]
    exit_code = main(
        [
            "check",
            "--output",
            "Paris is in France",
            "--contains",
            "Paris",
            "--contains",
            "France",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "score: 1.000" in captured.out


def test_cli_regex_works(capsys) -> None:  # type: ignore[no-untyped-def]
    exit_code = main(
        ["check", "--output", "Date: 2026-06-16", "--regex", r"\d{4}-\d{2}-\d{2}"]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "RegexChecker" in captured.out


def test_cli_invalid_regex_returns_code_2_gracefully(
    capsys,
) -> None:  # type: ignore[no-untyped-def]
    exit_code = main(["check", "--output", "hello", "--regex", "["])

    captured = capsys.readouterr()
    assert exit_code == 2
    assert "Invalid regex" in captured.err

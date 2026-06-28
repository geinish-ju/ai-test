from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from ai_testing.cli import main

Record = dict[str, Any]


def test_sample_export_and_preprocessing_smoke(
    tmp_path: Path,
    cli_args: Callable[..., list[str]],
    read_json_file: Callable[[Path], Any],
) -> None:
    raw_path = tmp_path / "sample_order_items.json"
    second_raw_path = tmp_path / "sample_order_items_second.json"
    processed_path = tmp_path / "processed_order_items.json"
    report_path = tmp_path / "preprocessing_report.json"

    assert main(cli_args("export-sample", "--output", str(raw_path))) == 0
    assert (
        main(
            cli_args(
                "preprocess-data",
                "--input",
                str(raw_path),
                "--output",
                str(processed_path),
                "--report-output",
                str(report_path),
                "--drop-exact-duplicates",
            )
        )
        == 0
    )
    assert main(cli_args("export-sample", "--output", str(second_raw_path))) == 0

    processed_records = read_json_file(processed_path)
    report = read_json_file(report_path)
    stage_history = read_json_file(tmp_path / "stage_history.json")
    stage_history_markdown = (tmp_path / "stage_history.md").read_text(encoding="utf-8")
    metric_history_markdown = (tmp_path / "stage_metric_history.md").read_text(encoding="utf-8")
    latest_sample_export = stage_history["latest_execution_by_stage"]["data_acquisition.sample"]
    assert len(processed_records) == 8
    assert report["input_record_count"] == 8
    assert report["output_record_count"] == 8
    assert latest_sample_export["previous_execution_id"] is not None
    assert "Stage Execution History" in stage_history_markdown
    assert "data_acquisition.sample" in stage_history_markdown
    assert "Stage Metric History" in metric_history_markdown
    assert "| Metric | Run 1 | Run 2 |" in metric_history_markdown
    assert "summary.record_count" in metric_history_markdown


def test_run_tracking_smoke(
    tmp_path: Path,
    cli_args: Callable[..., list[str]],
    read_json_file: Callable[[Path], Any],
) -> None:
    runs_dir = tmp_path / "runs"
    index_path = runs_dir / "run_index.json"

    assert (
        main(
            cli_args(
                "track-run",
                "--run-id",
                "smoke-run",
                "--runs-dir",
                str(runs_dir),
                "--index-output",
                str(index_path),
                "--no-compare-to-latest",
            )
        )
        == 0
    )

    run_report_path = runs_dir / "smoke-run" / "run_report.json"
    run_report_markdown_path = runs_dir / "smoke-run" / "run_report.md"
    run_report = read_json_file(run_report_path)
    run_index = read_json_file(index_path)
    stage_history = read_json_file(tmp_path / "stage_history.json")
    assert run_report["run_id"] == "smoke-run"
    assert run_report["report_type"] == "pipeline_run_report"
    assert run_report_markdown_path.exists()
    assert run_index["latest_run_id"] == "smoke-run"
    assert (
        stage_history["latest_execution_by_stage"]["run_tracking"]["stage_name"] == "run_tracking"
    )


def test_drift_and_markdown_report_smoke(
    tmp_path: Path,
    cli_args: Callable[..., list[str]],
    read_json_file: Callable[[Path], Any],
) -> None:
    runs_dir = tmp_path / "runs"
    index_path = runs_dir / "run_index.json"
    drift_path = tmp_path / "drift_report.json"
    markdown_path = tmp_path / "run_report.md"

    assert (
        main(
            cli_args(
                "track-run",
                "--run-id",
                "baseline-run",
                "--runs-dir",
                str(runs_dir),
                "--index-output",
                str(index_path),
                "--no-compare-to-latest",
            )
        )
        == 0
    )
    assert (
        main(
            cli_args(
                "track-run",
                "--run-id",
                "candidate-run",
                "--runs-dir",
                str(runs_dir),
                "--index-output",
                str(index_path),
            )
        )
        == 0
    )
    assert (
        main(
            cli_args(
                "test-drift",
                "--run-report-input",
                str(index_path),
                "--output",
                str(drift_path),
            )
        )
        == 0
    )
    assert (
        main(
            cli_args(
                "generate-run-report",
                "--run-report-input",
                str(index_path),
                "--drift-report-input",
                str(drift_path),
                "--output",
                str(markdown_path),
            )
        )
        == 0
    )

    drift_report = read_json_file(drift_path)
    markdown = markdown_path.read_text(encoding="utf-8")
    assert drift_report["report_type"] == "drift_quality_test"
    assert drift_report["status"] == "passed"
    assert "AI Testing Run Report" in markdown


def test_generate_markdown_reports_smoke(
    tmp_path: Path,
    cli_args: Callable[..., list[str]],
    write_json_file: Callable[[Path, Any], None],
) -> None:
    json_report_path = tmp_path / "input_data_report.json"
    output_dir = tmp_path / "reports"
    write_json_file(
        json_report_path,
        {
            "report_type": "input_data_test",
            "subject": "sample_dataset",
            "step": "Input data testing",
            "status": "passed",
            "summary": {"check_count": 1, "passed_count": 1, "failed_count": 0},
            "checks": [
                {
                    "id": "sample.required_fields",
                    "status": "passed",
                    "severity": "critical",
                    "message": "Required fields are present.",
                    "observed": {"missing": []},
                    "expected": {"missing": []},
                }
            ],
        },
    )

    assert (
        main(
            cli_args(
                "generate-markdown-reports",
                "--output-dir",
                str(output_dir),
                "--report",
                f"input_data={json_report_path}",
                "--no-include-run-report",
            )
        )
        == 0
    )

    markdown = (output_dir / "input_data.md").read_text(encoding="utf-8")
    stage_history_markdown = (tmp_path / "stage_history.md").read_text(encoding="utf-8")
    metric_history_markdown = (tmp_path / "stage_metric_history.md").read_text(encoding="utf-8")
    assert "# Input Data" in markdown
    assert "| ID | Status | Severity | What Was Checked | Observed | Expected |" in markdown
    assert "sample.required_fields" in markdown
    assert "reporting.markdown_reports" in stage_history_markdown
    assert "Stage Metric History" in metric_history_markdown


def test_run_quality_gates_cli_writes_decision_report(
    tmp_path: Path,
    cli_args: Callable[..., list[str]],
    read_json_file: Callable[[Path], Any],
    write_json_file: Callable[[Path, Any], None],
    passed_quality_report: Callable[[str], Record],
) -> None:
    input_report = tmp_path / "input_data_test_report.json"
    association_report = tmp_path / "association_model_report.json"
    category_report = tmp_path / "category_model_report.json"
    output_path = tmp_path / "project_quality_report.json"
    write_json_file(input_report, passed_quality_report("input_data_test"))
    write_json_file(association_report, passed_quality_report("model_quality_test"))
    write_json_file(category_report, passed_quality_report("model_quality_test"))

    assert (
        main(
            cli_args(
                "run-quality-gates",
                "--input-data-report",
                str(input_report),
                "--association-model-report",
                str(association_report),
                "--category-model-report",
                str(category_report),
                "--output",
                str(output_path),
            )
        )
        == 0
    )

    report = read_json_file(output_path)
    assert report["status"] == "passed"
    assert report["decision"]["outcome"] == "accepted"
    assert report["recommended_actions"] == [
        "Keep the report with the run evidence as the model acceptance record."
    ]


def test_llm_exploratory_plan_smoke(
    tmp_path: Path,
    cli_args: Callable[..., list[str]],
    read_json_file: Callable[[Path], Any],
) -> None:
    output_path = tmp_path / "llm_plan.json"

    assert main(cli_args("create-llm-exploratory-plan", "--output", str(output_path))) == 0

    report = read_json_file(output_path)
    assert report["report_type"] == "llm_exploratory_test_plan"
    assert report["summary"]["charter_count"] >= 5

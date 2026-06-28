from __future__ import annotations

import json
from pathlib import Path

from ai_testing.cli import main


def _cli_args(tmp_path: Path, *args: str) -> list[str]:
    return [
        "--stage-history-output",
        str(tmp_path / "stage_history.json"),
        "--stage-history-markdown-output",
        str(tmp_path / "stage_history.md"),
        "--stage-metric-history-markdown-output",
        str(tmp_path / "stage_metric_history.md"),
        *args,
    ]


def test_sample_export_and_preprocessing_smoke(tmp_path: Path) -> None:
    raw_path = tmp_path / "sample_order_items.json"
    second_raw_path = tmp_path / "sample_order_items_second.json"
    processed_path = tmp_path / "processed_order_items.json"
    report_path = tmp_path / "preprocessing_report.json"

    assert main(_cli_args(tmp_path, "export-sample", "--output", str(raw_path))) == 0
    assert (
        main(
            _cli_args(
                tmp_path,
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
    assert main(_cli_args(tmp_path, "export-sample", "--output", str(second_raw_path))) == 0

    processed_records = json.loads(processed_path.read_text(encoding="utf-8"))
    report = json.loads(report_path.read_text(encoding="utf-8"))
    stage_history = json.loads((tmp_path / "stage_history.json").read_text(encoding="utf-8"))
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


def test_run_tracking_smoke(tmp_path: Path) -> None:
    runs_dir = tmp_path / "runs"
    index_path = runs_dir / "run_index.json"

    assert (
        main(
            _cli_args(
                tmp_path,
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
    run_report = json.loads(run_report_path.read_text(encoding="utf-8"))
    run_index = json.loads(index_path.read_text(encoding="utf-8"))
    stage_history = json.loads((tmp_path / "stage_history.json").read_text(encoding="utf-8"))
    assert run_report["run_id"] == "smoke-run"
    assert run_report["report_type"] == "pipeline_run_report"
    assert run_report_markdown_path.exists()
    assert run_index["latest_run_id"] == "smoke-run"
    assert (
        stage_history["latest_execution_by_stage"]["run_tracking"]["stage_name"] == "run_tracking"
    )


def test_drift_and_markdown_report_smoke(tmp_path: Path) -> None:
    runs_dir = tmp_path / "runs"
    index_path = runs_dir / "run_index.json"
    drift_path = tmp_path / "drift_report.json"
    markdown_path = tmp_path / "run_report.md"

    assert (
        main(
            _cli_args(
                tmp_path,
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
            _cli_args(
                tmp_path,
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
            _cli_args(
                tmp_path,
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
            _cli_args(
                tmp_path,
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

    drift_report = json.loads(drift_path.read_text(encoding="utf-8"))
    markdown = markdown_path.read_text(encoding="utf-8")
    assert drift_report["report_type"] == "drift_quality_test"
    assert drift_report["status"] == "passed"
    assert "AI Testing Run Report" in markdown


def test_generate_markdown_reports_smoke(tmp_path: Path) -> None:
    json_report_path = tmp_path / "input_data_report.json"
    output_dir = tmp_path / "reports"
    json_report_path.write_text(
        json.dumps(
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
            }
        ),
        encoding="utf-8",
    )

    assert (
        main(
            _cli_args(
                tmp_path,
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


def test_llm_exploratory_plan_smoke(tmp_path: Path) -> None:
    output_path = tmp_path / "llm_plan.json"

    assert (
        main(_cli_args(tmp_path, "create-llm-exploratory-plan", "--output", str(output_path))) == 0
    )

    report = json.loads(output_path.read_text(encoding="utf-8"))
    assert report["report_type"] == "llm_exploratory_test_plan"
    assert report["summary"]["charter_count"] >= 5

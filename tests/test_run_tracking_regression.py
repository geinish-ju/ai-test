from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from ai_testing.observability.run_tracking import RunTrackingConfig, build_run_report

Record = dict[str, Any]


def test_run_tracking_marks_directional_metric_regression_and_quality_decision(
    tmp_path: Path,
    write_json_file: Callable[[Path, Any], None],
) -> None:
    category_test_report = tmp_path / "category_classifier_test_report.json"
    project_quality_report = tmp_path / "project_quality_report.json"
    write_json_file(category_test_report, {"test": {"accuracy": 0.84}})
    write_json_file(
        project_quality_report,
        {
            "report_type": "project_quality_gates",
            "status": "passed",
            "summary": {"check_count": 3, "passed_count": 3, "failed_count": 0},
            "decision": {
                "outcome": "accepted",
                "recommendation": "Accept the candidate.",
            },
        },
    )

    report = build_run_report(
        RunTrackingConfig(
            runs_dir=tmp_path / "runs",
            index_output=tmp_path / "runs" / "run_index.json",
            stage_report_paths={
                "category_testing": category_test_report,
                "project_quality": project_quality_report,
            },
            artifact_paths={},
        ),
        run_id="candidate",
        baseline_report={
            "run_id": "baseline",
            "metrics": {
                "category.test.accuracy": {
                    "value": 0.9,
                    "direction": "higher_is_better",
                }
            },
            "distributions": {},
        },
    )

    comparison = report["comparison"]
    metric_delta = comparison["metric_deltas"][0]

    assert report["metrics"]["category.test.accuracy"]["value"] == 0.84
    assert metric_delta["name"] == "category.test.accuracy"
    assert metric_delta["change"] == "regressed"
    assert comparison["regressed_metrics"] == ["category.test.accuracy"]
    assert report["quality_summary"]["reports"]["project_quality"]["decision_outcome"] == "accepted"

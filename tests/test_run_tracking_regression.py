from __future__ import annotations

import json
from pathlib import Path

from ai_testing.run_tracking import RunTrackingConfig, build_run_report


def test_run_tracking_marks_directional_metric_regression_and_quality_decision(
    tmp_path: Path,
) -> None:
    category_test_report = tmp_path / "category_classifier_test_report.json"
    project_quality_report = tmp_path / "project_quality_report.json"
    category_test_report.write_text(
        json.dumps({"test": {"accuracy": 0.84}}, ensure_ascii=False),
        encoding="utf-8",
    )
    project_quality_report.write_text(
        json.dumps(
            {
                "report_type": "project_quality_gates",
                "status": "passed",
                "summary": {"check_count": 3, "passed_count": 3, "failed_count": 0},
                "decision": {
                    "outcome": "accepted",
                    "recommendation": "Accept the candidate.",
                },
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
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

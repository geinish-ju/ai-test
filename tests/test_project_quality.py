from __future__ import annotations

from collections.abc import Callable
from typing import Any

from ai_testing.observability.reporting import build_markdown_report
from ai_testing.quality.project import aggregate_quality_reports

Record = dict[str, Any]


def test_project_quality_rejects_critical_child_failure(
    quality_report_factory: Callable[..., Record],
) -> None:
    report = aggregate_quality_reports(
        {
            "input_data": quality_report_factory("input_data_test", "passed"),
            "category_model": quality_report_factory(
                "model_quality_test",
                "failed",
                failed_check={
                    "id": "ml_model.source_feature_leakage",
                    "severity": "critical",
                    "message": "Source features contain target leakage.",
                    "observed": {"forbidden_intersection": ["main_category"]},
                    "expected": {"forbidden_intersection": []},
                    "diagnostics": {
                        "suggested_actions": [
                            "Remove target-derived fields from classifier text features."
                        ]
                    },
                },
            ),
        }
    ).report

    assert report["status"] == "failed"
    assert report["decision"]["outcome"] == "rejected"
    assert report["metrics"]["blocker_count"] == 1
    assert report["blockers"][0]["check_id"] == "ml_model.source_feature_leakage"
    assert (
        "Remove target-derived fields from classifier text features."
        in report["recommended_actions"]
    )


def test_project_quality_marks_major_only_failures_for_review(
    quality_report_factory: Callable[..., Record],
) -> None:
    report = aggregate_quality_reports(
        {
            "association_model": quality_report_factory(
                "model_quality_test",
                "failed",
                failed_check={
                    "id": "ml_model.test_lift",
                    "severity": "major",
                    "message": "Mean test lift is below the acceptance threshold.",
                    "observed": {"value": 0.96},
                    "expected": {">=": 1.0},
                },
            ),
        }
    ).report

    assert report["status"] == "failed"
    assert report["decision"]["outcome"] == "needs_review"
    assert report["metrics"]["blocker_count"] == 0
    assert report["metrics"]["warning_count"] == 1
    assert report["warnings"][0]["recommended_action"].startswith("Inspect association rule")


def test_project_quality_markdown_contains_decision_and_child_matrix(
    quality_report_factory: Callable[..., Record],
) -> None:
    report = aggregate_quality_reports(
        {
            "input_data": quality_report_factory("input_data_test", "passed"),
            "category_model": quality_report_factory(
                "model_quality_test",
                "failed",
                failed_check={
                    "id": "ml_model.test_macro_f1",
                    "severity": "critical",
                    "message": "Hold-out macro F1 is below threshold.",
                    "observed": {"value": 0.51},
                    "expected": {">=": 0.7},
                },
            ),
        }
    ).report

    markdown = build_markdown_report(report)

    assert "## Quality Decision" in markdown
    assert "| Outcome | `rejected` |" in markdown
    assert "### Blockers" in markdown
    assert "## Child Quality Reports" in markdown
    assert "`category_model`" in markdown

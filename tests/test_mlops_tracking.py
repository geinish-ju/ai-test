from __future__ import annotations

from pathlib import Path

from ai_testing.observability.reporting import build_run_markdown_report
from ai_testing.observability.run_tracking import (
    DVCVersioningConfig,
    MLflowTrackingConfig,
    MLOpsPublishConfig,
    publish_run_to_mlops,
)
from ai_testing.observability.run_tracking.mlops import _ensure_mlflow_tracking_storage


def test_mlops_publish_disabled_backends(tmp_path: Path) -> None:
    run_report_path = tmp_path / "run_report.json"
    markdown_report_path = tmp_path / "run_report.md"
    run_report_path.write_text("{}", encoding="utf-8")
    markdown_report_path.write_text("# Run", encoding="utf-8")

    result = publish_run_to_mlops(
        {
            "run_id": "test-run",
            "metrics": {"quality.failed_count": {"value": 0}},
            "artifacts": {},
        },
        run_report_path=run_report_path,
        markdown_report_path=markdown_report_path,
        config=MLOpsPublishConfig(),
    )

    assert result["mlflow"]["status"] == "disabled"
    assert result["dvc"]["status"] == "disabled"


def test_mlflow_default_uses_sqlite_tracking_backend() -> None:
    assert MLflowTrackingConfig().tracking_uri == "sqlite:///data/mlflow/mlflow.db"


def test_mlflow_sqlite_tracking_uri_prepares_parent_dir(tmp_path: Path) -> None:
    database_path = tmp_path / "tracking" / "mlflow.db"

    _ensure_mlflow_tracking_storage(f"sqlite:///{database_path.as_posix()}")

    assert database_path.parent.is_dir()


def test_dvc_missing_command_is_skipped(tmp_path: Path) -> None:
    artifact_path = tmp_path / "model.json"
    artifact_path.write_text("{}", encoding="utf-8")

    result = publish_run_to_mlops(
        {
            "run_id": "test-run",
            "metrics": {},
            "artifacts": {
                "model": {
                    "path": str(artifact_path),
                    "type": "model",
                    "exists": True,
                }
            },
        },
        run_report_path=tmp_path / "run_report.json",
        markdown_report_path=tmp_path / "run_report.md",
        config=MLOpsPublishConfig(
            dvc=DVCVersioningConfig(enabled=True, command="definitely-missing-dvc")
        ),
    )

    assert result["dvc"]["status"] == "skipped"
    assert result["dvc"]["reason"] == "dvc_not_installed"


def test_run_markdown_explains_skipped_mlops_backends() -> None:
    markdown = build_run_markdown_report(
        {
            "run_id": "test-run",
            "summary": {},
            "comparison": {},
            "quality_summary": {},
            "artifacts": {},
            "stage_reports": {},
            "mlops": {
                "mlflow": {
                    "status": "skipped",
                    "reason": "mlflow_not_installed",
                    "message": "No module named 'mlflow'",
                },
                "dvc": {
                    "status": "skipped",
                    "reason": "dvc_not_installed",
                    "message": "DVC command not found: dvc",
                },
            },
        }
    )

    assert "mlflow_not_installed: No module named 'mlflow'" in markdown
    assert "dvc_not_installed: DVC command not found: dvc" in markdown
    assert "tracked=0" not in markdown

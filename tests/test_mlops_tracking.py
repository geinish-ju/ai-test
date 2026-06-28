from __future__ import annotations

from pathlib import Path

from ai_testing.run_tracking import (
    DVCVersioningConfig,
    MLOpsPublishConfig,
    publish_run_to_mlops,
)


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

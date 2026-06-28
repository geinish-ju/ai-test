from __future__ import annotations

import importlib
import re
import shutil
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

Record = dict[str, Any]


@dataclass(frozen=True)
class MLflowTrackingConfig:
    enabled: bool = False
    tracking_uri: str = "sqlite:///data/mlflow/mlflow.db"
    experiment_name: str = "ai-grocery-testing"
    log_artifacts: bool = True
    artifact_types: tuple[str, ...] = ("report", "manifest", "model")
    fail_on_error: bool = False


@dataclass(frozen=True)
class DVCVersioningConfig:
    enabled: bool = False
    command: str = "dvc"
    artifact_names: tuple[str, ...] = ()
    push: bool = False
    remote: str = ""
    fail_on_error: bool = False


@dataclass(frozen=True)
class MLOpsPublishConfig:
    mlflow: MLflowTrackingConfig = MLflowTrackingConfig()
    dvc: DVCVersioningConfig = DVCVersioningConfig()


def publish_run_to_mlops(
    run_report: Mapping[str, Any],
    *,
    run_report_path: Path,
    markdown_report_path: Path,
    config: MLOpsPublishConfig,
) -> Record:
    return {
        "mlflow": _publish_to_mlflow(
            run_report,
            run_report_path=run_report_path,
            markdown_report_path=markdown_report_path,
            config=config.mlflow,
        ),
        "dvc": _publish_to_dvc(run_report, config=config.dvc),
    }


def _publish_to_mlflow(
    run_report: Mapping[str, Any],
    *,
    run_report_path: Path,
    markdown_report_path: Path,
    config: MLflowTrackingConfig,
) -> Record:
    if not config.enabled:
        return {"status": "disabled"}
    if config.tracking_uri:
        _ensure_mlflow_tracking_storage(config.tracking_uri)

    try:
        mlflow = importlib.import_module("mlflow")
    except ImportError as error:
        return _handle_optional_error(
            error,
            fail_on_error=config.fail_on_error,
            status="skipped",
            reason="mlflow_not_installed",
        )

    try:
        if config.tracking_uri:
            mlflow.set_tracking_uri(config.tracking_uri)
        mlflow.set_experiment(config.experiment_name)

        run_name = _string_value(run_report.get("run_name")) or _string_value(
            run_report.get("run_id")
        )
        logged_metric_count = 0
        logged_artifact_count = 0
        with mlflow.start_run(run_name=run_name) as active_run:
            mlflow.set_tags(_mlflow_tags(run_report))
            for name, value in _mlflow_params(run_report).items():
                mlflow.log_param(name, value)
            for name, metric in _mapping(run_report.get("metrics")).items():
                metric_value = _float_value(_mapping(metric).get("value"))
                if metric_value is None:
                    continue
                mlflow.log_metric(_mlflow_name(str(name)), metric_value)
                logged_metric_count += 1

            if config.log_artifacts:
                logged_artifact_count += _log_mlflow_artifact(
                    mlflow,
                    run_report_path,
                    artifact_path="run_reports",
                )
                logged_artifact_count += _log_mlflow_artifact(
                    mlflow,
                    markdown_report_path,
                    artifact_path="run_reports",
                )
                logged_artifact_count += _log_selected_mlflow_artifacts(
                    mlflow,
                    _mapping(run_report.get("artifacts")),
                    allowed_types=set(config.artifact_types),
                )

            return {
                "status": "logged",
                "tracking_uri": mlflow.get_tracking_uri(),
                "experiment_name": config.experiment_name,
                "mlflow_run_id": active_run.info.run_id,
                "logged_metric_count": logged_metric_count,
                "logged_artifact_count": logged_artifact_count,
            }
    except Exception as error:
        return _handle_optional_error(
            error,
            fail_on_error=config.fail_on_error,
            status="failed",
            reason="mlflow_error",
        )


def _ensure_mlflow_tracking_storage(tracking_uri: str) -> None:
    sqlite_prefix = "sqlite:///"
    if not tracking_uri.startswith(sqlite_prefix):
        return

    database_path = tracking_uri.removeprefix(sqlite_prefix)
    if not database_path or database_path == ":memory:":
        return

    Path(database_path).expanduser().parent.mkdir(parents=True, exist_ok=True)


def _publish_to_dvc(
    run_report: Mapping[str, Any],
    *,
    config: DVCVersioningConfig,
) -> Record:
    if not config.enabled:
        return {"status": "disabled"}
    if shutil.which(config.command) is None:
        return _handle_optional_error(
            RuntimeError(f"DVC command not found: {config.command}"),
            fail_on_error=config.fail_on_error,
            status="skipped",
            reason="dvc_not_installed",
        )

    selected_artifacts = _selected_dvc_artifacts(
        _mapping(run_report.get("artifacts")),
        artifact_names=config.artifact_names,
    )
    commands: list[Record] = []
    tracked_count = 0
    for name, artifact in selected_artifacts.items():
        path = Path(str(artifact.get("path")))
        if not path.exists():
            commands.append(
                {
                    "artifact": name,
                    "path": str(path),
                    "status": "skipped",
                    "reason": "missing",
                }
            )
            continue
        command_result = _run_dvc_command(config.command, "add", str(path))
        command_result["artifact"] = name
        command_result["path"] = str(path)
        commands.append(command_result)
        if command_result.get("status") == "completed":
            tracked_count += 1

    push_result: Record | None = None
    if config.push:
        push_args = ["push"]
        if config.remote:
            push_args.extend(["-r", config.remote])
        push_result = _run_dvc_command(config.command, *push_args)

    failed_commands = [command for command in commands if command.get("status") == "failed"]
    if push_result is not None and push_result.get("status") == "failed":
        failed_commands.append(push_result)
    status = "failed" if failed_commands else "completed"
    if status == "failed" and config.fail_on_error:
        raise RuntimeError("DVC versioning failed.")
    return {
        "status": status,
        "tracked_artifact_count": tracked_count,
        "selected_artifact_count": len(selected_artifacts),
        "commands": commands,
        "push": push_result,
    }


def _log_selected_mlflow_artifacts(
    mlflow: Any,
    artifacts: Mapping[str, Any],
    *,
    allowed_types: set[str],
) -> int:
    logged_count = 0
    for artifact in artifacts.values():
        artifact_map = _mapping(artifact)
        artifact_type = _string_value(artifact_map.get("type"))
        if artifact_type not in allowed_types:
            continue
        path = Path(str(artifact_map.get("path")))
        logged_count += _log_mlflow_artifact(
            mlflow,
            path,
            artifact_path=f"artifacts/{artifact_type or 'other'}",
        )
    return logged_count


def _log_mlflow_artifact(mlflow: Any, path: Path, *, artifact_path: str) -> int:
    if not path.is_file():
        return 0
    mlflow.log_artifact(str(path), artifact_path=artifact_path)
    return 1


def _selected_dvc_artifacts(
    artifacts: Mapping[str, Any],
    *,
    artifact_names: Sequence[str],
) -> dict[str, Mapping[str, Any]]:
    names = set(artifact_names)
    return {
        name: artifact
        for name, artifact in artifacts.items()
        if isinstance(name, str)
        and isinstance(artifact, Mapping)
        and (not names or name in names)
        and _string_value(artifact.get("path")) is not None
    }


def _run_dvc_command(command: str, *args: str) -> Record:
    completed = subprocess.run(
        [command, *args],
        check=False,
        capture_output=True,
        encoding="utf-8",
    )
    return {
        "command": [command, *args],
        "status": "completed" if completed.returncode == 0 else "failed",
        "returncode": completed.returncode,
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
    }


def _mlflow_tags(run_report: Mapping[str, Any]) -> dict[str, str]:
    summary = _mapping(run_report.get("summary"))
    quality_summary = _mapping(run_report.get("quality_summary"))
    git_state = _mapping(run_report.get("git"))
    tags = {
        "ai_test.run_id": _string_value(run_report.get("run_id")) or "",
        "ai_test.run_name": _string_value(run_report.get("run_name")) or "",
        "ai_test.quality_status": _string_value(quality_summary.get("status"))
        or _string_value(summary.get("quality_status"))
        or "",
        "ai_test.git_commit": _string_value(git_state.get("commit")) or "",
        "ai_test.git_branch": _string_value(git_state.get("branch")) or "",
    }
    return {key: value for key, value in tags.items() if value}


def _mlflow_params(run_report: Mapping[str, Any]) -> dict[str, str]:
    summary = _mapping(run_report.get("summary"))
    comparison = _mapping(run_report.get("comparison"))
    return {
        "run_id": _string_value(run_report.get("run_id")) or "",
        "run_name": _string_value(run_report.get("run_name")) or "",
        "baseline_run_id": _string_value(comparison.get("baseline_run_id")) or "",
        "available_stage_report_count": str(summary.get("available_stage_report_count") or ""),
        "available_artifact_count": str(summary.get("available_artifact_count") or ""),
        "metric_count": str(summary.get("metric_count") or ""),
    }


def _mlflow_name(value: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9_.\-/ ]+", "_", value.strip())
    return normalized[:250] or "metric"


def _handle_optional_error(
    error: Exception,
    *,
    fail_on_error: bool,
    status: str,
    reason: str,
) -> Record:
    if fail_on_error:
        raise error
    return {
        "status": status,
        "reason": reason,
        "message": str(error),
    }


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _string_value(value: Any) -> str | None:
    if value is None or isinstance(value, bool):
        return None
    text = str(value).strip()
    return text or None


def _float_value(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip())
        except ValueError:
            return None
    return None

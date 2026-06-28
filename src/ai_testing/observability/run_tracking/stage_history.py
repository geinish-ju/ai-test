from __future__ import annotations

import json
import os
import subprocess
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ai_testing.core import artifact_metadata

Record = dict[str, Any]

METRIC_TERMS = (
    "accuracy",
    "precision",
    "recall",
    "f1",
    "confidence",
    "lift",
    "coverage",
    "hit_rate",
    "gap",
    "count",
    "rate",
    "support",
    "vocabulary_size",
    "record_count",
    "class_count",
    "metric_count",
)

LOWER_IS_BETTER_TERMS = (
    "failed",
    "failure",
    "gap",
    "missing",
    "invalid",
    "duplicate",
    "dropped",
    "error",
    "std",
)

HIGHER_IS_BETTER_TERMS = (
    "accuracy",
    "precision",
    "recall",
    "f1",
    "confidence",
    "lift",
    "coverage",
    "hit_rate",
    "stable_rule_count",
)

SKIPPED_BRANCHES = {
    "artifacts",
    "inputs",
    "outputs",
    "checks",
    "diagnostics",
    "sample_records",
    "top_rules",
    "per_class",
    "classes",
    "charters",
    "folds",
    "frequent_itemsets",
    "association_rules",
}

LOCK_TIMEOUT_SECONDS = 30.0
LOCK_POLL_SECONDS = 0.05
LOCK_STALE_SECONDS = 600.0


@dataclass(frozen=True)
class StageExecutionTrackingConfig:
    history_output: Path
    markdown_output: Path
    metric_history_markdown_output: Path | None = None
    max_entries: int = 500
    recent_entries_per_stage: int = 5
    metric_history_recent_entries_per_stage: int = 10


def record_stage_execution(
    config: StageExecutionTrackingConfig,
    *,
    stage_name: str,
    command_name: str,
    artifacts: Mapping[str, Path],
    payload: Mapping[str, Any] | None = None,
    status: str | None = None,
    parameters: Mapping[str, Any] | None = None,
) -> Record:
    config.history_output.parent.mkdir(parents=True, exist_ok=True)
    config.markdown_output.parent.mkdir(parents=True, exist_ok=True)
    if config.metric_history_markdown_output is not None:
        config.metric_history_markdown_output.parent.mkdir(parents=True, exist_ok=True)

    with _StageHistoryLock(_lock_path(config.history_output)):
        history = _read_history(config.history_output)
        executions = _list_of_mappings(history.get("executions"))
        previous_execution = _latest_stage_execution(executions, stage_name)
        metrics = _extract_metrics(payload or {})
        checks = _extract_checks(payload or {})
        metric_deltas = _metric_deltas(metrics, _mapping(previous_execution.get("metrics")))
        execution = {
            "execution_id": _execution_id(stage_name),
            "stage_name": stage_name,
            "command_name": command_name,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "status": status or _payload_status(payload),
            "git": _git_state(),
            "parameters": dict(parameters or {}),
            "previous_execution_id": previous_execution.get("execution_id"),
            "metrics": metrics,
            "metric_deltas": metric_deltas,
            "metric_delta_summary": _metric_delta_summary(metric_deltas),
            "checks": checks,
            "artifacts": {
                name: artifact_metadata(path, _artifact_type(name), name)
                for name, path in sorted(artifacts.items())
            },
        }
        executions.append(execution)
        executions = executions[-max(config.max_entries, 1) :]
        updated_history = {
            "report_type": "stage_execution_history",
            "subject": "ai_grocery_testing_project",
            "updated_at": execution["created_at"],
            "execution_count": len(executions),
            "latest_execution_id": execution["execution_id"],
            "latest_execution_by_stage": _latest_execution_by_stage(executions),
            "executions": executions,
        }
        config.history_output.write_text(
            json.dumps(updated_history, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        config.markdown_output.write_text(
            build_stage_history_markdown(
                updated_history,
                recent_entries_per_stage=config.recent_entries_per_stage,
            ),
            encoding="utf-8",
        )
        if config.metric_history_markdown_output is not None:
            config.metric_history_markdown_output.write_text(
                build_stage_metric_history_markdown(
                    updated_history,
                    recent_entries_per_stage=config.metric_history_recent_entries_per_stage,
                ),
                encoding="utf-8",
            )
    return execution


def build_stage_history_markdown(
    history: Mapping[str, Any],
    *,
    recent_entries_per_stage: int = 5,
) -> str:
    executions = _list_of_mappings(history.get("executions"))
    lines = [
        "# Stage Execution History",
        "",
        "| Field | Value |",
        "|---|---:|",
        f"| Updated at | `{_text(history.get('updated_at'), '')}` |",
        f"| Execution count | {len(executions)} |",
        f"| Latest execution | `{_text(history.get('latest_execution_id'), '')}` |",
        "",
        "## Latest Stage Results",
        "",
        "| Stage | Command | Status | Created At | Metrics | Improved | Regressed | Report |",
        "|---|---|---:|---:|---:|---:|---:|---|",
    ]

    latest_by_stage = _mapping(history.get("latest_execution_by_stage"))
    for stage_name, execution in sorted(latest_by_stage.items()):
        execution_map = _mapping(execution)
        delta_summary = _mapping(execution_map.get("metric_delta_summary"))
        lines.append(
            f"| `{_escape(stage_name)}` | `{_escape(execution_map.get('command_name'))}` | "
            f"`{_escape(execution_map.get('status'))}` | "
            f"`{_escape(execution_map.get('created_at'))}` | "
            f"{len(_mapping(execution_map.get('metrics')))} | "
            f"{_number(delta_summary.get('improved_metric_count'))} | "
            f"{_number(delta_summary.get('regressed_metric_count'))} | "
            f"`{_escape(_primary_artifact_path(execution_map))}` |"
        )

    lines.extend(["", "## Stage Details"])
    for stage_name in sorted({str(execution.get("stage_name")) for execution in executions}):
        stage_executions = [
            execution for execution in executions if execution.get("stage_name") == stage_name
        ][-max(recent_entries_per_stage, 1) :]
        stage_executions.reverse()
        lines.extend(["", f"### {_escape(stage_name)}", ""])
        for execution in stage_executions:
            lines.extend(
                [
                    f"#### `{_escape(execution.get('execution_id'))}`",
                    "",
                    "| Field | Value |",
                    "|---|---:|",
                    f"| Command | `{_escape(execution.get('command_name'))}` |",
                    f"| Status | `{_escape(execution.get('status'))}` |",
                    f"| Created at | `{_escape(execution.get('created_at'))}` |",
                    f"| Previous execution | `{_escape(execution.get('previous_execution_id'))}` |",
                    "",
                ]
            )
            _add_metrics_table(lines, _mapping(execution.get("metrics")))
            _add_delta_table(lines, _list_of_mappings(execution.get("metric_deltas")))
            _add_artifacts_table(lines, _mapping(execution.get("artifacts")))

    lines.append("")
    return "\n".join(lines)


def build_stage_metric_history_markdown(
    history: Mapping[str, Any],
    *,
    recent_entries_per_stage: int = 10,
) -> str:
    executions = _list_of_mappings(history.get("executions"))
    lines = [
        "# Stage Metric History",
        "",
        "Each section compares recent executions of one pipeline stage. "
        "Run 1 is the oldest retained execution in that stage table; the last run is the latest.",
        "",
        "| Field | Value |",
        "|---|---:|",
        f"| Updated at | `{_text(history.get('updated_at'), '')}` |",
        f"| Execution count | {len(executions)} |",
        f"| Latest execution | `{_text(history.get('latest_execution_id'), '')}` |",
        "",
    ]

    stage_names = sorted({str(execution.get("stage_name")) for execution in executions})
    for stage_name in stage_names:
        stage_executions = [
            execution for execution in executions if execution.get("stage_name") == stage_name
        ][-max(recent_entries_per_stage, 1) :]
        if not stage_executions:
            continue

        lines.extend(["", f"## {_escape(stage_name)}", ""])
        _add_run_index_table(lines, stage_executions)
        _add_metric_history_table(lines, stage_executions)
        _add_check_history_table(lines, stage_executions)

    lines.append("")
    return "\n".join(lines)


def _add_run_index_table(lines: list[str], executions: Sequence[Mapping[str, Any]]) -> None:
    lines.extend(
        [
            "| Run | Execution | Command | Status | Created At |",
            "|---:|---|---|---:|---:|",
        ]
    )
    for index, execution in enumerate(executions, start=1):
        lines.append(
            f"| Run {index} | `{_escape(execution.get('execution_id'))}` | "
            f"`{_escape(execution.get('command_name'))}` | "
            f"`{_escape(execution.get('status'))}` | "
            f"`{_escape(execution.get('created_at'))}` |"
        )
    lines.append("")


def _add_metric_history_table(lines: list[str], executions: Sequence[Mapping[str, Any]]) -> None:
    metric_names = sorted(
        {str(name) for execution in executions for name in _mapping(execution.get("metrics"))}
    )
    if not metric_names:
        lines.extend(["No numeric metrics were extracted for these executions.", ""])
        return

    headers = [f"Run {index}" for index in range(1, len(executions) + 1)]
    lines.append("| Metric | " + " | ".join(headers) + " |")
    lines.append("|---|" + "---:|" * len(headers))
    for metric_name in metric_names:
        values = [
            _metric_history_value(_mapping(execution.get("metrics")), metric_name)
            for execution in executions
        ]
        lines.append(f"| `{_escape(metric_name)}` | " + " | ".join(values) + " |")
    lines.append("")


def _add_check_history_table(lines: list[str], executions: Sequence[Mapping[str, Any]]) -> None:
    check_ids = sorted(
        {
            str(check_id)
            for execution in executions
            for check_id in _mapping(execution.get("checks"))
        }
    )
    if not check_ids:
        return

    headers = [f"Run {index}" for index in range(1, len(executions) + 1)]
    lines.extend(["### Checks", ""])
    lines.append("| Check | Severity | " + " | ".join(headers) + " |")
    lines.append("|---|---|" + "---:|" * len(headers))
    for check_id in check_ids:
        severities = [
            _text(_mapping(_mapping(execution.get("checks")).get(check_id)).get("severity"), "")
            for execution in executions
        ]
        severity = next((value for value in severities if value), "")
        statuses = [f"`{_escape(_check_status(execution, check_id))}`" for execution in executions]
        lines.append(
            f"| `{_escape(check_id)}` | `{_escape(severity)}` | " + " | ".join(statuses) + " |"
        )
    lines.append("")


def _add_metrics_table(lines: list[str], metrics: Mapping[str, Any]) -> None:
    if not metrics:
        lines.extend(["Metrics were not extracted for this execution.", ""])
        return
    lines.extend(["| Metric | Value | Direction |", "|---|---:|---:|"])
    for name, metric in sorted(metrics.items())[:30]:
        metric_map = _mapping(metric)
        lines.append(
            f"| `{_escape(name)}` | {_number(metric_map.get('value'))} | "
            f"`{_escape(metric_map.get('direction'))}` |"
        )
    lines.append("")


def _add_delta_table(lines: list[str], deltas: Sequence[Mapping[str, Any]]) -> None:
    comparable = [
        delta for delta in deltas if delta.get("change") in {"improved", "regressed", "changed"}
    ]
    if not comparable:
        lines.extend(["No metric changes against the previous execution of this stage.", ""])
        return
    lines.extend(
        [
            "| Changed Metric | Previous | Current | Delta | Change |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for delta in comparable[:30]:
        lines.append(
            f"| `{_escape(delta.get('name'))}` | {_number(delta.get('previous'))} | "
            f"{_number(delta.get('current'))} | {_number(delta.get('delta'))} | "
            f"`{_escape(delta.get('change'))}` |"
        )
    lines.append("")


def _add_artifacts_table(lines: list[str], artifacts: Mapping[str, Any]) -> None:
    if not artifacts:
        return
    lines.extend(["| Artifact | Path | SHA256 |", "|---|---|---|"])
    for name, artifact in sorted(artifacts.items()):
        artifact_map = _mapping(artifact)
        lines.append(
            f"| `{_escape(name)}` | `{_escape(artifact_map.get('path'))}` | "
            f"`{_short_sha(artifact_map.get('sha256'))}` |"
        )
    lines.append("")


def _extract_metrics(payload: Mapping[str, Any]) -> Record:
    metrics: Record = {}
    _collect_metrics(payload, metrics=metrics)
    return dict(sorted(metrics.items()))


def _extract_checks(payload: Mapping[str, Any]) -> Record:
    checks = payload.get("checks")
    if not isinstance(checks, Sequence) or isinstance(checks, (str, bytes)):
        return {}

    extracted: Record = {}
    for index, check in enumerate(checks, start=1):
        check_map = _mapping(check)
        check_id = _string_value(check_map.get("id")) or f"check_{index}"
        extracted[check_id] = {
            "status": _text(check_map.get("status"), ""),
            "severity": _text(check_map.get("severity"), ""),
            "message": _text(check_map.get("message"), ""),
        }
    return dict(sorted(extracted.items()))


def _collect_metrics(
    value: Any,
    *,
    metrics: Record,
    path: str = "",
    depth: int = 0,
) -> None:
    if depth > 5:
        return
    if isinstance(value, Mapping):
        for key, child_value in value.items():
            key_text = str(key)
            if key_text in SKIPPED_BRANCHES:
                continue
            child_path = f"{path}.{key_text}" if path else key_text
            _collect_metrics(child_value, metrics=metrics, path=child_path, depth=depth + 1)
        return
    number = _float_value(value)
    if number is None or not _metric_candidate(path):
        return
    metrics[path] = {
        "value": round(number, 6),
        "direction": _metric_direction(path),
    }


def _metric_deltas(
    current_metrics: Mapping[str, Any],
    previous_metrics: Mapping[str, Any],
) -> list[Record]:
    deltas: list[Record] = []
    for name, metric in sorted(current_metrics.items()):
        metric_map = _mapping(metric)
        previous_metric = _mapping(previous_metrics.get(name))
        if not previous_metric:
            continue
        current_value = _float_value(metric_map.get("value"))
        previous_value = _float_value(previous_metric.get("value"))
        if current_value is None or previous_value is None:
            continue
        delta = round(current_value - previous_value, 6)
        direction = _text(metric_map.get("direction"), "neutral")
        deltas.append(
            {
                "name": name,
                "previous": previous_value,
                "current": current_value,
                "delta": delta,
                "direction": direction,
                "change": _metric_change(delta, direction),
            }
        )
    return deltas


def _metric_delta_summary(deltas: Sequence[Mapping[str, Any]]) -> Record:
    return {
        "compared_metric_count": len(deltas),
        "improved_metric_count": sum(1 for delta in deltas if delta.get("change") == "improved"),
        "regressed_metric_count": sum(1 for delta in deltas if delta.get("change") == "regressed"),
        "changed_metric_count": sum(1 for delta in deltas if delta.get("change") == "changed"),
    }


def _latest_execution_by_stage(executions: Sequence[Mapping[str, Any]]) -> Record:
    latest: Record = {}
    for execution in executions:
        stage_name = _text(execution.get("stage_name"), "")
        if stage_name:
            latest[stage_name] = dict(execution)
    return latest


def _latest_stage_execution(
    executions: Sequence[Mapping[str, Any]],
    stage_name: str,
) -> Mapping[str, Any]:
    for execution in reversed(executions):
        if execution.get("stage_name") == stage_name:
            return execution
    return {}


def _read_history(path: Path) -> Record:
    if not path.exists():
        return {
            "report_type": "stage_execution_history",
            "subject": "ai_grocery_testing_project",
            "executions": [],
        }
    payload = json.loads(path.read_text(encoding="utf-8"))
    return dict(payload) if isinstance(payload, Mapping) else {"executions": []}


class _StageHistoryLock:
    def __init__(self, path: Path) -> None:
        self.path = path
        self._fd: int | None = None

    def __enter__(self) -> _StageHistoryLock:
        deadline = time.monotonic() + LOCK_TIMEOUT_SECONDS
        while True:
            try:
                self._fd = os.open(str(self.path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            except FileExistsError:
                _remove_stale_lock(self.path)
                if time.monotonic() >= deadline:
                    raise TimeoutError(
                        f"Timed out waiting for stage history lock: {self.path}"
                    ) from None
                time.sleep(LOCK_POLL_SECONDS)
                continue

            os.write(
                self._fd,
                f"pid={os.getpid()} created_at={datetime.now(timezone.utc).isoformat()}".encode(),
            )
            return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        if self._fd is not None:
            os.close(self._fd)
            self._fd = None
        self.path.unlink(missing_ok=True)


def _lock_path(history_output: Path) -> Path:
    return history_output.with_name(f"{history_output.name}.lock")


def _remove_stale_lock(path: Path) -> None:
    try:
        lock_age_seconds = time.time() - path.stat().st_mtime
    except OSError:
        return
    if lock_age_seconds > LOCK_STALE_SECONDS:
        path.unlink(missing_ok=True)


def _payload_status(payload: Mapping[str, Any] | None) -> str:
    if payload is None:
        return "completed"
    status = _string_value(payload.get("status"))
    return status or "completed"


def _metric_candidate(path: str) -> bool:
    normalized = path.lower()
    return any(term in normalized for term in METRIC_TERMS)


def _metric_direction(path: str) -> str:
    normalized = path.lower()
    if any(term in normalized for term in LOWER_IS_BETTER_TERMS):
        return "lower_is_better"
    if any(term in normalized for term in HIGHER_IS_BETTER_TERMS):
        return "higher_is_better"
    return "neutral"


def _metric_change(delta: float, direction: str) -> str:
    if abs(delta) <= 0.000001:
        return "unchanged"
    if direction == "higher_is_better":
        return "improved" if delta > 0 else "regressed"
    if direction == "lower_is_better":
        return "improved" if delta < 0 else "regressed"
    return "changed"


def _metric_history_value(metrics: Mapping[str, Any], metric_name: str) -> str:
    metric = _mapping(metrics.get(metric_name))
    if not metric:
        return ""
    return _number(metric.get("value"))


def _check_status(execution: Mapping[str, Any], check_id: str) -> str:
    check = _mapping(_mapping(execution.get("checks")).get(check_id))
    return _text(check.get("status"), "")


def _primary_artifact_path(execution: Mapping[str, Any]) -> str:
    artifacts = _mapping(execution.get("artifacts"))
    for artifact in artifacts.values():
        artifact_map = _mapping(artifact)
        path = _string_value(artifact_map.get("path"))
        if path:
            return path
    return ""


def _artifact_type(name: str) -> str:
    if "report" in name:
        return "report"
    if "manifest" in name:
        return "manifest"
    if "model" in name or "estimator" in name:
        return "model"
    return "dataset"


def _execution_id(stage_name: str) -> str:
    safe_stage = "".join(char if char.isalnum() else "-" for char in stage_name.lower()).strip("-")
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    return f"{safe_stage}-{timestamp}"


def _git_state() -> Record:
    return {
        "commit": _git_output("rev-parse", "HEAD"),
        "branch": _git_output("branch", "--show-current"),
        "is_dirty": bool(_git_output("status", "--short")),
    }


def _git_output(*args: str) -> str | None:
    try:
        completed = subprocess.run(
            ["git", *args],
            check=False,
            capture_output=True,
            encoding="utf-8",
        )
    except OSError:
        return None
    if completed.returncode != 0:
        return None
    text = completed.stdout.strip()
    return text or None


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _list_of_mappings(value: Any) -> list[Record]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return []
    return [dict(item) for item in value if isinstance(item, Mapping)]


def _text(value: Any, default: str) -> str:
    if value is None:
        return default
    if isinstance(value, bool):
        return str(value).lower()
    text = str(value).strip()
    return text or default


def _string_value(value: Any) -> str | None:
    if value is None or isinstance(value, bool):
        return None
    text = str(value).strip()
    return text or None


def _number(value: Any) -> str:
    number = _float_value(value)
    if number is None:
        return _text(value, "")
    return f"{number:.6g}"


def _escape(value: Any) -> str:
    text = "" if value is None else str(value)
    return text.replace("\n", " ").replace("|", "\\|")


def _short_sha(value: Any) -> str:
    text = _string_value(value)
    return text[:12] if text else ""


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

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

from ai_testing.config import default_config_path, load_config
from ai_testing.run_tracking import StageExecutionTrackingConfig, record_stage_execution


def extract_config_path(argv: Sequence[str] | None) -> tuple[str, list[str]]:
    raw_args = list(sys.argv[1:] if argv is None else argv)
    config_path = default_config_path()
    cleaned_args: list[str] = []
    index = 0

    while index < len(raw_args):
        argument = raw_args[index]
        if argument == "--config":
            if index + 1 >= len(raw_args):
                raise SystemExit("--config requires a path value")
            config_path = raw_args[index + 1]
            index += 2
            continue
        if argument.startswith("--config="):
            config_path = argument.split("=", 1)[1]
            index += 1
            continue

        cleaned_args.append(argument)
        index += 1

    return config_path, cleaned_args


def load_config_for_cli(config_path: str) -> dict[str, object]:
    try:
        return load_config(config_path)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        raise SystemExit(f"Cannot load config {config_path}: {error}") from error


def read_records(input_path: Path) -> list[dict[str, object]]:
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    if not isinstance(payload, list) or not all(isinstance(item, dict) for item in payload):
        raise SystemExit(f"{input_path} must contain a JSON array of objects.")
    return [dict(item) for item in payload]


def read_json_object(input_path: Path) -> dict[str, object]:
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise SystemExit(f"{input_path} must contain a JSON object.")
    return dict(payload)


def write_json(payload: object, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def record_stage_result(
    args: argparse.Namespace,
    *,
    stage_name: str,
    artifacts: Mapping[str, Path],
    payload: Mapping[str, object] | None = None,
    status: str | None = None,
    parameters: Mapping[str, object] | None = None,
) -> None:
    if not bool(getattr(args, "stage_tracking_enabled", True)):
        return
    record_stage_execution(
        StageExecutionTrackingConfig(
            history_output=Path(
                str(getattr(args, "stage_history_output", "data/runs/stage_history.json"))
            ),
            markdown_output=Path(
                str(
                    getattr(
                        args,
                        "stage_history_markdown_output",
                        "data/reports/stage_history.md",
                    )
                )
            ),
            metric_history_markdown_output=Path(
                str(
                    getattr(
                        args,
                        "stage_metric_history_markdown_output",
                        "data/reports/stage_metric_history.md",
                    )
                )
            ),
            max_entries=int(getattr(args, "stage_history_max_entries", 500)),
            recent_entries_per_stage=int(
                getattr(args, "stage_history_recent_entries_per_stage", 5)
            ),
            metric_history_recent_entries_per_stage=int(
                getattr(args, "stage_metric_history_recent_entries_per_stage", 10)
            ),
        ),
        stage_name=stage_name,
        command_name=str(args.command),
        artifacts=artifacts,
        payload=payload,
        status=status,
        parameters=parameters,
    )


def print_report_summary(
    report: Mapping[str, object],
    output_path: Path,
    extra: Mapping[str, object] | None = None,
) -> None:
    summary = mapping(report.get("summary"))
    payload: dict[str, object] = {
        "output": str(output_path),
        "status": report.get("status"),
        "check_count": summary.get("check_count"),
        "passed_count": summary.get("passed_count"),
        "failed_count": summary.get("failed_count"),
    }
    if extra:
        payload.update(dict(extra))
    failed_checks = _failed_checks_for_cli(report)
    if failed_checks:
        payload["failed_checks"] = failed_checks
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def path_mapping(value: object) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    for name, path in mapping(value).items():
        if not isinstance(name, str) or not isinstance(path, (str, Path)):
            continue
        paths[name] = Path(path)
    return paths


def mapping(value: object) -> Mapping[str, object]:
    return value if isinstance(value, Mapping) else {}


def exit_if_report_failed(report: Mapping[str, object]) -> None:
    if report.get("status") == "failed":
        raise SystemExit(1)


def resolve_cookies(cookies: str | None, cookies_file: str | None, env_name: str) -> str:
    if cookies:
        return cookies

    if cookies_file:
        file_value = _read_cookies_file(Path(cookies_file))
        if file_value:
            return file_value

    env_value = os.getenv(env_name)
    if env_value:
        return env_value

    raise SystemExit(
        "Missing cookies. Put an authenticated Cookie header into "
        f"{cookies_file}, set {env_name}, or pass --cookies."
    )


def _failed_checks_for_cli(report: Mapping[str, object]) -> list[dict[str, object]]:
    checks = report.get("checks")
    if not isinstance(checks, Sequence) or isinstance(checks, (str, bytes)):
        return []

    failed_checks: list[dict[str, object]] = []
    for check in checks:
        if not isinstance(check, Mapping) or check.get("status") != "failed":
            continue
        failed_checks.append(
            {
                "id": check.get("id"),
                "severity": check.get("severity"),
                "message": check.get("message"),
                "observed": check.get("observed"),
                "expected": check.get("expected"),
                "diagnostics": _diagnostics_for_cli(check.get("diagnostics")),
            }
        )
    return failed_checks


def _diagnostics_for_cli(value: object) -> object:
    if not isinstance(value, Mapping):
        return None

    diagnostics: dict[str, object] = {}
    for key in (
        "summary",
        "failure_columns",
        "contract",
        "suggested_actions",
        "failed_child_checks",
    ):
        if key in value:
            diagnostics[key] = value[key]

    fields = value.get("fields")
    if isinstance(fields, Mapping):
        diagnostics["fields"] = {
            str(field): _diagnostic_field_for_cli(field_diagnostics)
            for field, field_diagnostics in fields.items()
        }

    sample_records = value.get("sample_records")
    if isinstance(sample_records, Sequence) and not isinstance(sample_records, (str, bytes)):
        diagnostics["sample_records"] = [item for item in sample_records[:3]]

    return diagnostics or dict(value)


def _diagnostic_field_for_cli(value: object) -> object:
    if not isinstance(value, Mapping):
        return value

    field_summary: dict[str, object] = {}
    for key in ("rule", "affected_record_count", "value_summary", "breakdown"):
        if key in value:
            field_summary[key] = value[key]
    sample_records = value.get("sample_records")
    if isinstance(sample_records, Sequence) and not isinstance(sample_records, (str, bytes)):
        field_summary["sample_records"] = [item for item in sample_records[:3]]
    return field_summary


def _read_cookies_file(path: Path) -> str | None:
    if not path.exists():
        return None

    lines = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    return " ".join(lines) if lines else None

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

Record = dict[str, Any]


@dataclass(frozen=True)
class RunMarkdownReportConfig:
    max_metric_deltas: int = 15
    max_drift_distributions: int = 5
    max_artifacts: int = 12


def build_run_markdown_report(
    run_report: Mapping[str, Any],
    *,
    drift_report: Mapping[str, Any] | None = None,
    config: RunMarkdownReportConfig | None = None,
) -> str:
    report_config = config or RunMarkdownReportConfig()
    lines: list[str] = []
    _add_title(lines, run_report)
    _add_summary(lines, run_report, drift_report)
    _add_quality(lines, run_report, drift_report)
    _add_metric_deltas(lines, run_report, report_config.max_metric_deltas)
    _add_drift(lines, run_report, report_config.max_drift_distributions)
    _add_versions(lines, run_report, report_config.max_artifacts)
    _add_stage_reports(lines, run_report)
    lines.append("")
    return "\n".join(lines)


def _add_title(lines: list[str], run_report: Mapping[str, Any]) -> None:
    run_id = _text(run_report.get("run_id"), "unknown-run")
    run_name = _text(run_report.get("run_name"), "")
    title = f"# AI Testing Run Report: {run_id}"
    lines.append(title)
    if run_name:
        lines.append("")
        lines.append(f"Run name: `{run_name}`")
    lines.append("")
    lines.append(f"Created at: `{_text(run_report.get('created_at'), 'unknown')}`")


def _add_summary(
    lines: list[str],
    run_report: Mapping[str, Any],
    drift_report: Mapping[str, Any] | None,
) -> None:
    summary = _mapping(run_report.get("summary"))
    comparison = _mapping(run_report.get("comparison"))
    delta_summary = _mapping(comparison.get("metric_delta_summary"))
    drift_status = _text(_mapping(drift_report).get("status"), "not_available")
    lines.extend(
        [
            "",
            "## Summary",
            "",
            "| Area | Value |",
            "|---|---:|",
            f"| Quality status | `{_text(summary.get('quality_status'), 'unknown')}` |",
            f"| Drift status | `{drift_status}` |",
            f"| Comparison status | `{_text(comparison.get('status'), 'unknown')}` |",
            f"| Baseline run | `{_text(comparison.get('baseline_run_id'), 'none')}` |",
            f"| Metrics tracked | {summary.get('metric_count', 0)} |",
            f"| Improved metrics | {delta_summary.get('improved_metric_count', 0)} |",
            f"| Regressed metrics | {delta_summary.get('regressed_metric_count', 0)} |",
            f"| Stage reports available | {summary.get('available_stage_report_count', 0)} |",
            f"| Artifacts available | {summary.get('available_artifact_count', 0)} |",
        ]
    )


def _add_quality(
    lines: list[str],
    run_report: Mapping[str, Any],
    drift_report: Mapping[str, Any] | None,
) -> None:
    quality_summary = _mapping(run_report.get("quality_summary"))
    quality_reports = _mapping(quality_summary.get("reports"))
    lines.extend(
        [
            "",
            "## Quality Gates",
            "",
            "| Report | Status | Failed Checks |",
            "|---|---:|---:|",
        ]
    )
    for name, report in sorted(quality_reports.items()):
        if not isinstance(report, Mapping):
            continue
        lines.append(
            f"| `{name}` | `{_text(report.get('status'), 'unknown')}` | "
            f"{report.get('failed_count', 0)} |"
        )

    if drift_report is not None:
        summary = _mapping(drift_report.get("summary"))
        lines.append(
            f"| `drift_testing` | `{_text(drift_report.get('status'), 'unknown')}` | "
            f"{summary.get('failed_count', 0)} |"
        )


def _add_metric_deltas(
    lines: list[str],
    run_report: Mapping[str, Any],
    limit: int,
) -> None:
    comparison = _mapping(run_report.get("comparison"))
    deltas = _list_of_mappings(comparison.get("metric_deltas"))
    interesting = [
        delta
        for delta in deltas
        if _text(delta.get("change"), "unchanged") in {"improved", "regressed", "changed"}
    ][: max(limit, 0)]
    lines.extend(["", "## Metric Changes", ""])
    if not interesting:
        lines.append("No metric changes were detected against the baseline.")
        return

    lines.extend(
        [
            "| Metric | Previous | Current | Delta | Change |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for delta in interesting:
        lines.append(
            f"| `{_text(delta.get('name'), '')}` | {_number(delta.get('previous'))} | "
            f"{_number(delta.get('current'))} | {_number(delta.get('delta'))} | "
            f"`{_text(delta.get('change'), '')}` |"
        )


def _add_drift(
    lines: list[str],
    run_report: Mapping[str, Any],
    limit: int,
) -> None:
    comparison = _mapping(run_report.get("comparison"))
    drift = _mapping(comparison.get("drift"))
    distributions = _mapping(drift.get("distributions"))
    lines.extend(["", "## Drift", ""])
    lines.append(
        f"Max total variation distance: `{_number(drift.get('max_total_variation_distance'))}`"
    )
    if not distributions:
        lines.append("")
        lines.append("No distribution drift details are available.")
        return

    lines.extend(
        [
            "",
            "| Distribution | Total Variation Distance | Largest Change |",
            "|---|---:|---|",
        ]
    )
    sorted_distributions = sorted(
        distributions.items(),
        key=lambda item: _float_value(_mapping(item[1]).get("total_variation_distance")) or 0.0,
        reverse=True,
    )
    for name, report in sorted_distributions[: max(limit, 0)]:
        if not isinstance(report, Mapping):
            continue
        top_changes = _list_of_mappings(report.get("top_changes"))
        largest = top_changes[0] if top_changes else {}
        largest_change = (
            f"`{_text(largest.get('value'), '')}` ({_number(largest.get('delta'))})"
            if largest
            else ""
        )
        lines.append(
            f"| `{name}` | {_number(report.get('total_variation_distance'))} | {largest_change} |"
        )


def _add_versions(
    lines: list[str],
    run_report: Mapping[str, Any],
    limit: int,
) -> None:
    git = _mapping(run_report.get("git"))
    artifacts = _mapping(run_report.get("artifacts"))
    lines.extend(
        [
            "",
            "## Versions",
            "",
            f"Git branch: `{_text(git.get('branch'), 'unknown')}`",
            f"Git commit: `{_text(git.get('commit'), 'unknown')}`",
            f"Dirty worktree: `{_text(git.get('is_dirty'), 'unknown')}`",
            "",
            "| Artifact | Type | Exists | SHA256 |",
            "|---|---:|---:|---|",
        ]
    )
    for name, artifact in sorted(artifacts.items())[: max(limit, 0)]:
        if not isinstance(artifact, Mapping):
            continue
        lines.append(
            f"| `{name}` | `{_text(artifact.get('type'), '')}` | "
            f"`{_text(artifact.get('exists'), '')}` | `{_short_sha(artifact.get('sha256'))}` |"
        )


def _add_stage_reports(lines: list[str], run_report: Mapping[str, Any]) -> None:
    stage_reports = _mapping(run_report.get("stage_reports"))
    lines.extend(
        [
            "",
            "## Stage Reports",
            "",
            "| Stage | Exists | Status | Report Type |",
            "|---|---:|---:|---|",
        ]
    )
    for name, report in sorted(stage_reports.items()):
        if not isinstance(report, Mapping):
            continue
        lines.append(
            f"| `{name}` | `{_text(report.get('exists'), '')}` | "
            f"`{_text(report.get('status'), '')}` | `{_text(report.get('report_type'), '')}` |"
        )


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


def _number(value: Any) -> str:
    number = _float_value(value)
    if number is None:
        return _text(value, "")
    return f"{number:.6g}"


def _short_sha(value: Any) -> str:
    text = _text(value, "")
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

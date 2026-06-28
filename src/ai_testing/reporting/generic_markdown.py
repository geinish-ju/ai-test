from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

Record = dict[str, Any]


@dataclass(frozen=True)
class MarkdownReportConfig:
    max_checks: int = 100
    max_rows: int = 30
    max_text_length: int = 180


def build_markdown_report(
    report: Mapping[str, Any],
    *,
    title: str | None = None,
    source_path: str | Path | None = None,
    config: MarkdownReportConfig | None = None,
) -> str:
    report_config = config or MarkdownReportConfig()
    lines: list[str] = []
    _add_title(lines, report, title=title, source_path=source_path)
    _add_overview(lines, report)
    _add_quality_decision(lines, report, report_config)
    _add_summary(lines, report)
    _add_metrics(lines, report)
    _add_project_quality_matrix(lines, report)
    _add_test_results(lines, report)
    _add_checks(lines, report, report_config)
    _add_per_class(lines, report, report_config)
    _add_folds(lines, report, report_config)
    _add_rules(lines, report, report_config)
    _add_explainability(lines, report, report_config)
    _add_llm_charters(lines, report, report_config)
    _add_inputs(lines, report)
    _add_artifacts(lines, report, report_config)
    lines.append("")
    return "\n".join(lines)


def _add_title(
    lines: list[str],
    report: Mapping[str, Any],
    *,
    title: str | None,
    source_path: str | Path | None,
) -> None:
    resolved_title = title or _report_title(report)
    lines.append(f"# {resolved_title}")
    if source_path is not None:
        lines.extend(["", f"Source JSON: `{source_path}`"])


def _add_overview(lines: list[str], report: Mapping[str, Any]) -> None:
    overview = {
        "Report type": report.get("report_type"),
        "Subject": report.get("subject"),
        "Step": report.get("step"),
        "Testing type": report.get("testing_type") or report.get("validation_type"),
        "Status": report.get("status"),
        "Model type": report.get("model_type"),
        "Algorithm": report.get("algorithm"),
        "Learning type": report.get("learning_type"),
        "Run id": report.get("run_id"),
        "Baseline run id": report.get("baseline_run_id"),
    }
    _add_key_value_table(lines, "Overview", overview)


def _add_summary(lines: list[str], report: Mapping[str, Any]) -> None:
    summary = _mapping(report.get("summary"))
    if summary:
        _add_key_value_table(lines, "Summary", summary)
        return

    summary_candidates = {
        key: value
        for key, value in report.items()
        if key.endswith("_count") or key in {"status", "accuracy", "macro_f1", "weighted_f1"}
    }
    _add_key_value_table(lines, "Summary", summary_candidates)


def _add_quality_decision(
    lines: list[str],
    report: Mapping[str, Any],
    config: MarkdownReportConfig,
) -> None:
    decision = _mapping(report.get("decision"))
    if not decision:
        return

    lines.extend(
        [
            "",
            "## Quality Decision",
            "",
            "| Field | Value |",
            "|---|---|",
            f"| Outcome | `{_escape(decision.get('outcome'))}` |",
            f"| Recommendation | {_cell(decision.get('recommendation'))} |",
            f"| Blockers | {_cell(decision.get('blocker_count'))} |",
            f"| Warnings | {_cell(decision.get('warning_count'))} |",
        ]
    )
    rationale = _sequence(decision.get("rationale"))
    if rationale:
        lines.extend(["", "### Rationale", ""])
        for item in rationale[: max(config.max_rows, 0)]:
            lines.append(f"- {_cell(item)}")

    _add_issue_table(lines, "Blockers", _list_of_mappings(report.get("blockers")), config)
    _add_issue_table(lines, "Warnings", _list_of_mappings(report.get("warnings")), config)

    actions = _sequence(report.get("recommended_actions") or decision.get("recommended_actions"))
    if actions:
        lines.extend(["", "### Recommended Actions", ""])
        for action in actions[: max(config.max_rows, 0)]:
            lines.append(f"- {_cell(action)}")


def _add_issue_table(
    lines: list[str],
    heading: str,
    issues: Sequence[Mapping[str, Any]],
    config: MarkdownReportConfig,
) -> None:
    if not issues:
        return
    lines.extend(
        [
            "",
            f"### {heading}",
            "",
            "| Report | Check | Severity | Finding | Action |",
            "|---|---|---:|---|---|",
        ]
    )
    for issue in issues[: max(config.max_rows, 0)]:
        lines.append(
            f"| `{_escape(issue.get('report'))}` | `{_escape(issue.get('check_id'))}` | "
            f"`{_escape(issue.get('severity'))}` | {_cell(issue.get('message'))} | "
            f"{_cell(issue.get('recommended_action'))} |"
        )


def _add_metrics(lines: list[str], report: Mapping[str, Any]) -> None:
    metrics = _mapping(report.get("metrics"))
    if not metrics:
        return
    lines.extend(["", "## Metrics", "", "| Metric | Value |", "|---|---:|"])
    for key, value in sorted(metrics.items()):
        if isinstance(value, Mapping) and "value" in value:
            value = value.get("value")
        lines.append(f"| `{_escape(key)}` | {_cell(value)} |")


def _add_project_quality_matrix(lines: list[str], report: Mapping[str, Any]) -> None:
    reports = _mapping(report.get("reports"))
    if not reports:
        return
    lines.extend(
        [
            "",
            "## Child Quality Reports",
            "",
            "| Report | Status | Type | Subject | Checks | Failed |",
            "|---|---:|---|---|---:|---:|",
        ]
    )
    for name, child_report in sorted(reports.items()):
        child = _mapping(child_report)
        lines.append(
            f"| `{_escape(name)}` | `{_escape(child.get('status'))}` | "
            f"`{_escape(child.get('report_type'))}` | `{_escape(child.get('subject'))}` | "
            f"{_cell(child.get('check_count'))} | {_cell(child.get('failed_count'))} |"
        )


def _add_test_results(lines: list[str], report: Mapping[str, Any]) -> None:
    for section_name in ("test", "validation", "train"):
        section = _mapping(report.get(section_name))
        if not section:
            continue
        values = {
            key: value for key, value in section.items() if _scalar(value) and _result_field(key)
        }
        _add_key_value_table(lines, f"{section_name.title()} Results", values)


def _add_checks(
    lines: list[str],
    report: Mapping[str, Any],
    config: MarkdownReportConfig,
) -> None:
    checks = _list_of_mappings(report.get("checks"))
    if not checks:
        lines.extend(
            [
                "",
                "## Checks",
                "",
                "This report does not contain formal pass/fail checks.",
            ]
        )
        return

    lines.extend(
        [
            "",
            "## Checks",
            "",
            "| ID | Status | Severity | What Was Checked | Observed | Expected |",
            "|---|---:|---:|---|---|---|",
        ]
    )
    for check in checks[: max(config.max_checks, 0)]:
        lines.append(
            f"| `{_escape(check.get('id'))}` | `{_escape(check.get('status'))}` | "
            f"`{_escape(check.get('severity'))}` | {_cell(check.get('message'))} | "
            f"{_cell(check.get('observed'))} | {_cell(check.get('expected'))} |"
        )

    failed_checks = [check for check in checks if check.get("status") == "failed"]
    if failed_checks:
        lines.extend(
            [
                "",
                "### Failed Check Diagnostics",
                "",
                "| ID | Diagnostics |",
                "|---|---|",
            ]
        )
        for check in failed_checks[: max(config.max_rows, 0)]:
            lines.append(f"| `{_escape(check.get('id'))}` | {_cell(check.get('diagnostics'))} |")


def _add_per_class(
    lines: list[str],
    report: Mapping[str, Any],
    config: MarkdownReportConfig,
) -> None:
    for section_name in ("test", "validation"):
        per_class = _mapping(_mapping(report.get(section_name)).get("per_class"))
        if not per_class:
            continue
        lines.extend(
            [
                "",
                f"## {section_name.title()} Per-Class Results",
                "",
                "| Class | Precision | Recall | F1 | Support |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        for label, metrics in list(sorted(per_class.items()))[: max(config.max_rows, 0)]:
            metric_map = _mapping(metrics)
            lines.append(
                f"| `{_escape(label)}` | {_cell(metric_map.get('precision'))} | "
                f"{_cell(metric_map.get('recall'))} | {_cell(metric_map.get('f1'))} | "
                f"{_cell(metric_map.get('support'))} |"
            )


def _add_folds(
    lines: list[str],
    report: Mapping[str, Any],
    config: MarkdownReportConfig,
) -> None:
    folds = _list_of_mappings(report.get("folds"))
    if not folds:
        return
    lines.extend(
        [
            "",
            "## Fold Results",
            "",
            "| Fold | Train Rows | Validation Rows | Accuracy | Macro F1 | Confidence | Lift |",
            "|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for fold in folds[: max(config.max_rows, 0)]:
        train = _mapping(fold.get("train"))
        validation = _mapping(fold.get("validation"))
        lines.append(
            f"| {_cell(fold.get('fold_index'))} | {_cell(train.get('record_count'))} | "
            f"{_cell(validation.get('record_count'))} | {_cell(validation.get('accuracy'))} | "
            f"{_cell(validation.get('macro_f1'))} | "
            f"{_cell(validation.get('mean_validation_confidence'))} | "
            f"{_cell(validation.get('mean_validation_lift'))} |"
        )


def _add_rules(
    lines: list[str],
    report: Mapping[str, Any],
    config: MarkdownReportConfig,
) -> None:
    rules = _list_of_mappings(report.get("top_rules"))
    if not rules:
        return
    lines.extend(
        [
            "",
            "## Top Association Rules",
            "",
            "| Antecedent | Consequent | Confidence | Lift | Support |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for rule in rules[: max(config.max_rows, 0)]:
        lines.append(
            f"| {_cell(rule.get('antecedent'))} | {_cell(rule.get('consequent'))} | "
            f"{_cell(rule.get('test_confidence') or rule.get('validation_confidence'))} | "
            f"{_cell(rule.get('test_lift') or rule.get('validation_lift'))} | "
            f"{_cell(rule.get('test_support') or rule.get('validation_support'))} |"
        )


def _add_explainability(
    lines: list[str],
    report: Mapping[str, Any],
    config: MarkdownReportConfig,
) -> None:
    classes = _mapping(report.get("classes"))
    if not classes:
        return
    lines.extend(
        [
            "",
            "## Class Explanations",
            "",
            "| Class | Top Tokens |",
            "|---|---|",
        ]
    )
    for label, payload in list(sorted(classes.items()))[: max(config.max_rows, 0)]:
        top_tokens = _list_of_mappings(_mapping(payload).get("top_tokens"))
        tokens = ", ".join(f"`{_escape(token.get('token'))}`" for token in top_tokens[:10])
        lines.append(f"| `{_escape(label)}` | {tokens} |")


def _add_llm_charters(
    lines: list[str],
    report: Mapping[str, Any],
    config: MarkdownReportConfig,
) -> None:
    charters = _list_of_mappings(report.get("charters"))
    if not charters:
        return
    lines.extend(
        [
            "",
            "## Exploratory Test Charters",
            "",
            "| ID | Title | Risks | Mission |",
            "|---|---|---|---|",
        ]
    )
    for charter in charters[: max(config.max_rows, 0)]:
        lines.append(
            f"| `{_escape(charter.get('id'))}` | {_cell(charter.get('title'))} | "
            f"{_cell(charter.get('risks'))} | {_cell(charter.get('mission'))} |"
        )


def _add_inputs(lines: list[str], report: Mapping[str, Any]) -> None:
    inputs = _mapping(report.get("inputs") or report.get("source_inputs"))
    if not inputs:
        return
    lines.extend(["", "## Inputs", "", "| Input | Path |", "|---|---|"])
    for name, path in sorted(inputs.items()):
        lines.append(f"| `{_escape(name)}` | `{_escape(path)}` |")


def _add_artifacts(
    lines: list[str],
    report: Mapping[str, Any],
    config: MarkdownReportConfig,
) -> None:
    artifacts = _mapping(report.get("artifacts") or report.get("outputs"))
    if not artifacts:
        return
    lines.extend(
        [
            "",
            "## Artifacts",
            "",
            "| Artifact | Type | Exists | Path | SHA256 |",
            "|---|---:|---:|---|---|",
        ]
    )
    for name, artifact in list(sorted(artifacts.items()))[: max(config.max_rows, 0)]:
        artifact_map = _mapping(artifact)
        if artifact_map:
            lines.append(
                f"| `{_escape(name)}` | `{_escape(artifact_map.get('type'))}` | "
                f"`{_escape(artifact_map.get('exists'))}` | "
                f"`{_escape(artifact_map.get('path'))}` | "
                f"`{_short_sha(artifact_map.get('sha256'))}` |"
            )
            continue
        lines.append(f"| `{_escape(name)}` |  |  | `{_escape(artifact)}` |  |")


def _add_key_value_table(
    lines: list[str],
    heading: str,
    values: Mapping[str, Any],
) -> None:
    compact_values = {
        key: value
        for key, value in values.items()
        if value is not None and value not in ({}, [], "")
    }
    if not compact_values:
        return
    lines.extend(["", f"## {heading}", "", "| Field | Value |", "|---|---:|"])
    for key, value in compact_values.items():
        lines.append(f"| `{_escape(key)}` | {_cell(value)} |")


def _report_title(report: Mapping[str, Any]) -> str:
    step = _string_value(report.get("step"))
    report_type = _string_value(report.get("report_type"))
    subject = _string_value(report.get("subject"))
    if step and subject:
        return f"{step}: {subject}"
    if step:
        return step
    if report_type and subject:
        return f"{report_type}: {subject}"
    return report_type or "AI Testing Report"


def _result_field(key: str) -> bool:
    return key.endswith("_count") or key in {
        "accuracy",
        "macro_precision",
        "macro_recall",
        "macro_f1",
        "weighted_precision",
        "weighted_recall",
        "weighted_f1",
        "mean_test_confidence",
        "mean_test_lift",
        "mean_validation_confidence",
        "mean_validation_lift",
        "antecedent_coverage",
        "hit_rate_per_covered_basket",
        "mean_abs_confidence_gap",
    }


def _cell(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return f"`{str(value).lower()}`"
    if isinstance(value, (int, float)):
        return f"{value:.6g}"
    if isinstance(value, str):
        return _escape(value)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return ", ".join(f"`{_escape(item)}`" for item in value[:8])
    return _escape(json.dumps(value, ensure_ascii=False, sort_keys=True))


def _escape(value: Any) -> str:
    text = "" if value is None else str(value)
    text = text.replace("\n", " ").replace("|", "\\|")
    return text if len(text) <= 180 else f"{text[:177]}..."


def _short_sha(value: Any) -> str:
    text = _string_value(value)
    return text[:12] if text else ""


def _scalar(value: Any) -> bool:
    return value is None or isinstance(value, (str, int, float, bool))


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _list_of_mappings(value: Any) -> list[Record]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return []
    return [dict(item) for item in value if isinstance(item, Mapping)]


def _sequence(value: Any) -> list[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return []
    return list(value)


def _string_value(value: Any) -> str | None:
    if value is None or isinstance(value, bool):
        return None
    text = str(value).strip()
    return text or None

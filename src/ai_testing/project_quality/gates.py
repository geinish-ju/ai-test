from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from ai_testing.core import add_check as _add_check
from ai_testing.core import build_standard_report

Record = dict[str, Any]


@dataclass(frozen=True)
class ProjectQualityResult:
    report: Record


def aggregate_quality_reports(
    reports: Mapping[str, Mapping[str, Any]],
) -> ProjectQualityResult:
    checks: list[Record] = []
    report_summaries: Record = {}
    failed_reports: list[str] = []
    total_child_checks = 0
    total_child_failures = 0

    for report_name, report in sorted(reports.items()):
        status = _string_value(report.get("status"))
        summary = _mapping(report.get("summary"))
        check_count = _int_value(summary.get("check_count"))
        failed_count = _int_value(summary.get("failed_count"))
        total_child_checks += check_count
        total_child_failures += failed_count
        if status != "passed":
            failed_reports.append(report_name)

        report_summaries[report_name] = {
            "status": status,
            "report_type": report.get("report_type"),
            "subject": report.get("subject"),
            "check_count": check_count,
            "failed_count": failed_count,
        }
        _add_check(
            checks,
            check_id=f"project_quality.{_safe_check_id(report_name)}",
            passed=status == "passed" and failed_count == 0,
            severity="critical",
            message=f"{report_name} quality report is passed.",
            observed={"status": status, "failed_count": failed_count},
            expected={"status": "passed", "failed_count": 0},
            diagnostics=_failed_report_diagnostics(report),
        )

    metrics = {
        "child_report_count": len(report_summaries),
        "passed_report_count": len(report_summaries) - len(failed_reports),
        "failed_report_count": len(failed_reports),
        "total_child_check_count": total_child_checks,
        "total_child_failed_check_count": total_child_failures,
    }

    return ProjectQualityResult(
        report=build_standard_report(
            report_type="project_quality_gates",
            subject="ai_grocery_testing_project",
            step="Project quality gates",
            testing_type="release acceptance quality gates",
            checks=checks,
            metrics=metrics,
            details={
                "reports": report_summaries,
                "failed_reports": failed_reports,
            },
        )
    )


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _failed_report_diagnostics(report: Mapping[str, Any]) -> Record:
    if report.get("status") == "passed":
        return {}

    return {
        "summary": "Child quality report failed. Inspect failed_child_checks for root causes.",
        "failed_child_checks": _failed_child_checks(report),
    }


def _failed_child_checks(report: Mapping[str, Any]) -> list[Record]:
    checks = report.get("checks")
    if not isinstance(checks, list):
        return []

    failed: list[Record] = []
    for check in checks:
        if not isinstance(check, Mapping) or check.get("status") != "failed":
            continue
        failed.append(
            {
                "id": check.get("id"),
                "severity": check.get("severity"),
                "message": check.get("message"),
                "observed": check.get("observed"),
                "expected": check.get("expected"),
                "diagnostics": _compact_diagnostics(check.get("diagnostics")),
            }
        )
    return failed


def _compact_diagnostics(value: Any) -> Any:
    if not isinstance(value, Mapping):
        return None

    compact: Record = {}
    for key in ("summary", "failure_columns", "contract", "suggested_actions"):
        if key in value:
            compact[key] = value[key]
    fields = value.get("fields")
    if isinstance(fields, Mapping):
        compact["fields"] = {
            str(field): _compact_field_diagnostics(field_diagnostics)
            for field, field_diagnostics in fields.items()
        }
    sample_records = value.get("sample_records")
    if isinstance(sample_records, list):
        compact["sample_records"] = sample_records[:3]
    return compact or dict(value)


def _compact_field_diagnostics(value: Any) -> Any:
    if not isinstance(value, Mapping):
        return value

    compact: Record = {}
    for key in ("rule", "affected_record_count", "value_summary", "breakdown"):
        if key in value:
            compact[key] = value[key]
    sample_records = value.get("sample_records")
    if isinstance(sample_records, list):
        compact["sample_records"] = sample_records[:3]
    return compact or dict(value)


def _int_value(value: Any) -> int:
    if isinstance(value, bool):
        return 0
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        try:
            return int(float(value.strip()))
        except ValueError:
            return 0
    return 0


def _string_value(value: Any) -> str | None:
    if value is None or isinstance(value, bool):
        return None
    text = str(value).strip()
    return text or None


def _safe_check_id(value: str) -> str:
    return "".join(char if char.isalnum() else "_" for char in value.lower()).strip("_")

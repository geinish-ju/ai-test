from __future__ import annotations

from collections.abc import Mapping, Sequence
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
    blockers: list[Record] = []
    warnings: list[Record] = []
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
            report_issues = _quality_issues(report_name, report)
            blockers.extend(issue for issue in report_issues if issue["severity"] == "critical")
            warnings.extend(issue for issue in report_issues if issue["severity"] != "critical")

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
        "blocker_count": len(blockers),
        "warning_count": len(warnings),
    }
    decision = _quality_decision(
        failed_reports=failed_reports,
        blockers=blockers,
        warnings=warnings,
        metrics=metrics,
    )

    return ProjectQualityResult(
        report=build_standard_report(
            report_type="project_quality_gates",
            subject="ai_grocery_testing_project",
            step="Project quality gates",
            testing_type="release acceptance quality gates",
            checks=checks,
            metrics=metrics,
            details={
                "decision": decision,
                "blockers": blockers,
                "warnings": warnings,
                "recommended_actions": decision["recommended_actions"],
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


def _quality_issues(report_name: str, report: Mapping[str, Any]) -> list[Record]:
    failed_checks = _failed_child_checks(report)
    if not failed_checks:
        return [
            {
                "report": report_name,
                "check_id": None,
                "severity": "critical",
                "message": f"{report_name} quality report did not pass.",
                "observed": {
                    "status": report.get("status"),
                    "failed_count": _int_value(_mapping(report.get("summary")).get("failed_count")),
                },
                "expected": {"status": "passed", "failed_count": 0},
                "recommended_action": _default_action(report_name),
            }
        ]

    return [_quality_issue(report_name, check) for check in failed_checks]


def _quality_issue(report_name: str, check: Mapping[str, Any]) -> Record:
    return {
        "report": report_name,
        "check_id": check.get("id"),
        "severity": _severity(check.get("severity")),
        "message": check.get("message"),
        "observed": check.get("observed"),
        "expected": check.get("expected"),
        "recommended_action": _recommended_action(report_name, check),
    }


def _quality_decision(
    *,
    failed_reports: list[str],
    blockers: Sequence[Mapping[str, Any]],
    warnings: Sequence[Mapping[str, Any]],
    metrics: Mapping[str, Any],
) -> Record:
    if blockers:
        outcome = "rejected"
        recommendation = "Do not accept the candidate model or data release."
        rationale = [
            "At least one critical quality gate failed.",
            (
                "Critical failures can indicate data leakage, invalid evaluation, "
                "missing evidence, or unsafe input data."
            ),
        ]
    elif warnings:
        outcome = "needs_review"
        recommendation = "Review non-critical failures before accepting the candidate."
        rationale = [
            "No critical blocker was found, but at least one major or minor check failed.",
            "Acceptance requires an explicit risk decision from the reviewer.",
        ]
    else:
        outcome = "accepted"
        recommendation = "Accept the candidate for the current learning project scope."
        rationale = [
            "All aggregated quality reports passed.",
            "Input data, model quality, and optional drift evidence meet configured gates.",
        ]

    return {
        "outcome": outcome,
        "recommendation": recommendation,
        "rationale": rationale,
        "failed_reports": failed_reports,
        "blocker_count": metrics.get("blocker_count", 0),
        "warning_count": metrics.get("warning_count", 0),
        "recommended_actions": _recommended_actions(blockers, warnings),
    }


def _recommended_actions(
    blockers: Sequence[Mapping[str, Any]],
    warnings: Sequence[Mapping[str, Any]],
) -> list[str]:
    issues = [*blockers, *warnings]
    actions: list[str] = []
    for issue in issues:
        action = _string_value(issue.get("recommended_action"))
        if action is not None and action not in actions:
            actions.append(action)
    if (
        issues
        and "Rerun the affected pipeline stage and regenerate project quality gates." not in actions
    ):
        actions.append("Rerun the affected pipeline stage and regenerate project quality gates.")
    if not issues:
        actions.append("Keep the report with the run evidence as the model acceptance record.")
    return actions


def _recommended_action(report_name: str, check: Mapping[str, Any]) -> str:
    diagnostics = _mapping(check.get("diagnostics"))
    suggested_actions = diagnostics.get("suggested_actions")
    if isinstance(suggested_actions, Sequence) and not isinstance(suggested_actions, (str, bytes)):
        for action in suggested_actions:
            action_text = _string_value(action)
            if action_text is not None:
                return action_text
    return _default_action(report_name)


def _default_action(report_name: str) -> str:
    if report_name == "input_data":
        return "Fix data acquisition or preprocessing, then rerun input data testing."
    if report_name == "category_model":
        return "Inspect classifier validation/test metrics, leakage checks, and class coverage."
    if report_name == "association_model":
        return "Inspect association rule stability, confidence, lift, and validation-test gaps."
    if report_name == "drift":
        return "Investigate metric regressions or distribution drift before accepting the run."
    return "Inspect the failed child report and fix the affected pipeline stage."


def _severity(value: Any) -> str:
    severity = _string_value(value)
    return severity if severity in {"critical", "major", "minor"} else "major"


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

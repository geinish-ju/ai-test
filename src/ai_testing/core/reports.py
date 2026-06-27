from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

Record = dict[str, Any]


def add_check(
    checks: list[Record],
    check_id: str,
    passed: bool,
    severity: str,
    message: str,
    observed: Any,
    expected: Any,
    diagnostics: Mapping[str, Any] | None = None,
) -> None:
    check: Record = {
        "id": check_id,
        "status": "passed" if passed else "failed",
        "severity": severity,
        "message": message,
        "observed": observed,
        "expected": expected,
    }
    if diagnostics:
        check["diagnostics"] = dict(diagnostics)
    checks.append(check)


def add_threshold_check(
    checks: list[Record],
    check_id: str,
    value: float | None,
    threshold: float,
    direction: str,
    severity: str,
    message: str,
) -> None:
    if direction not in {">=", "<="}:
        raise ValueError("direction must be either >= or <=")
    passed = value is not None and (value >= threshold if direction == ">=" else value <= threshold)
    add_check(
        checks,
        check_id=check_id,
        passed=passed,
        severity=severity,
        message=message,
        observed={"value": value},
        expected={direction: threshold},
    )


def build_standard_report(
    *,
    report_type: str,
    subject: str,
    step: str,
    checks: Sequence[Mapping[str, Any]],
    testing_type: str | None = None,
    metrics: Mapping[str, Any] | None = None,
    artifacts: Mapping[str, Any] | None = None,
    details: Mapping[str, Any] | None = None,
) -> Record:
    report: Record = {
        "report_type": report_type,
        "subject": subject,
        "step": step,
    }
    if testing_type is not None:
        report["testing_type"] = testing_type
    report.update(
        {
            "status": report_status(checks),
            "summary": summarize_checks(checks),
        }
    )
    if metrics is not None:
        report["metrics"] = dict(metrics)
    if artifacts is not None:
        report["artifacts"] = dict(artifacts)
    if details is not None:
        report.update(dict(details))
    report["checks"] = [dict(check) for check in checks]
    return report


def summarize_checks(checks: Sequence[Mapping[str, Any]]) -> Record:
    failed_count = sum(1 for check in checks if check.get("status") == "failed")
    passed_count = sum(1 for check in checks if check.get("status") == "passed")
    return {
        "check_count": len(checks),
        "passed_count": passed_count,
        "failed_count": failed_count,
    }


def report_status(checks: Sequence[Mapping[str, Any]]) -> str:
    return "failed" if any(check.get("status") == "failed" for check in checks) else "passed"

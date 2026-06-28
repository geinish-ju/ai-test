from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from ai_testing.core import add_check, build_standard_report

Record = dict[str, Any]


@dataclass(frozen=True)
class RunDriftTestConfig:
    require_baseline: bool = True
    max_regressed_metric_count: int = 0
    max_total_variation_distance: float = 0.1
    max_distribution_total_variation_distance: float = 0.15
    max_critical_metric_regression: float = 0.02
    critical_metrics: tuple[str, ...] = (
        "category.test.accuracy",
        "category.test.macro_f1",
        "category.test.weighted_f1",
        "association.test.confidence",
        "association.test.lift",
        "association.test.hit_rate",
        "quality.project.failed_report_count",
        "quality.project.total_child_failed_check_count",
    )


def test_run_drift(
    run_report: Mapping[str, Any],
    config: RunDriftTestConfig | None = None,
) -> Record:
    drift_config = config or RunDriftTestConfig()
    comparison = _mapping(run_report.get("comparison"))
    metric_delta_summary = _mapping(comparison.get("metric_delta_summary"))
    drift = _mapping(comparison.get("drift"))
    distribution_reports = _mapping(drift.get("distributions"))
    metric_deltas = _list_of_mappings(comparison.get("metric_deltas"))
    critical_regressions = _critical_metric_regressions(
        metric_deltas,
        critical_metrics=set(drift_config.critical_metrics),
        max_regression=drift_config.max_critical_metric_regression,
    )
    distribution_violations = _distribution_violations(
        distribution_reports,
        threshold=drift_config.max_distribution_total_variation_distance,
    )
    checks: list[Record] = []

    comparison_status = _string_value(comparison.get("status")) or "not_available"
    add_check(
        checks,
        check_id="drift.baseline_available",
        passed=(comparison_status == "compared" or not drift_config.require_baseline),
        severity="critical" if drift_config.require_baseline else "minor",
        message="Run has a baseline comparison for drift and regression evaluation.",
        observed={"comparison_status": comparison_status},
        expected={"comparison_status": "compared"},
    )

    regressed_metric_count = _int_value(metric_delta_summary.get("regressed_metric_count"))
    add_check(
        checks,
        check_id="drift.regressed_metric_count",
        passed=(
            comparison_status != "compared"
            or regressed_metric_count <= drift_config.max_regressed_metric_count
        ),
        severity="major",
        message="Number of regressed metrics is within the accepted limit.",
        observed={"regressed_metric_count": regressed_metric_count},
        expected={"<=": drift_config.max_regressed_metric_count},
        diagnostics={
            "regressed_metrics": comparison.get("regressed_metrics", []),
            "suggested_actions": [
                "Inspect metric_deltas in the run report.",
                "Check whether preprocessing, split, or model parameters changed.",
            ],
        },
    )

    max_total_variation_distance = _float_value(drift.get("max_total_variation_distance"))
    add_check(
        checks,
        check_id="drift.max_total_variation_distance",
        passed=(
            comparison_status != "compared"
            or (
                max_total_variation_distance is not None
                and max_total_variation_distance <= drift_config.max_total_variation_distance
            )
        ),
        severity="major",
        message="Maximum observed distribution drift is within the accepted threshold.",
        observed={"max_total_variation_distance": max_total_variation_distance},
        expected={"<=": drift_config.max_total_variation_distance},
        diagnostics={"distribution_count": drift.get("distribution_count")},
    )

    add_check(
        checks,
        check_id="drift.distribution_thresholds",
        passed=comparison_status != "compared" or not distribution_violations,
        severity="major",
        message="Each monitored distribution stays within the accepted drift threshold.",
        observed={"violating_distribution_count": len(distribution_violations)},
        expected={"violating_distribution_count": 0},
        diagnostics={"violations": distribution_violations},
    )

    add_check(
        checks,
        check_id="drift.critical_metric_regressions",
        passed=comparison_status != "compared" or not critical_regressions,
        severity="critical",
        message="Critical model and quality metrics did not regress beyond tolerance.",
        observed={"critical_regression_count": len(critical_regressions)},
        expected={"critical_regression_count": 0},
        diagnostics={
            "critical_metrics": list(drift_config.critical_metrics),
            "violations": critical_regressions,
        },
    )

    metrics = {
        "comparison_status": comparison_status,
        "baseline_run_id": comparison.get("baseline_run_id"),
        "regressed_metric_count": regressed_metric_count,
        "improved_metric_count": _int_value(metric_delta_summary.get("improved_metric_count")),
        "max_total_variation_distance": max_total_variation_distance,
        "violating_distribution_count": len(distribution_violations),
        "critical_regression_count": len(critical_regressions),
    }

    return build_standard_report(
        report_type="drift_quality_test",
        subject="pipeline_run",
        step="Drift testing",
        testing_type="run-to-baseline drift and metric regression checks",
        checks=checks,
        metrics=metrics,
        details={
            "run_id": run_report.get("run_id"),
            "baseline_run_id": comparison.get("baseline_run_id"),
            "thresholds": {
                "require_baseline": drift_config.require_baseline,
                "max_regressed_metric_count": drift_config.max_regressed_metric_count,
                "max_total_variation_distance": drift_config.max_total_variation_distance,
                "max_distribution_total_variation_distance": (
                    drift_config.max_distribution_total_variation_distance
                ),
                "max_critical_metric_regression": drift_config.max_critical_metric_regression,
            },
        },
    )


def _critical_metric_regressions(
    metric_deltas: Sequence[Mapping[str, Any]],
    *,
    critical_metrics: set[str],
    max_regression: float,
) -> list[Record]:
    violations: list[Record] = []
    for metric_delta in metric_deltas:
        name = _string_value(metric_delta.get("name"))
        if name is None or name not in critical_metrics:
            continue
        regression = _regression_magnitude(metric_delta)
        if regression is None or regression <= max_regression:
            continue
        violations.append(
            {
                "name": name,
                "previous": metric_delta.get("previous"),
                "current": metric_delta.get("current"),
                "delta": metric_delta.get("delta"),
                "direction": metric_delta.get("direction"),
                "regression": round(regression, 6),
                "threshold": max_regression,
            }
        )
    return violations


def _distribution_violations(
    distribution_reports: Mapping[str, Any],
    *,
    threshold: float,
) -> list[Record]:
    violations: list[Record] = []
    for name, report in sorted(distribution_reports.items()):
        if not isinstance(report, Mapping):
            continue
        distance = _float_value(report.get("total_variation_distance"))
        if distance is None or distance <= threshold:
            continue
        violations.append(
            {
                "name": str(name),
                "total_variation_distance": distance,
                "threshold": threshold,
                "top_changes": report.get("top_changes", []),
            }
        )
    return violations


def _regression_magnitude(metric_delta: Mapping[str, Any]) -> float | None:
    delta = _float_value(metric_delta.get("delta"))
    direction = _string_value(metric_delta.get("direction")) or "neutral"
    if delta is None:
        return None
    if direction == "higher_is_better" and delta < 0:
        return abs(delta)
    if direction == "lower_is_better" and delta > 0:
        return delta
    return 0.0


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _list_of_mappings(value: Any) -> list[Record]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return []
    return [dict(item) for item in value if isinstance(item, Mapping)]


def _string_value(value: Any) -> str | None:
    if value is None or isinstance(value, bool):
        return None
    text = str(value).strip()
    return text or None


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

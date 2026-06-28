from __future__ import annotations

import json
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ai_testing.core import artifact_metadata

Record = dict[str, Any]

METRIC_DIRECTIONS = {
    "higher_is_better",
    "lower_is_better",
    "neutral",
}


@dataclass(frozen=True)
class MetricSpec:
    name: str
    source: str
    path: str
    direction: str = "neutral"
    description: str = ""


@dataclass(frozen=True)
class DistributionSpec:
    name: str
    source: str
    path: str
    description: str = ""


@dataclass(frozen=True)
class RunTrackingConfig:
    runs_dir: Path
    index_output: Path
    stage_report_paths: Mapping[str, Path]
    artifact_paths: Mapping[str, Path]


DEFAULT_METRIC_SPECS = (
    MetricSpec("data.raw_record_count", "data_preprocessing", "input_record_count"),
    MetricSpec("data.cleaned_record_count", "data_preprocessing", "cleaned_input_record_count"),
    MetricSpec("data.processed_record_count", "data_preprocessing", "output_record_count"),
    MetricSpec(
        "data.dropped_incomplete_duplicate_rows",
        "data_preprocessing",
        "cleaning.incomplete_duplicate_order_items.dropped_record_count",
        "lower_is_better",
    ),
    MetricSpec(
        "data.dropped_zero_quantity_price_rows",
        "data_preprocessing",
        "cleaning.zero_quantity_price_items.dropped_record_count",
        "lower_is_better",
    ),
    MetricSpec("splits.group_count", "data_splitting", "group_count"),
    MetricSpec(
        "splits.train_validation_records",
        "data_splitting",
        "splits.train_validation.record_count",
    ),
    MetricSpec("splits.test_records", "data_splitting", "splits.test.record_count"),
    MetricSpec("splits.n_splits", "data_splitting", "parameters.n_splits"),
    MetricSpec(
        "classification.global_output_records",
        "classification_preprocessing",
        "global.output_record_count",
    ),
    MetricSpec(
        "classification.global_dropped_records",
        "classification_preprocessing",
        "global.dropped_record_count",
        "lower_is_better",
    ),
    MetricSpec(
        "classification.output_label_count",
        "classification_preprocessing",
        "global.output_label_count",
        "higher_is_better",
    ),
    MetricSpec(
        "category.training_examples",
        "category_training",
        "summary.training_example_count",
        "higher_is_better",
    ),
    MetricSpec(
        "category.class_count",
        "category_training",
        "summary.class_count",
        "higher_is_better",
    ),
    MetricSpec(
        "category.vocabulary_size",
        "category_training",
        "summary.vocabulary_size",
        "neutral",
    ),
    MetricSpec(
        "category.validation.accuracy",
        "category_validation",
        "summary.mean_accuracy",
        "higher_is_better",
    ),
    MetricSpec(
        "category.validation.macro_precision",
        "category_validation",
        "summary.mean_macro_precision",
        "higher_is_better",
    ),
    MetricSpec(
        "category.validation.macro_recall",
        "category_validation",
        "summary.mean_macro_recall",
        "higher_is_better",
    ),
    MetricSpec(
        "category.validation.macro_f1",
        "category_validation",
        "summary.mean_macro_f1",
        "higher_is_better",
    ),
    MetricSpec(
        "category.validation.weighted_f1",
        "category_validation",
        "summary.mean_weighted_f1",
        "higher_is_better",
    ),
    MetricSpec(
        "category.validation.accuracy_std",
        "category_validation",
        "summary.std_accuracy",
        "lower_is_better",
    ),
    MetricSpec("category.test.accuracy", "category_testing", "test.accuracy", "higher_is_better"),
    MetricSpec(
        "category.test.macro_precision",
        "category_testing",
        "test.macro_precision",
        "higher_is_better",
    ),
    MetricSpec(
        "category.test.macro_recall",
        "category_testing",
        "test.macro_recall",
        "higher_is_better",
    ),
    MetricSpec("category.test.macro_f1", "category_testing", "test.macro_f1", "higher_is_better"),
    MetricSpec(
        "category.test.weighted_f1",
        "category_testing",
        "test.weighted_f1",
        "higher_is_better",
    ),
    MetricSpec(
        "association.training.rule_count",
        "association_training",
        "summary.rule_count",
        "neutral",
    ),
    MetricSpec(
        "association.training.exported_rule_count",
        "association_training",
        "summary.exported_rule_count",
        "neutral",
    ),
    MetricSpec(
        "association.validation.confidence",
        "association_validation",
        "summary.mean_validation_confidence",
        "higher_is_better",
    ),
    MetricSpec(
        "association.validation.lift",
        "association_validation",
        "summary.mean_validation_lift",
        "higher_is_better",
    ),
    MetricSpec(
        "association.validation.coverage",
        "association_validation",
        "summary.mean_antecedent_coverage",
        "higher_is_better",
    ),
    MetricSpec(
        "association.validation.hit_rate",
        "association_validation",
        "summary.mean_hit_rate_per_covered_basket",
        "higher_is_better",
    ),
    MetricSpec(
        "association.validation.confidence_gap",
        "association_validation",
        "summary.mean_abs_confidence_gap",
        "lower_is_better",
    ),
    MetricSpec(
        "association.validation.stable_rule_count",
        "association_validation",
        "summary.mean_stable_rule_count",
        "higher_is_better",
    ),
    MetricSpec(
        "association.test.confidence",
        "association_testing",
        "test.mean_test_confidence",
        "higher_is_better",
    ),
    MetricSpec(
        "association.test.lift",
        "association_testing",
        "test.mean_test_lift",
        "higher_is_better",
    ),
    MetricSpec(
        "association.test.coverage",
        "association_testing",
        "test.antecedent_coverage",
        "higher_is_better",
    ),
    MetricSpec(
        "association.test.hit_rate",
        "association_testing",
        "test.hit_rate_per_covered_basket",
        "higher_is_better",
    ),
    MetricSpec(
        "association.test.confidence_gap",
        "association_testing",
        "test.mean_abs_confidence_gap",
        "lower_is_better",
    ),
    MetricSpec(
        "association.test.stable_rule_count",
        "association_testing",
        "test.stable_rule_count",
        "higher_is_better",
    ),
    MetricSpec(
        "quality.input_data.failed_count",
        "input_data_quality",
        "summary.failed_count",
        "lower_is_better",
    ),
    MetricSpec(
        "quality.category_model.failed_count",
        "category_model_quality",
        "summary.failed_count",
        "lower_is_better",
    ),
    MetricSpec(
        "quality.association_model.failed_count",
        "association_model_quality",
        "summary.failed_count",
        "lower_is_better",
    ),
    MetricSpec(
        "quality.project.failed_report_count",
        "project_quality",
        "metrics.failed_report_count",
        "lower_is_better",
    ),
    MetricSpec(
        "quality.project.total_child_failed_check_count",
        "project_quality",
        "metrics.total_child_failed_check_count",
        "lower_is_better",
    ),
)

DEFAULT_DISTRIBUTION_SPECS = (
    DistributionSpec(
        "classification.global_label_distribution",
        "classification_preprocessing",
        "global.output_label_distribution",
        "Distribution of supervised category labels after preprocessing.",
    ),
    DistributionSpec(
        "classification.train_validation_label_distribution",
        "classification_preprocessing",
        "train_validation.output_label_distribution",
        "Distribution of supervised category labels in train-validation data.",
    ),
    DistributionSpec(
        "classification.test_label_distribution",
        "classification_preprocessing",
        "test.output_label_distribution",
        "Distribution of supervised category labels in hold-out test data.",
    ),
    DistributionSpec(
        "splits.train_validation_shop_distribution",
        "data_splitting",
        "splits.train_validation.stratify_group_distribution",
        "Shop distribution by basket in train-validation split.",
    ),
    DistributionSpec(
        "splits.test_shop_distribution",
        "data_splitting",
        "splits.test.stratify_group_distribution",
        "Shop distribution by basket in hold-out test split.",
    ),
)


def build_run_report(
    config: RunTrackingConfig,
    *,
    run_id: str | None = None,
    run_name: str | None = None,
    baseline_report: Mapping[str, Any] | None = None,
) -> Record:
    resolved_run_id = run_id or _default_run_id()
    loaded_reports = _load_stage_reports(config.stage_report_paths)
    report_summaries = {
        name: _stage_report_summary(name, path, payload)
        for name, (path, payload) in loaded_reports.items()
    }
    metrics = _collect_metrics(loaded_reports)
    distributions = _collect_distributions(loaded_reports)
    artifacts = {
        name: artifact_metadata(path, _artifact_type(name), name)
        for name, path in sorted(config.artifact_paths.items())
    }
    quality_summary = _quality_summary(loaded_reports)
    comparison = _compare_to_baseline(
        current_metrics=metrics,
        current_distributions=distributions,
        baseline_report=baseline_report,
    )

    report: Record = {
        "report_type": "pipeline_run_report",
        "subject": "ai_grocery_testing_project",
        "step": "Run tracking",
        "run_id": resolved_run_id,
        "run_name": run_name,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git": _git_state(),
        "stage_reports": report_summaries,
        "artifacts": artifacts,
        "data_versions": _artifact_group(artifacts, "dataset"),
        "model_versions": _artifact_group(artifacts, "model"),
        "metrics": metrics,
        "distributions": distributions,
        "quality_summary": quality_summary,
        "comparison": comparison,
    }
    report["summary"] = {
        "stage_report_count": len(report_summaries),
        "available_stage_report_count": sum(
            1 for summary in report_summaries.values() if summary.get("exists") is True
        ),
        "artifact_count": len(artifacts),
        "available_artifact_count": sum(
            1 for artifact in artifacts.values() if artifact.get("exists") is True
        ),
        "metric_count": len(metrics),
        "distribution_count": len(distributions),
        "quality_status": quality_summary.get("status"),
        "comparison_status": comparison.get("status"),
    }
    return report


def latest_run_report_path(index_output: Path) -> Path | None:
    index = _read_json_object(index_output)
    latest_path = _string_value(index.get("latest_run_report"))
    if latest_path is None:
        return None
    path = Path(latest_path)
    return path if path.exists() else None


def update_run_index(
    index_output: Path,
    run_report: Mapping[str, Any],
    run_report_path: Path,
) -> Record:
    index = _read_json_object(index_output)
    runs = _list_of_mappings(index.get("runs"))

    run_id = _string_value(run_report.get("run_id")) or _default_run_id()
    compact = _compact_run_entry(run_report, run_report_path)
    runs = [run for run in runs if run.get("run_id") != run_id]
    runs.append(compact)
    runs.sort(key=lambda item: str(item.get("created_at") or ""))

    updated = {
        "report_type": "run_index",
        "subject": "ai_grocery_testing_project",
        "latest_run_id": run_id,
        "latest_run_report": str(run_report_path),
        "run_count": len(runs),
        "runs": runs,
    }
    index_output.parent.mkdir(parents=True, exist_ok=True)
    index_output.write_text(json.dumps(updated, ensure_ascii=False, indent=2), encoding="utf-8")
    return updated


def compact_run_history(index_output: Path, limit: int = 10) -> Record:
    index = _read_json_object(index_output)
    runs = _list_of_mappings(index.get("runs"))
    runs.sort(key=lambda item: str(item.get("created_at") or ""), reverse=True)
    return {
        "index": str(index_output),
        "run_count": len(runs),
        "latest_run_id": index.get("latest_run_id"),
        "runs": runs[: max(limit, 0)],
    }


def _load_stage_reports(paths: Mapping[str, Path]) -> dict[str, tuple[Path, Record | None]]:
    return {
        name: (path, _read_json_object(path) if path.exists() else None)
        for name, path in sorted(paths.items())
    }


def _stage_report_summary(name: str, path: Path, payload: Mapping[str, Any] | None) -> Record:
    metadata = artifact_metadata(path, "report", name)
    if payload is None:
        return {
            **metadata,
            "stage": name,
            "exists": False,
        }

    summary = _mapping(payload.get("summary"))
    return {
        **metadata,
        "stage": name,
        "step": payload.get("step"),
        "report_type": payload.get("report_type"),
        "status": payload.get("status"),
        "summary": dict(summary),
    }


def _collect_metrics(loaded_reports: Mapping[str, tuple[Path, Record | None]]) -> Record:
    metrics: Record = {}
    for spec in DEFAULT_METRIC_SPECS:
        report = _report_payload(loaded_reports, spec.source)
        if report is None:
            continue
        value = _path_value(report, spec.path)
        if value is None:
            continue
        metrics[spec.name] = {
            "value": value,
            "source": spec.source,
            "path": spec.path,
            "direction": _metric_direction(spec.direction),
        }
        if spec.description:
            metrics[spec.name]["description"] = spec.description
    return metrics


def _collect_distributions(loaded_reports: Mapping[str, tuple[Path, Record | None]]) -> Record:
    distributions: Record = {}
    for spec in DEFAULT_DISTRIBUTION_SPECS:
        report = _report_payload(loaded_reports, spec.source)
        if report is None:
            continue
        distribution = _distribution(_path_value(report, spec.path))
        if not distribution:
            continue
        distributions[spec.name] = {
            "source": spec.source,
            "path": spec.path,
            "description": spec.description,
            "values": distribution,
        }
    return distributions


def _quality_summary(loaded_reports: Mapping[str, tuple[Path, Record | None]]) -> Record:
    quality_sources = (
        "input_data_quality",
        "association_model_quality",
        "category_model_quality",
        "project_quality",
    )
    reports: Record = {}
    failed_reports: list[str] = []
    for source in quality_sources:
        report = _report_payload(loaded_reports, source)
        if report is None:
            continue
        status = _string_value(report.get("status")) or "unknown"
        summary = _mapping(report.get("summary"))
        failed_count = _int_value(summary.get("failed_count"))
        reports[source] = {
            "status": status,
            "check_count": _int_value(summary.get("check_count")),
            "failed_count": failed_count,
        }
        decision = _mapping(report.get("decision"))
        if decision:
            reports[source]["decision_outcome"] = decision.get("outcome")
            reports[source]["recommendation"] = decision.get("recommendation")
        if status != "passed" or failed_count > 0:
            failed_reports.append(source)

    return {
        "status": "passed" if reports and not failed_reports else "failed",
        "available_report_count": len(reports),
        "failed_report_count": len(failed_reports),
        "failed_reports": failed_reports,
        "reports": reports,
    }


def _compare_to_baseline(
    *,
    current_metrics: Mapping[str, Any],
    current_distributions: Mapping[str, Any],
    baseline_report: Mapping[str, Any] | None,
) -> Record:
    if baseline_report is None:
        return {
            "status": "not_available",
            "reason": "No baseline run report was provided.",
        }

    baseline_metrics = _mapping(baseline_report.get("metrics"))
    baseline_distributions = _mapping(baseline_report.get("distributions"))
    metric_deltas = _metric_deltas(current_metrics, baseline_metrics)
    drift = _distribution_drift(current_distributions, baseline_distributions)
    regressed_metrics = [
        delta["name"] for delta in metric_deltas if delta.get("change") == "regressed"
    ]
    improved_metrics = [
        delta["name"] for delta in metric_deltas if delta.get("change") == "improved"
    ]
    return {
        "status": "compared",
        "baseline_run_id": baseline_report.get("run_id"),
        "baseline_created_at": baseline_report.get("created_at"),
        "metric_delta_summary": {
            "compared_metric_count": len(metric_deltas),
            "improved_metric_count": len(improved_metrics),
            "regressed_metric_count": len(regressed_metrics),
            "unchanged_metric_count": sum(
                1 for delta in metric_deltas if delta.get("change") == "unchanged"
            ),
        },
        "regressed_metrics": regressed_metrics,
        "improved_metrics": improved_metrics,
        "metric_deltas": metric_deltas,
        "drift": drift,
    }


def _metric_deltas(
    current_metrics: Mapping[str, Any],
    baseline_metrics: Mapping[str, Any],
) -> list[Record]:
    deltas: list[Record] = []
    for name, metric in sorted(current_metrics.items()):
        if not isinstance(metric, Mapping):
            continue
        baseline_metric = baseline_metrics.get(name)
        if not isinstance(baseline_metric, Mapping):
            continue

        current_value = metric.get("value")
        baseline_value = baseline_metric.get("value")
        current_number = _float_value(current_value)
        baseline_number = _float_value(baseline_value)
        direction = _metric_direction(_string_value(metric.get("direction")) or "neutral")

        if current_number is None or baseline_number is None:
            deltas.append(
                {
                    "name": name,
                    "previous": baseline_value,
                    "current": current_value,
                    "direction": direction,
                    "change": "changed" if current_value != baseline_value else "unchanged",
                }
            )
            continue

        delta = round(current_number - baseline_number, 6)
        relative_delta = (
            round(delta / abs(baseline_number), 6) if abs(baseline_number) > 0.000001 else None
        )
        deltas.append(
            {
                "name": name,
                "previous": baseline_number,
                "current": current_number,
                "delta": delta,
                "relative_delta": relative_delta,
                "direction": direction,
                "change": _metric_change(delta, direction),
            }
        )
    return deltas


def _distribution_drift(
    current_distributions: Mapping[str, Any],
    baseline_distributions: Mapping[str, Any],
) -> Record:
    reports: Record = {}
    for name, distribution in sorted(current_distributions.items()):
        if not isinstance(distribution, Mapping):
            continue
        baseline_distribution = baseline_distributions.get(name)
        if not isinstance(baseline_distribution, Mapping):
            continue
        current_values = _distribution(distribution.get("values"))
        baseline_values = _distribution(baseline_distribution.get("values"))
        if not current_values or not baseline_values:
            continue
        reports[name] = _single_distribution_drift(current_values, baseline_values)

    return {
        "distribution_count": len(reports),
        "max_total_variation_distance": max(
            (float(report["total_variation_distance"]) for report in reports.values()),
            default=0.0,
        ),
        "distributions": reports,
    }


def _single_distribution_drift(
    current_values: Mapping[str, float],
    baseline_values: Mapping[str, float],
) -> Record:
    keys = sorted(set(current_values) | set(baseline_values))
    changes = [
        {
            "value": key,
            "previous": baseline_values.get(key, 0.0),
            "current": current_values.get(key, 0.0),
            "delta": round(current_values.get(key, 0.0) - baseline_values.get(key, 0.0), 6),
        }
        for key in keys
    ]
    changes.sort(key=lambda item: abs(_float_value(item["delta"]) or 0.0), reverse=True)
    total_variation_distance = round(
        0.5
        * sum(abs(current_values.get(key, 0.0) - baseline_values.get(key, 0.0)) for key in keys),
        6,
    )
    return {
        "total_variation_distance": total_variation_distance,
        "top_changes": changes[:10],
    }


def _distribution(value: Any) -> dict[str, float]:
    if not isinstance(value, Mapping):
        return {}
    if _looks_like_counter_report(value):
        return {
            str(key): _float_value(_mapping(item).get("rate")) or 0.0
            for key, item in value.items()
            if isinstance(item, Mapping)
        }

    counts = {
        str(key): number
        for key, raw_value in value.items()
        for number in [_float_value(raw_value)]
        if number is not None and number >= 0
    }
    total = sum(counts.values())
    if total <= 0:
        return {}
    return {key: round(count / total, 6) for key, count in counts.items()}


def _looks_like_counter_report(value: Mapping[str, Any]) -> bool:
    return any(isinstance(item, Mapping) and "rate" in item for item in value.values())


def _report_payload(
    loaded_reports: Mapping[str, tuple[Path, Record | None]],
    name: str,
) -> Record | None:
    report_entry = loaded_reports.get(name)
    if report_entry is None:
        return None
    return report_entry[1]


def _path_value(subject: Mapping[str, Any], path: str) -> Any:
    current: Any = subject
    for part in path.split("."):
        if not isinstance(current, Mapping):
            return None
        current = current.get(part)
    return current


def _artifact_group(artifacts: Mapping[str, Any], artifact_type: str) -> Record:
    return {
        name: dict(artifact)
        for name, artifact in artifacts.items()
        if isinstance(artifact, Mapping) and artifact.get("type") == artifact_type
    }


def _artifact_type(name: str) -> str:
    if "report" in name:
        return "report"
    if "manifest" in name:
        return "manifest"
    if "model" in name or "estimator" in name:
        return "model"
    return "dataset"


def _compact_run_entry(run_report: Mapping[str, Any], run_report_path: Path) -> Record:
    summary = _mapping(run_report.get("summary"))
    comparison = _mapping(run_report.get("comparison"))
    delta_summary = _mapping(comparison.get("metric_delta_summary"))
    quality_summary = _mapping(run_report.get("quality_summary"))
    git_state = _mapping(run_report.get("git"))
    return {
        "run_id": run_report.get("run_id"),
        "run_name": run_report.get("run_name"),
        "created_at": run_report.get("created_at"),
        "report": str(run_report_path),
        "git_commit": git_state.get("commit"),
        "quality_status": quality_summary.get("status") or summary.get("quality_status"),
        "metric_count": summary.get("metric_count"),
        "improved_metric_count": delta_summary.get("improved_metric_count"),
        "regressed_metric_count": delta_summary.get("regressed_metric_count"),
    }


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


def _read_json_object(path: Path) -> Record:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return dict(payload) if isinstance(payload, Mapping) else {}


def _default_run_id() -> str:
    return datetime.now(timezone.utc).strftime("run-%Y%m%dT%H%M%SZ")


def _metric_direction(direction: str) -> str:
    return direction if direction in METRIC_DIRECTIONS else "neutral"


def _metric_change(delta: float, direction: str) -> str:
    if abs(delta) <= 0.000001:
        return "unchanged"
    if direction == "higher_is_better":
        return "improved" if delta > 0 else "regressed"
    if direction == "lower_is_better":
        return "improved" if delta < 0 else "regressed"
    return "changed"


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

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path

from ai_testing.cli_common import (
    exit_if_report_failed,
    mapping,
    path_mapping,
    print_report_summary,
    read_json_object,
    record_stage_result,
    write_json,
)
from ai_testing.core import artifact_metadata
from ai_testing.drift_testing import RunDriftTestConfig, test_run_drift
from ai_testing.llm_exploratory_testing import (
    LLMExploratoryTestingConfig,
    create_llm_exploratory_test_plan,
)
from ai_testing.model_explainability import (
    TextClassifierExplainabilityConfig,
    explain_text_classifier,
)
from ai_testing.reporting import (
    MarkdownReportConfig,
    RunMarkdownReportConfig,
    build_markdown_report,
    build_run_markdown_report,
)
from ai_testing.run_tracking import (
    RunTrackingConfig,
    build_run_report,
    build_stage_history_markdown,
    build_stage_metric_history_markdown,
    compact_run_history,
    latest_run_report_path,
    update_run_index,
)


def test_drift_command(args: argparse.Namespace) -> None:
    run_report_path = _resolve_run_report_input(
        Path(args.run_report_input),
        bool(args.use_latest_run),
    )
    output_path = Path(args.output)
    run_report = read_json_object(run_report_path)
    result = test_run_drift(
        run_report,
        config=RunDriftTestConfig(
            require_baseline=bool(args.require_baseline),
            max_regressed_metric_count=int(args.max_regressed_metric_count),
            max_total_variation_distance=float(args.max_total_variation_distance),
            max_distribution_total_variation_distance=float(
                args.max_distribution_total_variation_distance
            ),
            max_critical_metric_regression=float(args.max_critical_metric_regression),
            critical_metrics=tuple(args.critical_metrics),
        ),
    )
    report = {
        **result,
        "inputs": {"run_report": str(run_report_path)},
        "artifacts": {"run_report": artifact_metadata(run_report_path, "report", "run_report")},
        "output": str(output_path),
    }
    write_json(report, output_path)
    record_stage_result(
        args,
        stage_name="drift_testing",
        artifacts={"drift_report": output_path},
        payload=report,
    )
    print_report_summary(
        report,
        output_path,
        extra={
            "run_id": report.get("run_id"),
            "baseline_run_id": report.get("baseline_run_id"),
            "max_total_variation_distance": mapping(report.get("metrics")).get(
                "max_total_variation_distance"
            ),
        },
    )
    exit_if_report_failed(report)


def generate_run_report_command(args: argparse.Namespace) -> None:
    run_report_path = _resolve_run_report_input(
        Path(args.run_report_input),
        bool(args.use_latest_run),
    )
    drift_report_path = Path(args.drift_report_input)
    drift_report = read_json_object(drift_report_path) if drift_report_path.exists() else None
    output_path = Path(args.output)
    markdown = build_run_markdown_report(
        read_json_object(run_report_path),
        drift_report=drift_report,
        config=RunMarkdownReportConfig(
            max_metric_deltas=int(args.max_metric_deltas),
            max_drift_distributions=int(args.max_drift_distributions),
            max_artifacts=int(args.max_artifacts),
        ),
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(markdown, encoding="utf-8")
    record_stage_result(
        args,
        stage_name="reporting.latest_run_report",
        artifacts={"markdown_report": output_path, "run_report": run_report_path},
        payload={"summary": {"generated_report_count": 1}},
    )
    print(
        json.dumps(
            {
                "output": str(output_path),
                "run_report_input": str(run_report_path),
                "drift_report_input": str(drift_report_path) if drift_report is not None else None,
                "format": "markdown",
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def generate_markdown_reports_command(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    report_paths = (
        _parse_report_specs(tuple(args.report)) if args.report else path_mapping(args.report_paths)
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    generated: list[dict[str, object]] = []
    skipped: list[dict[str, object]] = []
    markdown_config = MarkdownReportConfig(
        max_checks=int(args.max_checks),
        max_rows=int(args.max_rows),
    )

    for name, report_path in sorted(report_paths.items()):
        output_path = output_dir / f"{_safe_filename(name)}.md"
        if not report_path.exists():
            skipped.append({"name": name, "input": str(report_path), "reason": "missing"})
            continue
        markdown = build_markdown_report(
            read_json_object(report_path),
            title=_markdown_title(name),
            source_path=report_path,
            config=markdown_config,
        )
        output_path.write_text(markdown, encoding="utf-8")
        generated.append({"name": name, "input": str(report_path), "output": str(output_path)})

    if args.include_run_report:
        run_output_path = output_dir / "latest_run_report.md"
        try:
            run_report_path = _resolve_run_report_input(
                Path(args.run_report_input),
                bool(args.use_latest_run),
            )
        except SystemExit:
            skipped.append(
                {
                    "name": "latest_run_report",
                    "input": str(args.run_report_input),
                    "reason": "missing_latest_run",
                }
            )
        else:
            drift_report_path = Path(args.drift_report_input)
            drift_report = (
                read_json_object(drift_report_path) if drift_report_path.exists() else None
            )
            run_markdown = build_run_markdown_report(
                read_json_object(run_report_path),
                drift_report=drift_report,
            )
            run_output_path.write_text(run_markdown, encoding="utf-8")
            generated.append(
                {
                    "name": "latest_run_report",
                    "input": str(run_report_path),
                    "output": str(run_output_path),
                }
            )

    stage_history_path = Path(str(args.stage_history_output))
    stage_history_output_path = output_dir / "stage_history.md"
    stage_metric_history_output_path = output_dir / "stage_metric_history.md"
    if stage_history_path.exists():
        stage_history = read_json_object(stage_history_path)
        stage_history_output_path.write_text(
            build_stage_history_markdown(
                stage_history,
                recent_entries_per_stage=int(args.stage_history_recent_entries_per_stage),
            ),
            encoding="utf-8",
        )
        stage_metric_history_output_path.write_text(
            build_stage_metric_history_markdown(
                stage_history,
                recent_entries_per_stage=int(args.stage_metric_history_recent_entries_per_stage),
            ),
            encoding="utf-8",
        )
        generated.append(
            {
                "name": "stage_history",
                "input": str(stage_history_path),
                "output": str(stage_history_output_path),
            }
        )
        generated.append(
            {
                "name": "stage_metric_history",
                "input": str(stage_history_path),
                "output": str(stage_metric_history_output_path),
            }
        )
    else:
        skipped.append(
            {
                "name": "stage_history",
                "input": str(stage_history_path),
                "reason": "missing",
            }
        )
        skipped.append(
            {
                "name": "stage_metric_history",
                "input": str(stage_history_path),
                "reason": "missing",
            }
        )

    record_stage_result(
        args,
        stage_name="reporting.markdown_reports",
        artifacts={"output_dir": output_dir},
        payload={
            "summary": {
                "generated_report_count": len(generated),
                "skipped_report_count": len(skipped),
            }
        },
    )
    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "generated_count": len(generated),
                "skipped_count": len(skipped),
                "generated": generated,
                "skipped": skipped,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def track_run_command(args: argparse.Namespace) -> None:
    runs_dir = Path(args.runs_dir)
    index_output_path = Path(args.index_output)
    baseline_report = _baseline_run_report(
        explicit_baseline=str(args.baseline_run_report or ""),
        compare_to_latest=bool(args.compare_to_latest),
        index_output_path=index_output_path,
    )
    config = RunTrackingConfig(
        runs_dir=runs_dir,
        index_output=index_output_path,
        stage_report_paths=path_mapping(args.stage_report_paths),
        artifact_paths=path_mapping(args.artifact_paths),
    )
    report = build_run_report(
        config,
        run_id=str(args.run_id) if args.run_id else None,
        run_name=str(args.run_name) if args.run_name else None,
        baseline_report=baseline_report,
    )
    run_id = str(report["run_id"])
    output_path = Path(args.output) if args.output else runs_dir / run_id / "run_report.json"
    markdown_output_path = output_path.with_suffix(".md")
    write_json(report, output_path)
    markdown_output_path.write_text(build_run_markdown_report(report), encoding="utf-8")
    index = update_run_index(index_output_path, report, output_path)
    record_stage_result(
        args,
        stage_name="run_tracking",
        artifacts={
            "run_report": output_path,
            "run_report_markdown": markdown_output_path,
            "run_index": index_output_path,
        },
        payload=report,
    )
    comparison = mapping(report.get("comparison"))
    delta_summary = mapping(comparison.get("metric_delta_summary"))
    summary = mapping(report.get("summary"))
    print(
        json.dumps(
            {
                "run_id": run_id,
                "output": str(output_path),
                "index_output": str(index_output_path),
                "run_count": index.get("run_count"),
                "quality_status": summary.get("quality_status"),
                "available_stage_report_count": summary.get("available_stage_report_count"),
                "available_artifact_count": summary.get("available_artifact_count"),
                "metric_count": summary.get("metric_count"),
                "comparison_status": comparison.get("status"),
                "baseline_run_id": comparison.get("baseline_run_id"),
                "improved_metric_count": delta_summary.get("improved_metric_count"),
                "regressed_metric_count": delta_summary.get("regressed_metric_count"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def run_history_command(args: argparse.Namespace) -> None:
    history = compact_run_history(Path(args.index_output), limit=int(args.limit))
    print(json.dumps(history, ensure_ascii=False, indent=2))


def explain_category_classifier_command(args: argparse.Namespace) -> None:
    model_path = Path(args.model_input)
    output_path = Path(args.output)
    try:
        report = explain_text_classifier(
            read_json_object(model_path),
            config=TextClassifierExplainabilityConfig(
                top_features_per_class=int(args.top_features_per_class)
            ),
        )
    except ValueError as error:
        raise SystemExit(f"Cannot explain category classifier: {error}") from error
    report = {
        **report,
        "inputs": {"model": str(model_path)},
        "artifacts": {"model": artifact_metadata(model_path, "model", "model_input")},
        "output": str(output_path),
    }
    write_json(report, output_path)
    record_stage_result(
        args,
        stage_name="model_explainability.category_classifier",
        artifacts={"explainability_report": output_path},
        payload=report,
    )
    summary = mapping(report.get("summary"))
    print(
        json.dumps(
            {
                "output": str(output_path),
                "class_count": summary.get("class_count"),
                "feature_count": summary.get("feature_count"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def create_llm_exploratory_plan_command(args: argparse.Namespace) -> None:
    output_path = Path(args.output)
    try:
        report = create_llm_exploratory_test_plan(
            LLMExploratoryTestingConfig(
                target_model_name=str(args.target_model_name),
                domain=str(args.domain),
                session_duration_minutes=int(args.session_duration_minutes),
            )
        )
    except ValueError as error:
        raise SystemExit(f"Cannot create LLM exploratory plan: {error}") from error
    report = {**report, "output": str(output_path)}
    write_json(report, output_path)
    record_stage_result(
        args,
        stage_name="llm_exploratory_testing",
        artifacts={"llm_exploratory_plan": output_path},
        payload=report,
    )
    summary = mapping(report.get("summary"))
    print(
        json.dumps(
            {
                "output": str(output_path),
                "charter_count": summary.get("charter_count"),
                "target_model_name": report.get("subject"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def _baseline_run_report(
    *,
    explicit_baseline: str,
    compare_to_latest: bool,
    index_output_path: Path,
) -> Mapping[str, object] | None:
    baseline_path = Path(explicit_baseline) if explicit_baseline else None
    if baseline_path is None and compare_to_latest:
        baseline_path = latest_run_report_path(index_output_path)
    if baseline_path is None or not baseline_path.exists():
        return None
    return read_json_object(baseline_path)


def _resolve_run_report_input(input_path: Path, use_latest_run: bool) -> Path:
    if use_latest_run:
        latest_path = latest_run_report_path(input_path)
        if latest_path is None:
            raise SystemExit(f"Cannot resolve latest run report from {input_path}")
        return latest_path
    if not input_path.exists():
        raise SystemExit(f"Run report does not exist: {input_path}")
    return input_path


def _parse_report_specs(specs: Sequence[str]) -> dict[str, Path]:
    report_paths: dict[str, Path] = {}
    for spec in specs:
        if "=" not in spec:
            raise SystemExit(f"Invalid --report value {spec!r}. Expected NAME=PATH.")
        name, raw_path = spec.split("=", 1)
        name = name.strip()
        raw_path = raw_path.strip()
        if not name or not raw_path:
            raise SystemExit(f"Invalid --report value {spec!r}. Expected NAME=PATH.")
        report_paths[name] = Path(raw_path)
    return report_paths


def _safe_filename(value: str) -> str:
    safe = "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in value)
    return safe.strip("_") or "report"


def _markdown_title(value: str) -> str:
    return value.replace("_", " ").replace("-", " ").title()

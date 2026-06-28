from __future__ import annotations

import argparse
from collections.abc import Mapping
from pathlib import Path

from ai_testing.cli.commands.data_preparation import _read_input_data_folds
from ai_testing.cli_common import (
    exit_if_report_failed,
    mapping,
    print_report_summary,
    read_json_object,
    read_records,
    record_stage_result,
    write_json,
)
from ai_testing.core import artifact_metadata
from ai_testing.quality.input_data import InputDataTestConfig, test_input_data
from ai_testing.quality.ml_model import (
    AssociationMLModelTestConfig,
    TextClassifierMLModelTestConfig,
    test_association_ml_model,
    test_text_classifier_ml_model,
)
from ai_testing.quality.project import aggregate_quality_reports


def _test_input_data(args: argparse.Namespace) -> None:
    processed_input_path = Path(args.processed_input)
    train_validation_input_path = Path(args.train_validation_input)
    test_input_path = Path(args.test_input)
    folds_dir = Path(args.folds_dir)
    output_path = Path(args.output)
    result = test_input_data(
        processed_records=read_records(processed_input_path),
        train_validation_records=read_records(train_validation_input_path),
        test_records=read_records(test_input_path),
        folds=_read_input_data_folds(folds_dir),
        config=InputDataTestConfig(
            required_fields=tuple(args.required_fields),
            protected_fields=tuple(args.protected_fields),
            critical_fields=tuple(args.critical_fields),
            coverage_fields=tuple(args.coverage_fields),
            group_field=str(args.group_field),
            stratify_field=str(args.stratify_field),
            expected_shops=tuple(args.expected_shops),
            expected_currencies=tuple(args.expected_currencies),
            max_coverage_missing_rate=float(args.max_coverage_missing_rate),
            max_split_distribution_delta=float(args.max_split_distribution_delta),
        ),
    )
    report = {
        **result.report,
        "inputs": {
            "processed": str(processed_input_path),
            "train_validation": str(train_validation_input_path),
            "test": str(test_input_path),
            "folds_dir": str(folds_dir),
        },
        "artifacts": {
            "processed": artifact_metadata(processed_input_path, "dataset", "processed_input"),
            "train_validation": artifact_metadata(
                train_validation_input_path,
                "dataset",
                "train_validation_input",
            ),
            "test": artifact_metadata(test_input_path, "dataset", "test_input"),
            "folds_dir": artifact_metadata(folds_dir, "dataset_directory", "folds_input"),
        },
        "output": str(output_path),
    }
    write_json(report, output_path)
    record_stage_result(
        args,
        stage_name="input_data_testing",
        artifacts={"input_data_report": output_path},
        payload=report,
    )
    print_report_summary(report, output_path)
    exit_if_report_failed(report)


def _test_ml_model(args: argparse.Namespace) -> None:
    model_path = Path(args.model_input)
    validation_report_path = Path(args.validation_report_input)
    test_report_path = Path(args.test_report_input)
    test_dataset_path = Path(args.test_dataset_input)
    output_path = Path(args.output)
    result = test_association_ml_model(
        model=read_json_object(model_path),
        validation_report=read_json_object(validation_report_path),
        test_report=read_json_object(test_report_path),
        config=AssociationMLModelTestConfig(
            test_dataset_input=str(args.test_dataset_input),
            forbidden_feature_fields=tuple(args.forbidden_feature_fields),
            min_mean_test_confidence=float(args.min_mean_test_confidence),
            min_mean_test_lift=float(args.min_mean_test_lift),
            max_mean_abs_test_confidence_gap=float(args.max_mean_abs_test_confidence_gap),
            max_validation_test_confidence_delta=float(args.max_validation_test_confidence_delta),
            max_validation_test_lift_delta=float(args.max_validation_test_lift_delta),
        ),
    )
    report = {
        **result.report,
        "inputs": {
            "model": str(model_path),
            "validation_report": str(validation_report_path),
            "test_report": str(test_report_path),
            "test_dataset": str(test_dataset_path),
        },
        "artifacts": {
            "model": artifact_metadata(model_path, "model", "model_input"),
            "validation_report": artifact_metadata(
                validation_report_path,
                "report",
                "validation_report_input",
            ),
            "test_report": artifact_metadata(test_report_path, "report", "test_report_input"),
            "test_dataset": artifact_metadata(test_dataset_path, "dataset", "test_dataset_input"),
        },
        "output": str(output_path),
    }
    write_json(report, output_path)
    record_stage_result(
        args,
        stage_name="ml_model_testing.association_rules",
        artifacts={"ml_model_report": output_path},
        payload=report,
    )
    print_report_summary(report, output_path)
    exit_if_report_failed(report)


def _test_category_ml_model(args: argparse.Namespace) -> None:
    model_path = Path(args.model_input)
    validation_report_path = Path(args.validation_report_input)
    test_report_path = Path(args.test_report_input)
    classification_manifest_path = Path(args.classification_manifest_input)
    test_dataset_path = Path(args.test_dataset_input)
    output_path = Path(args.output)
    model = read_json_object(model_path)
    estimator_path = Path(str(model.get("estimator_path", "")))
    result = test_text_classifier_ml_model(
        model=model,
        validation_report=read_json_object(validation_report_path),
        test_report=read_json_object(test_report_path),
        classification_manifest=read_json_object(classification_manifest_path),
        config=TextClassifierMLModelTestConfig(
            test_dataset_input=str(args.test_dataset_input),
            forbidden_feature_fields=tuple(args.forbidden_feature_fields),
            min_test_accuracy=float(args.min_test_accuracy),
            min_test_macro_precision=float(args.min_test_macro_precision),
            min_test_macro_recall=float(args.min_test_macro_recall),
            min_test_macro_f1=float(args.min_test_macro_f1),
            min_test_weighted_f1=float(args.min_test_weighted_f1),
            max_validation_test_accuracy_delta=float(args.max_validation_test_accuracy_delta),
            max_validation_test_macro_f1_delta=float(args.max_validation_test_macro_f1_delta),
        ),
    )
    report = {
        **result.report,
        "inputs": {
            "model": str(model_path),
            "estimator": str(estimator_path),
            "validation_report": str(validation_report_path),
            "test_report": str(test_report_path),
            "classification_manifest": str(classification_manifest_path),
            "test_dataset": str(test_dataset_path),
        },
        "artifacts": {
            "model": artifact_metadata(model_path, "model", "model_input"),
            "estimator": artifact_metadata(estimator_path, "model", "estimator_input"),
            "validation_report": artifact_metadata(
                validation_report_path,
                "report",
                "validation_report_input",
            ),
            "test_report": artifact_metadata(test_report_path, "report", "test_report_input"),
            "classification_manifest": artifact_metadata(
                classification_manifest_path,
                "manifest",
                "classification_manifest_input",
            ),
            "test_dataset": artifact_metadata(test_dataset_path, "dataset", "test_dataset_input"),
        },
        "output": str(output_path),
    }
    write_json(report, output_path)
    record_stage_result(
        args,
        stage_name="ml_model_testing.category_classifier",
        artifacts={"ml_model_report": output_path},
        payload=report,
    )
    print_report_summary(report, output_path)
    exit_if_report_failed(report)


def _run_quality_gates(args: argparse.Namespace) -> None:
    input_data_report_path = Path(args.input_data_report)
    association_model_report_path = Path(args.association_model_report)
    category_model_report_path = Path(args.category_model_report)
    output_path = Path(args.output)
    reports: dict[str, Mapping[str, object]] = {
        "input_data": read_json_object(input_data_report_path),
        "association_model": read_json_object(association_model_report_path),
        "category_model": read_json_object(category_model_report_path),
    }
    drift_report_path = Path(str(args.drift_report)) if args.drift_report else None
    if drift_report_path is not None:
        if not drift_report_path.exists():
            raise SystemExit(f"Drift report does not exist: {drift_report_path}")
        reports["drift"] = read_json_object(drift_report_path)

    result = aggregate_quality_reports(
        reports,
    )
    report = {
        **result.report,
        "inputs": {
            "input_data_report": str(input_data_report_path),
            "association_model_report": str(association_model_report_path),
            "category_model_report": str(category_model_report_path),
            "drift_report": str(drift_report_path) if drift_report_path is not None else None,
        },
        "artifacts": {
            "input_data_report": artifact_metadata(
                input_data_report_path,
                "report",
                "input_data_report",
            ),
            "association_model_report": artifact_metadata(
                association_model_report_path,
                "report",
                "association_model_report",
            ),
            "category_model_report": artifact_metadata(
                category_model_report_path,
                "report",
                "category_model_report",
            ),
            "drift_report": (
                artifact_metadata(drift_report_path, "report", "drift_report")
                if drift_report_path is not None
                else {"path": None, "type": "report", "exists": False, "role": "drift_report"}
            ),
        },
        "output": str(output_path),
    }
    write_json(report, output_path)
    record_stage_result(
        args,
        stage_name="project_quality",
        artifacts={"project_quality_report": output_path},
        payload=report,
    )
    metrics = mapping(report.get("metrics"))
    print_report_summary(
        report,
        output_path,
        extra={
            "failed_report_count": metrics.get("failed_report_count"),
            "total_child_failed_check_count": metrics.get("total_child_failed_check_count"),
        },
    )
    exit_if_report_failed(report)

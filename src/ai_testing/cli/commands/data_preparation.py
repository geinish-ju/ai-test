from __future__ import annotations

import argparse
import json
from pathlib import Path

from ai_testing.cli_common import read_records, record_stage_result, write_json
from ai_testing.data.classification_preprocessing import (
    ClassificationPreprocessingConfig,
    build_classification_records,
)
from ai_testing.data.preprocessing import PreprocessingConfig, preprocess_grocery_records
from ai_testing.data.splitting import DatasetSplitConfig, split_dataset_records
from ai_testing.quality.input_data import InputDataFold


def _preprocess_data(args: argparse.Namespace) -> None:
    input_path = Path(args.input)
    output_path = Path(args.output)
    report_path = Path(args.report_output)
    raw_records = read_records(input_path)
    result = preprocess_grocery_records(
        raw_records,
        config=PreprocessingConfig(
            identifier_fields=tuple(args.identifier_fields),
            exact_time_fields=tuple(args.exact_time_fields),
            output_fields=tuple(args.output_fields),
            drop_exact_duplicates=bool(args.drop_exact_duplicates),
            basket_id_prefix=str(args.basket_id_prefix),
        ),
    )

    write_json(result.records, output_path)
    write_json(result.report, report_path)
    record_stage_result(
        args,
        stage_name="data_preprocessing",
        artifacts={"processed_dataset": output_path, "quality_report": report_path},
        payload=result.report,
    )
    print(
        json.dumps(
            {
                "input": str(input_path),
                "output": str(output_path),
                "report_output": str(report_path),
                "input_record_count": result.report["input_record_count"],
                "output_record_count": result.report["output_record_count"],
                "exact_duplicate_rows": result.report["duplicates"]["exact_duplicate_rows"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def _split_datasets(args: argparse.Namespace) -> None:
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    records = read_records(input_path)
    try:
        result = split_dataset_records(
            records,
            config=DatasetSplitConfig(
                group_field=str(args.group_field),
                stratify_field=str(args.stratify_field),
                test_size=float(args.test_size),
                n_splits=int(args.n_splits),
                random_seed=int(args.random_seed),
            ),
        )
    except ValueError as error:
        raise SystemExit(f"Cannot split datasets: {error}") from error

    output_dir.mkdir(parents=True, exist_ok=True)
    folds_dir = output_dir / "folds"
    folds_dir.mkdir(parents=True, exist_ok=True)

    train_validation_path = output_dir / "train_validation.json"
    test_path = output_dir / "test.json"
    manifest_path = output_dir / "split_manifest.json"
    write_json(result.train_validation_records, train_validation_path)
    write_json(result.test_records, test_path)
    for fold in result.folds:
        fold_prefix = f"fold_{fold.fold_index:02d}"
        write_json(fold.train_records, folds_dir / f"{fold_prefix}_train.json")
        write_json(fold.validation_records, folds_dir / f"{fold_prefix}_validation.json")

    manifest = {
        **result.manifest,
        "outputs": {
            "train_validation": str(train_validation_path),
            "test": str(test_path),
            "manifest": str(manifest_path),
            "folds_dir": str(folds_dir),
        },
    }
    write_json(manifest, manifest_path)
    record_stage_result(
        args,
        stage_name="data_splitting",
        artifacts={
            "train_validation": train_validation_path,
            "test": test_path,
            "manifest": manifest_path,
            "folds_dir": folds_dir,
        },
        payload=manifest,
    )
    print(
        json.dumps(
            {
                "input": str(input_path),
                "output_dir": str(output_dir),
                "input_record_count": manifest["input_record_count"],
                "group_count": manifest["group_count"],
                "train_validation_record_count": manifest["splits"]["train_validation"][
                    "record_count"
                ],
                "test_record_count": manifest["splits"]["test"]["record_count"],
                "n_splits": manifest["parameters"]["n_splits"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def _build_classification_dataset(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    folds_output_dir = output_dir / "folds"
    manifest_path = output_dir / "classification_manifest.json"
    config = ClassificationPreprocessingConfig(
        target_field=str(args.target_field),
        text_fields=tuple(args.text_fields),
        metadata_fields=tuple(args.metadata_fields),
        min_label_count=int(args.min_label_count),
    )
    processed_result = build_classification_records(
        read_records(Path(args.processed_input)),
        config=config,
    )
    train_validation_result = build_classification_records(
        read_records(Path(args.train_validation_input)),
        config=config,
        allowed_labels=processed_result.allowed_labels,
    )
    test_result = build_classification_records(
        read_records(Path(args.test_input)),
        config=config,
        allowed_labels=processed_result.allowed_labels,
    )
    folds = _read_input_data_folds(Path(args.folds_dir))

    output_dir.mkdir(parents=True, exist_ok=True)
    folds_output_dir.mkdir(parents=True, exist_ok=True)
    train_validation_path = output_dir / "train_validation.json"
    test_path = output_dir / "test.json"
    write_json(train_validation_result.records, train_validation_path)
    write_json(test_result.records, test_path)

    fold_reports: list[dict[str, object]] = []
    for fold in folds:
        train_result = build_classification_records(
            fold.train_records,
            config=config,
            allowed_labels=processed_result.allowed_labels,
        )
        validation_result = build_classification_records(
            fold.validation_records,
            config=config,
            allowed_labels=processed_result.allowed_labels,
        )
        fold_prefix = f"fold_{fold.fold_index:02d}"
        train_path = folds_output_dir / f"{fold_prefix}_train.json"
        validation_path = folds_output_dir / f"{fold_prefix}_validation.json"
        write_json(train_result.records, train_path)
        write_json(validation_result.records, validation_path)
        fold_reports.append(
            {
                "fold_index": fold.fold_index,
                "train": train_result.report,
                "validation": validation_result.report,
                "outputs": {
                    "train": str(train_path),
                    "validation": str(validation_path),
                },
            }
        )

    manifest = {
        "step": "Supervised classification dataset preprocessing",
        "task": "product category text classification",
        "source_inputs": {
            "processed": str(args.processed_input),
            "train_validation": str(args.train_validation_input),
            "test": str(args.test_input),
            "folds_dir": str(args.folds_dir),
        },
        "outputs": {
            "train_validation": str(train_validation_path),
            "test": str(test_path),
            "folds_dir": str(folds_output_dir),
            "manifest": str(manifest_path),
        },
        "global": processed_result.report,
        "train_validation": train_validation_result.report,
        "test": test_result.report,
        "folds": fold_reports,
    }
    write_json(manifest, manifest_path)
    record_stage_result(
        args,
        stage_name="classification_preprocessing",
        artifacts={
            "train_validation": train_validation_path,
            "test": test_path,
            "folds_dir": folds_output_dir,
            "manifest": manifest_path,
        },
        payload=manifest,
    )
    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "target_field": config.target_field,
                "text_fields": list(config.text_fields),
                "allowed_label_count": len(processed_result.allowed_labels),
                "train_validation_record_count": train_validation_result.report[
                    "output_record_count"
                ],
                "test_record_count": test_result.report["output_record_count"],
                "fold_count": len(fold_reports),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def _read_input_data_folds(folds_dir: Path) -> list[InputDataFold]:
    if not folds_dir.exists():
        raise SystemExit(f"{folds_dir} does not exist. Run ai-test split-datasets first.")

    folds: list[InputDataFold] = []
    for train_path in sorted(folds_dir.glob("fold_*_train.json")):
        fold_name = train_path.name.removesuffix("_train.json")
        validation_path = folds_dir / f"{fold_name}_validation.json"
        if not validation_path.exists():
            raise SystemExit(f"Missing validation file for {train_path}: {validation_path}")
        folds.append(
            InputDataFold(
                fold_index=_fold_index(fold_name),
                train_records=read_records(train_path),
                validation_records=read_records(validation_path),
            )
        )

    if not folds:
        raise SystemExit(f"No fold train files found in {folds_dir}.")
    return folds


def _fold_index(fold_name: str) -> int:
    try:
        return int(fold_name.removeprefix("fold_"))
    except ValueError as error:
        raise SystemExit(f"Cannot parse fold index from {fold_name}") from error

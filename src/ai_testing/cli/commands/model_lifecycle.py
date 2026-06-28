from __future__ import annotations

import argparse
import json
from pathlib import Path

from ai_testing.cli_common import read_json_object, read_records, record_stage_result, write_json
from ai_testing.model_testing import (
    AssociationTestConfig,
    TextClassifierTestConfig,
    test_association_rules,
    test_text_classifier,
)
from ai_testing.model_training import (
    AssociationRulesConfig,
    TextClassifierConfig,
    save_text_classifier_artifact,
    train_association_rules,
    train_text_classifier,
)
from ai_testing.model_validation import (
    AssociationValidationConfig,
    AssociationValidationFold,
    TextClassificationValidationConfig,
    TextClassificationValidationFold,
    validate_association_rules,
    validate_text_classifier,
)


def _train_category_classifier(args: argparse.Namespace) -> None:
    input_path = Path(args.input)
    output_path = Path(args.output)
    estimator_output_path = Path(args.estimator_output)
    records = read_records(input_path)
    try:
        result = train_text_classifier(
            records,
            config=TextClassifierConfig(
                text_field=str(args.text_field),
                label_field=str(args.label_field),
                alpha=float(args.alpha),
                min_token_length=int(args.min_token_length),
                max_vocabulary_size=int(args.max_vocabulary_size),
            ),
        )
    except ValueError as error:
        raise SystemExit(f"Cannot train category classifier: {error}") from error

    model = {
        **result.model,
        "training_input": str(input_path),
        "estimator_path": str(estimator_output_path),
        "note": "Trained on the supervised training split only. Hold-out test data is not used.",
    }
    try:
        save_text_classifier_artifact(
            model=model,
            estimator=result.estimator,
            manifest_path=output_path,
            estimator_path=estimator_output_path,
        )
    except ValueError as error:
        raise SystemExit(f"Cannot save category classifier: {error}") from error
    record_stage_result(
        args,
        stage_name="model_training.category_classifier",
        artifacts={"model": output_path, "estimator": estimator_output_path},
        payload=model,
    )
    summary = model["summary"]
    print(
        json.dumps(
            {
                "input": str(input_path),
                "output": str(output_path),
                "estimator_output": str(estimator_output_path),
                "model_type": model["model_type"],
                "algorithm": model["algorithm"],
                "framework": model["framework"],
                "training_example_count": summary["training_example_count"],
                "class_count": summary["class_count"],
                "vocabulary_size": summary["vocabulary_size"],
                "majority_class": summary["majority_class"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def _validate_category_classifier(args: argparse.Namespace) -> None:
    folds_dir = Path(args.folds_dir)
    output_path = Path(args.output)
    folds = _read_text_classification_folds(folds_dir)
    try:
        result = validate_text_classifier(
            folds,
            config=TextClassificationValidationConfig(
                text_field=str(args.text_field),
                label_field=str(args.label_field),
                alpha=float(args.alpha),
                min_token_length=int(args.min_token_length),
                max_vocabulary_size=int(args.max_vocabulary_size),
                top_confusions=int(args.top_confusions),
            ),
        )
    except ValueError as error:
        raise SystemExit(f"Cannot validate category classifier: {error}") from error

    report = {
        **result.report,
        "folds_dir": str(folds_dir),
        "output": str(output_path),
    }
    write_json(report, output_path)
    record_stage_result(
        args,
        stage_name="model_validation.category_classifier",
        artifacts={"validation_report": output_path},
        payload=report,
    )
    summary = report["summary"]
    print(
        json.dumps(
            {
                "folds_dir": str(folds_dir),
                "output": str(output_path),
                "fold_count": summary["fold_count"],
                "mean_accuracy": summary["mean_accuracy"],
                "std_accuracy": summary["std_accuracy"],
                "mean_macro_precision": summary["mean_macro_precision"],
                "mean_macro_recall": summary["mean_macro_recall"],
                "mean_macro_f1": summary["mean_macro_f1"],
                "mean_weighted_precision": summary["mean_weighted_precision"],
                "mean_weighted_recall": summary["mean_weighted_recall"],
                "mean_weighted_f1": summary["mean_weighted_f1"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def _test_category_classifier(args: argparse.Namespace) -> None:
    model_path = Path(args.model_input)
    test_input_path = Path(args.test_input)
    output_path = Path(args.output)
    model = read_json_object(model_path)
    test_records = read_records(test_input_path)
    try:
        result = test_text_classifier(
            model,
            test_records,
            config=TextClassifierTestConfig(
                text_field=str(args.text_field),
                label_field=str(args.label_field),
                top_confusions=int(args.top_confusions),
            ),
        )
    except ValueError as error:
        raise SystemExit(f"Cannot test category classifier: {error}") from error

    report = {
        **result.report,
        "model_input": str(model_path),
        "test_input": str(test_input_path),
        "output": str(output_path),
    }
    write_json(report, output_path)
    record_stage_result(
        args,
        stage_name="model_testing.category_classifier",
        artifacts={"test_report": output_path},
        payload=report,
    )
    test_summary = report["test"]
    print(
        json.dumps(
            {
                "model_input": str(model_path),
                "test_input": str(test_input_path),
                "output": str(output_path),
                "record_count": test_summary["record_count"],
                "evaluated_record_count": test_summary["evaluated_record_count"],
                "class_count": test_summary["class_count"],
                "accuracy": test_summary["accuracy"],
                "macro_precision": test_summary["macro_precision"],
                "macro_recall": test_summary["macro_recall"],
                "macro_f1": test_summary["macro_f1"],
                "weighted_precision": test_summary["weighted_precision"],
                "weighted_recall": test_summary["weighted_recall"],
                "weighted_f1": test_summary["weighted_f1"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def _train_associations(args: argparse.Namespace) -> None:
    input_path = Path(args.input)
    output_path = Path(args.output)
    records = read_records(input_path)
    try:
        result = train_association_rules(
            records,
            config=AssociationRulesConfig(
                basket_field=str(args.basket_field),
                item_field=str(args.item_field),
                min_support=float(args.min_support),
                min_confidence=float(args.min_confidence),
                min_lift=float(args.min_lift),
                max_itemset_size=int(args.max_itemset_size),
                max_rules=int(args.max_rules),
                max_itemsets=int(args.max_itemsets),
            ),
        )
    except ValueError as error:
        raise SystemExit(f"Cannot train association rules: {error}") from error

    model = {
        **result.model,
        "training_input": str(input_path),
        "note": "Trained on the training split only. Hold-out test data is not used.",
    }
    write_json(model, output_path)
    record_stage_result(
        args,
        stage_name="model_training.association_rules",
        artifacts={"model": output_path},
        payload=model,
    )
    summary = model["summary"]
    print(
        json.dumps(
            {
                "input": str(input_path),
                "output": str(output_path),
                "model_type": model["model_type"],
                "algorithm": model["algorithm"],
                "basket_count": summary["basket_count"],
                "item_count": summary["item_count"],
                "frequent_itemset_count": summary["frequent_itemset_count"],
                "rule_count": summary["rule_count"],
                "exported_rule_count": summary["exported_rule_count"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def _validate_associations(args: argparse.Namespace) -> None:
    folds_dir = Path(args.folds_dir)
    output_path = Path(args.output)
    folds = _read_association_validation_folds(folds_dir)
    try:
        result = validate_association_rules(
            folds,
            config=AssociationValidationConfig(
                basket_field=str(args.basket_field),
                item_field=str(args.item_field),
                min_support=float(args.min_support),
                min_confidence=float(args.min_confidence),
                min_lift=float(args.min_lift),
                max_itemset_size=int(args.max_itemset_size),
                max_rules=int(args.max_rules),
                max_itemsets=int(args.max_itemsets),
                top_rules=int(args.top_rules),
            ),
        )
    except ValueError as error:
        raise SystemExit(f"Cannot validate association rules: {error}") from error

    report = {
        **result.report,
        "folds_dir": str(folds_dir),
        "output": str(output_path),
    }
    write_json(report, output_path)
    record_stage_result(
        args,
        stage_name="model_validation.association_rules",
        artifacts={"validation_report": output_path},
        payload=report,
    )
    summary = report["summary"]
    print(
        json.dumps(
            {
                "folds_dir": str(folds_dir),
                "output": str(output_path),
                "fold_count": summary["fold_count"],
                "mean_train_rule_count": summary["mean_train_rule_count"],
                "mean_validation_confidence": summary["mean_validation_confidence"],
                "mean_validation_lift": summary["mean_validation_lift"],
                "mean_antecedent_coverage": summary["mean_antecedent_coverage"],
                "mean_hit_rate_per_covered_basket": summary["mean_hit_rate_per_covered_basket"],
                "mean_abs_confidence_gap": summary["mean_abs_confidence_gap"],
                "mean_stable_rule_count": summary["mean_stable_rule_count"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def _test_associations(args: argparse.Namespace) -> None:
    model_path = Path(args.model_input)
    test_input_path = Path(args.test_input)
    output_path = Path(args.output)
    model = read_json_object(model_path)
    test_records = read_records(test_input_path)
    try:
        result = test_association_rules(
            model,
            test_records,
            config=AssociationTestConfig(
                basket_field=str(args.basket_field),
                item_field=str(args.item_field),
                min_confidence=float(args.min_confidence),
                min_lift=float(args.min_lift),
                top_rules=int(args.top_rules),
            ),
        )
    except ValueError as error:
        raise SystemExit(f"Cannot test association rules: {error}") from error

    report = {
        **result.report,
        "model_input": str(model_path),
        "test_input": str(test_input_path),
        "output": str(output_path),
    }
    write_json(report, output_path)
    record_stage_result(
        args,
        stage_name="model_testing.association_rules",
        artifacts={"test_report": output_path},
        payload=report,
    )
    test_summary = report["test"]
    print(
        json.dumps(
            {
                "model_input": str(model_path),
                "test_input": str(test_input_path),
                "output": str(output_path),
                "record_count": test_summary["record_count"],
                "basket_count": test_summary["basket_count"],
                "evaluated_rule_count": test_summary["evaluated_rule_count"],
                "stable_rule_count": test_summary["stable_rule_count"],
                "mean_test_confidence": test_summary["mean_test_confidence"],
                "mean_test_lift": test_summary["mean_test_lift"],
                "antecedent_coverage": test_summary["antecedent_coverage"],
                "hit_rate_per_covered_basket": test_summary["hit_rate_per_covered_basket"],
                "mean_abs_confidence_gap": test_summary["mean_abs_confidence_gap"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def _read_association_validation_folds(folds_dir: Path) -> list[AssociationValidationFold]:
    if not folds_dir.exists():
        raise SystemExit(f"{folds_dir} does not exist. Run ai-test split-datasets first.")

    folds: list[AssociationValidationFold] = []
    for train_path in sorted(folds_dir.glob("fold_*_train.json")):
        fold_name = train_path.name.removesuffix("_train.json")
        validation_path = folds_dir / f"{fold_name}_validation.json"
        if not validation_path.exists():
            raise SystemExit(f"Missing validation file for {train_path}: {validation_path}")
        folds.append(
            AssociationValidationFold(
                fold_index=_fold_index(fold_name),
                train_records=read_records(train_path),
                validation_records=read_records(validation_path),
            )
        )

    if not folds:
        raise SystemExit(f"No fold train files found in {folds_dir}.")
    return folds


def _read_text_classification_folds(folds_dir: Path) -> list[TextClassificationValidationFold]:
    if not folds_dir.exists():
        raise SystemExit(
            f"{folds_dir} does not exist. Run ai-test build-classification-dataset first."
        )

    folds: list[TextClassificationValidationFold] = []
    for train_path in sorted(folds_dir.glob("fold_*_train.json")):
        fold_name = train_path.name.removesuffix("_train.json")
        validation_path = folds_dir / f"{fold_name}_validation.json"
        if not validation_path.exists():
            raise SystemExit(f"Missing validation file for {train_path}: {validation_path}")
        folds.append(
            TextClassificationValidationFold(
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

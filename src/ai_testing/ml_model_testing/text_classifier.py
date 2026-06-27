from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import PurePath
from typing import Any

from ai_testing.core import add_check as _add_check
from ai_testing.core import add_threshold_check as _add_threshold_check
from ai_testing.core import build_standard_report

Record = dict[str, Any]


@dataclass(frozen=True)
class TextClassifierMLModelTestConfig:
    expected_model_type: str = "text_classification"
    expected_algorithm: str = "multinomial_naive_bayes"
    expected_framework: str = "scikit-learn"
    expected_artifact_format: str = "sklearn_joblib"
    expected_learning_type: str = "supervised"
    test_dataset_input: str = "data/classification/category/test.json"
    forbidden_feature_fields: tuple[str, ...] = (
        "order_id",
        "order_number",
        "order_item_id",
        "product_id",
        "product_slug",
        "product_url",
        "main_category_id",
        "category_id",
        "category_ids",
        "main_category",
        "category",
        "category_path",
        "product_group",
    )
    min_validation_fold_count: int = 2
    min_training_example_count: int = 1
    min_train_class_count: int = 2
    min_train_vocabulary_size: int = 1
    min_test_record_count: int = 1
    min_test_class_count: int = 2
    min_mean_validation_accuracy: float = 0.9
    min_mean_validation_macro_precision: float = 0.7
    min_mean_validation_macro_recall: float = 0.7
    min_mean_validation_macro_f1: float = 0.7
    min_mean_validation_weighted_f1: float = 0.9
    max_validation_accuracy_std: float = 0.05
    max_validation_macro_f1_std: float = 0.08
    min_test_accuracy: float = 0.9
    min_test_macro_precision: float = 0.7
    min_test_macro_recall: float = 0.7
    min_test_macro_f1: float = 0.7
    min_test_weighted_f1: float = 0.9
    max_validation_test_accuracy_delta: float = 0.05
    max_validation_test_macro_f1_delta: float = 0.08


@dataclass(frozen=True)
class TextClassifierMLModelTestResult:
    report: Record


def test_text_classifier_ml_model(
    model: Mapping[str, Any],
    validation_report: Mapping[str, Any],
    test_report: Mapping[str, Any],
    classification_manifest: Mapping[str, Any],
    config: TextClassifierMLModelTestConfig | None = None,
) -> TextClassifierMLModelTestResult:
    test_config = config or TextClassifierMLModelTestConfig()
    checks: list[Record] = []
    model_summary = _mapping(model.get("summary"))
    model_parameters = _mapping(model.get("parameters"))
    validation_summary = _mapping(validation_report.get("summary"))
    test_summary = _mapping(test_report.get("test"))
    manifest_global = _mapping(classification_manifest.get("global"))
    source_text_fields = _string_sequence(manifest_global.get("text_fields"))
    target_field = _string_value(manifest_global.get("target_field"))
    forbidden_fields = set(test_config.forbidden_feature_fields)

    _add_check(
        checks,
        check_id="ml_model.artifact_identity",
        passed=(
            model.get("model_type") == test_config.expected_model_type
            and model.get("algorithm") == test_config.expected_algorithm
            and model.get("framework") == test_config.expected_framework
            and model.get("artifact_format") == test_config.expected_artifact_format
            and model.get("learning_type") == test_config.expected_learning_type
        ),
        severity="critical",
        message="Model artifact has the expected type, algorithm, and learning type.",
        observed={
            "model_type": model.get("model_type"),
            "algorithm": model.get("algorithm"),
            "framework": model.get("framework"),
            "artifact_format": model.get("artifact_format"),
            "learning_type": model.get("learning_type"),
        },
        expected={
            "model_type": test_config.expected_model_type,
            "algorithm": test_config.expected_algorithm,
            "framework": test_config.expected_framework,
            "artifact_format": test_config.expected_artifact_format,
            "learning_type": test_config.expected_learning_type,
        },
    )
    _add_check(
        checks,
        check_id="ml_model.estimator_artifact_reference",
        passed=_string_value(model.get("estimator_path")) is not None,
        severity="critical",
        message="Model manifest references the persisted estimator artifact.",
        observed={"estimator_path": model.get("estimator_path")},
        expected={"estimator_path": "non-empty"},
    )
    _add_check(
        checks,
        check_id="ml_model.training_input_is_not_test",
        passed=not _same_path(model.get("training_input"), test_config.test_dataset_input),
        severity="critical",
        message="Model training input is not the final hold-out test dataset.",
        observed={"training_input": model.get("training_input")},
        expected={"not_equal_to": test_config.test_dataset_input},
    )
    _add_check(
        checks,
        check_id="ml_model.text_feature_field_is_safe",
        passed=_string_value(model_parameters.get("text_field")) not in forbidden_fields,
        severity="critical",
        message="The trained text field is not a technical identifier or target leakage field.",
        observed={"text_field": model_parameters.get("text_field")},
        expected={"not_in": sorted(forbidden_fields)},
    )
    _add_check(
        checks,
        check_id="ml_model.label_field_shape",
        passed=_string_value(model_parameters.get("label_field")) == "label",
        severity="critical",
        message="The supervised label field uses the normalized classification label column.",
        observed={"label_field": model_parameters.get("label_field")},
        expected={"label_field": "label"},
    )
    _add_check(
        checks,
        check_id="ml_model.classification_manifest_shape",
        passed=(
            classification_manifest.get("step") == "Supervised classification dataset preprocessing"
            and manifest_global.get("learning_type") == test_config.expected_learning_type
            and len(source_text_fields) > 0
            and target_field is not None
        ),
        severity="critical",
        message="Classification manifest describes supervised text/label preprocessing.",
        observed={
            "step": classification_manifest.get("step"),
            "learning_type": manifest_global.get("learning_type"),
            "text_fields": list(source_text_fields),
            "target_field": target_field,
        },
        expected={"learning_type": test_config.expected_learning_type, "text_fields": "non-empty"},
    )
    _add_check(
        checks,
        check_id="ml_model.source_feature_leakage",
        passed=not (set(source_text_fields) & forbidden_fields)
        and target_field not in source_text_fields,
        severity="critical",
        message=(
            "Source text fields do not include identifiers, category targets, or derived labels."
        ),
        observed={
            "text_fields": list(source_text_fields),
            "target_field": target_field,
            "forbidden_intersection": sorted(set(source_text_fields) & forbidden_fields),
        },
        expected={"forbidden_intersection": [], "target_not_in_text_fields": True},
    )
    _add_threshold_check(
        checks,
        check_id="ml_model.training_example_count",
        value=_float_value(model_summary.get("training_example_count")),
        threshold=float(test_config.min_training_example_count),
        direction=">=",
        severity="critical",
        message="Model was trained on enough labeled examples.",
    )
    _add_threshold_check(
        checks,
        check_id="ml_model.train_class_count",
        value=_float_value(model_summary.get("class_count")),
        threshold=float(test_config.min_train_class_count),
        direction=">=",
        severity="critical",
        message="Model was trained with enough target classes.",
    )
    _add_threshold_check(
        checks,
        check_id="ml_model.train_vocabulary_size",
        value=_float_value(model_summary.get("vocabulary_size")),
        threshold=float(test_config.min_train_vocabulary_size),
        direction=">=",
        severity="critical",
        message="Model vocabulary is large enough to make text predictions.",
    )
    _add_check(
        checks,
        check_id="ml_model.validation_report_shape",
        passed=(
            validation_report.get("validation_type") == "k-fold cross-validation"
            and validation_report.get("model_type") == test_config.expected_model_type
            and validation_report.get("learning_type") == test_config.expected_learning_type
            and _int_value(validation_summary.get("fold_count"))
            >= test_config.min_validation_fold_count
        ),
        severity="critical",
        message="Validation report is based on k-fold supervised classification validation.",
        observed={
            "validation_type": validation_report.get("validation_type"),
            "model_type": validation_report.get("model_type"),
            "learning_type": validation_report.get("learning_type"),
            "fold_count": validation_summary.get("fold_count"),
        },
        expected={"fold_count": f">= {test_config.min_validation_fold_count}"},
    )
    _add_threshold_check(
        checks,
        check_id="ml_model.validation_accuracy",
        value=_float_value(validation_summary.get("mean_accuracy")),
        threshold=test_config.min_mean_validation_accuracy,
        direction=">=",
        severity="major",
        message="Mean validation accuracy meets the acceptance threshold.",
    )
    _add_threshold_check(
        checks,
        check_id="ml_model.validation_macro_precision",
        value=_float_value(validation_summary.get("mean_macro_precision")),
        threshold=test_config.min_mean_validation_macro_precision,
        direction=">=",
        severity="major",
        message="Mean validation macro precision meets the acceptance threshold.",
    )
    _add_threshold_check(
        checks,
        check_id="ml_model.validation_macro_recall",
        value=_float_value(validation_summary.get("mean_macro_recall")),
        threshold=test_config.min_mean_validation_macro_recall,
        direction=">=",
        severity="major",
        message="Mean validation macro recall meets the acceptance threshold.",
    )
    _add_threshold_check(
        checks,
        check_id="ml_model.validation_macro_f1",
        value=_float_value(validation_summary.get("mean_macro_f1")),
        threshold=test_config.min_mean_validation_macro_f1,
        direction=">=",
        severity="major",
        message="Mean validation macro F1 meets the acceptance threshold.",
    )
    _add_threshold_check(
        checks,
        check_id="ml_model.validation_weighted_f1",
        value=_float_value(validation_summary.get("mean_weighted_f1")),
        threshold=test_config.min_mean_validation_weighted_f1,
        direction=">=",
        severity="major",
        message="Mean validation weighted F1 meets the acceptance threshold.",
    )
    _add_threshold_check(
        checks,
        check_id="ml_model.validation_accuracy_std",
        value=_float_value(validation_summary.get("std_accuracy")),
        threshold=test_config.max_validation_accuracy_std,
        direction="<=",
        severity="major",
        message="Validation accuracy is stable across folds.",
    )
    _add_threshold_check(
        checks,
        check_id="ml_model.validation_macro_f1_std",
        value=_float_value(validation_summary.get("std_macro_f1")),
        threshold=test_config.max_validation_macro_f1_std,
        direction="<=",
        severity="major",
        message="Validation macro F1 is stable across folds.",
    )
    _add_check(
        checks,
        check_id="ml_model.test_report_shape",
        passed=(
            test_report.get("testing_type") == "hold-out test"
            and test_report.get("model_type") == test_config.expected_model_type
            and test_report.get("learning_type") == test_config.expected_learning_type
        ),
        severity="critical",
        message="Final test report is based on the hold-out supervised classification dataset.",
        observed={
            "testing_type": test_report.get("testing_type"),
            "model_type": test_report.get("model_type"),
            "learning_type": test_report.get("learning_type"),
        },
        expected={"testing_type": "hold-out test"},
    )
    _add_threshold_check(
        checks,
        check_id="ml_model.test_record_count",
        value=_float_value(test_summary.get("evaluated_record_count")),
        threshold=float(test_config.min_test_record_count),
        direction=">=",
        severity="critical",
        message="Hold-out test report evaluates enough labeled records.",
    )
    _add_threshold_check(
        checks,
        check_id="ml_model.test_class_count",
        value=_float_value(test_summary.get("class_count")),
        threshold=float(test_config.min_test_class_count),
        direction=">=",
        severity="critical",
        message="Hold-out test report evaluates enough target classes.",
    )
    _add_threshold_check(
        checks,
        check_id="ml_model.test_accuracy",
        value=_float_value(test_summary.get("accuracy")),
        threshold=test_config.min_test_accuracy,
        direction=">=",
        severity="major",
        message="Hold-out test accuracy meets the acceptance threshold.",
    )
    _add_threshold_check(
        checks,
        check_id="ml_model.test_macro_precision",
        value=_float_value(test_summary.get("macro_precision")),
        threshold=test_config.min_test_macro_precision,
        direction=">=",
        severity="major",
        message="Hold-out test macro precision meets the acceptance threshold.",
    )
    _add_threshold_check(
        checks,
        check_id="ml_model.test_macro_recall",
        value=_float_value(test_summary.get("macro_recall")),
        threshold=test_config.min_test_macro_recall,
        direction=">=",
        severity="major",
        message="Hold-out test macro recall meets the acceptance threshold.",
    )
    _add_threshold_check(
        checks,
        check_id="ml_model.test_macro_f1",
        value=_float_value(test_summary.get("macro_f1")),
        threshold=test_config.min_test_macro_f1,
        direction=">=",
        severity="major",
        message="Hold-out test macro F1 meets the acceptance threshold.",
    )
    _add_threshold_check(
        checks,
        check_id="ml_model.test_weighted_f1",
        value=_float_value(test_summary.get("weighted_f1")),
        threshold=test_config.min_test_weighted_f1,
        direction=">=",
        severity="major",
        message="Hold-out test weighted F1 meets the acceptance threshold.",
    )
    _add_threshold_check(
        checks,
        check_id="ml_model.validation_test_accuracy_delta",
        value=_absolute_delta(
            validation_summary.get("mean_accuracy"), test_summary.get("accuracy")
        ),
        threshold=test_config.max_validation_test_accuracy_delta,
        direction="<=",
        severity="major",
        message="Validation and test accuracy are close enough.",
    )
    _add_threshold_check(
        checks,
        check_id="ml_model.validation_test_macro_f1_delta",
        value=_absolute_delta(
            validation_summary.get("mean_macro_f1"),
            test_summary.get("macro_f1"),
        ),
        threshold=test_config.max_validation_test_macro_f1_delta,
        direction="<=",
        severity="major",
        message="Validation and test macro F1 are close enough.",
    )

    return TextClassifierMLModelTestResult(
        report=build_standard_report(
            report_type="model_quality_test",
            subject="category_classifier",
            step="ML model testing",
            testing_type="supervised classification model acceptance testing",
            checks=checks,
            details={
                "model": {
                    "model_type": model.get("model_type"),
                    "algorithm": model.get("algorithm"),
                    "framework": model.get("framework"),
                    "artifact_format": model.get("artifact_format"),
                    "learning_type": model.get("learning_type"),
                    "estimator_path": model.get("estimator_path"),
                    "training_input": model.get("training_input"),
                    "training_example_count": model_summary.get("training_example_count"),
                    "class_count": model_summary.get("class_count"),
                    "vocabulary_size": model_summary.get("vocabulary_size"),
                    "text_field": model_parameters.get("text_field"),
                    "label_field": model_parameters.get("label_field"),
                },
                "dataset_lineage": {
                    "test_dataset_input": test_config.test_dataset_input,
                    "source_text_fields": list(source_text_fields),
                    "target_field": target_field,
                },
                "validation_metrics": validation_summary,
                "test_metrics": test_summary,
            },
        )
    )


def _same_path(left: Any, right: str) -> bool:
    left_text = _string_value(left)
    if left_text is None:
        return False
    return PurePath(left_text).as_posix().lower() == PurePath(right).as_posix().lower()


def _absolute_delta(left: Any, right: Any) -> float | None:
    left_number = _float_value(left)
    right_number = _float_value(right)
    if left_number is None or right_number is None:
        return None
    return round(abs(left_number - right_number), 6)


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _int_value(value: Any) -> int:
    number = _float_value(value)
    return int(number) if number is not None else 0


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


def _string_value(value: Any) -> str | None:
    if value is None or isinstance(value, bool):
        return None
    text = str(value).strip()
    return text or None


def _string_sequence(value: Any) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return ()
    return tuple(text for item in value for text in [_string_value(item)] if text is not None)

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from statistics import mean, pstdev
from typing import Any

from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from ai_testing.model_training import (
    TextClassifierConfig,
    TextClassifierEstimator,
    predict_text_classifier,
    train_text_classifier,
)

Record = dict[str, Any]


@dataclass(frozen=True)
class TextClassificationEvaluationConfig:
    text_field: str = "text"
    label_field: str = "label"
    top_confusions: int = 20


@dataclass(frozen=True)
class TextClassificationValidationConfig:
    text_field: str = "text"
    label_field: str = "label"
    alpha: float = 1.0
    min_token_length: int = 2
    max_vocabulary_size: int = 5000
    top_confusions: int = 20


@dataclass(frozen=True)
class TextClassificationValidationFold:
    fold_index: int
    train_records: list[Record]
    validation_records: list[Record]


@dataclass(frozen=True)
class TextClassificationValidationResult:
    report: Record


@dataclass(frozen=True)
class TextClassificationEvaluationResult:
    report: Record


def validate_text_classifier(
    folds: Sequence[TextClassificationValidationFold],
    config: TextClassificationValidationConfig | None = None,
) -> TextClassificationValidationResult:
    validation_config = config or TextClassificationValidationConfig()
    _validate_validation_config(validation_config)
    if not folds:
        raise ValueError("At least one validation fold is required.")

    fold_reports = [
        _validate_fold(fold=fold, config=validation_config)
        for fold in sorted(folds, key=lambda item: item.fold_index)
    ]
    report = {
        "step": "Text classification validation",
        "validation_type": "k-fold cross-validation",
        "model_type": "text_classification",
        "algorithm": "multinomial_naive_bayes",
        "framework": "scikit-learn",
        "learning_type": "supervised",
        "parameters": _validation_parameters(validation_config),
        "summary": _summary(fold_reports),
        "folds": fold_reports,
    }
    return TextClassificationValidationResult(report=report)


def evaluate_text_classifier(
    model: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
    config: TextClassificationEvaluationConfig | None = None,
    dataset_name: str = "evaluation",
    estimator: TextClassifierEstimator | None = None,
) -> TextClassificationEvaluationResult:
    evaluation_config = config or TextClassificationEvaluationConfig()
    _validate_evaluation_config(evaluation_config)
    normalized_records = [dict(record) for record in records]
    labels = [
        label
        for record in normalized_records
        for label in [_string_value(record.get(evaluation_config.label_field))]
        if label is not None
    ]
    predictions = predict_text_classifier(
        model,
        normalized_records,
        text_field=evaluation_config.text_field,
        estimator=estimator,
    )
    paired = [
        (label, prediction)
        for record, prediction in zip(normalized_records, predictions, strict=True)
        for label in [_string_value(record.get(evaluation_config.label_field))]
        if label is not None
    ]
    metrics = _classification_metrics(paired, evaluation_config.top_confusions)
    return TextClassificationEvaluationResult(
        report={
            "dataset": dataset_name,
            "model_type": model.get("model_type"),
            "algorithm": model.get("algorithm"),
            "framework": model.get("framework"),
            "artifact_format": model.get("artifact_format"),
            "learning_type": model.get("learning_type"),
            "training_input": model.get("training_input"),
            "record_count": len(normalized_records),
            "labeled_record_count": len(labels),
            "metrics": metrics,
        }
    )


def _validate_validation_config(config: TextClassificationValidationConfig) -> None:
    TextClassifierConfig(
        text_field=config.text_field,
        label_field=config.label_field,
        alpha=config.alpha,
        min_token_length=config.min_token_length,
        max_vocabulary_size=config.max_vocabulary_size,
    )
    if config.top_confusions < 1:
        raise ValueError("top_confusions must be at least 1")


def _validate_evaluation_config(config: TextClassificationEvaluationConfig) -> None:
    if not config.text_field:
        raise ValueError("text_field must not be empty")
    if not config.label_field:
        raise ValueError("label_field must not be empty")
    if config.top_confusions < 1:
        raise ValueError("top_confusions must be at least 1")


def _validate_fold(
    fold: TextClassificationValidationFold,
    config: TextClassificationValidationConfig,
) -> Record:
    training_result = train_text_classifier(
        fold.train_records,
        config=TextClassifierConfig(
            text_field=config.text_field,
            label_field=config.label_field,
            alpha=config.alpha,
            min_token_length=config.min_token_length,
            max_vocabulary_size=config.max_vocabulary_size,
        ),
    )
    evaluation_result = evaluate_text_classifier(
        training_result.model,
        fold.validation_records,
        config=TextClassificationEvaluationConfig(
            text_field=config.text_field,
            label_field=config.label_field,
            top_confusions=config.top_confusions,
        ),
        dataset_name=f"fold_{fold.fold_index:02d}_validation",
        estimator=training_result.estimator,
    )
    return {
        "fold_index": fold.fold_index,
        "train": {
            "record_count": len(fold.train_records),
            **_training_summary(training_result.model),
        },
        "validation": {
            "record_count": len(fold.validation_records),
            **evaluation_result.report["metrics"],
        },
    }


def _classification_metrics(
    pairs: Sequence[tuple[str, str]],
    top_confusions: int,
) -> Record:
    labels = sorted({label for label, _ in pairs} | {prediction for _, prediction in pairs})
    y_true = [label for label, _ in pairs]
    y_pred = [prediction for _, prediction in pairs]
    precision_values, recall_values, f1_values, support_values = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels,
        zero_division=0,
    )
    per_class = {
        label: {
            "precision": _round(float(precision)),
            "recall": _round(float(recall)),
            "f1": _round(float(f1)),
            "support": int(support),
        }
        for label, precision, recall, f1, support in zip(
            labels,
            precision_values,
            recall_values,
            f1_values,
            support_values,
            strict=True,
        )
        if int(support) > 0
    }
    return {
        "evaluated_record_count": len(pairs),
        "class_count": len(per_class),
        "accuracy": _round(float(accuracy_score(y_true, y_pred))) if pairs else 0.0,
        "macro_precision": _average_metric(y_true, y_pred, labels, "macro", 0),
        "macro_recall": _average_metric(y_true, y_pred, labels, "macro", 1),
        "macro_f1": _average_metric(y_true, y_pred, labels, "macro", 2),
        "weighted_precision": _average_metric(y_true, y_pred, labels, "weighted", 0),
        "weighted_recall": _average_metric(y_true, y_pred, labels, "weighted", 1),
        "weighted_f1": _average_metric(y_true, y_pred, labels, "weighted", 2),
        "per_class": per_class,
        "top_confusions": _top_confusions(pairs, top_confusions),
    }


def _average_metric(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    labels: Sequence[str],
    average: str,
    metric_index: int,
) -> float:
    if not y_true:
        return 0.0
    metrics = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels,
        average=average,
        zero_division=0,
    )
    return _round(float(metrics[metric_index]))


def _top_confusions(pairs: Sequence[tuple[str, str]], limit: int) -> list[Record]:
    counts = Counter((actual, predicted) for actual, predicted in pairs if actual != predicted)
    return [
        {"actual": actual, "predicted": predicted, "count": count}
        for (actual, predicted), count in sorted(
            counts.items(),
            key=lambda item: (-item[1], item[0][0], item[0][1]),
        )[:limit]
    ]


def _summary(fold_reports: Sequence[Record]) -> Record:
    validation_reports = [dict(report["validation"]) for report in fold_reports]
    return {
        "fold_count": len(fold_reports),
        "mean_train_class_count": _mean(
            _number_path(report, "train", "class_count") for report in fold_reports
        ),
        "mean_train_vocabulary_size": _mean(
            _number_path(report, "train", "vocabulary_size") for report in fold_reports
        ),
        "mean_accuracy": _mean_metric(validation_reports, "accuracy"),
        "std_accuracy": _std_metric(validation_reports, "accuracy"),
        "mean_macro_precision": _mean_metric(validation_reports, "macro_precision"),
        "std_macro_precision": _std_metric(validation_reports, "macro_precision"),
        "mean_macro_recall": _mean_metric(validation_reports, "macro_recall"),
        "std_macro_recall": _std_metric(validation_reports, "macro_recall"),
        "mean_macro_f1": _mean_metric(validation_reports, "macro_f1"),
        "std_macro_f1": _std_metric(validation_reports, "macro_f1"),
        "mean_weighted_precision": _mean_metric(validation_reports, "weighted_precision"),
        "mean_weighted_recall": _mean_metric(validation_reports, "weighted_recall"),
        "mean_weighted_f1": _mean_metric(validation_reports, "weighted_f1"),
    }


def _validation_parameters(config: TextClassificationValidationConfig) -> Record:
    return {
        "text_field": config.text_field,
        "label_field": config.label_field,
        "alpha": config.alpha,
        "min_token_length": config.min_token_length,
        "max_vocabulary_size": config.max_vocabulary_size,
        "top_confusions": config.top_confusions,
    }


def _training_summary(model: Mapping[str, Any]) -> Record:
    summary = model.get("summary")
    return dict(summary) if isinstance(summary, Mapping) else {}


def _weighted_f1(per_class: Mapping[str, Mapping[str, Any]]) -> float | None:
    return _weighted_metric(per_class, "f1")


def _weighted_metric(
    per_class: Mapping[str, Mapping[str, Any]],
    metric_name: str,
) -> float | None:
    total_support = sum(_int_value(metric.get("support")) for metric in per_class.values())
    if total_support == 0:
        return None
    weighted_sum = sum(
        (_float_value(metric.get(metric_name)) or 0.0) * _int_value(metric.get("support"))
        for metric in per_class.values()
    )
    return _round(weighted_sum / total_support)


def _mean_metric(records: Sequence[Mapping[str, Any]], key: str) -> float | None:
    return _mean(_numbers(record.get(key) for record in records))


def _std_metric(records: Sequence[Mapping[str, Any]], key: str) -> float | None:
    values = _numbers(record.get(key) for record in records)
    if not values:
        return None
    return _round(pstdev(values))


def _numbers(values: Iterable[Any]) -> list[float]:
    return [number for value in values for number in [_float_value(value)] if number is not None]


def _number_path(record: Mapping[str, Any], first: str, second: str) -> float:
    nested = record.get(first)
    if not isinstance(nested, Mapping):
        return 0.0
    return _float_value(nested.get(second)) or 0.0


def _float_value(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _int_value(value: Any) -> int:
    number = _float_value(value)
    return int(number) if number is not None else 0


def _string_value(value: Any) -> str | None:
    if value is None or isinstance(value, bool):
        return None
    text = str(value).strip()
    return text or None


def _mean(values: Iterable[float]) -> float | None:
    numeric_values = list(values)
    if not numeric_values:
        return None
    return _round(mean(numeric_values))


def _round(value: float) -> float:
    return round(value, 6)

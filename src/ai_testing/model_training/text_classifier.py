from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

Record = dict[str, Any]

TOKEN_PATTERN = re.compile(r"[^\W_]+", re.UNICODE)


@dataclass(frozen=True)
class TextClassifierConfig:
    text_field: str = "text"
    label_field: str = "label"
    alpha: float = 1.0
    min_token_length: int = 2
    max_vocabulary_size: int = 5000


@dataclass(frozen=True)
class TextClassifierTrainingResult:
    model: Record


def train_text_classifier(
    records: Sequence[Mapping[str, Any]],
    config: TextClassifierConfig | None = None,
) -> TextClassifierTrainingResult:
    training_config = config or TextClassifierConfig()
    _validate_config(training_config)

    examples = _examples(records, training_config)
    if not examples:
        return TextClassifierTrainingResult(
            model=_empty_model(records, training_config, "No labeled text examples found.")
        )

    label_counts: Counter[str] = Counter(label for _, label in examples)
    vocabulary = _vocabulary(examples, training_config)
    vocabulary_set = set(vocabulary)
    token_counts_by_label: dict[str, Counter[str]] = {
        label: Counter() for label in sorted(label_counts)
    }
    for tokens, label in examples:
        token_counts_by_label[label].update(token for token in tokens if token in vocabulary_set)

    token_totals = {
        label: sum(token_counts.values()) for label, token_counts in token_counts_by_label.items()
    }
    model = {
        "model_type": "text_classification",
        "algorithm": "multinomial_naive_bayes",
        "learning_type": "supervised",
        "training_dataset": "training",
        "parameters": {
            "text_field": training_config.text_field,
            "label_field": training_config.label_field,
            "alpha": training_config.alpha,
            "min_token_length": training_config.min_token_length,
            "max_vocabulary_size": training_config.max_vocabulary_size,
        },
        "summary": {
            "input_record_count": len(records),
            "training_example_count": len(examples),
            "class_count": len(label_counts),
            "vocabulary_size": len(vocabulary),
            "majority_class": label_counts.most_common(1)[0][0],
        },
        "classes": sorted(label_counts),
        "class_counts": dict(sorted(label_counts.items())),
        "class_token_totals": dict(sorted(token_totals.items())),
        "vocabulary": vocabulary,
        "token_counts_by_class": {
            label: dict(sorted(token_counts_by_label[label].items()))
            for label in sorted(token_counts_by_label)
        },
    }
    return TextClassifierTrainingResult(model=model)


def predict_text_classifier(
    model: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
    text_field: str | None = None,
) -> list[str]:
    parameters = _mapping(model.get("parameters"))
    field = text_field or _string_value(parameters.get("text_field")) or "text"
    min_token_length = _int_value(parameters.get("min_token_length"), 2)
    return [_predict_one(model, _tokens(record.get(field), min_token_length)) for record in records]


def _validate_config(config: TextClassifierConfig) -> None:
    if not config.text_field:
        raise ValueError("text_field must not be empty")
    if not config.label_field:
        raise ValueError("label_field must not be empty")
    if config.alpha <= 0:
        raise ValueError("alpha must be greater than 0")
    if config.min_token_length < 1:
        raise ValueError("min_token_length must be at least 1")
    if config.max_vocabulary_size < 1:
        raise ValueError("max_vocabulary_size must be at least 1")


def _examples(
    records: Sequence[Mapping[str, Any]],
    config: TextClassifierConfig,
) -> list[tuple[tuple[str, ...], str]]:
    examples: list[tuple[tuple[str, ...], str]] = []
    for record in records:
        label = _string_value(record.get(config.label_field))
        tokens = _tokens(record.get(config.text_field), config.min_token_length)
        if label is None or not tokens:
            continue
        examples.append((tokens, label))
    return examples


def _vocabulary(
    examples: Sequence[tuple[tuple[str, ...], str]],
    config: TextClassifierConfig,
) -> list[str]:
    counts: Counter[str] = Counter(token for tokens, _ in examples for token in tokens)
    return [
        token
        for token, _ in sorted(counts.items(), key=lambda item: (-item[1], item[0]))[
            : config.max_vocabulary_size
        ]
    ]


def _predict_one(model: Mapping[str, Any], tokens: Sequence[str]) -> str:
    classes = _string_sequence(model.get("classes"))
    if not classes:
        return _string_value(_mapping(model.get("summary")).get("majority_class")) or ""

    vocabulary = set(_string_sequence(model.get("vocabulary")))
    vocabulary_size = len(vocabulary)
    class_counts = _number_mapping(model.get("class_counts"))
    token_totals = _number_mapping(model.get("class_token_totals"))
    token_counts_by_class = _token_counts_by_class(model.get("token_counts_by_class"))
    parameters = _mapping(model.get("parameters"))
    alpha = _float_value(parameters.get("alpha"), 1.0)
    total_examples = sum(class_counts.values())
    token_counter = Counter(token for token in tokens if token in vocabulary)
    if not token_counter:
        return _majority_class(class_counts)

    scores = {
        label: _score_label(
            label=label,
            token_counter=token_counter,
            class_counts=class_counts,
            token_totals=token_totals,
            token_counts_by_class=token_counts_by_class,
            alpha=alpha,
            vocabulary_size=vocabulary_size,
            total_examples=total_examples,
        )
        for label in classes
    }
    return sorted(scores.items(), key=lambda item: (-item[1], item[0]))[0][0]


def _score_label(
    label: str,
    token_counter: Counter[str],
    class_counts: Mapping[str, float],
    token_totals: Mapping[str, float],
    token_counts_by_class: Mapping[str, Mapping[str, float]],
    alpha: float,
    vocabulary_size: int,
    total_examples: float,
) -> float:
    class_count = class_counts.get(label, 0.0)
    if total_examples <= 0 or class_count <= 0:
        return float("-inf")

    score = math.log(class_count / total_examples)
    denominator = token_totals.get(label, 0.0) + alpha * vocabulary_size
    label_token_counts = token_counts_by_class.get(label, {})
    for token, count in token_counter.items():
        numerator = label_token_counts.get(token, 0.0) + alpha
        score += count * math.log(numerator / denominator)
    return score


def _empty_model(
    records: Sequence[Mapping[str, Any]],
    config: TextClassifierConfig,
    warning: str,
) -> Record:
    return {
        "model_type": "text_classification",
        "algorithm": "multinomial_naive_bayes",
        "learning_type": "supervised",
        "training_dataset": "training",
        "parameters": {
            "text_field": config.text_field,
            "label_field": config.label_field,
            "alpha": config.alpha,
            "min_token_length": config.min_token_length,
            "max_vocabulary_size": config.max_vocabulary_size,
        },
        "summary": {
            "input_record_count": len(records),
            "training_example_count": 0,
            "class_count": 0,
            "vocabulary_size": 0,
            "majority_class": None,
        },
        "warnings": [warning],
        "classes": [],
        "class_counts": {},
        "class_token_totals": {},
        "vocabulary": [],
        "token_counts_by_class": {},
    }


def _tokens(value: Any, min_token_length: int) -> tuple[str, ...]:
    text = _string_value(value)
    if text is None:
        return ()
    return tuple(
        token.lower() for token in TOKEN_PATTERN.findall(text) if len(token) >= min_token_length
    )


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _number_mapping(value: Any) -> dict[str, float]:
    if not isinstance(value, Mapping):
        return {}
    return {
        str(key): float(item)
        for key, item in value.items()
        if isinstance(item, (int, float)) and not isinstance(item, bool)
    }


def _token_counts_by_class(value: Any) -> dict[str, dict[str, float]]:
    if not isinstance(value, Mapping):
        return {}
    result: dict[str, dict[str, float]] = {}
    for label, token_counts in value.items():
        if isinstance(token_counts, Mapping):
            result[str(label)] = _number_mapping(token_counts)
    return result


def _string_sequence(value: Any) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, str):
        return ()
    return tuple(text for item in value for text in [_string_value(item)] if text is not None)


def _majority_class(class_counts: Mapping[str, float]) -> str:
    if not class_counts:
        return ""
    return sorted(class_counts.items(), key=lambda item: (-item[1], item[0]))[0][0]


def _int_value(value: Any, default: int) -> int:
    if isinstance(value, bool):
        return default
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    return default


def _float_value(value: Any, default: float) -> float:
    if isinstance(value, bool):
        return default
    if isinstance(value, (int, float)):
        return float(value)
    return default


def _string_value(value: Any) -> str | None:
    if value is None or isinstance(value, bool):
        return None
    text = str(value).strip()
    return text or None

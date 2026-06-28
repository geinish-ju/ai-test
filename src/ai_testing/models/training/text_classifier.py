from __future__ import annotations

import json
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, cast

import joblib
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

Record = dict[str, Any]


class TextClassifierEstimator(Protocol):
    named_steps: Mapping[str, Any]

    def fit(self, x: Sequence[str], y: Sequence[str]) -> object: ...

    def predict(self, x: Sequence[str]) -> Sequence[Any]: ...


TOKEN_PATTERN_TEMPLATE = r"(?u)\b\w{%d,}\b"


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
    estimator: TextClassifierEstimator | None


def train_text_classifier(
    records: Sequence[Mapping[str, Any]],
    config: TextClassifierConfig | None = None,
) -> TextClassifierTrainingResult:
    training_config = config or TextClassifierConfig()
    _validate_config(training_config)

    examples = _examples(records, training_config)
    if not examples:
        return TextClassifierTrainingResult(
            model=_empty_model(records, training_config, "No labeled text examples found."),
            estimator=None,
        )

    texts = [text for text, _ in examples]
    labels = [label for _, label in examples]
    label_counts: Counter[str] = Counter(labels)
    estimator = Pipeline(
        steps=[
            (
                "vectorizer",
                CountVectorizer(
                    lowercase=True,
                    max_features=training_config.max_vocabulary_size,
                    token_pattern=TOKEN_PATTERN_TEMPLATE % training_config.min_token_length,
                ),
            ),
            ("classifier", MultinomialNB(alpha=training_config.alpha)),
        ]
    )
    estimator.fit(texts, labels)

    vocabulary = _vocabulary(estimator)
    model = {
        "model_type": "text_classification",
        "algorithm": "multinomial_naive_bayes",
        "framework": "scikit-learn",
        "artifact_format": "sklearn_joblib",
        "learning_type": "supervised",
        "training_dataset": "training",
        "parameters": {
            "text_field": training_config.text_field,
            "label_field": training_config.label_field,
            "alpha": training_config.alpha,
            "min_token_length": training_config.min_token_length,
            "max_vocabulary_size": training_config.max_vocabulary_size,
            "vectorizer": "CountVectorizer",
            "classifier": "MultinomialNB",
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
        "vocabulary_preview": vocabulary[:100],
    }
    return TextClassifierTrainingResult(model=model, estimator=estimator)


def save_text_classifier_artifact(
    model: Mapping[str, Any],
    estimator: TextClassifierEstimator | None,
    manifest_path: str | Path,
    estimator_path: str | Path,
) -> None:
    if estimator is None:
        raise ValueError("Cannot save a classifier artifact without a trained estimator.")

    manifest_file = Path(manifest_path)
    estimator_file = Path(estimator_path)
    manifest_file.parent.mkdir(parents=True, exist_ok=True)
    estimator_file.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(estimator, estimator_file)
    manifest = {**dict(model), "estimator_path": str(estimator_file)}
    manifest_file.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def load_text_classifier_estimator(model: Mapping[str, Any]) -> TextClassifierEstimator:
    estimator_path = _string_value(model.get("estimator_path"))
    if estimator_path is None:
        raise ValueError("Classifier manifest does not contain estimator_path.")

    loaded = joblib.load(estimator_path)
    if not isinstance(loaded, Pipeline):
        raise ValueError(f"{estimator_path} does not contain a scikit-learn Pipeline.")
    return cast(TextClassifierEstimator, loaded)


def predict_text_classifier(
    model: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
    text_field: str | None = None,
    estimator: TextClassifierEstimator | None = None,
) -> list[str]:
    parameters = _mapping(model.get("parameters"))
    field = text_field or _string_value(parameters.get("text_field")) or "text"
    texts = [_string_value(record.get(field)) or "" for record in records]
    classifier = estimator or load_text_classifier_estimator(model)
    return [str(prediction) for prediction in classifier.predict(texts)]


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
) -> list[tuple[str, str]]:
    examples: list[tuple[str, str]] = []
    for record in records:
        label = _string_value(record.get(config.label_field))
        text = _string_value(record.get(config.text_field))
        if label is None or text is None:
            continue
        examples.append((text, label))
    return examples


def _empty_model(
    records: Sequence[Mapping[str, Any]],
    config: TextClassifierConfig,
    warning: str,
) -> Record:
    return {
        "model_type": "text_classification",
        "algorithm": "multinomial_naive_bayes",
        "framework": "scikit-learn",
        "artifact_format": "sklearn_joblib",
        "learning_type": "supervised",
        "training_dataset": "training",
        "parameters": {
            "text_field": config.text_field,
            "label_field": config.label_field,
            "alpha": config.alpha,
            "min_token_length": config.min_token_length,
            "max_vocabulary_size": config.max_vocabulary_size,
            "vectorizer": "CountVectorizer",
            "classifier": "MultinomialNB",
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
        "vocabulary_preview": [],
    }


def _vocabulary(estimator: TextClassifierEstimator) -> list[str]:
    vectorizer = estimator.named_steps.get("vectorizer")
    if not isinstance(vectorizer, CountVectorizer):
        raise ValueError("Classifier pipeline does not contain CountVectorizer.")
    vocabulary = vectorizer.vocabulary_
    return sorted(vocabulary, key=vocabulary.get)


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _string_value(value: Any) -> str | None:
    if value is None or isinstance(value, bool):
        return None
    text = str(value).strip()
    return text or None

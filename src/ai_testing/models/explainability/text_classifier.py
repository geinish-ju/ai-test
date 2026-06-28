from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

from ai_testing.models.training.text_classifier import load_text_classifier_estimator

Record = dict[str, Any]


@dataclass(frozen=True)
class TextClassifierExplainabilityConfig:
    top_features_per_class: int = 20


def explain_text_classifier(
    model: dict[str, Any],
    config: TextClassifierExplainabilityConfig | None = None,
) -> Record:
    explainability_config = config or TextClassifierExplainabilityConfig()
    if explainability_config.top_features_per_class < 1:
        raise ValueError("top_features_per_class must be at least 1")

    estimator = load_text_classifier_estimator(model)
    if not isinstance(estimator, Pipeline):
        raise ValueError("Classifier estimator must be a scikit-learn Pipeline.")

    vectorizer = estimator.named_steps.get("vectorizer")
    classifier = estimator.named_steps.get("classifier")
    if not isinstance(vectorizer, CountVectorizer):
        raise ValueError("Classifier pipeline does not contain CountVectorizer.")
    if not isinstance(classifier, MultinomialNB):
        raise ValueError("Classifier pipeline does not contain MultinomialNB.")

    feature_names = [str(feature) for feature in vectorizer.get_feature_names_out()]
    log_probabilities: Any = classifier.feature_log_prob_
    classes = [str(label) for label in classifier.classes_]
    class_explanations = {
        class_label: {
            "top_tokens": _top_tokens(
                feature_names=feature_names,
                log_probabilities=log_probabilities[class_index],
                limit=explainability_config.top_features_per_class,
            )
        }
        for class_index, class_label in enumerate(classes)
    }

    return {
        "step": "Model explainability",
        "report_type": "model_explainability_report",
        "subject": "category_classifier",
        "model_type": model.get("model_type"),
        "algorithm": model.get("algorithm"),
        "framework": model.get("framework"),
        "method": "top token log probabilities per class",
        "limitations": [
            (
                "Token weights explain the trained bag-of-words classifier, "
                "not causal product behavior."
            ),
            "Correlated tokens and rare classes can make individual token rankings unstable.",
        ],
        "parameters": {
            "top_features_per_class": explainability_config.top_features_per_class,
        },
        "summary": {
            "class_count": len(classes),
            "feature_count": len(feature_names),
        },
        "classes": class_explanations,
    }


def _top_tokens(
    *,
    feature_names: list[str],
    log_probabilities: Any,
    limit: int,
) -> list[Record]:
    indexes = sorted(
        range(len(feature_names)),
        key=lambda index: float(log_probabilities[index]),
        reverse=True,
    )[:limit]
    return [
        {
            "token": feature_names[index],
            "log_probability": round(float(log_probabilities[index]), 6),
        }
        for index in indexes
    ]

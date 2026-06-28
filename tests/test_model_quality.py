from __future__ import annotations

from collections.abc import Callable
from typing import Any

from ai_testing.ml_model_testing import (
    AssociationMLModelTestConfig,
    TextClassifierMLModelTestConfig,
)
from ai_testing.ml_model_testing import (
    test_association_ml_model as run_association_ml_model_test,
)
from ai_testing.ml_model_testing import (
    test_text_classifier_ml_model as run_text_classifier_ml_model_test,
)

Record = dict[str, Any]


def test_text_classifier_quality_rejects_target_leakage_feature(
    classification_validation_report: Record,
    classification_test_report: Record,
    failed_check_ids: Callable[[Record], set[str]],
) -> None:
    report = run_text_classifier_ml_model_test(
        model={
            "model_type": "text_classification",
            "algorithm": "multinomial_naive_bayes",
            "framework": "scikit-learn",
            "artifact_format": "sklearn_joblib",
            "learning_type": "supervised",
            "estimator_path": "data/models/category_classifier.joblib",
            "training_input": "data/classification/category/train.json",
            "summary": {
                "training_example_count": 120,
                "class_count": 4,
                "vocabulary_size": 500,
            },
            "parameters": {"text_field": "main_category", "label_field": "label"},
        },
        validation_report=classification_validation_report,
        test_report=classification_test_report,
        classification_manifest={
            "step": "Supervised classification dataset preprocessing",
            "global": {
                "learning_type": "supervised",
                "text_fields": ["product_name", "brand"],
                "target_field": "label",
            },
        },
        config=TextClassifierMLModelTestConfig(),
    ).report

    failed_ids = failed_check_ids(report)

    assert report["status"] == "failed"
    assert "ml_model.text_feature_field_is_safe" in failed_ids
    assert "ml_model.test_accuracy" not in failed_ids
    assert "ml_model.validation_test_accuracy_delta" not in failed_ids


def test_association_quality_rejects_training_on_holdout_test_dataset(
    failed_check_ids: Callable[[Record], set[str]],
) -> None:
    report = run_association_ml_model_test(
        model={
            "model_type": "association_rules",
            "algorithm": "apriori",
            "learning_type": "unsupervised",
            "training_input": "data/splits/test.json",
            "summary": {"rule_count": 1, "exported_rule_count": 1},
            "parameters": {"item_field": "product_name"},
            "rules": [{"antecedent": ["bread"], "consequent": ["milk"]}],
        },
        validation_report={
            "validation_type": "k-fold cross-validation",
            "summary": {
                "fold_count": 5,
                "mean_validation_confidence": 0.5,
                "mean_validation_lift": 1.3,
                "mean_abs_confidence_gap": 0.05,
            },
        },
        test_report={
            "testing_type": "hold-out test",
            "test": {
                "basket_count": 20,
                "evaluated_rule_count": 1,
                "stable_rule_count": 1,
                "mean_test_confidence": 0.48,
                "mean_test_lift": 1.25,
                "antecedent_coverage": 0.8,
                "hit_rate_per_covered_basket": 0.6,
                "mean_abs_confidence_gap": 0.04,
            },
        },
        config=AssociationMLModelTestConfig(),
    ).report

    assert report["status"] == "failed"
    assert "ml_model.training_input_is_not_test" in failed_check_ids(report)

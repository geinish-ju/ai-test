from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from ai_testing.model_validation import (
    AssociationEvaluationConfig,
    evaluate_association_rules_on_records,
)

Record = dict[str, Any]


@dataclass(frozen=True)
class AssociationTestConfig:
    basket_field: str = "basket_id"
    item_field: str = "product_group"
    min_confidence: float = 0.2
    min_lift: float = 1.05
    top_rules: int = 20


@dataclass(frozen=True)
class AssociationTestResult:
    report: Record


def test_association_rules(
    model: Mapping[str, Any],
    test_records: Sequence[Mapping[str, Any]],
    config: AssociationTestConfig | None = None,
) -> AssociationTestResult:
    test_config = config or AssociationTestConfig()
    evaluation_result = evaluate_association_rules_on_records(
        model=model,
        records=test_records,
        config=AssociationEvaluationConfig(
            basket_field=test_config.basket_field,
            item_field=test_config.item_field,
            min_confidence=test_config.min_confidence,
            min_lift=test_config.min_lift,
            top_rules=test_config.top_rules,
        ),
        dataset_name="hold-out test",
        metric_prefix="test",
    )
    evaluation_report = evaluation_result.report
    metrics = evaluation_report.get("metrics")
    test_summary: Record = {"record_count": evaluation_report.get("record_count", 0)}
    if isinstance(metrics, Mapping):
        test_summary.update(metrics)

    return AssociationTestResult(
        report={
            "step": "Association rules final test",
            "testing_type": "hold-out test",
            "model_type": evaluation_report.get("model_type"),
            "algorithm": evaluation_report.get("algorithm"),
            "learning_type": evaluation_report.get("learning_type"),
            "model_training_input": evaluation_report.get("training_input"),
            "dataset": evaluation_report.get("dataset"),
            "parameters": evaluation_report.get("parameters", {}),
            "test": test_summary,
            "top_rules": evaluation_report.get("top_rules", []),
            "note": (
                "Final test on the hold-out dataset only. Do not tune parameters on this report."
            ),
        }
    )

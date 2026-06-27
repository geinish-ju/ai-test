from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from ai_testing.model_validation import (
    TextClassificationEvaluationConfig,
    evaluate_text_classifier,
)

Record = dict[str, Any]


@dataclass(frozen=True)
class TextClassifierTestConfig:
    text_field: str = "text"
    label_field: str = "label"
    top_confusions: int = 20


@dataclass(frozen=True)
class TextClassifierTestResult:
    report: Record


def test_text_classifier(
    model: Mapping[str, Any],
    test_records: Sequence[Mapping[str, Any]],
    config: TextClassifierTestConfig | None = None,
) -> TextClassifierTestResult:
    test_config = config or TextClassifierTestConfig()
    evaluation_result = evaluate_text_classifier(
        model=model,
        records=test_records,
        config=TextClassificationEvaluationConfig(
            text_field=test_config.text_field,
            label_field=test_config.label_field,
            top_confusions=test_config.top_confusions,
        ),
        dataset_name="hold-out test",
    )
    evaluation_report = evaluation_result.report
    metrics = evaluation_report.get("metrics")
    test_summary: Record = {"record_count": evaluation_report.get("record_count", 0)}
    if isinstance(metrics, Mapping):
        test_summary.update(metrics)

    return TextClassifierTestResult(
        report={
            "step": "Text classification final test",
            "testing_type": "hold-out test",
            "model_type": evaluation_report.get("model_type"),
            "algorithm": evaluation_report.get("algorithm"),
            "learning_type": evaluation_report.get("learning_type"),
            "model_training_input": evaluation_report.get("training_input"),
            "dataset": evaluation_report.get("dataset"),
            "test": test_summary,
            "note": (
                "Final supervised classification test on the hold-out dataset only. "
                "Do not tune parameters on this report."
            ),
        }
    )

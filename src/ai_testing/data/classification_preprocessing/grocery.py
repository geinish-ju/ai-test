from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

Record = dict[str, Any]


@dataclass(frozen=True)
class ClassificationPreprocessingConfig:
    target_field: str = "main_category"
    text_fields: tuple[str, ...] = ("product_name", "brand")
    metadata_fields: tuple[str, ...] = ("basket_id", "shop", "order_date")
    min_label_count: int = 20


@dataclass(frozen=True)
class ClassificationPreprocessingResult:
    records: list[Record]
    report: Record
    allowed_labels: tuple[str, ...]


def build_classification_records(
    records: Sequence[Mapping[str, Any]],
    config: ClassificationPreprocessingConfig | None = None,
    allowed_labels: Sequence[str] | None = None,
) -> ClassificationPreprocessingResult:
    preprocessing_config = config or ClassificationPreprocessingConfig()
    if preprocessing_config.min_label_count < 1:
        raise ValueError("min_label_count must be at least 1")

    candidates: list[Record] = []
    for record in records:
        candidate = _candidate_record(record, preprocessing_config)
        if candidate is not None:
            candidates.append(candidate)
    label_counts = Counter(record["label"] for record in candidates)
    selected_labels = (
        tuple(sorted(set(allowed_labels)))
        if allowed_labels is not None
        else tuple(
            label
            for label, count in sorted(label_counts.items())
            if count >= preprocessing_config.min_label_count
        )
    )
    selected_label_set = set(selected_labels)
    output_records = [
        record for record in candidates if _string_value(record.get("label")) in selected_label_set
    ]
    report = _report(
        input_records=records,
        candidates=candidates,
        output_records=output_records,
        label_counts=label_counts,
        allowed_labels=selected_labels,
        config=preprocessing_config,
    )
    return ClassificationPreprocessingResult(
        records=output_records,
        report=report,
        allowed_labels=selected_labels,
    )


def _candidate_record(
    record: Mapping[str, Any],
    config: ClassificationPreprocessingConfig,
) -> Record | None:
    label = _string_value(record.get(config.target_field))
    text_parts = [
        text
        for field in config.text_fields
        for text in [_string_value(record.get(field))]
        if text is not None
    ]
    if label is None or not text_parts:
        return None

    classification_record: Record = {
        "text": " ".join(text_parts),
        "label": label,
    }
    for field in config.metadata_fields:
        classification_record[field] = record.get(field)
    return classification_record


def _report(
    input_records: Sequence[Mapping[str, Any]],
    candidates: Sequence[Mapping[str, Any]],
    output_records: Sequence[Mapping[str, Any]],
    label_counts: Counter[Any],
    allowed_labels: Sequence[str],
    config: ClassificationPreprocessingConfig,
) -> Record:
    output_label_counts = Counter(record["label"] for record in output_records)
    return {
        "step": "Classification dataset preprocessing",
        "learning_type": "supervised",
        "task": "product category text classification",
        "target_field": config.target_field,
        "text_fields": list(config.text_fields),
        "metadata_fields": list(config.metadata_fields),
        "min_label_count": config.min_label_count,
        "input_record_count": len(input_records),
        "candidate_record_count": len(candidates),
        "output_record_count": len(output_records),
        "dropped_record_count": len(input_records) - len(output_records),
        "missing_label_count": sum(
            1 for record in input_records if _string_value(record.get(config.target_field)) is None
        ),
        "empty_text_count": sum(1 for record in input_records if not _text_present(record, config)),
        "input_label_count": len(label_counts),
        "output_label_count": len(output_label_counts),
        "allowed_labels": list(allowed_labels),
        "input_label_distribution": _counter_report(label_counts),
        "output_label_distribution": _counter_report(output_label_counts),
    }


def _text_present(record: Mapping[str, Any], config: ClassificationPreprocessingConfig) -> bool:
    return any(_string_value(record.get(field)) is not None for field in config.text_fields)


def _counter_report(counter: Counter[Any]) -> Record:
    total = sum(counter.values())
    return {
        str(label): {
            "count": count,
            "rate": _rate(count, total),
        }
        for label, count in sorted(counter.items(), key=lambda item: (-item[1], str(item[0])))
    }


def _string_value(value: Any) -> str | None:
    if value is None or isinstance(value, bool):
        return None
    text = str(value).strip()
    return text or None


def _rate(count: int, total: int) -> float:
    if total == 0:
        return 0.0
    return round(count / total, 6)

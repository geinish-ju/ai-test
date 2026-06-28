from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from random import Random
from typing import Any

Record = dict[str, Any]

MISSING_LABEL = "__missing__"


@dataclass(frozen=True)
class DatasetSplitConfig:
    group_field: str = "basket_id"
    stratify_field: str = "shop"
    test_size: float = 0.2
    n_splits: int = 5
    random_seed: int = 42


@dataclass(frozen=True)
class DatasetFold:
    fold_index: int
    train_records: list[Record]
    validation_records: list[Record]
    train_group_count: int
    validation_group_count: int


@dataclass(frozen=True)
class DatasetSplitResult:
    train_validation_records: list[Record]
    test_records: list[Record]
    folds: list[DatasetFold]
    manifest: Record


def split_dataset_records(
    records: Sequence[Mapping[str, Any]],
    config: DatasetSplitConfig | None = None,
) -> DatasetSplitResult:
    split_config = config or DatasetSplitConfig()
    _validate_config(split_config)

    normalized_records = [dict(record) for record in records]
    grouped_records, missing_group_count = _group_records(normalized_records, split_config)
    if len(grouped_records) < split_config.n_splits + 1:
        raise ValueError(
            "Not enough groups for hold-out test and k-fold validation: "
            f"{len(grouped_records)} groups for {split_config.n_splits} folds."
        )

    group_labels = {
        group_id: _group_stratify_label(group_records, split_config.stratify_field)
        for group_id, group_records in grouped_records.items()
    }
    test_groups, train_validation_groups = _train_test_groups(
        group_labels=group_labels,
        test_size=split_config.test_size,
        seed=split_config.random_seed,
    )
    fold_groups = _fold_groups(
        group_labels={group_id: group_labels[group_id] for group_id in train_validation_groups},
        n_splits=split_config.n_splits,
        seed=split_config.random_seed,
    )

    test_records = _records_for_groups(normalized_records, test_groups, split_config.group_field)
    train_validation_records = _records_for_groups(
        normalized_records,
        train_validation_groups,
        split_config.group_field,
    )
    folds = [
        _build_fold(
            fold_index=index + 1,
            validation_groups=validation_groups,
            all_train_validation_groups=train_validation_groups,
            records=normalized_records,
            group_field=split_config.group_field,
        )
        for index, validation_groups in enumerate(fold_groups)
    ]
    manifest = _manifest(
        records=normalized_records,
        config=split_config,
        missing_group_count=missing_group_count,
        grouped_records=grouped_records,
        group_labels=group_labels,
        train_validation_groups=train_validation_groups,
        test_groups=test_groups,
        folds=folds,
    )
    return DatasetSplitResult(
        train_validation_records=train_validation_records,
        test_records=test_records,
        folds=folds,
        manifest=manifest,
    )


def _validate_config(config: DatasetSplitConfig) -> None:
    if not config.group_field:
        raise ValueError("group_field must not be empty")
    if not config.stratify_field:
        raise ValueError("stratify_field must not be empty")
    if not 0 < config.test_size < 1:
        raise ValueError("test_size must be in the (0, 1) range")
    if config.n_splits < 2:
        raise ValueError("n_splits must be at least 2")


def _group_records(
    records: Sequence[Record],
    config: DatasetSplitConfig,
) -> tuple[dict[str, list[Record]], int]:
    grouped: dict[str, list[Record]] = defaultdict(list)
    missing_group_count = 0
    for index, record in enumerate(records):
        group_id = _string_value(record.get(config.group_field))
        if group_id is None:
            group_id = f"{MISSING_LABEL}_{index:06d}"
            missing_group_count += 1
        grouped[group_id].append(record)
    return dict(grouped), missing_group_count


def _group_stratify_label(records: Sequence[Record], stratify_field: str) -> str:
    labels = [_string_value(record.get(stratify_field)) for record in records]
    counts = Counter(label for label in labels if label is not None)
    if not counts:
        return MISSING_LABEL
    return sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0][0]


def _train_test_groups(
    group_labels: Mapping[str, str],
    test_size: float,
    seed: int,
) -> tuple[set[str], set[str]]:
    rng = Random(seed)
    groups_by_label = _groups_by_label(group_labels)
    test_groups: set[str] = set()
    train_validation_groups: set[str] = set()

    for label in sorted(groups_by_label):
        groups = sorted(groups_by_label[label])
        rng.shuffle(groups)
        test_count = _test_group_count(len(groups), test_size)
        test_groups.update(groups[:test_count])
        train_validation_groups.update(groups[test_count:])

    return test_groups, train_validation_groups


def _test_group_count(group_count: int, test_size: float) -> int:
    if group_count <= 1:
        return 0
    test_count = round(group_count * test_size)
    if test_count <= 0:
        return 1
    if test_count >= group_count:
        return group_count - 1
    return test_count


def _fold_groups(
    group_labels: Mapping[str, str],
    n_splits: int,
    seed: int,
) -> list[set[str]]:
    if len(group_labels) < n_splits:
        raise ValueError(f"Not enough train-validation groups for {n_splits} folds.")

    rng = Random(seed)
    groups_by_label = _groups_by_label(group_labels)
    folds: list[set[str]] = [set() for _ in range(n_splits)]
    for label in sorted(groups_by_label):
        groups = sorted(groups_by_label[label])
        rng.shuffle(groups)
        for index, group_id in enumerate(groups):
            folds[index % n_splits].add(group_id)

    if any(not fold for fold in folds):
        raise ValueError("At least one validation fold is empty.")
    return folds


def _groups_by_label(group_labels: Mapping[str, str]) -> dict[str, list[str]]:
    groups_by_label: dict[str, list[str]] = defaultdict(list)
    for group_id, label in group_labels.items():
        groups_by_label[label].append(group_id)
    return dict(groups_by_label)


def _build_fold(
    fold_index: int,
    validation_groups: set[str],
    all_train_validation_groups: set[str],
    records: Sequence[Record],
    group_field: str,
) -> DatasetFold:
    train_groups = all_train_validation_groups - validation_groups
    train_records = _records_for_groups(records, train_groups, group_field)
    validation_records = _records_for_groups(records, validation_groups, group_field)
    return DatasetFold(
        fold_index=fold_index,
        train_records=train_records,
        validation_records=validation_records,
        train_group_count=len(train_groups),
        validation_group_count=len(validation_groups),
    )


def _records_for_groups(
    records: Sequence[Record],
    group_ids: set[str],
    group_field: str,
) -> list[Record]:
    return [
        dict(record)
        for index, record in enumerate(records)
        if _record_group_id(record, group_field, index) in group_ids
    ]


def _record_group_id(record: Mapping[str, Any], group_field: str, index: int) -> str:
    group_id = _string_value(record.get(group_field))
    return group_id if group_id is not None else f"{MISSING_LABEL}_{index:06d}"


def _manifest(
    records: Sequence[Record],
    config: DatasetSplitConfig,
    missing_group_count: int,
    grouped_records: Mapping[str, Sequence[Record]],
    group_labels: Mapping[str, str],
    train_validation_groups: set[str],
    test_groups: set[str],
    folds: Sequence[DatasetFold],
) -> Record:
    return {
        "step": "Training, validation, and test dataset splitting",
        "strategy": "group-aware hold-out test split plus stratified k-fold cross-validation",
        "input_record_count": len(records),
        "group_count": len(grouped_records),
        "missing_group_count": missing_group_count,
        "parameters": {
            "group_field": config.group_field,
            "stratify_field": config.stratify_field,
            "test_size": config.test_size,
            "n_splits": config.n_splits,
            "random_seed": config.random_seed,
        },
        "splits": {
            "train_validation": _split_summary(
                records=records,
                group_ids=train_validation_groups,
                group_field=config.group_field,
                group_labels=group_labels,
            ),
            "test": _split_summary(
                records=records,
                group_ids=test_groups,
                group_field=config.group_field,
                group_labels=group_labels,
            ),
        },
        "folds": [
            {
                "fold_index": fold.fold_index,
                "train": _record_summary(fold.train_records, config.group_field),
                "validation": _record_summary(fold.validation_records, config.group_field),
            }
            for fold in folds
        ],
    }


def _split_summary(
    records: Sequence[Record],
    group_ids: set[str],
    group_field: str,
    group_labels: Mapping[str, str],
) -> Record:
    split_records = _records_for_groups(records, group_ids, group_field)
    label_counts = Counter(group_labels[group_id] for group_id in group_ids)
    return {
        **_record_summary(split_records, group_field),
        "stratify_group_distribution": dict(sorted(label_counts.items())),
    }


def _record_summary(records: Sequence[Record], group_field: str) -> Record:
    groups = {_record_group_id(record, group_field, index) for index, record in enumerate(records)}
    return {
        "record_count": len(records),
        "group_count": len(groups),
    }


def _string_value(value: Any) -> str | None:
    if value is None or isinstance(value, bool):
        return None
    text = str(value).strip()
    return text or None

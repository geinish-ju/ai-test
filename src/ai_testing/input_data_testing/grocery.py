from __future__ import annotations

import importlib
import json
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from typing import Any

from ai_testing.core import add_check as _add_check
from ai_testing.core import build_standard_report

Record = dict[str, Any]


@dataclass(frozen=True)
class InputDataFold:
    fold_index: int
    train_records: list[Record]
    validation_records: list[Record]


@dataclass(frozen=True)
class InputDataTestConfig:
    required_fields: tuple[str, ...]
    protected_fields: tuple[str, ...]
    critical_fields: tuple[str, ...]
    coverage_fields: tuple[str, ...]
    group_field: str = "basket_id"
    stratify_field: str = "shop"
    date_field: str = "order_date"
    currency_field: str = "currency"
    expected_shops: tuple[str, ...] = ("kosik", "rohlik")
    expected_currencies: tuple[str, ...] = ("CZK",)
    positive_numeric_fields: tuple[str, ...] = ("quantity_ordered",)
    non_negative_numeric_fields: tuple[str, ...] = ("price_unit", "price_per_unit")
    min_record_count: int = 1
    min_group_count: int = 1
    max_critical_missing_rate: float = 0.0
    max_coverage_missing_rate: float = 0.4
    max_duplicate_rate: float = 0.0
    max_split_distribution_delta: float = 0.2


@dataclass(frozen=True)
class InputDataTestResult:
    report: Record


def test_input_data(
    processed_records: Sequence[Mapping[str, Any]],
    train_validation_records: Sequence[Mapping[str, Any]],
    test_records: Sequence[Mapping[str, Any]],
    folds: Sequence[InputDataFold],
    config: InputDataTestConfig,
) -> InputDataTestResult:
    normalized_processed = [dict(record) for record in processed_records]
    normalized_train_validation = [dict(record) for record in train_validation_records]
    normalized_test = [dict(record) for record in test_records]
    checks: list[Record] = []

    processed_groups = _group_ids(normalized_processed, config.group_field)
    train_validation_groups = _group_ids(normalized_train_validation, config.group_field)
    test_groups = _group_ids(normalized_test, config.group_field)
    duplicate_count = _exact_duplicate_count(normalized_processed)
    duplicate_rate = _rate(duplicate_count, len(normalized_processed))
    critical_missing_rates = _missing_rates(normalized_processed, config.critical_fields)
    coverage_missing_rates = _missing_rates(normalized_processed, config.coverage_fields)
    invalid_numeric_counts = _invalid_numeric_counts(normalized_processed, config)
    invalid_numeric_diagnostics = _invalid_numeric_diagnostics(normalized_processed, config)

    _add_check(
        checks,
        check_id="input_data.record_count",
        passed=len(normalized_processed) >= config.min_record_count,
        severity="critical",
        message="Processed dataset contains enough records.",
        observed=len(normalized_processed),
        expected={">=": config.min_record_count},
    )
    _add_check(
        checks,
        check_id="input_data.group_count",
        passed=len(processed_groups) >= config.min_group_count,
        severity="critical",
        message="Processed dataset contains enough baskets.",
        observed=len(processed_groups),
        expected={">=": config.min_group_count},
    )
    _add_check(
        checks,
        check_id="input_data.required_fields_present",
        passed=not _missing_required_field_counts(normalized_processed, config.required_fields),
        severity="critical",
        message="Every processed record contains the required feature columns.",
        observed=_missing_required_field_counts(normalized_processed, config.required_fields),
        expected={"missing_field_count": 0},
    )
    _add_check(
        checks,
        check_id="input_data.protected_fields_absent",
        passed=not _present_field_counts(
            [
                *normalized_processed,
                *normalized_train_validation,
                *normalized_test,
            ],
            config.protected_fields,
        ),
        severity="critical",
        message="Technical identifiers and removed raw fields are absent from model datasets.",
        observed=_present_field_counts(
            [
                *normalized_processed,
                *normalized_train_validation,
                *normalized_test,
            ],
            config.protected_fields,
        ),
        expected={"present_count": 0},
    )
    _add_check(
        checks,
        check_id="input_data.critical_missing_rate",
        passed=all(
            rate <= config.max_critical_missing_rate for rate in critical_missing_rates.values()
        ),
        severity="critical",
        message="Critical modeling fields meet the missing-value threshold.",
        observed=critical_missing_rates,
        expected={"max_missing_rate": config.max_critical_missing_rate},
    )
    _add_check(
        checks,
        check_id="input_data.coverage_missing_rate",
        passed=all(
            rate <= config.max_coverage_missing_rate for rate in coverage_missing_rates.values()
        ),
        severity="major",
        message="Coverage fields meet the configured missing-value threshold.",
        observed=coverage_missing_rates,
        expected={"max_missing_rate": config.max_coverage_missing_rate},
    )
    _add_check(
        checks,
        check_id="input_data.duplicates",
        passed=duplicate_rate <= config.max_duplicate_rate,
        severity="major",
        message="Processed dataset duplicate rate is within threshold.",
        observed={"exact_duplicate_rows": duplicate_count, "duplicate_rate": duplicate_rate},
        expected={"max_duplicate_rate": config.max_duplicate_rate},
    )
    _add_check(
        checks,
        check_id="input_data.numeric_values",
        passed=not invalid_numeric_counts,
        severity="major",
        message="Configured numeric fields have valid positive or non-negative values.",
        observed=invalid_numeric_counts,
        expected={"invalid_count": 0},
        diagnostics=invalid_numeric_diagnostics,
    )
    pandera_contract_violations = _pandera_contract_violations(normalized_processed, config)
    _add_check(
        checks,
        check_id="input_data.pandera_contract",
        passed=not pandera_contract_violations,
        severity="major",
        message="Processed dataset satisfies the dataframe schema contract.",
        observed=pandera_contract_violations,
        expected={"failure_case_count": 0},
        diagnostics=_pandera_contract_diagnostics(
            records=normalized_processed,
            violations=pandera_contract_violations,
            config=config,
        ),
    )
    _add_check(
        checks,
        check_id="input_data.date_values",
        passed=not _invalid_date_counts(normalized_processed, config.date_field),
        severity="major",
        message="Order dates are parseable and are not in the future.",
        observed=_invalid_date_counts(normalized_processed, config.date_field),
        expected={"invalid_count": 0},
    )
    _add_check(
        checks,
        check_id="input_data.calendar_features",
        passed=not _calendar_mismatch_counts(normalized_processed, config.date_field),
        severity="major",
        message="Derived calendar features match the order date.",
        observed=_calendar_mismatch_counts(normalized_processed, config.date_field),
        expected={"mismatch_count": 0},
    )
    _add_check(
        checks,
        check_id="input_data.expected_categories",
        passed=not _unexpected_category_values(normalized_processed, config),
        severity="major",
        message="Shop and currency values match the expected domain.",
        observed=_unexpected_category_values(normalized_processed, config),
        expected={
            "shops": sorted(config.expected_shops),
            "currencies": sorted(config.expected_currencies),
        },
    )
    _add_check(
        checks,
        check_id="input_data.train_test_record_coverage",
        passed=_same_record_multiset(
            normalized_processed,
            [*normalized_train_validation, *normalized_test],
        ),
        severity="critical",
        message="Train-validation plus test records cover the processed dataset exactly.",
        observed=_record_multiset_delta(
            normalized_processed,
            [*normalized_train_validation, *normalized_test],
        ),
        expected={"missing_records": 0, "unexpected_records": 0},
    )
    _add_check(
        checks,
        check_id="input_data.train_test_group_leakage",
        passed=not (train_validation_groups & test_groups),
        severity="critical",
        message="No basket appears in both train-validation and test datasets.",
        observed={"overlap_group_count": len(train_validation_groups & test_groups)},
        expected={"overlap_group_count": 0},
    )
    _add_check(
        checks,
        check_id="input_data.split_distribution",
        passed=_max_distribution_delta(
            normalized_train_validation,
            normalized_test,
            config.stratify_field,
        )
        <= config.max_split_distribution_delta,
        severity="major",
        message="Train-validation and test stratify distributions are close enough.",
        observed={
            "max_delta": _max_distribution_delta(
                normalized_train_validation,
                normalized_test,
                config.stratify_field,
            ),
            "train_validation": _distribution(normalized_train_validation, config.stratify_field),
            "test": _distribution(normalized_test, config.stratify_field),
        },
        expected={"max_delta": config.max_split_distribution_delta},
    )
    _add_fold_checks(
        checks=checks,
        folds=folds,
        train_validation_groups=train_validation_groups,
        test_groups=test_groups,
        group_field=config.group_field,
    )

    return InputDataTestResult(
        report=build_standard_report(
            report_type="input_data_test",
            subject="grocery_order_items",
            step="Input data testing",
            testing_type="input data quality and split integrity",
            checks=checks,
            details={
                "datasets": {
                    "processed": _dataset_summary(normalized_processed, config.group_field),
                    "train_validation": _dataset_summary(
                        normalized_train_validation,
                        config.group_field,
                    ),
                    "test": _dataset_summary(normalized_test, config.group_field),
                    "fold_count": len(folds),
                },
                "missing_values": {
                    "critical_fields": critical_missing_rates,
                    "coverage_fields": coverage_missing_rates,
                },
            },
        )
    )


def _add_fold_checks(
    checks: list[Record],
    folds: Sequence[InputDataFold],
    train_validation_groups: set[str],
    test_groups: set[str],
    group_field: str,
) -> None:
    validation_group_counter: Counter[str] = Counter()
    fold_summaries: list[Record] = []
    fold_overlap_count = 0
    fold_test_overlap_count = 0
    empty_fold_count = 0

    for fold in folds:
        train_groups = _group_ids(fold.train_records, group_field)
        validation_groups = _group_ids(fold.validation_records, group_field)
        validation_group_counter.update(validation_groups)
        fold_overlap_count += len(train_groups & validation_groups)
        fold_test_overlap_count += len(validation_groups & test_groups)
        if not train_groups or not validation_groups:
            empty_fold_count += 1
        fold_summaries.append(
            {
                "fold_index": fold.fold_index,
                "train_group_count": len(train_groups),
                "validation_group_count": len(validation_groups),
                "train_validation_overlap_group_count": len(train_groups & validation_groups),
                "test_overlap_group_count": len(validation_groups & test_groups),
            }
        )

    expected_once = {group_id for group_id in train_validation_groups}
    observed_once = {group_id for group_id, count in validation_group_counter.items() if count == 1}
    repeated_groups = {
        group_id: count for group_id, count in validation_group_counter.items() if count > 1
    }
    missing_groups = sorted(expected_once - set(validation_group_counter))

    _add_check(
        checks,
        check_id="input_data.fold_files_present",
        passed=bool(folds),
        severity="critical",
        message="K-fold train/validation files are present.",
        observed={"fold_count": len(folds)},
        expected={"fold_count": "> 0"},
    )
    _add_check(
        checks,
        check_id="input_data.fold_group_leakage",
        passed=fold_overlap_count == 0 and fold_test_overlap_count == 0,
        severity="critical",
        message="Fold validation baskets do not leak into fold training or hold-out test data.",
        observed={
            "fold_train_validation_overlap_group_count": fold_overlap_count,
            "fold_test_overlap_group_count": fold_test_overlap_count,
            "folds": fold_summaries,
        },
        expected={
            "fold_train_validation_overlap_group_count": 0,
            "fold_test_overlap_group_count": 0,
        },
    )
    _add_check(
        checks,
        check_id="input_data.fold_validation_coverage",
        passed=observed_once == expected_once and not repeated_groups and not missing_groups,
        severity="critical",
        message="Every train-validation basket appears in validation exactly once across folds.",
        observed={
            "validation_group_count": len(validation_group_counter),
            "observed_once_group_count": len(observed_once),
            "missing_group_count": len(missing_groups),
            "repeated_group_count": len(repeated_groups),
        },
        expected={"validation_group_count": len(expected_once), "repeat_count": 0},
    )
    _add_check(
        checks,
        check_id="input_data.fold_non_empty",
        passed=empty_fold_count == 0,
        severity="critical",
        message="Every fold contains both training and validation baskets.",
        observed={"empty_fold_count": empty_fold_count},
        expected={"empty_fold_count": 0},
    )


def _missing_required_field_counts(
    records: Sequence[Mapping[str, Any]],
    fields: Sequence[str],
) -> Record:
    missing = {
        field: sum(1 for record in records if field not in record)
        for field in fields
        if any(field not in record for record in records)
    }
    return dict(sorted(missing.items()))


def _present_field_counts(records: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> Record:
    present = {
        field: sum(1 for record in records if field in record)
        for field in fields
        if any(field in record for record in records)
    }
    return dict(sorted(present.items()))


def _missing_rates(records: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> Record:
    return {
        field: _rate(sum(1 for record in records if _is_missing(record.get(field))), len(records))
        for field in fields
    }


def _invalid_numeric_counts(
    records: Sequence[Mapping[str, Any]],
    config: InputDataTestConfig,
) -> Record:
    invalid: dict[str, int] = {}
    for field in config.positive_numeric_fields:
        count = sum(
            1
            for record in records
            if not _is_missing(record.get(field)) and not _number_gt(record.get(field), 0.0)
        )
        if count:
            invalid[field] = count
    for field in config.non_negative_numeric_fields:
        count = sum(
            1
            for record in records
            if not _is_missing(record.get(field)) and not _number_gte(record.get(field), 0.0)
        )
        if count:
            invalid[field] = count
    return dict(sorted(invalid.items()))


def _invalid_numeric_diagnostics(
    records: Sequence[Mapping[str, Any]],
    config: InputDataTestConfig,
) -> Record:
    field_rules = {
        **{field: "> 0" for field in config.positive_numeric_fields},
        **{field: ">= 0" for field in config.non_negative_numeric_fields},
    }
    field_diagnostics = {
        field: _invalid_numeric_field_diagnostics(records, field, rule)
        for field, rule in sorted(field_rules.items())
    }
    field_diagnostics = {
        field: diagnostics
        for field, diagnostics in field_diagnostics.items()
        if diagnostics["affected_record_count"] > 0
    }
    if not field_diagnostics:
        return {}

    return {
        "summary": (
            "One or more numeric fields violate the modeling contract. "
            "Rows with non-positive quantity usually mean cancelled, unavailable, "
            "or incorrectly mapped order items."
        ),
        "fields": field_diagnostics,
        "suggested_actions": [
            "Check whether affected rows represent cancelled or undelivered items.",
            "If they are not real purchases, filter them out during preprocessing.",
            "If they are real purchases, fix the acquisition/preprocessing quantity mapping.",
        ],
    }


def _invalid_numeric_field_diagnostics(
    records: Sequence[Mapping[str, Any]],
    field: str,
    rule: str,
) -> Record:
    invalid_records = [
        (index, record)
        for index, record in enumerate(records)
        if _invalid_numeric_value(record.get(field), rule)
    ]
    values = [_float_value(record.get(field)) for _, record in invalid_records]
    numeric_values = [value for value in values if value is not None]
    return {
        "rule": rule,
        "affected_record_count": len(invalid_records),
        "value_summary": _numeric_failure_summary(numeric_values),
        "breakdown": {
            "by_shop": _top_value_counts(record.get("shop") for _, record in invalid_records),
            "by_main_category": _top_value_counts(
                record.get("main_category") for _, record in invalid_records
            ),
            "by_unit": _top_value_counts(record.get("unit") for _, record in invalid_records),
        },
        "sample_records": [
            _diagnostic_record(index=index, record=record, fields=(field,))
            for index, record in invalid_records[:10]
        ],
    }


def _invalid_numeric_value(value: Any, rule: str) -> bool:
    if _is_missing(value):
        return False
    if rule == "> 0":
        return not _number_gt(value, 0.0)
    if rule == ">= 0":
        return not _number_gte(value, 0.0)
    return False


def _numeric_failure_summary(values: Sequence[float]) -> Record:
    if not values:
        return {"min": None, "max": None, "zero_count": 0, "negative_count": 0}
    return {
        "min": min(values),
        "max": max(values),
        "zero_count": sum(1 for value in values if value == 0),
        "negative_count": sum(1 for value in values if value < 0),
    }


def _pandera_contract_violations(
    records: Sequence[Mapping[str, Any]],
    config: InputDataTestConfig,
) -> Record:
    if not records:
        return {}

    pd = importlib.import_module("pandas")
    pa = importlib.import_module("pandera.pandas")
    columns: dict[str, Any] = {
        field: pa.Column(nullable=field not in set(config.critical_fields), required=True)
        for field in config.required_fields
    }
    columns[config.stratify_field] = pa.Column(
        str,
        pa.Check.isin(config.expected_shops),
        nullable=False,
        required=True,
        coerce=True,
    )
    columns[config.currency_field] = pa.Column(
        str,
        pa.Check.isin(config.expected_currencies),
        nullable=False,
        required=True,
        coerce=True,
    )
    for field in config.positive_numeric_fields:
        columns[field] = pa.Column(
            float,
            pa.Check.gt(0),
            nullable=False,
            required=True,
            coerce=True,
        )
    for field in config.non_negative_numeric_fields:
        columns[field] = pa.Column(
            float,
            pa.Check.ge(0),
            nullable=True,
            required=False,
            coerce=True,
        )

    schema = pa.DataFrameSchema(columns, strict=False, coerce=True)
    try:
        schema.validate(pd.DataFrame(records), lazy=True)
    except pa.errors.SchemaErrors as error:
        failure_cases = error.failure_cases
        column_counts = failure_cases["column"].dropna().astype(str).value_counts().to_dict()
        return {
            "failure_case_count": len(failure_cases),
            "columns": {str(column): int(count) for column, count in column_counts.items()},
            "sample": failure_cases.head(10).to_dict("records"),
        }
    return {}


def _pandera_contract_diagnostics(
    records: Sequence[Mapping[str, Any]],
    violations: Mapping[str, Any],
    config: InputDataTestConfig,
) -> Record:
    if not violations:
        return {}

    failure_columns = _mapping(violations.get("columns"))
    sample_indexes = _failure_sample_indexes(violations.get("sample"))
    return {
        "summary": (
            "Pandera validated the processed dataset as a dataframe and found schema "
            "violations. This is the formal schema-level view of the same data contract."
        ),
        "failure_columns": failure_columns,
        "sample_records": [
            _diagnostic_record(index=index, record=records[index], fields=tuple(failure_columns))
            for index in sample_indexes
            if 0 <= index < len(records)
        ],
        "suggested_actions": [
            "Use the failed column list to decide whether preprocessing should fix or filter rows.",
            "Keep Pandera failures aligned with domain checks so machine and human reports agree.",
        ],
        "contract": {
            "positive_numeric_fields": list(config.positive_numeric_fields),
            "non_negative_numeric_fields": list(config.non_negative_numeric_fields),
            "expected_shops": list(config.expected_shops),
            "expected_currencies": list(config.expected_currencies),
        },
    }


def _failure_sample_indexes(sample: Any) -> list[int]:
    if not isinstance(sample, Sequence) or isinstance(sample, (str, bytes)):
        return []
    indexes: list[int] = []
    for item in sample:
        if not isinstance(item, Mapping):
            continue
        index = item.get("index")
        if isinstance(index, int) and not isinstance(index, bool):
            indexes.append(index)
    return indexes[:10]


def _diagnostic_record(
    index: int,
    record: Mapping[str, Any],
    fields: Sequence[str],
) -> Record:
    diagnostic: Record = {
        "row_index": index,
        "shop": record.get("shop"),
        "order_date": record.get("order_date"),
        "product_name": record.get("product_name"),
        "main_category": record.get("main_category"),
        "category": record.get("category"),
        "unit": record.get("unit"),
    }
    for field in fields:
        diagnostic[field] = record.get(field)
    for field in ("price_unit", "price_per_unit", "currency"):
        if field in record:
            diagnostic[field] = record.get(field)
    return diagnostic


def _top_value_counts(values: Iterable[Any], limit: int = 10) -> list[Record]:
    counts = Counter(_string_value(value) or "__missing__" for value in values)
    return [
        {"value": value, "count": count}
        for value, count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:limit]
    ]


def _invalid_date_counts(records: Sequence[Mapping[str, Any]], date_field: str) -> Record:
    today = date.today()
    invalid_parse_count = 0
    future_count = 0
    for record in records:
        parsed = _parse_date(record.get(date_field))
        if parsed is None:
            invalid_parse_count += 1
            continue
        if parsed > today:
            future_count += 1

    invalid: dict[str, int] = {}
    if invalid_parse_count:
        invalid["invalid_parse_count"] = invalid_parse_count
    if future_count:
        invalid["future_date_count"] = future_count
    return invalid


def _calendar_mismatch_counts(records: Sequence[Mapping[str, Any]], date_field: str) -> Record:
    mismatches: dict[str, int] = {}
    for field in (
        "order_month",
        "order_week_of_year",
        "order_day_of_week",
        "order_quarter",
    ):
        count = 0
        for record in records:
            parsed = _parse_date(record.get(date_field))
            if parsed is None or _is_missing(record.get(field)):
                continue
            if _int_value(record.get(field)) != _expected_calendar_value(field, parsed):
                count += 1
        if count:
            mismatches[field] = count

    weekend_count = 0
    for record in records:
        parsed = _parse_date(record.get(date_field))
        if parsed is None or _is_missing(record.get("order_is_weekend")):
            continue
        if bool(record.get("order_is_weekend")) != (parsed.weekday() >= 5):
            weekend_count += 1
    if weekend_count:
        mismatches["order_is_weekend"] = weekend_count
    return dict(sorted(mismatches.items()))


def _expected_calendar_value(field: str, parsed: date) -> int:
    if field == "order_month":
        return parsed.month
    if field == "order_week_of_year":
        return parsed.isocalendar().week
    if field == "order_day_of_week":
        return parsed.weekday()
    if field == "order_quarter":
        return ((parsed.month - 1) // 3) + 1
    return 0


def _unexpected_category_values(
    records: Sequence[Mapping[str, Any]],
    config: InputDataTestConfig,
) -> Record:
    observed_shops = _non_missing_values(records, config.stratify_field)
    observed_currencies = _non_missing_values(records, config.currency_field)
    unexpected: dict[str, list[str]] = {}
    unexpected_shops = sorted(observed_shops - set(config.expected_shops))
    unexpected_currencies = sorted(observed_currencies - set(config.expected_currencies))
    if unexpected_shops:
        unexpected["shops"] = unexpected_shops
    if unexpected_currencies:
        unexpected["currencies"] = unexpected_currencies
    return unexpected


def _same_record_multiset(
    left: Sequence[Mapping[str, Any]],
    right: Sequence[Mapping[str, Any]],
) -> bool:
    return _record_counter(left) == _record_counter(right)


def _record_multiset_delta(
    left: Sequence[Mapping[str, Any]],
    right: Sequence[Mapping[str, Any]],
) -> Record:
    left_counter = _record_counter(left)
    right_counter = _record_counter(right)
    missing_records = sum((left_counter - right_counter).values())
    unexpected_records = sum((right_counter - left_counter).values())
    return {
        "missing_records": missing_records,
        "unexpected_records": unexpected_records,
    }


def _record_counter(records: Sequence[Mapping[str, Any]]) -> Counter[str]:
    return Counter(_canonical_record(record) for record in records)


def _exact_duplicate_count(records: Sequence[Mapping[str, Any]]) -> int:
    signatures = _record_counter(records)
    return sum(count - 1 for count in signatures.values() if count > 1)


def _group_ids(records: Sequence[Mapping[str, Any]], group_field: str) -> set[str]:
    return {
        group_id
        for index, record in enumerate(records)
        for group_id in [_string_value(record.get(group_field)) or f"__missing__{index:06d}"]
    }


def _distribution(records: Sequence[Mapping[str, Any]], field: str) -> Record:
    values = [_string_value(record.get(field)) or "__missing__" for record in records]
    total = len(values)
    counts = Counter(values)
    return {
        value: {
            "count": count,
            "rate": _rate(count, total),
        }
        for value, count in sorted(counts.items())
    }


def _max_distribution_delta(
    left_records: Sequence[Mapping[str, Any]],
    right_records: Sequence[Mapping[str, Any]],
    field: str,
) -> float:
    left = _distribution_rates(left_records, field)
    right = _distribution_rates(right_records, field)
    keys = set(left) | set(right)
    if not keys:
        return 0.0
    return round(max(abs(left.get(key, 0.0) - right.get(key, 0.0)) for key in keys), 6)


def _distribution_rates(records: Sequence[Mapping[str, Any]], field: str) -> dict[str, float]:
    values = [_string_value(record.get(field)) or "__missing__" for record in records]
    total = len(values)
    counts = Counter(values)
    return {value: _rate(count, total) for value, count in counts.items()}


def _dataset_summary(records: Sequence[Mapping[str, Any]], group_field: str) -> Record:
    return {
        "record_count": len(records),
        "group_count": len(_group_ids(records, group_field)),
    }


def _non_missing_values(records: Sequence[Mapping[str, Any]], field: str) -> set[str]:
    return {
        value
        for record in records
        for value in [_string_value(record.get(field))]
        if value is not None
    }


def _parse_date(value: Any) -> date | None:
    text = _string_value(value)
    if text is None:
        return None
    try:
        return date.fromisoformat(text[:10])
    except ValueError:
        return None


def _number_gt(value: Any, threshold: float) -> bool:
    number = _float_value(value)
    return number is not None and number > threshold


def _number_gte(value: Any, threshold: float) -> bool:
    number = _float_value(value)
    return number is not None and number >= threshold


def _float_value(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.replace(",", ".").strip())
        except ValueError:
            return None
    return None


def _int_value(value: Any) -> int | None:
    number = _float_value(value)
    return int(number) if number is not None else None


def _string_value(value: Any) -> str | None:
    if value is None or isinstance(value, bool):
        return None
    text = str(value).strip()
    return text or None


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _is_missing(value: Any) -> bool:
    return value in (None, "", [], {})


def _canonical_record(record: Mapping[str, Any]) -> str:
    return json.dumps(record, ensure_ascii=False, sort_keys=True, default=str)


def _rate(count: int, total: int) -> float:
    if total == 0:
        return 0.0
    return round(count / total, 6)

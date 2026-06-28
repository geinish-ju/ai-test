from __future__ import annotations

import json
import unicodedata
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime
from statistics import mean
from typing import Any

Record = dict[str, Any]

DEFAULT_IDENTIFIER_FIELDS = (
    "order_id",
    "order_number",
    "order_item_id",
    "product_id",
    "product_slug",
    "product_url",
    "main_category_id",
    "category_id",
    "category_ids",
)

DEFAULT_EXACT_TIME_FIELDS = ("order_created_at",)

DEFAULT_OUTPUT_FIELDS = (
    "shop",
    "order_date",
    "product_name",
    "brand",
    "main_category",
    "category",
    "is_gluten_free",
    "product_group",
    "meat_type",
    "quality_level",
    "quantity_ordered",
    "unit",
    "package_quantity",
    "package_unit",
    "price_unit",
    "price_per_unit",
    "price_per_unit_unit",
    "currency",
    "basket_id",
    "order_month",
    "order_week_of_year",
    "order_day_of_week",
    "order_is_weekend",
    "order_quarter",
)

PRODUCT_GROUP_RULES = (
    ("ham", ("sunk", "ham")),
    ("salami", ("salam", "salami")),
    ("sausage", ("klobas", "parek", "parky", "uzenin", "sausage")),
    ("bacon", ("slanin", "bacon")),
    ("meat", ("maso", "masa", "meat")),
    ("fish", ("ryb", "losos", "tunak", "fish", "salmon", "tuna")),
    ("cheese", ("syr", "syry", "cheese")),
    ("yogurt", ("jogurt", "yogurt", "yoghurt")),
    ("milk", ("mleko", "milk")),
    ("egg", ("vejce", "eggs", "egg")),
    ("bread", ("chleb", "pecivo", "bread", "bakery")),
    ("fruit", ("ovoce", "fruit")),
    ("vegetable", ("zelenin", "vegetable")),
    ("pasta", ("testovin", "pasta")),
    ("rice", ("ryze", "rice")),
    ("coffee", ("kava", "coffee")),
    ("tea", ("caj", "tea")),
    ("water", ("voda", "water")),
)

MEAT_TYPE_RULES = (
    ("chicken", ("kurec", "kure", "chicken")),
    ("turkey", ("krut", "turkey")),
    ("beef", ("hovez", "beef")),
    ("pork", ("veprov", "prase", "pork")),
    ("fish", ("ryb", "losos", "tunak", "fish", "salmon", "tuna")),
)

QUALITY_LEVEL_RULES = (
    ("vyberova", ("vyberov",)),
    ("premium", ("premium",)),
    ("bio", ("bio", "organic")),
    ("farm", ("farm", "farmar")),
)

GLUTEN_FREE_RULES = ("bezlepk", "bez lepku", "gluten free", "gluten-free")

REQUIRED_FIELDS = (
    "shop",
    "basket_id",
    "order_date",
    "product_name",
    "main_category",
    "category",
    "quantity_ordered",
    "unit",
    "currency",
)

NUMERIC_FIELDS = (
    "quantity_ordered",
    "package_quantity",
    "price_unit",
    "price_per_unit",
    "order_month",
    "order_week_of_year",
    "order_day_of_week",
    "order_quarter",
)

CATEGORICAL_FIELDS = (
    "shop",
    "main_category",
    "category",
    "brand",
    "unit",
    "package_unit",
    "price_per_unit_unit",
    "currency",
    "product_group",
    "meat_type",
    "quality_level",
)

EXPECTED_SHOPS = {"kosik", "rohlik"}
EXPECTED_CURRENCIES = {"CZK"}


@dataclass(frozen=True)
class PreprocessingConfig:
    identifier_fields: tuple[str, ...] = DEFAULT_IDENTIFIER_FIELDS
    exact_time_fields: tuple[str, ...] = DEFAULT_EXACT_TIME_FIELDS
    output_fields: tuple[str, ...] = DEFAULT_OUTPUT_FIELDS
    drop_exact_duplicates: bool = False
    basket_id_prefix: str = "basket"


@dataclass(frozen=True)
class PreprocessingResult:
    records: list[Record]
    report: Record


def preprocess_grocery_records(
    raw_records: Sequence[Mapping[str, Any]],
    config: PreprocessingConfig | None = None,
) -> PreprocessingResult:
    preprocessing_config = config or PreprocessingConfig()
    normalized_records = [_normalize_record(dict(record)) for record in raw_records]
    deduplicated_records, incomplete_duplicate_report = _drop_incomplete_duplicate_order_items(
        normalized_records
    )
    cleaned_records, zero_quantity_price_report = _drop_zero_quantity_price_items(
        deduplicated_records
    )
    basket_ids = _build_basket_ids(cleaned_records, preprocessing_config.basket_id_prefix)
    processed_records = [
        _preprocess_record(record, basket_ids, preprocessing_config) for record in cleaned_records
    ]

    exact_duplicate_count = _exact_duplicate_count(processed_records)
    if preprocessing_config.drop_exact_duplicates:
        processed_records = _drop_exact_duplicates(processed_records)

    report = build_quality_report(
        raw_records=normalized_records,
        cleaned_raw_records=cleaned_records,
        processed_records=processed_records,
        removed_fields=_removed_fields(normalized_records, preprocessing_config.output_fields),
        incomplete_duplicate_report=incomplete_duplicate_report,
        zero_quantity_price_report=zero_quantity_price_report,
        exact_duplicate_count=exact_duplicate_count,
        dropped_exact_duplicate_count=(
            len(cleaned_records) - len(processed_records)
            if preprocessing_config.drop_exact_duplicates
            else 0
        ),
    )
    return PreprocessingResult(records=processed_records, report=report)


def build_quality_report(
    raw_records: Sequence[Record],
    cleaned_raw_records: Sequence[Record],
    processed_records: Sequence[Record],
    removed_fields: Sequence[str],
    incomplete_duplicate_report: Record,
    zero_quantity_price_report: Record,
    exact_duplicate_count: int,
    dropped_exact_duplicate_count: int,
) -> Record:
    return {
        "step": "Data preprocessing",
        "input_record_count": len(raw_records),
        "cleaned_input_record_count": len(cleaned_raw_records),
        "output_record_count": len(processed_records),
        "removed_fields": sorted(set(removed_fields)),
        "cleaning": {
            "incomplete_duplicate_order_items": incomplete_duplicate_report,
            "zero_quantity_price_items": zero_quantity_price_report,
        },
        "missing_values": _missing_values_report(processed_records),
        "duplicates": {
            "exact_duplicate_rows": exact_duplicate_count,
            "dropped_exact_duplicate_rows": dropped_exact_duplicate_count,
            "logical_duplicate_rows": _logical_duplicate_count(processed_records),
        },
        "consistency": _consistency_report(
            raw_records=raw_records,
            processed_records=processed_records,
        ),
        "category_feature_coverage": _category_feature_coverage(processed_records),
        "categorical_cardinality": _categorical_cardinality(processed_records),
        "numeric_summary": _numeric_summary(processed_records),
        "outliers": _outlier_report(processed_records),
    }


def _preprocess_record(
    record: Record,
    basket_ids: Mapping[str, str],
    config: PreprocessingConfig,
) -> Record:
    processed = dict(record)
    processed["basket_id"] = basket_ids[_order_key(record)]
    processed["quantity_ordered"] = _quantity_ordered_feature(record)
    processed.update(_order_date_features(record))
    processed.update(_category_features(record))

    for field in (*config.identifier_fields, *config.exact_time_fields):
        processed.pop(field, None)

    return _select_fields(processed, config.output_fields)


def _drop_incomplete_duplicate_order_items(
    records: Sequence[Record],
) -> tuple[list[Record], Record]:
    groups: dict[tuple[str, str], list[tuple[int, Record]]] = {}
    for index, record in enumerate(records):
        product_key = _product_key(record)
        if product_key is None:
            continue
        groups.setdefault((_order_key(record), product_key), []).append((index, record))

    indexes_to_drop: set[int] = set()
    affected_groups: list[Record] = []
    for (order_key, product_key), items in sorted(groups.items()):
        if len(items) < 2:
            continue

        complete_items = [
            (index, record) for index, record in items if _has_quantity_and_price(record)
        ]
        incomplete_items = [
            (index, record) for index, record in items if _missing_quantity_and_price(record)
        ]
        if not complete_items or not incomplete_items:
            continue

        indexes_to_drop.update(index for index, _ in incomplete_items)
        affected_groups.append(
            {
                "order_key": order_key,
                "product_key": product_key,
                "dropped_record_count": len(incomplete_items),
                "kept_record_count": len(complete_items),
                "dropped_samples": [
                    _cleaning_sample(index=index, record=record)
                    for index, record in incomplete_items[:3]
                ],
                "kept_samples": [
                    _cleaning_sample(index=index, record=record)
                    for index, record in complete_items[:3]
                ],
            }
        )

    filtered_records = [
        dict(record) for index, record in enumerate(records) if index not in indexes_to_drop
    ]
    return filtered_records, {
        "rule": (
            "Within one order, drop duplicate product rows that have no usable quantity and "
            "no usable price when another row for the same product has both."
        ),
        "dropped_record_count": len(indexes_to_drop),
        "affected_group_count": len(affected_groups),
        "sample_groups": affected_groups[:10],
    }


def _drop_zero_quantity_price_items(records: Sequence[Record]) -> tuple[list[Record], Record]:
    indexes_to_drop = {
        index for index, record in enumerate(records) if _zero_quantity_price_item(record)
    }
    filtered_records = [
        dict(record) for index, record in enumerate(records) if index not in indexes_to_drop
    ]
    dropped_records = [
        _cleaning_sample(index=index, record=record)
        for index, record in enumerate(records)
        if index in indexes_to_drop
    ]
    return filtered_records, {
        "rule": (
            "Drop order-item rows where quantity, ordered quantity, delivered quantity, "
            "total price, unit price, and price per unit are all zero."
        ),
        "dropped_record_count": len(indexes_to_drop),
        "checked_fields": [
            "quantity",
            "quantity_ordered",
            "quantity_delivered",
            "price_total",
            "price_unit",
            "price_per_unit",
        ],
        "sample_records": dropped_records[:10],
    }


def _zero_quantity_price_item(record: Mapping[str, Any]) -> bool:
    fields = (
        "quantity",
        "quantity_ordered",
        "quantity_delivered",
        "price_total",
        "price_unit",
        "price_per_unit",
    )
    return all(_number_equals(record.get(field), 0.0) for field in fields)


def _product_key(record: Mapping[str, Any]) -> str | None:
    for field in ("product_id", "product_slug", "product_name"):
        value = _string_value(record.get(field))
        if value is not None:
            return f"{field}:{value.lower()}"
    return None


def _has_quantity_and_price(record: Mapping[str, Any]) -> bool:
    return _quantity_positive(record) and _price_positive(record)


def _missing_quantity_and_price(record: Mapping[str, Any]) -> bool:
    return not _quantity_positive(record) and not _price_positive(record)


def _quantity_positive(record: Mapping[str, Any]) -> bool:
    return any(
        _number_gt(record.get(field), 0.0)
        for field in ("quantity_ordered", "quantity_delivered", "quantity")
    )


def _price_positive(record: Mapping[str, Any]) -> bool:
    return any(
        _number_gt(record.get(field), 0.0)
        for field in ("price_total", "price_unit", "price_per_unit")
    )


def _cleaning_sample(index: int, record: Mapping[str, Any]) -> Record:
    return {
        "row_index": index,
        "shop": record.get("shop"),
        "order_date": record.get("order_date"),
        "product_name": record.get("product_name"),
        "main_category": record.get("main_category"),
        "category": record.get("category"),
        "quantity_ordered": record.get("quantity_ordered"),
        "quantity_delivered": record.get("quantity_delivered"),
        "quantity": record.get("quantity"),
        "unit": record.get("unit"),
        "price_total": record.get("price_total"),
        "price_unit": record.get("price_unit"),
        "currency": record.get("currency"),
    }


def _quantity_ordered_feature(record: Mapping[str, Any]) -> float | None:
    for field in ("quantity_ordered", "quantity_delivered", "quantity"):
        quantity = _to_float(record.get(field))
        if quantity is not None:
            return quantity
    return None


def _select_fields(record: Mapping[str, Any], fields: Sequence[str]) -> Record:
    return {field: record.get(field) for field in fields}


def _removed_fields(records: Sequence[Record], output_fields: Sequence[str]) -> list[str]:
    raw_fields = {field for record in records for field in record}
    return sorted(raw_fields - set(output_fields))


def _category_features(record: Mapping[str, Any]) -> Record:
    feature_texts = _feature_texts(record)
    product_group = _first_matching_rule(feature_texts, PRODUCT_GROUP_RULES)
    meat_type = _first_matching_rule(feature_texts, MEAT_TYPE_RULES)
    if meat_type is None and product_group in {"ham", "salami", "sausage", "bacon"}:
        meat_type = "pork"

    return {
        "is_gluten_free": _contains_any(feature_texts, GLUTEN_FREE_RULES),
        "product_group": product_group,
        "meat_type": meat_type,
        "quality_level": _first_matching_rule(feature_texts, QUALITY_LEVEL_RULES),
    }


def _feature_texts(record: Mapping[str, Any]) -> tuple[str, ...]:
    values: list[str] = []
    for field in ("product_name", "main_category", "category", "category_path"):
        values.extend(_flatten_text_values(record.get(field)))
    normalized_values = [_normalize_text(value) for value in values]
    return tuple(value for value in normalized_values if value)


def _flatten_text_values(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, Sequence) and not isinstance(value, str):
        values: list[str] = []
        for item in value:
            values.extend(_flatten_text_values(item))
        return values
    return []


def _normalize_text(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value)
    ascii_text = normalized.encode("ascii", "ignore").decode("ascii")
    return " ".join(ascii_text.lower().split())


def _first_matching_rule(
    texts: Sequence[str],
    rules: Sequence[tuple[str, Sequence[str]]],
) -> str | None:
    for label, patterns in rules:
        if _contains_any(texts, patterns):
            return label
    return None


def _contains_any(texts: Sequence[str], patterns: Sequence[str]) -> bool:
    normalized_patterns = [_normalize_text(pattern) for pattern in patterns]
    return any(pattern in text for text in texts for pattern in normalized_patterns)


def _normalize_record(record: Record) -> Record:
    normalized = {key: _normalize_value(value) for key, value in record.items()}
    for field in NUMERIC_FIELDS:
        if field in normalized:
            normalized[field] = _to_float(normalized[field])
    return normalized


def _normalize_value(value: Any) -> Any:
    if isinstance(value, str):
        stripped = value.strip()
        return stripped or None
    if isinstance(value, list):
        return [_normalize_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _normalize_value(nested) for key, nested in value.items()}
    return value


def _build_basket_ids(records: Sequence[Record], prefix: str) -> dict[str, str]:
    order_keys = sorted(
        {_order_key(record) for record in records},
        key=lambda key: (_order_sort_date(key, records), key),
    )
    return {key: f"{prefix}_{index:06d}" for index, key in enumerate(order_keys, start=1)}


def _order_sort_date(order_key: str, records: Sequence[Record]) -> str:
    for record in records:
        if _order_key(record) == order_key:
            return str(record.get("order_date") or record.get("order_created_at") or "")
    return ""


def _order_key(record: Mapping[str, Any]) -> str:
    shop = str(record.get("shop") or "unknown")
    raw_order_id = record.get("order_id") or record.get("order_number")
    if raw_order_id not in (None, ""):
        return f"{shop}:{raw_order_id}"
    return ":".join(
        [
            shop,
            str(record.get("order_date") or ""),
            str(record.get("order_created_at") or ""),
            str(record.get("product_name") or ""),
        ]
    )


def _order_date_features(record: Mapping[str, Any]) -> Record:
    parsed_date = _parse_date(record.get("order_date") or record.get("order_created_at"))
    if parsed_date is None:
        return {
            "order_year": None,
            "order_month": None,
            "order_week_of_year": None,
            "order_day_of_week": None,
            "order_is_weekend": None,
            "order_quarter": None,
        }

    return {
        "order_year": parsed_date.year,
        "order_month": parsed_date.month,
        "order_week_of_year": parsed_date.isocalendar().week,
        "order_day_of_week": parsed_date.weekday(),
        "order_is_weekend": parsed_date.weekday() >= 5,
        "order_quarter": ((parsed_date.month - 1) // 3) + 1,
    }


def _parse_date(value: Any) -> date | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None

    date_part = text[:10]
    try:
        return date.fromisoformat(date_part)
    except ValueError:
        pass

    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00")).date()
    except ValueError:
        return None


def _missing_values_report(records: Sequence[Record]) -> Record:
    fields = sorted({field for record in records for field in record})
    total = len(records)
    return {
        field: {
            "missing_count": missing_count,
            "missing_rate": _rate(missing_count, total),
        }
        for field in fields
        for missing_count in [sum(1 for record in records if _is_missing(record.get(field)))]
    }


def _exact_duplicate_count(records: Sequence[Record]) -> int:
    signatures = Counter(_canonical_record(record) for record in records)
    return sum(count - 1 for count in signatures.values() if count > 1)


def _drop_exact_duplicates(records: Sequence[Record]) -> list[Record]:
    seen: set[str] = set()
    deduplicated: list[Record] = []
    for record in records:
        signature = _canonical_record(record)
        if signature in seen:
            continue
        seen.add(signature)
        deduplicated.append(record)
    return deduplicated


def _logical_duplicate_count(records: Sequence[Record]) -> int:
    signatures = Counter(
        _canonical_value(
            {
                "shop": record.get("shop"),
                "basket_id": record.get("basket_id"),
                "product_name": record.get("product_name"),
                "category": record.get("category"),
                "quantity_ordered": record.get("quantity_ordered"),
                "unit": record.get("unit"),
                "price_unit": record.get("price_unit"),
                "currency": record.get("currency"),
            }
        )
        for record in records
    )
    return sum(count - 1 for count in signatures.values() if count > 1)


def _consistency_report(
    raw_records: Sequence[Record],
    processed_records: Sequence[Record],
) -> Record:
    required_missing = {
        field: sum(1 for record in processed_records if _is_missing(record.get(field)))
        for field in REQUIRED_FIELDS
    }
    invalid_numeric = {
        "quantity_ordered_lte_zero": sum(
            1 for record in processed_records if _number_lte(record.get("quantity_ordered"), 0.0)
        ),
        "package_quantity_lte_zero": sum(
            1
            for record in processed_records
            if not _is_missing(record.get("package_quantity"))
            and _number_lte(record.get("package_quantity"), 0.0)
        ),
        "price_unit_lt_zero": sum(
            1 for record in processed_records if _number_lt(record.get("price_unit"), 0.0)
        ),
        "price_per_unit_lt_zero": sum(
            1 for record in processed_records if _number_lt(record.get("price_per_unit"), 0.0)
        ),
    }
    unknown_shop_count = sum(
        1 for record in processed_records if record.get("shop") not in EXPECTED_SHOPS
    )
    unexpected_currency_count = sum(
        1
        for record in processed_records
        if record.get("currency") not in EXPECTED_CURRENCIES
        and not _is_missing(record.get("currency"))
    )

    return {
        "required_missing": required_missing,
        "invalid_numeric": invalid_numeric,
        "unknown_shop_count": unknown_shop_count,
        "unexpected_currency_count": unexpected_currency_count,
        "raw_quantity_ordered_delivered_mismatch_count": sum(
            1 for record in raw_records if _quantity_ordered_delivered_mismatch(record)
        ),
    }


def _category_feature_coverage(records: Sequence[Record]) -> Record:
    return {
        "is_gluten_free_true_count": sum(
            1 for record in records if record.get("is_gluten_free") is True
        ),
        "product_group_missing_count": sum(
            1 for record in records if _is_missing(record.get("product_group"))
        ),
        "meat_type_missing_count": sum(
            1 for record in records if _is_missing(record.get("meat_type"))
        ),
        "quality_level_missing_count": sum(
            1 for record in records if _is_missing(record.get("quality_level"))
        ),
    }


def _categorical_cardinality(records: Sequence[Record]) -> Record:
    return {
        field: len({record.get(field) for record in records if not _is_missing(record.get(field))})
        for field in CATEGORICAL_FIELDS
    }


def _numeric_summary(records: Sequence[Record]) -> Record:
    return {
        field: _numeric_field_summary(_numbers(record.get(field) for record in records))
        for field in NUMERIC_FIELDS
    }


def _numeric_field_summary(values: Sequence[float]) -> Record:
    if not values:
        return {
            "count": 0,
            "min": None,
            "max": None,
            "mean": None,
        }
    return {
        "count": len(values),
        "min": min(values),
        "max": max(values),
        "mean": round(mean(values), 4),
    }


def _outlier_report(records: Sequence[Record]) -> Record:
    return {
        field: _iqr_outlier_summary(_numbers(record.get(field) for record in records))
        for field in ("quantity_ordered", "package_quantity", "price_unit", "price_per_unit")
    }


def _iqr_outlier_summary(values: Sequence[float]) -> Record:
    if len(values) < 4:
        return {"count": 0, "lower_bound": None, "upper_bound": None}

    sorted_values = sorted(values)
    q1 = _percentile(sorted_values, 0.25)
    q3 = _percentile(sorted_values, 0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    count = sum(1 for value in sorted_values if value < lower_bound or value > upper_bound)
    return {
        "count": count,
        "lower_bound": round(lower_bound, 4),
        "upper_bound": round(upper_bound, 4),
    }


def _percentile(sorted_values: Sequence[float], percentile: float) -> float:
    index = (len(sorted_values) - 1) * percentile
    lower_index = int(index)
    upper_index = min(lower_index + 1, len(sorted_values) - 1)
    fraction = index - lower_index
    return (
        sorted_values[lower_index]
        + (sorted_values[upper_index] - sorted_values[lower_index]) * fraction
    )


def _quantity_ordered_delivered_mismatch(record: Mapping[str, Any]) -> bool:
    ordered = _to_float(record.get("quantity_ordered"))
    delivered = _to_float(record.get("quantity_delivered"))
    if ordered is None or delivered is None:
        return False
    return abs(ordered - delivered) > 0.0001


def _numbers(values: Iterable[Any]) -> list[float]:
    result: list[float] = []
    for value in values:
        number = _to_float(value)
        if number is not None:
            result.append(number)
    return result


def _number_lt(value: Any, threshold: float) -> bool:
    number = _to_float(value)
    return number is not None and number < threshold


def _number_lte(value: Any, threshold: float) -> bool:
    number = _to_float(value)
    return number is not None and number <= threshold


def _number_gt(value: Any, threshold: float) -> bool:
    number = _to_float(value)
    return number is not None and number > threshold


def _number_equals(value: Any, expected: float) -> bool:
    number = _to_float(value)
    return number is not None and abs(number - expected) <= 0.0001


def _to_float(value: Any) -> float | None:
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


def _string_value(value: Any) -> str | None:
    if value is None or isinstance(value, bool):
        return None
    text = str(value).strip()
    return text or None


def _is_missing(value: Any) -> bool:
    return value in (None, "", [], {})


def _canonical_record(record: Mapping[str, Any]) -> str:
    return _canonical_value(record)


def _canonical_value(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)


def _rate(count: int, total: int) -> float:
    if total == 0:
        return 0.0
    return round(count / total, 6)

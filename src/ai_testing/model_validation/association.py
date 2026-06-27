from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from statistics import mean, pstdev
from typing import Any

from ai_testing.model_training import AssociationRulesConfig, train_association_rules

Record = dict[str, Any]
Itemset = tuple[str, ...]


@dataclass(frozen=True)
class AssociationValidationConfig:
    basket_field: str = "basket_id"
    item_field: str = "product_group"
    min_support: float = 0.02
    min_confidence: float = 0.2
    min_lift: float = 1.05
    max_itemset_size: int = 2
    max_rules: int = 100
    max_itemsets: int = 500
    top_rules: int = 20


@dataclass(frozen=True)
class AssociationEvaluationConfig:
    basket_field: str = "basket_id"
    item_field: str = "product_group"
    min_confidence: float = 0.2
    min_lift: float = 1.05
    top_rules: int = 20


@dataclass(frozen=True)
class AssociationValidationFold:
    fold_index: int
    train_records: list[Record]
    validation_records: list[Record]


@dataclass(frozen=True)
class AssociationValidationResult:
    report: Record


@dataclass(frozen=True)
class AssociationEvaluationResult:
    report: Record


def validate_association_rules(
    folds: Sequence[AssociationValidationFold],
    config: AssociationValidationConfig | None = None,
) -> AssociationValidationResult:
    validation_config = config or AssociationValidationConfig()
    _validate_config(validation_config)
    if not folds:
        raise ValueError("At least one validation fold is required.")

    fold_reports = [
        _validate_fold(fold=fold, config=validation_config)
        for fold in sorted(folds, key=lambda item: item.fold_index)
    ]
    report = {
        "step": "Association rules validation",
        "validation_type": "k-fold cross-validation",
        "model_type": "association_rules",
        "algorithm": "apriori",
        "learning_type": "unsupervised",
        "parameters": _parameters(validation_config),
        "summary": _summary(fold_reports),
        "folds": fold_reports,
    }
    return AssociationValidationResult(report=report)


def evaluate_association_rules_on_records(
    model: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
    config: AssociationEvaluationConfig | None = None,
    dataset_name: str = "evaluation",
    metric_prefix: str = "evaluation",
) -> AssociationEvaluationResult:
    evaluation_config = config or AssociationEvaluationConfig()
    _validate_evaluation_config(evaluation_config)
    if not metric_prefix:
        raise ValueError("metric_prefix must not be empty")

    normalized_records = [dict(record) for record in records]
    rules = _rules_from_model(model)
    transactions = _transactions(
        records=normalized_records,
        basket_field=evaluation_config.basket_field,
        item_field=evaluation_config.item_field,
    )
    evaluated_rules = [
        _evaluate_rule(rule=rule, transactions=transactions, metric_prefix=metric_prefix)
        for rule in rules
    ]
    metrics = _evaluation_metrics(
        rules=evaluated_rules,
        transactions=transactions,
        min_confidence=evaluation_config.min_confidence,
        min_lift=evaluation_config.min_lift,
        metric_prefix=metric_prefix,
    )
    return AssociationEvaluationResult(
        report={
            "dataset": dataset_name,
            "model_type": model.get("model_type"),
            "algorithm": model.get("algorithm"),
            "learning_type": model.get("learning_type"),
            "training_input": model.get("training_input"),
            "record_count": len(normalized_records),
            "parameters": _evaluation_parameters(evaluation_config),
            "metrics": metrics,
            "top_rules": _top_rules(
                rules=evaluated_rules,
                limit=evaluation_config.top_rules,
                metric_prefix=metric_prefix,
            ),
        }
    )


def _validate_config(config: AssociationValidationConfig) -> None:
    AssociationRulesConfig(
        basket_field=config.basket_field,
        item_field=config.item_field,
        min_support=config.min_support,
        min_confidence=config.min_confidence,
        min_lift=config.min_lift,
        max_itemset_size=config.max_itemset_size,
        max_rules=config.max_rules,
        max_itemsets=config.max_itemsets,
    )
    if config.top_rules < 1:
        raise ValueError("top_rules must be at least 1")


def _validate_evaluation_config(config: AssociationEvaluationConfig) -> None:
    if not config.basket_field:
        raise ValueError("basket_field must not be empty")
    if not config.item_field:
        raise ValueError("item_field must not be empty")
    if not 0 < config.min_confidence <= 1:
        raise ValueError("min_confidence must be in the (0, 1] range")
    if config.min_lift < 0:
        raise ValueError("min_lift must be greater than or equal to 0")
    if config.top_rules < 1:
        raise ValueError("top_rules must be at least 1")


def _validate_fold(
    fold: AssociationValidationFold,
    config: AssociationValidationConfig,
) -> Record:
    training_result = train_association_rules(
        fold.train_records,
        config=AssociationRulesConfig(
            basket_field=config.basket_field,
            item_field=config.item_field,
            min_support=config.min_support,
            min_confidence=config.min_confidence,
            min_lift=config.min_lift,
            max_itemset_size=config.max_itemset_size,
            max_rules=config.max_rules,
            max_itemsets=config.max_itemsets,
        ),
    )
    model = training_result.model
    evaluation_result = evaluate_association_rules_on_records(
        model=model,
        records=fold.validation_records,
        config=_evaluation_config_from_validation_config(config),
        dataset_name=f"fold_{fold.fold_index:02d}_validation",
        metric_prefix="validation",
    )
    evaluation_report = evaluation_result.report
    return {
        "fold_index": fold.fold_index,
        "train": {
            "record_count": len(fold.train_records),
            "basket_count": _int_summary(model, "basket_count"),
            "item_count": _int_summary(model, "item_count"),
            "frequent_itemset_count": _int_summary(model, "frequent_itemset_count"),
            "rule_count": _int_summary(model, "rule_count"),
            "exported_rule_count": _int_summary(model, "exported_rule_count"),
        },
        "validation": {
            "record_count": len(fold.validation_records),
            **evaluation_report["metrics"],
        },
        "top_rules": evaluation_report["top_rules"],
    }


def _evaluation_config_from_validation_config(
    config: AssociationValidationConfig,
) -> AssociationEvaluationConfig:
    return AssociationEvaluationConfig(
        basket_field=config.basket_field,
        item_field=config.item_field,
        min_confidence=config.min_confidence,
        min_lift=config.min_lift,
        top_rules=config.top_rules,
    )


def _rules_from_model(model: Mapping[str, Any]) -> list[Record]:
    rules = model.get("rules")
    if not isinstance(rules, list):
        return []
    return [dict(rule) for rule in rules if isinstance(rule, Mapping)]


def _evaluate_rule(
    rule: Mapping[str, Any],
    transactions: Sequence[frozenset[str]],
    metric_prefix: str,
) -> Record:
    antecedent = _itemset(rule.get("antecedent"))
    consequent = _itemset(rule.get("consequent"))
    basket_count = len(transactions)
    antecedent_count = sum(1 for transaction in transactions if antecedent <= transaction)
    consequent_count = sum(1 for transaction in transactions if consequent <= transaction)
    support_count = sum(
        1 for transaction in transactions if antecedent <= transaction and consequent <= transaction
    )
    validation_support = _rate(support_count, basket_count)
    validation_antecedent_support = _rate(antecedent_count, basket_count)
    validation_consequent_support = _rate(consequent_count, basket_count)
    validation_confidence = _rate(support_count, antecedent_count) if antecedent_count else None
    validation_lift = (
        _round(validation_confidence / validation_consequent_support)
        if validation_confidence is not None and validation_consequent_support > 0
        else None
    )
    train_confidence = _float_value(rule.get("confidence"))
    train_lift = _float_value(rule.get("lift"))
    return {
        "antecedent": sorted(antecedent),
        "consequent": sorted(consequent),
        "train_support": _float_value(rule.get("support")),
        "train_confidence": train_confidence,
        "train_lift": train_lift,
        f"{metric_prefix}_support_count": support_count,
        f"{metric_prefix}_antecedent_count": antecedent_count,
        f"{metric_prefix}_consequent_count": consequent_count,
        f"{metric_prefix}_support": validation_support,
        f"{metric_prefix}_antecedent_support": validation_antecedent_support,
        f"{metric_prefix}_consequent_support": validation_consequent_support,
        f"{metric_prefix}_confidence": validation_confidence,
        f"{metric_prefix}_lift": validation_lift,
        "confidence_gap": _gap(train_confidence, validation_confidence),
        "lift_gap": _gap(train_lift, validation_lift),
    }


def _evaluation_metrics(
    rules: Sequence[Record],
    transactions: Sequence[frozenset[str]],
    min_confidence: float,
    min_lift: float,
    metric_prefix: str,
) -> Record:
    confidence_key = f"{metric_prefix}_confidence"
    lift_key = f"{metric_prefix}_lift"
    antecedent_count_key = f"{metric_prefix}_antecedent_count"
    confidences = _numbers(rule.get(confidence_key) for rule in rules)
    lifts = _numbers(rule.get(lift_key) for rule in rules)
    confidence_gaps = _numbers(rule.get("confidence_gap") for rule in rules)
    lift_gaps = _numbers(rule.get("lift_gap") for rule in rules)
    covered_baskets = _covered_basket_count(rules, transactions, require_consequent=False)
    hit_baskets = _covered_basket_count(rules, transactions, require_consequent=True)
    stable_rules = [
        rule
        for rule in rules
        if _passes_threshold(rule.get(confidence_key), min_confidence)
        and _passes_threshold(rule.get(lift_key), min_lift)
    ]
    zero_antecedent_rules = [rule for rule in rules if rule.get(antecedent_count_key) == 0]
    return {
        "basket_count": len(transactions),
        "evaluated_rule_count": len(rules),
        "stable_rule_count": len(stable_rules),
        "zero_antecedent_rule_count": len(zero_antecedent_rules),
        "antecedent_coverage_basket_count": covered_baskets,
        "antecedent_coverage": _rate(covered_baskets, len(transactions)),
        "hit_basket_count": hit_baskets,
        f"hit_rate_per_{metric_prefix}_basket": _rate(hit_baskets, len(transactions)),
        "hit_rate_per_covered_basket": _rate(hit_baskets, covered_baskets),
        f"mean_{metric_prefix}_confidence": _mean(confidences),
        f"mean_{metric_prefix}_lift": _mean(lifts),
        "mean_abs_confidence_gap": _mean([abs(value) for value in confidence_gaps]),
        "mean_abs_lift_gap": _mean([abs(value) for value in lift_gaps]),
    }


def _covered_basket_count(
    rules: Sequence[Mapping[str, Any]],
    transactions: Sequence[frozenset[str]],
    require_consequent: bool,
) -> int:
    count = 0
    for transaction in transactions:
        for rule in rules:
            antecedent = set(_itemset(rule.get("antecedent")))
            consequent = set(_itemset(rule.get("consequent")))
            if not antecedent <= transaction:
                continue
            if require_consequent and not consequent <= transaction:
                continue
            count += 1
            break
    return count


def _top_rules(
    rules: Sequence[Record],
    limit: int,
    metric_prefix: str = "validation",
) -> list[Record]:
    return sorted(
        rules,
        key=lambda rule: (
            -_sort_number(rule.get(f"{metric_prefix}_lift")),
            -_sort_number(rule.get(f"{metric_prefix}_confidence")),
            -_sort_number(rule.get(f"{metric_prefix}_support")),
            rule.get("antecedent", []),
            rule.get("consequent", []),
        ),
    )[:limit]


def _summary(fold_reports: Sequence[Record]) -> Record:
    validation = [dict(report["validation"]) for report in fold_reports]
    return {
        "fold_count": len(fold_reports),
        "mean_train_rule_count": _mean(
            _int_path(report, "train", "rule_count") for report in fold_reports
        ),
        "mean_validation_confidence": _mean_metric(
            validation,
            "mean_validation_confidence",
        ),
        "std_validation_confidence": _std_metric(
            validation,
            "mean_validation_confidence",
        ),
        "mean_validation_lift": _mean_metric(validation, "mean_validation_lift"),
        "std_validation_lift": _std_metric(validation, "mean_validation_lift"),
        "mean_antecedent_coverage": _mean_metric(validation, "antecedent_coverage"),
        "mean_hit_rate_per_covered_basket": _mean_metric(
            validation,
            "hit_rate_per_covered_basket",
        ),
        "mean_abs_confidence_gap": _mean_metric(validation, "mean_abs_confidence_gap"),
        "mean_stable_rule_count": _mean_metric(validation, "stable_rule_count"),
    }


def _parameters(config: AssociationValidationConfig) -> Record:
    return {
        "basket_field": config.basket_field,
        "item_field": config.item_field,
        "min_support": config.min_support,
        "min_confidence": config.min_confidence,
        "min_lift": config.min_lift,
        "max_itemset_size": config.max_itemset_size,
        "max_rules": config.max_rules,
        "max_itemsets": config.max_itemsets,
        "top_rules": config.top_rules,
    }


def _evaluation_parameters(config: AssociationEvaluationConfig) -> Record:
    return {
        "basket_field": config.basket_field,
        "item_field": config.item_field,
        "min_confidence": config.min_confidence,
        "min_lift": config.min_lift,
        "top_rules": config.top_rules,
    }


def _transactions(
    records: Sequence[Mapping[str, Any]],
    basket_field: str,
    item_field: str,
) -> list[frozenset[str]]:
    basket_items: dict[str, set[str]] = {}
    for record in records:
        basket_id = _string_value(record.get(basket_field))
        item = _string_value(record.get(item_field))
        if basket_id is None or item is None:
            continue
        basket_items.setdefault(basket_id, set()).add(item)
    return [frozenset(items) for _, items in sorted(basket_items.items()) if items]


def _itemset(value: Any) -> frozenset[str]:
    if not isinstance(value, Sequence) or isinstance(value, str):
        return frozenset()
    return frozenset(text for item in value for text in [_string_value(item)] if text is not None)


def _int_summary(model: Mapping[str, Any], key: str) -> int:
    summary = model.get("summary")
    if not isinstance(summary, Mapping):
        return 0
    value = summary.get(key)
    return value if isinstance(value, int) and not isinstance(value, bool) else 0


def _int_path(record: Mapping[str, Any], first: str, second: str) -> int:
    nested = record.get(first)
    if not isinstance(nested, Mapping):
        return 0
    value = nested.get(second)
    return value if isinstance(value, int) and not isinstance(value, bool) else 0


def _mean_metric(records: Sequence[Mapping[str, Any]], key: str) -> float | None:
    return _mean(_numbers(record.get(key) for record in records))


def _std_metric(records: Sequence[Mapping[str, Any]], key: str) -> float | None:
    values = _numbers(record.get(key) for record in records)
    if not values:
        return None
    return _round(pstdev(values))


def _numbers(values: Iterable[Any]) -> list[float]:
    return [number for value in values for number in [_float_value(value)] if number is not None]


def _float_value(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _string_value(value: Any) -> str | None:
    if value is None or isinstance(value, bool):
        return None
    text = str(value).strip()
    return text or None


def _passes_threshold(value: Any, threshold: float) -> bool:
    number = _float_value(value)
    return number is not None and number >= threshold


def _gap(left: float | None, right: float | None) -> float | None:
    if left is None or right is None:
        return None
    return _round(left - right)


def _mean(values: Iterable[float]) -> float | None:
    numeric_values = list(values)
    if not numeric_values:
        return None
    return _round(mean(numeric_values))


def _rate(count: int, total: int) -> float:
    if total == 0:
        return 0.0
    return _round(count / total)


def _sort_number(value: Any) -> float:
    number = _float_value(value)
    return number if number is not None else -1.0


def _round(value: float) -> float:
    return round(value, 6)

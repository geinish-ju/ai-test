from __future__ import annotations

import math
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import combinations
from typing import Any

Record = dict[str, Any]
Itemset = tuple[str, ...]


@dataclass(frozen=True)
class AssociationRulesConfig:
    basket_field: str = "basket_id"
    item_field: str = "product_group"
    min_support: float = 0.02
    min_confidence: float = 0.2
    min_lift: float = 1.05
    max_itemset_size: int = 2
    max_rules: int = 100
    max_itemsets: int = 500


@dataclass(frozen=True)
class AssociationTrainingResult:
    model: Record


@dataclass(frozen=True)
class _Rule:
    antecedent: Itemset
    consequent: Itemset
    support_count: int
    support: float
    antecedent_support: float
    consequent_support: float
    confidence: float
    lift: float
    leverage: float
    conviction: float | None


def train_association_rules(
    records: Sequence[Mapping[str, Any]],
    config: AssociationRulesConfig | None = None,
) -> AssociationTrainingResult:
    training_config = config or AssociationRulesConfig()
    _validate_config(training_config)

    transactions = _build_transactions(records, training_config)
    basket_count = len(transactions)
    if basket_count == 0:
        return AssociationTrainingResult(
            model=_empty_model(records, training_config, "No baskets with usable items found.")
        )

    min_support_count = max(1, math.ceil(training_config.min_support * basket_count))
    support_counts = _frequent_itemsets(
        transactions=transactions,
        min_support_count=min_support_count,
        max_itemset_size=training_config.max_itemset_size,
    )
    rules = _association_rules(
        support_counts=support_counts,
        basket_count=basket_count,
        min_confidence=training_config.min_confidence,
        min_lift=training_config.min_lift,
    )

    model = {
        "model_type": "association_rules",
        "algorithm": "apriori",
        "learning_type": "unsupervised",
        "training_dataset": "training",
        "parameters": {
            "basket_field": training_config.basket_field,
            "item_field": training_config.item_field,
            "min_support": training_config.min_support,
            "min_confidence": training_config.min_confidence,
            "min_lift": training_config.min_lift,
            "max_itemset_size": training_config.max_itemset_size,
            "max_rules": training_config.max_rules,
            "max_itemsets": training_config.max_itemsets,
            "min_support_count": min_support_count,
        },
        "summary": {
            "input_record_count": len(records),
            "basket_count": basket_count,
            "item_count": _item_count(transactions),
            "frequent_itemset_count": len(support_counts),
            "rule_count": len(rules),
            "exported_rule_count": min(len(rules), training_config.max_rules),
        },
        "frequent_itemsets": _serialize_itemsets(
            support_counts=support_counts,
            basket_count=basket_count,
            limit=training_config.max_itemsets,
        ),
        "rules": [_serialize_rule(rule) for rule in rules[: training_config.max_rules]],
    }
    return AssociationTrainingResult(model=model)


def _validate_config(config: AssociationRulesConfig) -> None:
    if not config.basket_field:
        raise ValueError("basket_field must not be empty")
    if not config.item_field:
        raise ValueError("item_field must not be empty")
    if not 0 < config.min_support <= 1:
        raise ValueError("min_support must be in the (0, 1] range")
    if not 0 < config.min_confidence <= 1:
        raise ValueError("min_confidence must be in the (0, 1] range")
    if config.min_lift < 0:
        raise ValueError("min_lift must be greater than or equal to 0")
    if config.max_itemset_size < 2:
        raise ValueError("max_itemset_size must be at least 2")
    if config.max_rules < 1:
        raise ValueError("max_rules must be at least 1")
    if config.max_itemsets < 1:
        raise ValueError("max_itemsets must be at least 1")


def _build_transactions(
    records: Sequence[Mapping[str, Any]],
    config: AssociationRulesConfig,
) -> list[Itemset]:
    basket_items: dict[str, set[str]] = {}
    for record in records:
        basket_id = _string_value(record.get(config.basket_field))
        item = _string_value(record.get(config.item_field))
        if basket_id is None or item is None:
            continue
        basket_items.setdefault(basket_id, set()).add(item)

    return [tuple(sorted(items)) for _, items in sorted(basket_items.items()) if items]


def _frequent_itemsets(
    transactions: Sequence[Itemset],
    min_support_count: int,
    max_itemset_size: int,
) -> dict[Itemset, int]:
    singleton_counts: Counter[Itemset] = Counter(
        (item,) for transaction in transactions for item in transaction
    )
    frequent_supports: dict[Itemset, int] = {
        itemset: support_count
        for itemset, support_count in singleton_counts.items()
        if support_count >= min_support_count
    }
    previous_frequent = set(frequent_supports)

    for size in range(2, max_itemset_size + 1):
        candidates = _join_candidates(previous_frequent, size)
        if not candidates:
            break

        counts: Counter[Itemset] = Counter()
        for transaction in transactions:
            frequent_items = tuple(item for item in transaction if (item,) in frequent_supports)
            if len(frequent_items) < size:
                continue
            for candidate in combinations(frequent_items, size):
                itemset = tuple(sorted(candidate))
                if itemset in candidates:
                    counts[itemset] += 1

        current_frequent = {
            itemset: support_count
            for itemset, support_count in counts.items()
            if support_count >= min_support_count
        }
        if not current_frequent:
            break

        frequent_supports.update(current_frequent)
        previous_frequent = set(current_frequent)

    return frequent_supports


def _join_candidates(previous_frequent: set[Itemset], size: int) -> set[Itemset]:
    candidates: set[Itemset] = set()
    previous = sorted(previous_frequent)
    for left_index, left in enumerate(previous):
        for right in previous[left_index + 1 :]:
            candidate = tuple(sorted(set(left) | set(right)))
            if len(candidate) != size:
                continue
            if all(
                tuple(sorted(subset)) in previous_frequent
                for subset in combinations(candidate, size - 1)
            ):
                candidates.add(candidate)
    return candidates


def _association_rules(
    support_counts: Mapping[Itemset, int],
    basket_count: int,
    min_confidence: float,
    min_lift: float,
) -> list[_Rule]:
    rules: list[_Rule] = []
    for itemset, support_count in support_counts.items():
        if len(itemset) < 2:
            continue
        for antecedent_size in range(1, len(itemset)):
            for antecedent in combinations(itemset, antecedent_size):
                consequent = tuple(item for item in itemset if item not in antecedent)
                if not consequent:
                    continue

                antecedent_count = support_counts.get(tuple(sorted(antecedent)))
                consequent_count = support_counts.get(tuple(sorted(consequent)))
                if antecedent_count is None or consequent_count is None:
                    continue

                support = support_count / basket_count
                antecedent_support = antecedent_count / basket_count
                consequent_support = consequent_count / basket_count
                confidence = support_count / antecedent_count
                lift = confidence / consequent_support if consequent_support else 0.0
                if confidence < min_confidence or lift < min_lift:
                    continue

                rules.append(
                    _Rule(
                        antecedent=tuple(sorted(antecedent)),
                        consequent=tuple(sorted(consequent)),
                        support_count=support_count,
                        support=_round(support),
                        antecedent_support=_round(antecedent_support),
                        consequent_support=_round(consequent_support),
                        confidence=_round(confidence),
                        lift=_round(lift),
                        leverage=_round(support - (antecedent_support * consequent_support)),
                        conviction=_conviction(confidence, consequent_support),
                    )
                )

    return sorted(
        rules,
        key=lambda rule: (
            -rule.lift,
            -rule.confidence,
            -rule.support,
            rule.antecedent,
            rule.consequent,
        ),
    )


def _serialize_itemsets(
    support_counts: Mapping[Itemset, int],
    basket_count: int,
    limit: int,
) -> list[Record]:
    sorted_itemsets = sorted(
        support_counts.items(),
        key=lambda item: (-len(item[0]), -item[1], item[0]),
    )
    return [
        {
            "items": list(itemset),
            "size": len(itemset),
            "support_count": support_count,
            "support": _round(support_count / basket_count),
        }
        for itemset, support_count in sorted_itemsets[:limit]
    ]


def _serialize_rule(rule: _Rule) -> Record:
    return {
        "antecedent": list(rule.antecedent),
        "consequent": list(rule.consequent),
        "support_count": rule.support_count,
        "support": rule.support,
        "antecedent_support": rule.antecedent_support,
        "consequent_support": rule.consequent_support,
        "confidence": rule.confidence,
        "lift": rule.lift,
        "leverage": rule.leverage,
        "conviction": rule.conviction,
    }


def _empty_model(
    records: Sequence[Mapping[str, Any]],
    config: AssociationRulesConfig,
    warning: str,
) -> Record:
    return {
        "model_type": "association_rules",
        "algorithm": "apriori",
        "learning_type": "unsupervised",
        "training_dataset": "training",
        "parameters": {
            "basket_field": config.basket_field,
            "item_field": config.item_field,
            "min_support": config.min_support,
            "min_confidence": config.min_confidence,
            "min_lift": config.min_lift,
            "max_itemset_size": config.max_itemset_size,
            "max_rules": config.max_rules,
            "max_itemsets": config.max_itemsets,
        },
        "summary": {
            "input_record_count": len(records),
            "basket_count": 0,
            "item_count": 0,
            "frequent_itemset_count": 0,
            "rule_count": 0,
            "exported_rule_count": 0,
        },
        "warnings": [warning],
        "frequent_itemsets": [],
        "rules": [],
    }


def _item_count(transactions: Sequence[Itemset]) -> int:
    return len({item for transaction in transactions for item in transaction})


def _string_value(value: Any) -> str | None:
    if value is None or isinstance(value, bool):
        return None
    text = str(value).strip()
    return text or None


def _conviction(confidence: float, consequent_support: float) -> float | None:
    if confidence >= 1:
        return None
    return _round((1 - consequent_support) / (1 - confidence))


def _round(value: float) -> float:
    return round(value, 6)

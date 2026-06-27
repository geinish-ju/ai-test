from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import PurePath
from typing import Any

from ai_testing.core import add_check as _add_check
from ai_testing.core import add_threshold_check as _add_threshold_check
from ai_testing.core import build_standard_report

Record = dict[str, Any]


@dataclass(frozen=True)
class AssociationMLModelTestConfig:
    expected_model_type: str = "association_rules"
    expected_algorithm: str = "apriori"
    expected_learning_type: str = "unsupervised"
    test_dataset_input: str = "data/splits/test.json"
    forbidden_feature_fields: tuple[str, ...] = (
        "order_id",
        "order_number",
        "order_item_id",
        "product_id",
        "product_slug",
        "product_url",
        "main_category_id",
    )
    min_validation_fold_count: int = 2
    min_train_rule_count: int = 1
    min_test_basket_count: int = 1
    min_evaluated_rule_count: int = 1
    min_stable_rule_count: int = 1
    min_mean_validation_confidence: float = 0.2
    min_mean_validation_lift: float = 1.0
    min_mean_test_confidence: float = 0.2
    min_mean_test_lift: float = 1.0
    min_test_antecedent_coverage: float = 0.5
    min_test_hit_rate_per_covered_basket: float = 0.5
    max_mean_abs_validation_confidence_gap: float = 0.2
    max_mean_abs_test_confidence_gap: float = 0.2
    max_validation_test_confidence_delta: float = 0.15
    max_validation_test_lift_delta: float = 0.25


@dataclass(frozen=True)
class AssociationMLModelTestResult:
    report: Record


def test_association_ml_model(
    model: Mapping[str, Any],
    validation_report: Mapping[str, Any],
    test_report: Mapping[str, Any],
    config: AssociationMLModelTestConfig | None = None,
) -> AssociationMLModelTestResult:
    test_config = config or AssociationMLModelTestConfig()
    checks: list[Record] = []
    model_summary = _mapping(model.get("summary"))
    model_parameters = _mapping(model.get("parameters"))
    validation_summary = _mapping(validation_report.get("summary"))
    test_summary = _mapping(test_report.get("test"))

    _add_check(
        checks,
        check_id="ml_model.artifact_identity",
        passed=(
            model.get("model_type") == test_config.expected_model_type
            and model.get("algorithm") == test_config.expected_algorithm
            and model.get("learning_type") == test_config.expected_learning_type
        ),
        severity="critical",
        message="Model artifact has the expected type, algorithm, and learning type.",
        observed={
            "model_type": model.get("model_type"),
            "algorithm": model.get("algorithm"),
            "learning_type": model.get("learning_type"),
        },
        expected={
            "model_type": test_config.expected_model_type,
            "algorithm": test_config.expected_algorithm,
            "learning_type": test_config.expected_learning_type,
        },
    )
    _add_check(
        checks,
        check_id="ml_model.training_input_is_not_test",
        passed=not _same_path(model.get("training_input"), test_config.test_dataset_input),
        severity="critical",
        message="Model training input is not the final hold-out test dataset.",
        observed={"training_input": model.get("training_input")},
        expected={"not_equal_to": test_config.test_dataset_input},
    )
    _add_check(
        checks,
        check_id="ml_model.feature_leakage",
        passed=_string_value(model_parameters.get("item_field"))
        not in set(test_config.forbidden_feature_fields),
        severity="critical",
        message="The trained item field is not a technical identifier or leakage-prone field.",
        observed={"item_field": model_parameters.get("item_field")},
        expected={"not_in": sorted(test_config.forbidden_feature_fields)},
    )
    _add_check(
        checks,
        check_id="ml_model.rules_available",
        passed=_int_value(model_summary.get("rule_count")) >= test_config.min_train_rule_count
        and _valid_rules(model.get("rules")),
        severity="critical",
        message="Model contains usable association rules.",
        observed={
            "rule_count": _int_value(model_summary.get("rule_count")),
            "exported_rule_count": _int_value(model_summary.get("exported_rule_count")),
            "invalid_rule_count": _invalid_rule_count(model.get("rules")),
        },
        expected={">=": test_config.min_train_rule_count, "invalid_rule_count": 0},
    )
    _add_check(
        checks,
        check_id="ml_model.validation_report_shape",
        passed=(
            validation_report.get("validation_type") == "k-fold cross-validation"
            and _int_value(validation_summary.get("fold_count"))
            >= test_config.min_validation_fold_count
        ),
        severity="critical",
        message="Validation report is based on k-fold cross-validation.",
        observed={
            "validation_type": validation_report.get("validation_type"),
            "fold_count": validation_summary.get("fold_count"),
        },
        expected={"fold_count": f">= {test_config.min_validation_fold_count}"},
    )
    _add_threshold_check(
        checks,
        check_id="ml_model.validation_confidence",
        value=_float_value(validation_summary.get("mean_validation_confidence")),
        threshold=test_config.min_mean_validation_confidence,
        direction=">=",
        severity="major",
        message="Mean validation confidence meets the acceptance threshold.",
    )
    _add_threshold_check(
        checks,
        check_id="ml_model.validation_lift",
        value=_float_value(validation_summary.get("mean_validation_lift")),
        threshold=test_config.min_mean_validation_lift,
        direction=">=",
        severity="major",
        message="Mean validation lift meets the acceptance threshold.",
    )
    _add_threshold_check(
        checks,
        check_id="ml_model.validation_confidence_gap",
        value=_float_value(validation_summary.get("mean_abs_confidence_gap")),
        threshold=test_config.max_mean_abs_validation_confidence_gap,
        direction="<=",
        severity="major",
        message="Mean train-vs-validation confidence gap is within threshold.",
    )
    _add_check(
        checks,
        check_id="ml_model.test_report_shape",
        passed=test_report.get("testing_type") == "hold-out test",
        severity="critical",
        message="Final test report is based on the hold-out test dataset.",
        observed={"testing_type": test_report.get("testing_type")},
        expected={"testing_type": "hold-out test"},
    )
    _add_threshold_check(
        checks,
        check_id="ml_model.test_basket_count",
        value=_float_value(test_summary.get("basket_count")),
        threshold=float(test_config.min_test_basket_count),
        direction=">=",
        severity="critical",
        message="Hold-out test report contains enough baskets.",
    )
    _add_threshold_check(
        checks,
        check_id="ml_model.test_rule_count",
        value=_float_value(test_summary.get("evaluated_rule_count")),
        threshold=float(test_config.min_evaluated_rule_count),
        direction=">=",
        severity="critical",
        message="Hold-out test report evaluates enough rules.",
    )
    _add_threshold_check(
        checks,
        check_id="ml_model.test_stable_rule_count",
        value=_float_value(test_summary.get("stable_rule_count")),
        threshold=float(test_config.min_stable_rule_count),
        direction=">=",
        severity="major",
        message="Enough rules remain stable on the hold-out test dataset.",
    )
    _add_threshold_check(
        checks,
        check_id="ml_model.test_confidence",
        value=_float_value(test_summary.get("mean_test_confidence")),
        threshold=test_config.min_mean_test_confidence,
        direction=">=",
        severity="major",
        message="Mean test confidence meets the acceptance threshold.",
    )
    _add_threshold_check(
        checks,
        check_id="ml_model.test_lift",
        value=_float_value(test_summary.get("mean_test_lift")),
        threshold=test_config.min_mean_test_lift,
        direction=">=",
        severity="major",
        message="Mean test lift meets the acceptance threshold.",
    )
    _add_threshold_check(
        checks,
        check_id="ml_model.test_coverage",
        value=_float_value(test_summary.get("antecedent_coverage")),
        threshold=test_config.min_test_antecedent_coverage,
        direction=">=",
        severity="major",
        message="Antecedent coverage on the test dataset meets threshold.",
    )
    _add_threshold_check(
        checks,
        check_id="ml_model.test_hit_rate",
        value=_float_value(test_summary.get("hit_rate_per_covered_basket")),
        threshold=test_config.min_test_hit_rate_per_covered_basket,
        direction=">=",
        severity="major",
        message="Hit rate on covered test baskets meets threshold.",
    )
    _add_threshold_check(
        checks,
        check_id="ml_model.test_confidence_gap",
        value=_float_value(test_summary.get("mean_abs_confidence_gap")),
        threshold=test_config.max_mean_abs_test_confidence_gap,
        direction="<=",
        severity="major",
        message="Mean train-vs-test confidence gap is within threshold.",
    )
    _add_threshold_check(
        checks,
        check_id="ml_model.validation_test_confidence_delta",
        value=_absolute_delta(
            validation_summary.get("mean_validation_confidence"),
            test_summary.get("mean_test_confidence"),
        ),
        threshold=test_config.max_validation_test_confidence_delta,
        direction="<=",
        severity="major",
        message="Validation and test confidence are close enough.",
    )
    _add_threshold_check(
        checks,
        check_id="ml_model.validation_test_lift_delta",
        value=_absolute_delta(
            validation_summary.get("mean_validation_lift"),
            test_summary.get("mean_test_lift"),
        ),
        threshold=test_config.max_validation_test_lift_delta,
        direction="<=",
        severity="major",
        message="Validation and test lift are close enough.",
    )

    return AssociationMLModelTestResult(
        report=build_standard_report(
            report_type="model_quality_test",
            subject="association_rules",
            step="ML model testing",
            testing_type="association model acceptance testing",
            checks=checks,
            details={
                "model": {
                    "model_type": model.get("model_type"),
                    "algorithm": model.get("algorithm"),
                    "learning_type": model.get("learning_type"),
                    "training_input": model.get("training_input"),
                    "rule_count": model_summary.get("rule_count"),
                    "exported_rule_count": model_summary.get("exported_rule_count"),
                    "item_field": model_parameters.get("item_field"),
                },
                "validation_metrics": validation_summary,
                "test_metrics": test_summary,
            },
        )
    )


def _valid_rules(value: Any) -> bool:
    return _invalid_rule_count(value) == 0


def _invalid_rule_count(value: Any) -> int:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return 1
    count = 0
    for rule in value:
        if not isinstance(rule, Mapping):
            count += 1
            continue
        if not _itemset(rule.get("antecedent")) or not _itemset(rule.get("consequent")):
            count += 1
    return count


def _itemset(value: Any) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return ()
    return tuple(item for raw in value for item in [_string_value(raw)] if item is not None)


def _same_path(left: Any, right: str) -> bool:
    left_text = _string_value(left)
    if left_text is None:
        return False
    return PurePath(left_text).as_posix().lower() == PurePath(right).as_posix().lower()


def _absolute_delta(left: Any, right: Any) -> float | None:
    left_number = _float_value(left)
    right_number = _float_value(right)
    if left_number is None or right_number is None:
        return None
    return round(abs(left_number - right_number), 6)


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _int_value(value: Any) -> int:
    number = _float_value(value)
    return int(number) if number is not None else 0


def _float_value(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip())
        except ValueError:
            return None
    return None


def _string_value(value: Any) -> str | None:
    if value is None or isinstance(value, bool):
        return None
    text = str(value).strip()
    return text or None

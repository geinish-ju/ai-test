from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import pytest

from ai_testing.input_data_testing import InputDataTestConfig

Record = dict[str, Any]


@pytest.fixture
def cli_args(tmp_path: Path) -> Callable[..., list[str]]:
    def build(*args: str) -> list[str]:
        return [
            "--stage-history-output",
            str(tmp_path / "stage_history.json"),
            "--stage-history-markdown-output",
            str(tmp_path / "stage_history.md"),
            "--stage-metric-history-markdown-output",
            str(tmp_path / "stage_metric_history.md"),
            *args,
        ]

    return build


@pytest.fixture
def read_json_file() -> Callable[[Path], Any]:
    def read(path: Path) -> Any:
        return json.loads(path.read_text(encoding="utf-8"))

    return read


@pytest.fixture
def write_json_file() -> Callable[[Path, Any], None]:
    def write(path: Path, payload: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    return write


@pytest.fixture
def quality_report_factory() -> Callable[..., Record]:
    def build(
        report_type: str,
        status: str = "passed",
        *,
        failed_check: Mapping[str, Any] | None = None,
        subject: str = "fixture",
    ) -> Record:
        checks: list[Record] = []
        if failed_check is not None:
            checks.append({**dict(failed_check), "status": "failed"})
        return {
            "report_type": report_type,
            "subject": subject,
            "status": status,
            "summary": {
                "check_count": len(checks) or 1,
                "passed_count": 0 if checks else 1,
                "failed_count": len(checks),
            },
            "checks": checks
            or [
                {
                    "id": "fixture.passed",
                    "status": "passed",
                    "severity": "critical",
                    "message": "Fixture check passed.",
                    "observed": {"status": "passed"},
                    "expected": {"status": "passed"},
                }
            ],
        }

    return build


@pytest.fixture
def passed_quality_report(quality_report_factory: Callable[..., Record]) -> Callable[[str], Record]:
    def build(report_type: str) -> Record:
        return quality_report_factory(report_type, "passed")

    return build


@pytest.fixture
def grocery_record_factory() -> Callable[..., Record]:
    def build(
        basket_id: str,
        shop: str,
        product_name: str,
        quantity_ordered: float,
    ) -> Record:
        return {
            "shop": shop,
            "order_date": "2025-01-06",
            "product_name": product_name,
            "brand": "Fixture",
            "main_category": "Dairy" if product_name != "Bread" else "Bakery",
            "category": "Fixture category",
            "quantity_ordered": quantity_ordered,
            "unit": "pcs",
            "package_quantity": 1.0,
            "package_unit": "pcs",
            "price_unit": 10.0,
            "price_per_unit": 10.0,
            "price_per_unit_unit": "pcs",
            "currency": "CZK",
            "basket_id": basket_id,
            "order_month": 1,
            "order_week_of_year": 2,
            "order_day_of_week": 0,
            "order_is_weekend": False,
            "order_quarter": 1,
        }

    return build


@pytest.fixture
def input_data_test_config() -> InputDataTestConfig:
    return InputDataTestConfig(
        required_fields=(
            "shop",
            "order_date",
            "product_name",
            "main_category",
            "category",
            "quantity_ordered",
            "price_unit",
            "price_per_unit",
            "currency",
            "basket_id",
            "order_month",
            "order_week_of_year",
            "order_day_of_week",
            "order_is_weekend",
            "order_quarter",
        ),
        protected_fields=("product_id", "order_item_id"),
        critical_fields=("shop", "order_date", "product_name", "quantity_ordered"),
        coverage_fields=("brand", "main_category", "category"),
    )


@pytest.fixture
def check_by_id() -> Callable[[Record, str], Record]:
    def find(report: Record, check_id: str) -> Record:
        checks = report["checks"]
        if not isinstance(checks, list):
            raise AssertionError("Report checks must be a list.")
        return next(
            check for check in checks if isinstance(check, dict) and check["id"] == check_id
        )

    return find


@pytest.fixture
def classification_validation_report() -> Record:
    return {
        "validation_type": "k-fold cross-validation",
        "model_type": "text_classification",
        "learning_type": "supervised",
        "summary": {
            "fold_count": 5,
            "mean_accuracy": 0.95,
            "mean_macro_precision": 0.9,
            "mean_macro_recall": 0.91,
            "mean_macro_f1": 0.9,
            "mean_weighted_f1": 0.95,
            "std_accuracy": 0.01,
            "std_macro_f1": 0.02,
        },
    }


@pytest.fixture
def classification_test_report() -> Record:
    return {
        "testing_type": "hold-out test",
        "model_type": "text_classification",
        "learning_type": "supervised",
        "test": {
            "evaluated_record_count": 40,
            "class_count": 4,
            "accuracy": 0.94,
            "macro_precision": 0.9,
            "macro_recall": 0.89,
            "macro_f1": 0.88,
            "weighted_f1": 0.94,
        },
    }


@pytest.fixture
def failed_check_ids() -> Callable[[Record], set[str]]:
    def collect(report: Record) -> set[str]:
        checks = report["checks"]
        if not isinstance(checks, list):
            return set()
        return {
            str(check["id"])
            for check in checks
            if isinstance(check, Mapping) and check.get("status") == "failed"
        }

    return collect

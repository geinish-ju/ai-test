from __future__ import annotations

from typing import Any

from ai_testing.input_data_testing import (
    InputDataFold,
    InputDataTestConfig,
)
from ai_testing.input_data_testing import (
    test_input_data as run_input_data_test,
)


def test_input_data_testing_reports_numeric_contract_diagnostics() -> None:
    records = [
        _record("basket-1", "kosik", "Milk", 1.0),
        _record("basket-2", "rohlik", "Bread", 2.0),
        _record("basket-3", "kosik", "Cheese", 1.0),
        _record("basket-4", "rohlik", "Cancelled item", 0.0),
    ]
    train_validation = records[:2]
    test = records[2:]
    folds = [
        InputDataFold(fold_index=0, train_records=[records[1]], validation_records=[records[0]]),
        InputDataFold(fold_index=1, train_records=[records[0]], validation_records=[records[1]]),
    ]

    report = run_input_data_test(
        processed_records=records,
        train_validation_records=train_validation,
        test_records=test,
        folds=folds,
        config=InputDataTestConfig(
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
        ),
    ).report

    numeric_check = _check(report, "input_data.numeric_values")
    pandera_check = _check(report, "input_data.pandera_contract")

    assert report["status"] == "failed"
    assert numeric_check["status"] == "failed"
    assert numeric_check["observed"] == {"quantity_ordered": 1}
    assert (
        numeric_check["diagnostics"]["fields"]["quantity_ordered"]["sample_records"][0][
            "product_name"
        ]
        == "Cancelled item"
    )
    assert pandera_check["status"] == "failed"
    assert pandera_check["diagnostics"]["contract"]["positive_numeric_fields"] == [
        "quantity_ordered"
    ]


def _record(
    basket_id: str, shop: str, product_name: str, quantity_ordered: float
) -> dict[str, Any]:
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


def _check(report: dict[str, Any], check_id: str) -> dict[str, Any]:
    checks = report["checks"]
    return next(check for check in checks if check["id"] == check_id)

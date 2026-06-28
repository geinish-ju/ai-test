from __future__ import annotations

from collections.abc import Callable

from ai_testing.input_data_testing import (
    InputDataFold,
    InputDataTestConfig,
)
from ai_testing.input_data_testing import (
    test_input_data as run_input_data_test,
)

Record = dict[str, object]


def test_input_data_testing_reports_numeric_contract_diagnostics(
    grocery_record_factory: Callable[..., Record],
    input_data_test_config: InputDataTestConfig,
    check_by_id: Callable[[Record, str], Record],
) -> None:
    records = [
        grocery_record_factory("basket-1", "kosik", "Milk", 1.0),
        grocery_record_factory("basket-2", "rohlik", "Bread", 2.0),
        grocery_record_factory("basket-3", "kosik", "Cheese", 1.0),
        grocery_record_factory("basket-4", "rohlik", "Cancelled item", 0.0),
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
        config=input_data_test_config,
    ).report

    numeric_check = check_by_id(report, "input_data.numeric_values")
    pandera_check = check_by_id(report, "input_data.pandera_contract")

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

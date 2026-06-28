from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from ai_testing.cli.commands.data_acquisition import _fetch_kosik, _fetch_rohlik, _write_records
from ai_testing.cli.commands.data_preparation import (
    _build_classification_dataset,
    _preprocess_data,
    _split_datasets,
)
from ai_testing.cli.commands.model_lifecycle import (
    _test_associations,
    _test_category_classifier,
    _train_associations,
    _train_category_classifier,
    _validate_associations,
    _validate_category_classifier,
)
from ai_testing.cli.commands.quality import (
    _run_quality_gates,
    _test_category_ml_model,
    _test_input_data,
    _test_ml_model,
)
from ai_testing.cli.commands.reporting import (
    create_llm_exploratory_plan_command,
    explain_category_classifier_command,
    generate_markdown_reports_command,
    generate_run_report_command,
    run_history_command,
    test_drift_command,
    track_run_command,
)
from ai_testing.cli.parser import build_parser
from ai_testing.cli_common import (
    extract_config_path,
    load_config_for_cli,
    record_stage_result,
)
from ai_testing.sample_data import sample_grocery_order_item_records


def main(argv: Sequence[str] | None = None) -> int:
    config_path, parser_argv = extract_config_path(argv)
    app_config = load_config_for_cli(config_path)
    parser = build_parser(config_path=config_path, app_config=app_config)
    args = parser.parse_args(parser_argv)
    if args.command == "export-kosik":
        records = _fetch_kosik(args)
        output_path = Path(args.output)
        _write_records(records, output_path)
        record_stage_result(
            args,
            stage_name="data_acquisition.kosik",
            artifacts={"output": output_path},
            payload={"summary": {"record_count": len(records)}},
        )
        return 0

    if args.command == "export-rohlik":
        records = _fetch_rohlik(args)
        output_path = Path(args.output)
        _write_records(records, output_path)
        record_stage_result(
            args,
            stage_name="data_acquisition.rohlik",
            artifacts={"output": output_path},
            payload={"summary": {"record_count": len(records)}},
        )
        return 0

    if args.command == "export-all":
        records = [
            *_fetch_kosik(
                argparse.Namespace(
                    base_url=args.kosik_base_url,
                    cookies=None,
                    cookies_file=args.kosik_cookies_file,
                    cookies_env=args.kosik_cookies_env,
                    include_raw=args.include_raw,
                    product_enrichment=args.product_enrichment,
                    order_page_limit=args.kosik_order_page_limit,
                    include_archived_orders=args.kosik_include_archived_orders,
                    order_list_path=args.kosik_order_list_path,
                    order_detail_path_template=args.kosik_order_detail_path_template,
                    product_by_slug_path_template=args.kosik_product_by_slug_path_template,
                )
            ),
            *_fetch_rohlik(
                argparse.Namespace(
                    base_url=args.rohlik_base_url,
                    cookies=None,
                    cookies_file=args.rohlik_cookies_file,
                    cookies_env=args.rohlik_cookies_env,
                    include_raw=args.include_raw,
                    include_product_content=args.include_product_content,
                    product_enrichment=args.product_enrichment,
                    order_page_limit=args.rohlik_order_page_limit,
                    include_archived_orders=args.rohlik_include_archived_orders,
                    delivered_orders_path=args.rohlik_delivered_orders_path,
                    order_detail_path_template=args.rohlik_order_detail_path_template,
                    product_card_path_template=args.rohlik_product_card_path_template,
                    product_detail_path_template=args.rohlik_product_detail_path_template,
                    product_detail_content_path_template=(
                        args.rohlik_product_detail_content_path_template
                    ),
                )
            ),
        ]
        output_path = Path(args.output)
        _write_records(records, output_path)
        record_stage_result(
            args,
            stage_name="data_acquisition.all",
            artifacts={"output": output_path},
            payload={"summary": {"record_count": len(records)}},
        )
        return 0

    if args.command == "export-sample":
        records = sample_grocery_order_item_records()
        output_path = Path(args.output)
        _write_records(records, output_path)
        record_stage_result(
            args,
            stage_name="data_acquisition.sample",
            artifacts={"output": output_path},
            payload={"summary": {"record_count": len(records)}},
        )
        return 0

    if args.command == "preprocess-data":
        _preprocess_data(args)
        return 0

    if args.command == "split-datasets":
        _split_datasets(args)
        return 0

    if args.command == "build-classification-dataset":
        _build_classification_dataset(args)
        return 0

    if args.command == "train-category-classifier":
        _train_category_classifier(args)
        return 0

    if args.command == "validate-category-classifier":
        _validate_category_classifier(args)
        return 0

    if args.command == "test-category-classifier":
        _test_category_classifier(args)
        return 0

    if args.command == "train-associations":
        _train_associations(args)
        return 0

    if args.command == "validate-associations":
        _validate_associations(args)
        return 0

    if args.command == "test-associations":
        _test_associations(args)
        return 0

    if args.command == "test-input-data":
        _test_input_data(args)
        return 0

    if args.command == "test-ml-model":
        _test_ml_model(args)
        return 0

    if args.command == "test-category-ml-model":
        _test_category_ml_model(args)
        return 0

    if args.command == "run-quality-gates":
        _run_quality_gates(args)
        return 0

    if args.command == "track-run":
        track_run_command(args)
        return 0

    if args.command == "run-history":
        run_history_command(args)
        return 0

    if args.command == "test-drift":
        test_drift_command(args)
        return 0

    if args.command == "generate-run-report":
        generate_run_report_command(args)
        return 0

    if args.command == "generate-markdown-reports":
        generate_markdown_reports_command(args)
        return 0

    if args.command == "explain-category-classifier":
        explain_category_classifier_command(args)
        return 0

    if args.command == "create-llm-exploratory-plan":
        create_llm_exploratory_plan_command(args)
        return 0

    return 2


if __name__ == "__main__":
    raise SystemExit(main())

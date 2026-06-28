from __future__ import annotations

import argparse
from collections.abc import Mapping
from pathlib import Path

from ai_testing.config import (
    config_bool,
    config_float,
    config_int,
    config_section,
    config_str,
    config_str_tuple,
)
from ai_testing.data_preprocessing import (
    DEFAULT_EXACT_TIME_FIELDS,
    DEFAULT_IDENTIFIER_FIELDS,
    DEFAULT_OUTPUT_FIELDS,
)
from ai_testing.drift_testing import RunDriftTestConfig


def build_parser(
    *,
    config_path: str | None,
    app_config: Mapping[str, object],
) -> argparse.ArgumentParser:
    acquisition_config = config_section(app_config, "data_acquisition")
    kosik_config = config_section(acquisition_config, "kosik")
    rohlik_config = config_section(acquisition_config, "rohlik")
    preprocessing_config = config_section(app_config, "data_preprocessing")
    classification_preprocessing_config = config_section(
        app_config,
        "classification_preprocessing",
        "category_classifier",
    )
    splitting_config = config_section(app_config, "data_splitting")
    model_training_config = config_section(app_config, "model_training")
    association_config = config_section(model_training_config, "association_rules")
    category_classifier_config = config_section(model_training_config, "category_classifier")
    model_validation_config = config_section(app_config, "model_validation")
    association_validation_config = config_section(
        model_validation_config,
        "association_rules",
    )
    category_classifier_validation_config = config_section(
        model_validation_config,
        "category_classifier",
    )
    model_testing_config = config_section(app_config, "model_testing")
    association_testing_config = config_section(
        model_testing_config,
        "association_rules",
    )
    category_classifier_testing_config = config_section(
        model_testing_config,
        "category_classifier",
    )
    input_data_testing_config = config_section(app_config, "input_data_testing")
    ml_model_testing_config = config_section(app_config, "ml_model_testing")
    association_ml_model_testing_config = config_section(
        ml_model_testing_config,
        "association_rules",
    )
    category_classifier_ml_model_testing_config = config_section(
        ml_model_testing_config,
        "category_classifier",
    )
    project_quality_config = config_section(app_config, "project_quality")
    run_tracking_config = config_section(app_config, "run_tracking")
    mlops_config = config_section(app_config, "mlops")
    mlflow_config = config_section(mlops_config, "mlflow")
    dvc_config = config_section(mlops_config, "dvc")
    drift_testing_config = config_section(app_config, "drift_testing")
    reporting_config = config_section(app_config, "reporting", "run_markdown")
    markdown_reports_config = config_section(app_config, "reporting", "markdown_reports")
    explainability_config = config_section(app_config, "explainability", "category_classifier")
    llm_exploratory_config = config_section(app_config, "llm_exploratory_testing")
    default_include_raw = config_bool(acquisition_config, "include_raw", False)
    default_product_enrichment = config_bool(acquisition_config, "product_enrichment", True)
    kosik_product_enrichment = config_bool(
        kosik_config,
        "product_enrichment",
        default_product_enrichment,
    )
    rohlik_product_enrichment = config_bool(
        rohlik_config,
        "product_enrichment",
        default_product_enrichment,
    )

    parser = argparse.ArgumentParser(prog="ai-test")
    parser.add_argument(
        "--config",
        default=config_path,
        help=(
            "Path to JSON or JSONC config file. "
            "Defaults to AI_TEST_CONFIG or config/defaults.jsonc."
        ),
    )
    parser.add_argument(
        "--stage-tracking",
        action=argparse.BooleanOptionalAction,
        default=config_bool(run_tracking_config, "stage_tracking_enabled", True),
        dest="stage_tracking_enabled",
        help="Record this command in the per-stage execution history",
    )
    parser.add_argument(
        "--stage-history-output",
        default=config_str(
            run_tracking_config,
            "stage_history_output",
            "data/runs/stage_history.json",
        ),
        help="Path to the per-stage execution history JSON",
    )
    parser.add_argument(
        "--stage-history-markdown-output",
        default=config_str(
            run_tracking_config,
            "stage_history_markdown_output",
            "data/reports/stage_history.md",
        ),
        help="Path to the per-stage execution history Markdown report",
    )
    parser.add_argument(
        "--stage-metric-history-markdown-output",
        default=config_str(
            run_tracking_config,
            "stage_metric_history_markdown_output",
            "data/reports/stage_metric_history.md",
        ),
        help="Path to the per-stage metric history matrix Markdown report",
    )
    parser.add_argument(
        "--stage-history-max-entries",
        type=int,
        default=config_int(run_tracking_config, "stage_history_max_entries", 500),
        help="Maximum number of stage executions kept in history",
    )
    parser.add_argument(
        "--stage-history-recent-entries-per-stage",
        type=int,
        default=config_int(
            run_tracking_config,
            "stage_history_recent_entries_per_stage",
            5,
        ),
        help="Recent executions per stage shown in the Markdown history report",
    )
    parser.add_argument(
        "--stage-metric-history-recent-entries-per-stage",
        type=int,
        default=config_int(
            run_tracking_config,
            "stage_metric_history_recent_entries_per_stage",
            10,
        ),
        help="Recent executions per stage shown in the metric history matrix report",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    kosik_parser = subparsers.add_parser("export-kosik", help="Export Kosik order items")
    _add_common_export_args(
        kosik_parser,
        default_output=config_str(kosik_config, "output", "data/raw/kosik_order_items.json"),
        default_env=config_str(kosik_config, "cookies_env", "KOSIK_COOKIES"),
        default_cookies_file=config_str(
            kosik_config,
            "cookies_file",
            "secrets/kosik.cookies.txt",
        ),
        default_base_url=config_str(kosik_config, "base_url", "https://www.kosik.cz"),
        default_include_raw=default_include_raw,
        default_product_enrichment=kosik_product_enrichment,
    )
    _add_kosik_endpoint_args(kosik_parser, kosik_config)
    kosik_parser.add_argument(
        "--order-page-limit",
        type=int,
        default=config_int(kosik_config, "order_page_limit", 50),
        help="Kosik order-list page size",
    )
    kosik_include_archived_default = config_bool(
        kosik_config,
        "include_archived_orders",
        True,
    )
    kosik_parser.add_argument(
        "--include-archived-orders",
        action=argparse.BooleanOptionalAction,
        default=kosik_include_archived_default,
        help="Request archived Kosik orders",
    )
    kosik_parser.add_argument(
        "--hide-archived-orders",
        dest="include_archived_orders",
        action="store_false",
        help=argparse.SUPPRESS,
    )

    rohlik_parser = subparsers.add_parser("export-rohlik", help="Export Rohlik order items")
    _add_common_export_args(
        rohlik_parser,
        default_output=config_str(rohlik_config, "output", "data/raw/rohlik_order_items.json"),
        default_env=config_str(rohlik_config, "cookies_env", "ROHLIK_COOKIES"),
        default_cookies_file=config_str(
            rohlik_config,
            "cookies_file",
            "secrets/rohlik.cookies.txt",
        ),
        default_base_url=config_str(rohlik_config, "base_url", "https://www.rohlik.cz"),
        default_include_raw=default_include_raw,
        default_product_enrichment=rohlik_product_enrichment,
    )
    _add_rohlik_endpoint_args(rohlik_parser, rohlik_config)
    rohlik_parser.add_argument(
        "--include-product-content",
        action=argparse.BooleanOptionalAction,
        default=config_bool(rohlik_config, "include_product_content", False),
        help="Also fetch Rohlik product detail/content for descriptions and ingredients",
    )
    rohlik_parser.add_argument(
        "--order-page-limit",
        type=int,
        default=config_int(rohlik_config, "order_page_limit", 10),
        help="Rohlik delivered-orders page size",
    )
    rohlik_include_archived_default = config_bool(
        rohlik_config,
        "include_archived_orders",
        True,
    )
    rohlik_parser.add_argument(
        "--include-archived-orders",
        action=argparse.BooleanOptionalAction,
        default=rohlik_include_archived_default,
        help="Request archived Rohlik orders",
    )
    rohlik_parser.add_argument(
        "--hide-archived-orders",
        dest="include_archived_orders",
        action="store_false",
        help=argparse.SUPPRESS,
    )

    all_parser = subparsers.add_parser("export-all", help="Export Kosik and Rohlik order items")
    all_parser.add_argument(
        "--output",
        default=config_str(
            acquisition_config,
            "combined_output",
            "data/raw/grocery_order_items.json",
        ),
        help="Path to write combined normalized order-item records as JSON",
    )
    all_parser.add_argument(
        "--kosik-cookies-env",
        default=config_str(kosik_config, "cookies_env", "KOSIK_COOKIES"),
    )
    all_parser.add_argument(
        "--rohlik-cookies-env",
        default=config_str(rohlik_config, "cookies_env", "ROHLIK_COOKIES"),
    )
    all_parser.add_argument(
        "--kosik-cookies-file",
        default=config_str(kosik_config, "cookies_file", "secrets/kosik.cookies.txt"),
    )
    all_parser.add_argument(
        "--rohlik-cookies-file",
        default=config_str(rohlik_config, "cookies_file", "secrets/rohlik.cookies.txt"),
    )
    all_parser.add_argument(
        "--kosik-base-url",
        default=config_str(kosik_config, "base_url", "https://www.kosik.cz"),
    )
    all_parser.add_argument(
        "--rohlik-base-url",
        default=config_str(rohlik_config, "base_url", "https://www.rohlik.cz"),
    )
    _add_kosik_endpoint_args(all_parser, kosik_config, option_prefix="kosik-", dest_prefix="kosik_")
    _add_rohlik_endpoint_args(
        all_parser,
        rohlik_config,
        option_prefix="rohlik-",
        dest_prefix="rohlik_",
    )
    all_parser.add_argument(
        "--kosik-order-page-limit",
        type=int,
        default=config_int(kosik_config, "order_page_limit", 50),
        help="Kosik order-list page size",
    )
    all_parser.add_argument(
        "--kosik-include-archived-orders",
        action=argparse.BooleanOptionalAction,
        default=kosik_include_archived_default,
        help="Request archived Kosik orders",
    )
    all_parser.add_argument(
        "--hide-kosik-archived-orders",
        dest="kosik_include_archived_orders",
        action="store_false",
        help=argparse.SUPPRESS,
    )
    all_parser.add_argument(
        "--rohlik-order-page-limit",
        type=int,
        default=config_int(rohlik_config, "order_page_limit", 10),
        help="Rohlik delivered-orders page size",
    )
    all_parser.add_argument(
        "--rohlik-include-archived-orders",
        action=argparse.BooleanOptionalAction,
        default=rohlik_include_archived_default,
        help="Request archived Rohlik orders",
    )
    all_parser.add_argument(
        "--hide-rohlik-archived-orders",
        dest="rohlik_include_archived_orders",
        action="store_false",
        help=argparse.SUPPRESS,
    )
    all_parser.add_argument(
        "--include-raw",
        action=argparse.BooleanOptionalAction,
        default=default_include_raw,
    )
    all_parser.add_argument(
        "--product-enrichment",
        action=argparse.BooleanOptionalAction,
        default=default_product_enrichment,
        help="Fetch product metadata endpoints",
    )
    all_parser.add_argument(
        "--skip-product-enrichment",
        dest="product_enrichment",
        action="store_false",
        help=argparse.SUPPRESS,
    )
    all_parser.add_argument(
        "--include-product-content",
        action=argparse.BooleanOptionalAction,
        default=config_bool(rohlik_config, "include_product_content", False),
        help="Also fetch Rohlik product detail/content for descriptions and ingredients",
    )

    sample_parser = subparsers.add_parser(
        "export-sample",
        help="Write sample normalized grocery order-item records",
    )
    sample_parser.add_argument(
        "--output",
        default=config_str(acquisition_config, "sample_output", "data/raw/sample_order_items.json"),
        help="Path to write sample normalized order-item records as JSON",
    )

    preprocessing_parser = subparsers.add_parser(
        "preprocess-data",
        help="Run Data preprocessing for acquired grocery records",
    )
    preprocessing_parser.add_argument(
        "--input",
        default=config_str(preprocessing_config, "input", "data/raw/grocery_order_items.json"),
        help="Path to acquired raw order-item JSON records",
    )
    preprocessing_parser.add_argument(
        "--output",
        default=config_str(
            preprocessing_config,
            "output",
            "data/processed/grocery_order_items.json",
        ),
        help="Path to write preprocessed order-item JSON records",
    )
    preprocessing_parser.add_argument(
        "--report-output",
        default=config_str(
            preprocessing_config,
            "report_output",
            "data/processed/grocery_data_quality_report.json",
        ),
        help="Path to write Data preprocessing quality report as JSON",
    )
    preprocessing_parser.add_argument(
        "--drop-exact-duplicates",
        action=argparse.BooleanOptionalAction,
        default=config_bool(preprocessing_config, "drop_exact_duplicates", False),
        help="Drop exact duplicate rows after identifier removal",
    )
    preprocessing_parser.add_argument(
        "--basket-id-prefix",
        default=config_str(preprocessing_config, "basket_id_prefix", "basket"),
        help="Prefix for synthetic basket ids",
    )
    preprocessing_parser.set_defaults(
        identifier_fields=config_str_tuple(
            preprocessing_config,
            "identifier_fields",
            DEFAULT_IDENTIFIER_FIELDS,
        ),
        exact_time_fields=config_str_tuple(
            preprocessing_config,
            "exact_time_fields",
            DEFAULT_EXACT_TIME_FIELDS,
        ),
        output_fields=config_str_tuple(
            preprocessing_config,
            "output_fields",
            DEFAULT_OUTPUT_FIELDS,
        ),
    )

    splitting_parser = subparsers.add_parser(
        "split-datasets",
        help="Create train, validation, and test datasets",
    )
    splitting_parser.add_argument(
        "--input",
        default=config_str(
            splitting_config,
            "input",
            "data/processed/grocery_order_items.json",
        ),
        help="Path to preprocessed grocery records",
    )
    splitting_parser.add_argument(
        "--output-dir",
        default=config_str(
            splitting_config,
            "output_dir",
            "data/splits",
        ),
        help="Directory to write train/test and k-fold validation datasets",
    )
    splitting_parser.add_argument(
        "--group-field",
        default=config_str(splitting_config, "group_field", "basket_id"),
        help="Field used to keep related rows in the same split",
    )
    splitting_parser.add_argument(
        "--stratify-field",
        default=config_str(splitting_config, "stratify_field", "shop"),
        help="Field used to balance group distribution across splits",
    )
    splitting_parser.add_argument(
        "--test-size",
        type=float,
        default=config_float(splitting_config, "test_size", 0.2),
        help="Hold-out test share",
    )
    splitting_parser.add_argument(
        "--n-splits",
        type=int,
        default=config_int(splitting_config, "n_splits", 5),
        help="Number of k-fold validation splits",
    )
    splitting_parser.add_argument(
        "--random-seed",
        type=int,
        default=config_int(splitting_config, "random_seed", 42),
        help="Deterministic split seed",
    )

    classification_dataset_parser = subparsers.add_parser(
        "build-classification-dataset",
        help="Build supervised text classification datasets from split grocery records",
    )
    classification_dataset_parser.add_argument(
        "--processed-input",
        default=config_str(
            classification_preprocessing_config,
            "processed_input",
            "data/processed/grocery_order_items.json",
        ),
        help="Path to preprocessed grocery records used to determine allowed labels",
    )
    classification_dataset_parser.add_argument(
        "--train-validation-input",
        default=config_str(
            classification_preprocessing_config,
            "train_validation_input",
            "data/splits/train_validation.json",
        ),
        help="Path to train-validation grocery records",
    )
    classification_dataset_parser.add_argument(
        "--test-input",
        default=config_str(
            classification_preprocessing_config,
            "test_input",
            "data/splits/test.json",
        ),
        help="Path to hold-out test grocery records",
    )
    classification_dataset_parser.add_argument(
        "--folds-dir",
        default=config_str(
            classification_preprocessing_config,
            "folds_dir",
            "data/splits/folds",
        ),
        help="Directory containing grocery fold files",
    )
    classification_dataset_parser.add_argument(
        "--output-dir",
        default=config_str(
            classification_preprocessing_config,
            "output_dir",
            "data/classification/category",
        ),
        help="Directory to write classification datasets",
    )
    classification_dataset_parser.add_argument(
        "--target-field",
        default=config_str(classification_preprocessing_config, "target_field", "main_category"),
        help="Label field for supervised classification",
    )
    classification_dataset_parser.add_argument(
        "--min-label-count",
        type=int,
        default=config_int(classification_preprocessing_config, "min_label_count", 20),
        help="Minimum global label count kept in classification datasets",
    )
    classification_dataset_parser.set_defaults(
        text_fields=config_str_tuple(
            classification_preprocessing_config,
            "text_fields",
            ("product_name", "brand"),
        ),
        metadata_fields=config_str_tuple(
            classification_preprocessing_config,
            "metadata_fields",
            ("basket_id", "shop", "order_date"),
        ),
    )

    category_classifier_parser = subparsers.add_parser(
        "train-category-classifier",
        help="Train a supervised product category text classifier",
    )
    category_classifier_parser.add_argument(
        "--input",
        default=config_str(
            category_classifier_config,
            "input",
            "data/classification/category/train_validation.json",
        ),
        help="Path to classification training records",
    )
    category_classifier_parser.add_argument(
        "--output",
        default=config_str(
            category_classifier_config,
            "output",
            "data/models/category_classifier.json",
        ),
        help="Path to write category classifier model JSON",
    )
    category_classifier_parser.add_argument(
        "--estimator-output",
        default=config_str(
            category_classifier_config,
            "estimator_output",
            "data/models/category_classifier.joblib",
        ),
        help="Path to write category classifier scikit-learn joblib artifact",
    )
    category_classifier_parser.add_argument(
        "--text-field",
        default=config_str(category_classifier_config, "text_field", "text"),
        help="Text feature field",
    )
    category_classifier_parser.add_argument(
        "--label-field",
        default=config_str(category_classifier_config, "label_field", "label"),
        help="Supervised label field",
    )
    category_classifier_parser.add_argument(
        "--alpha",
        type=float,
        default=config_float(category_classifier_config, "alpha", 1.0),
        help="Naive Bayes Laplace smoothing parameter",
    )
    category_classifier_parser.add_argument(
        "--min-token-length",
        type=int,
        default=config_int(category_classifier_config, "min_token_length", 2),
        help="Minimum token length",
    )
    category_classifier_parser.add_argument(
        "--max-vocabulary-size",
        type=int,
        default=config_int(category_classifier_config, "max_vocabulary_size", 5000),
        help="Maximum number of training tokens kept in the vocabulary",
    )

    category_classifier_validation_parser = subparsers.add_parser(
        "validate-category-classifier",
        help="Validate the supervised category classifier across k-fold datasets",
    )
    category_classifier_validation_parser.add_argument(
        "--folds-dir",
        default=config_str(
            category_classifier_validation_config,
            "folds_dir",
            "data/classification/category/folds",
        ),
        help="Directory containing classification fold files",
    )
    category_classifier_validation_parser.add_argument(
        "--output",
        default=config_str(
            category_classifier_validation_config,
            "output",
            "data/validation/category_classifier_validation_report.json",
        ),
        help="Path to write category classifier validation report JSON",
    )
    _add_text_classifier_args(
        category_classifier_validation_parser, category_classifier_validation_config
    )

    category_classifier_test_parser = subparsers.add_parser(
        "test-category-classifier",
        help="Test the trained supervised category classifier on the hold-out test dataset",
    )
    category_classifier_test_parser.add_argument(
        "--model-input",
        default=config_str(
            category_classifier_testing_config,
            "model_input",
            "data/models/category_classifier.json",
        ),
        help="Path to trained category classifier model JSON",
    )
    category_classifier_test_parser.add_argument(
        "--test-input",
        default=config_str(
            category_classifier_testing_config,
            "test_input",
            "data/classification/category/test.json",
        ),
        help="Path to classification hold-out test records",
    )
    category_classifier_test_parser.add_argument(
        "--output",
        default=config_str(
            category_classifier_testing_config,
            "output",
            "data/testing/category_classifier_test_report.json",
        ),
        help="Path to write category classifier final test report JSON",
    )
    category_classifier_test_parser.add_argument(
        "--text-field",
        default=config_str(category_classifier_testing_config, "text_field", "text"),
        help="Text feature field",
    )
    category_classifier_test_parser.add_argument(
        "--label-field",
        default=config_str(category_classifier_testing_config, "label_field", "label"),
        help="Supervised label field",
    )
    category_classifier_test_parser.add_argument(
        "--top-confusions",
        type=int,
        default=config_int(category_classifier_testing_config, "top_confusions", 20),
        help="Number of largest classification confusions to store",
    )

    association_parser = subparsers.add_parser(
        "train-associations",
        help="Train association rules on the training dataset",
    )
    association_parser.add_argument(
        "--input",
        default=config_str(
            association_config,
            "input",
            "data/splits/train_validation.json",
        ),
        help="Path to training grocery records",
    )
    association_parser.add_argument(
        "--output",
        default=config_str(
            association_config,
            "output",
            "data/models/association_rules.json",
        ),
        help="Path to write association rules model JSON",
    )
    association_parser.add_argument(
        "--basket-field",
        default=config_str(association_config, "basket_field", "basket_id"),
        help="Field used as the market basket identifier",
    )
    association_parser.add_argument(
        "--item-field",
        default=config_str(association_config, "item_field", "product_group"),
        help="Field used as the item in association rules",
    )
    association_parser.add_argument(
        "--min-support",
        type=float,
        default=config_float(association_config, "min_support", 0.02),
        help="Minimum itemset support ratio in training baskets",
    )
    association_parser.add_argument(
        "--min-confidence",
        type=float,
        default=config_float(association_config, "min_confidence", 0.2),
        help="Minimum rule confidence",
    )
    association_parser.add_argument(
        "--min-lift",
        type=float,
        default=config_float(association_config, "min_lift", 1.05),
        help="Minimum rule lift",
    )
    association_parser.add_argument(
        "--max-itemset-size",
        type=int,
        default=config_int(association_config, "max_itemset_size", 2),
        help="Maximum frequent itemset size",
    )
    association_parser.add_argument(
        "--max-rules",
        type=int,
        default=config_int(association_config, "max_rules", 100),
        help="Maximum number of rules to export",
    )
    association_parser.add_argument(
        "--max-itemsets",
        type=int,
        default=config_int(association_config, "max_itemsets", 500),
        help="Maximum number of frequent itemsets to export",
    )

    association_validation_parser = subparsers.add_parser(
        "validate-associations",
        help="Validate association rules across k-fold validation datasets",
    )
    association_validation_parser.add_argument(
        "--folds-dir",
        default=config_str(
            association_validation_config,
            "folds_dir",
            "data/splits/folds",
        ),
        help="Directory containing fold_NN_train and fold_NN_validation files",
    )
    association_validation_parser.add_argument(
        "--output",
        default=config_str(
            association_validation_config,
            "output",
            "data/validation/association_validation_report.json",
        ),
        help="Path to write association validation report JSON",
    )
    association_validation_parser.add_argument(
        "--basket-field",
        default=config_str(association_validation_config, "basket_field", "basket_id"),
        help="Field used as the market basket identifier",
    )
    association_validation_parser.add_argument(
        "--item-field",
        default=config_str(association_validation_config, "item_field", "product_group"),
        help="Field used as the item in association rules",
    )
    association_validation_parser.add_argument(
        "--min-support",
        type=float,
        default=config_float(association_validation_config, "min_support", 0.02),
        help="Minimum itemset support ratio in fold training baskets",
    )
    association_validation_parser.add_argument(
        "--min-confidence",
        type=float,
        default=config_float(association_validation_config, "min_confidence", 0.2),
        help="Minimum rule confidence",
    )
    association_validation_parser.add_argument(
        "--min-lift",
        type=float,
        default=config_float(association_validation_config, "min_lift", 1.05),
        help="Minimum rule lift",
    )
    association_validation_parser.add_argument(
        "--max-itemset-size",
        type=int,
        default=config_int(association_validation_config, "max_itemset_size", 2),
        help="Maximum frequent itemset size",
    )
    association_validation_parser.add_argument(
        "--max-rules",
        type=int,
        default=config_int(association_validation_config, "max_rules", 100),
        help="Maximum number of rules to evaluate per fold",
    )
    association_validation_parser.add_argument(
        "--max-itemsets",
        type=int,
        default=config_int(association_validation_config, "max_itemsets", 500),
        help="Maximum number of frequent itemsets to keep per fold",
    )
    association_validation_parser.add_argument(
        "--top-rules",
        type=int,
        default=config_int(association_validation_config, "top_rules", 20),
        help="Number of strongest validation rules to store per fold",
    )

    association_test_parser = subparsers.add_parser(
        "test-associations",
        help="Test the trained association rules model on the hold-out test dataset",
    )
    association_test_parser.add_argument(
        "--model-input",
        default=config_str(
            association_testing_config,
            "model_input",
            "data/models/association_rules.json",
        ),
        help="Path to a trained association rules model JSON",
    )
    association_test_parser.add_argument(
        "--test-input",
        default=config_str(
            association_testing_config,
            "test_input",
            "data/splits/test.json",
        ),
        help="Path to hold-out test grocery records",
    )
    association_test_parser.add_argument(
        "--output",
        default=config_str(
            association_testing_config,
            "output",
            "data/testing/association_test_report.json",
        ),
        help="Path to write final association test report JSON",
    )
    association_test_parser.add_argument(
        "--basket-field",
        default=config_str(association_testing_config, "basket_field", "basket_id"),
        help="Field used as the market basket identifier",
    )
    association_test_parser.add_argument(
        "--item-field",
        default=config_str(association_testing_config, "item_field", "product_group"),
        help="Field used as the item in association rules",
    )
    association_test_parser.add_argument(
        "--min-confidence",
        type=float,
        default=config_float(association_testing_config, "min_confidence", 0.2),
        help="Minimum test confidence used to count stable rules",
    )
    association_test_parser.add_argument(
        "--min-lift",
        type=float,
        default=config_float(association_testing_config, "min_lift", 1.05),
        help="Minimum test lift used to count stable rules",
    )
    association_test_parser.add_argument(
        "--top-rules",
        type=int,
        default=config_int(association_testing_config, "top_rules", 20),
        help="Number of strongest test rules to store",
    )

    input_data_test_parser = subparsers.add_parser(
        "test-input-data",
        help="Test input data quality and train/validation/test split integrity",
    )
    input_data_test_parser.add_argument(
        "--processed-input",
        default=config_str(
            input_data_testing_config,
            "processed_input",
            "data/processed/grocery_order_items.json",
        ),
        help="Path to preprocessed grocery records",
    )
    input_data_test_parser.add_argument(
        "--train-validation-input",
        default=config_str(
            input_data_testing_config,
            "train_validation_input",
            "data/splits/train_validation.json",
        ),
        help="Path to train-validation records",
    )
    input_data_test_parser.add_argument(
        "--test-input",
        default=config_str(input_data_testing_config, "test_input", "data/splits/test.json"),
        help="Path to hold-out test records",
    )
    input_data_test_parser.add_argument(
        "--folds-dir",
        default=config_str(input_data_testing_config, "folds_dir", "data/splits/folds"),
        help="Directory containing fold_NN_train and fold_NN_validation files",
    )
    input_data_test_parser.add_argument(
        "--output",
        default=config_str(
            input_data_testing_config,
            "output",
            "data/testing/input_data_test_report.json",
        ),
        help="Path to write input data test report JSON",
    )
    input_data_test_parser.add_argument(
        "--group-field",
        default=config_str(input_data_testing_config, "group_field", "basket_id"),
        help="Field used to keep related rows in the same split",
    )
    input_data_test_parser.add_argument(
        "--stratify-field",
        default=config_str(input_data_testing_config, "stratify_field", "shop"),
        help="Field used to compare train/test distributions",
    )
    input_data_test_parser.add_argument(
        "--max-coverage-missing-rate",
        type=float,
        default=config_float(input_data_testing_config, "max_coverage_missing_rate", 0.4),
        help="Maximum missing rate for configured coverage fields",
    )
    input_data_test_parser.add_argument(
        "--max-split-distribution-delta",
        type=float,
        default=config_float(input_data_testing_config, "max_split_distribution_delta", 0.2),
        help="Maximum absolute distribution delta between train-validation and test",
    )
    input_data_test_parser.set_defaults(
        required_fields=config_str_tuple(
            input_data_testing_config,
            "required_fields",
            DEFAULT_OUTPUT_FIELDS,
        ),
        protected_fields=config_str_tuple(
            input_data_testing_config,
            "protected_fields",
            (
                *DEFAULT_IDENTIFIER_FIELDS,
                *DEFAULT_EXACT_TIME_FIELDS,
                "quantity",
                "quantity_delivered",
                "price_total",
                "category_path",
                "product_enriched",
            ),
        ),
        critical_fields=config_str_tuple(
            input_data_testing_config,
            "critical_fields",
            ("shop", "order_date", "product_name", "quantity_ordered", "currency", "basket_id"),
        ),
        coverage_fields=config_str_tuple(
            input_data_testing_config,
            "coverage_fields",
            ("brand", "main_category", "category", "product_group"),
        ),
        expected_shops=config_str_tuple(
            input_data_testing_config,
            "expected_shops",
            ("kosik", "rohlik"),
        ),
        expected_currencies=config_str_tuple(
            input_data_testing_config,
            "expected_currencies",
            ("CZK",),
        ),
    )

    ml_model_test_parser = subparsers.add_parser(
        "test-ml-model",
        help="Test the association model against validation and final test acceptance criteria",
    )
    ml_model_test_parser.add_argument(
        "--model-input",
        default=config_str(
            association_ml_model_testing_config,
            "model_input",
            "data/models/association_rules.json",
        ),
        help="Path to trained association rules model JSON",
    )
    ml_model_test_parser.add_argument(
        "--validation-report-input",
        default=config_str(
            association_ml_model_testing_config,
            "validation_report_input",
            "data/validation/association_validation_report.json",
        ),
        help="Path to k-fold validation report JSON",
    )
    ml_model_test_parser.add_argument(
        "--test-report-input",
        default=config_str(
            association_ml_model_testing_config,
            "test_report_input",
            "data/testing/association_test_report.json",
        ),
        help="Path to final hold-out test report JSON",
    )
    ml_model_test_parser.add_argument(
        "--test-dataset-input",
        default=config_str(
            association_ml_model_testing_config,
            "test_dataset_input",
            "data/splits/test.json",
        ),
        help="Path used to ensure the model was not trained on the hold-out test dataset",
    )
    ml_model_test_parser.add_argument(
        "--output",
        default=config_str(
            association_ml_model_testing_config,
            "output",
            "data/testing/ml_model_test_report.json",
        ),
        help="Path to write ML model test report JSON",
    )
    ml_model_test_parser.add_argument(
        "--min-mean-test-confidence",
        type=float,
        default=config_float(association_ml_model_testing_config, "min_mean_test_confidence", 0.2),
    )
    ml_model_test_parser.add_argument(
        "--min-mean-test-lift",
        type=float,
        default=config_float(association_ml_model_testing_config, "min_mean_test_lift", 1.0),
    )
    ml_model_test_parser.add_argument(
        "--max-mean-abs-test-confidence-gap",
        type=float,
        default=config_float(
            association_ml_model_testing_config,
            "max_mean_abs_test_confidence_gap",
            0.2,
        ),
    )
    ml_model_test_parser.add_argument(
        "--max-validation-test-confidence-delta",
        type=float,
        default=config_float(
            association_ml_model_testing_config,
            "max_validation_test_confidence_delta",
            0.15,
        ),
    )
    ml_model_test_parser.add_argument(
        "--max-validation-test-lift-delta",
        type=float,
        default=config_float(
            association_ml_model_testing_config,
            "max_validation_test_lift_delta",
            0.25,
        ),
    )
    ml_model_test_parser.set_defaults(
        forbidden_feature_fields=config_str_tuple(
            association_ml_model_testing_config,
            "forbidden_feature_fields",
            DEFAULT_IDENTIFIER_FIELDS,
        ),
    )

    category_ml_model_test_parser = subparsers.add_parser(
        "test-category-ml-model",
        help="Test the supervised category classifier against acceptance criteria",
    )
    category_ml_model_test_parser.add_argument(
        "--model-input",
        default=config_str(
            category_classifier_ml_model_testing_config,
            "model_input",
            "data/models/category_classifier.json",
        ),
        help="Path to trained category classifier model JSON",
    )
    category_ml_model_test_parser.add_argument(
        "--validation-report-input",
        default=config_str(
            category_classifier_ml_model_testing_config,
            "validation_report_input",
            "data/validation/category_classifier_validation_report.json",
        ),
        help="Path to k-fold category classifier validation report JSON",
    )
    category_ml_model_test_parser.add_argument(
        "--test-report-input",
        default=config_str(
            category_classifier_ml_model_testing_config,
            "test_report_input",
            "data/testing/category_classifier_test_report.json",
        ),
        help="Path to final category classifier hold-out test report JSON",
    )
    category_ml_model_test_parser.add_argument(
        "--classification-manifest-input",
        default=config_str(
            category_classifier_ml_model_testing_config,
            "classification_manifest_input",
            "data/classification/category/classification_manifest.json",
        ),
        help="Path to classification preprocessing manifest JSON",
    )
    category_ml_model_test_parser.add_argument(
        "--test-dataset-input",
        default=config_str(
            category_classifier_ml_model_testing_config,
            "test_dataset_input",
            "data/classification/category/test.json",
        ),
        help="Path used to ensure the classifier was not trained on the hold-out test dataset",
    )
    category_ml_model_test_parser.add_argument(
        "--output",
        default=config_str(
            category_classifier_ml_model_testing_config,
            "output",
            "data/testing/category_ml_model_test_report.json",
        ),
        help="Path to write category classifier ML model test report JSON",
    )
    category_ml_model_test_parser.add_argument(
        "--min-test-accuracy",
        type=float,
        default=config_float(category_classifier_ml_model_testing_config, "min_test_accuracy", 0.9),
    )
    category_ml_model_test_parser.add_argument(
        "--min-test-macro-precision",
        type=float,
        default=config_float(
            category_classifier_ml_model_testing_config,
            "min_test_macro_precision",
            0.7,
        ),
    )
    category_ml_model_test_parser.add_argument(
        "--min-test-macro-recall",
        type=float,
        default=config_float(
            category_classifier_ml_model_testing_config,
            "min_test_macro_recall",
            0.7,
        ),
    )
    category_ml_model_test_parser.add_argument(
        "--min-test-macro-f1",
        type=float,
        default=config_float(category_classifier_ml_model_testing_config, "min_test_macro_f1", 0.7),
    )
    category_ml_model_test_parser.add_argument(
        "--min-test-weighted-f1",
        type=float,
        default=config_float(
            category_classifier_ml_model_testing_config,
            "min_test_weighted_f1",
            0.9,
        ),
    )
    category_ml_model_test_parser.add_argument(
        "--max-validation-test-accuracy-delta",
        type=float,
        default=config_float(
            category_classifier_ml_model_testing_config,
            "max_validation_test_accuracy_delta",
            0.05,
        ),
    )
    category_ml_model_test_parser.add_argument(
        "--max-validation-test-macro-f1-delta",
        type=float,
        default=config_float(
            category_classifier_ml_model_testing_config,
            "max_validation_test_macro_f1_delta",
            0.08,
        ),
    )
    category_ml_model_test_parser.set_defaults(
        forbidden_feature_fields=config_str_tuple(
            category_classifier_ml_model_testing_config,
            "forbidden_feature_fields",
            (
                *DEFAULT_IDENTIFIER_FIELDS,
                "main_category",
                "category",
                "category_path",
                "product_group",
            ),
        ),
    )

    project_quality_parser = subparsers.add_parser(
        "run-quality-gates",
        help="Aggregate input data and ML model quality reports into one project verdict",
    )
    project_quality_parser.add_argument(
        "--input-data-report",
        default=config_str(
            project_quality_config,
            "input_data_report",
            "data/testing/input_data_test_report.json",
        ),
        help="Path to input data quality report JSON",
    )
    project_quality_parser.add_argument(
        "--association-model-report",
        default=config_str(
            project_quality_config,
            "association_model_report",
            "data/testing/ml_model_test_report.json",
        ),
        help="Path to association ML model quality report JSON",
    )
    project_quality_parser.add_argument(
        "--category-model-report",
        default=config_str(
            project_quality_config,
            "category_model_report",
            "data/testing/category_ml_model_test_report.json",
        ),
        help="Path to category classifier ML model quality report JSON",
    )
    project_quality_parser.add_argument(
        "--output",
        default=config_str(
            project_quality_config,
            "output",
            "data/testing/project_quality_report.json",
        ),
        help="Path to write project quality report JSON",
    )
    project_quality_parser.add_argument(
        "--drift-report",
        default=config_str(project_quality_config, "drift_report", ""),
        help="Optional drift quality report JSON to include in project gates",
    )

    track_run_parser = subparsers.add_parser(
        "track-run",
        help="Record a versioned run report from current datasets, models, and stage reports",
    )
    track_run_parser.add_argument(
        "--run-id",
        default=None,
        help="Optional stable run id. Defaults to a UTC timestamp.",
    )
    track_run_parser.add_argument(
        "--run-name",
        default=None,
        help="Optional human-readable run name.",
    )
    track_run_parser.add_argument(
        "--runs-dir",
        default=config_str(run_tracking_config, "runs_dir", "data/runs"),
        help="Directory where versioned run reports are written",
    )
    track_run_parser.add_argument(
        "--index-output",
        default=config_str(run_tracking_config, "index_output", "data/runs/run_index.json"),
        help="Path to the run history index JSON",
    )
    track_run_parser.add_argument(
        "--output",
        default=None,
        help="Optional explicit run report output path",
    )
    track_run_parser.add_argument(
        "--baseline-run-report",
        default=config_str(run_tracking_config, "baseline_run_report", ""),
        help="Optional baseline run report used for metric deltas and drift checks",
    )
    track_run_parser.add_argument(
        "--compare-to-latest",
        action=argparse.BooleanOptionalAction,
        default=config_bool(run_tracking_config, "compare_to_latest", True),
        help="Compare this run to the latest run in the run index when no baseline is provided",
    )
    track_run_parser.add_argument(
        "--mlflow-tracking",
        action=argparse.BooleanOptionalAction,
        default=config_bool(mlflow_config, "enabled", False),
        help="Log the tracked run to MLflow",
    )
    track_run_parser.add_argument(
        "--mlflow-tracking-uri",
        default=config_str(mlflow_config, "tracking_uri", "sqlite:///data/mlflow/mlflow.db"),
        help="MLflow tracking URI",
    )
    track_run_parser.add_argument(
        "--mlflow-experiment-name",
        default=config_str(mlflow_config, "experiment_name", "ai-grocery-testing"),
        help="MLflow experiment name",
    )
    track_run_parser.add_argument(
        "--mlflow-log-artifacts",
        action=argparse.BooleanOptionalAction,
        default=config_bool(mlflow_config, "log_artifacts", True),
        help="Log selected run artifacts to MLflow",
    )
    track_run_parser.add_argument(
        "--mlflow-fail-on-error",
        action=argparse.BooleanOptionalAction,
        default=config_bool(mlflow_config, "fail_on_error", False),
        help="Fail track-run when MLflow logging fails",
    )
    track_run_parser.add_argument(
        "--dvc-versioning",
        action=argparse.BooleanOptionalAction,
        default=config_bool(dvc_config, "enabled", False),
        help="Run dvc add for selected data/model/report artifacts",
    )
    track_run_parser.add_argument(
        "--dvc-command",
        default=config_str(dvc_config, "command", "dvc"),
        help="DVC CLI command path",
    )
    track_run_parser.add_argument(
        "--dvc-push",
        action=argparse.BooleanOptionalAction,
        default=config_bool(dvc_config, "push", False),
        help="Run dvc push after dvc add",
    )
    track_run_parser.add_argument(
        "--dvc-remote",
        default=config_str(dvc_config, "remote", ""),
        help="Optional DVC remote name used with dvc push",
    )
    track_run_parser.add_argument(
        "--dvc-fail-on-error",
        action=argparse.BooleanOptionalAction,
        default=config_bool(dvc_config, "fail_on_error", False),
        help="Fail track-run when DVC versioning fails",
    )
    track_run_parser.set_defaults(
        stage_report_paths=_run_tracking_stage_report_paths(run_tracking_config),
        artifact_paths=_run_tracking_artifact_paths(run_tracking_config),
        mlflow_artifact_types=config_str_tuple(
            mlflow_config,
            "artifact_types",
            ("report", "manifest", "model"),
        ),
        dvc_artifact_names=config_str_tuple(
            dvc_config,
            "artifact_names",
            (
                "processed_grocery_dataset",
                "split_manifest",
                "train_validation_dataset",
                "test_dataset",
                "classification_manifest",
                "classification_train_validation_dataset",
                "classification_test_dataset",
                "association_model",
                "category_model",
                "category_estimator",
                "input_data_quality_report",
                "association_model_quality_report",
                "category_model_quality_report",
                "project_quality_report",
                "drift_test_report",
                "category_explainability_report",
                "llm_exploratory_test_plan",
                "latest_markdown_run_report",
            ),
        ),
    )

    run_history_parser = subparsers.add_parser(
        "run-history",
        help="Show compact history of recorded pipeline runs",
    )
    run_history_parser.add_argument(
        "--index-output",
        default=config_str(run_tracking_config, "index_output", "data/runs/run_index.json"),
        help="Path to the run history index JSON",
    )
    run_history_parser.add_argument(
        "--limit",
        type=int,
        default=config_int(run_tracking_config, "history_limit", 10),
        help="Maximum number of recent runs to print",
    )

    drift_test_parser = subparsers.add_parser(
        "test-drift",
        help="Test run-to-baseline data drift and metric regressions",
    )
    drift_test_parser.add_argument(
        "--run-report-input",
        default=config_str(
            drift_testing_config,
            "run_report_input",
            "data/runs/run_index.json",
        ),
        help="Path to a run report JSON, or run index JSON when --use-latest-run is enabled",
    )
    drift_test_parser.add_argument(
        "--use-latest-run",
        action=argparse.BooleanOptionalAction,
        default=config_bool(drift_testing_config, "use_latest_run", True),
        help="Resolve --run-report-input as a run index and test its latest run report",
    )
    drift_test_parser.add_argument(
        "--output",
        default=config_str(
            drift_testing_config,
            "output",
            "data/testing/drift_test_report.json",
        ),
        help="Path to write drift quality report JSON",
    )
    drift_test_parser.add_argument(
        "--require-baseline",
        action=argparse.BooleanOptionalAction,
        default=config_bool(drift_testing_config, "require_baseline", True),
        help="Fail when the run report has no baseline comparison",
    )
    drift_test_parser.add_argument(
        "--max-regressed-metric-count",
        type=int,
        default=config_int(drift_testing_config, "max_regressed_metric_count", 0),
        help="Maximum accepted number of regressed metrics",
    )
    drift_test_parser.add_argument(
        "--max-total-variation-distance",
        type=float,
        default=config_float(drift_testing_config, "max_total_variation_distance", 0.1),
        help="Maximum accepted overall total variation distance",
    )
    drift_test_parser.add_argument(
        "--max-distribution-total-variation-distance",
        type=float,
        default=config_float(
            drift_testing_config,
            "max_distribution_total_variation_distance",
            0.15,
        ),
        help="Maximum accepted total variation distance per monitored distribution",
    )
    drift_test_parser.add_argument(
        "--max-critical-metric-regression",
        type=float,
        default=config_float(drift_testing_config, "max_critical_metric_regression", 0.02),
        help="Maximum accepted absolute regression for critical metrics",
    )
    drift_test_parser.set_defaults(
        critical_metrics=config_str_tuple(
            drift_testing_config,
            "critical_metrics",
            RunDriftTestConfig().critical_metrics,
        )
    )

    run_markdown_parser = subparsers.add_parser(
        "generate-run-report",
        help="Generate a human-readable Markdown report for a tracked run",
    )
    run_markdown_parser.add_argument(
        "--run-report-input",
        default=config_str(
            reporting_config,
            "run_report_input",
            "data/runs/run_index.json",
        ),
        help="Path to a run report JSON, or run index JSON when --use-latest-run is enabled",
    )
    run_markdown_parser.add_argument(
        "--use-latest-run",
        action=argparse.BooleanOptionalAction,
        default=config_bool(reporting_config, "use_latest_run", True),
        help="Resolve --run-report-input as a run index and render its latest run report",
    )
    run_markdown_parser.add_argument(
        "--drift-report-input",
        default=config_str(
            reporting_config,
            "drift_report_input",
            "data/testing/drift_test_report.json",
        ),
        help="Optional drift quality report JSON included in the Markdown report",
    )
    run_markdown_parser.add_argument(
        "--output",
        default=config_str(
            reporting_config,
            "output",
            "data/reports/latest_run_report.md",
        ),
        help="Path to write Markdown run report",
    )
    run_markdown_parser.add_argument(
        "--max-metric-deltas",
        type=int,
        default=config_int(reporting_config, "max_metric_deltas", 15),
        help="Maximum number of metric changes to show",
    )
    run_markdown_parser.add_argument(
        "--max-drift-distributions",
        type=int,
        default=config_int(reporting_config, "max_drift_distributions", 5),
        help="Maximum number of drift distributions to show",
    )
    run_markdown_parser.add_argument(
        "--max-artifacts",
        type=int,
        default=config_int(reporting_config, "max_artifacts", 12),
        help="Maximum number of artifact hashes to show",
    )

    markdown_reports_parser = subparsers.add_parser(
        "generate-markdown-reports",
        help="Generate human-readable Markdown files for known JSON reports",
    )
    markdown_reports_parser.add_argument(
        "--output-dir",
        default=config_str(markdown_reports_config, "output_dir", "data/reports"),
        help="Directory where Markdown report files are written",
    )
    markdown_reports_parser.add_argument(
        "--report",
        action="append",
        default=None,
        metavar="NAME=PATH",
        help="Report JSON to render. Can be passed multiple times and overrides config list.",
    )
    markdown_reports_parser.add_argument(
        "--max-checks",
        type=int,
        default=config_int(markdown_reports_config, "max_checks", 100),
        help="Maximum number of checks rendered per report",
    )
    markdown_reports_parser.add_argument(
        "--max-rows",
        type=int,
        default=config_int(markdown_reports_config, "max_rows", 30),
        help="Maximum number of rows rendered for long report tables",
    )
    markdown_reports_parser.add_argument(
        "--include-run-report",
        action=argparse.BooleanOptionalAction,
        default=config_bool(markdown_reports_config, "include_run_report", True),
        help="Also render the latest tracked run report using the specialized run renderer",
    )
    markdown_reports_parser.set_defaults(
        report_paths=_markdown_report_paths(markdown_reports_config),
        run_report_input=config_str(
            markdown_reports_config,
            "run_report_input",
            "data/runs/run_index.json",
        ),
        drift_report_input=config_str(
            markdown_reports_config,
            "drift_report_input",
            "data/testing/drift_test_report.json",
        ),
        use_latest_run=config_bool(markdown_reports_config, "use_latest_run", True),
    )

    explain_classifier_parser = subparsers.add_parser(
        "explain-category-classifier",
        help="Generate a simple explainability report for the supervised category classifier",
    )
    explain_classifier_parser.add_argument(
        "--model-input",
        default=config_str(
            explainability_config,
            "model_input",
            "data/models/category_classifier.json",
        ),
        help="Path to category classifier model JSON",
    )
    explain_classifier_parser.add_argument(
        "--output",
        default=config_str(
            explainability_config,
            "output",
            "data/explainability/category_classifier_explanation_report.json",
        ),
        help="Path to write category classifier explainability report JSON",
    )
    explain_classifier_parser.add_argument(
        "--top-features-per-class",
        type=int,
        default=config_int(explainability_config, "top_features_per_class", 20),
        help="Number of top tokens to report per class",
    )

    llm_exploratory_parser = subparsers.add_parser(
        "create-llm-exploratory-plan",
        help="Create exploratory testing charters for an LLM-based feature",
    )
    llm_exploratory_parser.add_argument(
        "--target-model-name",
        default=config_str(llm_exploratory_config, "target_model_name", "not configured"),
        help="LLM or system under exploratory testing",
    )
    llm_exploratory_parser.add_argument(
        "--domain",
        default=config_str(llm_exploratory_config, "domain", "grocery AI assistant"),
        help="Business domain for exploratory testing charters",
    )
    llm_exploratory_parser.add_argument(
        "--session-duration-minutes",
        type=int,
        default=config_int(llm_exploratory_config, "session_duration_minutes", 45),
        help="Recommended session length for each exploratory run",
    )
    llm_exploratory_parser.add_argument(
        "--output",
        default=config_str(
            llm_exploratory_config,
            "output",
            "data/testing/llm_exploratory_test_plan.json",
        ),
        help="Path to write LLM exploratory testing plan JSON",
    )

    return parser


def _add_common_export_args(
    parser: argparse.ArgumentParser,
    default_output: str,
    default_env: str,
    default_cookies_file: str,
    default_base_url: str,
    default_include_raw: bool,
    default_product_enrichment: bool,
) -> None:
    parser.add_argument(
        "--output",
        default=default_output,
        help="Path to write normalized order-item records as JSON",
    )
    parser.add_argument(
        "--cookies",
        help="Authenticated browser Cookie header value. Prefer --cookies-file for local use.",
    )
    parser.add_argument(
        "--cookies-file",
        default=default_cookies_file,
        help="Local ignored file containing the authenticated browser Cookie header",
    )
    parser.add_argument(
        "--cookies-env",
        default=default_env,
        help="Environment variable containing the authenticated browser Cookie header",
    )
    parser.add_argument("--base-url", default=default_base_url)
    parser.add_argument(
        "--include-raw",
        action=argparse.BooleanOptionalAction,
        default=default_include_raw,
    )
    parser.add_argument(
        "--product-enrichment",
        action=argparse.BooleanOptionalAction,
        default=default_product_enrichment,
        help="Fetch product metadata endpoints",
    )
    parser.add_argument(
        "--skip-product-enrichment",
        dest="product_enrichment",
        action="store_false",
        help=argparse.SUPPRESS,
    )


def _add_text_classifier_args(
    parser: argparse.ArgumentParser,
    config: Mapping[str, object],
) -> None:
    parser.add_argument(
        "--text-field",
        default=config_str(config, "text_field", "text"),
        help="Text feature field",
    )
    parser.add_argument(
        "--label-field",
        default=config_str(config, "label_field", "label"),
        help="Supervised label field",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=config_float(config, "alpha", 1.0),
        help="Naive Bayes Laplace smoothing parameter",
    )
    parser.add_argument(
        "--min-token-length",
        type=int,
        default=config_int(config, "min_token_length", 2),
        help="Minimum token length",
    )
    parser.add_argument(
        "--max-vocabulary-size",
        type=int,
        default=config_int(config, "max_vocabulary_size", 5000),
        help="Maximum number of training tokens kept in the vocabulary",
    )
    parser.add_argument(
        "--top-confusions",
        type=int,
        default=config_int(config, "top_confusions", 20),
        help="Number of largest classification confusions to store",
    )


def _run_tracking_stage_report_paths(config: Mapping[str, object]) -> dict[str, Path]:
    stage_reports = config_section(config, "stage_reports")
    return {
        "data_preprocessing": Path(
            config_str(
                stage_reports,
                "data_preprocessing",
                "data/processed/grocery_data_quality_report.json",
            )
        ),
        "data_splitting": Path(
            config_str(stage_reports, "data_splitting", "data/splits/split_manifest.json")
        ),
        "classification_preprocessing": Path(
            config_str(
                stage_reports,
                "classification_preprocessing",
                "data/classification/category/classification_manifest.json",
            )
        ),
        "association_training": Path(
            config_str(stage_reports, "association_training", "data/models/association_rules.json")
        ),
        "association_validation": Path(
            config_str(
                stage_reports,
                "association_validation",
                "data/validation/association_validation_report.json",
            )
        ),
        "association_testing": Path(
            config_str(
                stage_reports,
                "association_testing",
                "data/testing/association_test_report.json",
            )
        ),
        "association_model_quality": Path(
            config_str(
                stage_reports,
                "association_model_quality",
                "data/testing/ml_model_test_report.json",
            )
        ),
        "category_training": Path(
            config_str(stage_reports, "category_training", "data/models/category_classifier.json")
        ),
        "category_validation": Path(
            config_str(
                stage_reports,
                "category_validation",
                "data/validation/category_classifier_validation_report.json",
            )
        ),
        "category_testing": Path(
            config_str(
                stage_reports,
                "category_testing",
                "data/testing/category_classifier_test_report.json",
            )
        ),
        "category_model_quality": Path(
            config_str(
                stage_reports,
                "category_model_quality",
                "data/testing/category_ml_model_test_report.json",
            )
        ),
        "input_data_quality": Path(
            config_str(
                stage_reports,
                "input_data_quality",
                "data/testing/input_data_test_report.json",
            )
        ),
        "project_quality": Path(
            config_str(
                stage_reports,
                "project_quality",
                "data/testing/project_quality_report.json",
            )
        ),
        "drift_testing": Path(
            config_str(
                stage_reports,
                "drift_testing",
                "data/testing/drift_test_report.json",
            )
        ),
        "category_explainability": Path(
            config_str(
                stage_reports,
                "category_explainability",
                "data/explainability/category_classifier_explanation_report.json",
            )
        ),
        "llm_exploratory_plan": Path(
            config_str(
                stage_reports,
                "llm_exploratory_plan",
                "data/testing/llm_exploratory_test_plan.json",
            )
        ),
    }


def _run_tracking_artifact_paths(config: Mapping[str, object]) -> dict[str, Path]:
    artifacts = config_section(config, "artifacts")
    return {
        "raw_grocery_dataset": Path(
            config_str(artifacts, "raw_grocery_dataset", "data/raw/grocery_order_items.json")
        ),
        "processed_grocery_dataset": Path(
            config_str(
                artifacts,
                "processed_grocery_dataset",
                "data/processed/grocery_order_items.json",
            )
        ),
        "split_manifest": Path(
            config_str(artifacts, "split_manifest", "data/splits/split_manifest.json")
        ),
        "train_validation_dataset": Path(
            config_str(artifacts, "train_validation_dataset", "data/splits/train_validation.json")
        ),
        "test_dataset": Path(config_str(artifacts, "test_dataset", "data/splits/test.json")),
        "classification_manifest": Path(
            config_str(
                artifacts,
                "classification_manifest",
                "data/classification/category/classification_manifest.json",
            )
        ),
        "classification_train_validation_dataset": Path(
            config_str(
                artifacts,
                "classification_train_validation_dataset",
                "data/classification/category/train_validation.json",
            )
        ),
        "classification_test_dataset": Path(
            config_str(
                artifacts,
                "classification_test_dataset",
                "data/classification/category/test.json",
            )
        ),
        "association_model": Path(
            config_str(artifacts, "association_model", "data/models/association_rules.json")
        ),
        "category_model": Path(
            config_str(artifacts, "category_model", "data/models/category_classifier.json")
        ),
        "category_estimator": Path(
            config_str(artifacts, "category_estimator", "data/models/category_classifier.joblib")
        ),
        "input_data_quality_report": Path(
            config_str(
                artifacts,
                "input_data_quality_report",
                "data/testing/input_data_test_report.json",
            )
        ),
        "association_model_quality_report": Path(
            config_str(
                artifacts,
                "association_model_quality_report",
                "data/testing/ml_model_test_report.json",
            )
        ),
        "category_model_quality_report": Path(
            config_str(
                artifacts,
                "category_model_quality_report",
                "data/testing/category_ml_model_test_report.json",
            )
        ),
        "project_quality_report": Path(
            config_str(
                artifacts,
                "project_quality_report",
                "data/testing/project_quality_report.json",
            )
        ),
        "drift_test_report": Path(
            config_str(
                artifacts,
                "drift_test_report",
                "data/testing/drift_test_report.json",
            )
        ),
        "category_explainability_report": Path(
            config_str(
                artifacts,
                "category_explainability_report",
                "data/explainability/category_classifier_explanation_report.json",
            )
        ),
        "llm_exploratory_test_plan": Path(
            config_str(
                artifacts,
                "llm_exploratory_test_plan",
                "data/testing/llm_exploratory_test_plan.json",
            )
        ),
        "latest_markdown_run_report": Path(
            config_str(
                artifacts,
                "latest_markdown_run_report",
                "data/reports/latest_run_report.md",
            )
        ),
    }


def _markdown_report_paths(config: Mapping[str, object]) -> dict[str, Path]:
    reports = config_section(config, "reports")
    return {
        "data_preprocessing": Path(
            config_str(
                reports,
                "data_preprocessing",
                "data/processed/grocery_data_quality_report.json",
            )
        ),
        "association_validation": Path(
            config_str(
                reports,
                "association_validation",
                "data/validation/association_validation_report.json",
            )
        ),
        "association_testing": Path(
            config_str(
                reports,
                "association_testing",
                "data/testing/association_test_report.json",
            )
        ),
        "association_model_quality": Path(
            config_str(
                reports,
                "association_model_quality",
                "data/testing/ml_model_test_report.json",
            )
        ),
        "category_validation": Path(
            config_str(
                reports,
                "category_validation",
                "data/validation/category_classifier_validation_report.json",
            )
        ),
        "category_testing": Path(
            config_str(
                reports,
                "category_testing",
                "data/testing/category_classifier_test_report.json",
            )
        ),
        "category_model_quality": Path(
            config_str(
                reports,
                "category_model_quality",
                "data/testing/category_ml_model_test_report.json",
            )
        ),
        "input_data_quality": Path(
            config_str(
                reports,
                "input_data_quality",
                "data/testing/input_data_test_report.json",
            )
        ),
        "drift_testing": Path(
            config_str(
                reports,
                "drift_testing",
                "data/testing/drift_test_report.json",
            )
        ),
        "project_quality": Path(
            config_str(
                reports,
                "project_quality",
                "data/testing/project_quality_report.json",
            )
        ),
        "category_explainability": Path(
            config_str(
                reports,
                "category_explainability",
                "data/explainability/category_classifier_explanation_report.json",
            )
        ),
        "llm_exploratory_plan": Path(
            config_str(
                reports,
                "llm_exploratory_plan",
                "data/testing/llm_exploratory_test_plan.json",
            )
        ),
    }


def _add_kosik_endpoint_args(
    parser: argparse.ArgumentParser,
    kosik_config: Mapping[str, object],
    option_prefix: str = "",
    dest_prefix: str = "",
) -> None:
    endpoints = config_section(kosik_config, "endpoints")
    parser.add_argument(
        f"--{option_prefix}order-list-path",
        dest=f"{dest_prefix}order_list_path",
        default=config_str(endpoints, "order_list_path", "/api/front/profile/order-list"),
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        f"--{option_prefix}order-detail-path-template",
        dest=f"{dest_prefix}order_detail_path_template",
        default=config_str(
            endpoints,
            "order_detail_path_template",
            "/api/front/profile/order/{order_id}",
        ),
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        f"--{option_prefix}product-by-slug-path-template",
        dest=f"{dest_prefix}product_by_slug_path_template",
        default=config_str(
            endpoints,
            "product_by_slug_path_template",
            "/api/front/product/slug/{slug}",
        ),
        help=argparse.SUPPRESS,
    )


def _add_rohlik_endpoint_args(
    parser: argparse.ArgumentParser,
    rohlik_config: Mapping[str, object],
    option_prefix: str = "",
    dest_prefix: str = "",
) -> None:
    endpoints = config_section(rohlik_config, "endpoints")
    parser.add_argument(
        f"--{option_prefix}delivered-orders-path",
        dest=f"{dest_prefix}delivered_orders_path",
        default=config_str(endpoints, "delivered_orders_path", "/api/v3/orders/delivered"),
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        f"--{option_prefix}order-detail-path-template",
        dest=f"{dest_prefix}order_detail_path_template",
        default=config_str(endpoints, "order_detail_path_template", "/api/v3/orders/{order_id}"),
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        f"--{option_prefix}product-card-path-template",
        dest=f"{dest_prefix}product_card_path_template",
        default=config_str(
            endpoints,
            "product_card_path_template",
            "/api/v1/products/{product_id}/card",
        ),
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        f"--{option_prefix}product-detail-path-template",
        dest=f"{dest_prefix}product_detail_path_template",
        default=config_str(
            endpoints,
            "product_detail_path_template",
            "/api/v1/products/{product_id}/detail",
        ),
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        f"--{option_prefix}product-detail-content-path-template",
        dest=f"{dest_prefix}product_detail_content_path_template",
        default=config_str(
            endpoints,
            "product_detail_content_path_template",
            "/api/v1/products/{product_id}/detail/content",
        ),
        help=argparse.SUPPRESS,
    )

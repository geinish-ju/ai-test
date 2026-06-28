from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

from ai_testing.classification_preprocessing import (
    ClassificationPreprocessingConfig,
    build_classification_records,
)
from ai_testing.config import (
    config_bool,
    config_float,
    config_int,
    config_section,
    config_str,
    config_str_tuple,
    default_config_path,
    load_config,
)
from ai_testing.core import artifact_metadata
from ai_testing.data_acquisition import (
    KosikApiClient,
    KosikOrderHistoryAdapter,
    RohlikApiClient,
    RohlikOrderHistoryAdapter,
)
from ai_testing.data_preprocessing import (
    DEFAULT_EXACT_TIME_FIELDS,
    DEFAULT_IDENTIFIER_FIELDS,
    DEFAULT_OUTPUT_FIELDS,
    PreprocessingConfig,
    preprocess_grocery_records,
)
from ai_testing.data_splitting import DatasetSplitConfig, split_dataset_records
from ai_testing.drift_testing import RunDriftTestConfig, test_run_drift
from ai_testing.input_data_testing import InputDataFold, InputDataTestConfig, test_input_data
from ai_testing.llm_exploratory_testing import (
    LLMExploratoryTestingConfig,
    create_llm_exploratory_test_plan,
)
from ai_testing.ml_model_testing import (
    AssociationMLModelTestConfig,
    TextClassifierMLModelTestConfig,
    test_association_ml_model,
    test_text_classifier_ml_model,
)
from ai_testing.model_explainability import (
    TextClassifierExplainabilityConfig,
    explain_text_classifier,
)
from ai_testing.model_testing import (
    AssociationTestConfig,
    TextClassifierTestConfig,
    test_association_rules,
    test_text_classifier,
)
from ai_testing.model_training import (
    AssociationRulesConfig,
    TextClassifierConfig,
    save_text_classifier_artifact,
    train_association_rules,
    train_text_classifier,
)
from ai_testing.model_validation import (
    AssociationValidationConfig,
    AssociationValidationFold,
    TextClassificationValidationConfig,
    TextClassificationValidationFold,
    validate_association_rules,
    validate_text_classifier,
)
from ai_testing.project_quality import aggregate_quality_reports
from ai_testing.reporting import (
    MarkdownReportConfig,
    RunMarkdownReportConfig,
    build_markdown_report,
    build_run_markdown_report,
)
from ai_testing.run_tracking import (
    RunTrackingConfig,
    StageExecutionTrackingConfig,
    build_run_report,
    build_stage_history_markdown,
    build_stage_metric_history_markdown,
    compact_run_history,
    latest_run_report_path,
    record_stage_execution,
    update_run_index,
)
from ai_testing.sample_data import sample_grocery_order_item_records


def main(argv: Sequence[str] | None = None) -> int:
    config_path, parser_argv = _extract_config_path(argv)
    app_config = _load_config_for_cli(config_path)
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
    track_run_parser.set_defaults(
        stage_report_paths=_run_tracking_stage_report_paths(run_tracking_config),
        artifact_paths=_run_tracking_artifact_paths(run_tracking_config),
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

    args = parser.parse_args(parser_argv)
    if args.command == "export-kosik":
        records = _fetch_kosik(args)
        output_path = Path(args.output)
        _write_records(records, output_path)
        _record_stage_result(
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
        _record_stage_result(
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
        _record_stage_result(
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
        _record_stage_result(
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
        _track_run(args)
        return 0

    if args.command == "run-history":
        _run_history(args)
        return 0

    if args.command == "test-drift":
        _test_drift(args)
        return 0

    if args.command == "generate-run-report":
        _generate_run_report(args)
        return 0

    if args.command == "generate-markdown-reports":
        _generate_markdown_reports(args)
        return 0

    if args.command == "explain-category-classifier":
        _explain_category_classifier(args)
        return 0

    if args.command == "create-llm-exploratory-plan":
        _create_llm_exploratory_plan(args)
        return 0

    return 2


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


def _extract_config_path(argv: Sequence[str] | None) -> tuple[str, list[str]]:
    raw_args = list(sys.argv[1:] if argv is None else argv)
    config_path = default_config_path()
    cleaned_args: list[str] = []
    index = 0

    while index < len(raw_args):
        argument = raw_args[index]
        if argument == "--config":
            if index + 1 >= len(raw_args):
                raise SystemExit("--config requires a path value")
            config_path = raw_args[index + 1]
            index += 2
            continue
        if argument.startswith("--config="):
            config_path = argument.split("=", 1)[1]
            index += 1
            continue

        cleaned_args.append(argument)
        index += 1

    return config_path, cleaned_args


def _load_config_for_cli(config_path: str) -> dict[str, object]:
    try:
        return load_config(config_path)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        raise SystemExit(f"Cannot load config {config_path}: {error}") from error


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


def _fetch_kosik(args: argparse.Namespace) -> list[dict[str, object]]:
    cookies = _resolve_cookies(args.cookies, args.cookies_file, args.cookies_env)
    client = KosikApiClient(
        base_url=args.base_url,
        cookies=cookies,
        order_list_path=args.order_list_path,
        order_detail_path_template=args.order_detail_path_template,
        product_by_slug_path_template=args.product_by_slug_path_template,
    )
    return KosikOrderHistoryAdapter(
        client,
        include_raw=bool(args.include_raw),
        enrich_products=bool(args.product_enrichment),
        order_page_limit=int(getattr(args, "order_page_limit", 50)),
        include_archived_orders=bool(args.include_archived_orders),
    ).fetch_order_item_records()


def _fetch_rohlik(args: argparse.Namespace) -> list[dict[str, object]]:
    cookies = _resolve_cookies(args.cookies, args.cookies_file, args.cookies_env)
    client = RohlikApiClient(
        base_url=args.base_url,
        cookies=cookies,
        delivered_orders_path=args.delivered_orders_path,
        order_detail_path_template=args.order_detail_path_template,
        product_card_path_template=args.product_card_path_template,
        product_detail_path_template=args.product_detail_path_template,
        product_detail_content_path_template=args.product_detail_content_path_template,
    )
    return RohlikOrderHistoryAdapter(
        client,
        include_raw=bool(args.include_raw),
        enrich_products=bool(args.product_enrichment),
        include_product_content=bool(getattr(args, "include_product_content", False)),
        order_page_limit=int(getattr(args, "order_page_limit", 10)),
        include_archived_orders=bool(args.include_archived_orders),
    ).fetch_order_item_records()


def _write_records(records: list[dict[str, object]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(output_path),
                "record_count": len(records),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def _preprocess_data(args: argparse.Namespace) -> None:
    input_path = Path(args.input)
    output_path = Path(args.output)
    report_path = Path(args.report_output)
    raw_records = _read_records(input_path)
    result = preprocess_grocery_records(
        raw_records,
        config=PreprocessingConfig(
            identifier_fields=tuple(args.identifier_fields),
            exact_time_fields=tuple(args.exact_time_fields),
            output_fields=tuple(args.output_fields),
            drop_exact_duplicates=bool(args.drop_exact_duplicates),
            basket_id_prefix=str(args.basket_id_prefix),
        ),
    )

    _write_json(result.records, output_path)
    _write_json(result.report, report_path)
    _record_stage_result(
        args,
        stage_name="data_preprocessing",
        artifacts={"processed_dataset": output_path, "quality_report": report_path},
        payload=result.report,
    )
    print(
        json.dumps(
            {
                "input": str(input_path),
                "output": str(output_path),
                "report_output": str(report_path),
                "input_record_count": result.report["input_record_count"],
                "output_record_count": result.report["output_record_count"],
                "exact_duplicate_rows": result.report["duplicates"]["exact_duplicate_rows"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def _split_datasets(args: argparse.Namespace) -> None:
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    records = _read_records(input_path)
    try:
        result = split_dataset_records(
            records,
            config=DatasetSplitConfig(
                group_field=str(args.group_field),
                stratify_field=str(args.stratify_field),
                test_size=float(args.test_size),
                n_splits=int(args.n_splits),
                random_seed=int(args.random_seed),
            ),
        )
    except ValueError as error:
        raise SystemExit(f"Cannot split datasets: {error}") from error

    output_dir.mkdir(parents=True, exist_ok=True)
    folds_dir = output_dir / "folds"
    folds_dir.mkdir(parents=True, exist_ok=True)

    train_validation_path = output_dir / "train_validation.json"
    test_path = output_dir / "test.json"
    manifest_path = output_dir / "split_manifest.json"
    _write_json(result.train_validation_records, train_validation_path)
    _write_json(result.test_records, test_path)
    for fold in result.folds:
        fold_prefix = f"fold_{fold.fold_index:02d}"
        _write_json(fold.train_records, folds_dir / f"{fold_prefix}_train.json")
        _write_json(fold.validation_records, folds_dir / f"{fold_prefix}_validation.json")

    manifest = {
        **result.manifest,
        "outputs": {
            "train_validation": str(train_validation_path),
            "test": str(test_path),
            "manifest": str(manifest_path),
            "folds_dir": str(folds_dir),
        },
    }
    _write_json(manifest, manifest_path)
    _record_stage_result(
        args,
        stage_name="data_splitting",
        artifacts={
            "train_validation": train_validation_path,
            "test": test_path,
            "manifest": manifest_path,
            "folds_dir": folds_dir,
        },
        payload=manifest,
    )
    print(
        json.dumps(
            {
                "input": str(input_path),
                "output_dir": str(output_dir),
                "input_record_count": manifest["input_record_count"],
                "group_count": manifest["group_count"],
                "train_validation_record_count": manifest["splits"]["train_validation"][
                    "record_count"
                ],
                "test_record_count": manifest["splits"]["test"]["record_count"],
                "n_splits": manifest["parameters"]["n_splits"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def _build_classification_dataset(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    folds_output_dir = output_dir / "folds"
    manifest_path = output_dir / "classification_manifest.json"
    config = ClassificationPreprocessingConfig(
        target_field=str(args.target_field),
        text_fields=tuple(args.text_fields),
        metadata_fields=tuple(args.metadata_fields),
        min_label_count=int(args.min_label_count),
    )
    processed_result = build_classification_records(
        _read_records(Path(args.processed_input)),
        config=config,
    )
    train_validation_result = build_classification_records(
        _read_records(Path(args.train_validation_input)),
        config=config,
        allowed_labels=processed_result.allowed_labels,
    )
    test_result = build_classification_records(
        _read_records(Path(args.test_input)),
        config=config,
        allowed_labels=processed_result.allowed_labels,
    )
    folds = _read_input_data_folds(Path(args.folds_dir))

    output_dir.mkdir(parents=True, exist_ok=True)
    folds_output_dir.mkdir(parents=True, exist_ok=True)
    train_validation_path = output_dir / "train_validation.json"
    test_path = output_dir / "test.json"
    _write_json(train_validation_result.records, train_validation_path)
    _write_json(test_result.records, test_path)

    fold_reports: list[dict[str, object]] = []
    for fold in folds:
        train_result = build_classification_records(
            fold.train_records,
            config=config,
            allowed_labels=processed_result.allowed_labels,
        )
        validation_result = build_classification_records(
            fold.validation_records,
            config=config,
            allowed_labels=processed_result.allowed_labels,
        )
        fold_prefix = f"fold_{fold.fold_index:02d}"
        train_path = folds_output_dir / f"{fold_prefix}_train.json"
        validation_path = folds_output_dir / f"{fold_prefix}_validation.json"
        _write_json(train_result.records, train_path)
        _write_json(validation_result.records, validation_path)
        fold_reports.append(
            {
                "fold_index": fold.fold_index,
                "train": train_result.report,
                "validation": validation_result.report,
                "outputs": {
                    "train": str(train_path),
                    "validation": str(validation_path),
                },
            }
        )

    manifest = {
        "step": "Supervised classification dataset preprocessing",
        "task": "product category text classification",
        "source_inputs": {
            "processed": str(args.processed_input),
            "train_validation": str(args.train_validation_input),
            "test": str(args.test_input),
            "folds_dir": str(args.folds_dir),
        },
        "outputs": {
            "train_validation": str(train_validation_path),
            "test": str(test_path),
            "folds_dir": str(folds_output_dir),
            "manifest": str(manifest_path),
        },
        "global": processed_result.report,
        "train_validation": train_validation_result.report,
        "test": test_result.report,
        "folds": fold_reports,
    }
    _write_json(manifest, manifest_path)
    _record_stage_result(
        args,
        stage_name="classification_preprocessing",
        artifacts={
            "train_validation": train_validation_path,
            "test": test_path,
            "folds_dir": folds_output_dir,
            "manifest": manifest_path,
        },
        payload=manifest,
    )
    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "target_field": config.target_field,
                "text_fields": list(config.text_fields),
                "allowed_label_count": len(processed_result.allowed_labels),
                "train_validation_record_count": train_validation_result.report[
                    "output_record_count"
                ],
                "test_record_count": test_result.report["output_record_count"],
                "fold_count": len(fold_reports),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def _train_category_classifier(args: argparse.Namespace) -> None:
    input_path = Path(args.input)
    output_path = Path(args.output)
    estimator_output_path = Path(args.estimator_output)
    records = _read_records(input_path)
    try:
        result = train_text_classifier(
            records,
            config=TextClassifierConfig(
                text_field=str(args.text_field),
                label_field=str(args.label_field),
                alpha=float(args.alpha),
                min_token_length=int(args.min_token_length),
                max_vocabulary_size=int(args.max_vocabulary_size),
            ),
        )
    except ValueError as error:
        raise SystemExit(f"Cannot train category classifier: {error}") from error

    model = {
        **result.model,
        "training_input": str(input_path),
        "estimator_path": str(estimator_output_path),
        "note": "Trained on the supervised training split only. Hold-out test data is not used.",
    }
    try:
        save_text_classifier_artifact(
            model=model,
            estimator=result.estimator,
            manifest_path=output_path,
            estimator_path=estimator_output_path,
        )
    except ValueError as error:
        raise SystemExit(f"Cannot save category classifier: {error}") from error
    _record_stage_result(
        args,
        stage_name="model_training.category_classifier",
        artifacts={"model": output_path, "estimator": estimator_output_path},
        payload=model,
    )
    summary = model["summary"]
    print(
        json.dumps(
            {
                "input": str(input_path),
                "output": str(output_path),
                "estimator_output": str(estimator_output_path),
                "model_type": model["model_type"],
                "algorithm": model["algorithm"],
                "framework": model["framework"],
                "training_example_count": summary["training_example_count"],
                "class_count": summary["class_count"],
                "vocabulary_size": summary["vocabulary_size"],
                "majority_class": summary["majority_class"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def _validate_category_classifier(args: argparse.Namespace) -> None:
    folds_dir = Path(args.folds_dir)
    output_path = Path(args.output)
    folds = _read_text_classification_folds(folds_dir)
    try:
        result = validate_text_classifier(
            folds,
            config=TextClassificationValidationConfig(
                text_field=str(args.text_field),
                label_field=str(args.label_field),
                alpha=float(args.alpha),
                min_token_length=int(args.min_token_length),
                max_vocabulary_size=int(args.max_vocabulary_size),
                top_confusions=int(args.top_confusions),
            ),
        )
    except ValueError as error:
        raise SystemExit(f"Cannot validate category classifier: {error}") from error

    report = {
        **result.report,
        "folds_dir": str(folds_dir),
        "output": str(output_path),
    }
    _write_json(report, output_path)
    _record_stage_result(
        args,
        stage_name="model_validation.category_classifier",
        artifacts={"validation_report": output_path},
        payload=report,
    )
    summary = report["summary"]
    print(
        json.dumps(
            {
                "folds_dir": str(folds_dir),
                "output": str(output_path),
                "fold_count": summary["fold_count"],
                "mean_accuracy": summary["mean_accuracy"],
                "std_accuracy": summary["std_accuracy"],
                "mean_macro_precision": summary["mean_macro_precision"],
                "mean_macro_recall": summary["mean_macro_recall"],
                "mean_macro_f1": summary["mean_macro_f1"],
                "mean_weighted_precision": summary["mean_weighted_precision"],
                "mean_weighted_recall": summary["mean_weighted_recall"],
                "mean_weighted_f1": summary["mean_weighted_f1"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def _test_category_classifier(args: argparse.Namespace) -> None:
    model_path = Path(args.model_input)
    test_input_path = Path(args.test_input)
    output_path = Path(args.output)
    model = _read_json_object(model_path)
    test_records = _read_records(test_input_path)
    try:
        result = test_text_classifier(
            model,
            test_records,
            config=TextClassifierTestConfig(
                text_field=str(args.text_field),
                label_field=str(args.label_field),
                top_confusions=int(args.top_confusions),
            ),
        )
    except ValueError as error:
        raise SystemExit(f"Cannot test category classifier: {error}") from error

    report = {
        **result.report,
        "model_input": str(model_path),
        "test_input": str(test_input_path),
        "output": str(output_path),
    }
    _write_json(report, output_path)
    _record_stage_result(
        args,
        stage_name="model_testing.category_classifier",
        artifacts={"test_report": output_path},
        payload=report,
    )
    test_summary = report["test"]
    print(
        json.dumps(
            {
                "model_input": str(model_path),
                "test_input": str(test_input_path),
                "output": str(output_path),
                "record_count": test_summary["record_count"],
                "evaluated_record_count": test_summary["evaluated_record_count"],
                "class_count": test_summary["class_count"],
                "accuracy": test_summary["accuracy"],
                "macro_precision": test_summary["macro_precision"],
                "macro_recall": test_summary["macro_recall"],
                "macro_f1": test_summary["macro_f1"],
                "weighted_precision": test_summary["weighted_precision"],
                "weighted_recall": test_summary["weighted_recall"],
                "weighted_f1": test_summary["weighted_f1"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def _train_associations(args: argparse.Namespace) -> None:
    input_path = Path(args.input)
    output_path = Path(args.output)
    records = _read_records(input_path)
    try:
        result = train_association_rules(
            records,
            config=AssociationRulesConfig(
                basket_field=str(args.basket_field),
                item_field=str(args.item_field),
                min_support=float(args.min_support),
                min_confidence=float(args.min_confidence),
                min_lift=float(args.min_lift),
                max_itemset_size=int(args.max_itemset_size),
                max_rules=int(args.max_rules),
                max_itemsets=int(args.max_itemsets),
            ),
        )
    except ValueError as error:
        raise SystemExit(f"Cannot train association rules: {error}") from error

    model = {
        **result.model,
        "training_input": str(input_path),
        "note": "Trained on the training split only. Hold-out test data is not used.",
    }
    _write_json(model, output_path)
    _record_stage_result(
        args,
        stage_name="model_training.association_rules",
        artifacts={"model": output_path},
        payload=model,
    )
    summary = model["summary"]
    print(
        json.dumps(
            {
                "input": str(input_path),
                "output": str(output_path),
                "model_type": model["model_type"],
                "algorithm": model["algorithm"],
                "basket_count": summary["basket_count"],
                "item_count": summary["item_count"],
                "frequent_itemset_count": summary["frequent_itemset_count"],
                "rule_count": summary["rule_count"],
                "exported_rule_count": summary["exported_rule_count"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def _validate_associations(args: argparse.Namespace) -> None:
    folds_dir = Path(args.folds_dir)
    output_path = Path(args.output)
    folds = _read_association_validation_folds(folds_dir)
    try:
        result = validate_association_rules(
            folds,
            config=AssociationValidationConfig(
                basket_field=str(args.basket_field),
                item_field=str(args.item_field),
                min_support=float(args.min_support),
                min_confidence=float(args.min_confidence),
                min_lift=float(args.min_lift),
                max_itemset_size=int(args.max_itemset_size),
                max_rules=int(args.max_rules),
                max_itemsets=int(args.max_itemsets),
                top_rules=int(args.top_rules),
            ),
        )
    except ValueError as error:
        raise SystemExit(f"Cannot validate association rules: {error}") from error

    report = {
        **result.report,
        "folds_dir": str(folds_dir),
        "output": str(output_path),
    }
    _write_json(report, output_path)
    _record_stage_result(
        args,
        stage_name="model_validation.association_rules",
        artifacts={"validation_report": output_path},
        payload=report,
    )
    summary = report["summary"]
    print(
        json.dumps(
            {
                "folds_dir": str(folds_dir),
                "output": str(output_path),
                "fold_count": summary["fold_count"],
                "mean_train_rule_count": summary["mean_train_rule_count"],
                "mean_validation_confidence": summary["mean_validation_confidence"],
                "mean_validation_lift": summary["mean_validation_lift"],
                "mean_antecedent_coverage": summary["mean_antecedent_coverage"],
                "mean_hit_rate_per_covered_basket": summary["mean_hit_rate_per_covered_basket"],
                "mean_abs_confidence_gap": summary["mean_abs_confidence_gap"],
                "mean_stable_rule_count": summary["mean_stable_rule_count"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def _test_associations(args: argparse.Namespace) -> None:
    model_path = Path(args.model_input)
    test_input_path = Path(args.test_input)
    output_path = Path(args.output)
    model = _read_json_object(model_path)
    test_records = _read_records(test_input_path)
    try:
        result = test_association_rules(
            model,
            test_records,
            config=AssociationTestConfig(
                basket_field=str(args.basket_field),
                item_field=str(args.item_field),
                min_confidence=float(args.min_confidence),
                min_lift=float(args.min_lift),
                top_rules=int(args.top_rules),
            ),
        )
    except ValueError as error:
        raise SystemExit(f"Cannot test association rules: {error}") from error

    report = {
        **result.report,
        "model_input": str(model_path),
        "test_input": str(test_input_path),
        "output": str(output_path),
    }
    _write_json(report, output_path)
    _record_stage_result(
        args,
        stage_name="model_testing.association_rules",
        artifacts={"test_report": output_path},
        payload=report,
    )
    test_summary = report["test"]
    print(
        json.dumps(
            {
                "model_input": str(model_path),
                "test_input": str(test_input_path),
                "output": str(output_path),
                "record_count": test_summary["record_count"],
                "basket_count": test_summary["basket_count"],
                "evaluated_rule_count": test_summary["evaluated_rule_count"],
                "stable_rule_count": test_summary["stable_rule_count"],
                "mean_test_confidence": test_summary["mean_test_confidence"],
                "mean_test_lift": test_summary["mean_test_lift"],
                "antecedent_coverage": test_summary["antecedent_coverage"],
                "hit_rate_per_covered_basket": test_summary["hit_rate_per_covered_basket"],
                "mean_abs_confidence_gap": test_summary["mean_abs_confidence_gap"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def _test_input_data(args: argparse.Namespace) -> None:
    processed_input_path = Path(args.processed_input)
    train_validation_input_path = Path(args.train_validation_input)
    test_input_path = Path(args.test_input)
    folds_dir = Path(args.folds_dir)
    output_path = Path(args.output)
    result = test_input_data(
        processed_records=_read_records(processed_input_path),
        train_validation_records=_read_records(train_validation_input_path),
        test_records=_read_records(test_input_path),
        folds=_read_input_data_folds(folds_dir),
        config=InputDataTestConfig(
            required_fields=tuple(args.required_fields),
            protected_fields=tuple(args.protected_fields),
            critical_fields=tuple(args.critical_fields),
            coverage_fields=tuple(args.coverage_fields),
            group_field=str(args.group_field),
            stratify_field=str(args.stratify_field),
            expected_shops=tuple(args.expected_shops),
            expected_currencies=tuple(args.expected_currencies),
            max_coverage_missing_rate=float(args.max_coverage_missing_rate),
            max_split_distribution_delta=float(args.max_split_distribution_delta),
        ),
    )
    report = {
        **result.report,
        "inputs": {
            "processed": str(processed_input_path),
            "train_validation": str(train_validation_input_path),
            "test": str(test_input_path),
            "folds_dir": str(folds_dir),
        },
        "artifacts": {
            "processed": artifact_metadata(processed_input_path, "dataset", "processed_input"),
            "train_validation": artifact_metadata(
                train_validation_input_path,
                "dataset",
                "train_validation_input",
            ),
            "test": artifact_metadata(test_input_path, "dataset", "test_input"),
            "folds_dir": artifact_metadata(folds_dir, "dataset_directory", "folds_input"),
        },
        "output": str(output_path),
    }
    _write_json(report, output_path)
    _record_stage_result(
        args,
        stage_name="input_data_testing",
        artifacts={"input_data_report": output_path},
        payload=report,
    )
    _print_report_summary(report, output_path)
    _exit_if_report_failed(report)


def _test_ml_model(args: argparse.Namespace) -> None:
    model_path = Path(args.model_input)
    validation_report_path = Path(args.validation_report_input)
    test_report_path = Path(args.test_report_input)
    test_dataset_path = Path(args.test_dataset_input)
    output_path = Path(args.output)
    result = test_association_ml_model(
        model=_read_json_object(model_path),
        validation_report=_read_json_object(validation_report_path),
        test_report=_read_json_object(test_report_path),
        config=AssociationMLModelTestConfig(
            test_dataset_input=str(args.test_dataset_input),
            forbidden_feature_fields=tuple(args.forbidden_feature_fields),
            min_mean_test_confidence=float(args.min_mean_test_confidence),
            min_mean_test_lift=float(args.min_mean_test_lift),
            max_mean_abs_test_confidence_gap=float(args.max_mean_abs_test_confidence_gap),
            max_validation_test_confidence_delta=float(args.max_validation_test_confidence_delta),
            max_validation_test_lift_delta=float(args.max_validation_test_lift_delta),
        ),
    )
    report = {
        **result.report,
        "inputs": {
            "model": str(model_path),
            "validation_report": str(validation_report_path),
            "test_report": str(test_report_path),
            "test_dataset": str(test_dataset_path),
        },
        "artifacts": {
            "model": artifact_metadata(model_path, "model", "model_input"),
            "validation_report": artifact_metadata(
                validation_report_path,
                "report",
                "validation_report_input",
            ),
            "test_report": artifact_metadata(test_report_path, "report", "test_report_input"),
            "test_dataset": artifact_metadata(test_dataset_path, "dataset", "test_dataset_input"),
        },
        "output": str(output_path),
    }
    _write_json(report, output_path)
    _record_stage_result(
        args,
        stage_name="ml_model_testing.association_rules",
        artifacts={"ml_model_report": output_path},
        payload=report,
    )
    _print_report_summary(report, output_path)
    _exit_if_report_failed(report)


def _test_category_ml_model(args: argparse.Namespace) -> None:
    model_path = Path(args.model_input)
    validation_report_path = Path(args.validation_report_input)
    test_report_path = Path(args.test_report_input)
    classification_manifest_path = Path(args.classification_manifest_input)
    test_dataset_path = Path(args.test_dataset_input)
    output_path = Path(args.output)
    model = _read_json_object(model_path)
    estimator_path = Path(str(model.get("estimator_path", "")))
    result = test_text_classifier_ml_model(
        model=model,
        validation_report=_read_json_object(validation_report_path),
        test_report=_read_json_object(test_report_path),
        classification_manifest=_read_json_object(classification_manifest_path),
        config=TextClassifierMLModelTestConfig(
            test_dataset_input=str(args.test_dataset_input),
            forbidden_feature_fields=tuple(args.forbidden_feature_fields),
            min_test_accuracy=float(args.min_test_accuracy),
            min_test_macro_precision=float(args.min_test_macro_precision),
            min_test_macro_recall=float(args.min_test_macro_recall),
            min_test_macro_f1=float(args.min_test_macro_f1),
            min_test_weighted_f1=float(args.min_test_weighted_f1),
            max_validation_test_accuracy_delta=float(args.max_validation_test_accuracy_delta),
            max_validation_test_macro_f1_delta=float(args.max_validation_test_macro_f1_delta),
        ),
    )
    report = {
        **result.report,
        "inputs": {
            "model": str(model_path),
            "estimator": str(estimator_path),
            "validation_report": str(validation_report_path),
            "test_report": str(test_report_path),
            "classification_manifest": str(classification_manifest_path),
            "test_dataset": str(test_dataset_path),
        },
        "artifacts": {
            "model": artifact_metadata(model_path, "model", "model_input"),
            "estimator": artifact_metadata(estimator_path, "model", "estimator_input"),
            "validation_report": artifact_metadata(
                validation_report_path,
                "report",
                "validation_report_input",
            ),
            "test_report": artifact_metadata(test_report_path, "report", "test_report_input"),
            "classification_manifest": artifact_metadata(
                classification_manifest_path,
                "manifest",
                "classification_manifest_input",
            ),
            "test_dataset": artifact_metadata(test_dataset_path, "dataset", "test_dataset_input"),
        },
        "output": str(output_path),
    }
    _write_json(report, output_path)
    _record_stage_result(
        args,
        stage_name="ml_model_testing.category_classifier",
        artifacts={"ml_model_report": output_path},
        payload=report,
    )
    _print_report_summary(report, output_path)
    _exit_if_report_failed(report)


def _run_quality_gates(args: argparse.Namespace) -> None:
    input_data_report_path = Path(args.input_data_report)
    association_model_report_path = Path(args.association_model_report)
    category_model_report_path = Path(args.category_model_report)
    output_path = Path(args.output)
    reports: dict[str, Mapping[str, object]] = {
        "input_data": _read_json_object(input_data_report_path),
        "association_model": _read_json_object(association_model_report_path),
        "category_model": _read_json_object(category_model_report_path),
    }
    drift_report_path = Path(str(args.drift_report)) if args.drift_report else None
    if drift_report_path is not None:
        if not drift_report_path.exists():
            raise SystemExit(f"Drift report does not exist: {drift_report_path}")
        reports["drift"] = _read_json_object(drift_report_path)

    result = aggregate_quality_reports(
        reports,
    )
    report = {
        **result.report,
        "inputs": {
            "input_data_report": str(input_data_report_path),
            "association_model_report": str(association_model_report_path),
            "category_model_report": str(category_model_report_path),
            "drift_report": str(drift_report_path) if drift_report_path is not None else None,
        },
        "artifacts": {
            "input_data_report": artifact_metadata(
                input_data_report_path,
                "report",
                "input_data_report",
            ),
            "association_model_report": artifact_metadata(
                association_model_report_path,
                "report",
                "association_model_report",
            ),
            "category_model_report": artifact_metadata(
                category_model_report_path,
                "report",
                "category_model_report",
            ),
            "drift_report": (
                artifact_metadata(drift_report_path, "report", "drift_report")
                if drift_report_path is not None
                else {"path": None, "type": "report", "exists": False, "role": "drift_report"}
            ),
        },
        "output": str(output_path),
    }
    _write_json(report, output_path)
    _record_stage_result(
        args,
        stage_name="project_quality",
        artifacts={"project_quality_report": output_path},
        payload=report,
    )
    metrics = _mapping(report.get("metrics"))
    _print_report_summary(
        report,
        output_path,
        extra={
            "failed_report_count": metrics.get("failed_report_count"),
            "total_child_failed_check_count": metrics.get("total_child_failed_check_count"),
        },
    )
    _exit_if_report_failed(report)


def _test_drift(args: argparse.Namespace) -> None:
    run_report_path = _resolve_run_report_input(
        Path(args.run_report_input),
        bool(args.use_latest_run),
    )
    output_path = Path(args.output)
    run_report = _read_json_object(run_report_path)
    result = test_run_drift(
        run_report,
        config=RunDriftTestConfig(
            require_baseline=bool(args.require_baseline),
            max_regressed_metric_count=int(args.max_regressed_metric_count),
            max_total_variation_distance=float(args.max_total_variation_distance),
            max_distribution_total_variation_distance=float(
                args.max_distribution_total_variation_distance
            ),
            max_critical_metric_regression=float(args.max_critical_metric_regression),
            critical_metrics=tuple(args.critical_metrics),
        ),
    )
    report = {
        **result,
        "inputs": {"run_report": str(run_report_path)},
        "artifacts": {"run_report": artifact_metadata(run_report_path, "report", "run_report")},
        "output": str(output_path),
    }
    _write_json(report, output_path)
    _record_stage_result(
        args,
        stage_name="drift_testing",
        artifacts={"drift_report": output_path},
        payload=report,
    )
    _print_report_summary(
        report,
        output_path,
        extra={
            "run_id": report.get("run_id"),
            "baseline_run_id": report.get("baseline_run_id"),
            "max_total_variation_distance": _mapping(report.get("metrics")).get(
                "max_total_variation_distance"
            ),
        },
    )
    _exit_if_report_failed(report)


def _generate_run_report(args: argparse.Namespace) -> None:
    run_report_path = _resolve_run_report_input(
        Path(args.run_report_input),
        bool(args.use_latest_run),
    )
    drift_report_path = Path(args.drift_report_input)
    drift_report = _read_json_object(drift_report_path) if drift_report_path.exists() else None
    output_path = Path(args.output)
    markdown = build_run_markdown_report(
        _read_json_object(run_report_path),
        drift_report=drift_report,
        config=RunMarkdownReportConfig(
            max_metric_deltas=int(args.max_metric_deltas),
            max_drift_distributions=int(args.max_drift_distributions),
            max_artifacts=int(args.max_artifacts),
        ),
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(markdown, encoding="utf-8")
    _record_stage_result(
        args,
        stage_name="reporting.latest_run_report",
        artifacts={"markdown_report": output_path, "run_report": run_report_path},
        payload={"summary": {"generated_report_count": 1}},
    )
    print(
        json.dumps(
            {
                "output": str(output_path),
                "run_report_input": str(run_report_path),
                "drift_report_input": str(drift_report_path) if drift_report is not None else None,
                "format": "markdown",
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def _generate_markdown_reports(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    report_paths = (
        _parse_report_specs(tuple(args.report)) if args.report else _path_mapping(args.report_paths)
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    generated: list[dict[str, object]] = []
    skipped: list[dict[str, object]] = []
    markdown_config = MarkdownReportConfig(
        max_checks=int(args.max_checks),
        max_rows=int(args.max_rows),
    )

    for name, report_path in sorted(report_paths.items()):
        output_path = output_dir / f"{_safe_filename(name)}.md"
        if not report_path.exists():
            skipped.append({"name": name, "input": str(report_path), "reason": "missing"})
            continue
        markdown = build_markdown_report(
            _read_json_object(report_path),
            title=_markdown_title(name),
            source_path=report_path,
            config=markdown_config,
        )
        output_path.write_text(markdown, encoding="utf-8")
        generated.append({"name": name, "input": str(report_path), "output": str(output_path)})

    if args.include_run_report:
        run_output_path = output_dir / "latest_run_report.md"
        try:
            run_report_path = _resolve_run_report_input(
                Path(args.run_report_input),
                bool(args.use_latest_run),
            )
        except SystemExit:
            skipped.append(
                {
                    "name": "latest_run_report",
                    "input": str(args.run_report_input),
                    "reason": "missing_latest_run",
                }
            )
        else:
            drift_report_path = Path(args.drift_report_input)
            drift_report = (
                _read_json_object(drift_report_path) if drift_report_path.exists() else None
            )
            run_markdown = build_run_markdown_report(
                _read_json_object(run_report_path),
                drift_report=drift_report,
            )
            run_output_path.write_text(run_markdown, encoding="utf-8")
            generated.append(
                {
                    "name": "latest_run_report",
                    "input": str(run_report_path),
                    "output": str(run_output_path),
                }
            )

    stage_history_path = Path(str(args.stage_history_output))
    stage_history_output_path = output_dir / "stage_history.md"
    stage_metric_history_output_path = output_dir / "stage_metric_history.md"
    if stage_history_path.exists():
        stage_history = _read_json_object(stage_history_path)
        stage_history_output_path.write_text(
            build_stage_history_markdown(
                stage_history,
                recent_entries_per_stage=int(args.stage_history_recent_entries_per_stage),
            ),
            encoding="utf-8",
        )
        stage_metric_history_output_path.write_text(
            build_stage_metric_history_markdown(
                stage_history,
                recent_entries_per_stage=int(args.stage_metric_history_recent_entries_per_stage),
            ),
            encoding="utf-8",
        )
        generated.append(
            {
                "name": "stage_history",
                "input": str(stage_history_path),
                "output": str(stage_history_output_path),
            }
        )
        generated.append(
            {
                "name": "stage_metric_history",
                "input": str(stage_history_path),
                "output": str(stage_metric_history_output_path),
            }
        )
    else:
        skipped.append(
            {
                "name": "stage_history",
                "input": str(stage_history_path),
                "reason": "missing",
            }
        )
        skipped.append(
            {
                "name": "stage_metric_history",
                "input": str(stage_history_path),
                "reason": "missing",
            }
        )

    _record_stage_result(
        args,
        stage_name="reporting.markdown_reports",
        artifacts={"output_dir": output_dir},
        payload={
            "summary": {
                "generated_report_count": len(generated),
                "skipped_report_count": len(skipped),
            }
        },
    )
    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "generated_count": len(generated),
                "skipped_count": len(skipped),
                "generated": generated,
                "skipped": skipped,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def _track_run(args: argparse.Namespace) -> None:
    runs_dir = Path(args.runs_dir)
    index_output_path = Path(args.index_output)
    baseline_report = _baseline_run_report(
        explicit_baseline=str(args.baseline_run_report or ""),
        compare_to_latest=bool(args.compare_to_latest),
        index_output_path=index_output_path,
    )
    config = RunTrackingConfig(
        runs_dir=runs_dir,
        index_output=index_output_path,
        stage_report_paths=_path_mapping(args.stage_report_paths),
        artifact_paths=_path_mapping(args.artifact_paths),
    )
    report = build_run_report(
        config,
        run_id=str(args.run_id) if args.run_id else None,
        run_name=str(args.run_name) if args.run_name else None,
        baseline_report=baseline_report,
    )
    run_id = str(report["run_id"])
    output_path = Path(args.output) if args.output else runs_dir / run_id / "run_report.json"
    markdown_output_path = output_path.with_suffix(".md")
    _write_json(report, output_path)
    markdown_output_path.write_text(build_run_markdown_report(report), encoding="utf-8")
    index = update_run_index(index_output_path, report, output_path)
    _record_stage_result(
        args,
        stage_name="run_tracking",
        artifacts={
            "run_report": output_path,
            "run_report_markdown": markdown_output_path,
            "run_index": index_output_path,
        },
        payload=report,
    )
    comparison = _mapping(report.get("comparison"))
    delta_summary = _mapping(comparison.get("metric_delta_summary"))
    summary = _mapping(report.get("summary"))
    print(
        json.dumps(
            {
                "run_id": run_id,
                "output": str(output_path),
                "index_output": str(index_output_path),
                "run_count": index.get("run_count"),
                "quality_status": summary.get("quality_status"),
                "available_stage_report_count": summary.get("available_stage_report_count"),
                "available_artifact_count": summary.get("available_artifact_count"),
                "metric_count": summary.get("metric_count"),
                "comparison_status": comparison.get("status"),
                "baseline_run_id": comparison.get("baseline_run_id"),
                "improved_metric_count": delta_summary.get("improved_metric_count"),
                "regressed_metric_count": delta_summary.get("regressed_metric_count"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def _run_history(args: argparse.Namespace) -> None:
    history = compact_run_history(Path(args.index_output), limit=int(args.limit))
    print(json.dumps(history, ensure_ascii=False, indent=2))


def _explain_category_classifier(args: argparse.Namespace) -> None:
    model_path = Path(args.model_input)
    output_path = Path(args.output)
    try:
        report = explain_text_classifier(
            _read_json_object(model_path),
            config=TextClassifierExplainabilityConfig(
                top_features_per_class=int(args.top_features_per_class)
            ),
        )
    except ValueError as error:
        raise SystemExit(f"Cannot explain category classifier: {error}") from error
    report = {
        **report,
        "inputs": {"model": str(model_path)},
        "artifacts": {"model": artifact_metadata(model_path, "model", "model_input")},
        "output": str(output_path),
    }
    _write_json(report, output_path)
    _record_stage_result(
        args,
        stage_name="model_explainability.category_classifier",
        artifacts={"explainability_report": output_path},
        payload=report,
    )
    summary = _mapping(report.get("summary"))
    print(
        json.dumps(
            {
                "output": str(output_path),
                "class_count": summary.get("class_count"),
                "feature_count": summary.get("feature_count"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def _create_llm_exploratory_plan(args: argparse.Namespace) -> None:
    output_path = Path(args.output)
    try:
        report = create_llm_exploratory_test_plan(
            LLMExploratoryTestingConfig(
                target_model_name=str(args.target_model_name),
                domain=str(args.domain),
                session_duration_minutes=int(args.session_duration_minutes),
            )
        )
    except ValueError as error:
        raise SystemExit(f"Cannot create LLM exploratory plan: {error}") from error
    report = {**report, "output": str(output_path)}
    _write_json(report, output_path)
    _record_stage_result(
        args,
        stage_name="llm_exploratory_testing",
        artifacts={"llm_exploratory_plan": output_path},
        payload=report,
    )
    summary = _mapping(report.get("summary"))
    print(
        json.dumps(
            {
                "output": str(output_path),
                "charter_count": summary.get("charter_count"),
                "target_model_name": report.get("subject"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def _baseline_run_report(
    *,
    explicit_baseline: str,
    compare_to_latest: bool,
    index_output_path: Path,
) -> Mapping[str, object] | None:
    baseline_path = Path(explicit_baseline) if explicit_baseline else None
    if baseline_path is None and compare_to_latest:
        baseline_path = latest_run_report_path(index_output_path)
    if baseline_path is None or not baseline_path.exists():
        return None
    return _read_json_object(baseline_path)


def _resolve_run_report_input(input_path: Path, use_latest_run: bool) -> Path:
    if use_latest_run:
        latest_path = latest_run_report_path(input_path)
        if latest_path is None:
            raise SystemExit(f"Cannot resolve latest run report from {input_path}")
        return latest_path
    if not input_path.exists():
        raise SystemExit(f"Run report does not exist: {input_path}")
    return input_path


def _parse_report_specs(specs: Sequence[str]) -> dict[str, Path]:
    report_paths: dict[str, Path] = {}
    for spec in specs:
        if "=" not in spec:
            raise SystemExit(f"Invalid --report value {spec!r}. Expected NAME=PATH.")
        name, raw_path = spec.split("=", 1)
        name = name.strip()
        raw_path = raw_path.strip()
        if not name or not raw_path:
            raise SystemExit(f"Invalid --report value {spec!r}. Expected NAME=PATH.")
        report_paths[name] = Path(raw_path)
    return report_paths


def _safe_filename(value: str) -> str:
    safe = "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in value)
    return safe.strip("_") or "report"


def _markdown_title(value: str) -> str:
    return value.replace("_", " ").replace("-", " ").title()


def _path_mapping(value: object) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    for name, path in _mapping(value).items():
        if not isinstance(name, str) or not isinstance(path, (str, Path)):
            continue
        paths[name] = Path(path)
    return paths


def _read_association_validation_folds(folds_dir: Path) -> list[AssociationValidationFold]:
    if not folds_dir.exists():
        raise SystemExit(f"{folds_dir} does not exist. Run ai-test split-datasets first.")

    folds: list[AssociationValidationFold] = []
    for train_path in sorted(folds_dir.glob("fold_*_train.json")):
        fold_name = train_path.name.removesuffix("_train.json")
        validation_path = folds_dir / f"{fold_name}_validation.json"
        if not validation_path.exists():
            raise SystemExit(f"Missing validation file for {train_path}: {validation_path}")
        folds.append(
            AssociationValidationFold(
                fold_index=_fold_index(fold_name),
                train_records=_read_records(train_path),
                validation_records=_read_records(validation_path),
            )
        )

    if not folds:
        raise SystemExit(f"No fold train files found in {folds_dir}.")
    return folds


def _read_text_classification_folds(folds_dir: Path) -> list[TextClassificationValidationFold]:
    if not folds_dir.exists():
        raise SystemExit(
            f"{folds_dir} does not exist. Run ai-test build-classification-dataset first."
        )

    folds: list[TextClassificationValidationFold] = []
    for train_path in sorted(folds_dir.glob("fold_*_train.json")):
        fold_name = train_path.name.removesuffix("_train.json")
        validation_path = folds_dir / f"{fold_name}_validation.json"
        if not validation_path.exists():
            raise SystemExit(f"Missing validation file for {train_path}: {validation_path}")
        folds.append(
            TextClassificationValidationFold(
                fold_index=_fold_index(fold_name),
                train_records=_read_records(train_path),
                validation_records=_read_records(validation_path),
            )
        )

    if not folds:
        raise SystemExit(f"No fold train files found in {folds_dir}.")
    return folds


def _read_input_data_folds(folds_dir: Path) -> list[InputDataFold]:
    if not folds_dir.exists():
        raise SystemExit(f"{folds_dir} does not exist. Run ai-test split-datasets first.")

    folds: list[InputDataFold] = []
    for train_path in sorted(folds_dir.glob("fold_*_train.json")):
        fold_name = train_path.name.removesuffix("_train.json")
        validation_path = folds_dir / f"{fold_name}_validation.json"
        if not validation_path.exists():
            raise SystemExit(f"Missing validation file for {train_path}: {validation_path}")
        folds.append(
            InputDataFold(
                fold_index=_fold_index(fold_name),
                train_records=_read_records(train_path),
                validation_records=_read_records(validation_path),
            )
        )

    if not folds:
        raise SystemExit(f"No fold train files found in {folds_dir}.")
    return folds


def _fold_index(fold_name: str) -> int:
    try:
        return int(fold_name.removeprefix("fold_"))
    except ValueError as error:
        raise SystemExit(f"Cannot parse fold index from {fold_name}") from error


def _read_records(input_path: Path) -> list[dict[str, object]]:
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    if not isinstance(payload, list) or not all(isinstance(item, dict) for item in payload):
        raise SystemExit(f"{input_path} must contain a JSON array of objects.")
    return [dict(item) for item in payload]


def _read_json_object(input_path: Path) -> dict[str, object]:
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise SystemExit(f"{input_path} must contain a JSON object.")
    return dict(payload)


def _write_json(payload: object, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _record_stage_result(
    args: argparse.Namespace,
    *,
    stage_name: str,
    artifacts: Mapping[str, Path],
    payload: Mapping[str, object] | None = None,
    status: str | None = None,
    parameters: Mapping[str, object] | None = None,
) -> None:
    if not bool(getattr(args, "stage_tracking_enabled", True)):
        return
    record_stage_execution(
        StageExecutionTrackingConfig(
            history_output=Path(
                str(getattr(args, "stage_history_output", "data/runs/stage_history.json"))
            ),
            markdown_output=Path(
                str(
                    getattr(
                        args,
                        "stage_history_markdown_output",
                        "data/reports/stage_history.md",
                    )
                )
            ),
            metric_history_markdown_output=Path(
                str(
                    getattr(
                        args,
                        "stage_metric_history_markdown_output",
                        "data/reports/stage_metric_history.md",
                    )
                )
            ),
            max_entries=int(getattr(args, "stage_history_max_entries", 500)),
            recent_entries_per_stage=int(
                getattr(args, "stage_history_recent_entries_per_stage", 5)
            ),
            metric_history_recent_entries_per_stage=int(
                getattr(args, "stage_metric_history_recent_entries_per_stage", 10)
            ),
        ),
        stage_name=stage_name,
        command_name=str(args.command),
        artifacts=artifacts,
        payload=payload,
        status=status,
        parameters=parameters,
    )


def _print_report_summary(
    report: Mapping[str, object],
    output_path: Path,
    extra: Mapping[str, object] | None = None,
) -> None:
    summary = _mapping(report.get("summary"))
    payload: dict[str, object] = {
        "output": str(output_path),
        "status": report.get("status"),
        "check_count": summary.get("check_count"),
        "passed_count": summary.get("passed_count"),
        "failed_count": summary.get("failed_count"),
    }
    if extra:
        payload.update(dict(extra))
    failed_checks = _failed_checks_for_cli(report)
    if failed_checks:
        payload["failed_checks"] = failed_checks
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def _failed_checks_for_cli(report: Mapping[str, object]) -> list[dict[str, object]]:
    checks = report.get("checks")
    if not isinstance(checks, Sequence) or isinstance(checks, (str, bytes)):
        return []

    failed_checks: list[dict[str, object]] = []
    for check in checks:
        if not isinstance(check, Mapping) or check.get("status") != "failed":
            continue
        failed_checks.append(
            {
                "id": check.get("id"),
                "severity": check.get("severity"),
                "message": check.get("message"),
                "observed": check.get("observed"),
                "expected": check.get("expected"),
                "diagnostics": _diagnostics_for_cli(check.get("diagnostics")),
            }
        )
    return failed_checks


def _diagnostics_for_cli(value: object) -> object:
    if not isinstance(value, Mapping):
        return None

    diagnostics: dict[str, object] = {}
    for key in (
        "summary",
        "failure_columns",
        "contract",
        "suggested_actions",
        "failed_child_checks",
    ):
        if key in value:
            diagnostics[key] = value[key]

    fields = value.get("fields")
    if isinstance(fields, Mapping):
        diagnostics["fields"] = {
            str(field): _diagnostic_field_for_cli(field_diagnostics)
            for field, field_diagnostics in fields.items()
        }

    sample_records = value.get("sample_records")
    if isinstance(sample_records, Sequence) and not isinstance(sample_records, (str, bytes)):
        diagnostics["sample_records"] = [item for item in sample_records[:3]]

    return diagnostics or dict(value)


def _diagnostic_field_for_cli(value: object) -> object:
    if not isinstance(value, Mapping):
        return value

    field_summary: dict[str, object] = {}
    for key in ("rule", "affected_record_count", "value_summary", "breakdown"):
        if key in value:
            field_summary[key] = value[key]
    sample_records = value.get("sample_records")
    if isinstance(sample_records, Sequence) and not isinstance(sample_records, (str, bytes)):
        field_summary["sample_records"] = [item for item in sample_records[:3]]
    return field_summary


def _mapping(value: object) -> Mapping[str, object]:
    return value if isinstance(value, Mapping) else {}


def _exit_if_report_failed(report: Mapping[str, object]) -> None:
    if report.get("status") == "failed":
        raise SystemExit(1)


def _resolve_cookies(cookies: str | None, cookies_file: str | None, env_name: str) -> str:
    if cookies:
        return cookies

    if cookies_file:
        file_value = _read_cookies_file(Path(cookies_file))
        if file_value:
            return file_value

    env_value = os.getenv(env_name)
    if env_value:
        return env_value

    raise SystemExit(
        "Missing cookies. Put an authenticated Cookie header into "
        f"{cookies_file}, set {env_name}, or pass --cookies."
    )


def _read_cookies_file(path: Path) -> str | None:
    if not path.exists():
        return None

    lines = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    return " ".join(lines) if lines else None


if __name__ == "__main__":
    raise SystemExit(main())

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
from ai_testing.input_data_testing import InputDataFold, InputDataTestConfig, test_input_data
from ai_testing.ml_model_testing import (
    AssociationMLModelTestConfig,
    test_association_ml_model,
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

    args = parser.parse_args(parser_argv)
    if args.command == "export-kosik":
        records = _fetch_kosik(args)
        _write_records(records, Path(args.output))
        return 0

    if args.command == "export-rohlik":
        records = _fetch_rohlik(args)
        _write_records(records, Path(args.output))
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
        _write_records(records, Path(args.output))
        return 0

    if args.command == "export-sample":
        _write_records(sample_grocery_order_item_records(), Path(args.output))
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
        "note": "Trained on the supervised training split only. Hold-out test data is not used.",
    }
    _write_json(model, output_path)
    summary = model["summary"]
    print(
        json.dumps(
            {
                "input": str(input_path),
                "output": str(output_path),
                "model_type": model["model_type"],
                "algorithm": model["algorithm"],
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
    summary = report["summary"]
    print(
        json.dumps(
            {
                "folds_dir": str(folds_dir),
                "output": str(output_path),
                "fold_count": summary["fold_count"],
                "mean_accuracy": summary["mean_accuracy"],
                "std_accuracy": summary["std_accuracy"],
                "mean_macro_f1": summary["mean_macro_f1"],
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
                "macro_f1": test_summary["macro_f1"],
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
        "output": str(output_path),
    }
    _write_json(report, output_path)
    summary = report["summary"]
    print(
        json.dumps(
            {
                "output": str(output_path),
                "status": report["status"],
                "check_count": summary["check_count"],
                "passed_count": summary["passed_count"],
                "failed_count": summary["failed_count"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    _exit_if_report_failed(report)


def _test_ml_model(args: argparse.Namespace) -> None:
    model_path = Path(args.model_input)
    validation_report_path = Path(args.validation_report_input)
    test_report_path = Path(args.test_report_input)
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
            "test_dataset": str(args.test_dataset_input),
        },
        "output": str(output_path),
    }
    _write_json(report, output_path)
    summary = report["summary"]
    print(
        json.dumps(
            {
                "output": str(output_path),
                "status": report["status"],
                "check_count": summary["check_count"],
                "passed_count": summary["passed_count"],
                "failed_count": summary["failed_count"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    _exit_if_report_failed(report)


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

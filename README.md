# AI Grocery Testing

Python project for preparing grocery order data for AI testing practice.

Current scope:

- collect order-item data from Kosik and Rohlik;
- normalize both shops into one raw dataset;
- preprocess raw data into a compact modeling dataset;
- create training, validation, and test datasets;
- build supervised classification datasets;
- train, validate, and test a supervised category classifier;
- train an association rules model on training data;
- validate association rules on k-fold validation data;
- test the selected association model on the hold-out test dataset;
- test input data quality and split integrity;
- test ML model reports against acceptance criteria;
- generate a data quality report.

Supervised classification and AI test scenarios will be added in the next project stage.

## Tech Stack

Current stack:

- Python 3.10+
- Python standard library
- `argparse` CLI exposed as `ai-test`
- JSON / JSONC configuration
- Apriori association rules implementation
- `ruff`, `mypy`, `pre-commit`
- Git

Planned ML stack:

- `pandas` / `numpy` for dataset work
- `scikit-learn` for supervised classification
- `matplotlib` / `seaborn` for analysis charts
- `joblib` for model persistence
- `pytest` for pipeline checks

## Project Structure

```text
config/
  defaults.jsonc              # non-secret runtime defaults
src/ai_testing/
  cli.py                      # ai-test commands
  config.py                   # JSONC config loader
  data_acquisition/
    common.py                 # shared HTTP/data helpers
    kosik.py                  # Kosik adapter and mapping
    rohlik.py                 # Rohlik adapter and mapping
  data_preprocessing/
    grocery.py                # feature selection and quality report
  data_splitting/
    grocery.py                # train/test and k-fold validation split
  classification_preprocessing/
    grocery.py                # supervised text/label dataset builder
  input_data_testing/
    grocery.py                # input data and split integrity tests
  model_training/
    text_classifier.py        # supervised category classifier training
    association.py            # association rules training
  model_validation/
    text_classifier.py        # k-fold classifier validation
    association.py            # k-fold association evaluation
  model_testing/
    text_classifier.py        # hold-out classifier test report
    association.py            # hold-out association test report
  ml_model_testing/
    association.py            # model acceptance tests
  sample_data.py              # synthetic data for smoke checks
data/                         # generated locally, ignored by git
secrets/                      # local cookies, ignored by git
```

## Configuration

Runtime defaults are in:

```text
config/defaults.jsonc
```

The config controls shop URLs, API paths, pagination limits, output paths, enrichment flags, and the
preprocessing feature whitelist. CLI arguments override config values.

## Secrets

Create local cookie files:

```text
secrets/kosik.cookies.txt
secrets/rohlik.cookies.txt
```

Each file should contain one authenticated browser `Cookie` header value. Do not commit secrets or
exported datasets.

## Commands

Install:

```powershell
python -m pip install -e ".[dev]"
```

Run synthetic sample export:

```powershell
ai-test export-sample
```

Run data acquisition:

```powershell
ai-test export-all
```

Run shops separately:

```powershell
ai-test export-kosik
ai-test export-rohlik
```

Run preprocessing:

```powershell
ai-test preprocess-data
```

Create training, validation, and test datasets:

```powershell
ai-test split-datasets
```

Build supervised classification datasets:

```powershell
ai-test build-classification-dataset
```

Train and evaluate the supervised category classifier:

```powershell
ai-test train-category-classifier
ai-test validate-category-classifier
ai-test test-category-classifier
```

Train association rules on training data:

```powershell
ai-test train-associations
```

Validate association rules across folds:

```powershell
ai-test validate-associations
```

Run the final hold-out test:

```powershell
ai-test test-associations
```

Run AI testing stages:

```powershell
ai-test test-input-data
ai-test test-ml-model
```

Use a custom config:

```powershell
ai-test --config config/defaults.jsonc export-all
```

Useful acquisition options:

```powershell
ai-test export-all --kosik-order-page-limit 150 --rohlik-order-page-limit 150
ai-test export-all --no-product-enrichment
ai-test export-all --include-product-content
ai-test export-all --no-kosik-include-archived-orders
ai-test export-all --no-rohlik-include-archived-orders
```

Run code checks:

```powershell
ruff format --check .
ruff check --no-cache .
mypy --no-incremental src
```

## Data Acquisition

Data acquisition reads authenticated order history from Kosik and Rohlik APIs. Each purchased product
line is mapped into a shared raw order-item schema. Product enrichment can add category, description,
ingredient, and package metadata when available.

Raw data is written under `data/raw/`.

## Data Preprocessing

Data preprocessing converts raw order items into a compact dataset for future modeling.

It keeps a feature whitelist from `config/defaults.jsonc`, removes technical identifiers and noisy
raw fields, and derives calendar/category features:

- `basket_id`
- `order_month`
- `order_week_of_year`
- `order_day_of_week`
- `order_is_weekend`
- `order_quarter`
- `is_gluten_free`
- `product_group`
- `meat_type`
- `quality_level`

`category_path`, product ids, order ids, URLs, `quantity`, `quantity_delivered`, `price_total`, and
`product_enriched` are removed from the processed dataset.

If `quantity_ordered` is missing, preprocessing fills it from delivered quantity, then from the raw
generic quantity field.

Processed data and the quality report are written under `data/processed/`.

## Dataset Splitting

Dataset splitting creates the datasets needed before supervised model training.

The split is group-aware by `basket_id`, so rows from the same basket stay in the same dataset. This
prevents leakage between training, validation, and test data.

Default strategy:

- hold out a final test dataset;
- use the remaining data for k-fold cross-validation;
- stratify groups by `shop`;
- use a fixed random seed for repeatability.

```powershell
ai-test split-datasets
ai-test split-datasets --n-splits 5 --test-size 0.2
```

Split artifacts are written under `data/splits/` and are ignored by git.

## Supervised Classification

The supervised model predicts product `main_category` from product text. By default, the text is built
from `product_name` and `brand`; category fields are not used as features to avoid leakage.

```powershell
ai-test build-classification-dataset
ai-test train-category-classifier
ai-test validate-category-classifier
ai-test test-category-classifier
```

Artifacts are written to `data/classification/category/`, `data/models/category_classifier.json`,
`data/validation/category_classifier_validation_report.json`, and
`data/testing/category_classifier_test_report.json`.

Validation accuracy, precision, recall, and F1 are used while evaluating and tuning the model. Test
metrics are the final hold-out result after the model and parameters are selected.

## Model Training

The project currently has two model families:

- supervised text classification for product categories;
- unsupervised association rules for market basket analysis.

Association training uses `data/splits/train_validation.json`, which excludes the hold-out test set.
For fold-specific association experiments, pass a fold training file explicitly:

```powershell
ai-test train-associations
ai-test train-associations --input data/splits/folds/fold_01_train.json
```

The model artifact is written under `data/models/` and contains frequent itemsets, association rules,
and metrics such as support, confidence, lift, leverage, and conviction.

## Model Validation

Association validation trains rules on each fold training dataset and evaluates those rules on the
matching validation dataset.

```powershell
ai-test validate-associations
```

The validation report is written under `data/validation/` and includes fold-level and aggregate
metrics:

- validation confidence;
- validation lift;
- antecedent coverage;
- hit rate;
- train-vs-validation confidence gap;
- stable rule count.

## Model Testing

Final testing evaluates the already trained association model on `data/splits/test.json`.

```powershell
ai-test test-associations
```

The test report is written under `data/testing/`. Use this report once after validation/tuning is
finished. It includes test confidence, test lift, coverage, hit rate, train-vs-test gaps, stable rule
count, and strongest test rules.

## Input Data Testing

Input data testing checks model-ready data and split artifacts:

- required feature columns exist;
- protected identifiers are absent;
- critical fields are not missing;
- category/product coverage stays within thresholds;
- dates, calendar features, numeric values, and duplicates are valid;
- train/test/fold baskets do not leak across datasets.

```powershell
ai-test test-input-data
```

The report is written to `data/testing/input_data_test_report.json`.

## ML Model Testing

ML model testing checks the trained association model and the validation/test reports against
acceptance criteria.

```powershell
ai-test test-ml-model
```

The report is written to `data/testing/ml_model_test_report.json`. It verifies model identity,
training/test separation, forbidden feature usage, validation metrics, final test metrics, and
validation-vs-test stability.

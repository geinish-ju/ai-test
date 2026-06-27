# AI Grocery Testing

Python project for preparing grocery order data for AI testing practice.

Current scope:

- collect order-item data from Kosik and Rohlik;
- normalize both shops into one raw dataset;
- preprocess raw data into a compact modeling dataset;
- create training, validation, and test datasets;
- train an association rules model on training data;
- validate association rules on k-fold validation data;
- test the selected association model on the hold-out test dataset;
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
  model_training/
    association.py            # association rules training
  model_validation/
    association.py            # k-fold association evaluation
  model_testing/
    association.py            # hold-out association test report
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

## Model Training

The current model is association rules for market basket analysis. It is unsupervised and is not the
same as the later supervised classification task.

By default, training uses `data/splits/train_validation.json`, which excludes the hold-out test set.
For fold-specific experiments, pass a fold training file explicitly:

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

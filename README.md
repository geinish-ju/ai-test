# AI Grocery Testing Framework

Python project for preparing grocery order data for AI testing practice.

Current scope:

- collect order-item data from Kosik and Rohlik;
- normalize both shops into one raw dataset;
- preprocess raw data into a compact modeling dataset;
- generate a data quality report.

Model training and AI test scenarios will be added in the next project stage.

## Tech Stack

Current stack:

- Python 3.10+
- Python standard library
- `argparse` CLI exposed as `ai-test`
- JSON / JSONC configuration
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

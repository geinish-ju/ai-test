# AI Grocery Testing

Python project for practicing AI testing on grocery order data.

The project builds a small end-to-end ML quality pipeline:

1. acquire order-item data from Kosik and Rohlik;
2. preprocess records into model-ready features;
3. create train, validation, and hold-out test datasets;
4. train and evaluate two model families;
5. test input data and ML model quality;
6. produce a project-level quality decision and run history.

## Tech Stack

- Python 3.10+
- `argparse` CLI exposed as `ai-test`
- JSON / JSONC configuration
- `pandas`, `pyarrow`, `pandera`
- `scikit-learn`, `joblib`
- Apriori-style association rules
- `pytest`, `ruff`, `mypy`
- GitHub Actions CI
- optional MLflow and DVC integration

## Project Structure

```text
config/
  defaults.jsonc
docs/
  index.md
  project_overview.md
  pipeline_walkthrough.md
  testing_strategy.md
  metrics_reference.md
  model_quality_decision.md
  configuration_reference.md
  interview_prep.md
src/ai_testing/
  cli/
    parser.py
    commands/
  core/
  data/
    acquisition/
    preprocessing/
    splitting/
    classification_preprocessing/
  models/
    training/
    validation/
    testing/
    explainability/
  quality/
    input_data/
    ml_model/
    project/
    drift/
    llm_exploratory/
  observability/
    reporting/
    run_tracking/
  cli_common.py
  cli_reporting.py
tests/
```

Generated data, reports, models, MLflow DB files, and secrets are local artifacts and are ignored by
git.

## Install

```powershell
python -m pip install -e ".[dev]"
```

Optional MLOps tools:

```powershell
python -m pip install -e ".[mlops]"
```

## Secrets

Create local cookie files:

```text
secrets/kosik.cookies.txt
secrets/rohlik.cookies.txt
```

Each file should contain one authenticated browser `Cookie` header value. Do not commit secrets or
exported personal data.

## Main Pipeline

Synthetic smoke data:

```powershell
ai-test export-sample
ai-test preprocess-data --input data/raw/sample_order_items.json
```

Real data acquisition:

```powershell
ai-test export-all
```

Data preparation:

```powershell
ai-test preprocess-data
ai-test split-datasets
ai-test build-classification-dataset
```

Supervised category classifier:

```powershell
ai-test train-category-classifier
ai-test validate-category-classifier
ai-test test-category-classifier
ai-test test-category-ml-model
```

Association rules:

```powershell
ai-test train-associations
ai-test validate-associations
ai-test test-associations
ai-test test-ml-model
```

Input data and project quality:

```powershell
ai-test test-input-data
ai-test run-quality-gates
```

Run tracking and reports:

```powershell
ai-test track-run --run-name "candidate-review"
ai-test test-drift
ai-test generate-markdown-reports
ai-test run-history
```

## Reports

Important artifacts:

- `data/testing/input_data_test_report.json`
- `data/testing/category_ml_model_test_report.json`
- `data/testing/ml_model_test_report.json`
- `data/testing/project_quality_report.json`
- `data/testing/drift_test_report.json`
- `data/runs/<run_id>/run_report.json`
- `data/reports/project_quality.md`
- `data/reports/latest_run_report.md`
- `data/reports/stage_metric_history.md`

The main acceptance artifact is `project_quality_report`. It contains:

- `accepted`, `needs_review`, or `rejected` decision;
- blockers and warnings;
- recommended actions;
- links to child quality reports.

Start with [docs/index.md](docs/index.md) for the full documentation package. A Russian copy is
available at [docs/ru/index.md](docs/ru/index.md).

Key pages:

- [Project overview](docs/project_overview.md)
- [Pipeline walkthrough](docs/pipeline_walkthrough.md)
- [Testing strategy](docs/testing_strategy.md)
- [Metrics reference](docs/metrics_reference.md)
- [Model quality decision guide](docs/model_quality_decision.md)
- [Configuration reference](docs/configuration_reference.md)
- [Interview preparation](docs/interview_prep.md)

## Quality Checks

Run locally:

```powershell
ruff format --check .
ruff check --no-cache .
mypy --no-incremental src
pytest
```

The same checks run in GitHub Actions for Python 3.10, 3.11, 3.12, and 3.13.

## MLOps

MLflow uses a local SQLite backend by default:

```text
sqlite:///data/mlflow/mlflow.db
```

Example:

```powershell
ai-test track-run --run-name "mlops-run" --mlflow-tracking
```

DVC can track selected generated datasets, model artifacts, and reports:

```powershell
dvc init
ai-test track-run --run-name "dvc-run" --dvc-versioning
```

Raw exports are not selected for DVC by default.

## Configuration

Runtime defaults live in [config/defaults.jsonc](config/defaults.jsonc).

Use a custom config:

```powershell
ai-test --config config/defaults.jsonc export-all
```

Common acquisition options:

```powershell
ai-test export-all --kosik-order-page-limit 150 --rohlik-order-page-limit 150
ai-test export-all --no-product-enrichment
ai-test export-all --include-product-content
```

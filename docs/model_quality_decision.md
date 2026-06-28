# Model Quality Decision Guide

This project treats model acceptance as a documented quality decision, not as a single accuracy
number.

## Primary Evidence

Use these artifacts together:

| Evidence | Purpose |
|---|---|
| `data/testing/project_quality_report.json` and `.md` | Main accept/review/reject decision |
| `data/testing/input_data_test_report.json` and `.md` | Input data contract, split integrity, leakage, missing values |
| `data/testing/category_ml_model_test_report.json` and `.md` | Supervised classifier acceptance checks |
| `data/testing/ml_model_test_report.json` and `.md` | Association model acceptance checks |
| `data/testing/drift_test_report.json` and `.md` | Regression and distribution drift evidence |
| `data/runs/<run_id>/run_report.json` and `.md` | Versioned run evidence, hashes, metric deltas |
| `data/explainability/category_classifier_explanation_report.json` and `.md` | Token-level sanity check for classifier behavior |

## Decision Levels

| Outcome | Meaning | Action |
|---|---|---|
| `accepted` | All aggregated quality reports passed. | Keep the report as the acceptance record. |
| `needs_review` | No critical blocker failed, but major/minor checks failed. | Review the risk and document the decision before accepting. |
| `rejected` | At least one critical quality gate failed. | Do not accept the model; fix the cause and rerun the pipeline. |

## Reject Immediately When

- Train, validation, and test datasets are not separated correctly.
- The model was trained on the hold-out test dataset.
- Identifier, target, or category-derived leakage appears in model features.
- Input data has invalid critical values, protected identifiers, or split leakage.
- Hold-out test metrics fail critical acceptance thresholds.
- Validation metrics are good but hold-out test metrics drop beyond the configured tolerance.
- Drift or metric regression exceeds the configured gate for the current candidate.
- Required evidence is missing, stale, or cannot be linked to the current data/model artifacts.

## Review Before Accepting When

- Macro precision, macro recall, or macro F1 is weaker than weighted metrics.
- Some classes have low support or unstable per-class metrics.
- Association rules pass average thresholds but coverage or stable-rule count is weak.
- Explainability shows tokens that look like leakage, store-specific artifacts, or accidental labels.
- Drift is present but below the hard rejection threshold.

## Metric Interpretation

Accuracy answers: how often is the classifier correct overall?

Precision answers: when the model predicts a category, how often is that prediction correct?

Recall answers: for real items of a category, how many does the model find?

Macro metrics give every class equal weight and expose weak minority classes.

Weighted metrics follow class frequency and can look strong even when rare classes are poor.

For this project, prefer accepting a classifier only when:

- hold-out accuracy meets the configured threshold;
- macro precision, macro recall, and macro F1 meet the configured thresholds;
- validation-to-test deltas are within tolerance;
- no leakage or dataset separation checks fail.

## Validation vs Test

Use validation and k-fold cross-validation to tune data preparation, features, and model parameters.

Use the hold-out test report as final evidence. Do not tune thresholds or model parameters after
looking at hold-out test results unless you create a new candidate and rerun the full process.

## Recommended Review Workflow

```powershell
ai-test test-input-data
ai-test test-category-ml-model
ai-test test-ml-model
ai-test test-drift
ai-test run-quality-gates --drift-report data/testing/drift_test_report.json
ai-test track-run --run-name "candidate-review"
ai-test generate-markdown-reports
```

Then inspect:

1. `data/reports/project_quality.md`
2. `data/reports/latest_run_report.md`
3. `data/reports/stage_metric_history.md`
4. model-specific Markdown reports for failed or warning checks

The decision should be based on the main project quality report first, then explained with the
specific child reports that created blockers or warnings.

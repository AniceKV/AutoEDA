# Executive Summary Report

## 1. Executive Overview

The automated EDA analyzed `dataset_2191_sleep.csv`, a small biological dataset containing 62 observations and 8 raw columns. The stated modeling objective is supervised regression: predicting continuous `total_sleep`.

Key conclusions:

- The dataset is small, with only 62 rows, so statistical estimates, correlations, and model validation results may be unstable.
- Several numeric-looking fields were stored as strings and used placeholder tokens such as `?`.
- The raw `body_weight` and `brain_weight` variables are highly right-skewed and contain substantial IQR-defined outlier rates.
- All evaluated predictors showed statistically significant Pearson associations with `total_sleep` at the unadjusted 0.05 level.
- The strongest reported association was between `gestation_sleep_ratio` and `total_sleep`, with correlation `-0.6461`.
- However, `gestation_sleep_ratio` is defined as `gestation_time / total_sleep`. Because it directly uses the target, it is target-derived and should be treated as a leakage-prone feature. It should not be used for prospective prediction of `total_sleep`.
- The strongest non-target-derived relationships include negative associations between sleep duration and sleep exposure, gestation time, transformed body or brain weight, danger, and predation indices.
- A transparent median baseline and regularized regression should be established before comparing constrained tree-based models.
- Any final model should use pipeline-based preprocessing and repeated cross-validation, with particular attention to influential observations and leakage control.

---

## 2. Dataset Scope and Structure

### 2.1 Dataset profile

| Attribute | Value |
|---|---:|
| Dataset | `dataset_2191_sleep.csv` |
| Rows | 62 |
| Raw columns | 8 |
| Processed modeling columns | 12 |
| Target | `total_sleep` |
| Problem type | Supervised regression |
| Target definition | Continuous total sleep duration parsed from `total_sleep` |

The processed dataset contains the 8 original fields plus 4 engineered features:

- `brain_body_ratio`
- `log_body_weight`
- `log_brain_weight`
- `gestation_sleep_ratio`

### 2.2 Raw schema

| Column | Raw type | Processed type | Cardinality | Key observations |
|---|---|---|---:|---|
| `body_weight` | float64 | float64 | 60 | Range 0.01 to 6654.00; median 3.34; skewness 6.56 |
| `brain_weight` | float64 | float64 | 59 | Range 0.14 to 5712.00; median 17.25; skewness 5.07 |
| `max_life_span` | string | float64 | 48 | Placeholder values present; median imputation applied |
| `gestation_time` | string | float64 | 50 | Placeholder values present; median imputation applied |
| `predation_index` | int64 | int64 | 5 | Range 1 to 5; mean 2.87 |
| `sleep_exposure_index` | int64 | int64 | 5 | Range 1 to 5; mean 2.42 |
| `danger_index` | int64 | int64 | 5 | Range 1 to 5; mean 2.61 |
| `total_sleep` | string | float64 | 45 | Modeling target; placeholder values present; mean imputation applied |

---

## 3. Data Quality Assessment

### 3.1 Missing and placeholder values

The metadata profile reports no missing values because the missing entries were represented as strings rather than native null values. The processing metrics identify four missing values in each of the following fields:

| Column | Missing before processing | Missing after processing | Imputation method | Fill value |
|---|---:|---:|---|---:|
| `max_life_span` | 4 | 0 | Median | 15.1 |
| `gestation_time` | 4 | 0 | Median | 79.0 |
| `total_sleep` | 4 | 0 | Mean | 10.5328 |

The pipeline replaced the following tokens with missing values:

- `?`
- Empty strings
- `NA`
- `N/A`
- `null`
- `None`
- `nan`

No missing values remained after processing.

### 3.2 Imputation rules

The pipeline used the following rules:

- Median imputation for numeric variables with skewness greater than 1 or less than -1.
- Mean imputation for numeric variables with skewness between -1 and 1.
- Mode imputation, or `Unknown` where no mode exists, for categorical variables.
- Explicit conversion of numeric-looking strings to floating-point values.

The imputation treatment was:

| Column | Skewness | Method |
|---|---:|---|
| `body_weight` | 6.5636 | Median |
| `brain_weight` | 5.0716 | Median |
| `max_life_span` | 2.0138 | Median |
| `gestation_time` | 1.6835 | Median |
| `predation_index` | 0.2295 | Mean |
| `sleep_exposure_index` | 0.6795 | Mean |
| `danger_index` | 0.3777 | Mean |
| `total_sleep` | 0.2012 | Mean |

### 3.3 Outlier profile

Outliers were identified using the IQR rule:

`below Q1 - 1.5 * IQR` or `above Q3 + 1.5 * IQR`

| Column | Outlier count | Outlier percentage |
|---|---:|---:|
| `body_weight` | 10 | 16.13% |
| `brain_weight` | 9 | 14.52% |
| `max_life_span` | 2 | 3.23% |
| `gestation_time` | 3 | 4.84% |
| `predation_index` | 0 | 0.00% |
| `sleep_exposure_index` | 0 | 0.00% |
| `danger_index` | 0 | 0.00% |
| `total_sleep` | 0 | 0.00% |
| `brain_body_ratio` | 2 | 3.23% |
| `log_body_weight` | 0 | 0.00% |
| `log_brain_weight` | 0 | 0.00% |
| `gestation_sleep_ratio` | 4 | 6.45% |

The primary data quality concern is the extreme skew and outlier concentration in body and brain weight. These values may materially affect Pearson correlations and model coefficients. Logarithmic transformations reduce the measured outlier burden for the transformed variables, although the underlying observations still require domain review.

---

## 4. Distributional Findings

### 4.1 Body weight

- Range: `0.01` to `6654.00`
- Mean: `198.79`
- Median: `3.34`
- Skewness: `6.56`
- IQR-defined outliers: 10, or `16.13%`

The large difference between the mean and median indicates a highly right-skewed distribution. The automated analysis recommends a logarithmic transformation.

### 4.2 Brain weight

- Range: `0.14` to `5712.00`
- Mean: `283.13`
- Median: `17.25`
- Skewness: `5.07`
- IQR-defined outliers: 9, or `14.52%`

Brain weight is also highly right-skewed. A log transformation is recommended to stabilize its scale and reduce the influence of extreme values.

### 4.3 Index variables

The three index variables each have five distinct levels from 1 to 5:

| Variable | Mean | Median | Range |
|---|---:|---:|---:|
| `predation_index` | 2.87 | 3.00 | 1 to 5 |
| `sleep_exposure_index` | 2.42 | 2.00 | 1 to 5 |
| `danger_index` | 2.61 | 2.00 | 1 to 5 |

These variables are ordinal in meaning, although they were processed as integer-valued numeric variables. Their use in linear models should therefore be evaluated carefully, since a one-unit increase may not represent an equal substantive change across all levels.

---

## 5. Correlation Analysis

The reported correlations are Pearson correlations computed after processing and imputation.

### 5.1 Target correlations

| Feature | Correlation with `total_sleep` | Direction and strength |
|---|---:|---|
| `gestation_sleep_ratio` | -0.6461 | Strongest absolute association |
| `sleep_exposure_index` | -0.5941 | Moderate to strong negative |
| `gestation_time` | -0.5648 | Moderate to strong negative |
| `log_brain_weight` | -0.5504 | Moderate to strong negative |
| `danger_index` | -0.5487 | Moderate to strong negative |
| `log_body_weight` | -0.5063 | Moderate negative |
| `max_life_span` | -0.4079 | Moderate negative |
| `predation_index` | -0.3815 | Moderate negative |
| `brain_weight` | -0.3570 | Moderate negative |
| `body_weight` | -0.3066 | Weak to moderate negative |
| `brain_body_ratio` | 0.2511 | Weak positive |

The general pattern is that higher size, gestation, exposure, danger, and predation values are associated with lower total sleep in this sample. These are associations only and do not establish causal effects.

### 5.2 Important target leakage concern

`gestation_sleep_ratio` was engineered as:

`gestation_time / total_sleep`

It therefore incorporates the target variable directly. Its reported correlation of `-0.6461` is not an independent predictor-target relationship. The ratio is mathematically coupled to the target and is inappropriate for a model intended to predict `total_sleep` before the outcome is observed.

Recommended treatment:

- Exclude `gestation_sleep_ratio` from prospective modeling.
- Retain it only for descriptive analysis or post-outcome exploratory analysis.
- If a ratio is required for deployment, redesign it so that it uses only variables available before prediction.
- Recalculate feature rankings and model comparisons after removing this feature.

This is the most important methodological issue identified in the artifact outputs.

### 5.3 Strong predictor interrelationships

Several predictors are strongly correlated with each other:

| Feature pair | Correlation |
|---|---:|
| `body_weight` and `brain_weight` | 0.9342 |
| `log_body_weight` and `log_brain_weight` | 0.9303 |
| `predation_index` and `danger_index` | 0.9160 |
| `gestation_time` and `gestation_sleep_ratio` | 0.8902 |
| `gestation_time` and `log_body_weight` | 0.8263 |
| `brain_weight` and `gestation_sleep_ratio` | 0.8228 |
| `gestation_time` and `log_brain_weight` | 0.8061 |
| `sleep_exposure_index` and `danger_index` | 0.7872 |

These relationships indicate substantial multicollinearity and redundancy. In particular:

- Body and brain weight should not necessarily be included in raw and log-transformed forms simultaneously in a simple linear model.
- Predation, danger, and sleep exposure indices may represent overlapping ecological risk constructs.
- Coefficient interpretation may be unstable when highly correlated predictors are included together.
- Regularization, feature selection, dimensionality reduction, or compact domain-informed feature sets should be considered.

---

## 6. Statistical Hypothesis Testing

The pipeline applied Pearson correlation tests at significance level `alpha = 0.05`.

### 6.1 Results

| Feature | Pearson correlation | P-value | Unadjusted result |
|---|---:|---:|---|
| `body_weight` | -0.3066 | 0.01535 | Significant |
| `brain_weight` | -0.3570 | 0.00439 | Significant |
| `max_life_span` | -0.4079 | 0.00100 | Significant |
| `gestation_time` | -0.5648 | 0.00000173 | Significant |
| `predation_index` | -0.3815 | 0.00222 | Significant |
| `sleep_exposure_index` | -0.5941 | 0.000000357 | Significant |
| `danger_index` | -0.5487 | 0.00000389 | Significant |
| `brain_body_ratio` | 0.2511 | 0.04903 | Significant, borderline |
| `log_body_weight` | -0.5063 | 0.0000268 | Significant |
| `log_brain_weight` | -0.5504 | 0.00000358 | Significant |
| `gestation_sleep_ratio` | -0.6461 | 0.0000000142 | Significant, but target-derived |

All 11 evaluated predictors were marked statistically significant by the pipeline under the unadjusted threshold.

### 6.2 Interpretation limitations

These results require caution:

1. The sample contains only 62 rows.
2. Pearson correlation measures linear association and may be sensitive to extreme observations.
3. Multiple hypothesis tests were performed. The artifact explicitly recommends false discovery rate adjustment for formal inference.
4. The borderline result for `brain_body_ratio`, with `p = 0.04903`, may not remain significant after multiplicity adjustment.
5. Statistical significance does not imply practical importance or causation.
6. The `gestation_sleep_ratio` result is invalid as an independent predictor result because the ratio uses `total_sleep`.
7. Imputed observations contribute to the reported correlations and may affect estimates.

---

## 7. Feature Engineering Highlights

### 7.1 Engineered features

| Feature | Formula | Intended purpose | Correlation with target |
|---|---|---|---:|
| `brain_body_ratio` | `brain_weight / body_weight` | Relative brain investment independent of absolute size | 0.2511 |
| `log_body_weight` | `log1p(body_weight)` | Reduce extreme right skew and large-value influence | -0.5063 |
| `log_brain_weight` | `log1p(brain_weight)` | Stabilize the brain-weight distribution | -0.5504 |
| `gestation_sleep_ratio` | `gestation_time / total_sleep` | Capture relative gestation and sleep relationship | -0.6461 |

### 7.2 Recommended feature handling

- Use `log_body_weight` and `log_brain_weight` as candidates for modeling instead of, or alongside carefully selected versions of, the raw weight fields.
- Evaluate `brain_body_ratio` as a biologically motivated feature, while recognizing that its association is relatively weak and borderline under hypothesis testing.
- Exclude `gestation_sleep_ratio` from any model that predicts total sleep from pre-outcome information.
- Recalculate outlier diagnostics and correlations after finalizing the leakage-safe feature set.
- Use feature selection methods that account for redundancy rather than selecting solely on univariate correlation.

---

## 8. Image Artifact Descriptions

### 8.1 `correlation_matrix.png`

- File type: Image visualization
- File size: 388.47 KB
- Purpose: Visual representation of pairwise correlations among raw and engineered numeric features.
- Supported interpretation from the metrics:
  - Strong positive association between body and brain weight.
  - Strong positive association between predation and danger indices.
  - Strong negative associations between total sleep and several ecological, gestational, and transformed size variables.
  - Strong redundancy among raw and logarithmic size features.

The text correlation matrix in `metrics.json` provides the numeric values underlying this artifact.

### 8.2 `target_interactions.png`

- File type: Image visualization
- File size: 55.81 KB
- Purpose: Visualization of feature interactions or relationships with the target variable.
- The supplied artifact metadata does not provide chart-specific visual details, point-level observations, or interaction statistics. Therefore, no additional visual conclusions should be inferred beyond the numerical correlation and hypothesis-testing results reported above.

---

## 9. Predictive Modeling Blueprint

### 9.1 Modeling objective

Develop a regression model to predict continuous `total_sleep`.

### 9.2 Recommended model sequence

The pipeline recommends evaluating:

1. Median-target baseline regressor.
2. Regularized linear regression.
3. Random Forest Regressor.
4. Gradient Boosting Regressor.
5. HistGradientBoostingRegressor.

The median baseline is essential because it establishes the minimum performance standard for a useful predictive model.

### 9.3 Leakage-safe feature strategy

A leakage-safe initial feature set should include:

- `body_weight` or `log_body_weight`, with careful evaluation of which representation is preferable.
- `brain_weight` or `log_brain_weight`.
- `max_life_span`.
- `gestation_time`.
- `predation_index`.
- `sleep_exposure_index`.
- `danger_index`.
- Potentially `brain_body_ratio`.

The following should be excluded from prospective prediction:

- `gestation_sleep_ratio`, because it directly uses `total_sleep`.

The feature set should remain compact because the dataset has only 62 rows and contains highly correlated predictors.

### 9.4 Preprocessing pipeline

All preprocessing should be performed inside a cross-validation pipeline to avoid information leakage between training and validation folds.

Recommended steps:

1. Replace placeholder strings with missing values.
2. Parse numeric-looking strings explicitly as `float64`.
3. Impute skewed numeric variables with medians.
4. Impute approximately symmetric numeric variables with means, subject to review.
5. Apply `log1p` transformations to heavily skewed positive variables.
6. Standardize predictors for regularized linear models.
7. One-hot encode any remaining categorical variables, if applicable.
8. Generate engineered features only within the pipeline when feature generation depends on learned quantities or imputed values.
9. Exclude target-derived features from the predictive feature matrix.

### 9.5 Validation design

The recommended validation strategy is:

- Repeated 5-fold cross-validation.
- Pipeline-based imputation, transformations, scaling, and encoding.
- Comparison against the median-target baseline.
- Reporting of:
  - Mean absolute error, or MAE.
  - Root mean squared error, or RMSE.
  - R-squared.
  - Variability estimates across repeated folds.

Because there are only 62 observations, a single train-test split would produce highly variable results. Repeated cross-validation is more informative, although its estimates will still have wide uncertainty.

### 9.6 Overfitting controls

Recommended controls include:

- Use regularization for linear models.
- Limit tree depth.
- Impose minimum leaf-size constraints for tree ensembles.
- Avoid repeated tuning against the final evaluation data.
- Use repeated cross-validation rather than relying on one split.
- Investigate influential observations and robust alternatives.
- Prefer stable performance over marginal improvements.
- Avoid overly broad feature engineering relative to the sample size.

### 9.7 Primary evaluation priority

MAE should be emphasized because it is directly interpretable in the units of total sleep and is less sensitive to extreme errors than RMSE. RMSE and R-squared should still be reported, but conclusions should focus on stability across folds and comparison with the baseline rather than on a single best score.

---

## 10. Executive Recommendations

### Immediate priorities

1. **Correct the leakage issue.** Remove `gestation_sleep_ratio` from predictive modeling of `total_sleep`.
2. **Re-run feature analysis without the leakage-prone ratio.** This will provide a valid ranking of candidate predictors.
3. **Review extreme body and brain weight observations.** Confirm whether they are valid biological values or data-entry/unit issues.
4. **Apply multiplicity adjustment** before treating the correlation tests as formal inferential evidence.
5. **Use a compact, leakage-safe pipeline** with preprocessing learned inside each cross-validation fold.
6. **Benchmark against the median baseline** before interpreting any model improvement.
7. **Prefer robust and regularized methods** given the small sample and substantial predictor redundancy.

### Overall assessment

The dataset contains meaningful exploratory signals, particularly negative associations between total sleep and gestation time, sleep exposure, danger, predation, and transformed body or brain weight. However, the evidence is exploratory rather than confirmatory. The combination of small sample size, imputation, extreme skew, influential observations, correlated predictors, multiple testing, and one target-derived engineered feature makes disciplined validation essential.

The most defensible next step is a leakage-safe comparison of a median baseline, regularized regression, and constrained tree ensembles using repeated cross-validation and compact feature sets.
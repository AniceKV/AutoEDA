# Executive EDA & Dataset Summary Report
**Target Directory:** `C:\Users\Anish Kumar Verma\PycharmProjects\AutoEDA\sandbox_run`
**Processed Files:** `agent_plan_log.json`, `agent_state.json`, `bivariate_gender_vs_math_score.png`, `bivariate_gender_vs_reading_score.png`, `bivariate_gender_vs_writing_score.png`, `bivariate_lunch_vs_math_score.png`, `bivariate_lunch_vs_reading_score.png`, `bivariate_lunch_vs_writing_score.png`, `bivariate_math_score_vs_reading_score.png`, `bivariate_math_score_vs_writing_score.png`, `bivariate_parental_level_of_education_vs_math_score.png`, `bivariate_parental_level_of_education_vs_reading_score.png`, `bivariate_parental_level_of_education_vs_writing_score.png`, `bivariate_race_ethnicity_vs_math_score.png`, `bivariate_race_ethnicity_vs_reading_score.png`, `bivariate_race_ethnicity_vs_writing_score.png`, `bivariate_reading_score_vs_writing_score.png`, `bivariate_test_preparation_course_vs_math_score.png`, `bivariate_test_preparation_course_vs_reading_score.png`, `bivariate_test_preparation_course_vs_writing_score.png`, `categorical_association_matrix.png`, `correlation_matrix.png`, `current_df.csv`, `dist_gender.png`, `dist_lunch.png`, `dist_math_score.png`, `dist_parental_level_of_education.png`, `dist_race_ethnicity.png`, `dist_reading_score.png`, `dist_test_preparation_course.png`, `dist_writing_score.png`, `eda_report.html`, `metadata_profile.json`, `metrics.json`
**Excluded Files:** `generated_analysis.py` (Script excluded from summary)

---

## 1. Dataset Overview
- **Dataset Identifier:** `StudentsPerformance.csv`
- **Dimensions:** `1000` rows x `8` columns
- **Target Variable:** `writing score`
- **Data Quality:** No missing values detected in raw profile.

---

## 1.5 Full Column Statistics
| Column | Type | Missing % | Unique % | Mean | Median | Std | Skew | Kurtosis |
|---|---|---|---|---|---|---|---|---|
| `gender` | `object` | 0.0% | 0.2% | N/A | N/A | N/A | N/A | N/A |
| `race/ethnicity` | `object` | 0.0% | 0.5% | N/A | N/A | N/A | N/A | N/A |
| `parental level of education` | `object` | 0.0% | 0.6% | N/A | N/A | N/A | N/A | N/A |
| `lunch` | `object` | 0.0% | 0.2% | N/A | N/A | N/A | N/A | N/A |
| `test preparation course` | `object` | 0.0% | 0.2% | N/A | N/A | N/A | N/A | N/A |
| `math score` | `int64` | 0.0% | 8.1% | 66.09 | 66.0 | 15.16 | -0.28 | 0.27 |
| `reading score` | `int64` | 0.0% | 7.2% | 69.17 | 70.0 | 14.6 | -0.26 | -0.07 |
| `writing score` | `int64` | 0.0% | 7.7% | 68.05 | 69.0 | 15.2 | -0.29 | -0.03 |

---

## 2. Data Imputation & Preprocessing
**Rules Applied:**
- Standardized missing string placeholders ('?', 'NA', 'N/A', 'null') to NaN.
- Numeric columns with skewness > 1.0 or < -1.0 use median imputation.
- Numeric columns with skewness between -1.0 and 1.0 use mean imputation.
- Categorical/String columns use mode imputation with 'Unknown' fallback.

No columns required imputation this run.

---

## 3. Outlier Analysis (IQR Method)
| Column | Outlier Count | Outlier Percentage | Bounds (Lower / Upper) |
|---|---|---|---|
| `math score` | 8 | 0.8% | [27.0, 107.0] |
| `reading score` | 6 | 0.6% | [29.0, 109.0] |
| `writing score` | 5 | 0.5% | [25.875, 110.875] |

---

## 4. Derived Domain Attributes & Composite Metrics
No custom derived domain metrics synthesized during this run.

---

## 5. Statistical Hypothesis Testing & Key Predictors
- **Statistically Significant Predictors:** `gender`, `race/ethnicity`, `parental level of education`, `lunch`, `test preparation course`, `math score`, `reading score`
_Detailed effect sizes unavailable -- `ranked_significant_details` missing from metrics.json._

---

## 6. Redundancy & Multicollinearity Analysis
No high-correlation or cross-type redundant feature pairs detected (threshold: |r| or Eta >= 0.85).

---

## 7. Generated Visual Artifacts
- **`bivariate_gender_vs_math_score.png`** (54.12 KB) -- Relationship between `gender` and `math score`.
- **`bivariate_gender_vs_reading_score.png`** (63.27 KB) -- Relationship between `gender` and `reading score`.
- **`bivariate_gender_vs_writing_score.png`** (62.33 KB) -- Relationship between `gender` and `writing score`.
- **`bivariate_lunch_vs_math_score.png`** (56.78 KB) -- Relationship between `lunch` and `math score`.
- **`bivariate_lunch_vs_reading_score.png`** (66.72 KB) -- Relationship between `lunch` and `reading score`.
- **`bivariate_lunch_vs_writing_score.png`** (65.14 KB) -- Relationship between `lunch` and `writing score`.
- **`bivariate_math_score_vs_reading_score.png`** (89.18 KB) -- Relationship between `math score` and `reading score`.
- **`bivariate_math_score_vs_writing_score.png`** (89.59 KB) -- Relationship between `math score` and `writing score`.
- **`bivariate_parental_level_of_education_vs_math_score.png`** (89.22 KB) -- Relationship between `parental level of education` and `math score`.
- **`bivariate_parental_level_of_education_vs_reading_score.png`** (94.65 KB) -- Relationship between `parental level of education` and `reading score`.
- **`bivariate_parental_level_of_education_vs_writing_score.png`** (92.33 KB) -- Relationship between `parental level of education` and `writing score`.
- **`bivariate_race_ethnicity_vs_math_score.png`** (62.82 KB) -- Relationship between `race ethnicity` and `math score`.
- **`bivariate_race_ethnicity_vs_reading_score.png`** (73.98 KB) -- Relationship between `race ethnicity` and `reading score`.
- **`bivariate_race_ethnicity_vs_writing_score.png`** (71.91 KB) -- Relationship between `race ethnicity` and `writing score`.
- **`bivariate_reading_score_vs_writing_score.png`** (91.75 KB) -- Relationship between `reading score` and `writing score`.
- **`bivariate_test_preparation_course_vs_math_score.png`** (57.68 KB) -- Relationship between `test preparation course` and `math score`.
- **`bivariate_test_preparation_course_vs_reading_score.png`** (66.15 KB) -- Relationship between `test preparation course` and `reading score`.
- **`bivariate_test_preparation_course_vs_writing_score.png`** (65.72 KB) -- Relationship between `test preparation course` and `writing score`.
- **`categorical_association_matrix.png`** (78.0 KB) -- Cramer's V association heatmap across categorical features.
- **`correlation_matrix.png`** (49.42 KB) -- Pearson correlation heatmap across numeric features.
- **`dist_gender.png`** (24.74 KB) -- Distribution of `gender`.
- **`dist_lunch.png`** (28.92 KB) -- Distribution of `lunch`.
- **`dist_math_score.png`** (44.93 KB) -- Distribution of `math score`.
- **`dist_parental_level_of_education.png`** (51.28 KB) -- Distribution of `parental level of education`.
- **`dist_race_ethnicity.png`** (37.93 KB) -- Distribution of `race ethnicity`.
- **`dist_reading_score.png`** (45.42 KB) -- Distribution of `reading score`.
- **`dist_test_preparation_course.png`** (30.31 KB) -- Distribution of `test preparation course`.
- **`dist_writing_score.png`** (45.29 KB) -- Distribution of `writing score`.

---

## 8. Categorical Associations (Cramer's V)
| Feature 1 | Feature 2 | Cramer's V |
|---|---|---|
| `gender` | `race/ethnicity` | 0.0709 |
| `parental level of education` | `test preparation course` | 0.0674 |
| `race/ethnicity` | `parental level of education` | 0.0487 |
| `race/ethnicity` | `test preparation course` | 0.0385 |
| `gender` | `parental level of education` | 0.0 |
| `gender` | `lunch` | 0.0 |
| `gender` | `test preparation course` | 0.0 |
| `race/ethnicity` | `lunch` | 0.0 |
| `parental level of education` | `lunch` | 0.0 |
| `lunch` | `test preparation course` | 0.0 |

---

## 9. Predictive Modeling Strategy Blueprint
- **Target Definition:** writing score
- **Problem Type:** Regression
### Recommended Algorithms
- Regularized Linear Regression (Ridge / Lasso)
- Random Forest Regressor
- Gradient Boosting Regressor
- Support Vector Regressor (SVR)
### Feature Selection Strategy
- Exclude high-cardinality ID or text name columns
- Rank features using cross-validated permutation importance and mutual information
- Remove collinear features exceeding correlation threshold > 0.85
### Validation Strategy
- K-Fold Cross-Validation (5 folds)
- Evaluate MAE, RMSE, R-Squared, and Residual Error distribution
### Overfitting Risk Mitigation
- Apply regularization penalties (L1/L2)
- Limit tree depth and enforce minimum samples per leaf
- Perform hyperparameter tuning strictly within cross-validation folds
- **Executive Summary:** Target: writing score (Regression). Use robust cross-validation on 1000 rows x 8 columns.

---

*Report generated automatically by `summary_generator.py`*
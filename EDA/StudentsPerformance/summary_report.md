# Executive EDA & Dataset Summary Report
**Target Directory:** `C:\Users\Anish Kumar Verma\PycharmProjects\AutoEDA\sandbox_run\1da9e373-8d94-40ad-b6b7-b370e045ad89`
**Processed Files:** `agent_plan_log.json`, `agent_state.json`, `current_df.csv`, `metadata_profile.json`, `metrics.json`
**Excluded Files:** `generated_analysis.py` (Script excluded from summary)

---

## 1. Dataset Overview
- **Dataset Identifier:** `StudentsPerformance.csv`
- **Dimensions:** `1000` rows x `8` columns
- **Target Variable:** `average_score`
- **Data Quality:** No missing values detected in raw profile.

---

## 1.5 Full Column Statistics
| Column | Type | Missing % | Unique % | Mean | Median | Std | Skew | Kurtosis |
|---|---|---|---|---|---|---|---|---|
| `gender` | `str` | 0.0% | 0.2% | N/A | N/A | N/A | N/A | N/A |
| `race/ethnicity` | `str` | 0.0% | 0.5% | N/A | N/A | N/A | N/A | N/A |
| `parental level of education` | `str` | 0.0% | 0.6% | N/A | N/A | N/A | N/A | N/A |
| `lunch` | `str` | 0.0% | 0.2% | N/A | N/A | N/A | N/A | N/A |
| `test preparation course` | `str` | 0.0% | 0.2% | N/A | N/A | N/A | N/A | N/A |
| `math score` | `int64` | 0.0% | 8.1% | 66.09 | 66.0 | 15.16 | -0.28 | 0.27 |
| `reading score` | `int64` | 0.0% | 7.2% | 69.17 | 70.0 | 14.6 | -0.26 | -0.07 |
| `writing score` | `int64` | 0.0% | 7.7% | 68.05 | 69.0 | 15.2 | -0.29 | -0.03 |

---

## 2. Data Imputation & Preprocessing
- **status:** Imputation completed

---

## 3. Outlier Analysis (IQR Method)
No numeric outlier statistics reported.

---

## 4. Derived Domain Attributes & Composite Metrics
- **`average_score`**: Formula: `(`math score` + `reading score` + `writing score`) / 3` | Purpose: Composite academic performance metric as requested by the user.

---

## 5. Statistical Hypothesis Testing & Key Predictors
All predictors below were tested against `average_score` and found statistically significant (p < 0.05), ranked by effect size.

| Feature | Test Type | Effect Size | Label | P-Value | Why It Matters |
|---|---|---|---|---|---|
| `reading score` | Pearson Correlation | 0.9703 | Strong correlation | 0.0000e+00 | High proficiency in reading is closely linked to a student's overall academic success across all subjects. |
| `writing score` | Pearson Correlation | 0.9657 | Strong correlation | 0.0000e+00 | Strong writing skills are a primary indicator of a student's ability to achieve high average marks. |
| `math score` | Pearson Correlation | 0.9187 | Strong correlation | 0.0000e+00 | Performance in mathematics shows a very strong connection to the total average grade earned by students. |
| `lunch` | ANOVA | 0.2901 | Large effect | 7.7368e-21 | The type of lunch a student receives is associated with significant differences in their overall academic results. |
| `test preparation course` | ANOVA | 0.2567 | Large effect | 1.6338e-16 | Completing a preparatory course relates to a notable increase in a student's average performance levels. |
| `parental level of education` | ANOVA | 0.2265 | Large effect | 4.3810e-10 | The educational background of parents is a meaningful factor linked to their children's academic achievement. |
| `race/ethnicity` | ANOVA | 0.1878 | Large effect | 3.2259e-07 | Average scores show distinct variations when analyzed across different student ethnic groups. |
| `gender` | ANOVA | 0.1309 | Medium effect | 3.3120e-05 | There is a measurable difference in the average academic performance between male and female students. |

---

## 6. Redundancy & Multicollinearity Analysis
**Numeric-Numeric High Correlation Pairs (|r| >= 0.85):**

| Feature 1 | Feature 2 | Correlation (r) | Interpretation |
|---|---|---|---|
| `math score` | `average_score` | 0.9187 | Strong correlation |
| `reading score` | `writing score` | 0.9546 | Strong correlation |
| `reading score` | `average_score` | 0.9703 | Strong correlation |
| `writing score` | `average_score` | 0.9657 | Strong correlation |


---

## 7. Generated Visualizations
No custom chart image assets found in directory (Interactive Plotly visualizations generated directly in HTML report).

---

## 8. Categorical Associations (Cramer's V)
| Feature 1 | Feature 2 | Cramer's V |
|---|---|---|
| `gender` | `race/ethnicity` | 0.0709 |
| `gender` | `parental level of education` | 0.0 |
| `gender` | `lunch` | 0.0 |
| `gender` | `test preparation course` | 0.0 |
| `race/ethnicity` | `parental level of education` | 0.0487 |
| `race/ethnicity` | `lunch` | 0.0 |
| `race/ethnicity` | `test preparation course` | 0.0385 |
| `parental level of education` | `lunch` | 0.0 |
| `parental level of education` | `test preparation course` | 0.0674 |
| `lunch` | `test preparation course` | 0.0 |

---

## 9. Predictive Modeling Strategy Blueprint
- **Target Definition:** average_score
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
- **Executive Summary:** Target: 'average_score' (Regression). Model recommendations and validation strategy tailored for 1000 rows x 9 columns.

---

*Report generated automatically by `summary_generator.py`*
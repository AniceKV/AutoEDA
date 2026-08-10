# Executive EDA & Dataset Summary Report
**Target Directory:** `C:\Users\Anish Kumar Verma\PycharmProjects\AutoEDA\sandbox_run\eb8252cd-73f1-4c19-afab-3b1db3c3cd95`
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
- **`average_score`**: Formula: `(`math score` + `reading score` + `writing score`) / 3` | Purpose: High-signal feature engineering transformation

---

## 5. Statistical Hypothesis Testing & Key Predictors
All predictors below were tested against `average_score` and found statistically significant (p < 0.05), ranked by effect size.

| Feature | Test Type | Effect Size | Label | P-Value | Why It Matters |
|---|---|---|---|---|---|
| `reading score` | Pearson Correlation | 0.9703 | Strong correlation | 0.0000e+00 | Higher reading scores tend to accompany higher overall average scores, showing reading ability closely aligns with overall academic performance. |
| `writing score` | Pearson Correlation | 0.9657 | Strong correlation | 0.0000e+00 | Students with stronger writing scores usually achieve higher average scores, highlighting the importance of writing skills for overall achievement. |
| `math score` | Pearson Correlation | 0.9187 | Strong correlation | 0.0000e+00 | Better math scores are linked to higher average scores, indicating math proficiency is a key component of overall success. |
| `lunch` | ANOVA | 0.2901 | Large effect | 7.7368e-21 | Students receiving standard lunch tend to have higher average scores than those with free/reduced lunch, reflecting socioeconomic influences on performance. |
| `test preparation course` | ANOVA | 0.2567 | Large effect | 1.6338e-16 | Completing a test preparation course is associated with higher average scores, suggesting focused study boosts overall performance. |
| `parental level of education` | ANOVA | 0.2265 | Large effect | 4.3810e-10 | Higher parental education levels correspond with higher student average scores, indicating family educational background supports student achievement. |
| `race/ethnicity` | ANOVA | 0.1878 | Large effect | 3.2259e-07 | Average scores vary across race/ethnicity groups, reflecting broader societal factors that affect student outcomes. |
| `gender` | ANOVA | 0.1309 | Medium effect | 3.3120e-05 | Male and female students show modest differences in average scores, highlighting slight gender-related performance variations. |

---

## 6. Redundancy & Multicollinearity Analysis
No high-correlation or cross-type redundant feature pairs detected (threshold: |r| or Eta >= 0.85).

---

## 7. Generated Visualizations
No custom chart image assets found in directory (Interactive Plotly visualizations generated directly in HTML report).

---

## 8. Categorical Associations (Cramer's V)
No categorical associations available.

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
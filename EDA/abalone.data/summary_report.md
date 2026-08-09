# Executive EDA & Dataset Summary Report
**Target Directory:** `C:\Users\Anish Kumar Verma\PycharmProjects\AutoEDA\sandbox_run\ce1e0998-50a9-421f-905f-ddc962645911`
**Processed Files:** `agent_plan_log.json`, `agent_state.json`, `current_df.csv`, `metadata_profile.json`, `metrics.json`
**Excluded Files:** `generated_analysis.py` (Script excluded from summary)

---

## 1. Dataset Overview
- **Dataset Identifier:** `abalone.data.csv`
- **Dimensions:** `4177` rows x `9` columns
- **Target Variable:** `Rings`
- **Data Quality:** No missing values detected in raw profile.

---

## 1.5 Full Column Statistics
| Column | Type | Missing % | Unique % | Mean | Median | Std | Skew | Kurtosis |
|---|---|---|---|---|---|---|---|---|
| `gender` | `str` | 0.0% | 0.07% | N/A | N/A | N/A | N/A | N/A |
| `Length` | `float64` | 0.0% | 3.21% | 0.52 | 0.55 | 0.12 | -0.64 | 0.06 |
| `Diameter` | `float64` | 0.0% | 2.66% | 0.41 | 0.42 | 0.1 | -0.61 | -0.05 |
| `Height` | `float64` | 0.0% | 1.22% | 0.14 | 0.14 | 0.04 | 3.13 | 76.03 |
| `Whole weight` | `float64` | 0.0% | 58.15% | 0.83 | 0.8 | 0.49 | 0.53 | -0.02 |
| `Shucked weight` | `float64` | 0.0% | 36.27% | 0.36 | 0.34 | 0.22 | 0.72 | 0.6 |
| `Viscera weight` | `float64` | 0.0% | 21.07% | 0.18 | 0.17 | 0.11 | 0.59 | 0.08 |
| `Shell weight` | `float64` | 0.0% | 22.17% | 0.24 | 0.23 | 0.14 | 0.62 | 0.53 |
| `Rings` | `int64` | 0.0% | 0.67% | 9.93 | 9.0 | 3.22 | 1.11 | 2.33 |

---

## 2. Data Imputation & Preprocessing
- **status:** Imputation completed

---

## 3. Outlier Analysis (IQR Method)
No numeric outlier statistics reported.

---

## 4. Derived Domain Attributes & Composite Metrics
No custom derived domain metrics synthesized during this run.

---

## 5. Statistical Hypothesis Testing & Key Predictors
All predictors below were tested against `Rings` and found statistically significant (p < 0.05), ranked by effect size.

| Feature | Test Type | Effect Size | Label | P-Value | Why It Matters |
|---|---|---|---|---|---|
| `Shell weight` | Pearson Correlation | 0.6276 | Strong correlation | 0.0000e+00 | Larger shell weight tends to be linked with older abalone that have more rings. |
| `Diameter` | Pearson Correlation | 0.5747 | Strong correlation | 0.0000e+00 | Wider diameter often indicates a bigger, older abalone, which usually shows a higher ring count. |
| `Height` | Pearson Correlation | 0.5575 | Strong correlation | 0.0000e+00 | Greater height of the shell generally corresponds to older abalone, reflected in more rings. |
| `Length` | Pearson Correlation | 0.5567 | Strong correlation | 0.0000e+00 | Longer shells are typically found on older abalone, which tend to have more rings. |
| `Whole weight` | Pearson Correlation | 0.5404 | Strong correlation | 1.8887e-315 | Heavier overall weight usually signals an older abalone, associated with a higher number of rings. |
| `Viscera weight` | Pearson Correlation | 0.5038 | Strong correlation | 8.5747e-268 | More viscera weight often reflects larger, older abalone, which typically have more rings. |
| `gender` | ANOVA | 0.4394 | Large effect | 3.7246e-195 | Female abalone generally have more rings than males, indicating they are often older. |
| `Shucked weight` | Pearson Correlation | 0.4209 | Moderate correlation | 5.0875e-179 | Higher shucked weight is usually linked to older abalone, which tend to possess more rings. |

---

## 6. Redundancy & Multicollinearity Analysis
**Numeric-Numeric High Correlation Pairs (|r| >= 0.85):**

| Feature 1 | Feature 2 | Correlation (r) | Interpretation |
|---|---|---|---|
| `Length` | `Diameter` | 0.9868 | Strong correlation |
| `Length` | `Whole weight` | 0.9253 | Strong correlation |
| `Length` | `Shucked weight` | 0.8979 | Strong correlation |
| `Length` | `Viscera weight` | 0.903 | Strong correlation |
| `Length` | `Shell weight` | 0.8977 | Strong correlation |
| `Diameter` | `Whole weight` | 0.9255 | Strong correlation |
| `Diameter` | `Shucked weight` | 0.8932 | Strong correlation |
| `Diameter` | `Viscera weight` | 0.8997 | Strong correlation |
| `Diameter` | `Shell weight` | 0.9053 | Strong correlation |
| `Whole weight` | `Shucked weight` | 0.9694 | Strong correlation |
| `Whole weight` | `Viscera weight` | 0.9664 | Strong correlation |
| `Whole weight` | `Shell weight` | 0.9554 | Strong correlation |
| `Shucked weight` | `Viscera weight` | 0.932 | Strong correlation |
| `Shucked weight` | `Shell weight` | 0.8826 | Strong correlation |
| `Viscera weight` | `Shell weight` | 0.9077 | Strong correlation |


---

## 7. Generated Visualizations
No custom chart image assets found in directory (Interactive Plotly visualizations generated directly in HTML report).

---

## 8. Categorical Associations (Cramer's V)
No categorical associations available.

---

## 9. Predictive Modeling Strategy Blueprint
- **Target Definition:** Rings
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
- **Executive Summary:** Target: 'Rings' (Regression). Model recommendations and validation strategy tailored for 4177 rows x 9 columns.

---

*Report generated automatically by `summary_generator.py`*
# Executive EDA & Dataset Summary Report
**Target Directory:** `C:\Users\Anish Kumar Verma\PycharmProjects\AutoEDA\tests\benchmark_sandbox\modified_data`
**Processed Files:** `metadata_profile.json`, `metrics.json`
**Excluded Files:** `generated_analysis.py` (Script excluded from summary)

---

## 1. Dataset Overview
- **Dataset Identifier:** `modified_data.csv`
- **Dimensions:** `4600` rows x `18` columns
- **Target Variable:** `Not Specified`
- **Data Quality:** No missing values detected in raw profile.

---

## 1.5 Full Column Statistics
| Column | Type | Missing % | Unique % | Mean | Median | Std | Skew | Kurtosis |
|---|---|---|---|---|---|---|---|---|
| `date` | `str` | 0.0% | 1.52% | N/A | N/A | N/A | N/A | N/A |
| `price` | `float64` | 0.0% | 37.85% | 551962.99 | 460943.46 | 563834.7 | 24.79 | 1044.35 |
| `bedrooms` | `float64` | 0.0% | 0.22% | 3.4 | 3.0 | 0.91 | 0.46 | 1.24 |
| `bathrooms` | `float64` | 0.0% | 0.57% | 2.16 | 2.25 | 0.78 | 0.62 | 1.87 |
| `sqft_living` | `int64` | 0.0% | 12.3% | 2139.35 | 1980.0 | 963.21 | 1.72 | 8.29 |
| `sqft_lot` | `int64` | 0.0% | 67.67% | 14852.52 | 7683.0 | 35884.44 | 11.31 | 219.87 |
| `floors` | `float64` | 0.0% | 0.13% | 1.51 | 1.5 | 0.54 | 0.55 | -0.54 |
| `waterfront` | `int64` | 0.0% | 0.04% | 0.01 | 0.0 | 0.08 | 11.68 | 134.55 |
| `view` | `int64` | 0.0% | 0.11% | 0.24 | 0.0 | 0.78 | 3.34 | 10.46 |
| `condition` | `int64` | 0.0% | 0.11% | 3.45 | 3.0 | 0.68 | 0.96 | 0.2 |
| `sqft_above` | `int64` | 0.0% | 11.11% | 1827.27 | 1590.0 | 862.17 | 1.49 | 4.07 |
| `sqft_basement` | `int64` | 0.0% | 4.5% | 312.08 | 0.0 | 464.14 | 1.64 | 4.08 |
| `yr_built` | `int64` | 0.0% | 2.5% | 1970.79 | 1976.0 | 29.73 | -0.5 | -0.67 |
| `yr_renovated` | `int64` | 0.0% | 1.3% | 808.61 | 0.0 | 979.41 | 0.39 | -1.85 |
| `street` | `str` | 0.0% | 98.37% | N/A | N/A | N/A | N/A | N/A |
| `city` | `str` | 0.0% | 0.96% | N/A | N/A | N/A | N/A | N/A |
| `statezip` | `str` | 0.0% | 1.67% | N/A | N/A | N/A | N/A | N/A |
| `price_per_sqft` | `float64` | 0.0% | 87.07% | 265.88 | 243.86 | 357.5 | 53.47 | 3287.77 |

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
No statistically significant predictors identified.

---

## 6. Redundancy & Multicollinearity Analysis
**Numeric-Numeric High Correlation Pairs (|r| >= 0.85):**

| Feature 1 | Feature 2 | Correlation (r) | Interpretation |
|---|---|---|---|
| `sqft_living` | `sqft_above` | 0.8764 | Strong correlation |


---

## 7. Generated Visualizations
No custom chart image assets found in directory (Interactive Plotly visualizations generated directly in HTML report).

---

## 8. Categorical Associations (Cramer's V)
| Feature 1 | Feature 2 | Cramer's V |
|---|---|---|
| `bedrooms` | `floors` | 0.2701 |
| `bedrooms` | `waterfront` | 0.0 |
| `bedrooms` | `view` | 0.0855 |
| `bedrooms` | `condition` | 0.066 |
| `bedrooms` | `city` | 0.0978 |
| `floors` | `waterfront` | 0.0 |
| `floors` | `view` | 0.0334 |
| `floors` | `condition` | 0.1864 |
| `floors` | `city` | 0.1897 |
| `waterfront` | `view` | 0.4826 |
| `waterfront` | `condition` | 0.0 |
| `waterfront` | `city` | 0.2342 |
| `view` | `condition` | 0.0272 |
| `view` | `city` | 0.1131 |
| `condition` | `city` | 0.1306 |

---

## 9. Predictive Modeling Strategy Blueprint
- **Target Definition:** price
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
- **Executive Summary:** Target: 'price' (Regression). Model recommendations and validation strategy tailored for 4600 rows x 18 columns.

---

*Report generated automatically by `summary_generator.py`*
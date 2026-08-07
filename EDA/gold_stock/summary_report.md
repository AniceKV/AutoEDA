# Executive EDA & Dataset Summary Report
**Target Directory:** `C:\Users\Anish Kumar Verma\PycharmProjects\AutoEDA\tests\benchmark_sandbox\gold_stock`
**Processed Files:** `metadata_profile.json`, `metrics.json`
**Excluded Files:** `generated_analysis.py` (Script excluded from summary)

---

## 1. Dataset Overview
- **Dataset Identifier:** `gold_stock.csv`
- **Dimensions:** `2970` rows x `6` columns
- **Target Variable:** `Not Specified`
- **Missing Value Columns:** 5
  - `Close`: 1 (0.0%)
  - `High`: 1 (0.0%)
  - `Low`: 1 (0.0%)
  - `Open`: 1 (0.0%)
  - `Volume`: 1 (0.0%)

---

## 1.5 Full Column Statistics
| Column | Type | Missing % | Unique % | Mean | Median | Std | Skew | Kurtosis |
|---|---|---|---|---|---|---|---|---|
| `Price` | `str` | 0.0% | 100.0% | N/A | N/A | N/A | N/A | N/A |
| `Close` | `str` | 0.03% | 83.1% | N/A | N/A | N/A | N/A | N/A |
| `High` | `str` | 0.03% | 98.25% | N/A | N/A | N/A | N/A | N/A |
| `Low` | `str` | 0.03% | 98.96% | N/A | N/A | N/A | N/A | N/A |
| `Open` | `str` | 0.03% | 99.36% | N/A | N/A | N/A | N/A | N/A |
| `Volume` | `str` | 0.03% | 50.37% | N/A | N/A | N/A | N/A | N/A |

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
No high-correlation or cross-type redundant feature pairs detected (threshold: |r| or Eta >= 0.85).

---

## 7. Generated Visual Artifacts
No PNG/SVG image assets found in directory.

---

## 8. Categorical Associations (Cramer's V)
No categorical associations available.

---

## 9. Predictive Modeling Strategy Blueprint
- **Target Definition:** Price
- **Problem Type:** Multiclass Classification
### Recommended Algorithms
- Regularized Logistic Regression (baseline)
- Random Forest Classifier
- Gradient Boosting Classifier (XGBoost / LightGBM)
- Support Vector Classifier (SVM)
### Feature Selection Strategy
- Exclude high-cardinality ID or text name columns
- Rank features using cross-validated permutation importance and mutual information
- Remove collinear features exceeding correlation threshold > 0.85
### Validation Strategy
- Stratified K-Fold Cross-Validation (5 folds)
- Evaluate Balanced Accuracy, Macro F1, Precision-Recall AUC, and Confusion Matrix
### Overfitting Risk Mitigation
- Apply regularization penalties (L1/L2)
- Limit tree depth and enforce minimum samples per leaf
- Perform hyperparameter tuning strictly within cross-validation folds
- **Executive Summary:** Target: 'Price' (Multiclass Classification). Model recommendations and validation strategy tailored for 2970 rows x 6 columns.

---

*Report generated automatically by `summary_generator.py`*
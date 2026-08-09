# Executive EDA & Dataset Summary Report
**Target Directory:** `C:\Users\Anish Kumar Verma\AppData\Local\Temp\pytest-of-Anish Kumar Verma\pytest-31\test_executive_summary_generat0\summary_ws`
**Processed Files:** `metadata_profile.json`, `metrics.json`
**Excluded Files:** `generated_analysis.py` (Script excluded from summary)

---

## 1. Dataset Overview
- **Dataset Identifier:** `iris_sample.csv`
- **Dimensions:** `10` rows x `4` columns
- **Target Variable:** `survived`
- **Missing Value Columns:** 1
  - `age`: 1 (10.0%)

---

## 1.5 Full Column Statistics
| Column | Type | Missing % | Unique % | Mean | Median | Std | Skew | Kurtosis |
|---|---|---|---|---|---|---|---|---|
| `age` | `float64` | 10.0% | 80.0% | 28.11 | 27.0 | 14.95 | -0.09 | 0.63 |
| `fare` | `float64` | 0.0% | 100.0% | 27.02 | 16.1 | 23.6 | 0.93 | -0.63 |
| `sex` | `str` | 0.0% | 20.0% | N/A | N/A | N/A | N/A | N/A |
| `survived` | `int64` | 0.0% | 20.0% | 0.5 | 0.5 | 0.53 | 0.0 | -2.57 |

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
**Cross-Type Redundant Pairs (categorical vs. its own numeric/ordinal encoding, Eta >= 0.85):**

| Categorical Feature | Numeric Feature | Correlation Ratio (Eta) | Interpretation |
|---|---|---|---|
| `age` | `fare` | 0.8837 | High cross-type redundancy between 'age' and 'fare' (Eta = 0.8837). |
| `age` | `survived` | 0.8803 | High cross-type redundancy between 'age' and 'survived' (Eta = 0.8803). |
| `fare` | `age` | 1.0 | High cross-type redundancy between 'fare' and 'age' (Eta = 1.0000). |
| `fare` | `survived` | 1.0 | High cross-type redundancy between 'fare' and 'survived' (Eta = 1.0000). |
| `sex` | `survived` | 1.0 | High cross-type redundancy between 'sex' and 'survived' (Eta = 1.0000). |

_Recommendation: drop one feature from each redundant pair before modeling to avoid multicollinearity._

---

## 7. Generated Visualizations
No custom chart image assets found in directory (Interactive Plotly visualizations generated directly in HTML report).

---

## 8. Categorical Associations (Cramer's V)
| Feature 1 | Feature 2 | Cramer's V |
|---|---|---|
| `age` | `fare` | 0.0 |
| `age` | `sex` | 0.0 |
| `age` | `survived` | 0.0 |
| `fare` | `sex` | 0.0 |
| `fare` | `survived` | 0.0 |
| `sex` | `survived` | 1.0 |

---

## 9. Predictive Modeling Strategy Blueprint
- **Target Definition:** survived
- **Problem Type:** Binary Classification
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
- Stratified K-Fold Cross-Validation (Repeated 5 folds)
- Evaluate Balanced Accuracy, Macro F1, Precision-Recall AUC, and Confusion Matrix
### Overfitting Risk Mitigation
- Apply regularization penalties (L1/L2)
- Limit tree depth and enforce minimum samples per leaf
- Perform hyperparameter tuning strictly within cross-validation folds
- **Executive Summary:** Target: 'survived' (Binary Classification). Model recommendations and validation strategy tailored for 10 rows x 4 columns.

---

*Report generated automatically by `summary_generator.py`*
# Executive EDA & Dataset Summary Report
**Target Directory:** `C:\Users\Anish Kumar Verma\PycharmProjects\AutoEDA\sandbox_run\36313acd-2e56-4fa9-9f08-a783b3bf3da8`
**Processed Files:** `agent_plan_log.json`, `agent_state.json`, `current_df.csv`, `metadata_profile.json`, `metrics.json`
**Excluded Files:** `generated_analysis.py` (Script excluded from summary)

---

## 1. Dataset Overview
- **Dataset Identifier:** `BankNoteAuthentication.csv`
- **Dimensions:** `1372` rows x `5` columns
- **Target Variable:** `class`
- **Data Quality:** No missing values detected in raw profile.

---

## 1.5 Full Column Statistics
| Column | Type | Missing % | Unique % | Mean | Median | Std | Skew | Kurtosis |
|---|---|---|---|---|---|---|---|---|
| `variance` | `float64` | 0.0% | 97.52% | 0.43 | 0.5 | 2.84 | -0.15 | -0.75 |
| `skewness` | `float64` | 0.0% | 91.55% | 1.92 | 2.32 | 5.87 | -0.39 | -0.44 |
| `curtosis` | `float64` | 0.0% | 92.57% | 1.4 | 0.62 | 4.31 | 1.09 | 1.27 |
| `entropy` | `float64` | 0.0% | 84.26% | -1.19 | -0.59 | 2.1 | -1.02 | 0.5 |
| `class` | `int64` | 0.0% | 0.15% | 0.44 | 0.0 | 0.5 | 0.22 | -1.95 |

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
All predictors below were tested against `class` and found statistically significant (p < 0.05), ranked by effect size.

| Feature | Test Type | Effect Size | Label | P-Value | Why It Matters |
|---|---|---|---|---|---|
| `variance` | Pearson Correlation | 0.7248 | Strong correlation | 5.7410e-224 | Higher variance in the note’s image signals a stronger likelihood of being authentic. |
| `skewness` | Pearson Correlation | 0.4447 | Moderate correlation | 1.3721e-67 | Skewness reflects asymmetry in the note’s texture, with larger values often linked to genuine notes. |
| `curtosis` | Pearson Correlation | 0.1559 | Weak correlation | 6.4655e-09 | Curtosis measures peakedness of the image distribution; higher values modestly associate with authentic banknotes. |

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
- **Target Definition:** class
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
- Stratified K-Fold Cross-Validation (5 folds)
- Evaluate Balanced Accuracy, Macro F1, Precision-Recall AUC, and Confusion Matrix
### Overfitting Risk Mitigation
- Apply regularization penalties (L1/L2)
- Limit tree depth and enforce minimum samples per leaf
- Perform hyperparameter tuning strictly within cross-validation folds
- **Executive Summary:** Target: 'class' (Binary Classification). Model recommendations and validation strategy tailored for 1372 rows x 5 columns.

---

*Report generated automatically by `summary_generator.py`*
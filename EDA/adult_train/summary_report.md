# Executive EDA & Dataset Summary Report
**Target Directory:** `C:\Users\Anish Kumar Verma\PycharmProjects\AutoEDA\sandbox_run\ff353bd1-081c-4a0a-bc11-8f8ffccb9c78`
**Processed Files:** `agent_plan_log.json`, `agent_state.json`, `current_df.csv`, `metadata_profile.json`, `metrics.json`
**Excluded Files:** `generated_analysis.py` (Script excluded from summary)

---

## 1. Dataset Overview
- **Dataset Identifier:** `adult_train.csv`
- **Dimensions:** `32561` rows x `15` columns
- **Target Variable:** `Target`
- **Missing Value Columns:** 3
  - `Workclass`: 1836 (5.6%)
  - `Occupation`: 1843 (5.7%)
  - `Country`: 583 (1.8%)

---

## 1.5 Full Column Statistics
| Column | Type | Missing % | Unique % | Mean | Median | Std | Skew | Kurtosis |
|---|---|---|---|---|---|---|---|---|
| `Age` | `int64` | 0.0% | 0.22% | 38.58 | 37.0 | 13.64 | 0.56 | -0.17 |
| `Workclass` | `str` | 5.64% | 0.02% | N/A | N/A | N/A | N/A | N/A |
| `fnlwgt` | `int64` | 0.0% | 66.48% | 189778.37 | 178356.0 | 105549.98 | 1.45 | 6.22 |
| `Education` | `str` | 0.0% | 0.05% | N/A | N/A | N/A | N/A | N/A |
| `Education_Num` | `int64` | 0.0% | 0.05% | 10.08 | 10.0 | 2.57 | -0.31 | 0.62 |
| `Martial_Status` | `str` | 0.0% | 0.02% | N/A | N/A | N/A | N/A | N/A |
| `Occupation` | `str` | 5.66% | 0.04% | N/A | N/A | N/A | N/A | N/A |
| `Relationship` | `str` | 0.0% | 0.02% | N/A | N/A | N/A | N/A | N/A |
| `Race` | `str` | 0.0% | 0.02% | N/A | N/A | N/A | N/A | N/A |
| `Sex` | `str` | 0.0% | 0.01% | N/A | N/A | N/A | N/A | N/A |
| `Capital_Gain` | `int64` | 0.0% | 0.37% | 1077.65 | 0.0 | 7385.29 | 11.95 | 154.8 |
| `Capital_Loss` | `int64` | 0.0% | 0.28% | 87.3 | 0.0 | 402.96 | 4.59 | 20.38 |
| `Hours_per_week` | `int64` | 0.0% | 0.29% | 40.44 | 40.0 | 12.35 | 0.23 | 2.92 |
| `Country` | `str` | 1.79% | 0.13% | N/A | N/A | N/A | N/A | N/A |
| `Target` | `str` | 0.0% | 0.01% | N/A | N/A | N/A | N/A | N/A |

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
All predictors below were tested against `Target` and found statistically significant (p < 0.05), ranked by effect size.

| Feature | Test Type | Effect Size | Label | P-Value | Why It Matters |
|---|---|---|---|---|---|
| `Relationship` | Chi-Square | 0.4534 | Medium association | 0.0000e+00 | Family roles and household structures are strongly linked to the target outcome. |
| `Martial_Status` | Chi-Square | 0.4472 | Medium association | 0.0000e+00 | Legal union status shows a significant connection with the observed target results. |
| `Education` | Chi-Square | 0.3682 | Medium association | 0.0000e+00 | The highest level of schooling completed relates closely to the target variable. |
| `Occupation` | Chi-Square | 0.3486 | Medium association | 0.0000e+00 | The specific type of work performed is a key indicator for the target. |
| `Education_Num` | ANOVA | 0.3352 | Large effect | 0.0000e+00 | The total years spent in formal education show a strong link to the outcome. |
| `Age` | ANOVA | 0.234 | Large effect | 0.0000e+00 | The stage of life and maturity level are highly relevant to the target. |
| `Hours_per_week` | ANOVA | 0.2297 | Large effect | 0.0000e+00 | The amount of time dedicated to work each week relates to the target. |
| `Capital_Gain` | ANOVA | 0.2233 | Large effect | 0.0000e+00 | Profits from investments show a notable association with the target variable. |
| `Sex` | Chi-Square | 0.2159 | Small association | 0.0000e+00 | Gender differences are linked to variations in the target outcome. |
| `Workclass` | Chi-Square | 0.1634 | Small association | 1.9338e-174 | The sector of employment shows a relevant connection to the target results. |
| `Capital_Loss` | ANOVA | 0.1505 | Large effect | 2.6865e-164 | Financial losses from investments are associated with changes in the target. |
| `Race` | Chi-Square | 0.1002 | Small association | 2.3060e-70 | Self-identified racial background shows a statistical link to the target variable. |
| `Country` | Chi-Square | 0.0931 | Negligible association | 8.2804e-45 | The individual's nation of origin relates to the observed target outcome. |

---

## 6. Redundancy & Multicollinearity Analysis
**Cross-Type Redundant Pairs (categorical vs. its own numeric/ordinal encoding, Eta >= 0.85):**

| Categorical Feature | Numeric Feature | Correlation Ratio (Eta) | Interpretation |
|---|---|---|---|
| `Education` | `Education_Num` | 1.0 | High cross-type redundancy between 'Education' and 'Education_Num' (Eta = 1.0000). |

_Recommendation: drop one feature from each redundant pair before modeling to avoid multicollinearity._

---

## 7. Generated Visual Artifacts
No PNG/SVG image assets found in directory.

---

## 8. Categorical Associations (Cramer's V)
| Feature 1 | Feature 2 | Cramer's V |
|---|---|---|
| `Workclass` | `Education` | 0.0998 |
| `Workclass` | `Martial_Status` | 0.0763 |
| `Workclass` | `Occupation` | 0.215 |
| `Workclass` | `Relationship` | 0.0886 |
| `Workclass` | `Race` | 0.0549 |
| `Workclass` | `Sex` | 0.1431 |
| `Workclass` | `Country` | 0.0298 |
| `Workclass` | `Target` | 0.1634 |
| `Education` | `Martial_Status` | 0.089 |
| `Education` | `Occupation` | 0.1963 |
| `Education` | `Relationship` | 0.1208 |
| `Education` | `Race` | 0.0718 |
| `Education` | `Sex` | 0.0932 |
| `Education` | `Country` | 0.1292 |
| `Education` | `Target` | 0.3682 |
| `Martial_Status` | `Occupation` | 0.1299 |
| `Martial_Status` | `Relationship` | 0.4878 |
| `Martial_Status` | `Race` | 0.0831 |
| `Martial_Status` | `Sex` | 0.4616 |
| `Martial_Status` | `Country` | 0.0639 |
| `Martial_Status` | `Target` | 0.4472 |
| `Occupation` | `Relationship` | 0.1768 |
| `Occupation` | `Race` | 0.0802 |
| `Occupation` | `Sex` | 0.4338 |
| `Occupation` | `Country` | 0.0679 |
| `Occupation` | `Target` | 0.3486 |
| `Relationship` | `Race` | 0.0973 |
| `Relationship` | `Sex` | 0.6489 |
| `Relationship` | `Country` | 0.0781 |
| `Relationship` | `Target` | 0.4534 |
| `Race` | `Sex` | 0.1176 |
| `Race` | `Country` | 0.4207 |
| `Race` | `Target` | 0.1002 |
| `Sex` | `Country` | 0.0558 |
| `Sex` | `Target` | 0.2159 |
| `Country` | `Target` | 0.0931 |

---

## 9. Predictive Modeling Strategy Blueprint
- **Target Definition:** Target
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
- **Executive Summary:** Target: 'Target' (Binary Classification). Model recommendations and validation strategy tailored for 32561 rows x 15 columns.

---

*Report generated automatically by `summary_generator.py`*
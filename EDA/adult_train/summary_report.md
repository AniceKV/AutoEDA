# Executive EDA & Dataset Summary Report
**Target Directory:** `C:\Users\Anish Kumar Verma\PycharmProjects\AutoEDA\sandbox_run\ce1e0998-50a9-421f-905f-ddc962645911`
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
| `Relationship` | Chi-Square | 0.4534 | Medium association | 0.0000e+00 | People’s relationship status (e.g., spouse, child) relates to income because family responsibilities can affect work hours and earnings. |
| `Martial_Status` | Chi-Square | 0.4472 | Medium association | 0.0000e+00 | Marital status influences earnings since married individuals often have different financial needs and job stability than singles. |
| `Education` | Chi-Square | 0.3682 | Medium association | 0.0000e+00 | Education level matters because higher schooling typically opens doors to better-paying jobs. |
| `Occupation` | Chi-Square | 0.3486 | Medium association | 0.0000e+00 | Occupation type affects income as different jobs pay varying wages. |
| `Education_Num` | ANOVA | 0.3352 | Large effect | 0.0000e+00 | More years of schooling are linked to higher earnings, reflecting skill accumulation. |
| `Age` | ANOVA | 0.234 | Large effect | 0.0000e+00 | Age correlates with income because experience and career progression often increase earnings over time. |
| `Hours_per_week` | ANOVA | 0.2297 | Large effect | 0.0000e+00 | Working more hours per week generally leads to higher pay, though overtime rules apply. |
| `Capital_Gain` | ANOVA | 0.2233 | Large effect | 0.0000e+00 | Capital gains boost total income, indicating investment returns contribute to earnings. |
| `Sex` | Chi-Square | 0.2159 | Small association | 0.0000e+00 | Gender shows income differences, reflecting broader societal wage gaps. |
| `Workclass` | Chi-Square | 0.1634 | Small association | 1.9338e-174 | Work class (e.g., private, self‑employed) influences earnings due to varying employment conditions. |
| `Capital_Loss` | ANOVA | 0.1505 | Large effect | 2.6865e-164 | Capital losses reduce overall income, showing negative investment outcomes affect earnings. |
| `Race` | Chi-Square | 0.1002 | Small association | 2.3060e-70 | Race is associated with income disparities, highlighting systemic inequality. |
| `Country` | Chi-Square | 0.0931 | Negligible association | 8.2804e-45 | Country of origin relates to earnings, reflecting economic differences across regions. |

---

## 6. Redundancy & Multicollinearity Analysis
**Cross-Type Redundant Pairs (categorical vs. its own numeric/ordinal encoding, Eta >= 0.85):**

| Categorical Feature | Numeric Feature | Correlation Ratio (Eta) | Interpretation |
|---|---|---|---|
| `Education` | `Education_Num` | 1.0 | High cross-type redundancy between 'Education' and 'Education_Num' (Eta = 1.0000). |

_Recommendation: drop one feature from each redundant pair before modeling to avoid multicollinearity._

---

## 7. Generated Visualizations
No custom chart image assets found in directory (Interactive Plotly visualizations generated directly in HTML report).

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
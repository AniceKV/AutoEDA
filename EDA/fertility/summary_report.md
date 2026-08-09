# Executive EDA & Dataset Summary Report
**Target Directory:** `C:\Users\Anish Kumar Verma\PycharmProjects\AutoEDA\my_analysis_output`
**Processed Files:** `agent_plan_log.json`, `agent_state.json`, `current_df.csv`, `metadata_profile.json`, `metrics.json`
**Excluded Files:** `generated_analysis.py` (Script excluded from summary)

---

## 1. Dataset Overview
- **Dataset Identifier:** `fertility.csv`
- **Dimensions:** `100` rows x `10` columns
- **Target Variable:** `Diagnosis`
- **Data Quality:** No missing values detected in raw profile.

---

## 1.5 Full Column Statistics
| Column | Type | Missing % | Unique % | Mean | Median | Std | Skew | Kurtosis |
|---|---|---|---|---|---|---|---|---|
| `Season` | `str` | 0.0% | 4.0% | N/A | N/A | N/A | N/A | N/A |
| `Age` | `int64` | 0.0% | 10.0% | 30.11 | 30.0 | 2.25 | 0.67 | -0.21 |
| `Childish diseases` | `str` | 0.0% | 2.0% | N/A | N/A | N/A | N/A | N/A |
| `Accident or serious trauma` | `str` | 0.0% | 2.0% | N/A | N/A | N/A | N/A | N/A |
| `Surgical intervention` | `str` | 0.0% | 2.0% | N/A | N/A | N/A | N/A | N/A |
| `High fevers in the last year` | `str` | 0.0% | 3.0% | N/A | N/A | N/A | N/A | N/A |
| `Frequency of alcohol consumption` | `str` | 0.0% | 5.0% | N/A | N/A | N/A | N/A | N/A |
| `Smoking habit` | `str` | 0.0% | 3.0% | N/A | N/A | N/A | N/A | N/A |
| `Number of hours spent sitting per day` | `int64` | 0.0% | 14.0% | 10.8 | 7.0 | 33.62 | 9.85 | 98.02 |
| `Diagnosis` | `str` | 0.0% | 2.0% | N/A | N/A | N/A | N/A | N/A |

---

## 2. Data Imputation & Preprocessing
- **status:** Imputation completed

---

## 3. Outlier Analysis (IQR Method)
No numeric outlier statistics reported.

---

## 4. Derived Domain Attributes & Composite Metrics
- **`log_hours_sitting`**: Formula: `log1p(`Number of hours spent sitting per day`)` | Purpose: High-signal feature engineering transformation
- **`age_hours_ratio`**: Formula: ``Age` / (`Number of hours spent sitting per day` + 1)` | Purpose: High-signal feature engineering transformation

---

## 5. Statistical Hypothesis Testing & Key Predictors
No statistically significant predictors identified.

---

## 6. Redundancy & Multicollinearity Analysis
No high-correlation or cross-type redundant feature pairs detected (threshold: |r| or Eta >= 0.85).

---

## 7. Generated Visualizations
No custom chart image assets found in directory (Interactive Plotly visualizations generated directly in HTML report).

---

## 8. Categorical Associations (Cramer's V)
| Feature 1 | Feature 2 | Cramer's V |
|---|---|---|
| `Season` | `Age` | 0.1063 |
| `Season` | `Childish diseases` | 0.1049 |
| `Season` | `Accident or serious trauma` | 0.2253 |
| `Season` | `Surgical intervention` | 0.0 |
| `Season` | `High fevers in the last year` | 0.2027 |
| `Season` | `Frequency of alcohol consumption` | 0.0 |
| `Season` | `Smoking habit` | 0.0 |
| `Season` | `Diagnosis` | 0.1069 |
| `Age` | `Childish diseases` | 0.1588 |
| `Age` | `Accident or serious trauma` | 0.2386 |
| `Age` | `Surgical intervention` | 0.1911 |
| `Age` | `High fevers in the last year` | 0.2229 |
| `Age` | `Frequency of alcohol consumption` | 0.2367 |
| `Age` | `Smoking habit` | 0.1109 |
| `Age` | `Diagnosis` | 0.1796 |
| `Childish diseases` | `Accident or serious trauma` | 0.1289 |
| `Childish diseases` | `Surgical intervention` | 0.0994 |
| `Childish diseases` | `High fevers in the last year` | 0.0 |
| `Childish diseases` | `Frequency of alcohol consumption` | 0.0 |
| `Childish diseases` | `Smoking habit` | 0.0 |
| `Childish diseases` | `Diagnosis` | 0.0 |
| `Accident or serious trauma` | `Surgical intervention` | 0.0234 |
| `Accident or serious trauma` | `High fevers in the last year` | 0.0 |
| `Accident or serious trauma` | `Frequency of alcohol consumption` | 0.177 |
| `Accident or serious trauma` | `Smoking habit` | 0.0 |
| `Accident or serious trauma` | `Diagnosis` | 0.0999 |
| `Surgical intervention` | `High fevers in the last year` | 0.2437 |
| `Surgical intervention` | `Frequency of alcohol consumption` | 0.0 |
| `Surgical intervention` | `Smoking habit` | 0.0 |
| `Surgical intervention` | `Diagnosis` | 0.0 |
| `High fevers in the last year` | `Frequency of alcohol consumption` | 0.0 |
| `High fevers in the last year` | `Smoking habit` | 0.0218 |
| `High fevers in the last year` | `Diagnosis` | 0.0 |
| `Frequency of alcohol consumption` | `Smoking habit` | 0.2102 |
| `Frequency of alcohol consumption` | `Diagnosis` | 0.0 |
| `Smoking habit` | `Diagnosis` | 0.0 |

---

## 9. Predictive Modeling Strategy Blueprint
- **Target Definition:** Diagnosis
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
- **Executive Summary:** Target: 'Diagnosis' (Binary Classification). Model recommendations and validation strategy tailored for 100 rows x 12 columns.

---

*Report generated automatically by `summary_generator.py`*
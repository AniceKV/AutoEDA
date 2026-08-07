# Executive EDA & Dataset Summary Report
**Target Directory:** `C:\Users\Anish Kumar Verma\PycharmProjects\AutoEDA\tests\benchmark_sandbox\Titanic-Dataset`
**Processed Files:** `metadata_profile.json`, `metrics.json`
**Excluded Files:** `generated_analysis.py` (Script excluded from summary)

---

## 1. Dataset Overview
- **Dataset Identifier:** `Titanic-Dataset.csv`
- **Dimensions:** `891` rows x `12` columns
- **Target Variable:** `Not Specified`
- **Missing Value Columns:** 3
  - `Age`: 177 (19.9%)
  - `Cabin`: 687 (77.1%)
  - `Embarked`: 2 (0.2%)

---

## 1.5 Full Column Statistics
| Column | Type | Missing % | Unique % | Mean | Median | Std | Skew | Kurtosis |
|---|---|---|---|---|---|---|---|---|
| `PassengerId` | `int64` | 0.0% | 100.0% | 446.0 | 446.0 | 257.35 | 0.0 | -1.2 |
| `Survived` | `int64` | 0.0% | 0.22% | 0.38 | 0.0 | 0.49 | 0.48 | -1.78 |
| `Pclass` | `int64` | 0.0% | 0.34% | 2.31 | 3.0 | 0.84 | -0.63 | -1.28 |
| `Name` | `str` | 0.0% | 100.0% | N/A | N/A | N/A | N/A | N/A |
| `Sex` | `str` | 0.0% | 0.22% | N/A | N/A | N/A | N/A | N/A |
| `Age` | `float64` | 19.87% | 9.88% | 29.7 | 28.0 | 14.53 | 0.39 | 0.18 |
| `SibSp` | `int64` | 0.0% | 0.79% | 0.52 | 0.0 | 1.1 | 3.7 | 17.88 |
| `Parch` | `int64` | 0.0% | 0.79% | 0.38 | 0.0 | 0.81 | 2.75 | 9.78 |
| `Ticket` | `str` | 0.0% | 76.43% | N/A | N/A | N/A | N/A | N/A |
| `Fare` | `float64` | 0.0% | 27.83% | 32.2 | 14.45 | 49.69 | 4.79 | 33.4 |
| `Cabin` | `str` | 77.1% | 16.5% | N/A | N/A | N/A | N/A | N/A |
| `Embarked` | `str` | 0.22% | 0.34% | N/A | N/A | N/A | N/A | N/A |

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

## 7. Generated Visualizations
No custom chart image assets found in directory (Interactive Plotly visualizations generated directly in HTML report).

---

## 8. Categorical Associations (Cramer's V)
| Feature 1 | Feature 2 | Cramer's V |
|---|---|---|
| `Survived` | `Pclass` | 0.3367 |
| `Survived` | `Sex` | 0.5426 |
| `Survived` | `SibSp` | 0.1874 |
| `Survived` | `Parch` | 0.1569 |
| `Survived` | `Embarked` | 0.1661 |
| `Pclass` | `Sex` | 0.1297 |
| `Pclass` | `SibSp` | 0.1478 |
| `Pclass` | `Parch` | 0.022 |
| `Pclass` | `Embarked` | 0.2598 |
| `Sex` | `SibSp` | 0.2059 |
| `Sex` | `Parch` | 0.2471 |
| `Sex` | `Embarked` | 0.1131 |
| `SibSp` | `Parch` | 0.2399 |
| `SibSp` | `Embarked` | 0.0919 |
| `Parch` | `Embarked` | 0.0518 |

---

## 9. Predictive Modeling Strategy Blueprint
- **Target Definition:** Survived
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
- **Executive Summary:** Target: 'Survived' (Binary Classification). Model recommendations and validation strategy tailored for 891 rows x 12 columns.

---

*Report generated automatically by `summary_generator.py`*
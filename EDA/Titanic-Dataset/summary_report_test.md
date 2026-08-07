# Executive EDA & Dataset Summary Report
**Target Directory:** `c:\Users\Anish Kumar Verma\PycharmProjects\AutoEDA\EDA\Titanic-Dataset`
**Processed Files:** `age_vs_survived.png`, `agent_plan_log.json`, `agent_state.json`, `bivariate_Age_vs_Fare.png`, `bivariate_Pclass_vs_Fare.png`, `bivariate_Sex_vs_Survived.png`, `categorical_association_matrix.png`, `correlation_matrix.png`, `dist_Age.png`, `dist_Embarked.png`, `dist_Fare.png`, `dist_Parch.png`, `dist_Pclass.png`, `dist_Sex.png`, `dist_SibSp.png`, `eda_report.html`, `fare_vs_survived.png`, `metadata_profile.json`, `metrics.json`, `pairplot.png`
**Excluded Files:** `generated_analysis.py` (Script excluded from summary)

---

## 1. Dataset Overview
- **Dataset Identifier:** `Titanic-Dataset.csv`
- **Dimensions:** `891` rows x `12` columns
- **Target Variable:** `Survived`
- **Missing Value Columns:** 3
  - `Age`: 177 (19.9%)
  - `Cabin`: 687 (77.1%)
  - `Embarked`: 2 (0.2%)

---

## 1.5 Full Column Statistics
| Column | Type | Missing % | Unique % | Mean | Median | Std | Skew | Kurtosis |
|---|---|---|---|---|---|---|---|---|
| `PassengerId` | `int64` | 0.0% | N/A% | N/A | N/A | N/A | N/A | N/A |
| `Survived` | `int64` | 0.0% | N/A% | N/A | N/A | N/A | N/A | N/A |
| `Pclass` | `int64` | 0.0% | N/A% | N/A | N/A | N/A | N/A | N/A |
| `Name` | `object` | 0.0% | N/A% | N/A | N/A | N/A | N/A | N/A |
| `Sex` | `object` | 0.0% | N/A% | N/A | N/A | N/A | N/A | N/A |
| `Age` | `float64` | 19.87% | N/A% | N/A | N/A | N/A | N/A | N/A |
| `SibSp` | `int64` | 0.0% | N/A% | N/A | N/A | N/A | N/A | N/A |
| `Parch` | `int64` | 0.0% | N/A% | N/A | N/A | N/A | N/A | N/A |
| `Ticket` | `object` | 0.0% | N/A% | N/A | N/A | N/A | N/A | N/A |
| `Fare` | `float64` | 0.0% | N/A% | N/A | N/A | N/A | N/A | N/A |
| `Cabin` | `object` | 77.1% | N/A% | N/A | N/A | N/A | N/A | N/A |
| `Embarked` | `object` | 0.22% | N/A% | N/A | N/A | N/A | N/A | N/A |

---

## 2. Data Imputation & Preprocessing
**Rules Applied:**
- Standardized missing string placeholders ('?', 'NA', 'N/A', 'null') to NaN.
- Numeric columns with skewness > 1.0 or < -1.0 use median imputation.
- Numeric columns with skewness between -1.0 and 1.0 use mean imputation.
- Categorical/String columns use mode imputation with 'Unknown' fallback.

| Column | Missing (Before) | Method | Fill Value |
|---|---|---|---|
| `Age` | 177 | mean | 29.69911764705882 |
| `Cabin` | 687 | mode | B96 B98 |
| `Embarked` | 2 | mode | S |

---

## 3. Outlier Analysis (IQR Method)
| Column | Outlier Count | Outlier Percentage | Bounds (Lower / Upper) |
|---|---|---|---|
| `Age` | 66 | 7.41% | [2.5, 54.5] |
| `SibSp` | 46 | 5.16% | [-1.5, 2.5] |
| `Parch` | 213 | 23.91% | [0.0, 0.0] |
| `Fare` | 116 | 13.02% | [-26.724, 65.6344] |

---

## 4. Derived Domain Attributes & Composite Metrics
No custom derived domain metrics synthesized during this run.

---

## 5. Statistical Hypothesis Testing & Key Predictors
- **Statistically Significant Predictors:** `Pclass`, `Sex`, `Age`, `Parch`, `Ticket`, `Fare`, `Cabin`, `Embarked`
_Detailed effect sizes unavailable -- `ranked_significant_details` missing from metrics.json._

---

## 6. Redundancy & Multicollinearity Analysis
No high-correlation or cross-type redundant feature pairs detected (threshold: |r| or Eta >= 0.85).

---

## 7. Generated Visual Artifacts
- **`age_vs_survived.png`** (58.98 KB) -- Generated analysis artifact.
- **`bivariate_Age_vs_Fare.png`** (119.72 KB) -- Relationship between `Age` and `Fare`.
- **`bivariate_Pclass_vs_Fare.png`** (50.16 KB) -- Relationship between `Pclass` and `Fare`.
- **`bivariate_Sex_vs_Survived.png`** (24.69 KB) -- Relationship between `Sex` and `Survived`.
- **`categorical_association_matrix.png`** (31.91 KB) -- Cramer's V association heatmap across categorical features.
- **`correlation_matrix.png`** (92.44 KB) -- Pearson correlation heatmap across numeric features.
- **`dist_Age.png`** (37.17 KB) -- Distribution of `Age`.
- **`dist_Embarked.png`** (21.7 KB) -- Distribution of `Embarked`.
- **`dist_Fare.png`** (31.93 KB) -- Distribution of `Fare`.
- **`dist_Parch.png`** (22.26 KB) -- Distribution of `Parch`.
- **`dist_Pclass.png`** (18.96 KB) -- Distribution of `Pclass`.
- **`dist_Sex.png`** (22.63 KB) -- Distribution of `Sex`.
- **`dist_SibSp.png`** (22.54 KB) -- Distribution of `SibSp`.
- **`fare_vs_survived.png`** (66.56 KB) -- Generated analysis artifact.
- **`pairplot.png`** (196.94 KB) -- Pairwise scatter/distribution grid across key numeric features, colored by target.

---

## 8. Categorical Associations (Cramer's V)
| Feature 1 | Feature 2 | Cramer's V |
|---|---|---|
| `Sex` | `Embarked` | 0.1107 |

---

## 9. Predictive Modeling Strategy Blueprint
- **Target Definition:** Survived
- **Problem Type:** Classification
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
- **Executive Summary:** Target: Survived (Classification). Use robust cross-validation on 891 rows x 12 columns.

---

*Report generated automatically by `summary_generator.py`*
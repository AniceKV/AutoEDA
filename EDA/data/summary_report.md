# Executive EDA & Dataset Summary Report
**Target Directory:** `C:\Users\Anish Kumar Verma\PycharmProjects\AutoEDA\sandbox_run\e71bd788-597f-419d-8fde-ddececcbd3db`
**Processed Files:** `agent_plan_log.json`, `agent_state.json`, `current_df.csv`, `metadata_profile.json`, `metrics.json`
**Excluded Files:** `generated_analysis.py` (Script excluded from summary)

---

## 1. Dataset Overview
- **Dataset Identifier:** `data.csv`
- **Dimensions:** `891` rows x `14` columns
- **Target Variable:** `Survived`
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
**Rules Applied:**
- Standardized missing string placeholders ('?', 'NA', 'N/A', 'null') to NaN.
- Numeric columns with skewness > 1.0 or < -1.0 use median imputation.
- Numeric columns with skewness between -1.0 and 1.0 use mean imputation.
- Categorical/String columns use mode imputation with 'Unknown' fallback.

| Column | Missing (Before) | Method | Fill Value |
|---|---|---|---|
| `Age` | 177 | median | 28.0 |
| `Cabin` | 687 | mode | B96 B98 |
| `Embarked` | 2 | constant | Unknown |

---

## 3. Outlier Analysis (IQR Method)
No numeric outlier statistics reported.

---

## 4. Derived Domain Attributes & Composite Metrics
- **`FamilySize`**: Formula: `SibSp + Parch + 1` | Purpose: Total number of family members on board.
- **`IsAlone`**: Formula: `FamilySize == 1` | Purpose: Indicator for passengers traveling without family.

---

## 5. Statistical Hypothesis Testing & Key Predictors
All predictors below were tested against `Survived` and found statistically significant (p < 0.05), ranked by effect size.

| Feature | Test Type | Effect Size | Label | P-Value | Why It Matters |
|---|---|---|---|---|---|
| `Sex` | ANOVA | 0.5434 | Large effect | 1.4061e-69 | Gender is a primary indicator of survival, reflecting historical emergency protocols that prioritized specific groups. |
| `Pclass` | Pearson Correlation | 0.3385 | Moderate correlation | 2.5370e-25 | Socioeconomic status is strongly linked to survival, likely due to the location and accessibility of different cabin tiers. |
| `Fare` | Pearson Correlation | 0.2573 | Weak correlation | 6.1202e-15 | The amount paid for a ticket relates to survival, as higher spending often granted better access to safety resources. |
| `IsAlone` | ANOVA | 0.2034 | Large effect | 9.0095e-10 | Traveling without family is associated with different survival outcomes compared to those moving in groups. |
| `Embarked` | ANOVA | 0.1825 | Large effect | 1.3423e-06 | The location where a passenger boarded shows a connection to survival, possibly reflecting different passenger demographics from each port. |
| `Parch` | Pearson Correlation | 0.0816 | Negligible correlation | 1.4799e-02 | The number of parents or children traveling together is linked to survival rates during the evacuation process. |

---

## 6. Redundancy & Multicollinearity Analysis
**Numeric-Numeric High Correlation Pairs (|r| >= 0.85):**

| Feature 1 | Feature 2 | Correlation (r) | Interpretation |
|---|---|---|---|
| `SibSp` | `FamilySize` | 0.8907 | Strong correlation |

**Cross-Type Redundant Pairs (categorical vs. its own numeric/ordinal encoding, Eta >= 0.85):**

| Categorical Feature | Numeric Feature | Correlation Ratio (Eta) | Interpretation |
|---|---|---|---|
| `SibSp` | `FamilySize` | 0.8929 | High cross-type redundancy between 'SibSp' and 'FamilySize' (Eta = 0.8929). |
| `FamilySize` | `SibSp` | 0.9173 | High cross-type redundancy between 'FamilySize' and 'SibSp' (Eta = 0.9173). |

_Recommendation: drop one feature from each redundant pair before modeling to avoid multicollinearity._

---

## 7. Generated Visual Artifacts
No PNG/SVG image assets found in directory.

---

## 8. Categorical Associations (Cramer's V)
| Feature 1 | Feature 2 | Cramer's V |
|---|---|---|
| `Survived` | `Pclass` | 0.3367 |
| `Survived` | `Sex` | 0.5426 |
| `Survived` | `SibSp` | 0.1874 |
| `Survived` | `Parch` | 0.1569 |
| `Survived` | `Embarked` | 0.1731 |
| `Survived` | `FamilySize` | 0.2857 |
| `Survived` | `IsAlone` | 0.2007 |
| `Pclass` | `Sex` | 0.1297 |
| `Pclass` | `SibSp` | 0.1478 |
| `Pclass` | `Parch` | 0.022 |
| `Pclass` | `Embarked` | 0.2637 |
| `Pclass` | `FamilySize` | 0.2029 |
| `Pclass` | `IsAlone` | 0.1275 |
| `Sex` | `SibSp` | 0.2059 |
| `Sex` | `Parch` | 0.2471 |
| `Sex` | `Embarked` | 0.1255 |
| `Sex` | `FamilySize` | 0.3128 |
| `Sex` | `IsAlone` | 0.302 |
| `SibSp` | `Parch` | 0.2399 |
| `SibSp` | `Embarked` | 0.0612 |
| `SibSp` | `FamilySize` | 0.7645 |
| `SibSp` | `IsAlone` | 0.8367 |
| `Parch` | `Embarked` | 0.0 |
| `Parch` | `FamilySize` | 0.4901 |
| `Parch` | `IsAlone` | 0.6858 |
| `Embarked` | `FamilySize` | 0.098 |
| `Embarked` | `IsAlone` | 0.1118 |
| `FamilySize` | `IsAlone` | 0.9961 |

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
- **Executive Summary:** Target: 'Survived' (Binary Classification). Model recommendations and validation strategy tailored for 891 rows x 14 columns.

---

*Report generated automatically by `summary_generator.py`*
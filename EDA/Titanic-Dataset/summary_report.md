# Executive EDA & Dataset Summary Report
**Target Directory:** `C:\Users\Anish Kumar Verma\PycharmProjects\AutoEDA\sandbox_run\eb8252cd-73f1-4c19-afab-3b1db3c3cd95`
**Processed Files:** `agent_plan_log.json`, `agent_state.json`, `current_df.csv`, `metadata_profile.json`, `metrics.json`
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
- **`FamilySize`**: Formula: `SibSp + Parch + 1` | Purpose: High-signal feature engineering transformation
- **`IsAlone`**: Formula: `(SibSp + Parch) == 0` | Purpose: High-signal feature engineering transformation

---

## 5. Statistical Hypothesis Testing & Key Predictors
All predictors below were tested against `Survived` and found statistically significant (p < 0.05), ranked by effect size.

| Feature | Test Type | Effect Size | Label | P-Value | Why It Matters |
|---|---|---|---|---|---|
| `Sex` | ANOVA | 0.5434 | Large effect | 1.4061e-69 | Gender influenced survival chances, with women more likely to survive than men. |
| `Pclass` | Pearson Correlation | 0.3385 | Moderate correlation | 2.5370e-25 | Ticket class reflected socioeconomic status, affecting passengers' access to lifeboats and rescue. |
| `Fare` | Pearson Correlation | 0.2573 | Weak correlation | 6.1202e-15 | Higher ticket fares indicated better accommodations, which were linked to higher survival rates. |
| `IsAlone` | ANOVA | 0.2034 | Large effect | 9.0095e-10 | Traveling alone reduced chances of survival, as companions could assist during evacuation. |
| `Embarked` | ANOVA | 0.1726 | Large effect | 1.5143e-06 | Port of embarkation affected survival, reflecting differences in passenger groups and cabin locations. |
| `Parch` | Pearson Correlation | 0.0816 | Negligible correlation | 1.4799e-02 | Having more parents or children aboard slightly changed survival odds, likely due to family dynamics. |
| `Age` | Pearson Correlation | 0.0772 | Negligible correlation | 3.9125e-02 | Age influenced survival, with younger passengers generally having better chances than older ones. |

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
| `Survived` | `FamilySize` | 0.2857 |
| `Survived` | `IsAlone` | 0.2007 |
| `Pclass` | `Sex` | 0.1297 |
| `Pclass` | `SibSp` | 0.1478 |
| `Pclass` | `Parch` | 0.022 |
| `Pclass` | `Embarked` | 0.2598 |
| `Pclass` | `FamilySize` | 0.2029 |
| `Pclass` | `IsAlone` | 0.1275 |
| `Sex` | `SibSp` | 0.2059 |
| `Sex` | `Parch` | 0.2471 |
| `Sex` | `Embarked` | 0.1131 |
| `Sex` | `FamilySize` | 0.3128 |
| `Sex` | `IsAlone` | 0.302 |
| `SibSp` | `Parch` | 0.2399 |
| `SibSp` | `Embarked` | 0.0919 |
| `SibSp` | `FamilySize` | 0.7645 |
| `SibSp` | `IsAlone` | 0.8367 |
| `Parch` | `Embarked` | 0.0518 |
| `Parch` | `FamilySize` | 0.4901 |
| `Parch` | `IsAlone` | 0.6858 |
| `Embarked` | `FamilySize` | 0.1347 |
| `Embarked` | `IsAlone` | 0.1102 |
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
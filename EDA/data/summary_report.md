# Executive EDA & Dataset Summary Report
**Target Directory:** `C:\Users\Anish Kumar Verma\PycharmProjects\AutoEDA\sandbox_run\f4673b7e-81d1-4823-a0f0-70734d95721b`
**Processed Files:** `agent_plan_log.json`, `agent_state.json`, `bivariate_Age_vs_Fare.png`, `bivariate_FamilySize_vs_Survived.png`, `bivariate_Pclass_vs_Fare.png`, `bivariate_Sex_vs_Pclass.png`, `current_df.csv`, `dist_Age.png`, `dist_Embarked.png`, `dist_FamilySize.png`, `dist_Fare.png`, `dist_Pclass.png`, `dist_Sex.png`, `dist_Survived.png`, `metadata_profile.json`, `metrics.json`, `pairplot.png`, `target_interactions.png`
**Excluded Files:** `generated_analysis.py` (Script excluded from summary)

---

## 1. Dataset Overview
- **Dataset Identifier:** `data.csv`
- **Dimensions:** `891` rows x `15` columns
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
- **`HasCabin`**: Formula: `Cabin.notnull()` | Purpose: Binary indicator for whether a cabin number was recorded, often linked to socio-economic status.

---

## 5. Statistical Hypothesis Testing & Key Predictors
All predictors below were tested against `Survived` and found statistically significant (p < 0.05), ranked by effect size.

| Feature | Test Type | Effect Size | Label | P-Value | Why It Matters |
|---|---|---|---|---|---|
| `Sex` | ANOVA | 0.5434 | Large effect | 1.4061e-69 | Gender is a primary indicator of survival, reflecting historical emergency protocols that prioritized specific groups. |
| `Pclass` | Pearson Correlation | 0.3385 | Moderate correlation | 2.5370e-25 | Socioeconomic status is strongly linked to survival, likely due to the location and accessibility of different cabin tiers. |
| `Fare` | Pearson Correlation | 0.2573 | Weak correlation | 6.1202e-15 | The amount paid for a ticket relates to survival, as higher spending often granted better access to safety resources. |
| `IsAlone` | ANOVA | 0.2034 | Large effect | 9.0095e-10 | Whether a passenger traveled solo or with others is associated with their likelihood of reaching safety. |
| `Embarked` | ANOVA | 0.1825 | Large effect | 1.3423e-06 | The port where a passenger boarded shows a connection to survival rates, possibly reflecting different passenger demographics. |
| `Parch` | Pearson Correlation | 0.0816 | Negligible correlation | 1.4799e-02 | The number of parents or children traveling with a passenger is linked to their survival outcome. |

---

## 6. Redundancy & Multicollinearity Analysis
No high-correlation or cross-type redundant feature pairs detected (threshold: |r| or Eta >= 0.85).

---

## 7. Generated Visual Artifacts
- **`bivariate_Age_vs_Fare.png`** (99.44 KB) -- Relationship between `Age` and `Fare`.
- **`bivariate_FamilySize_vs_Survived.png`** (26.18 KB) -- Relationship between `FamilySize` and `Survived`.
- **`bivariate_Pclass_vs_Fare.png`** (40.58 KB) -- Relationship between `Pclass` and `Fare`.
- **`bivariate_Sex_vs_Pclass.png`** (25.23 KB) -- Relationship between `Sex` and `Pclass`.
- **`dist_Age.png`** (37.47 KB) -- Distribution of `Age`.
- **`dist_Embarked.png`** (24.6 KB) -- Distribution of `Embarked`.
- **`dist_FamilySize.png`** (23.14 KB) -- Distribution of `FamilySize`.
- **`dist_Fare.png`** (31.93 KB) -- Distribution of `Fare`.
- **`dist_Pclass.png`** (18.96 KB) -- Distribution of `Pclass`.
- **`dist_Sex.png`** (22.63 KB) -- Distribution of `Sex`.
- **`dist_Survived.png`** (19.92 KB) -- Distribution of `Survived`.
- **`pairplot.png`** (219.2 KB) -- Pairwise scatter/distribution grid across key numeric features, colored by target.
- **`target_interactions.png`** (58.22 KB) -- Overview of how the top features interact with the target variable.

---

## 8. Categorical Associations (Cramer's V)
No categorical associations available.

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
- **Executive Summary:** Target: 'Survived' (Binary Classification). Model recommendations and validation strategy tailored for 891 rows x 15 columns.

---

*Report generated automatically by `summary_generator.py`*
# Executive EDA & Dataset Summary Report
**Target Directory:** `C:\Users\Anish Kumar Verma\PycharmProjects\AutoEDA\test_sandbox`
**Processed Files:** `categorical_association_matrix.png`, `correlation_matrix.png`, `metadata_profile.json`, `metrics.json`
**Excluded Files:** `generated_analysis.py` (Script excluded from summary)

---

## 1. Dataset Overview
- **Dataset Identifier:** `temp_test.csv`
- **Dimensions:** `891` rows x `13` columns
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
| `Name` | `object` | 0.0% | 100.0% | N/A | N/A | N/A | N/A | N/A |
| `Sex` | `object` | 0.0% | 0.22% | N/A | N/A | N/A | N/A | N/A |
| `Age` | `float64` | 19.87% | 9.88% | 29.7 | 28.0 | 14.53 | 0.39 | 0.18 |
| `SibSp` | `int64` | 0.0% | 0.79% | 0.52 | 0.0 | 1.1 | 3.7 | 17.88 |
| `Parch` | `int64` | 0.0% | 0.79% | 0.38 | 0.0 | 0.81 | 2.75 | 9.78 |
| `Ticket` | `object` | 0.0% | 76.43% | N/A | N/A | N/A | N/A | N/A |
| `Fare` | `float64` | 0.0% | 27.83% | 32.2 | 14.45 | 49.69 | 4.79 | 33.4 |
| `Cabin` | `object` | 77.1% | 16.5% | N/A | N/A | N/A | N/A | N/A |
| `Embarked` | `object` | 0.22% | 0.34% | N/A | N/A | N/A | N/A | N/A |
| `dummy_cat` | `object` | 0.0% | 0.67% | N/A | N/A | N/A | N/A | N/A |

---

## 2. Data Imputation & Preprocessing
- **status:** Imputation completed

---

## 3. Outlier Analysis (IQR Method)
No numeric outlier statistics reported.

---

## 4. Engineered Features
No custom engineered features recorded.

---

## 5. Statistical Hypothesis Testing

---

## 6. Generated Visual Artifacts
- **![categorical_association_matrix.png](categorical_association_matrix.png)** - `categorical_association_matrix.png` (46.83 KB)
- **![correlation_matrix.png](correlation_matrix.png)** - `correlation_matrix.png` (100.77 KB)

---

## Categorical Associations (Cramér's V)
| Feature 1 | Feature 2 | Cramér's V |
|---|---|---|
| `Sex` | `dummy_cat` | 0.9977 |
| `Embarked` | `dummy_cat` | 0.2802 |
| `Sex` | `Embarked` | 0.1131 |

---

## 7. Predictive Modeling Strategy Blueprint
- **Target Definition:** Undefined (Unsupervised)
- **Problem Type:** Unsupervised / Exploratory
### Recommended Algorithms
- K-Means Clustering
- Hierarchical Agglomerative Clustering
- Principal Component Analysis (PCA) for Dimensionality Reduction
### Feature Selection Strategy
- Exclude high-cardinality ID or text name columns
- Rank features using cross-validated permutation importance and mutual information
- Remove collinear features exceeding correlation threshold > 0.85
### Validation Strategy
- Evaluate Silhouette Score and Inertia elbow curve
### Overfitting Risk Mitigation
- Apply regularization penalties (L1/L2)
- Limit tree depth and enforce minimum samples per leaf
- Perform hyperparameter tuning strictly within cross-validation folds
- **Executive Summary:** Target: Undefined (Unsupervised) (Unsupervised / Exploratory). Use robust cross-validation on 891 rows x 13 columns.

---

*Report generated automatically by `summary_generator.py`*
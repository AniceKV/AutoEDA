# Executive EDA & Dataset Summary Report
**Target Directory:** `C:\Users\Anish Kumar Verma\PycharmProjects\AutoEDA\tests\benchmark_sandbox\adult_test-selected-columns`
**Processed Files:** `metadata_profile.json`, `metrics.json`
**Excluded Files:** `generated_analysis.py` (Script excluded from summary)

---

## 1. Dataset Overview
- **Dataset Identifier:** `adult_test-selected-columns.csv`
- **Dimensions:** `924` rows x `10` columns
- **Target Variable:** `Not Specified`
- **Missing Value Columns:** 10
  - `Age`: 924 (100.0%)
  - `Workclass`: 924 (100.0%)
  - `fnlwgt`: 924 (100.0%)
  - `Education`: 924 (100.0%)
  - `Education_Num`: 924 (100.0%)
  - `Martial_Status`: 924 (100.0%)
  - `Occupation`: 924 (100.0%)
  - `Relationship`: 924 (100.0%)
  - `Race`: 924 (100.0%)
  - `Sex`: 924 (100.0%)

---

## 1.5 Full Column Statistics
| Column | Type | Missing % | Unique % | Mean | Median | Std | Skew | Kurtosis |
|---|---|---|---|---|---|---|---|---|
| `Age` | `float64` | 100.0% | 0.0% | N/A | N/A | N/A | N/A | N/A |
| `Workclass` | `float64` | 100.0% | 0.0% | N/A | N/A | N/A | N/A | N/A |
| `fnlwgt` | `float64` | 100.0% | 0.0% | N/A | N/A | N/A | N/A | N/A |
| `Education` | `float64` | 100.0% | 0.0% | N/A | N/A | N/A | N/A | N/A |
| `Education_Num` | `float64` | 100.0% | 0.0% | N/A | N/A | N/A | N/A | N/A |
| `Martial_Status` | `float64` | 100.0% | 0.0% | N/A | N/A | N/A | N/A | N/A |
| `Occupation` | `float64` | 100.0% | 0.0% | N/A | N/A | N/A | N/A | N/A |
| `Relationship` | `float64` | 100.0% | 0.0% | N/A | N/A | N/A | N/A | N/A |
| `Race` | `float64` | 100.0% | 0.0% | N/A | N/A | N/A | N/A | N/A |
| `Sex` | `float64` | 100.0% | 0.0% | N/A | N/A | N/A | N/A | N/A |

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

## 7. Generated Visual Artifacts
No PNG/SVG image assets found in directory.

---

## 8. Categorical Associations (Cramer's V)
No categorical associations available.

---

## 9. Predictive Modeling Strategy Blueprint
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
- **Executive Summary:** Target: 'Undefined (Unsupervised)' (Unsupervised / Exploratory). Model recommendations and validation strategy tailored for 924 rows x 10 columns.

---

*Report generated automatically by `summary_generator.py`*
# Executive EDA & Dataset Summary Report
**Target Directory:** `C:\Users\Anish Kumar Verma\PycharmProjects\AutoEDA\tests\benchmark_sandbox\EV_Adoption_and_Range_Anxiety_Dataset-selected-columns`
**Processed Files:** `metadata_profile.json`, `metrics.json`
**Excluded Files:** `generated_analysis.py` (Script excluded from summary)

---

## 1. Dataset Overview
- **Dataset Identifier:** `EV_Adoption_and_Range_Anxiety_Dataset-selected-columns.csv`
- **Dimensions:** `10000` rows x `10` columns
- **Target Variable:** `Not Specified`
- **Missing Value Columns:** 2
  - `Annual_Income_USD`: 178 (1.8%)
  - `Daily_Commute_km`: 181 (1.8%)

---

## 1.5 Full Column Statistics
| Column | Type | Missing % | Unique % | Mean | Median | Std | Skew | Kurtosis |
|---|---|---|---|---|---|---|---|---|
| `Buyer_ID` | `str` | 0.0% | 100.0% | N/A | N/A | N/A | N/A | N/A |
| `Age` | `int64` | 0.0% | 0.45% | 46.94 | 47.0 | 12.95 | 0.01 | -1.19 |
| `Gender` | `str` | 0.0% | 0.03% | N/A | N/A | N/A | N/A | N/A |
| `Annual_Income_USD` | `float64` | 1.78% | 89.15% | 85378.49 | 84708.0 | 32838.68 | 0.31 | -0.18 |
| `City_Type` | `str` | 0.0% | 0.03% | N/A | N/A | N/A | N/A | N/A |
| `Daily_Commute_km` | `float64` | 1.81% | 9.91% | 41.11 | 40.2 | 23.35 | 0.34 | -0.38 |
| `Number_of_Cars_Owned` | `int64` | 0.0% | 0.04% | 1.86 | 2.0 | 0.86 | 0.76 | -0.11 |
| `Current_Car_Type` | `str` | 0.0% | 0.04% | N/A | N/A | N/A | N/A | N/A |
| `Charging_Stations_Near_Home` | `int64` | 0.0% | 0.15% | 5.35 | 5.0 | 4.05 | 0.57 | -0.72 |
| `Charging_Stations_Near_Work` | `int64` | 0.0% | 0.2% | 7.46 | 6.0 | 5.26 | 0.61 | -0.65 |

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
| `Gender` | `City_Type` | 0.0 |
| `Gender` | `Number_of_Cars_Owned` | 0.0 |
| `Gender` | `Current_Car_Type` | 0.0 |
| `City_Type` | `Number_of_Cars_Owned` | 0.0053 |
| `City_Type` | `Current_Car_Type` | 0.0 |
| `Number_of_Cars_Owned` | `Current_Car_Type` | 0.0 |

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
- **Executive Summary:** Target: 'Undefined (Unsupervised)' (Unsupervised / Exploratory). Model recommendations and validation strategy tailored for 10000 rows x 10 columns.

---

*Report generated automatically by `summary_generator.py`*
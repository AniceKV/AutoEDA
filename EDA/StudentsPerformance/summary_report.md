# Executive EDA & Dataset Summary Report
**Target Directory:** `C:\Users\Anish Kumar Verma\PycharmProjects\AutoEDA\sandbox_run\ce1e0998-50a9-421f-905f-ddc962645911`
**Processed Files:** `agent_plan_log.json`, `agent_state.json`, `current_df.csv`, `metadata_profile.json`, `metrics.json`
**Excluded Files:** `generated_analysis.py` (Script excluded from summary)

---

## 1. Dataset Overview
- **Dataset Identifier:** `StudentsPerformance.csv`
- **Dimensions:** `1000` rows x `8` columns
- **Target Variable:** `average_score`
- **Data Quality:** No missing values detected in raw profile.

---

## 1.5 Full Column Statistics
| Column | Type | Missing % | Unique % | Mean | Median | Std | Skew | Kurtosis |
|---|---|---|---|---|---|---|---|---|
| `gender` | `str` | 0.0% | 0.2% | N/A | N/A | N/A | N/A | N/A |
| `race/ethnicity` | `str` | 0.0% | 0.5% | N/A | N/A | N/A | N/A | N/A |
| `parental level of education` | `str` | 0.0% | 0.6% | N/A | N/A | N/A | N/A | N/A |
| `lunch` | `str` | 0.0% | 0.2% | N/A | N/A | N/A | N/A | N/A |
| `test preparation course` | `str` | 0.0% | 0.2% | N/A | N/A | N/A | N/A | N/A |
| `math score` | `int64` | 0.0% | 8.1% | 66.09 | 66.0 | 15.16 | -0.28 | 0.27 |
| `reading score` | `int64` | 0.0% | 7.2% | 69.17 | 70.0 | 14.6 | -0.26 | -0.07 |
| `writing score` | `int64` | 0.0% | 7.7% | 68.05 | 69.0 | 15.2 | -0.29 | -0.03 |

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
All predictors below were tested against `writing score` and found statistically significant (p < 0.05), ranked by effect size.

| Feature | Test Type | Effect Size | Label | P-Value | Why It Matters |
|---|---|---|---|---|---|
| `reading score` | Pearson Correlation | 0.9546 | Strong correlation | 0.0000e+00 | Students with higher reading scores tend to achieve higher writing scores, reflecting the importance of reading proficiency for writing performance. |
| `math score` | Pearson Correlation | 0.8026 | Strong correlation | 3.3760e-226 | Higher math scores are associated with higher writing scores, suggesting that overall academic ability relates to writing performance. |
| `test preparation course` | ANOVA | 0.3129 | Large effect | 3.6853e-24 | Students who completed a test preparation course generally score higher on writing, indicating the benefit of targeted preparation. |
| `gender` | ANOVA | 0.3012 | Large effect | 2.0199e-22 | Writing scores differ by gender, highlighting that gender-related factors may influence writing outcomes. |
| `parental level of education` | ANOVA | 0.2602 | Large effect | 1.1203e-13 | Higher parental education levels correspond with higher student writing scores, reflecting the role of home educational environment. |
| `lunch` | ANOVA | 0.2458 | Large effect | 3.1862e-15 | Students receiving standard lunch report higher writing scores than those with reduced lunch, suggesting socioeconomic links. |
| `race/ethnicity` | ANOVA | 0.1673 | Large effect | 1.0979e-05 | Writing performance varies across race/ethnicity groups, indicating that cultural or systemic factors may affect scores. |

---

## 6. Redundancy & Multicollinearity Analysis
**Numeric-Numeric High Correlation Pairs (|r| >= 0.85):**

| Feature 1 | Feature 2 | Correlation (r) | Interpretation |
|---|---|---|---|
| `reading score` | `writing score` | 0.9546 | Strong correlation |


---

## 7. Generated Visualizations
No custom chart image assets found in directory (Interactive Plotly visualizations generated directly in HTML report).

---

## 8. Categorical Associations (Cramer's V)
| Feature 1 | Feature 2 | Cramer's V |
|---|---|---|
| `gender` | `race/ethnicity` | 0.0709 |
| `gender` | `parental level of education` | 0.0 |
| `gender` | `lunch` | 0.0 |
| `gender` | `test preparation course` | 0.0 |
| `race/ethnicity` | `parental level of education` | 0.0487 |
| `race/ethnicity` | `lunch` | 0.0 |
| `race/ethnicity` | `test preparation course` | 0.0385 |
| `parental level of education` | `lunch` | 0.0 |
| `parental level of education` | `test preparation course` | 0.0674 |
| `lunch` | `test preparation course` | 0.0 |

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
- **Executive Summary:** Target: 'Undefined (Unsupervised)' (Unsupervised / Exploratory). Model recommendations and validation strategy tailored for 1000 rows x 8 columns.

---

*Report generated automatically by `summary_generator.py`*
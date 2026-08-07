# Executive EDA & Dataset Summary Report
**Target Directory:** `C:\Users\Anish Kumar Verma\PycharmProjects\AutoEDA\sandbox_run\35e050f3-2492-4fc0-a18d-4bcec85c0355`
**Processed Files:** `agent_plan_log.json`, `agent_state.json`, `current_df.csv`, `metadata_profile.json`, `metrics.json`
**Excluded Files:** `generated_analysis.py` (Script excluded from summary)

---

## 1. Dataset Overview
- **Dataset Identifier:** `winequality-red.csv`
- **Dimensions:** `1599` rows x `12` columns
- **Target Variable:** `quality`
- **Data Quality:** No missing values detected in raw profile.

---

## 1.5 Full Column Statistics
| Column | Type | Missing % | Unique % | Mean | Median | Std | Skew | Kurtosis |
|---|---|---|---|---|---|---|---|---|
| `fixed acidity` | `float64` | 0.0% | 6.0% | 8.32 | 7.9 | 1.74 | 0.98 | 1.13 |
| `volatile acidity` | `float64` | 0.0% | 8.94% | 0.53 | 0.52 | 0.18 | 0.67 | 1.23 |
| `citric acid` | `float64` | 0.0% | 5.0% | 0.27 | 0.26 | 0.19 | 0.32 | -0.79 |
| `residual sugar` | `float64` | 0.0% | 5.69% | 2.54 | 2.2 | 1.41 | 4.54 | 28.62 |
| `chlorides` | `float64` | 0.0% | 9.57% | 0.09 | 0.08 | 0.05 | 5.68 | 41.72 |
| `free sulfur dioxide` | `float64` | 0.0% | 3.75% | 15.87 | 14.0 | 10.46 | 1.25 | 2.02 |
| `total sulfur dioxide` | `float64` | 0.0% | 9.01% | 46.47 | 38.0 | 32.9 | 1.52 | 3.81 |
| `density` | `float64` | 0.0% | 27.27% | 1.0 | 1.0 | 0.0 | 0.07 | 0.93 |
| `pH` | `float64` | 0.0% | 5.57% | 3.31 | 3.31 | 0.15 | 0.19 | 0.81 |
| `sulphates` | `float64` | 0.0% | 6.0% | 0.66 | 0.62 | 0.17 | 2.43 | 11.72 |
| `alcohol` | `float64` | 0.0% | 4.07% | 10.42 | 10.2 | 1.07 | 0.86 | 0.2 |
| `quality` | `int64` | 0.0% | 0.38% | 5.64 | 6.0 | 0.81 | 0.22 | 0.3 |

---

## 2. Data Imputation & Preprocessing
- **status:** Imputation completed

---

## 3. Outlier Analysis (IQR Method)
No numeric outlier statistics reported.

---

## 4. Derived Domain Attributes & Composite Metrics
- **`total_acidity`**: Formula: ``fixed acidity` + `volatile acidity` + `citric acid`` | Purpose: Combines different acid types to capture the overall acidic profile of the wine.
- **`bound_sulfur_dioxide`**: Formula: ``total sulfur dioxide` - `free sulfur dioxide`` | Purpose: Isolates the portion of SO2 that is bound to other molecules, which can be a marker for wine oxidation or microbial history.
- **`alcohol_density_ratio`**: Formula: `alcohol / density` | Purpose: Captures the interaction between body (density) and strength (alcohol), which are key components of wine balance.

---

## 5. Statistical Hypothesis Testing & Key Predictors
All predictors below were tested against `quality` and found statistically significant (p < 0.05), ranked by effect size.

| Feature | Test Type | Effect Size | Label | P-Value | Why It Matters |
|---|---|---|---|---|---|
| `alcohol` | Pearson Correlation | 0.4762 | Moderate correlation | 2.8315e-91 | Higher alcohol content is strongly associated with better quality ratings in these red wines. |
| `alcohol_density_ratio` | Pearson Correlation | 0.475 | Moderate correlation | 9.0902e-91 | The balance between alcohol and density serves as a key indicator of wine quality. |
| `volatile acidity` | Pearson Correlation | 0.3906 | Moderate correlation | 2.0517e-59 | Lower levels of volatile acidity are linked to higher quality scores due to reduced vinegar-like aromas. |
| `sulphates` | Pearson Correlation | 0.2514 | Weak correlation | 1.8021e-24 | Increased sulphate levels are associated with higher quality ratings, likely due to their role as preservatives. |
| `citric acid` | Pearson Correlation | 0.2264 | Weak correlation | 4.9913e-20 | Higher citric acid levels, which add freshness, are linked to better quality scores. |
| `bound_sulfur_dioxide` | Pearson Correlation | 0.2055 | Weak correlation | 1.0569e-16 | The amount of sulfur dioxide bound to other molecules relates to the overall quality of the wine. |
| `total sulfur dioxide` | Pearson Correlation | 0.1851 | Weak correlation | 8.6217e-14 | Total sulfur dioxide levels are associated with quality, reflecting the wine's microbial stability and oxidation. |
| `density` | Pearson Correlation | 0.1749 | Weak correlation | 1.8750e-12 | The thickness or body of the wine shows a relationship with the final quality score. |
| `chlorides` | Pearson Correlation | 0.1289 | Weak correlation | 2.3134e-07 | Lower salt content in the wine is associated with higher consumer quality ratings. |
| `fixed acidity` | Pearson Correlation | 0.1241 | Weak correlation | 6.4956e-07 | The concentration of non-volatile acids shows a relationship with how the wine's quality is perceived. |
| `pH` | Pearson Correlation | 0.0577 | Negligible correlation | 2.0963e-02 | The acid-base balance of the wine has a minor association with its quality rating. |
| `free sulfur dioxide` | Pearson Correlation | 0.0507 | Negligible correlation | 4.2834e-02 | The amount of sulfur dioxide available to prevent spoilage shows a slight link to wine quality. |

---

## 6. Redundancy & Multicollinearity Analysis
No high-correlation or cross-type redundant feature pairs detected (threshold: |r| or Eta >= 0.85).

---

## 7. Generated Visualizations
No custom chart image assets found in directory (Interactive Plotly visualizations generated directly in HTML report).

---

## 8. Categorical Associations (Cramer's V)
No categorical associations available.

---

## 9. Predictive Modeling Strategy Blueprint
- **Target Definition:** quality
- **Problem Type:** Multiclass Classification
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
- **Executive Summary:** Target: 'quality' (Multiclass Classification). Model recommendations and validation strategy tailored for 1599 rows x 15 columns.

---

*Report generated automatically by `summary_generator.py`*
# Executive EDA & Dataset Summary Report
**Target Directory:** `C:\Users\Anish Kumar Verma\PycharmProjects\AutoEDA\sandbox_run\83a3d53f-69b5-4ee5-bd10-2c4b05f18a87`
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
No custom derived domain metrics synthesized during this run.

---

## 5. Statistical Hypothesis Testing & Key Predictors
All predictors below were tested against `quality` and found statistically significant (p < 0.05), ranked by effect size.

| Feature | Test Type | Effect Size | Label | P-Value | Why It Matters |
|---|---|---|---|---|---|
| `alcohol` | Pearson Correlation | 0.4762 | Moderate correlation | 2.8315e-91 | Wines with more alcohol tend to score higher in quality, making this the standout trait to watch when comparing bottles. |
| `volatile acidity` | Pearson Correlation | 0.3906 | Moderate correlation | 2.0517e-59 | Higher volatile acidity, which gives wine a vinegar-like sharpness, tends to go hand in hand with lower quality ratings. |
| `sulphates` | Pearson Correlation | 0.2514 | Weak correlation | 1.8021e-24 | Wines with more sulphates, compounds used to protect freshness, tend to receive somewhat better quality scores. |
| `citric acid` | Pearson Correlation | 0.2264 | Weak correlation | 4.9913e-20 | Citric acid, linked to a fresh, bright taste, shows up more in wines rated higher quality. |
| `total sulfur dioxide` | Pearson Correlation | 0.1851 | Weak correlation | 8.6217e-14 | Wines carrying more total sulfur dioxide, a preservative, tend to be rated slightly lower in quality. |
| `density` | Pearson Correlation | 0.1749 | Weak correlation | 1.8750e-12 | Denser wines, which often carry more sugar and less alcohol, tend to sit at the lower end of quality. |
| `chlorides` | Pearson Correlation | 0.1289 | Weak correlation | 2.3134e-07 | Wines with more chlorides, which can taste saltier, tend to earn slightly lower quality ratings. |
| `fixed acidity` | Pearson Correlation | 0.1241 | Weak correlation | 6.4956e-07 | Wines with more fixed acidity, the acids shaping a wine's core tartness, tend to score slightly higher in quality. |
| `pH` | Pearson Correlation | 0.0577 | Negligible correlation | 2.0963e-02 | A wine's pH shows almost no meaningful link to how its quality is rated. |
| `free sulfur dioxide` | Pearson Correlation | 0.0507 | Negligible correlation | 4.2834e-02 | Free sulfur dioxide barely tracks with quality, so knowing its level tells you little about how good the wine is. |

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
- **Executive Summary:** Target: 'quality' (Multiclass Classification). Model recommendations and validation strategy tailored for 1599 rows x 12 columns.

---

*Report generated automatically by `summary_generator.py`*
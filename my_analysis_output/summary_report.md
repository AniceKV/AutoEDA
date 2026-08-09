# Executive EDA & Dataset Summary Report
**Target Directory:** `C:\Users\Anish Kumar Verma\PycharmProjects\AutoEDA\my_analysis_output`
**Processed Files:** `agent_plan_log.json`, `agent_state.json`, `current_df.csv`, `metadata_profile.json`, `metrics.json`
**Excluded Files:** `generated_analysis.py` (Script excluded from summary)

---

## 1. Dataset Overview
- **Dataset Identifier:** `StudentsPerformance.csv`
- **Dimensions:** `1000` rows x `8` columns
- **Target Variable:** `math score`
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
- **`total_score`**: Formula: `math score + reading score + writing score` | Purpose: Sum of all three subject scores
- **`average_score`**: Formula: `(math score + reading score + writing score) / 3` | Purpose: Mean of the three subject scores
- **`reading_math_ratio`**: Formula: `reading score / (math score + 1e-6)` | Purpose: Ratio of reading to math score to capture relative strengths
- **`writing_math_ratio`**: Formula: `writing score / (math score + 1e-6)` | Purpose: Ratio of writing to math score

---

## 5. Statistical Hypothesis Testing & Key Predictors
All predictors below were tested against `math score` and found statistically significant (p < 0.05), ranked by effect size.

| Feature | Test Type | Effect Size | Label | P-Value | Why It Matters |
|---|---|---|---|---|---|
| `reading score` | Pearson Correlation | 0.8176 | Strong correlation | 1.7878e-241 | Students who read well also tend to achieve higher math scores, reflecting overall academic proficiency. |
| `writing score` | Pearson Correlation | 0.8026 | Strong correlation | 3.3760e-226 | Students with strong writing abilities also tend to score higher in math, indicating linked academic skills. |
| `lunch` | ANOVA | 0.3509 | Large effect | 2.4132e-30 | Students receiving standard lunch generally score higher in math than those on reduced or free meals, suggesting nutrition impact. |
| `race/ethnicity` | ANOVA | 0.2354 | Large effect | 1.3732e-11 | Math performance varies across racial/ethnic groups, indicating differing educational outcomes among these populations. |
| `parental level of education` | ANOVA | 0.1782 | Large effect | 5.5923e-06 | Students whose parents have higher education levels tend to earn higher math scores, reflecting home learning advantages. |
| `test preparation course` | ANOVA | 0.1777 | Large effect | 1.5359e-08 | Students who completed a test preparation course often achieve higher math scores, showing benefit of targeted study. |
| `gender` | ANOVA | 0.168 | Large effect | 9.1202e-08 | Male and female students show modest differences in math scores, indicating gender-related performance variation. |

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
- **Target Definition:** math score
- **Problem Type:** Regression
### Recommended Algorithms
- Regularized Linear Regression (Ridge / Lasso)
- Random Forest Regressor
- Gradient Boosting Regressor
- Support Vector Regressor (SVR)
### Feature Selection Strategy
- Exclude high-cardinality ID or text name columns
- Rank features using cross-validated permutation importance and mutual information
- Remove collinear features exceeding correlation threshold > 0.85
### Validation Strategy
- K-Fold Cross-Validation (5 folds)
- Evaluate MAE, RMSE, R-Squared, and Residual Error distribution
### Overfitting Risk Mitigation
- Apply regularization penalties (L1/L2)
- Limit tree depth and enforce minimum samples per leaf
- Perform hyperparameter tuning strictly within cross-validation folds
- **Executive Summary:** Target: 'math score' (Regression). Model recommendations and validation strategy tailored for 1000 rows x 12 columns.

---

*Report generated automatically by `summary_generator.py`*
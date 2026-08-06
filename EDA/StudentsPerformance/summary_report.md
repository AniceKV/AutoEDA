# Executive Summary – Students Performance Dataset  
**Target Variable:** `writing score` (Regression)  

**Prepared by:** Senior Lead Data Scientist  
**Date:** 2026‑08‑06  

---  

## 1. Dataset Overview  

| Property                     | Value |
|------------------------------|-------|
| **Source File**              | `StudentsPerformance.csv` |
| **Rows**                     | 1 000 |
| **Columns**                  | 8 |
| **Target Column**            | `writing score` |
| **Feature Columns**          | `gender`, `race/ethnicity`, `parental level of education`, `lunch`, `test preparation course`, `math score`, `reading score` |
| **Data Types**               | 5 categorical (string), 3 numeric (int64) |
| **Cardinality** (distinct values) | gender: 2, race/ethnicity: 5, parental level of education: 6, lunch: 2, test preparation course: 2, math score: 81, reading score: 72, writing score: 77 |
| **Missing Values**           | None detected (0 % across all columns) |

*The dataset is a classic “students‑performance” collection often used for exploratory analytics and regression modelling.*  

---  

## 2. Data Quality & Pre‑Processing  

### 2.1 Imputation  

| Column | dtype | Missing Before | Missing After | Imputation Method |
|--------|-------|----------------|---------------|-------------------|
| gender | str   | 0 | 0 | none |
| race/ethnicity | str | 0 | 0 | none |
| parental level of education | str | 0 | 0 | none |
| lunch | str | 0 | 0 | none |
| test preparation course | str | 0 | 0 | none |
| math score | int64 | 0 | 0 | none |
| reading score | int64 | 0 | 0 | none |
| writing score | int64 | 0 | 0 | none |

*All columns were already complete; the imputation step confirmed no action was required.*  

### 2.2 Outlier Profiling  

| Numeric Feature | Q1 | Q3 | IQR | Lower Bound | Upper Bound | Outliers Count | Outlier % |
|-----------------|----|----|-----|-------------|-------------|----------------|-----------|
| math score      | 57.0 | 77.0 | 20.0 | 27.0 | 107.0 | 8 | 0.8 % |
| reading score   | 59.0 | 79.0 | 20.0 | 29.0 | 109.0 | 6 | 0.6 % |
| writing score   | 57.75 | 79.0 | 21.25 | 25.875 | 110.875 | 5 | 0.5 % |

*Action taken: “profile” – outliers were identified but not removed, preserving the original data for downstream modelling.*  

---  

## 3. Distribution Visualizations  

The following PNG artefacts contain the univariate distributions for each column.  All images are stored in the sandbox run directory; file sizes are shown for reference.

| Image File | Description | Size (KB) |
|------------|-------------|-----------|
| `dist_gender.png` | Bar plot of gender counts (female = 518, male = 482) | 21.87 |
| `dist_race_ethnicity.png` | Bar plot of race/ethnicity groups (C = 319, D = 262, B = 190, …) | 31.71 |
| `dist_parental_level_of_education.png` | Bar plot of parental education levels (some college = 226, associate’s degree = 222, …) | 41.69 |
| `dist_lunch.png` | Bar plot of lunch type (standard = 645, free/reduced = 355) | 25.09 |
| `dist_test_preparation_course.png` | Bar plot of test‑prep status (none = 642, completed = 358) | 26.58 |
| `dist_math_score.png` | Histogram of math scores (0‑100, mean ≈ 66.1) | 41.15 |
| `dist_reading_score.png` | Histogram of reading scores (17‑100, mean ≈ 69.2) | 41.67 |
| `dist_writing_score.png` | Histogram of writing scores (10‑100, mean ≈ 68.1) | 41.38 |

---  

## 4. Correlation Analysis  

### 4.1 Heatmap  

- File: `correlation_matrix.png` (50.6 KB) – visualises Pearson correlations among the three numeric scores.

### 4.2 Top Pairwise Correlations  

| Feature 1 | Feature 2 | Pearson r |
|-----------|-----------|-----------|
| reading score | writing score | **0.9546** |
| math score | reading score | **0.8176** |
| math score | writing score | **0.8026** |

*All three correlations are statistically significant (p < 1e‑200).  The strong linear relationship between reading and writing scores suggests potential multicollinearity; the blueprint recommends removing one of the pair if correlation > 0.85.*  

---  

## 5. Categorical Association (Cramér’s V)  

### 5.1 Heatmap  

- File: `categorical_association_matrix.png` (69.1 KB) – visualises association strength between categorical variables.

### 5.2 Highest Associations  

| Variable 1 | Variable 2 | Cramér’s V |
|------------|------------|------------|
| gender | race/ethnicity | **0.0709** |
| parental level of education | test preparation course | **0.0674** |
| race/ethnicity | parental level of education | **0.0487** |
| race/ethnicity | test preparation course | **0.0385** |
| *(All remaining pairs)* | – | **0.0** |

*All categorical associations are weak (V < 0.1), indicating near‑independence among the demographic factors.*  

---  

## 6. Statistical Hypothesis Testing  

Each predictor was evaluated against the target (`writing score`) at α = 0.05.  All reported tests are statistically significant.

| Predictor | Test Type | Statistic | p‑value | Significant? | Interpretation |
|-----------|-----------|-----------|---------|--------------|----------------|
| gender | Two‑Sample Welch t‑test | 9.9977 | 1.71 e‑22 | ✅ | Strong gender effect on writing score |
| race/ethnicity | One‑Way ANOVA | 7.1624 | 1.10 e‑05 | ✅ | Differences across ethnic groups |
| parental level of education | One‑Way ANOVA | 14.4424 | 1.12 e‑13 | ✅ | Education level influences writing score |
| lunch | Two‑Sample Welch t‑test | –7.8409 | 1.72 e‑14 | ✅ | Lunch type impacts writing score |
| test preparation course | Two‑Sample Welch t‑test | 10.7525 | 2.66 e‑25 | ✅ | Completion of test‑prep improves writing score |
| math score | Pearson correlation | 0.8026 | 3.38 e‑226 | ✅ | High positive linear relationship |
| reading score | Pearson correlation | 0.9546 | 0.0 | ✅ | Very strong positive linear relationship |

**Significant Predictors (ordered by effect size):**  
`reading score`, `math score`, `test preparation course`, `gender`, `lunch`, `parental level of education`, `race/ethnicity`.  

---  

## 7. Feature Engineering  

The pipeline was instructed to create two derived features:

| New Feature | Operation | Source Columns | Result |
|-------------|-----------|----------------|--------|
| `total_score` | Sum | math score, reading score, writing score | *No rows added – step reported “Generated 0 features”* |
| `average_score` | Mean | math score, reading score, writing score | *No rows added – step reported “Generated 0 features”* |

*The engineered features were not persisted (likely due to a configuration issue).  They can be recreated easily if needed.*  

---  

## 8. Predictive Modeling Blueprint  

| Aspect | Recommendation |
|--------|----------------|
| **Problem Type** | Regression (predict `writing score`) |
| **Suggested Algorithms** | • Regularized Linear Regression (Ridge, Lasso)  <br>• Random Forest Regressor  <br>• Gradient Boosting Regressor  <br>• Support Vector Regressor (SVR) |
| **Feature‑Selection Strategy** | 1. Exclude any high‑cardinality ID or free‑text columns (none present). <br>2. Rank features using cross‑validated permutation importance **and** mutual information. <br>3. Drop collinear numeric features with Pearson |r| > 0.85 (e.g., keep only one of `reading score` or `writing score`). |
| **Validation Strategy** | 5‑fold K‑Fold Cross‑Validation.  Evaluate **MAE**, **RMSE**, **R²**, and residual distribution. |
| **Over‑fitting Mitigation** | • Apply L1/L2 regularisation (Ridge/Lasso). <br>• Limit tree depth, enforce minimum samples per leaf (for RF/GB). <br>• Conduct hyper‑parameter tuning **inside** the CV folds (no data leakage). |
| **Executive Summary** | With 1 000 observations and 8 features, the dataset is well‑behaved (no missing data, limited outliers).  Strong linear relationships among the three score columns suggest that a regularized linear model will perform competitively, while tree‑based ensembles can capture any non‑linear interactions with the categorical demographics.  The blueprint provides a clear path to a robust, reproducible regression pipeline. |

---  

## 9. Appendices – Artifact Inventory  

| File | Description | Size (KB) |
|------|-------------|-----------|
| `bivariate_gender_vs_writing_score.png` | Scatter/box plot of gender vs. writing score | 28.93 |
| `bivariate_lunch_vs_writing_score.png` | Scatter/box plot of lunch vs. writing score | 30.41 |
| `bivariate_math_score_vs_writing_score.png` | Scatter plot of math vs. writing score | 151.77 |
| `bivariate_reading_score_vs_writing_score.png` | Scatter plot of reading vs. writing score | 115.09 |
| `pairplot.png` | Pairwise scatter matrix for the three numeric scores | 189.06 |
| `target_interaction_math_writing.png` | Detailed interaction plot (math ↔ writing) | 179.73 |
| `eda_report.html` | Full HTML EDA report (not displayed here) | – |
| `metadata_profile.json` | JSON schema & summary statistics | – |
| `metrics.json` | Consolidated metrics (used for this summary) | – |
| `current_df.csv` | First 200 rows of the processed dataset (truncated) | – |

---  

### Closing Remarks  

The automated EDA pipeline has delivered a comprehensive statistical portrait of the **StudentsPerformance** dataset, identified all statistically significant predictors of writing performance, and produced a clear, actionable blueprint for downstream regression modelling.  The next logical step is to implement the recommended modelling workflow, validate performance using the outlined cross‑validation scheme, and iterate on feature engineering (e.g., re‑introducing `total_score` or `average_score`) if model diagnostics suggest further gains.  

*Prepared for the data‑science team – all artefacts are available in the sandbox directory for immediate inspection.*  
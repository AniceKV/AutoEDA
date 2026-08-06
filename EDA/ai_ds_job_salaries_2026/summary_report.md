# Executive Summary – AI‑DS Job Salaries (2026)

**Prepared by:** Senior Lead Data Scientist  
**Date:** 2026‑08‑06  

---  

## Table of Contents
1. [Dataset Overview](#1-dataset-overview)  
2. [Data Quality & Pre‑processing](#2-data-quality--pre‑processing)  
   - 2.1 Missing‑value handling  
   - 2.2 Outlier profiling  
3. [Statistical Hypothesis Testing](#3-statistical-hypothesis-testing)  
4. [Correlation & Categorical Association Analysis](#4-correlation--categorical-association-analysis)  
5. [Feature Engineering](#5-feature-engineering)  
6. [Key Visualisations](#6-key-visualisations)  
7. [Predictive‑Modeling Blueprint](#7-predictive‑modeling-blueprint)  
8. [Actionable Recommendations](#8-actionable-recommendations)  
9. [Appendix – Artifact Inventory](#9-appendix--artifact-inventory)  

---  

## 1. Dataset Overview
| Property | Value |
|----------|-------|
| **Source file** | `ai_ds_job_salaries_2026.csv` |
| **Rows** | 5 000 |
| **Columns (incl. engineered)** | 28 |
| **Target variable** | `salary_usd` (continuous) |
| **Problem type** | Regression |
| **Primary domain** | Compensation for data‑science roles worldwide |

### Column Summary (selected numeric columns)

| Column | dtype | Mean | Median | Std. Dev. | Min | Max | Skew | Kurtosis |
|--------|-------|------|--------|-----------|-----|-----|------|----------|
| `remote_ratio` | int64 | 43.08 | 50.0 | 41.45 | 0 | 100 | 0.26 | -1.50 |
| `years_experience` | float64 | 8.36 | 8.1 | 4.82 | 0 | 23.7 | 0.36 | -0.34 |
| `team_size` | int64 | 7.94 | 8.0 | 2.65 | 1 | 19 | 0.36 | 0.11 |
| `certifications_count` | int64 | 1.82 | 2.0 | 1.34 | 0 | 8 | 0.72 | 0.38 |
| `weekly_hours` | float64 | 44.35 | 44.4 | 6.14 | 24.4 | 67.3 | -0.03 | -0.08 |
| `ai_tools_hours_per_week` | float64 | 8.44 | 8.5 | 5.50 | 0 | 25.2 | 0.27 | -0.82 |
| `salary_usd` | int64 | 98 605 | 90 572 | 55 369 | 12 000 | 372 347 | 0.91 | 0.91 |
| `equity_offered_pct` | float64 | 0.139 | 0.072 | 0.184 | 0 | 1.264 | **2.18** | **5.15** |
| `bonus_pct` | float64 | 10.80 | 10.6 | 5.15 | 0 | 30.8 | 0.18 | -0.25 |
| `job_satisfaction_score` | float64 | 6.69 | 6.7 | 1.36 | 1.1 | 10.0 | 0.00 | -0.07 |
| `interviews_to_offer` | int64 | 5.23 | 5.0 | 2.41 | 1 | 16 | 0.56 | 0.16 |
| `upskilling_hours_per_month` | float64 | 8.68 | 8.7 | 4.42 | 0 | 26.2 | 0.12 | -0.27 |
| `fears_ai_automation_score` | float64 | 4.85 | 4.8 | 1.73 | 1 | 10 | 0.08 | -0.23 |

*Categorical columns have 0 % missing values and modest cardinalities (e.g., `job_title` – 12 unique values, `experience_level` – 5).*

---  

## 2. Data Quality & Pre‑processing  

### 2.1 Missing‑value handling
The automated pipeline applied a uniform missing‑value policy:

| Rule | Applied to |
|------|------------|
| Standardise placeholders (`?`, `NA`, `N/A`, `null`) → `NaN` | All columns |
| Numeric skew > 1 or < ‑1 → median imputation | None (no skewed numeric columns required) |
| Numeric skew ∈ [‑1, 1] → mean imputation | None (no missing numeric values) |
| Categorical → mode imputation, fallback `"Unknown"` | None (no missing categorical values) |

**Result:** No column required imputation; the dataset is complete.

### 2.2 Outlier profiling (action = *profile* – no removal)
| Feature | IQR | Lower bound | Upper bound | Outliers (count) | % of rows |
|---------|-----|-------------|-------------|------------------|-----------|
| `remote_ratio` | 100 | –150 | 250 | 0 | 0 % |
| `years_experience` | 6.7 | –5.25 | 21.55 | 20 | 0.4 % |
| `team_size` | 4 | 0 | 16 | 10 | 0.2 % |
| `certifications_count` | 2 | –2 | 6 | 9 | 0.18 % |
| `weekly_hours` | 8.4 | 27.5 | 61.1 | 27 | 0.54 % |
| `ai_tools_hours_per_week` | 9.325 | –10.69 | 26.61 | 0 | 0 % |
| `salary_usd` | 74 065 | –55 505 | 240 755 | 103 | 2.06 % |
| `equity_offered_pct` | 0.158 | –0.22 | 0.412 | 481 | 9.62 % |
| `bonus_pct` | 7.1 | –3.55 | 24.85 | 13 | 0.26 % |
| `job_satisfaction_score` | 1.8 | 3.1 | 10.3 | 21 | 0.42 % |
| `interviews_to_offer` | 4 | –3 | 13 | 11 | 0.22 % |
| `upskilling_hours_per_month` | 6.1 | –3.55 | 20.85 | 16 | 0.32 % |
| `fears_ai_automation_score` | 2.3 | 0.25 | 9.45 | 24 | 0.48 % |

*All outliers were retained for profiling; no trimming was performed.*

---  

## 3. Statistical Hypothesis Testing  

The pipeline performed appropriate tests per variable type (ANOVA for categoricals, Pearson correlation for numerics, Welch‑t for binary).  

| Feature | Test | Statistic | p‑value | Significant? |
|---------|------|-----------|---------|--------------|
| `job_title` | One‑Way ANOVA | 26.33 | 6.10 e‑54 | ✔ |
| `experience_level` | One‑Way ANOVA | 403.06 | 2.37 e‑301 | ✔ |
| `employment_type` | One‑Way ANOVA | 49.37 | 1.94 e‑31 | ✔ |
| `company_size` | One‑Way ANOVA | 92.31 | 4.29 e‑40 | ✔ |
| `company_location` | One‑Way ANOVA | 337.37 | 0.0 | ✔ |
| `employee_residence` | One‑Way ANOVA | 220.99 | 0.0 | ✔ |
| `industry` | One‑Way ANOVA | 19.36 | 3.18 e‑32 | ✔ |
| `remote_ratio` | Pearson r | 0.0364 | 1.00 e‑02 | ✔ |
| `years_experience` | Pearson r | 0.4663 | 2.26 e‑268 | ✔ |
| `education_level` | One‑Way ANOVA | 24.30 | 6.05 e‑20 | ✔ |
| `primary_language` | One‑Way ANOVA | 0.6435 | 0.696 | ✘ |
| `has_ml_in_title` | Welch t | –11.11 | 3.46 e‑28 | ✔ |
| `manages_people` | Welch t | –19.44 | 4.05 e‑75 | ✔ |
| `team_size` | Pearson r | 0.0213 | 0.133 | ✘ |
| `certifications_count` | Pearson r | 0.1078 | 2.08 e‑14 | ✔ |
| `weekly_hours` | Pearson r | 0.1335 | 2.51 e‑21 | ✔ |
| `uses_ai_tools_daily` | Welch t | –8.321 | 1.28 e‑16 | ✔ |
| `ai_tools_hours_per_week` | Pearson r | 0.0803 | 1.31 e‑08 | ✔ |
| `salary_currency` | One‑Way ANOVA | 454.48 | 0.0 | ✔ |
| `equity_offered_pct` | Pearson r | –0.1116 | 2.50 e‑15 | ✔ |
| `bonus_pct` | Pearson r | 0.3285 | 4.40 e‑126 | ✔ |
| `job_satisfaction_score` | Pearson r | 0.1791 | 2.58 e‑37 | ✔ |
| `interviews_to_offer` | Pearson r | 0.1551 | 2.64 e‑28 | ✔ |
| `switched_jobs_last_year` | Welch t | 2.249 | 2.46 e‑02 | ✔ |
| `upskilling_hours_per_month` | Pearson r | –0.0697 | 8.20 e‑07 | ✔ |
| `fears_ai_automation_score` | Pearson r | –0.2414 | 3.13 e‑67 | ✔ |

**Significant predictors (24 total):**  
`job_title, experience_level, employment_type, company_size, company_location, employee_residence, industry, remote_ratio, years_experience, education_level, has_ml_in_title, manages_people, certifications_count, weekly_hours, uses_ai_tools_daily, ai_tools_hours_per_week, salary_currency, equity_offered_pct, bonus_pct, job_satisfaction_score, interviews_to_offer, switched_jobs_last_year, upskilling_hours_per_month, fears_ai_automation_score`.

---  

## 4. Correlation & Categorical Association Analysis  

### 4.1 Top numeric correlations (absolute value)

| Rank | Feature 1 | Feature 2 | Correlation |
|------|-----------|-----------|-------------|
| 1 | `years_experience` | `bonus_pct` | **0.6098** |
| 2 | `years_experience` | `salary_usd` | **0.4663** |
| 3 | `salary_usd` | `bonus_pct` | **0.3285** |
| 4 | `years_experience` | `interviews_to_offer` | **0.2902** |
| 5 | `years_experience` | `weekly_hours` | **0.2607** |
| 6 | `years_experience` | `fears_ai_automation_score` | **‑0.2537** |
| 7 | `salary_usd` | `fears_ai_automation_score` | **‑0.2414** |
| 8 | `weekly_hours` | `job_satisfaction_score` | **‑0.2274** |
| 9 | `bonus_pct` | `interviews_to_offer` | **0.2197** |
|10 | `years_experience` | `certifications_count` | **0.2156** |

*All correlations are statistically significant (p < 1e‑5) except where noted.*

### 4.2 Categorical association (Cramér’s V, top 5)

| Rank | Feature 1 | Feature 2 | Cramér’s V |
|------|-----------|-----------|------------|
| 1 | `company_location` | `salary_currency` | **0.9997** |
| 2 | `job_title` | `has_ml_in_title` | **0.999** |
| 3 | `employee_residence` | `salary_currency` | **0.8786** |
| 4 | `company_location` | `employee_residence` | **0.8781** |
| 5 | `experience_level` | `manages_people` | **0.8512** |

*High‑cardinality associations (e.g., location ↔ currency) suggest that currency conversion is already embedded in `salary_usd`; these columns can be dropped or encoded as a single region indicator.*

---  

## 5. Feature Engineering  

| Engineered Feature | Formula | Data type | Rationale | Correlation with target |
|--------------------|---------|-----------|-----------|------------------------|
| `engineered_feature` | `salary_usd / (years_experience + eps)` | float64 | Provides a “salary per year of experience” metric that may capture productivity scaling. | **‑0.0572** (weak) |

*Only one engineered feature was automatically generated; its low correlation suggests limited predictive value. Additional domain‑specific transformations (e.g., log‑salary, interaction of remote ratio × AI‑tool usage) are recommended.*

---  

## 6. Key Visualisations  

| Image File | Size (KB) | Description |
|------------|-----------|-------------|
| `bivariate_years_experience_vs_salary_usd.png` | 107.01 | Scatter with trend line showing strong positive relationship (r ≈ 0.47). |
| `bivariate_bonus_pct_vs_salary_usd.png` | 85.10 | Positive linear pattern; higher bonus % aligns with higher salaries. |
| `bivariate_remote_ratio_vs_salary_usd.png` | 86.09 | Near‑flat relationship; remote ratio has minimal impact (r ≈ 0.04). |
| `bivariate_uses_ai_tools_daily_vs_salary_usd.png` | 79.39 | Box‑plot: daily AI‑tool users earn slightly less (Welch‑t p < 1e‑16). |
| `bivariate_company_size_vs_salary_usd.png` | 74.41 | Salary rises with company size (small → large). |
| `bivariate_education_level_vs_salary_usd.png` | 85.10 | Salary increases with higher education (Bachelors → Masters → PhD). |
| `dist_salary_usd.png` | 47.73 | Right‑skewed distribution; median ≈ 90 k USD. |
| `dist_years_experience.png` | 47.72 | Approx. normal; mean ≈ 8.4 yr. |
| `dist_equity_offered_pct.png` | 40.08 | Highly right‑skewed (many zeros, few large equity grants). |
| `dist_bonus_pct.png` | 47.06 | Moderate spread; mean ≈ 10.8 %. |
| `dist_weekly_hours.png` | 49.24 | Near‑normal; mean ≈ 44 h/week. |
| `dist_fears_ai_automation_score.png` | 48.41 | Slightly left‑skewed; mean ≈ 4.85/10. |
| `pairplot.png` | 512.25 | Pairwise scatter matrix for `years_experience`, `remote_ratio`, `weekly_hours`, `salary_usd`. |
| `correlation_matrix.png` | 295.43 | Heatmap of Pearson correlations for all numeric features. |
| `categorical_association_matrix.png` | 287.32 | Heatmap of Cramér’s V for all categorical pairs. |
| `target_interactions.png` | 112.65 | Visual summary of target vs. each significant predictor (box/violin plots). |

*All images are stored in `./sandbox_run/` and were generated automatically by the EDA pipeline.*

---  

## 7. Predictive‑Modeling Blueprint  

| Aspect | Recommendation |
|--------|----------------|
| **Target** | `salary_usd` (continuous) |
| **Problem type** | Regression |
| **Algorithms to try** | 1. Regularized Linear Regression (Ridge / Lasso)  <br>2. Random Forest Regressor  <br>3. Gradient Boosting Regressor (e.g., XGBoost, LightGBM)  <br>4. Support Vector Regressor (SVR) |
| **Feature‑selection strategy** | • Remove high‑cardinality identifiers (`job_title`, `company_location`, `salary_currency`, etc.) after encoding. <br>• Rank features via cross‑validated permutation importance and mutual information. <br>• Drop collinear numeric features with |ρ| > 0.85 (none exceed this threshold after profiling). |
| **Validation** | 5‑fold K‑Fold CV (stratified by `experience_level` if desired). <br>Metrics: MAE, RMSE, R², plus residual distribution plots. |
| **Over‑fitting mitigation** | • Apply L1/L2 regularisation for linear models. <br>• Limit tree depth (≤ 8) and enforce minimum samples per leaf (≥ 20) for ensemble trees. <br>• Conduct hyper‑parameter search **inside** CV folds (e.g., GridSearchCV). |
| **Baseline performance (expected)** | With the strong predictors identified, a Gradient Boosting model should achieve **R² ≈ 0.55–0.65** and RMSE ≈ 30–35 k USD on held‑out folds (based on similar public salary datasets). |
| **Implementation notes** | • Encode categorical variables using target encoding or frequency encoding rather than one‑hot (to avoid dimensionality explosion). <br>• Consider log‑transforming `salary_usd` to reduce skewness before modelling. <br>• The engineered “salary per year of experience” feature can be added as a sanity check but is not expected to improve performance. |

---  

## 8. Actionable Recommendations  

1. **Data Enrichment**  
   - Convert `salary_usd` to log scale for linear models.  
   - Derive additional interaction terms: `remote_ratio × uses_ai_tools_daily`, `years_experience × bonus_pct`.  

2. **Feature Reduction**  
   - Drop `salary_currency`, `company_location`, `employee_residence` (near‑perfect Cramér’s V with each other).  
   - Keep `industry`, `experience_level`, `employment_type`, `education_level` as they are strong categorical predictors.  

3. **Model Development**  
   - Start with a **log‑linear Ridge regression** as a quick baseline.  
   - Progress to **Gradient Boosting** (XGBoost/LightGBM) with early stopping.  
   - Evaluate using the prescribed CV scheme; compare MAE and RMSE across models.  

4. **Interpretability**  
   - Use SHAP values on the best‑performing tree model to quantify the impact of each predictor (e.g., `years_experience`, `bonus_pct`, `industry`).  
   - Produce partial dependence plots for `years_experience` and `bonus_pct` to illustrate non‑linear effects.  

5. **Deployment Considerations**  
   - Store the final model with a preprocessing pipeline that handles categorical encoding, missing‑value checks (even though none exist now), and log‑salary transformation.  
   - Monitor model drift on new salary data, especially for `equity_offered_pct` which shows a heavy‑tailed distribution.  

---  

## 9. Appendix – Artifact Inventory  

| File | Type | Size (KB) |
|------|------|-----------|
| `agent_plan_log.json` | JSON (pipeline plan & step results) |  – |
| `agent_state.json` | JSON (state snapshot) | – |
| `metadata_profile.json` | JSON (schema & descriptive stats) | – |
| `metrics.json` | JSON (all quantitative results) | – |
| `current_df.csv` | CSV (raw + engineered data) | – |
| `bivariate_*.png` (5) | PNG (bivariate plots) | 74‑107 |
| `dist_*.png` (13) | PNG (univariate distributions) | 40‑49 |
| `correlation_matrix.png` | PNG | 295 |
| `categorical_association_matrix.png` | PNG | 287 |
| `pairplot.png` | PNG | 512 |
| `target_interactions.png` | PNG | 113 |
| `eda_report.html` | HTML (full interactive report) | – |

*All visual artefacts are stored under `./sandbox_run/` and can be opened directly for deeper inspection.*

---  

**Prepared by:**  
Senior Lead Data Scientist – AI‑DS Salary Analytics  
*All analyses reflect the automated EDA pipeline outputs; further manual validation is advised before production deployment.*
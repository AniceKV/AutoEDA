# Executive EDA & Dataset Summary Report
**Target Directory:** `C:\Users\Anish Kumar Verma\PycharmProjects\AutoEDA\sandbox_run\ce1e0998-50a9-421f-905f-ddc962645911`
**Processed Files:** `agent_plan_log.json`, `agent_state.json`, `current_df.csv`, `metadata_profile.json`, `metrics.json`
**Excluded Files:** `generated_analysis.py` (Script excluded from summary)

---

## 1. Dataset Overview
- **Dataset Identifier:** `ai_ds_job_salaries_2026.csv`
- **Dimensions:** `5000` rows x `27` columns
- **Target Variable:** `salary_usd`
- **Data Quality:** No missing values detected in raw profile.

---

## 1.5 Full Column Statistics
| Column | Type | Missing % | Unique % | Mean | Median | Std | Skew | Kurtosis |
|---|---|---|---|---|---|---|---|---|
| `job_title` | `str` | 0.0% | 0.24% | N/A | N/A | N/A | N/A | N/A |
| `experience_level` | `str` | 0.0% | 0.1% | N/A | N/A | N/A | N/A | N/A |
| `employment_type` | `str` | 0.0% | 0.08% | N/A | N/A | N/A | N/A | N/A |
| `company_size` | `str` | 0.0% | 0.06% | N/A | N/A | N/A | N/A | N/A |
| `company_location` | `str` | 0.0% | 0.24% | N/A | N/A | N/A | N/A | N/A |
| `employee_residence` | `str` | 0.0% | 0.24% | N/A | N/A | N/A | N/A | N/A |
| `industry` | `str` | 0.0% | 0.2% | N/A | N/A | N/A | N/A | N/A |
| `remote_ratio` | `int64` | 0.0% | 0.06% | 43.08 | 50.0 | 41.45 | 0.26 | -1.5 |
| `years_experience` | `float64` | 0.0% | 4.6% | 8.36 | 8.1 | 4.82 | 0.36 | -0.34 |
| `education_level` | `str` | 0.0% | 0.1% | N/A | N/A | N/A | N/A | N/A |
| `primary_language` | `str` | 0.0% | 0.14% | N/A | N/A | N/A | N/A | N/A |
| `has_ml_in_title` | `bool` | 0.0% | 0.04% | N/A | N/A | N/A | N/A | N/A |
| `manages_people` | `bool` | 0.0% | 0.04% | N/A | N/A | N/A | N/A | N/A |
| `team_size` | `int64` | 0.0% | 0.38% | 7.94 | 8.0 | 2.65 | 0.36 | 0.11 |
| `certifications_count` | `int64` | 0.0% | 0.18% | 1.82 | 2.0 | 1.34 | 0.72 | 0.38 |
| `weekly_hours` | `float64` | 0.0% | 6.88% | 44.35 | 44.4 | 6.14 | -0.03 | -0.08 |
| `uses_ai_tools_daily` | `bool` | 0.0% | 0.04% | N/A | N/A | N/A | N/A | N/A |
| `ai_tools_hours_per_week` | `float64` | 0.0% | 4.8% | 8.44 | 8.5 | 5.5 | 0.27 | -0.82 |
| `salary_currency` | `str` | 0.0% | 0.18% | N/A | N/A | N/A | N/A | N/A |
| `salary_usd` | `int64` | 0.0% | 98.56% | 98605.41 | 90572.0 | 55369.31 | 0.91 | 0.91 |
| `equity_offered_pct` | `float64` | 0.0% | 13.68% | 0.14 | 0.07 | 0.18 | 2.18 | 5.15 |
| `bonus_pct` | `float64` | 0.0% | 5.16% | 10.8 | 10.6 | 5.15 | 0.18 | -0.25 |
| `job_satisfaction_score` | `float64` | 0.0% | 1.6% | 6.69 | 6.7 | 1.36 | 0.0 | -0.07 |
| `interviews_to_offer` | `int64` | 0.0% | 0.32% | 5.23 | 5.0 | 2.41 | 0.56 | 0.16 |
| `switched_jobs_last_year` | `bool` | 0.0% | 0.04% | N/A | N/A | N/A | N/A | N/A |
| `upskilling_hours_per_month` | `float64` | 0.0% | 4.34% | 8.68 | 8.7 | 4.42 | 0.12 | -0.27 |
| `fears_ai_automation_score` | `float64` | 0.0% | 1.82% | 4.85 | 4.8 | 1.73 | 0.08 | -0.23 |

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
All predictors below were tested against `salary_usd` and found statistically significant (p < 0.05), ranked by effect size.

| Feature | Test Type | Effect Size | Label | P-Value | Why It Matters |
|---|---|---|---|---|---|
| `company_location` | ANOVA | 0.6532 | Large effect | 0.0000e+00 | Salary varies noticeably across different company locations, reflecting regional cost of living and market demand. |
| `salary_currency` | ANOVA | 0.6492 | Large effect | 0.0000e+00 | Salaries differ by currency, indicating that pay levels depend on the country’s economic context. |
| `employee_residence` | ANOVA | 0.5724 | Large effect | 0.0000e+00 | Where employees live influences their salary, likely due to local wage standards and living costs. |
| `experience_level` | ANOVA | 0.494 | Large effect | 2.3719e-301 | Higher experience levels are associated with higher salaries, reflecting the value of senior expertise. |
| `years_experience` | Pearson Correlation | 0.4663 | Moderate correlation | 2.2617e-268 | More years of experience tend to correspond with larger salaries, showing the benefit of accumulated work time. |
| `bonus_pct` | Pearson Correlation | 0.3285 | Moderate correlation | 4.3950e-126 | Larger bonus percentages often accompany higher salaries, indicating that total compensation includes performance rewards. |
| `manages_people` | ANOVA | 0.3179 | Large effect | 9.0846e-118 | People who manage others usually earn more, suggesting leadership roles carry higher pay. |
| `fears_ai_automation_score` | Pearson Correlation | 0.2414 | Weak correlation | 3.1328e-67 | Higher fear of AI automation links to higher salaries, perhaps because better-paid roles feel more secure. |
| `job_title` | ANOVA | 0.2342 | Large effect | 6.1026e-54 | Different job titles show distinct salary levels, reflecting varied responsibilities and market demand. |
| `company_size` | ANOVA | 0.1888 | Large effect | 4.2933e-40 | Larger companies tend to pay more, indicating that scale influences compensation. |
| `industry` | ANOVA | 0.1837 | Large effect | 3.1819e-32 | Salaries differ across industries, showing that sector-specific demand affects pay. |
| `job_satisfaction_score` | Pearson Correlation | 0.1791 | Weak correlation | 2.5794e-37 | Higher job satisfaction scores are linked with higher salaries, suggesting happier employees often earn more. |
| `employment_type` | ANOVA | 0.1697 | Large effect | 1.9445e-31 | Full-time positions generally offer higher salaries than part-time or contract roles. |
| `has_ml_in_title` | ANOVA | 0.1596 | Large effect | 6.8519e-30 | Jobs mentioning machine learning in the title usually have higher salaries, reflecting demand for specialized skills. |
| `interviews_to_offer` | Pearson Correlation | 0.1551 | Weak correlation | 2.6357e-28 | More interviews before an offer correlates with higher salaries, perhaps indicating competitive hiring processes. |
| `education_level` | ANOVA | 0.1382 | Medium effect | 6.0550e-20 | Higher education levels tend to be associated with higher salaries, showing the value of advanced qualifications. |
| `weekly_hours` | Pearson Correlation | 0.1335 | Weak correlation | 2.5081e-21 | Working more weekly hours often aligns with higher salaries, reflecting compensation for extra time. |
| `equity_offered_pct` | Pearson Correlation | 0.1116 | Weak correlation | 2.4989e-15 | Higher equity percentages offered correlate with higher salaries, indicating total compensation includes ownership stakes. |
| `uses_ai_tools_daily` | ANOVA | 0.1106 | Medium effect | 4.3584e-15 | Daily use of AI tools is linked to higher salaries, suggesting productivity gains affect pay. |
| `certifications_count` | Pearson Correlation | 0.1078 | Weak correlation | 2.0751e-14 | More certifications often relate to higher salaries, reflecting the market value of recognized skills. |
| `ai_tools_hours_per_week` | Pearson Correlation | 0.0803 | Negligible correlation | 1.3118e-08 | Spending more hours weekly on AI tools shows a slight link to higher salaries. |
| `upskilling_hours_per_month` | Pearson Correlation | 0.0697 | Negligible correlation | 8.1962e-07 | More monthly upskilling hours modestly associate with higher salaries, indicating continuous learning benefits. |
| `remote_ratio` | Pearson Correlation | 0.0364 | Negligible correlation | 1.0037e-02 | Higher remote work ratios have a very small connection to salary differences. |
| `switched_jobs_last_year` | ANOVA | 0.0315 | Small effect | 2.5765e-02 | Switching jobs within the last year shows a tiny effect on salary. |

---

## 6. Redundancy & Multicollinearity Analysis
**Cross-Type Redundant Pairs (categorical vs. its own numeric/ordinal encoding, Eta >= 0.85):**

| Categorical Feature | Numeric Feature | Correlation Ratio (Eta) | Interpretation |
|---|---|---|---|
| `experience_level` | `years_experience` | 0.9301 | High cross-type redundancy between 'experience_level' and 'years_experience' (Eta = 0.9301). |

_Recommendation: drop one feature from each redundant pair before modeling to avoid multicollinearity._

---

## 7. Generated Visualizations
No custom chart image assets found in directory (Interactive Plotly visualizations generated directly in HTML report).

---

## 8. Categorical Associations (Cramer's V)
| Feature 1 | Feature 2 | Cramer's V |
|---|---|---|
| `job_title` | `experience_level` | 0.0204 |
| `job_title` | `employment_type` | 0.0 |
| `job_title` | `company_size` | 0.0 |
| `job_title` | `company_location` | 0.0 |
| `job_title` | `employee_residence` | 0.0188 |
| `job_title` | `industry` | 0.0 |
| `job_title` | `remote_ratio` | 0.0 |
| `job_title` | `education_level` | 0.0 |
| `job_title` | `primary_language` | 0.0106 |
| `job_title` | `has_ml_in_title` | 0.999 |
| `job_title` | `manages_people` | 0.0 |
| `job_title` | `certifications_count` | 0.0 |
| `job_title` | `uses_ai_tools_daily` | 0.1891 |
| `job_title` | `salary_currency` | 0.0036 |
| `job_title` | `switched_jobs_last_year` | 0.0101 |
| `experience_level` | `employment_type` | 0.0075 |
| `experience_level` | `company_size` | 0.015 |
| `experience_level` | `company_location` | 0.0 |
| `experience_level` | `employee_residence` | 0.0 |
| `experience_level` | `industry` | 0.0253 |
| `experience_level` | `remote_ratio` | 0.0 |
| `experience_level` | `education_level` | 0.0223 |
| `experience_level` | `primary_language` | 0.0114 |
| `experience_level` | `has_ml_in_title` | 0.0 |
| `experience_level` | `manages_people` | 0.8512 |
| `experience_level` | `certifications_count` | 0.1174 |
| `experience_level` | `uses_ai_tools_daily` | 0.0995 |
| `experience_level` | `salary_currency` | 0.0 |
| `experience_level` | `switched_jobs_last_year` | 0.0 |
| `employment_type` | `company_size` | 0.0137 |
| `employment_type` | `company_location` | 0.0297 |
| `employment_type` | `employee_residence` | 0.0299 |
| `employment_type` | `industry` | 0.0 |
| `employment_type` | `remote_ratio` | 0.0 |
| `employment_type` | `education_level` | 0.0 |
| `employment_type` | `primary_language` | 0.0043 |
| `employment_type` | `has_ml_in_title` | 0.0 |
| `employment_type` | `manages_people` | 0.0 |
| `employment_type` | `certifications_count` | 0.0266 |
| `employment_type` | `uses_ai_tools_daily` | 0.0 |
| `employment_type` | `salary_currency` | 0.0194 |
| `employment_type` | `switched_jobs_last_year` | 0.0312 |
| `company_size` | `company_location` | 0.0413 |
| `company_size` | `employee_residence` | 0.0373 |
| `company_size` | `industry` | 0.0 |
| `company_size` | `remote_ratio` | 0.0 |
| `company_size` | `education_level` | 0.0079 |
| `company_size` | `primary_language` | 0.02 |
| `company_size` | `has_ml_in_title` | 0.0 |
| `company_size` | `manages_people` | 0.0129 |
| `company_size` | `certifications_count` | 0.0 |
| `company_size` | `uses_ai_tools_daily` | 0.0044 |
| `company_size` | `salary_currency` | 0.0373 |
| `company_size` | `switched_jobs_last_year` | 0.0 |
| `company_location` | `employee_residence` | 0.8781 |
| `company_location` | `industry` | 0.0 |
| `company_location` | `remote_ratio` | 0.0 |
| `company_location` | `education_level` | 0.0 |
| `company_location` | `primary_language` | 0.0144 |
| `company_location` | `has_ml_in_title` | 0.0097 |
| `company_location` | `manages_people` | 0.0295 |
| `company_location` | `certifications_count` | 0.0 |
| `company_location` | `uses_ai_tools_daily` | 0.0 |
| `company_location` | `salary_currency` | 0.9997 |
| `company_location` | `switched_jobs_last_year` | 0.007 |
| `employee_residence` | `industry` | 0.0 |
| `employee_residence` | `remote_ratio` | 0.0 |
| `employee_residence` | `education_level` | 0.0 |
| `employee_residence` | `primary_language` | 0.0 |
| `employee_residence` | `has_ml_in_title` | 0.0211 |
| `employee_residence` | `manages_people` | 0.0069 |
| `employee_residence` | `certifications_count` | 0.0 |
| `employee_residence` | `uses_ai_tools_daily` | 0.0 |
| `employee_residence` | `salary_currency` | 0.8786 |
| `employee_residence` | `switched_jobs_last_year` | 0.0 |
| `industry` | `remote_ratio` | 0.0 |
| `industry` | `education_level` | 0.0215 |
| `industry` | `primary_language` | 0.0097 |
| `industry` | `has_ml_in_title` | 0.0 |
| `industry` | `manages_people` | 0.0418 |
| `industry` | `certifications_count` | 0.0 |
| `industry` | `uses_ai_tools_daily` | 0.031 |
| `industry` | `salary_currency` | 0.0 |
| `industry` | `switched_jobs_last_year` | 0.0302 |
| `remote_ratio` | `education_level` | 0.0 |
| `remote_ratio` | `primary_language` | 0.0 |
| `remote_ratio` | `has_ml_in_title` | 0.0 |
| `remote_ratio` | `manages_people` | 0.0 |
| `remote_ratio` | `certifications_count` | 0.0 |
| `remote_ratio` | `uses_ai_tools_daily` | 0.0 |
| `remote_ratio` | `salary_currency` | 0.0 |
| `remote_ratio` | `switched_jobs_last_year` | 0.0263 |
| `education_level` | `primary_language` | 0.0 |
| `education_level` | `has_ml_in_title` | 0.0 |
| `education_level` | `manages_people` | 0.0 |
| `education_level` | `certifications_count` | 0.0077 |
| `education_level` | `uses_ai_tools_daily` | 0.0116 |
| `education_level` | `salary_currency` | 0.0068 |
| `education_level` | `switched_jobs_last_year` | 0.0 |
| `primary_language` | `has_ml_in_title` | 0.0309 |
| `primary_language` | `manages_people` | 0.014 |
| `primary_language` | `certifications_count` | 0.0 |
| `primary_language` | `uses_ai_tools_daily` | 0.0 |
| `primary_language` | `salary_currency` | 0.0106 |
| `primary_language` | `switched_jobs_last_year` | 0.0247 |
| `has_ml_in_title` | `manages_people` | 0.0038 |
| `has_ml_in_title` | `certifications_count` | 0.0 |
| `has_ml_in_title` | `uses_ai_tools_daily` | 0.1909 |
| `has_ml_in_title` | `salary_currency` | 0.0 |
| `has_ml_in_title` | `switched_jobs_last_year` | 0.0 |
| `manages_people` | `certifications_count` | 0.1531 |
| `manages_people` | `uses_ai_tools_daily` | 0.0614 |
| `manages_people` | `salary_currency` | 0.0 |
| `manages_people` | `switched_jobs_last_year` | 0.0 |
| `certifications_count` | `uses_ai_tools_daily` | 0.0269 |
| `certifications_count` | `salary_currency` | 0.0 |
| `certifications_count` | `switched_jobs_last_year` | 0.0 |
| `uses_ai_tools_daily` | `salary_currency` | 0.0 |
| `uses_ai_tools_daily` | `switched_jobs_last_year` | 0.0 |
| `salary_currency` | `switched_jobs_last_year` | 0.0204 |

---

## 9. Predictive Modeling Strategy Blueprint
- **Target Definition:** salary_usd
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
- **Executive Summary:** Target: 'salary_usd' (Regression). Model recommendations and validation strategy tailored for 5000 rows x 27 columns.

---

*Report generated automatically by `summary_generator.py`*
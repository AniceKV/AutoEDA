# Executive Summary – Student Performance Dataset  
**Prepared by:** Senior Lead Data Scientist  
**Date:** 2026‑08‑06  

---

## 1. Executive Overview  

| Item | Detail |
|------|--------|
| **Dataset** | `student_performance_dataset-selected-columns.csv` |
| **Rows / Columns** | 1,000 rows × 10 columns |
| **Target Variable** | `previous_grade` (continuous, 31.30 – 100.00) |
| **Problem Type** | Regression |
| **Key Take‑aways** | • All numeric columns are complete; only `parental_education` had missing values (10.2 %).  <br>• Imputation used mode (“High School”) – no further missing data remain. <br>• Outliers are minimal (< 1 % for each numeric feature). <br>• Pairwise Pearson correlations are uniformly weak (|r| ≤ 0.045). <br>• No statistically‑significant predictors were identified by univariate hypothesis tests. <br>• The data are well‑behaved, but predictive power will rely on multivariate interactions and regularized models. |

---

## 2. Dataset Profile  

### 2.1 Schema & Summary Statistics  

| Column | Data Type | Cardinality | Range / Levels | Mean | Median |
|--------|-----------|------------|----------------|------|--------|
| `student_id` | int64 | 1,000 | 1 – 1,000 | 500.5 | 500.5 |
| `gender` | object | 2 | Female = 510, Male = 490 | – | – |
| `study_time_hours` | float64 | 70 | 0.50 – 8.10 | 3.57 | 3.60 |
| `attendance_percent` | float64 | 327 | 54.80 – 100.00 | 85.09 | 85.20 |
| `sleep_hours` | float64 | 65 | 3.20 – 10.00 | 6.80 | 6.80 |
| `parental_education` | object | 4 | High School = 356, Bachelors = 308, Masters = 184 (102 missing → imputed) | – | – |
| `internet_access` | object | 2 | Yes = 854, No = 146 | – | – |
| `extracurricular_activities` | object | 2 | Yes = 572, No = 428 | – | – |
| `part_time_job` | object | 2 | No = 684, Yes = 316 | – | – |
| `previous_grade` (target) | float64 | 434 | 31.30 – 100.00 | 69.74 | 69.60 |

### 2.2 Missing‑Value Summary  

| Column | Missing Count | Missing % | Imputation Method |
|--------|---------------|----------|-------------------|
| `parental_education` | 102 | 10.2 % | Mode (“High School”) |
| All other columns | 0 | 0 % | – |

---

## 3. Data Quality Checks  

### 3.1 Outlier Analysis (IQR method)  

| Feature | Q1 | Q3 | IQR | Lower Bound | Upper Bound | Outliers (count, %) |
|---------|----|----|-----|-------------|-------------|----------------------|
| `study_time_hours` | 2.6 | 4.5 | 1.9 | –0.25 | 7.35 | 4 (0.4 %) |
| `attendance_percent` | 78.8 | 91.9 | 13.1 | 59.15 | 111.55 | 6 (0.6 %) |
| `sleep_hours` | 5.9 | 7.6 | 1.7 | 3.35 | 10.15 | 2 (0.2 %) |
| `previous_grade` | 61.0 | 78.4 | 17.4 | 34.9 | 104.5 | 3 (0.3 %) |

*Action:* Outliers were **profiled only** (no removal) because they represent a negligible portion of the data.

### 3.2 Imputation Details  

* `parental_education` – 102 missing values replaced with the most frequent category (“High School”).  
* No numeric imputation required (all numeric columns complete).  

---

## 4. Visual Artefacts  

| Image File | Description |
|------------|-------------|
| `dist_student_id.png` | Histogram of unique student identifiers (uniform distribution). |
| `dist_gender.png` | Bar chart showing near‑balanced gender split (Female = 51 %). |
| `dist_study_time_hours.png` | Distribution of study time (right‑skewed, peak around 3–4 h). |
| `dist_attendance_percent.png` | Attendance percentage – tight cluster near 85 % with slight left tail. |
| `dist_sleep_hours.png` | Sleep hours – roughly normal, centred at 6.8 h. |
| `dist_parental_education.png` | Bar chart of parental education levels (High School dominant). |
| `dist_internet_access.png` | Internet access – majority “Yes”. |
| `dist_extracurricular_activities.png` | Participation in extracurriculars – slight majority “Yes”. |
| `dist_part_time_job.png` | Part‑time job status – majority “No”. |
| `dist_previous_grade.png` | Target distribution – bell‑shaped around 70 pts. |
| `bivariate_attendance_percent_vs_previous_grade.png` | Scatter plot of attendance vs. previous grade (no clear trend). |
| `bivariate_sleep_hours_vs_previous_grade.png` | Scatter plot of sleep hours vs. previous grade (flat cloud). |
| `bivariate_study_time_hours_vs_previous_grade.png` | Scatter plot of study time vs. previous grade (no visible pattern). |
| `pairplot.png` | Matrix of pairwise scatterplots & histograms for all numeric features. |
| `correlation_matrix.png` | Heat‑map of Pearson correlations (all values near zero). |
| `target_interactions.png` | Visualisation of the target variable against each categorical predictor (box‑plots). |

*All images are stored in the working directory; they can be opened with any image viewer for detailed inspection.*

---

## 5. Correlation Analysis  

### 5.1 Correlation Matrix (excerpt)  

```
                student_id  study_time_hours  attendance_percent  sleep_hours  previous_grade
student_id           1.000          0.028              -0.045          -0.012          0.012
study_time_hours     0.028          1.000              -0.044           0.006         -0.036
attendance_percent  -0.045        -0.044               1.000           0.034        -0.014
sleep_hours          -0.012         0.006               0.034           1.000         0.001
previous_grade        0.012        -0.036              -0.014           0.001         1.000
```

### 5.2 Top 10 Absolute Correlations  

| Rank | Feature 1 | Feature 2 | Pearson r |
|------|-----------|-----------|-----------|
| 1 | `student_id` | `attendance_percent` | **‑0.045** |
| 2 | `study_time_hours` | `attendance_percent` | **‑0.044** |
| 3 | `study_time_hours` | `previous_grade` | **‑0.036** |
| 4 | `attendance_percent` | `sleep_hours` | **0.034** |
| 5 | `student_id` | `study_time_hours` | **0.028** |
| 6 | `attendance_percent` | `previous_grade` | **‑0.014** |
| 7 | `student_id` | `sleep_hours` | **‑0.012** |
| 8 | `student_id` | `previous_grade` | **0.012** |
| 9 | `study_time_hours` | `sleep_hours` | **0.006** |
|10 | `sleep_hours` | `previous_grade` | **0.001** |

*Interpretation:* All absolute correlations are **< 0.05**, indicating very weak linear relationships between any pair of features and the target.

---

## 6. Univariate Statistical Tests  

| Feature | Test | Statistic | p‑value | Significant? | Interpretation |
|---------|------|------------|---------|--------------|----------------|
| `gender` | Welch t‑test (two‑sample) | 0.2269 | 0.821 | No | No grade difference between genders. |
| `study_time_hours` | Pearson r | –0.0357 | 0.259 | No | Negligible negative association with grade. |
| `attendance_percent` | Pearson r | –0.0143 | 0.651 | No | No linear link to grade. |
| `sleep_hours` | Pearson r | 0.0009 | 0.976 | No | No association. |
| `parental_education` | One‑Way ANOVA | 0.118 | 0.950 | No | Education level does not affect grades. |
| `internet_access` | Welch t‑test | 1.0114 | 0.313 | No | Internet access not predictive. |
| `extracurricular_activities` | Welch t‑test | 1.5010 | 0.134 | No | Participation not linked to grades. |
| `part_time_job` | Welch t‑test | 0.9358 | 0.350 | No | Part‑time job status not predictive. |
| `student_id` | Pearson r | 0.0119 | 0.707 | No | Identifier unrelated to grade. |

**Result:** No single predictor reaches conventional significance (α = 0.05).  

---

## 7. Feature Engineering  

*The automated pipeline did **not** create additional engineered features.*  
Given the weak univariate signals, future work may explore interaction terms (e.g., `study_time_hours × attendance_percent`) or non‑linear transformations (log, polynomial) to capture hidden patterns.

---

## 8. Predictive Modeling Blueprint  

| Aspect | Recommendation |
|--------|----------------|
| **Target** | `previous_grade` (continuous) |
| **Problem Type** | Regression |
| **Algorithms to Try** | • Regularized Linear Regression (Ridge, Lasso) <br>• Random Forest Regressor <br>• Gradient Boosting Regressor (e.g., XGBoost, LightGBM) <br>• Support Vector Regressor (SVR) |
| **Feature‑Selection Strategy** | 1. Drop high‑cardinality identifier (`student_id`). <br>2. Compute permutation importance & mutual information via cross‑validation; keep top‑k features. <br>3. Remove any pair of features with |r| > 0.85 (none observed). |
| **Validation** | 5‑fold cross‑validation. Evaluate **MAE**, **RMSE**, **R²**, and inspect residual distribution for heteroscedasticity. |
| **Over‑fitting Mitigation** | • Apply L1/L2 regularization (Ridge/Lasso). <br>• For tree‑based models, limit max depth, set minimum samples per leaf, and use early stopping. <br>• Perform hyper‑parameter tuning **inside** each CV fold (e.g., GridSearchCV or Bayesian optimization). |
| **Baseline Expectation** | Given the low linear correlations, expect modest R² (≈ 0.02‑0.05) with simple models; ensemble methods may improve performance by capturing non‑linear interactions. |
| **Next Steps** | 1. Encode categorical variables (one‑hot or target encoding). <br>2. Experiment with interaction features (e.g., `study_time_hours * attendance_percent`). <br>3. Run the recommended models and compare performance against the baseline. |

---

## 9. Conclusions & Recommendations  

1. **Data Integrity** – The dataset is clean after imputation; missingness is limited to a single categorical column.  
2. **Predictive Power** – Univariate analyses reveal no strong predictors; multivariate, non‑linear models are required.  
3. **Modeling Path** – Begin with regularized linear models for interpretability, then progress to tree‑based ensembles to capture complex relationships.  
4. **Feature Expansion** – Consider engineered interaction terms and possibly dimensionality reduction (e.g., PCA) if multicollinearity emerges in later stages.  
5. **Evaluation** – Use robust cross‑validation and multiple error metrics; monitor residual plots for systematic patterns.  

*Prepared for the data science team to guide the next phase of model development.*
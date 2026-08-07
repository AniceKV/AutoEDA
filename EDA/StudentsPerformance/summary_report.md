# Executive Summary – Automated Exploratory Data Analysis (EDA)  
**Dataset:** *StudentsPerformance.csv* (1000 rows × 9 columns)  
**Analysis Run:** 2026‑08‑07  

---

## 1. Objective  

The automated EDA pipeline was executed to:

1. **Create a composite target** – `total_score` (sum of Math, Reading, and Writing scores).  
2. **Assess data quality** – missing values, outliers, and distributional characteristics.  
3. **Identify statistically significant relationships** between all features and the new target.  
4. **Generate a set of bivariate visualizations** and a pair‑plot to aid interpretation.  
5. **Provide a predictive‑modeling blueprint** (unsupervised clustering / dimensionality‑reduction) for downstream work.

All steps completed without error; the artefacts are listed in the *File Scan Metadata* section.

---

## 2. Dataset Overview  

| Attribute | Type | Cardinality | Missing % | Key Statistics |
|-----------|------|-------------|----------|----------------|
| gender | string | 2 | 0.0 | female = 518, male = 482 |
| race/ethnicity | string | 5 | 0.0 | group C = 319, group D = 262, group B = 190 |
| parental level of education | string | 6 | 0.0 | some college = 226, associate’s degree = 222 |
| lunch | string | 2 | 0.0 | standard = 645, free/reduced = 355 |
| test preparation course | string | 2 | 0.0 | none = 642, completed = 358 |
| math score | int64 | 81 | 0.0 | Mean = 66.09, Median = 66, Range = 0‑100 |
| reading score | int64 | 72 | 0.0 | Mean = 69.17, Median = 70, Range = 17‑100 |
| writing score | int64 | 77 | 0.0 | Mean = 68.05, Median = 69, Range = 10‑100 |
| **total_score** (engineered) | int64 | 194 | 0.0 | Range = 88‑320, Mean ≈ 203, Median ≈ 204 |

*All columns are complete; no imputation was required.*

---

## 3. Data Quality – Outlier Profiling  

| Feature | Q1 | Q3 | IQR | Lower Bound | Upper Bound | Outliers (Count) | Outlier % |
|---------|----|----|-----|-------------|-------------|------------------|-----------|
| math score | 57.0 | 77.0 | 20.0 | 27.0 | 107.0 | 8 | 0.8 |
| reading score | 59.0 | 79.0 | 20.0 | 29.0 | 109.0 | 6 | 0.6 |
| writing score | 57.75 | 79.0 | 21.25 | 25.875 | 110.875 | 5 | 0.5 |
| total_score | 175.0 | 233.0 | 58.0 | 88.0 | 320.0 | 6 | 0.6 |

*Action:* “profile” – outliers were recorded but **not removed**, preserving the original data for downstream modelling.

---

## 4. Feature Engineering  

| Engineered Feature | Formula | Data Type | Rationale |
|--------------------|---------|-----------|-----------|
| total_score | `math score + reading score + writing score` | int64 | Provides a single, high‑signal measure of overall academic performance; serves as the analysis target. |

No additional derived features were created.

---

## 5. Statistical Relationship Summary  

### 5.1 Hypothesis‑Test Results  

| Feature | Test | Statistic | p‑value | Effect Size* | Significant? |
|---------|------|------------|---------|--------------|--------------|
| gender | Welch t‑test (2‑sample) | 4.1789 | 3.19 e‑05 | 0.1321 (Cohen’s d = 0.2642) | Yes |
| race/ethnicity | One‑Way ANOVA | 9.0961 | 3.23 e‑07 | 0.0353 (η²) | Yes |
| parental level of education | One‑Way ANOVA | 10.7531 | 4.38 e‑10 | 0.0513 (η²) | Yes |
| lunch | Welch t‑test | -9.3232 | 1.58 e‑19 | 0.3121 (Cohen’s d = 0.6243) | Yes |
| test preparation course | Welch t‑test | 8.5945 | 4.43 e‑17 | 0.2800 (Cohen’s d = 0.5601) | Yes |
| math score | Pearson r | 0.9187 | 0.0 | 0.9187 | Yes |
| reading score | Pearson r | 0.9703 | 0.0 | 0.9703 | Yes |
| writing score | Pearson r | 0.9657 | 0.0 | 0.9657 | Yes |

\*Effect size is reported as Cohen’s d for t‑tests, η² for ANOVA, and Pearson r for correlation tests.

### 5.2 Ranked Significant Predictors  

| Rank | Feature | Effect Size | p‑value |
|------|---------|-------------|---------|
| 1 | reading score | 0.9703 | 0.0 |
| 2 | writing score | 0.9657 | 0.0 |
| 3 | math score | 0.9187 | 0.0 |
| 4 | lunch | 0.3121 | 1.58 e‑19 |
| 5 | test preparation course | 0.2800 | 4.43 e‑17 |
| 6 | gender | 0.1321 | 3.19 e‑05 |
| 7 | parental level of education | 0.0513 | 4.38 e‑10 |
| 8 | race/ethnicity | 0.0353 | 3.23 e‑07 |

All listed features are statistically significant at α = 0.05.

---

## 6. Visual Artefacts  

| Image File | Description |
|------------|-------------|
| `bivariate_math_score_vs_reading_score.png` | Scatter plot of Math vs. Reading scores (strong positive linear relationship). |
| `bivariate_math_score_vs_writing_score.png` | Scatter plot of Math vs. Writing scores (high correlation). |
| `bivariate_reading_score_vs_writing_score.png` | Scatter plot of Reading vs. Writing scores (tight clustering). |
| `bivariate_parental_level_of_education_vs_test_preparation_course.png` | Mosaic/stacked bar showing test‑prep completion rates across parental education levels. |
| `bivariate_lunch_vs_test_preparation_course.png` | Bar chart of test‑prep completion by lunch type (standard vs. free/reduced). |
| `pairplot_scores.png` | Pair‑plot of the three numeric scores, colored by `test preparation course` (visualizes class separation). |
| `target_interaction_gender.png` | Box‑plot of `total_score` split by gender. |
| `target_interaction_race.png` | Box‑plot of `total_score` across race/ethnicity groups. |
| `target_interaction_parent_education.png` | Box‑plot of `total_score` across parental education categories. |
| `target_interaction_lunch.png` | Box‑plot of `total_score` for standard vs. free/reduced lunch. |
| `target_interaction_test_prep.png` | Box‑plot of `total_score` for students with/without test‑prep completion. |

*All images are stored in the working directory; each file size is listed in the scan metadata.*

---

## 7. Predictive‑Modeling Blueprint  

| Aspect | Recommendation |
|--------|----------------|
| **Problem Type** | Unsupervised / Exploratory (no external label beyond `total_score`). |
| **Suggested Algorithms** | 1. **K‑Means Clustering** – to discover natural student groups.<br>2. **Hierarchical Agglomerative Clustering** – for dendrogram‑based insights.<br>3. **Principal Component Analysis (PCA)** – to reduce dimensionality and visualise variance. |
| **Feature‑Selection Strategy** | • Remove any high‑cardinality identifiers (none present).<br>• Rank features using cross‑validated permutation importance and mutual information.<br>• Drop collinear features with Pearson |r| > 0.85 (math, reading, and writing scores are highly correlated). |
| **Validation Strategy** | • **Silhouette Score** to assess cluster cohesion.<br>• **Elbow method (Inertia)** to choose optimal K for K‑Means. |
| **Over‑fitting Mitigation** | • Apply regularisation (L1/L2) if moving to supervised models later.<br>• Limit tree depth / enforce minimum samples per leaf for tree‑based methods.<br>• Perform hyper‑parameter tuning strictly within cross‑validation folds. |
| **Execution Environment** | 1000 rows × 8 predictor columns (excluding the engineered target). All steps completed in a sandboxed Python environment. |

---

## 8. Key Take‑aways & Recommendations  

1. **Strong linear relationships** exist among the three core scores (Math, Reading, Writing). For any downstream supervised model, retaining only one of these (e.g., `reading score`) would avoid multicollinearity while preserving most of the variance.  
2. **Lunch type** and **test‑preparation course** show the largest non‑numeric effect sizes (Cohen’s d ≈ 0.62 and 0.56 respectively). These categorical variables are prime candidates for targeted interventions.  
3. **Gender**, **parental education**, and **race/ethnicity** are statistically significant but have modest effect sizes; they should be included for fairness audits.  
4. **Outlier prevalence** is low (< 1 % for each numeric column). Since the outliers are not extreme, they can be kept for modelling, but a sensitivity analysis is advisable.  
5. **Clustering** on the engineered `total_score` together with the categorical variables may reveal distinct student performance groups (e.g., high‑achieving vs. low‑achieving clusters).  
6. **Visualization** confirms the numerical relationships and highlights disparities across socio‑demographic groups; these plots should be incorporated into stakeholder presentations.

---

## 9. Appendices  

### 9.1 Full Hypothesis‑Test Output (excerpt)

```json
{
  "gender": {"test_name":"Two-Sample Welch T-Test","statistic":4.1789,"p_value":3.186e-05,"effect_size":0.1321},
  "race/ethnicity": {"test_name":"One-Way ANOVA","statistic":9.0961,"p_value":3.226e-07,"effect_size":0.0353},
  "parental level of education": {"test_name":"One-Way ANOVA","statistic":10.7531,"p_value":4.381e-10,"effect_size":0.0513},
  "lunch": {"test_name":"Two-Sample Welch T-Test","statistic":-9.3232,"p_value":1.583e-19,"effect_size":0.3121},
  "test preparation course": {"test_name":"Two-Sample Welch T-Test","statistic":8.5945,"p_value":4.427e-17,"effect_size":0.2800},
  "math score": {"test_name":"Pearson Correlation Test","statistic":0.9187,"p_value":0.0,"effect_size":0.9187},
  "reading score": {"test_name":"Pearson Correlation Test","statistic":0.9703,"p_value":0.0,"effect_size":0.9703},
  "writing score": {"test_name":"Pearson Correlation Test","statistic":0.9657,"p_value":0.0,"effect_size":0.9657}
}
```

### 9.2 File Inventory  

| File | Type | Size (KB) |
|------|------|-----------|
| `bivariate_math_score_vs_reading_score.png` | Image (scatter) | 149.54 |
| `bivariate_math_score_vs_writing_score.png` | Image (scatter) | 151.41 |
| `bivariate_reading_score_vs_writing_score.png` | Image (scatter) | 115.22 |
| `bivariate_parental_level_of_education_vs_test_preparation_course.png` | Image (categorical) | 56.36 |
| `bivariate_lunch_vs_test_preparation_course.png` | Image (categorical) | 36.89 |
| `pairplot_scores.png` | Image (pair‑plot) | 253.11 |
| `target_interaction_gender.png` | Image (box‑plot) | 31.28 |
| `target_interaction_race.png` | Image (box‑plot) | 40.45 |
| `target_interaction_parent_education.png` | Image (box‑plot) | 56.26 |
| `target_interaction_lunch.png` | Image (box‑plot) | 33.29 |
| `target_interaction_test_prep.png` | Image (box‑plot) | 34.34 |
| `current_df.csv` | CSV (final dataset) | – |
| `metadata_profile.json` | JSON (schema) | – |
| `metrics.json` | JSON (full metrics) | – |
| `agent_plan_log.json` | JSON (pipeline plan) | – |
| `agent_state.json` | JSON (state & results) | – |

---

**Prepared by:**  
Senior Lead Data Scientist – Automated EDA Team  
*Date: 2026‑08‑07*  
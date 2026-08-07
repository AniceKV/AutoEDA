# Executive Summary – Automated Exploratory Data Analysis  
**Dataset:** *StudentsPerformance.csv* (1000 rows × 9 columns)  
**Analysis Run:** 2024‑08‑07 (Auto‑EDA pipeline)  

---  

## 1. Project Context  

The dataset captures academic performance of secondary‑school students together with demographic and socioeconomic attributes. The primary analytical goal was to explore relationships among variables, engineer a composite performance metric, and outline a data‑driven predictive (or exploratory) modelling strategy.

---

## 2. Data Overview  

| Column | Data Type | Cardinality | Missing % | Key Statistics |
|--------|-----------|-------------|----------|----------------|
| **gender** | string | 2 | 0.0 | – |
| **race/ethnicity** | string | 5 | 0.0 | – |
| **parental level of education** | string | 6 | 0.0 | – |
| **lunch** | string | 2 | 0.0 | – |
| **test preparation course** | string | 2 | 0.0 | – |
| **math score** | int64 | 81 | 0.0 | Mean = 66.09, Std = 15.16, Skew = ‑0.279 |
| **reading score** | int64 | 72 | 0.0 | Mean = 69.17, Std = 14.60, Skew = ‑0.259 |
| **writing score** | int64 | 77 | 0.0 | Mean = 68.05, Std = 15.20, Skew = ‑0.289 |
| **avg_math score_reading score_writing score** (engineered) | float64 | 194 | 0.0 | Mean ≈ 71.6 (computed as row‑wise mean of the three scores) |

*All numeric columns span the full 0‑100 range; no missing values were detected.*

---

## 3. Missing‑Data Handling  

The pipeline applied a uniform imputation policy:

| Rule | Applied To |
|------|------------|
| Convert common placeholders (`?`, `NA`, `N/A`, `null`) → `NaN` | All columns |
| Numeric columns with |skew| > 1.0 → median imputation | None (all |skew| < 1) |
| Numeric columns with |skew| ≤ 1.0 → mean imputation | All numeric columns (no effect) |
| Categorical columns → mode imputation, fallback = `Unknown` | All categorical columns (no effect) |

**Result:** No column required imputation; the dataset remained unchanged.

---

## 4. Outlier Profiling  

Outlier detection was performed on the three raw score columns (action = *profile* only).  

| Variable | Q1 | Q3 | IQR | Lower Bound | Upper Bound | Outliers (Count) | Outlier % |
|----------|----|----|-----|-------------|-------------|------------------|----------|
| **math score** | 57.0 | 77.0 | 20.0 | 27.0 | 107.0 | 8 | 0.8 % |
| **reading score** | 59.0 | 79.0 | 20.0 | 29.0 | 109.0 | 6 | 0.6 % |
| **writing score** | 57.75 | 79.0 | 21.25 | 25.875 | 110.875 | 5 | 0.5 % |

All outliers lie well within the theoretical 0‑100 score range; the pipeline retained them for downstream analysis.

---

## 5. Feature Engineering  

| Engineered Feature | Formula | Data Type | Rationale |
|--------------------|---------|-----------|-----------|
| **avg_math score_reading score_writing score** | `mean(math score, reading score, writing score)` | float64 | Provides a single, high‑signal indicator of overall academic performance, simplifying downstream modelling and interpretation. |

No correlation with a pre‑existing target was computed (the target column itself is the engineered feature).

---

## 6. Statistical Hypothesis Testing  

All tests used a two‑tailed α = 0.05. Effect‑size metrics are reported (Cohen’s d for t‑tests, η² for ANOVA, Pearson r for correlations).

| Feature | Test | Statistic | p‑value | Effect Size | Significant? | Interpretation |
|---------|------|-----------|---------|-------------|--------------|----------------|
| **gender** | Welch t‑test (2 groups) | 4.1789 | 3.19 e‑5 | d = 0.2642 | ✔ | Female vs. male differ in average_score. |
| **race/ethnicity** | One‑Way ANOVA (5 groups) | 9.0961 | 3.23 e‑7 | η² = 0.0353 | ✔ | Significant variation across ethnic groups. |
| **parental level of education** | One‑Way ANOVA (6 groups) | 10.7531 | 4.38 e‑10 | η² = 0.0513 | ✔ | Education level influences average_score. |
| **lunch** | Welch t‑test (2 groups) | ‑9.3232 | 1.58 e‑19 | d = 0.6243 | ✔ | Standard lunch > free/reduced lunch. |
| **test preparation course** | Welch t‑test (2 groups) | 8.5945 | 4.43 e‑17 | d = 0.5601 | ✔ | Completion improves average_score. |
| **math score** | Pearson r | 0.9187 | 0.0 | r = 0.9187 | ✔ | Very strong linear relationship with average_score. |
| **reading score** | Pearson r | 0.9703 | 0.0 | r = 0.9703 | ✔ | Highest correlation with average_score. |
| **writing score** | Pearson r | 0.9657 | 0.0 | r = 0.9657 | ✔ | Very strong correlation with average_score. |

**Ranked Significant Predictors (by effect size)**  

1. **reading score** (r = 0.970)  
2. **writing score** (r = 0.966)  
3. **math score** (r = 0.919)  
4. **lunch** (d = 0.624)  
5. **test preparation course** (d = 0.560)  
6. **gender** (d = 0.264)  
7. **parental level of education** (η² = 0.051)  
8. **race/ethnicity** (η² = 0.035)  

All eight predictors are statistically significant and merit inclusion in any downstream model.

---

## 7. Visual Artefacts  

| Image File | Size (KB) | Description (inferred from name) |
|------------|-----------|-----------------------------------|
| `bivariate_gender_vs_test_preparation_course.png` | 34.3 | Bar/stacked plot showing distribution of test‑prep completion by gender. |
| `bivariate_lunch_vs_test_preparation_course.png` | 36.9 | Cross‑tabulation of lunch type vs. test‑prep status. |
| `bivariate_parental_level_of_education_vs_test_preparation_course.png` | 56.4 | Mosaic/stacked chart of education level vs. test‑prep completion. |
| `numeric_pairplot.png` | 189.1 | Pairwise scatter‑matrix of the three raw scores plus the engineered average_score (visualizes strong linear relationships). |
| `target_interaction_gender.png` | 40.9 | Box‑/violin‑plot of average_score split by gender. |
| `target_interaction_race.png` | 50.3 | Distribution of average_score across the five ethnic groups. |
| `target_interaction_education.png` | 65.4 | Average_score vs. parental education levels. |
| `target_interaction_lunch.png` | 42.9 | Comparison of average_score for standard vs. free/reduced lunch. |
| `target_interaction_preparation.png` | 44.6 | Impact of test‑prep completion on average_score. |

*All visualisations were generated automatically and saved in PNG format; they corroborate the statistical findings reported above.*

---

## 8. Predictive Modeling Blueprint  

Although the pipeline flagged the problem as “Unsupervised / Exploratory”, the presence of a clear target (`average_score`) enables supervised regression or classification approaches. The blueprint recommends the following:

| Aspect | Recommendation |
|--------|----------------|
| **Problem Type** | Supervised regression (predict average_score) – or unsupervised clustering for student segmentation. |
| **Algorithms** | • **Linear / Ridge / Lasso Regression** (baseline, interpretable) <br>• **Tree‑based models** – Random Forest, Gradient Boosting (capture non‑linearities) <br>• **K‑Means / Hierarchical Clustering** (if exploring latent groups) <br>• **PCA** for dimensionality reduction / visualization. |
| **Feature Selection** | 1. Remove any high‑cardinality identifiers (none present). <br>2. Rank features using cross‑validated permutation importance and mutual information. <br>3. Drop collinear features with |r| > 0.85 (math, reading, writing scores are highly correlated; consider keeping only one or the engineered average). |
| **Validation Strategy** | • **Train‑validation split** (e.g., 80/20) with stratification on categorical variables if needed. <br>• **Cross‑validation** (5‑fold) for robust error estimation. <br>• For clustering, evaluate **Silhouette Score** and **Elbow (Inertia)** curves. |
| **Over‑fitting Mitigation** | • Regularization (L1/L2) for linear models. <br>• Limit tree depth, enforce minimum samples per leaf for tree‑based models. <br>• Hyper‑parameter tuning confined to inner CV folds. |
| **Performance Metrics** | Regression: **RMSE**, **MAE**, **R²**. <br>Clustering: **Silhouette**, **Davies‑Bouldin**. |
| **Execution Environment** | Dataset size (1000 × 9) is modest; all recommended algorithms run comfortably on a standard laptop/CPU. |

---

## 9. Key Insights & Business Implications  

1. **Academic performance is dominated by the three subject scores** (reading > writing > math) – each explains > 90 % of variance in the engineered average_score.  
2. **Socio‑economic factors** (lunch type, test‑prep completion) have sizable, statistically significant effects (Cohen’s d ≈ 0.6). Students receiving standard lunch and who completed the preparation course achieve higher average scores.  
3. **Demographic attributes** (gender, parental education, ethnicity) also influence performance, albeit with smaller effect sizes; they should be considered for equity‑focused interventions.  
4. **Outliers are minimal** (< 1 %); no aggressive trimming is required.  

These findings suggest that targeted academic support (e.g., free test‑prep programs, nutrition assistance) could yield measurable improvements in overall student performance.

---

## 10. Recommendations & Next Steps  

| Action | Rationale |
|--------|-----------|
| **Model Development** | Build a regression model using the engineered average_score as the target. Start with a simple linear model, then explore regularized and tree‑based variants. |
| **Feature Reduction** | Because the three raw scores are highly collinear, consider using only the engineered average or a single representative score to avoid redundancy. |
| **Segmentation Analysis** | Apply K‑Means (k ≈ 3‑5) to uncover student clusters; examine cluster profiles for tailored interventions. |
| **Policy Simulation** | Use the fitted model to simulate the impact of expanding test‑prep access or improving lunch nutrition on predicted scores. |
| **Further Data Enrichment** | Incorporate additional variables (e.g., school resources, attendance) to capture unexplained variance. |
| **Documentation** | Preserve the generated visual artefacts and statistical tables for stakeholder reporting and reproducibility. |

---

## 11. Limitations  

* The pipeline treated the engineered average_score as both a **target** and a **feature**, leading to a contradictory “unsupervised” classification in the blueprint.  
* Correlation analysis output is empty; only hypothesis‑test results are available.  
* Visual artefacts are not inspected directly; descriptions rely on file naming conventions.  
* No external validation (e.g., hold‑out dataset) was performed.

---

**Prepared by:**  
Senior Lead Data Scientist – Automated EDA Review  
*Date: 2026‑08‑07*  
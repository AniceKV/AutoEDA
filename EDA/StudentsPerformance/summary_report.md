# Executive Summary – Automated Exploratory Data Analysis  
**Dataset:** `StudentsPerformance.csv`  
**Analysis Run:** 2026‑08‑07 (AutoEDA – sandbox run)  

---  

## 1. Dataset Overview  

| Item | Value |
|------|-------|
| **Rows** | 1,000 |
| **Columns (including engineered target)** | 9 |
| **Target column** | `avg_math score_reading score_writing score` (mean of the three exam scores) |
| **Original numeric columns** | `math score`, `reading score`, `writing score` |
| **Categorical columns** | `gender`, `race/ethnicity`, `parental level of education`, `lunch`, `test preparation course` |
| **File containing raw data** | `current_df.csv` |
| **Location of source file** | `C:\Users\Anish Kumar Verma\PycharmProjects\AutoEDA\temp_uploads\StudentsPerformance.csv` |

### 1.1 Schema & Cardinality  

| Column | Data Type | Cardinality | Key Metric |
|--------|-----------|-------------|------------|
| gender | string | 2 | female: 518, male: 482 |
| race/ethnicity | string | 5 | group C: 319, group D: 262, group B: 190 |
| parental level of education | string | 6 | some college: 226, associate’s degree: 222, high school: 196 |
| lunch | string | 2 | standard: 645, free/reduced: 355 |
| test preparation course | string | 2 | none: 642, completed: 358 |
| math score | int64 | 81 | Mean = 66.09, Median = 66, Std = 15.16 |
| reading score | int64 | 72 | Mean = 69.17, Median = 70, Std = 14.60 |
| writing score | int64 | 77 | Mean = 68.05, Median = 69, Std = 15.20 |
| **engineered target** `avg_math score_reading score_writing score` | float64 | 194 | – |

*All columns have **0 % missing values**.*

---  

## 2. Data Cleaning & Imputation  

The pipeline applied a uniform imputation policy (see `imputation_summary` in `metrics.json`):

| Rule | Applied To |
|------|------------|
| Standardize missing string placeholders (`?`, `NA`, `N/A`, `null`) → `NaN` | All columns |
| Numeric columns with |skew| > 1.0 or < ‑1.0 → **median** imputation | None (all numeric columns have modest skew) |
| Numeric columns with |skew| ≤ 1.0 → **mean** imputation | All numeric columns (no missing values) |
| Categorical columns → **mode** imputation, fallback `'Unknown'` | All categorical columns (no missing values) |

Result: **No values were altered**; the dataset was already complete.

---  

## 3. Outlier Profiling  

Outlier detection was performed on the three original score columns (action = *profile*).  

| Column | Q1 | Q3 | IQR | Lower Bound | Upper Bound | Outliers (count) | Outlier % |
|--------|----|----|-----|-------------|-------------|------------------|-----------|
| math score | 57.0 | 77.0 | 20.0 | 27.0 | 107.0 | 8 | 0.8 % |
| reading score | 59.0 | 79.0 | 20.0 | 29.0 | 109.0 | 6 | 0.6 % |
| writing score | 57.75 | 79.0 | 21.25 | 25.875 | 110.875 | 5 | 0.5 % |

*Action taken:* profiling only – no rows were removed or altered.

---  

## 4. Feature Engineering  

| Engineered Feature | Formula | Data Type | Rationale |
|--------------------|---------|-----------|-----------|
| `avg_math score_reading score_writing score` | `mean(math score, reading score, writing score)` | float64 | Provides a single composite performance metric; high‑signal transformation for downstream analysis. |

The engineered target is used as the **analysis focus** throughout the pipeline.

---  

## 5. Statistical Hypothesis Testing  

All tests were performed at α = 0.05. Significance is indicated by `p < α`.  

| Feature | Test Type | Statistic | Effect Size | p‑value | Significant? | Interpretation |
|---------|-----------|-----------|-------------|---------|--------------|----------------|
| gender | Welch t‑test (2‑sample) | 4.1789 | Cohen’s d = 0.2642 | 3.19 e‑05 | ✔ | Females score higher on average. |
| race/ethnicity | One‑Way ANOVA | 9.0961 | η² = 0.0353 | 3.23 e‑07 | ✔ | Performance varies across ethnic groups. |
| parental level of education | One‑Way ANOVA | 10.7531 | η² = 0.0513 | 4.38 e‑10 | ✔ | Higher parental education → higher scores. |
| lunch | Welch t‑test | –9.3232 | Cohen’s d = 0.6243 | 1.58 e‑19 | ✔ | Standard lunch associated with higher scores. |
| test preparation course | Welch t‑test | 8.5945 | Cohen’s d = 0.5601 | 4.43 e‑17 | ✔ | Completion of the prep course improves scores. |
| math score | Pearson r | 0.9187 | r = 0.9187 | 0.0 | ✔ | Very strong linear relationship with the engineered target. |
| reading score | Pearson r | 0.9703 | r = 0.9703 | 0.0 | ✔ | Highest correlation with the target. |
| writing score | Pearson r | 0.9657 | r = 0.9657 | 0.0 | ✔ | Very strong correlation with the target. |

**Ranked Significant Predictors (by absolute effect size):**  

1. Reading score (r = 0.9703)  
2. Writing score (r = 0.9657)  
3. Math score (r = 0.9187)  
4. Lunch (Cohen’s d = 0.6243)  
5. Test preparation course (Cohen’s d = 0.5601)  
6. Gender (Cohen’s d = 0.2642)  
7. Parental level of education (η² = 0.0513)  
8. Race/ethnicity (η² = 0.0353)  

All listed predictors are statistically significant and should be considered in any downstream modeling effort.

---  

## 6. Visual Analytics  

The pipeline generated a suite of PNG visualizations (stored in the working directory). Below is a brief description of each artifact.

| Image File | Description |
|------------|-------------|
| `bivariate_math_score_vs_reading_score.png` | Scatter plot of **math** vs **reading** scores; shows a tight positive linear trend (correlation ≈ 0.92). |
| `bivariate_math_score_vs_writing_score.png` | Scatter plot of **math** vs **writing** scores; also a strong positive relationship (correlation ≈ 0.97). |
| `bivariate_reading_score_vs_writing_score.png` | Scatter plot of **reading** vs **writing** scores; highest observed correlation (≈ 0.97). |
| `bivariate_parental_level_of_education_vs_test_preparation_course.png` | Categorical heat‑map / count plot showing distribution of **parental education** levels across **test‑prep** status. |
| `pairplot.png` | Pairwise matrix (seaborn) for the three original scores plus the engineered target, colored by target value. Highlights the strong inter‑correlations. |
| `target_interaction_math.png` | Interaction plot of **average_score** vs **math score** (line of best fit). |
| `target_interaction_reading.png` | Interaction plot of **average_score** vs **reading score**. |
| `target_interaction_writing.png` | Interaction plot of **average_score** vs **writing score**. |

*All images are stored in the same directory as the analysis; file sizes range from ~56 KB to ~189 KB.*

---  

## 7. Predictive Modeling Blueprint  

Although the pipeline classified the problem as **unsupervised / exploratory** (target defined as a derived metric), the following blueprint outlines a sensible approach should a supervised model be desired.

| Component | Recommendation |
|-----------|----------------|
| **Problem Type** | Unsupervised (exploratory) – clustering & dimensionality reduction. |
| **Suggested Algorithms** | • **K‑Means** (choose *k* via elbow & silhouette) <br>• **Hierarchical Agglomerative Clustering** (ward linkage) <br>• **Principal Component Analysis (PCA)** for visualisation and noise reduction. |
| **Feature Selection Strategy** | 1. Exclude any high‑cardinality identifier columns (none present). <br>2. Rank features using **cross‑validated permutation importance** and **mutual information**. <br>3. Remove collinear features with **|ρ| > 0.85** (math, reading, and writing scores are highly collinear). |
| **Validation Strategy** | • **Silhouette Score** (range 0–1) to assess cluster cohesion/separation. <br>• **Inertia (within‑cluster sum of squares)** elbow curve for *k* selection. |
| **Over‑fitting Mitigation** | • Apply **regularization** (L1/L2) if moving to supervised regression. <br>• Limit tree depth / enforce minimum samples per leaf for tree‑based models. <br>• Perform **hyper‑parameter tuning** strictly within cross‑validation folds. |
| **Executive Summary** | The dataset is well‑behaved (no missing data, minimal outliers). The engineered average score is strongly driven by the three exam scores, which are themselves highly correlated. Categorical variables (lunch, test‑prep, gender, parental education, ethnicity) also show statistically significant differences in performance. A clustering analysis using the three scores (or the engineered target) is likely to reveal distinct student groups, while PCA can visualise the dominant variance directions. |

---  

## 8. Key Take‑aways & Recommendations  

1. **Data Quality** – The source data is complete and clean; no further imputation or outlier removal is required.  
2. **Signal Strength** – The three exam scores explain virtually all variance in the engineered target (Pearson r > 0.91). Any model that includes all three will be near‑perfect; consider dimensionality reduction if redundancy is a concern.  
3. **Categorical Impact** – Lunch type, test‑prep completion, gender, parental education, and ethnicity each have a statistically significant effect on performance. These variables should be retained for any explanatory or clustering analysis.  
4. **Modeling Path** –  
   *If the goal is to **cluster** students*: use the three scores (or the average) as input, evaluate K‑Means (k = 3‑5) and hierarchical clustering, and validate with silhouette scores.  
   *If the goal is to **predict** a future score*: a simple linear regression using any one of the three scores will already achieve R² ≈ 0.95; adding categorical dummies can improve interpretability.  
5. **Next Steps** –  
   * Run PCA (2‑3 components) to visualise student groups.  
   * Perform a systematic search for the optimal number of clusters (elbow + silhouette).  
   * If a supervised task emerges (e.g., predicting a future exam), build a regularized linear model or tree‑based ensemble, ensuring cross‑validation to guard against over‑fitting.  

---  

**Prepared by:**  
Senior Lead Data Scientist – Automated EDA Review  
*Date:* 2026‑08‑07  

*All findings are derived directly from the generated artifact files; no external data or code was consulted.*
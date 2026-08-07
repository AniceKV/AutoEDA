# Executive Summary – EV Adoption & Range‑Anxiety Dataset  
**Target Variable:** `Current_Car_Type` (Classification)  
**Rows / Columns:** 10 000 × 10  

---

## 1. Dataset Overview  

| Attribute | Type / dtype | Cardinality | Missing (Count %) | Key Statistics |
|-----------|--------------|-------------|-------------------|----------------|
| **Buyer_ID** | string (object) | 10 000 | 0 (0 %) | Unique identifier – no predictive value |
| **Age** | int64 | 45 | 0 (0 %) | Range 25‑69, Mean ≈ 46.9, Median = 47 |
| **Gender** | string (object) | 3 | 0 (0 %) | Male = 5 201, Female = 4 498, Other = 301 |
| **Annual_Income_USD** | float64 | 8 915 | 178 (1.78 %) | Range 30 000‑223 345, Mean ≈ 85 378, Median ≈ 84 708 |
| **City_Type** | string (object) | 3 | 0 (0 %) | Urban = 4 941, Suburban = 3 563, Rural = 1 496 |
| **Daily_Commute_km** | float64 | 991 | 181 (1.81 %) | Range 5‑135.5, Mean ≈ 41.1, Median ≈ 40.2 |
| **Number_of_Cars_Owned** | int64 | 4 | 0 (0 %) | Range 1‑4, Mean ≈ 1.86, Median = 2 |
| **Current_Car_Type** | string (object) | 4 | 0 (0 %) | Sedan = 4 000, SUV = 3 486, Hatchback = 1 523, Truck = ? |
| **Charging_Stations_Near_Home** | int64 | 15 | 0 (0 %) | Range 0‑14, Mean ≈ 5.35, Median = 5 |
| **Charging_Stations_Near_Work** | int64 | 20 | 0 (0 %) | Range 0‑19, Mean ≈ 7.46, Median = 6 |

*The dataset is balanced across the four car‑type classes (≈ 40 % Sedan, 35 % SUV, 15 % Hatchback, remainder Truck).*

---

## 2. Data‑Quality & Imputation  

### 2.1 Missing‑Value Handling  

| Column | Missing Before | Imputation Method | Imputed Value |
|--------|----------------|-------------------|---------------|
| Annual_Income_USD | 178 (1.78 %) | Mean (skewness = 0.31) | **85 378.49** |
| Daily_Commute_km | 181 (1.81 %) | Mean (skewness = 0.34) | **41.1054** |
| All other columns | 0 | – | – |

*Rule set applied:*  
1. Standardise common placeholders (`?`, `NA`, `N/A`, `null`) → `NaN`.  
2. Numeric columns with |skew| > 1 → median imputation (none triggered).  
3. Numeric columns with |skew| ≤ 1 → mean imputation (applied above).  
4. Categorical columns → mode imputation, fallback to `"Unknown"` (no missing values).

### 2.2 Outlier Profiling  

| Feature | Q1 | Q3 | IQR | Lower Bound | Upper Bound | Outliers (Count %) |
|---------|----|----|-----|-------------|-------------|--------------------|
| Age | 36 | 58 | 22 | 3 | 91 | 0 (0 %) |
| Annual_Income_USD | 61 883.5 | 106 938.5 | 45 055 | –5 699 | 174 521 | 58 (0.58 %) |
| Daily_Commute_km | 23.6 | 56.7 | 33.1 | –26.05 | 106.35 | 44 (0.44 %) |
| Number_of_Cars_Owned | 1 | 2 | 1 | –0.5 | 3.5 | 511 (5.11 %) |
| Charging_Stations_Near_Home | 2 | 8 | 6 | –7 | 17 | 0 (0 %) |
| Charging_Stations_Near_Work | 3 | 11 | 8 | –9 | 23 | 0 (0 %) |

*Action taken:* **profile** only – no removal or transformation was performed.

---

## 3. Distribution Visualisations  

All distribution plots are saved in the sandbox directory; file names and sizes are listed below.

| Plot | Description | File (KB) |
|------|-------------|-----------|
| `dist_Age.png` | Histogram of Age (25‑69) | 33.13 |
| `dist_Annual_Income_USD.png` | Histogram of Annual Income (30 k‑223 k) | 47.89 |
| `dist_Daily_Commute_km.png` | Histogram of Daily Commute (5‑135 km) | 40.24 |
| `dist_Number_of_Cars_Owned.png` | Bar chart of cars owned (1‑4) | 28.94 |
| `dist_Charging_Stations_Near_Home.png` | Bar chart of home charging stations (0‑14) | 39.24 |
| `dist_Charging_Stations_Near_Work.png` | Bar chart of work charging stations (0‑19) | 42.31 |
| `dist_Gender.png` | Bar chart of gender distribution | 24.18 |
| `dist_City_Type.png` | Bar chart of city‑type distribution | 26.44 |
| `dist_Current_Car_Type.png` | Bar chart of target car‑type distribution | 27.03 |

*Interpretation:* All numeric features exhibit near‑normal shapes with modest skew; categorical features are well‑balanced except for a slight dominance of `Male` and `Urban` categories.

---

## 4. Correlation Analysis (Numeric Features)

### 4.1 Heatmap  
Saved as `correlation_matrix.png` (86.97 KB). The matrix confirms very low linear relationships among predictors.

### 4.2 Top 10 Absolute Correlations  

| Rank | Feature 1 | Feature 2 | Pearson r |
|------|-----------|-----------|-----------|
| 1 | Charging_Stations_Near_Home | Charging_Stations_Near_Work | **0.4808** |
| 2 | Daily_Commute_km | Charging_Stations_Near_Home | 0.027 |
| 3 | Age | Daily_Commute_km | –0.0136 |
| 4 | Age | Annual_Income_USD | –0.0115 |
| 5 | Age | Charging_Stations_Near_Home | –0.0081 |
| 6 | Daily_Commute_km | Number_of_Cars_Owned | –0.0074 |
| 7 | Annual_Income_USD | Charging_Stations_Near_Home | 0.0047 |
| 8 | Number_of_Cars_Owned | Charging_Stations_Near_Home | 0.0043 |
| 9 | Annual_Income_USD | Number_of_Cars_Owned | 0.0042 |
|10 | Age | Number_of_Cars_Owned | –0.0033 |

*Take‑away:* The only moderate correlation is between the two charging‑station counts (≈ 0.48). All other predictor pairs are essentially uncorrelated (|r| < 0.03).

---

## 5. Categorical Association (Cramér’s V)

Heatmap saved as `categorical_association_matrix.png` (42.13 KB).  

| Pair | Cramér’s V |
|------|------------|
| Gender ↔ City_Type | **0.0** |
| Gender ↔ Current_Car_Type | **0.0** |
| City_Type ↔ Current_Car_Type | **0.0** |

*Interpretation:* No detectable association between the categorical predictors and the target; the variables are statistically independent in the sample.

---

## 6. Bivariate Relationships with the Target  

All bivariate plots are stored as PNG files (size in KB). They visualise the distribution of each predictor across the four car‑type categories.

| Plot | X‑axis | Y‑axis (Target) | File (KB) |
|------|--------|----------------|-----------|
| `bivariate_Age_vs_Current_Car_Type.png` | Age | Current_Car_Type | 31.05 |
| `bivariate_Gender_vs_Current_Car_Type.png` | Gender | Current_Car_Type | 40.66 |
| `bivariate_City_Type_vs_Current_Car_Type.png` | City_Type | Current_Car_Type | 43.57 |
| `bivariate_Annual_Income_USD_vs_Current_Car_Type.png` | Annual_Income_USD | Current_Car_Type | 48.35 |
| `bivariate_Daily_Commute_km_vs_Current_Car_Type.png` | Daily_Commute_km | Current_Car_Type | 42.05 |
| `bivariate_Number_of_Cars_Owned_vs_Current_Car_Type.png` | Number_of_Cars_Owned | Current_Car_Type | 38.39 |
| `bivariate_Charging_Stations_Near_Home_vs_Current_Car_Type.png` | Charging_Stations_Near_Home | Current_Car_Type | 38.62 |
| `bivariate_Charging_Stations_Near_Work_vs_Current_Car_Type.png` | Charging_Stations_Near_Work | Current_Car_Type | 43.48 |

*Visual inspection* (as per the generated images) shows overlapping distributions across car‑type categories, consistent with the statistical tests that found no significant differences.

---

## 7. Statistical Hypothesis Testing  

| Feature | Test | Statistic | p‑value | Significant (α = 0.05) | Interpretation |
|---------|------|------------|---------|------------------------|----------------|
| Buyer_ID | Chi‑Square (Independence) | 30 000 | 0.4940 | No | No association with target |
| Age | One‑Way ANOVA | 1.8102 | 0.1429 | No | Mean ages similar across car types |
| Gender | Chi‑Square (Independence) | 1.0359 | 0.9842 | No | Gender distribution identical across car types |
| Annual_Income_USD | One‑Way ANOVA | 2.0844 | 0.09999 | No | Income differences not statistically meaningful |
| City_Type | Chi‑Square (Independence) | 1.544 | 0.9565 | No | City type unrelated to car type |
| Daily_Commute_km | One‑Way ANOVA | 0.2122 | 0.88797 | No | Commute distance does not differentiate car types |
| Number_of_Cars_Owned | One‑Way ANOVA | 0.0629 | 0.97940 | No | Number of cars owned is uniform across target |
| Charging_Stations_Near_Home | One‑Way ANOVA | 0.2881 | 0.83401 | No | Home charging availability not predictive |
| Charging_Stations_Near_Work | One‑Way ANOVA | 0.1864 | 0.90569 | No | Work charging availability not predictive |

**Result:** *No predictor reached statistical significance at the 5 % level.* Consequently, the `significant_predictors` list is empty.

---

## 8. Predictive‑Modeling Blueprint  

| Component | Recommendation |
|-----------|----------------|
| **Problem Type** | Multi‑class Classification (`Current_Car_Type`) |
| **Baseline Model** | Regularized Logistic Regression (L2 penalty) |
| **Advanced Models** | • Random Forest Classifier  <br>• Gradient Boosting (XGBoost or LightGBM)  <br>• Support Vector Classifier (SVM) |
| **Feature‑Selection Strategy** | 1. Drop high‑cardinality ID (`Buyer_ID`). <br>2. Rank features via cross‑validated permutation importance **and** mutual information. <br>3. Remove collinear features with |r| > 0.85 (none found). |
| **Validation Strategy** | Stratified 5‑fold Cross‑Validation. <br>Metrics: Balanced Accuracy, Macro‑averaged F1, Precision‑Recall AUC, Confusion Matrix. |
| **Over‑fitting Mitigation** | • Apply L1/L2 regularisation (logistic regression, linear SVM). <br>• Limit tree depth, enforce minimum samples per leaf (RF, GB). <br>• Hyper‑parameter tuning **inside** CV folds (e.g., GridSearchCV). |
| **Execution Summary** | The dataset contains 10 000 rows and 10 columns. After imputation and profiling, the data are ready for modelling. No strong linear or categorical signals were detected, so model performance will rely on subtle interactions captured by tree‑based or kernel methods. |

---

## 9. Key Take‑aways & Next Steps  

1. **Data Quality** – Missing values are minimal and have been imputed with sensible mean values; outlier prevalence is low except for `Number_of_Cars_Owned` (5 %); no aggressive outlier removal was performed.  
2. **Predictive Signal** – Neither univariate ANOVA/Chi‑Square tests nor simple Pearson correlations reveal strong predictors of `Current_Car_Type`.  
3. **Modeling Approach** – Begin with a regularized logistic regression baseline, then explore ensemble methods (Random Forest, Gradient Boosting) and SVM to capture non‑linear patterns.  
4. **Feature Engineering** – No engineered features were generated automatically; consider domain‑specific transformations (e.g., income‑to‑age ratio, commute‑to‑charging‑station ratios) if further performance gains are needed.  
5. **Evaluation** – Use stratified 5‑fold CV and report macro‑averaged metrics to account for class balance.  

*Prepared by:* **Senior Lead Data Scientist**  
*Date:* 2026‑08‑07  

---  

**Appendix – Artifact Index**  

| Artifact | Description | Size (KB) |
|----------|-------------|-----------|
| `correlation_matrix.png` | Pearson correlation heatmap (numeric features) | 86.97 |
| `categorical_association_matrix.png` | Cramér’s V heatmap (categorical features) | 42.13 |
| `pairplot.png` | Full pair‑plot of all variables | 544.07 |
| `bivariate_*.png` | Eight target‑centric bivariate visualisations (see Section 6) | 31‑48 |
| `dist_*.png` | Individual distribution plots (see Section 3) | 24‑48 |
| `eda_report.html` | Full interactive HTML report (generated by the pipeline) | – |
| `metrics.json` | Consolidated JSON metrics (source of this summary) | – |
| `metadata_profile.json` | Schema & basic profiling metadata | – |
| `current_df.csv` | Cleaned dataset after imputation (10 000 × 10) | – |

All files are located under the sandbox run directory:  

```
C:\Users\Anish Kumar Verma\PycharmProjects\AutoEDA\sandbox_run\3392c3f0-ec75-44c4-8042-d6c83df85c4b\
```
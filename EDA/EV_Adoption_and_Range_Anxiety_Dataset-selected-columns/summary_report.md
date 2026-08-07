# Executive Summary – EV Adoption & Range‑Anxiety Dataset  
**Target Variable:** `Annual_Income_USD` (Regression)  

Prepared from the automated EDA pipeline (run on 10 000 × 10 data matrix). All findings below are derived from the generated artifact files listed in the working directory.

---  

## 1. Dataset Overview  

| Property                     | Value |
|------------------------------|-------|
| **Rows**                     | 10 000 |
| **Columns**                  | 10 |
| **Target Column**            | `Annual_Income_USD` |
| **Source File**              | `EV_Adoption_and_Range_Anxiety_Dataset-selected-columns.csv` |
| **Key Categorical Variables** | `Gender` (3 levels), `City_Type` (3 levels), `Current_Car_Type` (4 levels) |
| **Numeric Variables**        | `Age`, `Annual_Income_USD`, `Daily_Commute_km`, `Number_of_Cars_Owned`, `Charging_Stations_Near_Home`, `Charging_Stations_Near_Work` |

### 1.1 Column‑wise Metadata  

| Column                     | dtype   | Missing % | Cardinality | Key Stats (where applicable) |
|----------------------------|---------|----------|-------------|------------------------------|
| Buyer_ID                   | object  | 0.0 %    | 10 000      | – |
| Age                        | int64   | 0.0 %    | 45          | Mean = 46.94, Median = 47 |
| Gender                     | object  | 0.0 %    | 3           | Male = 5 201, Female = 4 498, Other = 301 |
| Annual_Income_USD          | float64 | 1.78 %   | 8 915       | Mean = 85 378.49, Median = 84 708 |
| City_Type                  | object  | 0.0 %    | 3           | Urban = 4 941, Suburban = 3 563, Rural = 1 496 |
| Daily_Commute_km           | float64 | 1.81 %   | 991         | Mean = 41.11, Median = 40.20 |
| Number_of_Cars_Owned      | int64   | 0.0 %    | 4           | Mean = 1.86, Median = 2 |
| Current_Car_Type           | object  | 0.0 %    | 4           | Sedan = 4 000, SUV = 3 486, Hatchback = 1 523 |
| Charging_Stations_Near_Home| int64   | 0.0 %    | 15          | Mean = 5.35, Median = 5 |
| Charging_Stations_Near_Work| int64   | 0.0 %    | 20          | Mean = 7.46, Median = 6 |

---  

## 2. Data Quality & Pre‑processing  

### 2.1 Missing‑value Handling  

* **Columns with missing data:** `Annual_Income_USD` (178 rows) and `Daily_Commute_km` (181 rows).  
* **Imputation strategy:**  
  * Numeric columns with skewness between –1.0 and +1.0 → **mean imputation**.  
  * Both columns had low skewness (0.31 & 0.34) → mean values used.  

| Column                | Missing Before | Imputed Method | Imputed Value |
|-----------------------|----------------|----------------|---------------|
| Annual_Income_USD    | 178 (1.78 %)   | Mean           | 85 378.49 |
| Daily_Commute_km     | 181 (1.81 %)   | Mean           | 41.1054 |

All other columns required no imputation.

### 2.2 Outlier Profiling  

Outliers were **profiled only** (no removal). The table below summarises the IQR‑based bounds and the proportion of records flagged as outliers.

| Feature                | Q1   | Q3   | IQR  | Lower Bound | Upper Bound | Outlier Count | Outlier % |
|------------------------|------|------|------|-------------|------------|---------------|-----------|
| Age                    | 36   | 58   | 22   | 3           | 91         | 0             | 0.00 % |
| Annual_Income_USD      | 61 883.5 | 106 938.5 | 45 055 | –5 699 | 174 521 | 58 | 0.58 % |
| Daily_Commute_km       | 23.6 | 56.7 | 33.1 | –26.05 | 106.35 | 44 | 0.44 % |
| Number_of_Cars_Owned   | 1    | 2    | 1    | –0.5 | 3.5 | 511 | 5.11 % |
| Charging_Stations_Near_Home | 2 | 8 | 6 | –7 | 17 | 0 | 0.00 % |
| Charging_Stations_Near_Work | 3 | 11 | 8 | –9 | 23 | 0 | 0.00 % |

*Only `Number_of_Cars_Owned` shows a modest outlier proportion (≈5 %). No automatic trimming was performed.*

---  

## 3. Exploratory Analysis  

### 3.1 Correlation Matrix (Numeric Features)  

The full heatmap is saved as **`correlation_matrix.png`** (≈ 87 KB). The strongest linear relationships (|r| > 0.25) are limited to the pair **Charging_Stations_Near_Home ↔ Charging_Stations_Near_Work** (r ≈ 0.48). All other absolute correlations are ≤ 0.03, indicating weak linear dependence.

| Rank | Feature 1                     | Feature 2                     | Pearson r |
|------|------------------------------|------------------------------|-----------|
| 1    | Charging_Stations_Near_Home  | Charging_Stations_Near_Work  | **0.4808** |
| 2    | Daily_Commute_km             | Charging_Stations_Near_Home  | 0.027 |
| 3    | Age                          | Daily_Commute_km             | –0.0136 |
| 4    | Age                          | Annual_Income_USD            | –0.0115 |
| 5    | Age                          | Charging_Stations_Near_Home  | –0.0081 |
| …    | *(remaining correlations ≤ 0.007)* | | |

*Interpretation:* The modest correlation between home‑ and work‑charging infrastructure suggests that households with more home chargers also tend to have more work‑place chargers, but none of the other numeric predictors show a meaningful linear link to the target.

### 3.2 Categorical Association  

Cramér’s V was computed for the three categorical variables (`Gender`, `City_Type`, `Current_Car_Type`). All pairwise associations are **0.0**, confirming statistical independence among these categories.

| Feature 1 | Feature 2 | Cramér’s V |
|-----------|-----------|------------|
| Gender    | City_Type | 0.0 |
| Gender    | Current_Car_Type | 0.0 |
| City_Type | Current_Car_Type | 0.0 |

### 3.3 Statistical Hypothesis Tests  

Each predictor was tested against the target (`Annual_Income_USD`) at α = 0.05.

| Predictor                | Test Type | Statistic | p‑value | Significant? | Effect Size |
|--------------------------|-----------|-----------|---------|--------------|-------------|
| Age                      | Pearson r | –0.0115   | 0.2513  | No | 0.0115 |
| Gender (3‑group ANOVA)   | ANOVA     | 0.5635    | 0.5692  | No | 0.0001 |
| City_Type (3‑group ANOVA)| ANOVA    | 0.6030    | 0.5472  | No | 0.0001 |
| Daily_Commute_km         | Pearson r | 0.0023    | 0.8143  | No | 0.0023 |
| Number_of_Cars_Owned     | Pearson r | 0.0042    | 0.6712  | No | 0.0042 |
| Current_Car_Type (4‑group ANOVA) | ANOVA | 2.0844 | 0.09999 | No | 0.0006 |
| Charging_Stations_Near_Home | Pearson r | 0.0047 | 0.6403 | No | 0.0047 |
| Charging_Stations_Near_Work | Pearson r | 0.0026 | 0.7938 | No | 0.0026 |

**Result:** *No predictor reached statistical significance.* Consequently, the `significant_predictors` list is empty.

---  

## 4. Visual Artifacts  

| File (PNG) | Approx. Size | Description |
|------------|--------------|-------------|
| `bivariate_Age_vs_Annual_Income_USD.png` | 218 KB | Scatter plot of Age vs. Income (no visible trend). |
| `bivariate_Age_vs_Daily_Commute_km.png` | 219 KB | Age vs. Commute distance – shows a flat cloud. |
| `bivariate_Charging_Stations_Near_Home_vs_Annual_Income_USD.png` | 96 KB | Home‑charging stations vs. Income – negligible slope. |
| `bivariate_Charging_Stations_Near_Home_vs_Charging_Stations_Near_Work.png` | 76 KB | Positive association (r ≈ 0.48). |
| `bivariate_City_Type_vs_Annual_Income_USD.png` | 45 KB | Box‑plot per city type – overlapping distributions. |
| `bivariate_Daily_Commute_km_vs_Annual_Income_USD.png` | 209 KB | Commute distance vs. Income – no pattern. |
| `bivariate_Gender_vs_Annual_Income_USD.png` | 42 KB | Income distribution by gender – virtually identical. |
| `bivariate_Number_of_Cars_Owned_vs_Annual_Income_USD.png` | 58 KB | Cars owned vs. Income – flat relationship. |
| `bivariate_Number_of_Cars_Owned_vs_Charging_Stations_Near_Work.png` | 50 KB | Cars owned vs. work‑charging stations – weak link. |
| `pairplot.png` | 545 KB | Pairwise scatter/ KDE plots for Age, Income, Commute, Cars owned. |
| `correlation_matrix.png` | 87 KB | Heatmap of Pearson correlations among numeric features. |
| `categorical_association_matrix.png` | 42 KB | Heatmap of Cramér’s V for the three categorical variables. |

*All images are stored in the sandbox run directory and can be opened directly for visual inspection.*

---  

## 5. Predictive Modeling Blueprint  

| Aspect | Recommendation |
|--------|----------------|
| **Problem Type** | Regression (`Annual_Income_USD`) |
| **Suggested Algorithms** | • Regularized Linear Regression (Ridge, Lasso) <br>• Random Forest Regressor <br>• Gradient Boosting Regressor <br>• Support Vector Regressor (SVR) |
| **Feature‑Selection Strategy** | 1. Drop high‑cardinality identifiers (`Buyer_ID`). <br>2. Rank features using cross‑validated permutation importance **and** mutual information. <br>3. Remove collinear features with |r| > 0.85 (none detected). |
| **Validation Strategy** | 5‑fold **K‑Fold Cross‑Validation**. <br>Metrics to report: MAE, RMSE, R², and residual‑error distribution. |
| **Over‑fitting Mitigation** | • Apply L1/L2 regularization (Ridge/Lasso). <br>• Limit tree depth / enforce minimum samples per leaf for ensemble methods. <br>• Conduct hyper‑parameter tuning **inside** the CV folds (no data leakage). |
| **Executive Summary (from blueprint)** | “Target: `Annual_Income_USD` (Regression). Use robust cross‑validation on 10 000 rows × 10 columns.” |

---  

## 6. Key Take‑aways & Recommendations  

1. **Data Quality** – After mean imputation, the dataset is complete; only a modest outlier proportion exists for `Number_of_Cars_Owned`. No aggressive cleaning is required.  
2. **Predictive Power** – None of the examined predictors shows a statistically significant linear relationship with income. This suggests that income may be driven by factors **outside** the current feature set (e.g., education, occupation, regional economics).  
3. **Modeling Strategy** – Because linear relationships are weak, **non‑linear ensemble methods** (Random Forest, Gradient Boosting) are advisable alongside regularized linear models for baseline comparison.  
4. **Feature Engineering** – No engineered features were generated automatically. Potential next steps:  
   * Create interaction terms (e.g., `Charging_Stations_Near_Home × Charging_Stations_Near_Work`).  
   * Encode categorical variables with target‑aware encodings (e.g., mean‑encoding of `Gender`).  
   * Consider external socioeconomic data (e.g., ZIP‑code median income) to enrich the model.  
5. **Further Analysis** –  
   * Perform **partial dependence** or **SHAP** analysis on tree‑based models to uncover hidden non‑linear effects.  
   * Explore **cluster analysis** on the charging‑station variables to see if distinct user groups exist.  

---  

## 7. Appendices  

### A. Full Correlation Matrix (numeric)  

```
                Age  Annual_Income_USD  Daily_Commute_km  Number_of_Cars_Owned  Charging_Stations_Near_Home  Charging_Stations_Near_Work
Age                1.0           -0.011           -0.014                -0.003                     -0.008                     -0.002
Annual_Income_USD -0.011            1.0            0.002                 0.004                      0.005                      0.003
Daily_Commute_km  -0.014            0.002            1.0                -0.007                      0.027                      0.003
Number_of_Cars_Owned -0.003        0.004           -0.007                1.0                      0.004                      0.000
Charging_Stations_Near_Home -0.008 0.005           0.027                0.004                      1.0                      0.481
Charging_Stations_Near_Work -0.002 0.003           0.003                0.000                      0.481                      1.0
```

### B. Full Hypothesis‑Test Summary  

(Repeated from Section 3.3 for completeness.)

| Predictor                | Test | Statistic | p‑value | Significant? |
|--------------------------|------|-----------|---------|--------------|
| Age                      | Pearson r | –0.0115 | 0.2513 | No |
| Gender                   | ANOVA | 0.5635 | 0.5692 | No |
| City_Type                | ANOVA | 0.6030 | 0.5472 | No |
| Daily_Commute_km         | Pearson r | 0.0023 | 0.8143 | No |
| Number_of_Cars_Owned     | Pearson r | 0.0042 | 0.6712 | No |
| Current_Car_Type         | ANOVA | 2.0844 | 0.09999 | No |
| Charging_Stations_Near_Home | Pearson r | 0.0047 | 0.6403 | No |
| Charging_Stations_Near_Work | Pearson r | 0.0026 | 0.7938 | No |

---  

**Prepared by:** Senior Lead Data Scientist  
**Date:** 2026‑08‑07  

*All tables, statistics, and visual references are derived from the automatically generated EDA artifacts.*
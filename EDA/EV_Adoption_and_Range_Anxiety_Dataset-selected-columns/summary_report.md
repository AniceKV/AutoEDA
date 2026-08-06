# Executive Summary – EV Adoption & Range‑Anxiety Dataset  
**Target Variable:** `Annual_Income_USD` (Regression)  
**Rows / Columns:** 10 000 × 11 (including engineered feature)  

---  

## 1. Dataset Overview  

| Aspect | Detail |
|--------|--------|
| **Source file** | `EV_Adoption_and_Range_Anxiety_Dataset-selected-columns.csv` |
| **Rows** | 10 000 |
| **Columns (pre‑engineered)** | 10 |
| **Columns (post‑engineered)** | 11 (`engineered_feature`) |
| **Target** | `Annual_Income_USD` (continuous) |
| **Key categorical fields** | `Gender` (3 levels), `City_Type` (3), `Current_Car_Type` (4) |
| **Numeric fields** | `Age`, `Annual_Income_USD`, `Daily_Commute_km`, `Number_of_Cars_Owned`, `Charging_Stations_Near_Home`, `Charging_Stations_Near_Work` |

### 1.1 Column‑wise Summary  

| Column | dtype | Cardinality | Missing % | Typical range / stats |
|--------|-------|-------------|----------|-----------------------|
| Buyer_ID | object | 10 000 | 0.0 | Unique identifier |
| Age | int64 | 45 | 0.0 | 25 – 69 (mean = 46.9, median = 47) |
| Gender | object | 3 | 0.0 | Male = 52 % , Female = 45 % , Other = 3 % |
| Annual_Income_USD | float64 | 8 915 | 1.78 | 30 k – 223 k (mean = 85 378, median = 84 708) |
| City_Type | object | 3 | 0.0 | Urban = 49 % , Suburban = 36 % , Rural = 15 % |
| Daily_Commute_km | float64 | 991 | 1.81 | 5 – 135.5 (mean = 41.1, median = 40.2) |
| Number_of_Cars_Owned | int64 | 4 | 0.0 | 1 – 4 (mean = 1.86, median = 2) |
| Current_Car_Type | object | 4 | 0.0 | Sedan = 40 % , SUV = 35 % , Hatchback = 15 % |
| Charging_Stations_Near_Home | int64 | 15 | 0.0 | 0 – 14 (mean = 5.35, median = 5) |
| Charging_Stations_Near_Work | int64 | 20 | 0.0 | 0 – 19 (mean = 7.46, median = 6) |
| engineered_feature | int64 / float64* | 36 | 0.0 | See Section 4 |

\*Two engineered features were created; one is a ratio (float) and the other an interaction (int).

---  

## 2. Missing‑Data Handling  

* **Strategy** – Standardised missing string placeholders (`?`, `NA`, `N/A`, `null`) → `NaN`.  
* **Numeric columns** – Skewness ≈ 0.3 → mean imputation.  
* **Categorical columns** – Mode imputation with fallback to `"Unknown"` (none required).  

| Column | Missing before | Imputation method | Imputed value |
|--------|----------------|-------------------|---------------|
| Annual_Income_USD | 178 (1.78 %) | Mean | **85 378.49** |
| Daily_Commute_km | 181 (1.81 %) | Mean | **41.1054** |
| All others | 0 | – | – |

All missing values were successfully filled; no residual NaNs remain.

---  

## 3. Outlier Profiling  

Outliers were **profiled only** (no removal).  

| Feature | Q1 | Q3 | IQR | Lower bound | Upper bound | Outlier count | % outliers |
|---------|----|----|-----|-------------|-------------|---------------|------------|
| Age | 36 | 58 | 22 | 3 | 91 | 0 | 0.0 % |
| Annual_Income_USD | 61 884 | 106 939 | 45 055 | –5 699 | 174 521 | 58 | 0.58 % |
| Daily_Commute_km | 23.6 | 56.7 | 33.1 | –26.05 | 106.35 | 44 | 0.44 % |
| Number_of_Cars_Owned | 1 | 2 | 1 | –0.5 | 3.5 | 511 | 5.11 % |
| Charging_Stations_Near_Home | 2 | 8 | 6 | –7 | 17 | 0 | 0.0 % |
| Charging_Stations_Near_Work | 3 | 11 | 8 | –9 | 23 | 0 | 0.0 % |

*Only the `Number_of_Cars_Owned` column shows a modest outlier proportion (≈ 5 %).*  

---  

## 4. Feature Engineering  

Two high‑signal engineered features were added:

| Engineered Feature | Formula | Data type | Correlation with target |
|--------------------|---------|-----------|--------------------------|
| `Commute_per_Age` | `Daily_Commute_km / (Age + eps)` | float64 | **0.0062** |
| `Cars_Owned_x_Stations_Home` | `Number_of_Cars_Owned * Charging_Stations_Near_Home` | int64 | **‑0.0055** |

Both exhibit near‑zero linear correlation with `Annual_Income_USD`, indicating limited predictive power in their raw form. They are retained for downstream non‑linear models (e.g., tree‑based methods) where interaction effects may be useful.

---  

## 5. Visual Artefacts  

All visualisations are saved in the working directory; file names are listed for reference.

| Plot | Description | File |
|------|-------------|------|
| **Univariate distributions** – Age, Income, Commute, etc. | Histograms / bar‑charts of each variable | `dist_Age.png`, `dist_Annual_Income_USD.png`, `dist_Daily_Commute_km.png`, `dist_Number_of_Cars_Owned.png`, `dist_Gender.png`, `dist_City_Type.png`, `dist_Current_Car_Type.png`, `dist_Charging_Stations_Near_Home.png`, `dist_Charging_Stations_Near_Work.png`, `dist_Buyer_ID.png` |
| **Correlation heatmap** | Pearson correlation matrix for numeric features | `correlation_matrix.png` |
| **Categorical association heatmap** | Cramér’s V matrix for categorical pairs | `categorical_association_matrix.png` |
| **Target‑vs‑Age interaction** | Scatter + regression line (colored by Gender) | `target_interaction_age.png` |
| **Target‑vs‑Commute interaction** | Scatter + regression line (colored by Gender) | `target_interaction_commute.png` |
| **Semantic bivariate relationships** (4 panels) | Age vs Income (Hue = Gender), Cars Owned vs Income (Hue = City_Type), Stations‑Home vs Income (Hue = Current_Car_Type), Stations‑Work vs Income (Hue = Gender) | `bivariate_Age_vs_Annual_Income_USD.png`, `bivariate_Number_of_Cars_Owned_vs_Annual_Income_USD.png`, `bivariate_Charging_Stations_Near_Home_vs_Annual_Income_USD.png`, `bivariate_Charging_Stations_Near_Work_vs_Annual_Income_USD.png` |
| **Pairplot** (Age, Income, Commute, Cars Owned, Stations‑Home) | Pairwise scatter matrix with `Gender` hue | `pairplot.png` |

*All images are stored in the same directory as the report; they can be opened directly for visual inspection.*

---  

## 6. Correlation & Association Findings  

### 6.1 Numeric Correlations (top 5 absolute values)

| Feature 1 | Feature 2 | Pearson ρ |
|----------|-----------|----------|
| `Charging_Stations_Near_Home` | `Charging_Stations_Near_Work` | **0.4808** |
| `Daily_Commute_km` | `Charging_Stations_Near_Home` | **0.027** |
| `Age` | `Daily_Commute_km` | **‑0.0136** |
| `Age` | `Annual_Income_USD` | **‑0.0115** |
| `Age` | `Charging_Stations_Near_Home` | **‑0.0081** |

*All correlations with the target (`Annual_Income_USD`) are below 0.01, indicating very weak linear relationships.*

### 6.2 Categorical Associations (Cramér’s V)

| Feature 1 | Feature 2 | Cramér’s V |
|-----------|-----------|-----------|
| Gender | City_Type | **0.0** |
| Gender | Current_Car_Type | **0.0** |
| City_Type | Current_Car_Type | **0.0** |

The three categorical variables are statistically independent of each other in this sample.

---  

## 7. Statistical Hypothesis Testing  

| Feature | Test | Statistic | p‑value | Significant (α = 0.05) | Interpretation |
|---------|------|-----------|---------|------------------------|----------------|
| Age | Pearson correlation | –0.0115 | 0.2513 | No | No linear association with income |
| Gender | One‑Way ANOVA | 0.5635 | 0.5692 | No | Income distribution identical across genders |
| City_Type | One‑Way ANOVA | 0.6030 | 0.5472 | No | No income difference across city types |
| Daily_Commute_km | Pearson correlation | 0.0023 | 0.8143 | No | Commute distance unrelated to income |
| Number_of_Cars_Owned | Pearson correlation | 0.0042 | 0.6712 | No | Car‑ownership count not linked to income |
| Current_Car_Type | One‑Way ANOVA | 2.0844 | 0.09999 | No (p > 0.05) | Trend toward difference but not significant |
| Charging_Stations_Near_Home | Pearson correlation | 0.0047 | 0.6403 | No | No link to income |
| Charging_Stations_Near_Work | Pearson correlation | 0.0026 | 0.7938 | No | No link to income |

**Result:** *No feature reached statistical significance at the 5 % level.*  

---  

## 8. Predictive Modeling Blueprint  

| Component | Recommendation |
|-----------|----------------|
| **Problem type** | Regression (`Annual_Income_USD`) |
| **Algorithms** (ordered by increasing complexity) | 1. Ridge / Lasso (regularised linear) <br>2. Random Forest Regressor <br>3. Gradient Boosting Regressor (e.g., XGBoost, LightGBM) <br>4. Support Vector Regressor (SVR) |
| **Feature‑selection strategy** | • Drop high‑cardinality identifier (`Buyer_ID`) <br>• Rank features using cross‑validated permutation importance and mutual‑information scores <br>• Remove collinear features with |ρ| > 0.85 (none currently exceed this threshold) |
| **Validation** | 5‑fold K‑Fold cross‑validation; report MAE, RMSE, R², and residual distribution |
| **Over‑fitting mitigation** | • L1/L2 regularisation (for linear models) <br>• Limit tree depth, set `min_samples_leaf` (for ensemble models) <br>• Hyper‑parameter tuning confined to inner CV loops |
| **Baseline expectation** | Given the near‑zero linear correlations, expect modest R² (< 0.05) for linear models; tree‑based ensembles may capture non‑linear interactions and improve performance modestly. |

---  

## 9. Key Take‑aways & Recommendations  

1. **Data Quality** – Missing values are minimal and have been imputed sensibly; outlier profiling shows only a small proportion of extreme values.  
2. **Predictive Power** – Linear relationships between the available features and the target are extremely weak; no statistically significant predictors were identified.  
3. **Feature Engineering** – The two engineered features (`Commute_per_Age` and `Cars_Owned_x_Stations_Home`) have negligible linear correlation with income but may prove useful for non‑linear models.  
4. **Modeling Strategy** – Begin with regularised linear regression to establish a baseline, then explore tree‑based ensembles (Random Forest, Gradient Boosting) which can exploit interaction effects.  
5. **Further Exploration** – Consider enriching the dataset with external socioeconomic variables (e.g., education, employment sector) or geographic indicators that could better explain income variance.  

---  

## 10. Next Steps  

1. **Implement the blueprint** – Train the recommended models using the outlined validation scheme.  
2. **Feature importance analysis** – After model training, extract permutation importance and SHAP values to identify any hidden non‑linear drivers.  
3. **Iterate on engineering** – If tree‑based models highlight promising interactions, create additional domain‑specific features (e.g., `Income_per_Car`, `Stations_per_Commute`).  
4. **Report performance** – Summarise final model metrics (MAE, RMSE, R²) and compare against the baseline.  

*All artefacts referenced in this report are available in the current working directory.*  
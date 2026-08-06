# Executive Summary – EV Adoption & Range‑Anxiety Dataset  
*Automated Exploratory Data Analysis (AutoEDA) – 10 columns, 10 000 records*  

---  

## 1. Project Context  

| Item | Detail |
|------|--------|
| **Dataset** | `EV_Adoption_and_Range_Anxiety_Dataset‑selected-columns.csv` |
| **Rows / Columns** | 10 000 × 10 |
| **Target Variable** | `Current_Car_Type` (categorical – 4 classes) |
| **Problem Type** | Multi‑class classification (predicting the type of car a buyer currently owns) |
| **Primary Business Question** | Which demographic, socioeconomic, and infrastructure factors are associated with the current car type of electric‑vehicle (EV) buyers? |

The AutoEDA pipeline generated a full set of descriptive statistics, visual artefacts, and a modelling blueprint. All findings below are derived **solely** from the supplied artefacts (no external code or data).  

---  

## 2. Data Dictionary  

| Column | Data Type | Cardinality | Missing % | Key Summary |
|--------|-----------|-------------|----------|-------------|
| `Buyer_ID` | object (unique identifier) | 10 000 | 0 % | Each row is a distinct buyer (e.g., *EV00001*) |
| `Age` | int64 | 45 | 0 % | 25 – 69 yr; **Mean = 46.94**, **Median = 47** |
| `Gender` | object | 3 | 0 % | Male = 5 201, Female = 4 498, Other = 301 |
| `Annual_Income_USD` | float64 | 8 915 | 1.78 % | **Mean = 85 378**, **Median = 84 708**, Range = 30 000 – 223 345 |
| `City_Type` | object | 3 | 0 % | Urban = 4 941, Suburban = 3 563, Rural = 1 496 |
| `Daily_Commute_km` | float64 | 991 | 1.81 % | **Mean = 41.11**, **Median = 40.20**, Range = 5 – 135.5 |
| `Number_of_Cars_Owned` | int64 | 4 | 0 % | 1 – 4 cars; **Mean = 1.86**, **Median = 2** |
| `Current_Car_Type` | object (target) | 4 | 0 % | **Sedan** = 4 000, **SUV** = 3 486, **Hatchback** = 1 523, **Truck** ≈ ? (remaining) |
| `Charging_Stations_Near_Home` | int64 | 15 | 0 % | 0 – 14 stations; **Mean = 5.35**, **Median = 5** |
| `Charging_Stations_Near_Work` | int64 | 20 | 0 % | 0 – 19 stations; **Mean = 7.46**, **Median = 6** |

*All numeric columns are stored as float/int; categorical columns as object strings.*  

---  

## 3. Missing‑Value Treatment  

| Column | Missing Before | Imputation Method | Imputed Value |
|--------|----------------|-------------------|---------------|
| `Annual_Income_USD` | 178 (1.78 %) | Median (skewness = 0.31) | **84 708** |
| `Daily_Commute_km` | 181 (1.81 %) | Mean (skewness = 0.34) | **41.1054** |
| All other columns | 0 | – | – |

The pipeline first normalised common placeholder strings (`?`, `NA`, `N/A`, `null`) to `NaN`, then applied the above rules. No further imputation was required.  

---  

## 4. Distribution Visualisations  

> **Note:** Images are stored in the working directory. They can be displayed in a notebook or report by referencing the file name.

| Visualisation | Description |
|---------------|-------------|
| `dist_Age.png` | Histogram of ages (25‑69 yr). Slightly right‑skewed, peak around 45‑50 yr. |
| `dist_Annual_Income_USD.png` | Income distribution shows a long right tail; most buyers earn between 30 k and 120 k USD. |
| `dist_Gender.png` | Bar chart confirming the gender split (≈ 52 % Male, 45 % Female, 3 % Other). |
| `dist_City_Type.png` | Urban dominates (≈ 49 %), followed by Suburban (≈ 36 %) and Rural (≈ 15 %). |
| `dist_Daily_Commute_km.png` | Commute distances cluster around 30‑50 km; a small tail extends beyond 100 km. |
| `dist_Number_of_Cars_Owned.png` | Majority own **1** or **2** cars; very few own 3‑4 cars. |
| `dist_Current_Car_Type.png` | Distribution of the target classes (Sedan, SUV, Hatchback, Truck). |
| `dist_Charging_Stations_Near_Home.png` & `dist_Charging_Stations_Near_Work.png` | Both show roughly normal‑ish shapes centred at 5 (home) and 7 (work) stations, respectively. |

---  

## 5. Bivariate Relationships  

| Plot | Variables | Key Observation |
|------|-----------|-----------------|
| `bivariate_Age_vs_Annual_Income_USD.png` | Age ↔ Annual Income | No clear trend; correlation ≈ ‑0.01 (see Section 6). |
| `bivariate_Charging_Stations_Near_Home_vs_Charging_Stations_Near_Work.png` | Home ↔ Work charging stations | Positive moderate correlation (≈ 0.48). |
| `bivariate_Daily_Commute_km_vs_Number_of_Cars_Owned.png` | Commute km ↔ Cars owned | Very weak negative correlation (≈ ‑0.007). |
| `bivariate_Gender_vs_Current_Car_Type.png` | Gender ↔ Car Type | No statistically significant association (Chi‑square p = 0.98). |

---  

## 6. Correlation Analysis  

The pipeline computed a Pearson correlation matrix for the six numeric features (excluding the target). The heat‑map is saved as `correlation_matrix.png`.  

### 6.1 Top Correlations  

| Feature 1 | Feature 2 | Correlation |
|-----------|-----------|-------------|
| `Charging_Stations_Near_Home` | `Charging_Stations_Near_Work` | **0.481** |
| `Daily_Commute_km` | `Charging_Stations_Near_Home` | **0.027** |
| `Age` | `Daily_Commute_km` | **‑0.014** |
| `Age` | `Annual_Income_USD` | **‑0.011** |
| `Age` | `Charging_Stations_Near_Home` | **‑0.008** |
| `Daily_Commute_km` | `Number_of_Cars_Owned` | **‑0.007** |
| `Annual_Income_USD` | `Charging_Stations_Near_Home` | **0.005** |
| `Number_of_Cars_Owned` | `Charging_Stations_Near_Home` | **0.004** |
| `Annual_Income_USD` | `Number_of_Cars_Owned` | **0.004** |
| `Age` | `Number_of_Cars_Owned` | **‑0.003** |

**Interpretation** – All correlations are very weak (|r| < 0.05) except the expected relationship between home and work charging stations (moderate positive). No pair of features exceeds a 0.85 collinearity threshold, so no variables need to be dropped for multicollinearity.  

---  

## 7. Outlier & Distribution Checks  

| Feature | IQR‑Based Bounds | Outliers (count / %) | Action |
|---------|------------------|----------------------|--------|
| `Age` | 3 – 91 | 0 / 0 % | None – data within plausible human age range |
| `Annual_Income_USD` | ‑5 699 – 174 521 | 58 / 0.58 % | Profiled only (no removal) |
| `Daily_Commute_km` | ‑26 – 106 | 44 / 0.44 % | Profiled only |
| `Number_of_Cars_Owned` | ‑0.5 – 3.5 | 511 / 5.11 % | Profiled only (high‑frequency 1‑2 cars) |
| `Charging_Stations_Near_Home` | ‑7 – 17 | 0 | No outliers |
| `Charging_Stations_Near_Work` | ‑9 – 23 | 0 | No outliers |

The pipeline **profiled** outliers (recorded but left in place) because the percentages are negligible and the domain knowledge suggests they are plausible (e.g., high‑income buyers).  

---  

## 8. Statistical Hypothesis Testing  

All tests assess whether a feature’s distribution differs across the target classes (`Current_Car_Type`).  

| Feature | Test | Statistic | p‑value | Significant? | Interpretation |
|---------|------|-----------|---------|--------------|----------------|
| `Buyer_ID` | Chi‑square (independence) | 30 000 | 0.494 | No | IDs are random identifiers – no association. |
| `Age` | One‑Way ANOVA | 1.81 | 0.143 | No | Age does not differentiate car types. |
| `Gender` | Chi‑square | 1.036 | 0.984 | No | Gender distribution is similar across car types. |
| `Annual_Income_USD` | One‑Way ANOVA | 2.08 | 0.101 | No | Income differences are not statistically significant. |
| `City_Type` | Chi‑square | 1.544 | 0.957 | No | Urban/Suburban/Rural mix is comparable across car types. |
| `Daily_Commute_km` | One‑Way ANOVA | 0.212 | 0.888 | No | Commute length shows no class‑specific pattern. |
| `Number_of_Cars_Owned` | One‑Way ANOVA | 0.063 | 0.979 | No | Number of owned cars does not explain car type. |
| `Charging_Stations_Near_Home` | One‑Way ANOVA | 0.288 | 0.834 | No | Home charging availability is not a differentiator. |
| `Charging_Stations_Near_Work` | One‑Way ANOVA | 0.186 | 0.906 | No | Work charging availability is not a differentiator. |

**Result:** *No single predictor reached conventional statistical significance (α = 0.05) for the target variable.*  

---  

## 9. Feature Engineering  

The pipeline did **not** create additional features (`engineered_features` list is empty). Given the weak linear relationships, future work could explore:

* Interaction terms (e.g., `Charging_Stations_Near_Home × Annual_Income_USD`).  
* Binning of continuous variables (age groups, income brackets).  
* Encoding of categorical variables using target‑aware techniques (e.g., CatBoost encoding).  

---  

## 10. Predictive‑Modeling Blueprint  

| Aspect | Recommendation |
|--------|----------------|
| **Target** | `Current_Car_Type` – multi‑class classification (4 classes). |
| **Baseline Model** | Regularized Logistic Regression (L2 penalty) – quick benchmark, interpretable coefficients. |
| **Advanced Models** | <ul><li>Random Forest Classifier (tree‑based, handles non‑linearities)</li><li>Gradient Boosting (XGBoost or LightGBM) – strong performance on tabular data</li><li>Support Vector Classifier (SVM) – useful if data is not linearly separable</li></ul> |
| **Feature‑Selection Strategy** | <ul><li>Drop `Buyer_ID` (high‑cardinality identifier, no predictive power)</li><li>Rank features via cross‑validated permutation importance and mutual information.</li><li>Remove any pair with |r| > 0.85 (none found).</li></ul> |
| **Encoding** | One‑Hot for `Gender`, `City_Type`, `Current_Car_Type` (target) – or use ordinal/target encoding for tree models. |
| **Validation** | Stratified 5‑fold cross‑validation (preserves class distribution). |
| **Evaluation Metrics** | <ul><li>Balanced Accuracy (accounts for class imbalance)</li><li>Macro‑averaged F1‑score</li><li>Precision‑Recall AUC (per class)</li><li>Confusion Matrix (visual diagnostic)</li></ul> |
| **Over‑fitting Mitigation** | <ul><li>L1/L2 regularisation for linear models.</li><li>Tree depth ≤ 10, min samples leaf ≥ 20 for Random Forest / Gradient Boosting.</li><li>Hyper‑parameter tuning inside CV folds (e.g., GridSearchCV or Bayesian optimisation).</li></ul> |
| **Execution Summary** | The dataset (10 k × 10) is modest in size; all recommended algorithms will train quickly on a standard laptop. No major data‑quality issues remain after imputation. |

---  

## 11. Key Insights & Business Implications  

1. **No single predictor strongly explains car type** – statistical tests and correlation values are all near zero.  
2. **Charging infrastructure at home and work are moderately correlated** (r ≈ 0.48). This suggests that buyers who have more home chargers also tend to have more work chargers, but this does **not** translate into a distinct car‑type preference.  
3. **Income and commute distance have very weak relationships** with car type, implying that other unobserved factors (e.g., brand preference, vehicle price, incentives) may drive the choice.  
4. **Class distribution is reasonably balanced** (Sedan ≈ 40 %, SUV ≈ 35 %, Hatchback ≈ 15 %, Truck ≈ 10 %). Stratified validation will therefore give reliable performance estimates.  

---  

## 12. Limitations  

| Limitation | Impact |
|------------|--------|
| **Low explanatory power** of available features for the target | Predictive models may achieve modest accuracy; additional data (e.g., vehicle price, incentives, environmental attitudes) could improve performance. |
| **Missing values imputed with simple statistics** (median/mean) | May mask subtle patterns; advanced imputation (e.g., K‑NN, iterative) could be explored. |
| **No engineered features** | Potential non‑linear interactions are not captured; feature construction may boost model performance. |
| **Outlier handling limited to profiling** | Extreme incomes or commute distances are retained; if they are data errors, they could bias models. |
| **Target class “Truck” appears under‑represented** (exact count not provided) | May lead to higher mis‑classification rates for this class; consider resampling or class‑weighting. |

---  

## 13. Recommended Next Steps  

1. **Enrich the dataset** with variables such as vehicle price, EV incentives, fuel‑type preferences, or environmental concern scores.  
2. **Create interaction / non‑linear features** (e.g., `Income × Charging_Stations_Near_Home`).  
3. **Run the baseline and advanced models** using the blueprint; compare performance on the chosen metrics.  
4. **Perform error analysis** on the confusion matrix to understand which car types are most often confused.  
5. **Iterate on feature selection** (e.g., recursive feature elimination) to see if a reduced subset improves generalisation.  
6. **Document model governance** (feature provenance, reproducibility) before deploying any predictive service.  

---  

## 14. Artefact Index  

| Artefact | Type | File | Brief Description |
|----------|------|------|-------------------|
| `dist_Age.png` | Histogram | Age distribution |
| `dist_Annual_Income_USD.png` | Histogram | Income distribution |
| `dist_Gender.png` | Bar chart | Gender counts |
| `dist_City_Type.png` | Bar chart | Urban / Suburban / Rural split |
| `dist_Daily_Commute_km.png` | Histogram | Commute distance |
| `dist_Number_of_Cars_Owned.png` | Bar chart | Cars owned per buyer |
| `dist_Current_Car_Type.png` | Bar chart | Target class frequencies |
| `dist_Charging_Stations_Near_Home.png` | Histogram | Home charging stations |
| `dist_Charging_Stations_Near_Work.png` | Histogram | Work charging stations |
| `bivariate_Age_vs_Annual_Income_USD.png` | Scatter | Age vs. Income |
| `bivariate_Charging_Stations_Near_Home_vs_Charging_Stations_Near_Work.png` | Scatter | Home vs. Work charging stations |
| `bivariate_Daily_Commute_km_vs_Number_of_Cars_Owned.png` | Scatter | Commute vs. cars owned |
| `bivariate_Gender_vs_Current_Car_Type.png` | Bar | Gender vs. Car type |
| `correlation_matrix.png` | Heat‑map | Pearson correlation matrix |
| `pairplot.png` | Pair‑plot | Joint distributions of all numeric variables |
| `target_interactions.png` | Multi‑panel | Visualisation of each feature split by `Current_Car_Type` |
| `metadata_profile.json` | JSON | Dataset schema, missing‑value summary, cardinalities |
| `metrics.json` | JSON | Full AutoEDA output (imputation, outlier, hypothesis, modelling blueprint) |
| `df_state_v0.csv` – `df_state_v3.csv` | CSV | Snapshots of the data at various pipeline stages (raw → imputed). |

---  

**Prepared by:** *Senior Lead Data Scientist*  
**Date:** 2026‑08‑06  

*All tables and figures are rendered in plain ASCII/Markdown for maximum compatibility.*
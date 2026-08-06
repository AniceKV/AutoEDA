# Executive Summary – Fertility Dataset (Diagnosis Prediction)

**Prepared by:** Senior Lead Data Scientist  
**Date:** 2026‑08‑06  

---  

## 1. Dataset Overview  

| Property | Value |
|----------|-------|
| **File** | `fertility.csv` (referenced as `current_df.csv`) |
| **Rows** | 100 |
| **Columns** | 10 |
| **Target column** | `Diagnosis` (binary: *Normal* / *Altered*) |
| **Column Types** | 2 numeric (`Age`, `Number of hours spent sitting per day`); 8 categorical (object) |
| **Cardinality** | 4–5 distinct values for most categorical fields; 10 distinct ages; 14 distinct sitting‑hour values |
| **Class balance** | *Normal*: 88 % (88 rows) – *Altered*: 12 % (12 rows) |

### 1.1 Schema Snapshot  

| Column | Type | Distinct values | Key stats / notes |
|--------|------|----------------|-------------------|
| Season | object | 4 | `spring` (37), `fall` (31), `winter` (28) |
| Age | int64 | 10 | Mean = 30.11, Median = 30, Std = 2.25, Skew = 0.67 |
| Childish diseases | object | 2 | `yes` (87), `no` (13) |
| Accident or serious trauma | object | 2 | `yes` (44), `no` (56) |
| Surgical intervention | object | 2 | `yes` (51), `no` (49) |
| High fevers in the last year | object | 3 | `more than 3 months ago` (63), `no` (28), `less than 3 months ago` (9) |
| Frequency of alcohol consumption | object | 5 | `hardly ever or never` (40), `once a week` (39), `several times a week` (19) |
| Smoking habit | object | 3 | `never` (56), `occasional` (23), `daily` (21) |
| Number of hours spent sitting per day | int64 | 14 | Mean = 10.8, Median = 7, **Highly skewed** (skew = 9.85, kurtosis ≈ 98) |
| Diagnosis | object | 2 | `Normal` (88), `Altered` (12) |

*No missing values were detected in any column.*

---  

## 2. Data Quality & Pre‑processing  

| Step | Action | Outcome |
|------|--------|---------|
| **Missing‑value handling** | Standardised placeholders → `NaN`; numeric imputation based on skewness; categorical mode imputation with fallback “Unknown”. | No missing values before or after – nothing imputed. |
| **Outlier profiling** | IQR method on `Age` and `Number of hours spent sitting per day`. | `Age`: 0 % outliers (bounds 22‑38).<br>`Sitting hours`: 5 % outliers (5 rows > 15 h). Action = *profile* (no removal). |
| **Feature engineering** | Requested: `log1p(Number of hours spent sitting per day)` → `log_hours`; interaction `Age × Number of hours spent sitting per day` → `age_hours_interaction`. | **0 new features generated** – the step completed but produced no columns (likely due to implementation guard). |
| **Imputation summary** | See Table 1. | All columns retained original values. |

---  

## 3. Exploratory Visualisations  

| Image | Size (KB) | Description |
|-------|-----------|-------------|
| `dist_Age.png` | 39.78 | Histogram + KDE of age (27‑36). Slight right‑skew, consistent with numeric summary. |
| `dist_Number_of_hours_spent_sitting_per_day.png` | 37.85 | Highly right‑skewed distribution; long tail up to 342 h (outlier region). |
| `dist_Smoking_habit.png` | 31.87 | Bar chart of smoking frequency (`never`, `occasional`, `daily`). |
| `dist_Frequency_of_alcohol_consumption.png` | 52.88 | Bar chart of alcohol consumption categories. |
| `dist_Season.png` | 32.40 | Bar chart of seasonal counts. |
| `correlation_matrix.png` | 40.30 | Heatmap of Pearson correlations (numeric only). Only non‑trivial entry: Age ↔ Sitting hours = **‑0.047** (practically zero). |
| `categorical_association_matrix.png` | 140.25 | Heatmap of Cramér’s V for all categorical pairs. Highest association: **Surgical intervention ↔ High fevers** (V = 0.244). |
| `bivariate_Age_vs_Diagnosis.png` | 29.24 | Box/violin plot of age by diagnosis – overlapping distributions. |
| `bivariate_Number_of_hours_spent_sitting_per_day_vs_Diagnosis.png` | 44.91 | Box/violin plot of sitting hours by diagnosis – no clear separation. |
| `bivariate_Smoking_habit_vs_Diagnosis.png` | 39.72 | Bar plot of smoking habit vs. diagnosis – proportions similar across classes. |
| `bivariate_Frequency_of_alcohol_consumption_vs_Diagnosis.png` | 63.48 | Bar plot of alcohol frequency vs. diagnosis – no visible trend. |
| `bivariate_Season_vs_Diagnosis.png` | 39.57 | Bar plot of season vs. diagnosis – slight variation, not statistically significant. |

*All visual assets are stored under `./sandbox_run/` (paths shown in the JSON metadata).*

---  

## 4. Correlation & Association Analysis  

### 4.1 Numeric Correlations  

Only two numeric columns exist; the absolute Pearson correlation is **0.047** (negative). This indicates virtually no linear relationship between age and sitting time.

| Feature 1 | Feature 2 | Pearson r |
|-----------|-----------|-----------|
| Age | Number of hours spent sitting per day | **‑0.047** |

### 4.2 Categorical Associations (Cramér’s V)  

| Feature 1 | Feature 2 | Cramér’s V |
|-----------|-----------|------------|
| Surgical intervention | High fevers in the last year | **0.244** |
| Season | Accident or serious trauma | **0.225** |
| Frequency of alcohol consumption | Smoking habit | **0.210** |
| Season | High fevers in the last year | **0.203** |
| Accident or serious trauma | Frequency of alcohol consumption | **0.177** |
| … | … | (others ≤ 0.13) |
| **Season ↔ Diagnosis** | **0.107** |
| **Accident or serious trauma ↔ Diagnosis** | **0.100** |

*All Cramér’s V values are modest (< 0.25), indicating weak to moderate association. None reach a level that would suggest strong predictive power.*

---  

## 5. Statistical Hypothesis Testing  

A two‑sample Welch t‑test was used for numeric variables, and a chi‑square test of independence for categorical variables, with α = 0.05.

| Variable | Test | Statistic | p‑value | Significant? | Interpretation |
|----------|------|-----------|---------|--------------|----------------|
| Age | Welch t‑test | 1.0435 | 0.313 | **No** | Age distributions for *Normal* vs *Altered* overlap. |
| Number of hours spent sitting per day | Welch t‑test | –0.9024 | 0.369 | **No** | No evidence of difference in sitting time across diagnoses. |
| Season | Chi‑square | 4.1613 | 0.245 | **No** | Seasonal distribution does not differ by diagnosis. |
| Childish diseases | Chi‑square | 0.0000 | 1.000 | **No** | No association. |
| Accident or serious trauma | Chi‑square | 1.2177 | 0.270 | **No** | No association. |
| Surgical intervention | Chi‑square | 0.0547 | 0.815 | **No** | No association. |
| High fevers in the last year | Chi‑square | 1.5452 | 0.462 | **No** | No association. |
| Frequency of alcohol consumption | Chi‑square | 4.0263 | 0.402 | **No** | No association. |
| Smoking habit | Chi‑square | 0.2153 | 0.898 | **No** | No association. |

**Result:** *No predictor reached statistical significance.* The `significant_predictors` list is empty.

---  

## 6. Predictive Modeling Blueprint  

| Aspect | Recommendation |
|--------|----------------|
| **Problem type** | Binary classification (`Diagnosis`). |
| **Baseline algorithm** | Regularized Logistic Regression (L1/L2). |
| **Strong candidates** | - Random Forest Classifier  <br> - Gradient Boosting (XGBoost / LightGBM) <br> - Support Vector Classifier (SVM) |
| **Feature selection** | 1. Remove any high‑cardinality ID/text columns (none present). <br> 2. Rank features using cross‑validated permutation importance **and** mutual information. <br> 3. Drop collinear features with |r| > 0.85 (none observed). |
| **Validation strategy** | Stratified 5‑fold cross‑validation (preserves 12 % *Altered* class). |
| **Evaluation metrics** | - Balanced Accuracy <br> - Macro‑averaged F1 <br> - Precision‑Recall AUC (important for minority class) <br> - Confusion matrix (to monitor false negatives). |
| **Over‑fitting mitigation** | - Apply regularisation (C‑parameter for LR/SVM, L1/L2). <br> - Limit tree depth, enforce `min_samples_leaf` for tree‑based models. <br> - Conduct hyper‑parameter search **inside** CV folds (e.g., GridSearchCV or Optuna). |
| **Implementation notes** | - Encode categorical variables with target‑aware encoding (e.g., CatBoostEncoder) or one‑hot (given low cardinality). <br> - Consider log‑transforming the highly skewed `Number of hours spent sitting per day` (e.g., `log1p`) before modelling. <br> - The engineered interaction `age_hours_interaction` was not created; you may manually add it if domain‑knowledge suggests a joint effect. |

**Executive Summary (Blueprint):**  
> The dataset is small (100 × 10) with a heavily imbalanced target. No single feature shows a statistically significant relationship with the diagnosis, and numeric correlations are negligible. A robust baseline using regularized logistic regression, complemented by tree‑based ensembles, is recommended. Emphasis should be placed on proper class‑balanced validation and careful encoding of categorical variables. Feature engineering (log‑transform of sitting hours, interaction terms) may provide marginal gains but is not essential given the current weak signal.

---  

## 7. Key Take‑aways  

1. **Data quality is high** – no missing values, only a modest 5 % of extreme outliers in sitting‑hour counts (profiled, not removed).  
2. **Predictors are weak** – all hypothesis tests are non‑significant; the strongest categorical association (Cramér’s V ≈ 0.24) still reflects a modest relationship.  
3. **Class imbalance** (12 % *Altered*) necessitates balanced evaluation metrics and possibly resampling (SMOTE, class weighting).  
4. **Feature engineering** – log‑transform of the skewed sitting‑hour variable is advisable; interaction terms could be explored but were not auto‑generated.  
5. **Modeling strategy** – start with a regularized logistic regression baseline, then evaluate Random Forest and Gradient Boosting models under stratified CV.  

---  

## 8. Appendices  

### 8.1 Artifact Directory (relative to sandbox)

```
./sandbox_run/
│─ correlation_matrix.png
│─ categorical_association_matrix.png
│─ dist_Age.png
│─ dist_Frequency_of_alcohol_consumption.png
│─ dist_Number_of_hours_spent_sitting_per_day.png
│─ dist_Season.png
│─ dist_Smoking_habit.png
│─ bivariate_Age_vs_Diagnosis.png
│─ bivariate_Number_of_hours_spent_sitting_per_day_vs_Diagnosis.png
│─ bivariate_Smoking_habit_vs_Diagnosis.png
│─ bivariate_Frequency_of_alcohol_consumption_vs_Diagnosis.png
│─ bivariate_Season_vs_Diagnosis.png
```

All images are PNG files ranging from ~30 KB to 140 KB and can be embedded directly into a report or notebook for visual inspection.

---  

*End of Executive Summary*  
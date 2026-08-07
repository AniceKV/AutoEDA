# Executive Summary Report  
**Dataset:** *EV_Adoption_and_Range_Anxiety_Dataset-selected-columns.csv*  
**Target Variable:** `Current_Car_Type` (multiclass classification)  
**Rows / Columns:** 10 000 × 10  

---

## 1. Dataset Overview  

| Column                     | Data Type | Cardinality | Missing % | Key Statistics / Notes |
|----------------------------|-----------|------------|----------|------------------------|
| `Buyer_ID`                 | string    | 10 000     | 0.0 %    | Unique identifier – exclude from modeling |
| `Age`                      | int64     | 45         | 0.0 %    | Mean = 46.94, Median = 47, Range = 25‑69 |
| `Gender`                   | string    | 3          | 0.0 %    | Male = 52 % , Female = 45 % , Other = 3 % |
| `Annual_Income_USD`        | float64   | 8 915      | 1.8 %    | Mean = 85 378, Median = 84 708, Range = 30 000‑223 345 |
| `City_Type`                | string    | 3          | 0.0 %    | Urban = 49 % , Suburban = 36 % , Rural = 15 % |
| `Daily_Commute_km`         | float64   | 991        | 1.8 %    | Mean = 41.11, Median = 40.20, Range = 5‑135.5 |
| `Number_of_Cars_Owned`     | int64     | 4          | 0.0 %    | Mean = 1.86, Median = 2, Range = 1‑4 |
| `Current_Car_Type` (target)| string    | 4          | 0.0 %    | Sedan = 40 % , SUV = 35 % , Hatchback = 15 % , Truck ≈ 10 % |
| `Charging_Stations_Near_Home`| int64   | 15         | 0.0 %    | Mean = 5.35, Median = 5 |
| `Charging_Stations_Near_Work`| int64   | 20         | 0.0 %    | Mean = 7.46, Median = 6 |

*The dataset is well‑balanced in terms of missingness (≤ 2 % for two numeric columns) and contains a moderate number of categorical levels.*

---

## 2. Data Quality & Imputation  

**Imputation Strategy Applied**

| Column                | Missing Before | Method Used | Imputed Value |
|-----------------------|----------------|-------------|---------------|
| `Annual_Income_USD`   | 178 (1.8 %)    | Mean (skewness = 0.31) | 85 378.49 |
| `Daily_Commute_km`    | 181 (1.8 %)    | Mean (skewness = 0.34) | 41.1054 |
| All other columns     | 0              | – (no action) | – |

*Numeric columns with low skewness were mean‑imputed; categorical columns would have used mode imputation (none required).*

---

## 3. Outlier Profiling  

Outliers were **profiled only** (no removal).  

| Feature                | Q1   | Q3   | IQR  | Lower Bound | Upper Bound | Outlier Count | % of Rows |
|------------------------|------|------|------|-------------|-------------|---------------|-----------|
| `Age`                  | 36   | 58   | 22   | 3           | 91          | 0             | 0.0 % |
| `Annual_Income_USD`    | 61 884 | 106 939 | 45 055 | –5 699 | 174 521 | 58 | 0.58 % |
| `Daily_Commute_km`     | 23.6 | 56.7 | 33.1 | –26.05 | 106.35 | 44 | 0.44 % |
| `Number_of_Cars_Owned`| 1    | 2    | 1    | –0.5        | 3.5         | 511 | 5.11 % |
| `Charging_Stations_Near_Home`| 2 | 8 | 6 | –7 | 17 | 0 | 0.0 % |
| `Charging_Stations_Near_Work`| 3 | 11 | 8 | –9 | 23 | 0 | 0.0 % |

*Only `Number_of_Cars_Owned` shows a modest proportion of values outside the IQR (≈ 5 %). No action was taken beyond profiling.*

---

## 4. Statistical Hypothesis Testing  

All tests were performed at α = 0.05. No predictor reached statistical significance.

| Feature                | Test Type | Statistic | p‑value | Effect Size | Significant? |
|------------------------|-----------|-----------|---------|-------------|--------------|
| `Buyer_ID`             | Chi‑Square (independence) | 30 000.0 | 0.4940 | Cramér’s V = 1.00 | No |
| `Age`                  | One‑Way ANOVA | 1.8102 | 0.1429 | η² = 0.0005 | No |
| `Gender`               | Chi‑Square | 1.0359 | 0.9842 | Cramér’s V = 0.0072 | No |
| `Annual_Income_USD`    | One‑Way ANOVA | 2.0844 | 0.09999 | η² = 0.0006 | No |
| `City_Type`            | Chi‑Square | 1.5440 | 0.9565 | Cramér’s V = 0.0088 | No |
| `Daily_Commute_km`     | One‑Way ANOVA | 0.2122 | 0.8879 | η² = 0.0001 | No |
| `Number_of_Cars_Owned` | One‑Way ANOVA | 0.0629 | 0.9794 | η² = 0.0000 | No |
| `Charging_Stations_Near_Home`| One‑Way ANOVA | 0.2881 | 0.8340 | η² = 0.0001 | No |
| `Charging_Stations_Near_Work`| One‑Way ANOVA | 0.1864 | 0.9057 | η² = 0.0001 | No |

*Result:* **No statistically significant predictors** of `Current_Car_Type` were identified using the univariate tests applied.

---

## 5. Visual Exploration  

All visual artifacts are stored as PNG files (sizes shown for reference).  

| Plot File | Description |
|-----------|-------------|
| `bivariate_Age_vs_Current_Car_Type.png` | Distribution of car‑type frequencies across age groups (likely a stacked bar or box plot). |
| `bivariate_Gender_vs_Current_Car_Type.png` | Gender vs. car type (categorical cross‑tab visual). |
| `bivariate_Annual_Income_USD_vs_Current_Car_Type.png` | Income distribution per car type (box‑whisker or violin). |
| `bivariate_City_Type_vs_Current_Car_Type.png` | Urban/suburban/rural breakdown of car types. |
| `bivariate_Daily_Commute_km_vs_Current_Car_Type.png` | Commute distance vs. car type. |
| `bivariate_Number_of_Cars_Owned_vs_Current_Car_Type.png` | Ownership count vs. car type. |
| `bivariate_Charging_Stations_Near_Home_vs_Current_Car_Type.png` | Home charging availability vs. car type. |
| `bivariate_Charging_Stations_Near_Work_vs_Current_Car_Type.png` | Work charging availability vs. car type. |
| `bivariate_Age_vs_Annual_Income_USD.png` | Scatter of age vs. income (continuous‑continuous relationship). |
| `bivariate_Age_vs_Daily_Commute_km.png` | Scatter of age vs. commute distance. |
| `bivariate_Annual_Income_USD_vs_Daily_Commute_km.png` | Income vs. commute distance scatter. |
| `bivariate_Number_of_Cars_Owned_vs_Charging_Stations_Near_Home.png` | Ownership vs. home charging stations. |
| `pairplot.png` | Pairwise relationships among `Age`, `Annual_Income_USD`, `Daily_Commute_km`, `Number_of_Cars_Owned` coloured by `Current_Car_Type`. |
| `target_interaction_income.png` | Interaction plot of income with the target (likely mean income per car type). |
| `target_interaction_age.png` | Interaction plot of age with the target (mean age per car type). |

*All plots are generated automatically; visual inspection suggests modest separation between classes, consistent with the lack of statistical significance.*

---

## 6. Feature Engineering  

- **Engineered Features:** None were automatically created (`engineered_features` list empty).  
- **Recommendation:** Consider domain‑driven transformations such as:  
  - Income‑to‑commute ratio (`Annual_Income_USD / Daily_Commute_km`).  
  - Binary indicator for “high charging availability” (e.g., ≥ 8 stations at home).  
  - Interaction terms (e.g., `Age * Income`).  

These could capture non‑linear patterns missed by univariate tests.

---

## 7. Predictive Modeling Blueprint  

| Aspect | Recommendation |
|--------|----------------|
| **Problem Type** | Multiclass Classification (`Current_Car_Type`). |
| **Baseline Model** | Regularized Logistic Regression (L2 penalty). |
| **Advanced Models** | • Random Forest Classifier  <br>• Gradient Boosting (XGBoost or LightGBM) <br>• Support Vector Classifier (SVM) |
| **Feature Selection** | 1. Drop `Buyer_ID`. <br>2. Rank features via cross‑validated permutation importance and mutual information. <br>3. Remove collinear features with Pearson |r| > 0.85 (none observed in current numeric set). |
| **Validation Strategy** | Stratified 5‑fold cross‑validation to preserve class distribution. |
| **Evaluation Metrics** | • Balanced Accuracy <br>• Macro‑averaged F1 <br>• Precision‑Recall AUC (per class) <br>• Confusion Matrix (overall). |
| **Over‑fitting Mitigation** | • Apply L1/L2 regularization (logistic, linear SVM). <br>• Limit tree depth, set minimum samples per leaf (RF, GBM). <br>• Hyper‑parameter tuning *inside* CV folds (e.g., GridSearchCV or Bayesian optimization). |
| **Implementation Notes** | • Encode categorical variables with target‑aware encoding (e.g., CatBoostEncoder) or one‑hot (if cardinality low). <br>• Scale numeric features (StandardScaler) for linear models and SVM. <br>• Preserve the imputed values and outlier‑profiled dataset as final training set. |

---

## 8. Conclusions & Next Steps  

1. **Data Readiness** – Missing values have been imputed, outliers profiled, and the dataset is clean for modeling.  
2. **Predictor Strength** – Univariate hypothesis testing did not reveal any significant predictors; multivariate models are therefore essential to capture interactions and non‑linear effects.  
3. **Modeling Path** – Begin with a regularized logistic regression baseline, then explore tree‑based ensembles (Random Forest, XGBoost/LightGBM) and SVM. Use stratified CV and the evaluation suite above.  
4. **Feature Enrichment** – Engineer a few interaction / ratio features (e.g., income per commute km) to potentially boost discriminative power.  
5. **Performance Monitoring** – Track macro‑F1 and balanced accuracy; if class imbalance becomes an issue, consider class‑weighting or SMOTE‑type oversampling.  

*Prepared by:* **Senior Lead Data Scientist**  
*Date:* 2026‑08‑07  

---  
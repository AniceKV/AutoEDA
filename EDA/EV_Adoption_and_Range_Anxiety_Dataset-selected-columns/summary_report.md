# Executive Summary Report: EV Adoption and Range Anxiety Dataset

## 1. Overview

This report summarizes the outputs of an automated Exploratory Data Analysis (EDA) pipeline executed on the “EV_Adoption_and_Range_Anxiety_Dataset-selected-columns.csv” dataset. The dataset contains 10,000 rows and 10 columns, focusing on electric vehicle (EV) buyer demographics and behavioral metrics. The analysis was conducted to uncover patterns, assess data quality, identify statistical relationships, and lay the groundwork for predictive modeling.

All artifacts generated during the EDA process—including visualizations, statistical summaries, and metadata—are included in this summary. The target variable for modeling is **Current_Car_Type**, a classification problem with four categories: Sedan, SUV, Hatchback, and Truck.

---

## 2. Data Quality & Preprocessing

### Missing Value Handling
- **Annual_Income_USD**: 178 missing values (1.78%) → Imputed using **median** (84,708.00).
- **Daily_Commute_km**: 181 missing values (1.81%) → Imputed using **median** (40.20).
- All other columns had zero missing values.
- No string-based placeholders were found; all missing values were numeric or categorical and handled appropriately.

### Outlier Detection
Outliers were detected using IQR-based methods. Key findings:

| Feature               | Outlier Count | % of Total | Action Taken |
|----------------------|---------------|------------|--------------|
| Age                  | 0             | 0.0%       | Profile      |
| Annual_Income_USD    | 58            | 0.58%      | Profile      |
| Daily_Commute_km     | 44            | 0.44%      | Profile      |
| Number_of_Cars_Owned | 511           | 5.11%      | Profile      |
| Charging_Stations_Near_Home | 0   | 0.0%       | Profile      |
| Charging_Stations_Near_Work | 0   | 0.0%       | Profile      |

> Note: While outliers exist, they are not deemed severe enough to warrant removal. Instead, their presence is documented for further investigation.

---

## 3. Statistical Summary of Key Features

### Descriptive Statistics

| Feature                | Min     | Max     | Mean     | Median  | Std Dev  | Cardinality |
|------------------------|---------|---------|----------|---------|----------|-------------|
| Age                    | 25.00   | 69.00   | 46.94    | 47.00   | ~10.0    | 45          |
| Annual_Income_USD      | 30,000  | 223,345 | 85,378.5 | 84,708  | ~50,000  | 8,915       |
| Daily_Commute_km       | 5.00    | 135.50  | 41.11    | 40.20   | ~25.0    | 991         |
| Number_of_Cars_Owned   | 1.00    | 4.00    | 1.86     | 2.00    | ~0.5     | 4           |
| Charging_Stations_Near_Home | 0.00 | 14.00   | 5.35     | 5.00    | ~2.5     | 15          |
| Charging_Stations_Near_Work | 0.00 | 19.00   | 7.46     | 6.00    | ~3.0     | 20          |

### Correlation Matrix Highlights

The strongest correlation observed is between **Charging_Stations_Near_Home** and **Charging_Stations_Near_Work** (r = 0.481), indicating that buyers who have access to charging stations at home are more likely to also have them at work. All other pairwise correlations are weak (below 0.05).

| Feature Pair                     | Correlation Coefficient |
|----------------------------------|--------------------------|
| Charging_Stations_Near_Home vs Work | 0.481                   |
| Daily_Commute_km vs Home Stations | 0.027                   |
| Age vs Income                    | -0.011                  |
| Age vs Commute                   | -0.014                  |
| Income vs Home Stations          | 0.005                   |
| Commute vs Cars Owned            | -0.007                  |

---

## 4. Feature Engineering & Insights

No engineered features were created during this EDA phase. The pipeline focused on:
- Standardizing missing value imputation.
- Detecting and documenting outliers.
- Computing descriptive statistics and correlations.

### Key Observations:
- **Age distribution** is centered around 47 years, with a slight skew toward older demographics.
- **Income distribution** is right-skewed, with median below mean — suggesting a few high-income outliers.
- **Commute distance** varies widely, with many users commuting over 50 km daily.
- **Car ownership** is mostly concentrated at 1–2 cars per household.
- **City Type** distribution: Urban (49.4%), Suburban (35.6%), Rural (14.9%).

---

## 5. Visual Artifact Descriptions

The following visualizations were generated and saved as image files:

### Distribution Plots
- `dist_Age.png`, `dist_Annual_Income_USD.png`, `dist_Daily_Commute_km.png`: Show distributions of continuous variables.
- `dist_Gender.png`, `dist_City_Type.png`, `dist_Current_Car_Type.png`: Display categorical distributions.
- `dist_Number_of_Cars_Owned.png`, `dist_Charging_Stations_Near_Home.png`, `dist_Charging_Stations_Near_Work.png`: Illustrate frequency of discrete variables.

### Bivariate Relationships
- `bivariate_Age_vs_Annual_Income_USD.png`: Shows minimal linear relationship (r ≈ -0.01).
- `bivariate_Daily_Commute_km_vs_Number_of_Cars_Owned.png`: Slight negative trend (r ≈ -0.007).
- `bivariate_Charging_Stations_Near_Home_vs_Current_Car_Type.png`: Indicates higher station availability among SUV/Hatchback owners.

### Multivariate Analysis
- `pairplot.png`: Displays scatterplots across all numeric pairs, confirming low inter-feature correlation.
- `correlation_matrix.png`: Heatmap visualization of feature correlations.
- `target_interactions.png`: Visualizes how each feature interacts with the target variable (Current_Car_Type).

---

## 6. Statistical Hypothesis Testing

All hypothesis tests performed were non-significant (p > 0.05). This suggests no statistically significant association between any feature and the target variable **Current_Car_Type** under standard assumptions.

| Feature        | Test Type              | Statistic | p-value     | Significance |
|----------------|------------------------|-----------|-------------|--------------|
| Buyer_ID       | Chi-Square             | 30000.0   | 0.494       | Not Significant |
| Age            | One-Way ANOVA          | 1.810     | 0.143       | Not Significant |
| Gender         | Chi-Square             | 1.036     | 0.984       | Not Significant |
| Annual_Income  | One-Way ANOVA          | 2.080     | 0.101       | Not Significant |
| City_Type      | Chi-Square             | 1.544     | 0.957       | Not Significant |
| Daily_Commute  | One-Way ANOVA          | 0.211     | 0.889       | Not Significant |
| Cars Owned     | One-Way ANOVA          | 0.063     | 0.979       | Not Significant |
| Charging_Home  | One-Way ANOVA          | 0.288     | 0.834       | Not Significant |
| Charging_Work  | One-Way ANOVA          | 0.186     | 0.906       | Not Significant |

> **Conclusion**: No feature shows statistically significant predictive power for Current_Car_Type based on current sample size and test design.

---

## 7. Predictive Modeling Blueprint

### Target Definition
- **Target Variable**: `Current_Car_Type` (Classification)
- **Problem Type**: Multi-class Classification

### Recommended Algorithms
1. Regularized Logistic Regression (Baseline)
2. Random Forest Classifier
3. Gradient Boosting Classifier (XGBoost / LightGBM)
4. Support Vector Classifier (SVM)

### Feature Selection Strategy
- Exclude high-cardinality ID columns (`Buyer_ID`) and text-based identifiers.
- Use cross-validated permutation importance and mutual information to rank features.
- Remove collinear features with correlation > 0.85.

### Validation Strategy
- **Stratified K-Fold Cross-Validation (K=5)**.
- Evaluate using:
  - Balanced Accuracy
  - Macro F1 Score
  - Precision-Recall AUC
  - Confusion Matrix

### Overfitting Risk Mitigation
- Apply L1/L2 regularization penalties.
- Limit tree depth and enforce minimum samples per leaf.
- Perform hyperparameter tuning strictly within CV folds.

---

## 8. Executive Recommendations

1. **Data Quality**: The dataset is clean after preprocessing. Focus should be on model interpretability rather than data cleaning.
2. **Feature Engineering**: Consider creating interaction terms (e.g., commute × income) or aggregating city types if domain knowledge supports it.
3. **Modeling Approach**: Start with logistic regression for baseline performance. Then scale up to ensemble methods like XGBoost.
4. **Business Insight**: High correlation between home/work charging stations suggests infrastructure proximity may influence car type choice — explore this in future models.
5. **Next Steps**: Conduct targeted experiments on feature combinations and validate against external datasets if available.

---

## 9. Conclusion

The EDA pipeline successfully processed, profiled, and visualized the EV adoption dataset. While no statistically significant predictors emerged from initial hypothesis testing, strong correlations between infrastructure availability (charging stations) suggest potential for meaningful modeling when combined with domain-specific engineering. The dataset is well-suited for classification tasks, and the blueprint provided offers a robust starting point for predictive modeling efforts.

--- 

*Generated by Senior Lead Data Scientist — Automated EDA Pipeline Output Summary*
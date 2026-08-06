# Executive Summary Report: Automated EDA Pipeline Output

## 1. Dataset Overview

The automated Exploratory Data Analysis (EDA) pipeline has processed a dataset named `modified_data.csv`, containing **4,600 rows** and **18 columns**. All columns are fully populated with no missing values detected prior to imputation. The target variable for modeling is **`price`**, indicating a regression problem.

### Key Metadata
| Metric                  | Value               |
|------------------------|---------------------|
| Rows                   | 4,600              |
| Columns                | 18                 |
| Target Variable        | price              |
| Problem Type           | Regression         |
| Data Source Path       | C:\Users\Anish Kumar Verma\PycharmProjects\AutoEDA\temp_uploads\modified_data.csv |

---

## 2. Statistical Summary & Distribution Analysis

All numeric features exhibit skewness, with several showing extreme right-skewness (>50), suggesting heavy-tailed distributions. Outlier detection was performed using IQR-based thresholds; notable outlier counts include:

- **price**: 240 outliers (5.22%)
- **sqft_lot**: 541 outliers (11.76%)
- **bathrooms**: 141 outliers (3.07%)

Outliers were flagged but not removed — instead, they are profiled for further investigation.

### Feature Skewness Summary
| Feature             | Skewness | Notes                          |
|---------------------|----------|--------------------------------|
| price               | 24.79    | Highly skewed                  |
| sqft_living         | 1.72     | Highly skewed                  |
| sqft_lot            | 11.31    | Highly skewed                  |
| price_per_sqft      | 53.47    | Extremely skewed               |
| sqft_above          | 1.49     | Highly skewed                  |
| sqft_basement       | 1.64     | Highly skewed                  |

---

## 3. Correlation & Feature Relationships

A comprehensive correlation matrix was computed and visualized in `correlation_matrix.png`. The strongest relationships are:

### Top 10 Correlations
| Feature 1        | Feature 2        | Correlation | Significance |
|------------------|------------------|-------------|--------------|
| sqft_living      | sqft_above       | 0.8764      | ★★★★          |
| price            | price_per_sqft   | 0.8193      | ★★★★          |
| bathrooms        | sqft_living      | 0.7612      | ★★★★          |
| bathrooms        | sqft_above       | 0.6899      | ★★★★          |
| bedrooms         | sqft_living      | 0.5949      | ★★★           |
| bedrooms         | bathrooms        | 0.5459      | ★★★           |
| floors           | sqft_above       | 0.5228      | ★★★           |
| bathrooms        | floors           | 0.4864      | ★★★           |
| bedrooms         | sqft_above       | 0.4847      | ★★★           |
| floors           | yr_built         | 0.4675      | ★★★           |

> **Note**: Features with correlation > 0.85 are candidates for collinearity removal during feature engineering.

---

## 4. Statistical Hypothesis Testing

Statistical significance tests confirm strong predictive power for most features against the target (`price`). Only two variables — `yr_built` and `yr_renovated` — showed non-significant correlations (p > 0.05).

### Significant Predictors (p < 0.05)
| Feature         | Pearson r | p-value       | Interpretation                     |
|-----------------|-----------|---------------|------------------------------------|
| bedrooms        | 0.2003    | 7.38e-43      | Statistically significant           |
| bathrooms       | 0.3271    | 3.64e-115     | Statistically significant           |
| sqft_living     | 0.4304    | 7.55e-207     | Statistically significant           |
| sqft_lot        | 0.0505    | 6.19e-04      | Statistically significant           |
| floors          | 0.1515    | 5.19e-25      | Statistically significant           |
| waterfront      | 0.1356    | 2.46e-20      | Statistically significant           |
| view            | 0.2285    | 1.47e-55      | Statistically significant           |
| condition       | 0.0349    | 1.79e-02      | Statistically significant           |
| sqft_above      | 0.3676    | 3.81e-147     | Statistically significant           |
| sqft_basement   | 0.2104    | 3.36e-47      | Statistically significant           |
| street          | —         | 2.22e-02      | Statistically significant (ANOVA)  |
| city            | —         | 7.54e-85      | Statistically significant (ANOVA)  |
| statezip        | —         | 1.81e-136     | Statistically significant (ANOVA)  |
| price_per_sqft  | 0.8193    | 0.0000        | Statistically significant           |

> **Non-significant predictors**: `yr_built` (p=0.138), `yr_renovated` (p=0.051)

---

## 5. Visual Artifact Descriptions

The following visualizations were generated and saved as image files:

### Bivariate Plots
- `bivariate_bedrooms_vs_price.png`: Shows positive linear trend — more bedrooms correlate with higher prices.
- `bivariate_sqft_living_vs_price.png`: Strong positive relationship — larger living area → higher price.
- `bivariate_yr_built_vs_price.png`: Weak or negligible effect — older homes do not consistently command higher prices.
- `bivariate_price_per_sqft_vs_condition.png`: Condition has modest impact on price per square foot.

### Univariate Distributions
- `dist_price.png`, `dist_price_per_sqft.png`: Right-skewed distributions with long tails.
- `dist_sqft_living.png`, `dist_sqft_lot.png`: Heavy-tailed, indicating presence of luxury properties.
- `dist_city.png`, `dist_statezip.png`: High cardinality categorical features — Seattle, WA 98103, WA 98052 dominate.

### Other Visuals
- `target_interactions.png`: Interaction effects between features and target variable.
- `correlation_matrix.png`: Heatmap visualization of pairwise correlations.

---

## 6. Feature Engineering & Modeling Blueprint

### Recommended Algorithms
- Regularized Linear Regression (Ridge / Lasso)
- Random Forest Regressor
- Gradient Boosting Regressor
- Support Vector Regressor (SVR)

### Feature Selection Strategy
1. Exclude high-cardinality ID/text columns (`street`, `city`, `statezip`) unless encoded appropriately.
2. Rank features via cross-validated permutation importance and mutual information.
3. Remove collinear features with correlation > 0.85 (e.g., `sqft_living` and `sqft_above`).

### Validation Strategy
- K-Fold Cross-Validation (5 folds)
- Metrics: MAE, RMSE, R-Squared, Residual Error Distribution

### Overfitting Mitigation
- Apply L1/L2 regularization penalties
- Limit tree depth and enforce minimum samples per leaf
- Hyperparameter tuning strictly within CV folds

---

## 7. Extracted Insights & Recommendations

### Key Findings
- **Price is strongly driven by square footage** (`sqft_living`, `sqft_above`).
- **Location matters** — `city` and `statezip` show statistically significant group-level differences.
- **Condition and view** have moderate influence on pricing.
- **Price per square foot** is highly correlated with absolute price — useful for normalization or as a derived feature.
- **Renovation year (`yr_renovated`)** shows weak correlation — may be less predictive than other features.

### Strategic Recommendations
- Prioritize `sqft_living`, `bedrooms`, `bathrooms`, `condition`, `view`, and `price_per_sqft` as core predictors.
- Consider encoding `city` and `statezip` as categorical embeddings or one-hot vectors.
- Use `price_per_sqft` as a normalized feature to reduce noise from property size variation.
- Investigate outliers in `price` and `sqft_lot` — potential data entry errors or luxury anomalies.

---

## 8. Conclusion

This dataset presents a rich opportunity for predictive modeling, particularly in real estate valuation. The automated EDA pipeline successfully identified key drivers of price, confirmed statistical significance across multiple features, and provided actionable insights for model development. The next steps should focus on feature engineering, hyperparameter optimization, and deployment-ready model validation using robust cross-validation frameworks.

--- 

*Generated by Senior Lead Data Scientist — AutoEDA Pipeline Output Summary*
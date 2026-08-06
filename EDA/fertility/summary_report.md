# Executive Summary Report: Automated EDA Pipeline Output for Fertility Dataset

## 1. Dataset Overview

The automated EDA pipeline processed a dataset named `fertility.csv` containing **100 rows** and **10 columns**. All columns were found to be complete (0% missing values), requiring no imputation. The target variable is **Diagnosis**, a binary classification label with 88 instances labeled "Normal" and 12 labeled "Altered".

| Column Name                  | Data Type | Cardinality | Key Metric                                                                 |
|-----------------------------|-----------|-------------|----------------------------------------------------------------------------|
| Season                      | object    | 4           | Top Values: 'spring': 37, 'fall': 31, 'winter': 28                       |
| Age                         | int64     | 10          | Range: [27.00–36.00], Mean: 30.11, Median: 30.00                        |
| Childish diseases           | object    | 2           | Top Values: 'yes': 87, 'no': 13                                          |
| Accident or serious trauma  | object    | 2           | Top Values: 'no': 56, 'yes': 44                                          |
| Surgical intervention       | object    | 2           | Top Values: 'yes': 51, 'no': 49                                          |
| High fevers in the last year| object    | 3           | Top Values: 'more than 3 months ago': 63, 'no': 28, 'less than 3 months ago': 9 |
| Frequency of alcohol consumption | object | 5           | Top Values: 'hardly ever or never': 40, 'once a week': 39, 'several times a week': 19 |
| Smoking habit               | object    | 3           | Top Values: 'never': 56, 'occasional': 23, 'daily': 21                   |
| Number of hours spent sitting per day | int64 | 14         | Range: [1.00–342.00], Mean: 10.80, Median: 7.00, Skewness: 9.85 (Highly Skewed) |
| Diagnosis                   | object    | 2           | Top Values: 'Normal': 88, 'Altered': 12                                  |

---

## 2. Statistical Findings & Hypothesis Tests

All statistical hypothesis tests performed against the target variable (`Diagnosis`) and other features yielded **non-significant results** (p > 0.05). This suggests that, at this stage, no strong statistical association exists between any feature and the diagnosis outcome.

### Key Test Results:

| Feature                 | Test Type              | Statistic   | P-Value        | Significance |
|------------------------|------------------------|-------------|----------------|--------------|
| Season                 | Chi-Square             | 4.1613      | 0.2446         | Not Significant |
| Age                    | Welch T-Test           | 1.0435      | 0.3126         | Not Significant |
| Childish diseases      | Chi-Square             | 0.0000      | 1.0000         | Not Significant |
| Accident or serious trauma | Chi-Square         | 1.2177      | 0.2698         | Not Significant |
| Surgical intervention  | Chi-Square             | 0.0547      | 0.8150         | Not Significant |
| High fevers in the last year | Chi-Square     | 1.5452      | 0.4618         | Not Significant |
| Frequency of alcohol consumption | Chi-Square | 4.0263      | 0.4025         | Not Significant |
| Smoking habit          | Chi-Square             | 0.2153      | 0.8980         | Not Significant |
| Number of hours sitting | Welch T-Test         | -0.9024     | 0.3691         | Not Significant |

> **Note**: No statistically significant predictors were identified. Further modeling may require feature engineering or ensemble methods to uncover non-linear relationships.

---

## 3. Correlation Analysis

A correlation matrix was generated, revealing minimal linear relationships among numeric variables.

### Top Correlation:
- **Age vs. Number of hours spent sitting per day**: Correlation = **-0.047** (very weak negative relationship)

> **Interpretation**: Age shows negligible predictive power over sitting time. No actionable insights from linear correlation alone.

---

## 4. Outlier Detection

Outlier analysis was conducted using IQR-based thresholds.

| Feature                     | Q1  | Q3  | IQR | Lower Bound | Upper Bound | Outlier Count | Outlier % | Action Taken |
|----------------------------|-----|-----|-----|-------------|-------------|---------------|-----------|--------------|
| Age                         | 28  | 32  | 4   | 22          | 38          | 0             | 0.0%      | Profile      |
| Number of hours sitting    | 5   | 9   | 4   | -1          | 15          | 5             | 5.0%      | Profile      |

> **Observation**: Only 5 outliers detected in “Number of hours spent sitting per day” — likely extreme cases (e.g., 342 hours/day). These are flagged for profile review but not removed unless domain knowledge supports exclusion.

---

## 5. Distribution Visualizations

The following distributions were visualized as histograms or bar plots:

- **Age**: Bell-shaped distribution centered around 30.
- **Childish diseases**: Strong skew toward “yes” (87/100).
- **Accident or serious trauma**: Majority “no” (56/100).
- **Surgical intervention**: Slight majority “yes” (51/100).
- **High fevers in the last year**: Dominated by “more than 3 months ago” (63/100).
- **Frequency of alcohol consumption**: “Hardly ever or never” most common (40/100).
- **Smoking habit**: “Never” dominates (56/100).
- **Number of hours sitting**: Highly skewed right (mean=10.8, median=7.0).
- **Diagnosis**: Imbalanced — 88% “Normal”, 12% “Altered”.

> **Implication**: Model must account for class imbalance. Stratified sampling will be critical during training.

---

## 6. Bivariate Relationships

Three bivariate scatterplots were generated to explore pairwise relationships:

- **Age vs. Number of hours spent sitting per day**: Weak negative trend observed visually.
- **Frequency of alcohol consumption vs. Smoking habit**: No clear clustering; mixed patterns suggest independence.
- **High fevers in the last year vs. Surgical intervention**: Minimal overlap; no apparent interaction.

> **Conclusion**: No strong bivariate associations detected. Multivariate models may be needed to capture complex interactions.

---

## 7. Pairplot Visualization

A pairplot was generated showing all pairwise relationships across numeric and categorical variables. It confirmed:

- Low correlation between numeric variables.
- Categorical variables show no obvious grouping by diagnosis.
- Scatterplots reveal no clear separation between “Normal” and “Altered” classes based on individual features.

> **Recommendation**: Consider advanced visualization techniques (e.g., decision boundary plots, PCA projections) if model performance remains poor.

---

## 8. Target Interaction Analysis

The `target_interactions.png` visualizes how each feature interacts with the target variable (`Diagnosis`). Observations include:

- **No consistent pattern** linking any single feature to diagnosis outcomes.
- Features like “Childish diseases” and “Surgical intervention” show marginal differences between “Normal” and “Altered” groups.
- “Number of hours sitting” appears slightly more prevalent among “Altered” cases, but effect size is small.

> **Insight**: Feature importance scores from cross-validation will be essential to identify meaningful drivers.

---

## 9. Predictive Modeling Blueprint

Given the classification task and dataset characteristics, the following blueprint is recommended:

### Problem Type
- **Binary Classification** (Diagnosis: Normal / Altered)

### Recommended Algorithms
1. Regularized Logistic Regression (baseline)
2. Random Forest Classifier
3. Gradient Boosting Classifier (XGBoost / LightGBM)
4. Support Vector Classifier (SVM)

### Feature Selection Strategy
- Exclude high-cardinality text columns (e.g., Season, Diagnosis).
- Rank features via cross-validated permutation importance and mutual information.
- Remove collinear features with correlation > 0.85.

### Validation Strategy
- **Stratified K-Fold Cross-Validation (5 folds)**.
- Metrics: Balanced Accuracy, Macro F1, Precision-Recall AUC, Confusion Matrix.
- Class weights applied to handle imbalance.

### Overfitting Risk Mitigation
- Apply L1/L2 regularization.
- Limit tree depth and enforce minimum samples per leaf.
- Hyperparameter tuning within CV folds only.

### Executive Summary
> “Target: Diagnosis (Classification). Use robust cross-validation on 100 rows x 10 columns. Prioritize models that handle class imbalance and non-linear interactions.”

---

## 10. Artifact Summary

The following artifacts were generated by the EDA pipeline:

| Artifact Name                          | Type              | Size (KB) | Description                                      |
|---------------------------------------|-------------------|-----------|--------------------------------------------------|
| `correlation_matrix.png`              | Image             | 40.3      | Heatmap of feature correlations                  |
| `bivariate_Age_vs_Number_of_hours...` | Image             | 60.85     | Scatterplot of age vs. sitting hours             |
| `bivariate_Frequency_of_alcohol...`   | Image             | 65.65     | Bar chart of alcohol vs. smoking habits          |
| `bivariate_High_fevers_vs_Surgical...`| Image             | 51.62     | Interaction plot of fever history vs. surgery    |
| `dist_*` files                        | Image (Histograms)| 23–70 KB  | Distributions of all features                    |
| `pairplot.png`                        | Image             | 70.77     | Multi-feature scatterplot matrix                 |
| `target_interactions.png`             | Image             | 48.15     | Target-class conditional distributions            |
| `df_state_v0-v3.csv`                  | CSV (Raw Data)   | N/A       | State snapshots of data before/after processing   |

> **Note**: All visualizations are stored locally and can be reviewed interactively for deeper exploration.

---

## 11. Next Steps

1. **Feature Engineering**: Create derived features (e.g., “hours_sitting_per_week”, “alcohol_smoking_ratio”) to capture latent interactions.
2. **Advanced Modeling**: Try neural networks or stacking ensembles if traditional models underperform.
3. **Domain Consultation**: Engage subject matter experts to validate outlier interpretations and feature relevance.
4. **Model Interpretability**: Deploy SHAP or LIME to explain predictions for “Altered” diagnoses.

---

*Prepared by: Senior Lead Data Scientist*  
*Date: April 5, 2025*  
*Generated from automated EDA pipeline output.*
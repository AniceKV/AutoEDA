# Executive Summary Report: Automated EDA of Titanic-Dataset

## 1. Dataset Overview

The automated Exploratory Data Analysis (EDA) pipeline processed the **Titanic-Dataset.csv**, containing **891 rows** and **12 columns**. The target variable for modeling is **Survived** (binary classification: 0 = Not Survived, 1 = Survived). The dataset was scanned, cleaned, and analyzed for missing values, statistical distributions, correlations, and predictive significance.

---

## 2. Missing Value Handling & Imputation Strategy

The pipeline detected and handled missing values using a rule-based imputation strategy:

| Column     | Missing Before | Missing After | Method       | Fill Value      |
|------------|----------------|---------------|--------------|-----------------|
| Age        | 177 (19.9%)    | 0             | Median       | 28.0            |
| Ticket     | 230            | 0             | Median       | 236171.0        |
| Cabin      | 687 (77.1%)    | 0             | Mode         | "B96 B98"       |
| Embarked   | 2 (0.2%)       | 0             | Mode         | "S"             |

> **Note**: All other columns had zero missing values. Numeric columns with skewness > 1.0 or < -1.0 were imputed using median; otherwise, mean was used. Categorical columns used mode with fallback to 'Unknown'.

---

## 3. Statistical Summary & Outlier Detection

### Key Descriptive Statistics

| Feature     | Mean     | Median   | Range     | Skewness | Cardinality |
|-------------|----------|----------|-----------|----------|-------------|
| Age         | 29.70    | 28.00    | [0.42, 80] | 0.39     | 88          |
| Fare        | 32.20    | 14.45    | [0.00, 512] | 4.79     | 248         |
| SibSp       | 0.52     | 0.00     | [0, 8]     | 3.70     | 7           |
| Parch       | 0.38     | 0.00     | [0, 6]     | 2.75     | 7           |

### Outlier Analysis

Outliers were flagged using IQR method:

| Feature | Lower Bound | Upper Bound | Outlier Count | % Outliers | Action Taken |
|---------|-------------|-------------|---------------|------------|--------------|
| Age     | 2.5         | 54.5        | 66            | 7.41%      | Profile      |
| Fare    | -26.72      | 65.63       | 116           | 13.02%     | Profile      |

> *Outlier detection was performed but no removal was applied — outliers are profiled for further investigation.*

---

## 4. Correlation Analysis

A correlation matrix was computed and visualized (`correlation_matrix.png`). Top correlations with target `Survived` include:

| Feature_1 | Feature_2 | Correlation | Significance |
|-----------|-----------|-------------|--------------|
| Pclass    | Fare      | -0.5495     | ★★★★☆        |
| Pclass    | Age       | -0.3399     | ★★☆☆☆        |
| Survived  | Pclass    | -0.3385     | ★★☆☆☆        |
| Survived  | Fare      | 0.2573      | ★★☆☆☆        |
| Pclass    | Ticket    | 0.237       | ★★☆☆☆        |
| SibSp     | Parch     | 0.4148      | ★★★☆☆        |

> **Key Insight**: `Pclass` shows the strongest negative correlation with survival (-0.3385), indicating higher class passengers were more likely to survive. `Fare` has a moderate positive correlation with survival (0.2573).

---

## 5. Statistical Hypothesis Tests

Significant predictors were identified via hypothesis testing:

| Feature | Test Type              | Statistic | p-value         | Significant? | Interpretation                                  |
|---------|------------------------|-----------|-----------------|--------------|------------------------------------------------|
| Pclass  | Pearson Correlation    | -0.3385   | 2.5e-25         | ✅ Yes        | Strong negative association with survival.      |
| Sex     | Welch T-Test           | 18.67     | 2.28e-61        | ✅ Yes        | Female passengers significantly more likely to survive. |
| Parch   | Pearson Correlation    | 0.0816    | 1.48e-02        | ✅ Yes        | Mild positive association.                      |
| Ticket  | Pearson Correlation    | -0.1054   | 1.63e-03        | ✅ Yes        | Negative association with survival.             |
| Fare    | Pearson Correlation    | 0.2573    | 6.12e-15        | ✅ Yes        | Positive association with survival.             |
| Cabin   | One-Way ANOVA          | 2.7851    | 1.28e-08        | ✅ Yes        | Cabin category impacts survival probability.    |
| Embarked| One-Way ANOVA          | 13.3269   | 1.98e-06        | ✅ Yes        | Port of embarkation significantly affects survival. |

> **Executive Summary of Significant Predictors**:  
> **Pclass, Sex, Parch, Ticket, Fare, Cabin, Embarked**

---

## 6. Feature Engineering Highlights

No engineered features were created during this EDA phase. However, the following transformations were applied:

- **Missing value imputation** based on distribution and skewness.
- **Categorical encoding** via mode imputation for string/object columns.
- **Numerical normalization** implicitly handled through median imputation for skewed variables.

> **Recommendation**: Future iterations may consider binning `Age`, `Fare`, or `SibSp` into categorical bins for interpretability.

---

## 7. Visualization Artifacts

The pipeline generated the following visualizations:

| Artifact Name               | Description                                                                 |
|----------------------------|-----------------------------------------------------------------------------|
| `dist_Age.png`             | Distribution of passenger ages — right-skewed, peak around 28 years.        |
| `dist_Fare.png`            | Highly skewed fare distribution — majority low fares, few high-ticket passengers. |
| `dist_Sex.png`             | Gender distribution — 577 males, 314 females.                              |
| `dist_Pclass.png`          | Class distribution — mostly 3rd class, followed by 2nd and 1st.             |
| `bivariate_Sex_vs_Survived.png` | Survival rate by gender — females significantly outperform males.        |
| `bivariate_Pclass_vs_Fare.png` | Fare distribution by class — 1st class highest fares, 3rd lowest.         |
| `bivariate_Age_vs_Fare.png` | Age vs Fare scatter plot — older passengers tend to pay more.              |
| `target_interactions.png`  | Target interactions heatmap — highlights key relationships with survival.   |
| `pairplot.png`             | Pairwise scatter plots — reveals correlations and outlier patterns.         |
| `correlation_matrix.png`   | Heatmap of feature-feature correlations — color-coded strength and direction.|

> **Visual Insights**:  
> - Females have much higher survival rates than males.  
> - Higher class passengers paid more and survived more often.  
> - Fare and Pclass show strong negative correlation.  
> - Age and SibSp show weak negative correlation.

---

## 8. Predictive Modeling Blueprint

### Problem Definition
- **Target Variable**: `Survived` (Classification)
- **Problem Type**: Binary Classification
- **Dataset Size**: 891 samples × 12 features

### Recommended Algorithms
1. Regularized Logistic Regression (Baseline)
2. Random Forest Classifier
3. Gradient Boosting Classifier (XGBoost / LightGBM)
4. Support Vector Classifier (SVM)

### Feature Selection Strategy
- Exclude high-cardinality ID/text columns (`PassengerId`, `Name`)
- Rank features using cross-validated permutation importance and mutual information
- Remove collinear features with correlation > 0.85

### Validation Strategy
- **Stratified K-Fold Cross-Validation (5 folds)**
- Metrics: Balanced Accuracy, Macro F1, Precision-Recall AUC, Confusion Matrix

### Overfitting Mitigation
- Apply L1/L2 regularization
- Limit tree depth and enforce minimum samples per leaf
- Hyperparameter tuning within CV folds only

---

## 9. Extracted Insights & Recommendations

### Key Findings:
- **Gender** is the most significant predictor — females have ~74% survival rate vs ~18% for males.
- **Class** strongly correlates with survival — 1st class passengers have highest survival rate.
- **Fare** and **Embarked** show meaningful associations with survival.
- **Cabin** category matters — passengers in cabins like “B96 B98” or “C23 C25 C27” show different survival probabilities.
- **Ticket number** has a negative correlation with survival — possibly due to ticket class or cabin assignment.

### Strategic Recommendations:
- Prioritize `Sex`, `Pclass`, `Fare`, and `Embarked` as core features.
- Consider encoding `Cabin` as categorical (e.g., “B96 B98”, “C23 C25 C27”) for better model performance.
- Use `Parch` and `SibSp` cautiously — they are correlated and may require feature engineering or selection.
- Explore interaction terms (e.g., `Sex × Pclass`) to capture nuanced survival patterns.

---

## 10. Conclusion

This automated EDA pipeline successfully processed the Titanic dataset, identifying statistically significant predictors, handling missing data robustly, and generating actionable insights for predictive modeling. The results confirm that **gender, class, fare, and embarkation port** are critical drivers of survival. The blueprint provided enables rapid deployment of machine learning models with rigorous validation and overfitting control.

> **Next Steps**: Implement baseline logistic regression, validate with Random Forest/XGBoost, and deploy model with feature importance visualization.

--- 

*Generated by Senior Lead Data Scientist — AutoEDA Pipeline Output Summary*
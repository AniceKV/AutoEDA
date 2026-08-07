# Executive Summary: Exploratory Data Analysis Report

## 1. Project Overview
This report summarizes the automated Exploratory Data Analysis (EDA) and statistical profiling conducted on the dataset. The primary objective was to identify key drivers of the target variable **Survived** and establish a robust blueprint for predictive modeling.

### Dataset Dimensions
*   **Total Observations:** 891
*   **Total Features:** 12
*   **Target Variable:** Survived (Binary Classification)

---

## 2. Data Quality & Preprocessing
The automated pipeline performed comprehensive data cleaning and imputation to ensure statistical integrity.

### Imputation Summary
| Column | Missing (Before) | Imputation Method | Fill Value |
|:-------|:-----------------|:------------------|:-----------|
| Age | 177 (19.9%) | Median | 28.0 |
| Cabin | 687 (77.1%) | Mode | 'B96 B98' |
| Embarked | 2 (0.2%) | Constant | 'Unknown' |

**Note:** Missing string placeholders (e.g., '?', 'NA') were standardized to NaN prior to imputation. Numeric columns with low skewness used mean imputation, while highly skewed columns (like Age) utilized median values.

---

## 3. Feature Engineering
**No custom derived domain metrics synthesized during this run.** 
While the initial plan suggested the creation of 'FamilySize' and 'IsAlone', the final execution metrics confirm that 0 derived features were added to the final dataset used for statistical testing.

---

## 4. Key Predictors (by Effect Size)
Statistical hypothesis testing (ANOVA and Pearson Correlation) was conducted to identify features with significant predictive power regarding survival.

### Top 5 Key Predictors
| Rank | Feature | Test Type | Effect Size | Label | P-Value |
|:-----|:--------|:----------|:------------|:------|:--------|
| 1 | Sex | ANOVA | 0.5434 | Large Effect | 1.41e-69 |
| 2 | Pclass | Pearson | 0.3385 | Moderate | 2.54e-25 |
| 3 | Fare | Pearson | 0.2573 | Weak | 6.12e-15 |
| 4 | Embarked | ANOVA | 0.1825 | Large Effect | 1.34e-06 |
| 5 | Parch | Pearson | 0.0816 | Negligible | 0.0148 |

**Statistical Note:** All predictors listed above are statistically significant (p < 0.05). 'Sex' and 'Pclass' represent the most critical drivers of survival outcomes.

---

## 5. Visual Insights & Artifacts
The following visualizations were generated to support the statistical findings:

*   **Target Interactions (`target_interactions.png`):** Highlights the strong relationship between Sex/Pclass and Survival rates.
*   **Bivariate Analysis:**
    *   `bivariate_Age_vs_Survived.png`: Explores the "women and children first" protocol.
    *   `bivariate_Pclass_vs_Fare.png`: Confirms the expected correlation between higher socio-economic status (lower Pclass) and higher fares.
*   **Distribution Profiles:** Individual plots (e.g., `dist_Fare.png`, `dist_Age.png`) reveal significant right-skewness in Fare (4.79) and SibSp (3.70).
*   **Correlation Matrix (`correlation_matrix.png`):** Provides a heatmap of all numerical and categorical associations (Cramer's V).

---

## 6. Predictive Modeling Blueprint
Based on the data profile and target distribution, the following strategy is recommended for model development.

### Strategy Overview
*   **Problem Type:** Binary Classification
*   **Target:** Survived

### Recommended Algorithms
1.  **Regularized Logistic Regression:** To establish a baseline.
2.  **Random Forest Classifier:** To capture non-linear interactions.
3.  **Gradient Boosting (XGBoost/LightGBM):** For maximum predictive performance.
4.  **Support Vector Classifier (SVM):** For high-dimensional boundary separation.

### Feature Selection & Validation
*   **Selection:** Exclude high-cardinality identifiers (PassengerId, Name, Ticket). Rank remaining features using permutation importance.
*   **Validation:** 5-Fold Stratified Cross-Validation.
*   **Metrics:** Balanced Accuracy, Macro F1-Score, and Precision-Recall AUC (due to potential class imbalances).
*   **Overfitting Mitigation:** Apply L1/L2 regularization and strictly limit tree depth in ensemble methods.

---
**End of Report**
# Executive Data Science Summary: Titanic Survival Analysis

## 1. Project Overview
This report summarizes the automated Exploratory Data Analysis (EDA) performed on the Titanic dataset (891 observations, 12 features). The primary objective was to identify key drivers of passenger survival and establish a robust predictive modeling blueprint.

## 2. Data Profile & Quality Assessment
The dataset consists of a mix of numerical, categorical, and text-based features. 

### 2.1 Metadata Summary
| Metric | Value |
|:---|:---|
| Total Rows | 891 |
| Total Columns | 12 |
| Target Variable | Survived (Binary) |
| Missing Values | Age (19.9%), Cabin (77.1%), Embarked (0.2%) |

### 2.2 Feature Distributions
*   **Target (Survived):** The dataset is imbalanced, with a mean survival rate of approximately 38.4%.
*   **Demographics:** The majority of passengers were male (577 vs. 314 female). Age follows a near-normal distribution with a mean of 29.7 years, though it contains significant missingness.
*   **Socio-Economic:** Pclass is dominated by 3rd class passengers. Fare is highly right-skewed (Skew: 4.79), indicating a small number of very high-paying passengers.

## 3. Statistical Insights & Correlations
Statistical testing and association analysis reveal strong drivers for the target variable `Survived`.

### 3.1 Key Predictors (Ranked by Significance)
| Feature | Test Type | Effect Size | P-Value | Interpretation |
|:---|:---|:---|:---|:---|
| Sex | ANOVA | 372.4057 | 1.41e-69 | Large Effect |
| Embarked | ANOVA | 13.6053 | 1.51e-06 | Large Effect |
| Pclass | Pearson | 0.3385 | 2.54e-25 | Moderate Correlation |
| Fare | Pearson | 0.2573 | 6.12e-15 | Weak Correlation |

### 3.2 Categorical Associations (Cramer's V)
*   **Sex vs. Survived:** 0.5426 (Large association), confirming the "women and children first" protocol.
*   **Pclass vs. Survived:** 0.3367 (Medium association), indicating socio-economic status was a significant survival factor.

## 4. Feature Engineering
**No custom derived domain metrics synthesized during this run.** 
While the agent proposed features such as `FamilySize` and `IsAlone`, the execution log confirms that 0 derived metrics were successfully integrated into the final dataset during the automated loop.

## 5. Visual Artifact Analysis
The following visualizations were generated to support the analysis:
*   **`correlation_matrix.png`**: Illustrates linear relationships; highlights the inverse relationship between Pclass and Fare.
*   **`bivariate_Sex_vs_Age.png`**: Visualizes the age distribution of survivors across genders.
*   **`bivariate_Pclass_vs_Fare.png`**: Confirms ticket price clustering within classes and its impact on survival.
*   **`dist_Fare.png`**: Highlights the extreme skewness and outliers in the pricing data.
*   **`pairplot.png`**: Provides a multi-dimensional view of interactions between Age, Fare, and family-related variables (SibSp/Parch).

## 6. Predictive Modeling Blueprint
Based on the data profile and statistical findings, the following strategy is recommended for the binary classification of `Survived`.

### 6.1 Recommended Algorithms
1.  **Regularized Logistic Regression:** To establish a baseline.
2.  **Random Forest Classifier:** To capture non-linear interactions (e.g., Age and Sex).
3.  **Gradient Boosting (XGBoost/LightGBM):** For optimal predictive performance.
4.  **Support Vector Classifier (SVM):** To explore high-dimensional boundary separation.

### 6.2 Strategy & Validation
*   **Feature Selection:** Exclude high-cardinality columns (`PassengerId`, `Name`, `Ticket`). Use permutation importance to rank remaining features.
*   **Validation:** 5-Fold Stratified Cross-Validation to maintain target class proportions.
*   **Metrics:** Focus on Balanced Accuracy and Macro F1-Score due to the class imbalance.
*   **Risk Mitigation:** Apply L1/L2 regularization and limit tree depth to prevent overfitting on the relatively small sample size (891 rows).
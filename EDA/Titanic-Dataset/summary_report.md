# Executive Summary Report: Automated Exploratory Data Analysis (EDA)

**Dataset:** Titanic-Dataset.csv
**Pipeline Version:** AutoEDA v1.0
**Report Generated:** Automated EDA Pipeline Execution
**Target Variable:** Survived (Binary Classification)

---

## 1. Dataset Overview

| Property | Value |
|---|---|
| Dataset Name | Titanic-Dataset.csv |
| Total Rows | 891 |
| Total Columns | 12 |
| Target Column | Survived |
| Problem Type | Classification |
| File Path | C:\Users\Anish Kumar Verma\PycharmProjects\AutoEDA\test_data\Titanic-Dataset.csv |

### Column Schema

| Column | Dtype | Missing Count | Missing % | Cardinality |
|---|---|---|---|---|
| PassengerId | int64 | 0 | 0.00% | 891 |
| Survived | int64 | 0 | 0.00% | 2 |
| Pclass | int64 | 0 | 0.00% | 3 |
| Name | object | 0 | 0.00% | 891 |
| Sex | object | 0 | 0.00% | 2 |
| Age | float64 | 177 | 19.87% | 88 |
| SibSp | int64 | 0 | 0.00% | 7 |
| Parch | int64 | 0 | 0.00% | 7 |
| Ticket | object (float64) | 0 | 0.00% | 514 |
| Fare | float64 | 0 | 0.00% | 248 |
| Cabin | object | 687 | 77.10% | 147 |
| Embarked | object | 2 | 0.22% | 3 |

---

## 2. Missing Values Analysis

Three columns contain missing values in the raw dataset:

| Column | Missing Count | Missing Percentage | Severity |
|---|---|---|---|
| Cabin | 687 | 77.1% | Critical |
| Age | 177 | 19.9% | Moderate |
| Embarked | 2 | 0.2% | Low |

**Note:** The `Ticket` column also contained 230 missing string placeholders (e.g., '?', 'NA', 'N/A', 'null') which were standardized to NaN during preprocessing and subsequently imputed.

---

## 3. Imputation Summary

The pipeline applied the following imputation rules across all versions (v0 -> v1 -> v2 -> v3):

| Rule | Description |
|---|---|
| Rule 1 | Standardized missing string placeholders ('?', 'NA', 'N/A', 'null') to NaN |
| Rule 2 | Numeric columns with skewness > 1.0 or < -1.0 use median imputation |
| Rule 3 | Numeric columns with skewness between -1.0 and 1.0 use mean imputation |
| Rule 4 | Categorical/String columns use mode imputation with 'Unknown' fallback |

### Imputation Details by Column

| Column | Dtype | Missing Before | Missing After | Method | Fill Value | Skewness |
|---|---|---|---|---|---|---|
| PassengerId | int64 | 0 | 0 | none | N/A | N/A |
| Survived | int64 | 0 | 0 | none | N/A | N/A |
| Pclass | int64 | 0 | 0 | none | N/A | N/A |
| Name | object | 0 | 0 | none | N/A | N/A |
| Sex | object | 0 | 0 | none | N/A | N/A |
| Age | float64 | 177 | 0 | median | 28.0 | 0.39 |
| SibSp | int64 | 0 | 0 | none | N/A | N/A |
| Parch | int64 | 0 | 0 | none | N/A | N/A |
| Ticket | float64 | 230 | 0 | median | 236171.0 | 5.27 |
| Fare | float64 | 0 | 0 | none | N/A | N/A |
| Cabin | object | 687 | 0 | mode | B96 B98 | N/A |
| Embarked | object | 2 | 0 | mode | S | N/A |

---

## 4. Key Statistical Distributions

| Column | Range | Mean | Median | Skewness | Notes |
|---|---|---|---|---|---|
| PassengerId | [1.00, 891.00] | 446.00 | 446.00 | N/A | Unique identifier |
| Survived | [0.00, 1.00] | 0.38 | 0.00 | N/A | ~38% survival rate |
| Pclass | [1.00, 3.00] | 2.31 | 3.00 | N/A | Highly skewed toward 3rd class |
| Sex | N/A | N/A | N/A | N/A | male: 577 (64.8%), female: 314 (35.2%) |
| Age | [0.42, 80.00] | 29.70 | 28.00 | 0.39 | Moderate right skew |
| SibSp | [0.00, 8.00] | 0.52 | 0.00 | 3.70 | Highly skewed |
| Parch | [0.00, 6.00] | 0.38 | 0.00 | 2.75 | Highly skewed |
| Fare | [0.00, 512.33] | 32.20 | 14.45 | 4.79 | Highly skewed |
| Embarked | N/A | N/A | N/A | N/A | S: 644 (72.3%), C: 168 (18.9%), Q: 77 (8.7%) |

---

## 5. Correlation Analysis

### Top 10 Feature Correlations

| Rank | Feature 1 | Feature 2 | Correlation | Strength |
|---|---|---|---|---|
| 1 | Pclass | Fare | -0.5495 | Moderate Negative |
| 2 | SibSp | Parch | 0.4148 | Moderate Positive |
| 3 | Pclass | Age | -0.3399 | Moderate Negative |
| 4 | Survived | Pclass | -0.3385 | Moderate Negative |
| 5 | Survived | Fare | 0.2573 | Weak-Moderate Positive |
| 6 | Pclass | Ticket | 0.2370 | Weak-Moderate Positive |
| 7 | Age | SibSp | -0.2333 | Weak Negative |
| 8 | Parch | Fare | 0.2162 | Weak-Moderate Positive |
| 9 | SibSp | Ticket | 0.1836 | Weak Positive |
| 10 | Age | Parch | -0.1725 | Weak Negative |

### Full Correlation Matrix (Survived Row)

| Feature | Correlation with Survived |
|---|---|
| PassengerId | -0.005 |
| Pclass | -0.338 |
| Age | -0.065 |
| SibSp | -0.035 |
| Parch | 0.082 |
| Ticket | -0.105 |
| Fare | 0.257 |

**Key Insight:** `Pclass` and `Fare` show the strongest inverse relationship (-0.5495), confirming that higher-class passengers paid more. `Survived` is most correlated with `Pclass` (-0.338) and `Fare` (0.257), suggesting socioeconomic status was a significant survival factor.

---

## 6. Statistical Hypothesis Testing Results

| Feature | Test Type | Statistic | P-Value | Significant? |
|---|---|---|---|---|
| PassengerId | Pearson Correlation | -0.005 | 0.8814 | No |
| Pclass | Pearson Correlation | -0.3385 | 2.54e-25 | Yes |
| Name | One-Way ANOVA | NaN | 1.0000 | No |
| Sex | Two-Sample Welch T-Test | 18.6718 | 2.28e-61 | Yes |
| Age | Pearson Correlation | -0.0649 | 0.0528 | No |
| SibSp | Pearson Correlation | -0.0353 | 0.2922 | No |
| Parch | Pearson Correlation | 0.0816 | 0.0148 | Yes |
| Ticket | Pearson Correlation | -0.1054 | 0.0016 | Yes |
| Fare | Pearson Correlation | 0.2573 | 6.12e-15 | Yes |
| Cabin | One-Way ANOVA | 1.8019 | 4.20e-07 | Yes |
| Embarked | One-Way ANOVA | 13.3269 | 1.98e-06 | Yes |

### Statistically Significant Predictors (p < 0.05)

1. Pclass
2. Sex
3. Parch
4. Ticket
5. Fare
6. Cabin
7. Embarked

---

## 7. Outlier Analysis

| Feature | Q1 | Q3 | IQR | Lower Bound | Upper Bound | Outlier Count | Outlier % | Action |
|---|---|---|---|---|---|---|---|---|
| Age | 22.0 | 35.0 | 13.0 | 2.5 | 54.5 | 66 | 7.41% | Profile |
| SibSp | 0.0 | 1.0 | 1.0 | -1.5 | 2.5 | 46 | 5.16% | Profile |
| Parch | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 213 | 23.91% | Profile |
| Fare | 7.91 | 31.0 | 23.09 | -26.72 | 65.63 | 116 | 13.02% | Profile |

**Note:** All outliers were profiled (not removed) to preserve the integrity of the dataset for downstream modeling.

---

## 8. Image Artifacts

| Artifact | Type | File Size | Description |
|---|---|---|---|
| correlation_matrix.png | Image Visualization | 120.69 KB | Heatmap displaying pairwise Pearson correlation coefficients across all numeric features. Highlights strong negative correlation between Pclass and Fare (-0.55) and moderate positive correlation between SibSp and Parch (0.41). |
| feature_distributions.png | Image Visualization | 187.86 KB | Distribution plots for all numeric and categorical features. Shows the highly right-skewed distribution of Fare (skewness 4.79), the bimodal nature of Survived, and the uniform spread of Pclass categories. |
| target_interactions.png | Image Visualization | 52.89 KB | Target variable interaction plots showing how Survived varies across feature combinations. Illustrates survival rate differences by Pclass, Sex, and Embarked. |

---

## 9. Feature Engineering Highlights

| Aspect | Detail |
|---|---|
| Engineered Features | None generated in current pipeline run |
| Data Versions | 4 states tracked (v0 raw -> v1-v3 progressive imputation) |
| Key Transformations | (1) Standardized missing string placeholders to NaN, (2) Median imputation for skewed numeric columns (Age, Ticket), (3) Mode imputation for categorical columns (Cabin, Embarked) |
| Ticket Column | Converted from object to float64 (230 missing standardized and imputed with median 236171.0) |
| Cabin Column | 77.1% missing; imputed with mode 'B96 B98' |

---

## 10. Predictive Modeling Blueprint

### Target Definition
- **Target:** Survived
- **Problem Type:** Classification
- **Dataset Dimensions:** 891 rows x 12 columns

### Recommended Algorithms

| Priority | Algorithm | Purpose |
|---|---|---|
| 1 | Regularized Logistic Regression | Baseline model |
| 2 | Random Forest Classifier | Non-linear ensemble baseline |
| 3 | Gradient Boosting Classifier (XGBoost / LightGBM) | High-performance boosted trees |
| 4 | Support Vector Classifier (SVM) | Margin-based classifier |

### Feature Selection Strategy

1. Exclude high-cardinality ID or text name columns (PassengerId, Name)
2. Rank features using cross-validated permutation importance and mutual information
3. Remove collinear features exceeding correlation threshold > 0.85

### Validation Strategy

- Stratified K-Fold Cross-Validation (5 folds)
- Evaluation Metrics: Balanced Accuracy, Macro F1, Precision-Recall AUC, Confusion Matrix

### Overfitting Risk Mitigation

1. Apply regularization penalties (L1/L2)
2. Limit tree depth and enforce minimum samples per leaf
3. Perform hyperparameter tuning strictly within cross-validation folds

---

## 11. Key Findings & Executive Summary

### Key Findings

1. **Dataset Composition:** The Titanic dataset contains 891 passenger records across 12 features, with a binary target variable (Survived: 0/1) representing approximately 38% survival rate.

2. **Data Quality Issues:** Three columns have missing values -- Cabin (77.1% missing, critical), Age (19.9% missing, moderate), and Embarked (0.2% missing, low). The Ticket column contained 230 invalid string placeholders that were standardized and imputed.

3. **Survival Correlates:** The strongest statistically significant predictors of survival are Sex (p < 0.001), Pclass (p < 0.001), Fare (p < 0.001), Embarked (p < 0.001), Cabin (p < 0.001), Parch (p = 0.015), and Ticket (p = 0.002).

4. **Socioeconomic Bias:** There is a clear socioeconomic gradient in survival -- higher Pclass (lower number) passengers had significantly higher survival rates, correlated with higher Fare payments (-0.55 Pclass-Fare correlation).

5. **Gender Disparity:** Sex is the strongest predictor of survival (T-statistic = 18.67, p < 0.001), consistent with the "women and children first" evacuation protocol.

6. **Skewness Concerns:** Fare (skewness 4.79), SibSp (3.70), and Parch (2.75) are highly skewed, requiring careful handling in modeling (median imputation, potential log-transformation).

7. **No Engineered Features:** The current pipeline did not generate any engineered features. Recommended future enhancements include family size (SibSp + Parch + 1), title extraction from Name, and cabin deck extraction from Cabin.

### Executive Summary

> **Target:** Survived (Classification). Use robust cross-validation on 891 rows x 12 columns. The dataset is well-suited for binary classification with 7 statistically significant predictors identified. Recommended modeling approach begins with Regularized Logistic Regression as a baseline, progressing to Gradient Boosting for performance optimization. All missing values have been imputed and outliers profiled. No data leakage risks were identified in the preprocessing pipeline.
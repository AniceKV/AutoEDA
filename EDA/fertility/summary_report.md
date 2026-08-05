# Executive Summary: Automated Exploratory Data Analysis (EDA) Report

**Dataset:** `fertility.csv`  
**Target Variable:** `Diagnosis`  
**Scope:** Comprehensive profiling, statistical testing, feature engineering assessment, and predictive modeling blueprint.  
**Generated Artifacts:** 23 files (12 distributions, 5 bivariate plots, 1 correlation matrix, 1 pairplot, 1 target interaction plot, 4 data state snapshots, 2 metrics/profile JSONs).

---

## 1. Dataset Profile

The dataset comprises 100 rows and 12 columns (10 original features + 2 engineered features). There are zero missing values across all columns. All features are either categorical (object) or integer/float numeric types.

### 1.1 Schema and Data Types

| Column | Dtype | Missing Count | Missing % | Cardinality |
|---|---|---|---|---|
| Season | object | 0 | 0.0 | 4 |
| Age | int64 | 0 | 0.0 | 10 |
| Childish diseases | object | 0 | 0.0 | 2 |
| Accident or serious trauma | object | 0 | 0.0 | 2 |
| Surgical intervention | object | 0 | 0.0 | 2 |
| High fevers in the last year | object | 0 | 0.0 | 3 |
| Frequency of alcohol consumption | object | 0 | 0.0 | 5 |
| Smoking habit | object | 0 | 0.0 | 3 |
| Number of hours spent sitting per day | int64 | 0 | 0.0 | 14 |
| Diagnosis | object | 0 | 0.0 | 2 |
| Age_SittingHours_interaction | int64 | 0 | 0.0 | 43 |
| Age_to_SittingHours_ratio | float64 | 0 | 0.0 | 47 |

### 1.2 Target Variable Distribution

The target variable `Diagnosis` is imbalanced, which must be accounted for during modeling.

| Diagnosis Class | Count | Proportion |
|---|---|---|
| Normal | 88 | 88.0% |
| Altered | 12 | 12.0% |

---

## 2. Key Statistical Findings

### 2.1 Descriptive Statistics for Numeric Features

| Feature | Min | Max | Mean | Median | Skewness |
|---|---|---|---|---|---|
| Age | 27.00 | 36.00 | 30.11 | 30.00 | N/A (Uniform-like) |
| Number of hours spent sitting per day | 1.00 | 342.00 | 10.80 | 7.00 | 9.85 (Highly Skewed) |

### 2.2 Outlier Analysis

The automated outlier detection (IQR method) flagged records in the `Number of hours spent sitting per day` feature.

| Feature | Q1 | Q3 | IQR | Lower Bound | Upper Bound | Outlier Count | Outlier % | Action |
|---|---|---|---|---|---|---|---|---|
| Age | 28.0 | 32.0 | 4.0 | 22.0 | 38.0 | 0 | 0.0% | Profile only |
| Number of hours spent sitting per day | 5.0 | 9.0 | 4.0 | -1.0 | 15.0 | 5 | 5.0% | Profile only |

**Note:** The extreme maximum value of 342.0 in `Number of hours spent sitting per day` represents a severe data entry anomaly or extreme outlier that warrants manual verification, as it heavily distorts the mean and skewness metrics.

---

## 3. Feature Engineering Highlights

Two features were engineered to capture the relationship between age and sedentary time.

| Feature Name | Formula | Data Type | Rationale |
|---|---|---|---|
| Age_SittingHours_interaction | Age * Number of hours spent sitting per day | int64 | Capture whether sedentary exposure has different implications across age levels. |
| Age_to_SittingHours_ratio | Age / (Number of hours spent sitting per day + eps) | float64 | Create a normalized age-to-sedentary-time measure while avoiding division by zero. |

**Critical Observation:** The `Age_SittingHours_interaction` feature exhibits a near-perfect multicollinearity with its parent feature `Number of hours spent sitting per day` (correlation = 0.9998). This will require careful handling (e.g., removal or regularization) during the modeling phase to avoid inflating coefficient variance in linear models.

---

## 4. Correlation Analysis

The correlation matrix was computed for all numeric features. The top pairwise correlations are summarized below.

| Feature 1 | Feature 2 | Correlation |
|---|---|---|
| Number of hours spent sitting per day | Age_SittingHours_interaction | 0.9998 |
| Age | Age_to_SittingHours_ratio | 0.3434 |
| Number of hours spent sitting per day | Age_to_SittingHours_ratio | -0.1811 |
| Age_SittingHours_interaction | Age_to_SittingHours_ratio | -0.1767 |
| Age | Number of hours spent sitting per day | -0.0466 |
| Age | Age_SittingHours_interaction | -0.0305 |

**Implication:** The `Age_SittingHours_interaction` and `Number of hours spent sitting per day` columns provide redundant information. The feature selection strategy must drop one of these to satisfy the collinearity threshold (< 0.85) before model training.

---

## 5. Hypothesis Testing Results

Bivariate statistical tests were conducted to evaluate the independence of each feature against the target variable `Diagnosis`.

| Feature | Test Name | Statistic | P-Value | Significant? |
|---|---|---|---|---|
| Season | Chi-Square Test of Independence | 4.1613 | 0.2446 | No |
| Age | Two-Sample Welch T-Test | 1.0435 | 0.3126 | No |
| Childish diseases | Chi-Square Test of Independence | 0.0000 | 1.0000 | No |
| Accident or serious trauma | Chi-Square Test of Independence | 1.2177 | 0.2698 | No |
| Surgical intervention | Chi-Square Test of Independence | 0.0547 | 0.8150 | No |
| High fevers in the last year | Chi-Square Test of Independence | 1.5452 | 0.4618 | No |
| Frequency of alcohol consumption | Chi-Square Test of Independence | 4.0263 | 0.4025 | No |
| Smoking habit | Chi-Square Test of Independence | 0.2153 | 0.8980 | No |
| Number of hours spent sitting per day | Two-Sample Welch T-Test | -0.9024 | 0.3691 | No |
| Age_SittingHours_interaction | Two-Sample Welch T-Test | -0.8751 | 0.3837 | No |
| Age_to_SittingHours_ratio | Two-Sample Welch T-Test | 0.4771 | 0.6421 | No |

**Finding:** At a standard significance level of alpha = 0.05, **none of the features demonstrate a statistically significant association with the target variable `Diagnosis`**. The `significant_predictors` list is empty. This suggests that the relationship between the observed features and the diagnosis may be non-linear, highly complex, or that the current feature set lacks sufficient predictive power without further domain-specific transformation.

---

## 6. Image Artifact Inventory

The EDA pipeline generated the following visual artifacts to support the analysis:

### 6.1 Distribution Plots (Univariate)
- `dist_Age.png`
- `dist_Childish_diseases.png`
- `dist_Accident_or_serious_trauma.png`
- `dist_Surgical_intervention.png`
- `dist_High_fevers_in_the_last_year.png`
- `dist_Frequency_of_alcohol_consumption.png`
- `dist_Smoking_habit.png`
- `dist_Number_of_hours_spent_sitting_per_day.png`
- `dist_Season.png`
- `dist_Diagnosis.png`

### 6.2 Bivariate and Target Interaction Plots
- `bivariate_Age_vs_Number_of_hours_spent_sitting_per_day.png`
- `bivariate_Frequency_of_alcohol_consumption_vs_Diagnosis.png`
- `bivariate_High_fevers_in_the_last_year_vs_Diagnosis.png`
- `bivariate_Surgical_intervention_vs_Diagnosis.png`
- `target_interactions.png`

### 6.3 Multivariate and Correlation Visuals
- `correlation_matrix.png`
- `pairplot.png`

---

## 7. Predictive Modeling Blueprint

Based on the EDA findings, the following blueprint is recommended for the subsequent modeling phase.

### 7.1 Problem Definition
- **Target:** `Diagnosis`
- **Problem Type:** Classification (Binary)
- **Data Dimensions:** 100 rows x 12 columns

### 7.2 Recommended Algorithms
1. Regularized Logistic Regression (baseline)
2. Random Forest Classifier
3. Gradient Boosting Classifier (XGBoost / LightGBM)
4. Support Vector Classifier (SVM)

### 7.3 Feature Selection Strategy
- Exclude high-cardinality ID or text name columns.
- Rank features using cross-validated permutation importance and mutual information.
- **Remove collinear features exceeding correlation threshold > 0.85** (specifically, drop either `Number of hours spent sitting per day` or `Age_SittingHours_interaction`).

### 7.4 Validation Strategy
- **Cross-Validation:** Stratified K-Fold (5 folds) to preserve the class imbalance ratio in each fold.
- **Evaluation Metrics:** Balanced Accuracy, Macro F1, Precision-Recall AUC, and Confusion Matrix.

### 7.5 Overfitting Risk Mitigation
- Apply regularization penalties (L1/L2).
- Limit tree depth and enforce minimum samples per leaf.
- Perform hyperparameter tuning strictly within cross-validation folds.

---

## 8. Conclusion and Next Steps

The automated EDA pipeline successfully profiled the `fertility.csv` dataset, confirming data completeness (0 missing values) and identifying critical structural issues. The primary findings are:

1. **Class Imbalance:** The target `Diagnosis` is heavily skewed towards "Normal" (88% vs 12%).
2. **Data Quality Anomaly:** The `Number of hours spent sitting per day` contains an extreme outlier (max = 342) and is highly skewed (9.85), requiring robust scaling or transformation.
3. **Multicollinearity:** The engineered feature `Age_SittingHours_interaction` is redundant with its parent feature (correlation = 0.9998) and must be dropped prior to linear modeling.
4. **Lack of Linear Signal:** Hypothesis testing revealed no statistically significant univariate predictors of `Diagnosis` at the 0.05 level.

**Recommended Next Steps:**
- Investigate the 342-hour sitting outlier to determine if it is a valid extreme case or a data entry error.
- Apply non-linear feature transformations (e.g., polynomial, binning) to capture potential complex interactions missed by linear tests.
- Proceed with the predictive modeling blueprint, strictly adhering to the collinearity removal and stratified cross-validation protocols.
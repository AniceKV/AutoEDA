# Executive Summary Report: Automated Exploratory Data Analysis (EDA)

**Dataset:** Titanic-Dataset.csv
**Pipeline Output Directory:** Working Directory
**Report Generated From:** Automated EDA Pipeline Artifacts
**Excluded Artifacts:** generated_analysis.py (script file, excluded per request)

---

## 1. Dataset Overview

| Property              | Value          |
|-----------------------|----------------|
| Dataset Name          | Titanic-Dataset.csv |
| Total Rows            | 891            |
| Total Columns         | 12             |
| Target Column         | Survived       |
| Problem Type          | Classification |
| File Path             | C:\Users\Anish Kumar Verma\PycharmProjects\AutoEDA\test_data\Titanic-Dataset.csv |

The dataset contains passenger records from the Titanic disaster, with the binary target variable `Survived` (0 = Did Not Survive, 1 = Survived). The mean survival rate across the dataset is **0.38** (approximately 38%), indicating class imbalance favoring non-survivors.

---

## 2. Schema and Column Profiles

| Column       | Dtype    | Missing Count | Missing % | Cardinality | Key Metric                                      |
|--------------|----------|---------------|-----------|-------------|-------------------------------------------------|
| PassengerId  | int64    | 0             | 0.0%      | 891         | Range: [1, 891] | Mean: 446.00 | Median: 446.00 |
| Survived     | int64    | 0             | 0.0%      | 2           | Range: [0, 1] | Mean: 0.38 | Median: 0.00 |
| Pclass       | int64    | 0             | 0.0%      | 3           | Range: [1, 3] | Mean: 2.31 | Median: 3.00 |
| Name         | object   | 0             | 0.0%      | 891         | All unique values (high cardinality)            |
| Sex          | object   | 0             | 0.0%      | 2           | male: 577 (64.8%), female: 314 (35.2%)         |
| Age          | float64  | 177             | 19.87%    | 88          | Range: [0.42, 80.00] | Mean: 29.70 | Median: 28.00 |
| SibSp        | int64    | 0             | 0.0%      | 7           | Range: [0, 8] | Mean: 0.52 | Median: 0.00 | Skewed: 3.70 |
| Parch        | int64    | 0             | 0.0%      | 7           | Range: [0, 6] | Mean: 0.38 | Median: 0.00 | Skewed: 2.75 |
| Ticket       | object   | 0             | 0.0%      | 681         | Top: '347082' (7), '1601' (7), 'CA. 2343' (7) |
| Fare         | float64  | 0             | 0.0%      | 248         | Range: [0.00, 512.33] | Mean: 32.20 | Median: 14.45 | Skewed: 4.79 |
| Cabin        | object   | 687             | 77.1%     | 147         | Top: 'G6' (4), 'C23 C25 C27' (4), 'B96 B98' (4) |
| Embarked     | object   | 2             | 0.22%     | 3           | Top: 'S' (644), 'C' (168), 'Q' (77)            |

---

## 3. Missing Values Analysis

Three columns contain missing values:

| Column    | Missing Count | Missing Percentage | Severity   |
|-----------|---------------|--------------------|------------|
| Age       | 177           | 19.87%             | Moderate   |
| Cabin     | 687           | 77.10%             | Critical   |
| Embarked  | 2             | 0.22%              | Low        |

**Imputation Strategy Applied:**

| Column    | Method   | Skewness | Fill Value   | Rationale                                      |
|-----------|----------|----------|--------------|------------------------------------------------|
| Age       | Median   | 0.39     | 28.0         | Numeric, low skewness; median is robust        |
| Ticket    | Median   | 5.27     | 236171.0     | Numeric, high skewness; median is robust       |
| Cabin     | Mode     | N/A      | B96 B98      | Categorical; mode imputation with fallback     |
| Embarked  | Mode     | N/A      | S            | Categorical; mode imputation with fallback     |

**Note:** The `metrics.json` column summary reports `missing_count: 0` for `Age`, `Ticket`, and `Cabin` after imputation, confirming all missing values were successfully filled. The `df_state_v0.csv` (pre-imputation) retains original missing values (empty fields), while `df_state_v1.csv` through `df_state_v3.csv` reflect the imputed state.

---

## 4. Statistical Hypothesis Testing Results

The following features were tested for statistical significance against the target variable `Survived`:

| Feature    | Test Name                     | Statistic    | P-Value           | Significant? |
|------------|-------------------------------|--------------|-------------------|--------------|
| Pclass     | Pearson Correlation Test      | -0.3385      | 2.5370e-25        | YES          |
| Sex        | Two-Sample Welch T-Test       | 18.6718      | 2.2836e-61        | YES          |
| Parch      | Pearson Correlation Test      | 0.0816       | 1.4799e-02        | YES          |
| Ticket     | Pearson Correlation Test      | -0.1054      | 1.6253e-03        | YES          |
| Fare       | Pearson Correlation Test      | 0.2573       | 6.1202e-15        | YES          |
| Cabin      | One-Way ANOVA                 | 2.7851       | 1.2811e-08        | YES          |
| Embarked   | One-Way ANOVA                 | 13.3269      | 1.9832e-06        | YES          |
| PassengerId| Pearson Correlation Test      | -0.0050      | 8.8137e-01        | NO           |
| Age        | Pearson Correlation Test      | -0.0649      | 5.2761e-02        | NO           |
| SibSp      | Pearson Correlation Test      | -0.0353      | 2.9224e-01        | NO           |

**Statistically Significant Predictors (7 of 10):** Pclass, Sex, Parch, Ticket, Fare, Cabin, Embarked.

---

## 5. Correlation Analysis

### Top Correlations (Absolute Value, Descending)

| Feature 1 | Feature 2 | Correlation | Strength       |
|-----------|-----------|-------------|----------------|
| Pclass    | Fare      | -0.5495     | Moderate       |
| SibSp     | Parch     | 0.4148      | Moderate       |
| Pclass    | Age       | -0.3399     | Moderate       |
| Survived  | Pclass    | -0.3385     | Moderate       |
| Survived  | Fare      | 0.2573      | Weak-Moderate  |
| Pclass    | Ticket    | 0.2370      | Weak-Moderate  |
| Age       | SibSp     | -0.2333     | Weak           |
| Parch     | Fare      | 0.2162      | Weak           |
| SibSp     | Ticket    | 0.1836      | Weak           |
| Age       | Parch     | -0.1725     | Weak           |

### Key Correlation Observations

- **Pclass and Fare** exhibit the strongest linear relationship (r = -0.5495), confirming that higher passenger classes paid higher fares.
- **Survived and Pclass** are moderately negatively correlated (r = -0.3385), indicating lower-class passengers had lower survival rates.
- **Survived and Fare** show a weak positive correlation (r = 0.2573), suggesting wealthier passengers had a slight survival advantage.
- **SibSp and Parch** are moderately positively correlated (r = 0.4148), reflecting family group travel patterns.
- **PassengerId** shows negligible correlation with all other features, confirming it is a non-informative identifier.

---

## 6. Outlier Analysis

| Feature  | Q1      | Q3      | IQR      | Lower Bound | Upper Bound | Outlier Count | Outlier % | Action  |
|----------|---------|---------|----------|-------------|-------------|---------------|-----------|---------|
| Age      | 22.0    | 35.0    | 13.0     | 2.5         | 54.5        | 66            | 7.41%     | Profile |
| SibSp    | 0.0     | 1.0     | 1.0      | -1.5        | 2.5         | 46            | 5.16%     | Profile |
| Parch    | 0.0     | 0.0     | 0.0      | 0.0         | 0.0         | 213           | 23.91%    | Profile |
| Fare     | 7.9104  | 31.0    | 23.0896  | -26.724     | 65.6344     | 116           | 13.02%    | Profile |

**Interpretation:**
- **Parch** has the highest outlier rate (23.91%), which is expected given its heavily zero-inflated distribution (median = 0).
- **Fare** exhibits 13.02% outliers, consistent with its high skewness (4.79) and the presence of very high fare values (max: 512.33).
- **Age** outliers (7.41%) are within acceptable bounds for a real-world demographic dataset.
- All outliers were profiled (not removed), preserving the integrity of the dataset for modeling.

---

## 7. Feature Engineering Highlights

| Aspect                  | Status       | Details                                                        |
|-------------------------|--------------|----------------------------------------------------------------|
| Engineered Features     | None         | The `engineered_features` list in metrics.json is empty.       |
| Missing Value Handling  | Completed    | 4 columns imputed using median/mode strategies.                |
| String Standardization  | Completed    | Missing string placeholders ('?', 'NA', 'N/A', 'null') converted to NaN. |
| Skewness-Based Imputation | Applied    | Numeric columns with |skew| > 1.0 use median; otherwise mean. |

**Recommendation:** Consider engineering features such as:
- Family size (SibSp + Parch + 1)
- IsAlone flag (binary)
- Title extraction from Name (Mr., Mrs., Miss., etc.)
- Fare per person (Fare / family size)
- Cabin deck letter extraction

---

## 8. Image Artifacts Inventory

### 8.1 Distribution Plots (Univariate)

| File                  | Type                | Size (KB) | Description                                      |
|-----------------------|---------------------|-----------|--------------------------------------------------|
| dist_Age.png          | image_visualization | 41.31     | Age distribution histogram/kde                    |
| dist_Embarked.png     | image_visualization | 23.47     | Embarked port distribution (S, C, Q)              |
| dist_Fare.png         | image_visualization | 32.79     | Fare distribution (right-skewed)                  |
| dist_Parch.png        | image_visualization | 37.55     | Parch (parents/children) count distribution       |
| dist_Pclass.png       | image_visualization | 41.48     | Passenger class distribution (1, 2, 3)            |
| dist_Sex.png          | image_visualization | 24.80     | Gender distribution (male vs. female)             |
| dist_SibSp.png        | image_visualization | 32.97     | SibSp (siblings/spouses) count distribution       |
| dist_Survived.png     | image_visualization | 36.18     | Target variable distribution (0 vs. 1)            |
| feature_distributions.png | image_visualization | 187.86 | Multi-panel overview of all feature distributions |

### 8.2 Bivariate Plots

| File                          | Type                | Size (KB) | Description                                          |
|-------------------------------|---------------------|-----------|------------------------------------------------------|
| bivariate_Age_vs_Fare.png     | image_visualization | 128.24    | Age vs. Fare scatter/bivariate plot                  |
| bivariate_Age_vs_Survived.png | image_visualization | 81.53     | Age vs. Survived relationship                        |
| bivariate_Fare_vs_Survived.png| image_visualization | 75.33     | Fare vs. Survived relationship                       |
| bivariate_Pclass_vs_Fare.png  | image_visualization | 55.64     | Passenger class vs. Fare comparison                  |
| bivariate_Sex_vs_Pclass.png   | image_visualization | 38.26     | Gender vs. passenger class cross-tabulation          |
| bivariate_Sex_vs_Survived.png | image_visualization | 34.70     | Gender vs. survival rate comparison                  |

### 8.3 Multivariate and Summary Visualizations

| File                  | Type                | Size (KB) | Description                                      |
|-----------------------|---------------------|-----------|--------------------------------------------------|
| pairplot.png          | image_visualization | 156.96    | Pairwise scatter matrix of numeric features      |
| correlation_matrix.png| image_visualization | 120.69    | Heatmap of feature correlation coefficients      |
| target_interactions.png| image_visualization | 53.17    | Target variable interaction plots                |

---

## 9. Predictive Modeling Blueprint

### 9.1 Problem Definition

| Property              | Value                          |
|-----------------------|--------------------------------|
| Target                | Survived                       |
| Problem Type          | Classification (Binary)        |
| Dataset Dimensions    | 891 rows x 12 columns          |

### 9.2 Recommended Algorithms

| Priority | Algorithm                              | Role           |
|----------|----------------------------------------|----------------|
| 1        | Regularized Logistic Regression        | Baseline       |
| 2        | Random Forest Classifier               | Ensemble       |
| 3        | Gradient Boosting (XGBoost / LightGBM) | Ensemble       |
| 4        | Support Vector Classifier (SVM)        | Kernel-based   |

### 9.3 Feature Selection Strategy

1. Exclude high-cardinality ID or text name columns (e.g., PassengerId, Name).
2. Rank features using cross-validated permutation importance and mutual information.
3. Remove collinear features exceeding a correlation threshold of > 0.85.

### 9.4 Validation Strategy

- **Method:** Stratified K-Fold Cross-Validation (5 folds)
- **Metrics:** Balanced Accuracy, Macro F1, Precision-Recall AUC, Confusion Matrix

### 9.5 Overfitting Risk Mitigation

- Apply regularization penalties (L1 / L2)
- Limit tree depth and enforce minimum samples per leaf
- Perform hyperparameter tuning strictly within cross-validation folds

---

## 10. Key Findings and Executive Summary

### 10.1 Data Quality

- The dataset is relatively clean with only 3 columns containing missing values.
- **Cabin** has the most severe missingness (77.1%), making it a challenging feature to leverage without significant imputation or encoding strategy.
- **Age** has moderate missingness (19.87%) and was imputed using the median (28.0).
- **Embarked** has minimal missingness (2 records) and was imputed using the mode ('S').

### 10.2 Statistical Significance

- **7 out of 10** features show statistically significant associations with survival.
- **Sex** is the strongest predictor (Welch T-test statistic = 18.67, p < 0.001).
- **Pclass** is the second strongest predictor (Pearson r = -0.3385, p < 0.001).
- **Age** and **SibSp** are not statistically significant at the conventional alpha = 0.05 level.

### 10.3 Correlation Insights

- Passenger class and fare are moderately negatively correlated (r = -0.5495).
- Survival and fare show a weak positive correlation (r = 0.2573).
- PassengerId is uncorrelated with all other features, confirming its non-informative nature.

### 10.4 Outlier Profile

- **Parch** has the highest outlier rate (23.91%), driven by its zero-inflated distribution.
- **Fare** has 13.02% outliers, consistent with its high skewness (4.79).
- No outliers were removed; all were profiled for downstream consideration.

### 10.5 Modeling Readiness

- The dataset is ready for predictive modeling after imputation.
- Recommended starting model: **Regularized Logistic Regression** as a baseline.
- Suggested next steps: Feature engineering (family size, title extraction, fare per person), followed by ensemble methods (Random Forest, Gradient Boosting).
- Validation should use **Stratified 5-Fold Cross-Validation** with metrics including Balanced Accuracy, Macro F1, and Precision-Recall AUC.

---

## 11. Data State Versions

| File          | Description                          | State          |
|---------------|--------------------------------------|----------------|
| df_state_v0.csv | Raw data with original missing values | Pre-imputation |
| df_state_v1.csv | Imputed data (Ticket, Cabin, Embarked) | Post-imputation v1 |
| df_state_v2.csv | Imputed data (consistent with v1)    | Post-imputation v2 |
| df_state_v3.csv | Imputed data (consistent with v1/v2) | Post-imputation v3 |

All three imputed states (v1-v3) show consistent transformations: Ticket values standardized to numeric, Cabin values filled with mode 'B96 B98', and Embarked values filled with mode 'S'.

---

*End of Executive Summary Report*
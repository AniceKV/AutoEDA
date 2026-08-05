# Executive Summary Report: Automated Exploratory Data Analysis (EDA) Pipeline

**Dataset:** Titanic-Dataset.csv
**Pipeline Output Directory:** Working Directory
**Report Generated From:** Automated EDA Artifact Files
**Excluded Artifact:** `generated_analysis.py` (script file, excluded per request)

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

### 1.1 Column Schema

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
| Ticket | object | 0 | 0.00% | 681 |
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

**Note:** The `Ticket` column also exhibited missing values (230 entries) in the raw state prior to imputation, as confirmed by the imputation summary.

---

## 3. Imputation Summary

The pipeline applied the following imputation rules:

| Rule | Description |
|---|---|
| Rule 1 | Standardized missing string placeholders ('?', 'NA', 'N/A', 'null') to NaN |
| Rule 2 | Numeric columns with skewness > 1.0 or < -1.0 use median imputation |
| Rule 3 | Numeric columns with skewness between -1.0 and 1.0 use mean imputation |
| Rule 4 | Categorical/String columns use mode imputation with 'Unknown' fallback |

### 3.1 Imputation Details Per Column

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

## 4. Statistical Distributions & Key Metrics

### 4.1 Numerical Feature Summaries

| Feature | Range | Mean | Median | Skewness | Notes |
|---|---|---|---|---|---|
| PassengerId | [1.00, 891.00] | 446.00 | 446.00 | N/A | Unique identifier |
| Age | [0.42, 80.00] | 29.70 | 28.00 | 0.39 | Moderate right skew |
| SibSp | [0.00, 8.00] | 0.52 | 0.00 | 3.70 | Highly skewed |
| Parch | [0.00, 6.00] | 0.38 | 0.00 | 2.75 | Highly skewed |
| Fare | [0.00, 512.33] | 32.20 | 14.45 | 4.79 | Highly skewed |
| Survived | [0.00, 1.00] | 0.38 | 0.00 | N/A | Binary target (38% survival rate) |
| Pclass | [1.00, 3.00] | 2.31 | 3.00 | N/A | Categorical ordinal |

### 4.2 Categorical Feature Summaries

| Feature | Cardinality | Top Values |
|---|---|---|
| Sex | 2 | male: 577, female: 314 |
| Embarked | 3 | S: 644, C: 168, Q: 77 |
| Pclass | 3 | 1, 2, 3 (ordinal) |
| Name | 891 | All unique (high cardinality) |
| Ticket | 681 | '347082': 7, '1601': 7, 'CA. 2343': 7 |
| Cabin | 147 | 'G6': 4, 'C23 C25 C27': 4, 'B96 B98': 4 |

---

## 5. Correlation Analysis

### 5.1 Top Correlations (Absolute Magnitude)

| Rank | Feature 1 | Feature 2 | Correlation | Direction |
|---|---|---|---|---|
| 1 | Pclass | Fare | -0.5495 | Negative |
| 2 | SibSp | Parch | 0.4148 | Positive |
| 3 | Pclass | Age | -0.3399 | Negative |
| 4 | Survived | Pclass | -0.3385 | Negative |
| 5 | Survived | Fare | 0.2573 | Positive |
| 6 | Pclass | Ticket | 0.2370 | Positive |
| 7 | Age | SibSp | -0.2333 | Negative |
| 8 | Parch | Fare | 0.2162 | Positive |
| 9 | SibSp | Ticket | 0.1836 | Positive |
| 10 | Age | Parch | -0.1725 | Negative |

### 5.2 Correlation Matrix (Full)

| | PassengerId | Survived | Pclass | Age | SibSp | Parch | Ticket | Fare |
|---|---|---|---|---|---|---|---|---|
| PassengerId | 1.000 | -0.005 | -0.035 | 0.034 | -0.058 | -0.002 | -0.064 | 0.013 |
| Survived | -0.005 | 1.000 | -0.338 | -0.065 | -0.035 | 0.082 | -0.105 | 0.257 |
| Pclass | -0.035 | -0.338 | 1.000 | -0.340 | 0.083 | 0.018 | 0.237 | -0.549 |
| Age | 0.034 | -0.065 | -0.340 | 1.000 | -0.233 | -0.172 | -0.125 | 0.097 |
| SibSp | -0.058 | -0.035 | 0.083 | -0.233 | 1.000 | 0.415 | 0.184 | 0.160 |
| Parch | -0.002 | 0.082 | 0.018 | -0.172 | 0.415 | 1.000 | 0.074 | 0.216 |
| Ticket | -0.064 | -0.105 | 0.237 | -0.125 | 0.184 | 0.074 | 1.000 | -0.091 |
| Fare | 0.013 | 0.257 | -0.549 | 0.097 | 0.160 | 0.216 | -0.091 | 1.000 |

### 5.3 Key Correlation Observations

- **Pclass and Fare** exhibit the strongest linear relationship (r = -0.5495), indicating that higher passenger classes paid substantially higher fares.
- **Survived and Pclass** show a moderate negative correlation (r = -0.3385), suggesting that lower-class passengers had lower survival rates.
- **Survived and Fare** show a moderate positive correlation (r = 0.2573), indicating that passengers who paid higher fares had better survival outcomes.
- **SibSp and Parch** are moderately positively correlated (r = 0.4148), reflecting family group travel patterns.
- **PassengerId** shows negligible correlation with all other features, confirming it is a non-informative identifier.

---

## 6. Hypothesis Testing Results

Statistical tests were performed to assess the relationship between each feature and the target variable (`Survived`).

| Feature | Test Type | Statistic | P-Value | Significant? | Interpretation |
|---|---|---|---|---|---|
| Pclass | Pearson Correlation | -0.3385 | 2.537e-25 | YES | Statistically Significant |
| Sex | Two-Sample Welch T-Test | 18.6718 | 2.284e-61 | YES | Statistically Significant |
| Parch | Pearson Correlation | 0.0816 | 1.480e-02 | YES | Statistically Significant |
| Ticket | Pearson Correlation | -0.1054 | 1.625e-03 | YES | Statistically Significant |
| Fare | Pearson Correlation | 0.2573 | 6.120e-15 | YES | Statistically Significant |
| Cabin | One-Way ANOVA | 1.8019 | 4.198e-07 | YES | Statistically Significant |
| Embarked | One-Way ANOVA | 13.3269 | 1.983e-06 | YES | Statistically Significant |
| Age | Pearson Correlation | -0.0649 | 5.276e-02 | NO | Not Significant |
| SibSp | Pearson Correlation | -0.0353 | 2.922e-01 | NO | Not Significant |
| Name | One-Way ANOVA | NaN | 1.0 | NO | Not Significant |
| PassengerId | Pearson Correlation | -0.005 | 8.814e-01 | NO | Not Significant |

### 6.1 Statistically Significant Predictors (p < 0.05)

1. Pclass
2. Sex
3. Parch
4. Ticket
5. Fare
6. Cabin
7. Embarked

---

## 7. Outlier Analysis

Outliers were identified using the Interquartile Range (IQR) method: Outlier if value < Q1 - 1.5*IQR or value > Q3 + 1.5*IQR.

| Feature | Q1 | Q3 | IQR | Lower Bound | Upper Bound | Outlier Count | Outlier % | Action |
|---|---|---|---|---|---|---|---|---|
| Age | 22.0 | 35.0 | 13.0 | 2.5 | 54.5 | 66 | 7.41% | Profile |
| Fare | 7.91 | 31.00 | 23.09 | -26.72 | 65.63 | 116 | 13.02% | Profile |
| SibSp | 0.0 | 1.0 | 1.0 | -1.5 | 2.5 | 46 | 5.16% | Profile |
| Parch | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 213 | 23.91% | Profile |

**Key Observations:**
- **Parch** has the highest outlier percentage (23.91%), driven by a zero IQR where all non-zero values are flagged as outliers.
- **Fare** has 13.02% outliers, consistent with its high skewness (4.79) and the presence of very high fares (max: 512.33).
- **Age** has 7.41% outliers, which is within an acceptable range for a real-world dataset.
- **SibSp** has 5.16% outliers, with a relatively tight IQR of 1.0.

All outlier actions were set to "profile" -- meaning outliers were documented but not removed or capped, preserving the integrity of the raw data distribution.

---

## 8. Feature Engineering Highlights

The pipeline's `engineered_features` list is currently **empty**. No additional features were created during this EDA run. The following columns are candidates for future feature engineering:

| Candidate Feature | Source Column(s) | Rationale |
|---|---|---|
| FamilySize | SibSp + Parch | Combined family group size |
| IsAlone | FamilySize | Binary flag for solo travelers |
| Title | Name | Extract titles (Mr., Mrs., Miss., etc.) |
| Deck | Cabin | Extract deck letter from cabin codes |
| FarePerPerson | Fare / FamilySize | Normalized fare by group size |
| AgeGroup | Age | Categorical binning of age |

---

## 9. Image Artifact Descriptions

The pipeline generated the following visual artifacts. All files are PNG image visualizations.

### 9.1 Distribution Plots (Univariate)

| File | Size (KB) | Description |
|---|---|---|
| dist_Age.png | 41.31 | Univariate distribution of passenger age |
| dist_Embarked.png | 23.47 | Univariate distribution of embarkation port (S, C, Q) |
| dist_Fare.png | 32.79 | Univariate distribution of ticket fare |
| dist_Parch.png | 37.55 | Univariate distribution of number of parents/children aboard |
| dist_Pclass.png | 41.48 | Univariate distribution of passenger class (1, 2, 3) |
| dist_Sex.png | 24.80 | Univariate distribution of gender (male/female) |
| dist_SibSp.png | 32.97 | Univariate distribution of number of siblings/spouses aboard |
| dist_Survived.png | 36.18 | Univariate distribution of survival outcome (0/1) |
| feature_distributions.png | 187.42 | Combined multi-feature distribution overview |

### 9.2 Bivariate & Interaction Plots

| File | Size (KB) | Description |
|---|---|---|
| bivariate_Age_vs_Fare.png | 128.24 | Scatter/relationship plot of Age vs Fare |
| bivariate_Pclass_vs_Fare.png | 55.64 | Relationship between passenger class and fare |
| bivariate_Sex_vs_Pclass.png | 38.26 | Cross-tabulation of gender and passenger class |
| pairplot.png | 145.75 | Pairwise scatter plot matrix of all numerical features |
| target_interactions.png | 68.90 | Interaction plots showing feature relationships with the Survived target |

### 9.3 Correlation Visualization

| File | Size (KB) | Description |
|---|---|---|
| correlation_matrix.png | 120.69 | Heatmap visualization of the full correlation matrix |

---

## 10. Predictive Modeling Blueprint

### 10.1 Problem Definition

| Property | Value |
|---|---|
| Target Variable | Survived |
| Problem Type | Classification |
| Dataset Dimensions | 891 rows x 12 columns |

### 10.2 Recommended Algorithms

1. **Regularized Logistic Regression** -- Baseline model; interpretable and robust for binary classification.
2. **Random Forest Classifier** -- Ensemble method capturing non-linear feature interactions.
3. **Gradient Boosting Classifier (XGBoost / LightGBM)** -- High-performance boosting approach for structured tabular data.
4. **Support Vector Classifier (SVM)** -- Effective in high-dimensional spaces with clear margin separation.

### 10.3 Feature Selection Strategy

1. Exclude high-cardinality ID or text name columns (e.g., PassengerId, Name).
2. Rank features using cross-validated permutation importance and mutual information.
3. Remove collinear features exceeding a correlation threshold of > 0.85.

### 10.4 Validation Strategy

- **Stratified K-Fold Cross-Validation** with 5 folds to ensure class balance across folds.
- **Evaluation Metrics:** Balanced Accuracy, Macro F1, Precision-Recall AUC, and Confusion Matrix.

### 10.5 Overfitting Risk Mitigation

- Apply regularization penalties (L1/L2) to penalize model complexity.
- Limit tree depth and enforce minimum samples per leaf in tree-based models.
- Perform hyperparameter tuning strictly within cross-validation folds to avoid data leakage.

### 10.6 Executive Summary

> Target: Survived (Classification). Use robust cross-validation on 891 rows x 12 columns. The dataset is moderately sized with 7 statistically significant predictors identified through hypothesis testing. Key features for modeling include Pclass, Sex, Fare, Cabin, Embarked, Parch, and Ticket.

---

## 11. Key Findings & Insights

### 11.1 Summary of Key Findings

1. The dataset contains **891 rows and 12 columns**, representing passenger records from the Titanic disaster.
2. **Missing values** were identified in three columns: Cabin (77.1%), Age (19.9%), and Embarked (0.2%). All were successfully imputed using the pipeline's rule-based strategy.
3. The **survival rate** is approximately 38% (mean of Survived = 0.38), indicating class imbalance in the target variable.
4. **Seven features** were found to be statistically significant predictors of survival (p < 0.05): Pclass, Sex, Parch, Ticket, Fare, Cabin, and Embarked.
5. The strongest predictor of survival is **Sex** (Welch T-Test statistic = 18.67, p = 2.28e-61), followed by **Pclass** (Pearson r = -0.3385, p = 2.54e-25).
6. **Fare** shows a moderate positive correlation with survival (r = 0.2573), while **Pclass** shows a moderate negative correlation (r = -0.3385), reflecting the "women and children first" protocol and class-based access to lifeboats.
7. **Parch** (number of parents/children) has a weak but statistically significant positive correlation with survival (r = 0.0816, p = 0.015).
8. **Age** was found to be NOT statistically significant (p = 0.053), despite its intuitive importance -- this may be due to the moderate missing rate and the imputation strategy employed.

### 11.2 Data Quality Notes

- The `Ticket` column was originally stored as an object type but was converted to float64 during imputation (230 missing values imputed with median 236171.0). This conversion may warrant review, as Ticket is a categorical identifier rather than a numeric quantity.
- The `Cabin` column has 77.1% missing values, making it a high-noise feature. The mode imputation fill value "B96 B98" represents the most common cabin code among passengers with recorded cabin data.
- The `Name` column has 891 unique values (100% cardinality) and is not suitable for direct modeling without feature extraction (e.g., title extraction).

### 11.3 Pipeline Artifacts Summary

| Artifact Type | Count | Files |
|---|---|---|
| CSV Data States | 4 | df_state_v0.csv (raw), df_state_v1.csv, df_state_v2.csv, df_state_v3.csv (imputed) |
| Distribution Plots | 9 | dist_*.png, feature_distributions.png |
| Bivariate Plots | 3 | bivariate_*.png |
| Correlation Visualizations | 1 | correlation_matrix.png |
| Pairwise/Interaction Plots | 2 | pairplot.png, target_interactions.png |
| **Total Image Artifacts** | **15** | |
| **Total CSV Artifacts** | **4** | |
| **Total JSON Artifacts** | **2** | metadata_profile.json, metrics.json |
| **Grand Total** | **21** | |

---

*End of Executive Summary Report*
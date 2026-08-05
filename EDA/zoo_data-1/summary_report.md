# Executive Summary Report: Automated Exploratory Data Analysis (EDA) Pipeline

---

## 1. Executive Overview

This report summarizes the outputs of an automated Exploratory Data Analysis (EDA) pipeline executed against the **zoo_data-1.csv** dataset. The pipeline performed data profiling, missing value assessment, statistical distribution analysis, correlation computation, hypothesis testing, outlier detection, and predictive modeling blueprinting. All artifacts -- including visualizations, data snapshots, and metrics -- were generated and stored in the working directory. The script file `generated_analysis.py` was excluded from this review per directive.

The dataset comprises **101 rows** and **17 columns**, with the target variable `catsize` (binary: 0 or 1) designated for a classification task. No missing values were detected in any column. All features are numeric (int64) except `animal_name`, which is an object-type identifier with 100 unique values.

---

## 2. Dataset Profile

### 2.1 Schema Overview

| Column       | Dtype   | Missing Count | Missing % | Cardinality |
|--------------|---------|---------------|-----------|-------------|
| animal_name  | object  | 0             | 0.0%      | 100         |
| hair         | int64   | 0             | 0.0%      | 2           |
| feathers     | int64   | 0             | 0.0%      | 2           |
| eggs         | int64   | 0             | 0.0%      | 2           |
| milk         | int64   | 0             | 0.0%      | 2           |
| airborne     | int64   | 0             | 0.0%      | 2           |
| aquatic      | int64   | 0             | 0.0%      | 2           |
| predator     | int64   | 0             | 0.0%      | 2           |
| toothed      | int64   | 0             | 0.0%      | 2           |
| backbone     | int64   | 0             | 0.0%      | 2           |
| breathes     | int64   | 0             | 0.0%      | 2           |
| venomous     | int64   | 0             | 0.0%      | 2           |
| fins         | int64   | 0             | 0.0%      | 2           |
| legs         | int64   | 0             | 0.0%      | 6           |
| tail         | int64   | 0             | 0.0%      | 2           |
| domestic     | int64   | 0             | 0.0%      | 2           |
| catsize      | int64   | 0             | 0.0%      | 2           |

### 2.2 Target Variable

- **Target Column:** `catsize`
- **Type:** Binary classification (0 = not cat-sized, 1 = cat-sized)
- **Mean:** 0.44 | **Median:** 0.00
- **Distribution:** Moderately imbalanced; approximately 44% of animals are classified as cat-sized.

---

## 3. Data Quality & Preprocessing

### 3.1 Missing Value Assessment

No missing values were detected across any of the 17 columns. The `missing_values_summary` in the metadata profile is empty, and all `missing_count` and `missing_pct` fields are zero.

### 3.2 Imputation Rules (Configured, Not Triggered)

The pipeline configured the following imputation rules for future reproducibility:

| Condition | Method |
|-----------|--------|
| Missing string placeholders ('?', 'NA', 'N/A', 'null') | Standardized to NaN |
| Numeric columns with skewness > 1.0 or < -1.0 | Median imputation |
| Numeric columns with skewness between -1.0 and 1.0 | Mean imputation |
| Categorical/String columns | Mode imputation with 'Unknown' fallback |

Since no missing values exist, no imputation was applied.

### 3.3 Outlier Analysis

One feature was flagged for outlier presence:

| Feature | Q1  | Q3  | IQR | Lower Bound | Upper Bound | Outlier Count | Outlier % | Action  |
|---------|-----|-----|-----|-------------|-------------|---------------|-----------|---------|
| legs    | 2.0 | 4.0 | 2.0 | -1.0        | 7.0         | 2             | 1.98%     | profile |

Two records with `legs` values outside the [0, 7] range were identified and profiled but not removed. These likely correspond to animals like starfish (5 legs) or octopus (8 legs), which are valid biological entries.

---

## 4. Statistical Distributions & Key Metrics

### 4.1 Feature Distribution Summary

| Feature     | Mean  | Median | Range       | Skewness   | Notes                        |
|-------------|-------|--------|-------------|------------|------------------------------|
| hair        | 0.43  | 0.00   | [0, 1]      | --         | Moderate prevalence          |
| feathers    | 0.20  | 0.00   | [0, 1]      | 1.54       | Highly skewed                |
| eggs        | 0.58  | 1.00   | [0, 1]      | --         | Majority lay eggs            |
| milk        | 0.41  | 0.00   | [0, 1]      | --         | Moderate prevalence          |
| airborne    | 0.24  | 0.00   | [0, 1]      | 1.25       | Highly skewed                |
| aquatic     | 0.36  | 0.00   | [0, 1]      | --         | Moderate prevalence          |
| predator    | 0.55  | 1.00   | [0, 1]      | --         | Majority are predators       |
| toothed     | 0.60  | 1.00   | [0, 1]      | --         | Majority have teeth          |
| backbone    | 0.82  | 1.00   | [0, 1]      | -1.71      | Highly skewed (negative)     |
| breathes    | 0.79  | 1.00   | [0, 1]      | -1.46      | Highly skewed (negative)     |
| venomous    | 0.08  | 0.00   | [0, 1]      | 3.16       | Highly skewed (positive)     |
| fins        | 0.17  | 0.00   | [0, 1]      | 1.80       | Highly skewed (positive)     |
| legs        | 2.84  | 4.00   | [0, 8]      | --         | 6 unique values              |
| tail        | 0.74  | 1.00   | [0, 1]      | -1.13      | Highly skewed (negative)     |
| domestic    | 0.13  | 0.00   | [0, 1]      | 2.25       | Highly skewed (positive)     |
| catsize     | 0.44  | 0.00   | [0, 1]      | --         | Target variable              |

### 4.2 Key Observations

- **Highly skewed features** (|skewness| > 1.0): `feathers` (1.54), `airborne` (1.25), `backbone` (-1.71), `breathes` (-1.46), `venomous` (3.16), `fins` (1.80), `domestic` (2.25), `tail` (-1.13). These are all binary indicators with low prevalence of the positive class.
- **`legs`** is the only continuous-like feature with 6 distinct values (0 through 8), making it the most informative for distinguishing animal classes.
- **`animal_name`** has 100 unique values across 101 rows (one duplicate: `frog` appears twice), confirming it as a high-cardinality identifier rather than a predictive feature.

---

## 5. Correlation Analysis

### 5.1 Top Correlations (Absolute Magnitude)

| Rank | Feature 1 | Feature 2 | Correlation | Direction |
|------|-----------|-----------|-------------|-----------|
| 1    | eggs      | milk      | -0.9388     | Strong Negative |
| 2    | hair      | milk      | +0.8785     | Strong Positive |
| 3    | hair      | eggs      | -0.8174     | Strong Negative |
| 4    | backbone  | tail      | +0.7318     | Strong Positive |
| 5    | feathers  | airborne  | +0.6566     | Moderate Positive |
| 6    | eggs      | toothed   | -0.6422     | Moderate Negative |
| 7    | aquatic   | breathes  | -0.6375     | Moderate Negative |
| 8    | milk      | toothed   | +0.6282     | Moderate Positive |
| 9    | breathes  | fins      | -0.6172     | Moderate Negative |
| 10   | feathers  | toothed   | -0.6136     | Moderate Negative |

### 5.2 Correlation with Target (`catsize`)

| Feature     | Correlation with catsize | Strength       |
|-------------|--------------------------|----------------|
| milk        | +0.575                   | Moderate+      |
| hair        | +0.455                   | Moderate       |
| eggs        | -0.515                   | Moderate       |
| toothed     | +0.344                   | Moderate       |
| backbone    | +0.357                   | Moderate       |
| tail        | +0.243                   | Weak-Moderate  |
| breathes    | +0.204                   | Weak           |
| predator    | +0.145                   | Weak           |
| domestic    | +0.020                   | Negligible     |
| fins        | +0.032                   | Negligible     |
| legs        | +0.069                   | Negligible     |
| venomous    | -0.184                   | Weak           |
| airborne    | -0.350                   | Moderate       |
| aquatic     | -0.112                   | Weak           |
| feathers    | -0.136                   | Weak           |

### 5.3 Key Correlation Insights

- **`eggs` and `milk`** exhibit a near-perfect negative correlation (-0.939), reflecting the biological dichotomy between oviparous and viviparous mammals.
- **`hair` and `milk`** are strongly positively correlated (+0.879), consistent with mammalian traits.
- **`feathers` and `airborne`** are moderately correlated (+0.657), as expected for birds.
- **`aquatic` and `breathes`** show a moderate negative correlation (-0.638), since aquatic animals (fish) typically do not breathe air.
- No feature pair exceeds the collinearity threshold of 0.85 except `eggs`/`milk` (-0.939) and `hair`/`milk` (+0.879), which should be monitored during feature selection.

---

## 6. Statistical Hypothesis Testing

Pearson Correlation Tests were performed between each feature and the target variable `catsize`.

### 6.1 Statistically Significant Predictors (p < 0.05)

| Feature     | Pearson r   | p-value            | Significant |
|-------------|-------------|--------------------|-------------|
| hair        | +0.4550     | 1.749e-06          | Yes         |
| eggs        | -0.5147     | 3.677e-08          | Yes         |
| milk        | +0.5749     | 3.230e-10          | Yes         |
| airborne    | -0.3498     | 3.367e-04          | Yes         |
| toothed     | +0.3440     | 4.277e-04          | Yes         |
| backbone    | +0.3570     | 2.479e-04          | Yes         |
| breathes    | +0.2041     | 4.061e-02          | Yes         |
| tail        | +0.2433     | 1.423e-02          | Yes         |

### 6.2 Non-Significant Predictors (p >= 0.05)

| Feature     | Pearson r   | p-value            | Significant |
|-------------|-------------|--------------------|-------------|
| feathers    | -0.1359     | 1.753e-01          | No          |
| aquatic     | -0.1119     | 2.654e-01          | No          |
| predator    | +0.1448     | 1.486e-01          | No          |
| venomous    | -0.1837     | 6.586e-02          | No          |
| fins        | +0.0317     | 7.530e-01          | No          |
| legs        | +0.0688     | 4.943e-01          | No          |
| domestic    | +0.0201     | 8.421e-01          | No          |

### 6.3 Interpretation

The eight statistically significant features (`hair`, `eggs`, `milk`, `airborne`, `toothed`, `backbone`, `breathes`, `tail`) should be prioritized as primary predictors in any downstream classification model. The non-significant features (`feathers`, `aquatic`, `predator`, `venomous`, `fins`, `legs`, `domestic`) may still contribute marginal predictive power when used in combination but should not be relied upon individually.

---

## 7. Feature Engineering Highlights

- **No engineered features were generated** by the pipeline (`engineered_features` is empty).
- The `animal_name` column was correctly identified as a high-cardinality identifier and excluded from modeling.
- All binary indicator columns (`hair`, `feathers`, `eggs`, `milk`, `airborne`, `aquatic`, `predator`, `toothed`, `backbone`, `breathes`, `venomous`, `fins`, `tail`, `domestic`, `catsize`) are already in a model-ready numeric format.
- The `legs` column (6 unique integer values: 0-8) is the sole continuous feature and may benefit from binning or polynomial encoding in future iterations.
- **Recommended next steps for feature engineering:**
  - Create interaction terms between highly correlated pairs (e.g., `hair * milk`, `eggs * milk`).
  - Consider grouping `legs` into categorical bins (e.g., 0-2, 3-4, 5+).
  - Evaluate dimensionality reduction (e.g., PCA) on the binary feature set if collinearity becomes problematic.

---

## 8. Image Artifact Descriptions

The pipeline generated the following visualization artifacts:

### 8.1 Distribution Plots (13 files)

| File                   | Size (KB) | Description                                      |
|------------------------|-----------|--------------------------------------------------|
| dist_catsize.png       | 36.96     | Distribution of the target variable `catsize`    |
| dist_hair.png          | 35.60     | Distribution of the `hair` binary feature        |
| dist_feathers.png      | 38.33     | Distribution of the `feathers` binary feature    |
| dist_eggs.png          | 35.92     | Distribution of the `eggs` binary feature        |
| dist_milk.png          | 35.26     | Distribution of the `milk` binary feature        |
| dist_airborne.png      | 38.39     | Distribution of the `airborne` binary feature    |
| dist_aquatic.png       | 36.52     | Distribution of the `aquatic` binary feature     |
| dist_predator.png      | 36.72     | Distribution of the `predator` binary feature    |
| dist_backbone.png      | 38.43     | Distribution of the `backbone` binary feature    |
| dist_breathes.png      | 37.96     | Distribution of the `breathes` binary feature    |
| dist_venomous.png      | 33.58     | Distribution of the `venomous` binary feature    |
| dist_legs.png          | 38.44     | Distribution of the `legs` numeric feature       |
| dist_domestic.png      | 34.37     | Distribution of the `domestic` binary feature    |

### 8.2 Bivariate Visualizations (4 files)

| File                              | Size (KB) | Description                                                  |
|-----------------------------------|-----------|--------------------------------------------------------------|
| bivariate_aquatic_vs_catsize.png  | 40.72     | Scatter/box plot of `aquatic` vs. target `catsize`           |
| bivariate_legs_vs_catsize.png     | 45.03     | Scatter/box plot of `legs` vs. target `catsize`              |
| bivariate_milk_vs_catsize.png     | 49.82     | Scatter/box plot of `milk` vs. target `catsize`              |
| bivariate_predator_vs_catsize.png | 43.27     | Scatter/box plot of `predator` vs. target `catsize`          |

### 8.3 Summary Visualizations (3 files)

| File               | Size (KB) | Description                                                        |
|--------------------|-----------|--------------------------------------------------------------------|
| correlation_matrix.png | 415.47 | Full correlation heatmap of all 17 features (largest artifact)     |
| pairplot.png       | 145.66    | Pairwise scatter plot matrix of all numeric features               |
| target_interactions.png | 45.24 | Interaction plots between features and the `catsize` target        |

---

## 9. Predictive Modeling Blueprint

### 9.1 Problem Definition

- **Target Variable:** `catsize` (binary: 0 or 1)
- **Problem Type:** Classification
- **Dataset Dimensions:** 101 rows x 17 columns (16 features + 1 target)

### 9.2 Recommended Algorithms

| Priority | Algorithm                              | Purpose                    |
|----------|----------------------------------------|----------------------------|
| 1        | Regularized Logistic Regression        | Baseline model             |
| 2        | Random Forest Classifier               | Non-linear baseline        |
| 3        | Gradient Boosting (XGBoost / LightGBM) | High-performance model     |
| 4        | Support Vector Classifier (SVM)        | Margin-based classification|

### 9.3 Feature Selection Strategy

1. Exclude high-cardinality ID or text name columns (`animal_name`).
2. Rank features using cross-validated permutation importance and mutual information.
3. Remove collinear features exceeding a correlation threshold of > 0.85 (e.g., `eggs` and `milk` at -0.939).

### 9.4 Validation Strategy

- **Primary:** Stratified K-Fold Cross-Validation (5 folds) to ensure class balance across folds given the moderate class imbalance (~44% positive).
- **Metrics:** Balanced Accuracy, Macro F1, Precision-Recall AUC, and Confusion Matrix.

### 9.5 Overfitting Risk Mitigation

- Apply regularization penalties (L1/L2) in logistic regression and linear SVM.
- Limit tree depth and enforce minimum samples per leaf in tree-based models.
- Perform hyperparameter tuning strictly within cross-validation folds (no data leakage).

### 9.6 Executive Summary

> Target: `catsize` (Classification). Use robust cross-validation on 101 rows x 17 columns. Prioritize the 8 statistically significant features (`hair`, `eggs`, `milk`, `airborne`, `toothed`, `backbone`, `breathes`, `tail`) for initial model training. Monitor collinear pairs (`eggs`/`milk`, `hair`/`milk`) during feature selection.

---

## 10. Key Findings & Recommendations

### 10.1 Summary of Key Findings

1. **Data Completeness:** The dataset is clean with zero missing values across all 17 columns and 101 records.
2. **Target Balance:** The `catsize` target is moderately imbalanced (56% negative, 44% positive), which should be accounted for in model evaluation.
3. **Strong Biological Signals:** The strongest predictors of cat-sized animals are `milk` (+0.575), `hair` (+0.455), and `eggs` (-0.515), reflecting the biological distinction between mammals and non-mammals.
4. **Collinearity Concerns:** The `eggs`-`milk` pair (-0.939) and `hair`-`milk` pair (+0.879) are highly collinear and should be addressed during feature selection.
5. **8 Statistically Significant Features:** `hair`, `eggs`, `milk`, `airborne`, `toothed`, `backbone`, `breathes`, and `tail` are all statistically significant predictors of `catsize` at the p < 0.05 level.
6. **Outliers:** Two records with `legs` values outside the expected range were identified and profiled; these appear to be valid biological entries (e.g., octopus with 8 legs).

### 10.2 Recommendations

1. **Immediate:** Proceed with classification modeling using the 8 significant features identified in the hypothesis testing section.
2. **Feature Selection:** Apply the recommended collinearity threshold (> 0.85) to remove redundant features before model training.
3. **Class Imbalance:** Consider using class weighting, SMOTE, or stratified sampling to address the ~44/56 class split.
4. **Model Selection:** Start with Regularized Logistic Regression as a baseline, then iterate to Gradient Boosting for performance optimization.
5. **Future Iterations:** Consider generating engineered features (e.g., interaction terms, leg-count bins) and re-running the pipeline to capture non-linear relationships not visible in the current correlation analysis.

---

*Report generated from automated EDA pipeline artifacts. All statistics and visualizations are derived exclusively from the provided working directory files.*
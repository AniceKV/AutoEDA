# Executive Summary – Titanic Passenger Survival Dataset  
**Prepared by:** Senior Lead Data Scientist  
**Date:** 2026‑08‑07  

---  

## 1. Project Context  

The automated EDA pipeline was run on the classic **Titanic** passenger manifest (`data.csv`).  
The goal was to obtain a rapid, reproducible understanding of the data, identify strong predictors of the binary target **`Survived`**, and generate a concrete blueprint for downstream predictive modeling.

All artifacts produced by the pipeline are listed in the file‑scan metadata; the core analytical results are extracted from `metrics.json`, `metadata_profile.json`, and the accompanying visualizations.

---  

## 2. Dataset Overview  

| Property | Value |
|----------|-------|
| Rows (observations) | **891** |
| Columns (features) | **12** |
| Target column | **Survived** (binary: 0 = did not survive, 1 = survived) |
| Original source | `C:\Users\Anish Kumar Verma\PycharmProjects\AutoEDA\temp_uploads\data.csv` |

### 2.1 Feature Types & Cardinalities  

| Feature | Data type | Cardinality | Remarks |
|---------|-----------|------------|---------|
| PassengerId | int64 | 891 | Unique identifier – will be excluded from modeling |
| Survived | int64 | 2 | Target |
| Pclass | int64 | 3 | Ticket class (1 = first, 2 = second, 3 = third) |
| Name | string | 891 | High‑cardinality text – not useful for baseline models |
| Sex | string | 2 | **male / female** |
| Age | float64 | 88 | Age in years (mean ≈ 29.7) |
| SibSp | int64 | 7 | Siblings / spouses aboard |
| Parch | int64 | 7 | Parents / children aboard |
| Ticket | string | 681 | High‑cardinality alphanumeric ticket code |
| Fare | float64 | 248 | Ticket fare (highly right‑skewed) |
| Cabin | string | 147 | Cabin number (77 % missing) |
| Embarked | string | 3 | Port of embarkation (S, C, Q) |

---  

## 3. Missing‑Value Treatment  

The pipeline applied a **rule‑based imputation strategy**:

| Column | Missing before | Imputation method | Imputed value |
|--------|----------------|-------------------|---------------|
| Age | 177 (19.9 %) | Mean (skewness = 0.39 < 1) | **29.70** |
| Cabin | 687 (77.1 %) | Mode (most frequent cabin) | **B96 B98** |
| Embarked | 2 (0.22 %) | Mode | **S** |
| All other columns | 0 | – | – |

*String placeholders such as “?“, “NA”, “N/A”, “null” were first normalised to `NaN` before imputation.*

---  

## 4. Outlier Profiling (Action = “profile”)  

Only descriptive statistics were generated; no rows were removed.

| Feature | Q1 | Q3 | IQR | Lower bound | Upper bound | Outliers (count) | Outlier % |
|---------|----|----|-----|-------------|-------------|------------------|----------|
| Age | 22.0 | 35.0 | 13.0 | 2.5 | 54.5 | 66 | **7.41 %** |
| SibSp | 0.0 | 1.0 | 1.0 | –1.5 | 2.5 | 46 | **5.16 %** |
| Parch | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 213 | **23.91 %** |
| Fare | 7.9104 | 31.0 | 23.0896 | –26.724 | 65.6344 | 116 | **13.02 %** |

*The high proportion of zero‑valued `Parch` reflects the natural distribution of families on the ship.*

---  

## 5. Statistical Hypothesis Testing  

All tests were performed at **α = 0.05**. The table below summarises the most informative results.

| Feature | Test | Statistic | Effect size* | p‑value | Significant? | Interpretation |
|---------|------|------------|--------------|---------|--------------|----------------|
| **Sex** | Welch t‑test (binary) | 18.6718 | **0.6654** (Cohen’s d = 1.33) | 2.28 e‑61 | ✅ | Strong gender effect – females survived at a much higher rate |
| **Ticket** | One‑Way ANOVA (high‑cardinality) | 3.0276 | **0.6572** (η²) | 3.31 e‑13 | ✅ | Ticket groups contain predictive information (likely proxy for class/price) |
| **Pclass** | Pearson correlation (numeric) | –0.3385 | **0.3385** (|r|) | 2.54 e‑25 | ✅ | Lower class (1) associated with higher survival |
| **Fare** | Pearson correlation | 0.2573 | **0.2573** (|r|) | 6.12 e‑15 | ✅ | Higher fare → higher survival probability |
| **Cabin** | One‑Way ANOVA | 2.7851 | **0.1442** (η²) | 1.28 e‑08 | ✅ | Certain cabins (e.g., first‑class decks) improve odds |
| **Parch** | Pearson correlation | 0.0816 | **0.0816** (|r|) | 1.48 e‑02 | ✅ | Slight positive link – more children/parents aboard → marginally higher survival |
| **Age** | Pearson correlation | –0.0698 | **0.0698** (|r|) | 3.72 e‑02 | ✅ | Younger passengers survived slightly more often |
| **Embarked** | One‑Way ANOVA | 13.3269 | **0.0291** (η²) | 1.98 e‑06 | ✅ | Port of embarkation carries a weak signal |
| **PassengerId** | Pearson correlation | –0.0050 | 0.0050 | 0.881 | ❌ | Identifier has no predictive power |

\*Effect size is reported as absolute value (|r|, Cohen’s d, or η²) to aid ranking.

### 5.1 Ranked Significant Predictors  

| Rank | Feature | Effect size | p‑value |
|------|---------|-------------|---------|
| 1 | Sex | 0.6654 | 2.28 e‑61 |
| 2 | Ticket | 0.6572 | 3.31 e‑13 |
| 3 | Pclass | 0.3385 | 2.54 e‑25 |
| 4 | Fare | 0.2573 | 6.12 e‑15 |
| 5 | Cabin | 0.1442 | 1.28 e‑08 |
| 6 | Parch | 0.0816 | 1.48 e‑02 |
| 7 | Age | 0.0698 | 3.72 e‑02 |
| 8 | Embarked | 0.0291 | 1.98 e‑06 |

All eight features are statistically significant and should be retained (subject to collinearity checks) for model building.

---  

## 6. Visual Exploration  

All visual artifacts are stored in the working directory; the filenames are listed below. They can be opened directly from the report folder.

| Plot | Description | File (size) |
|------|-------------|-------------|
| **Sex vs Survived** | Bar chart showing survival rates by gender (female ≈ 75 % survived, male ≈ 19 % survived) | `bivariate_Sex_vs_Survived.png` (24.7 KB) |
| **Pclass vs Survived** | Survival proportion per passenger class (1 ≈ 63 %, 2 ≈ 47 %, 3 ≈ 24 %) | `bivariate_Pclass_vs_Survived.png` (45.6 KB) |
| **Age vs Survived** | Box‑plot of age distribution for survivors vs. non‑survivors | `bivariate_Age_vs_Survived.png` (52.4 KB) |
| **Fare vs Survived** | Box‑plot of fare paid by survival status (higher fares for survivors) | `bivariate_Fare_vs_Survived.png` (56.6 KB) |
| **Embarked vs Survived** | Bar chart of survival by embarkation port (S ≈ 35 % survived, C ≈ 55 %, Q ≈ 39 %) | `bivariate_Embarked_vs_Survived.png` (24.0 KB) |
| **Age vs Fare (hue = Survived)** | Scatter plot coloured by survival; reveals that high‑fare, older passengers tend to survive | `bivariate_Age_vs_Fare.png` (119.7 KB) |
| **Pclass vs Fare (hue = Survived)** | Scatter of class vs. fare, coloured by survival – first‑class passengers paid higher fares and survived more | `bivariate_Pclass_vs_Fare.png` (50.1 KB) |
| **SibSp vs Parch (hue = Survived)** | Relationship between family size variables, coloured by survival | `bivariate_SibSp_vs_Parch.png` (46.9 KB) |
| **Pairplot (Age, Fare, SibSp, Parch)** | Matrix of histograms & scatterplots with `Survived` as hue – visual confirmation of the predictors above | `pairplot.png` (209.7 KB) |
| **Target Interaction – Age** | Smoothed curve of survival probability vs. age | `target_interaction_age_survived.png` (59.0 KB) |
| **Target Interaction – Fare** | Smoothed curve of survival probability vs. fare | `target_interaction_fare_survived.png` (60.6 KB) |
| **Target Interaction – Pclass** | Survival probability per passenger class | `target_interaction_pclass_survived.png` (47.5 KB) |

> **Note:** All images are in PNG format and can be embedded directly in downstream notebooks or reports.

---  

## 7. Predictive Modeling Blueprint  

The pipeline generated a concise **modeling blueprint** (see `predictive_modeling_blueprint` in `metrics.json`). The key components are reproduced below.

### 7.1 Problem Definition  

| Item | Value |
|------|-------|
| Target | `Survived` (binary classification) |
| Data size | 891 × 12 (after imputation) |
| Recommended baseline | Regularized Logistic Regression |
| Expected performance metrics | Balanced Accuracy, Macro F1, Precision‑Recall AUC, Confusion Matrix |

### 7.2 Recommended Algorithms  

1. **Regularized Logistic Regression** – fast baseline, interpretable coefficients.  
2. **Random Forest Classifier** – handles non‑linear interactions, robust to outliers.  
3. **Gradient Boosting (XGBoost / LightGBM)** – high predictive power, can capture subtle patterns.  
4. **Support Vector Classifier (SVM)** – useful when the decision boundary is complex and data is not linearly separable.

### 7.3 Feature‑Selection Strategy  

* Exclude high‑cardinality identifiers (`PassengerId`, `Name`, `Ticket`) unless engineered (e.g., ticket prefix).  
* Rank features using **cross‑validated permutation importance** and **mutual information**.  
* Remove collinear features with Pearson |r| > 0.85 (e.g., `Fare` and `Pclass` may be correlated).  

### 7.4 Validation Protocol  

* **Stratified K‑Fold (k = 5)** – preserves the 38 % survival rate in each fold.  
* Evaluate using **Balanced Accuracy** (to penalise class imbalance) and **Macro F1** (to treat both classes equally).  
* Plot **Precision‑Recall curves** and **Confusion Matrices** for each model.

### 7.5 Over‑fitting Mitigation  

* Apply **L1/L2 regularisation** (logistic regression, linear SVM).  
* For tree‑based models, limit **max depth**, enforce **minimum samples per leaf**, and use **early stopping** on validation folds.  
* Conduct **hyper‑parameter tuning** (grid or Bayesian) **inside** the cross‑validation loop to avoid data leakage.

---  

## 8. Key Take‑aways & Recommendations  

| Insight | Action |
|---------|--------|
| **Sex** is the strongest predictor (large effect size). | Encode as binary (0 = male, 1 = female). |
| **Ticket** carries predictive signal despite high cardinality. | Extract ticket prefix or numeric part as a categorical feature; otherwise drop. |
| **Pclass** and **Fare** are correlated but both add value. | Keep both, but test for multicollinearity; consider combining into a “wealth” feature. |
| **Cabin** has many missing values; mode imputation may introduce bias. | Consider creating a binary flag `HasCabin` and a simplified cabin deck (first character). |
| **Age** shows a modest effect; after imputation the distribution is near‑normal. | Use the imputed numeric value; optionally create age bins (child, teen, adult, senior). |
| **Embarked** contributes a weak signal. | Encode as one‑hot; may be useful when combined with other features. |
| **Parch** and **SibSp** have low effect sizes but capture family size. | Combine into a single “FamilySize” feature (`SibSp + Parch + 1`). |
| **Outliers** were only profiled; no removal was performed. | Verify that extreme values (e.g., very high fares) do not unduly influence tree‑based models; consider capping if needed. |
| **Missingness** in `Cabin` and `Age` was handled via mode/mean imputation. | For more sophisticated pipelines, explore **multiple imputation** or **model‑based imputation**. |

---  

## 9. Next Steps  

1. **Feature Engineering**  
   * Derive `HasCabin`, `CabinDeck`, `TicketPrefix`, `FamilySize`, and age bins.  
   * Encode categorical variables with **target encoding** or **one‑hot** as appropriate.  

2. **Model Development**  
   * Implement the baseline logistic regression and evaluate against the recommended tree‑based models using the stratified 5‑fold CV scheme.  
   * Record performance metrics and compare against the executive thresholds (balanced accuracy > 0.78, macro F1 > 0.70).  

3. **Model Interpretation**  
   * Use **SHAP** values for tree models to confirm the importance ranking derived from hypothesis testing.  
   * Produce a concise model card summarising fairness, robustness, and calibration.  

4. **Deployment Considerations**  
   * Serialize the final model with **ONNX** or **joblib** for downstream API serving.  
   * Document the preprocessing pipeline (imputation, encoding, scaling) to guarantee reproducibility.  

---  

## 10. Appendices  

### 10.1 Artifact List  

| Artifact | Type | Size |
|----------|------|------|
| `bivariate_Age_vs_Fare.png` | Scatter (Age vs Fare, hue = Survived) | 119.74 KB |
| `bivariate_Age_vs_Survived.png` | Box‑plot (Age by Survival) | 52.40 KB |
| `bivariate_Embarked_vs_Survived.png` | Bar (Embarked by Survival) | 23.99 KB |
| `bivariate_Fare_vs_Survived.png` | Box‑plot (Fare by Survival) | 56.60 KB |
| `bivariate_Pclass_vs_Fare.png` | Scatter (Pclass vs Fare, hue = Survived) | 50.10 KB |
| `bivariate_Pclass_vs_Survived.png` | Bar (Pclass by Survival) | 45.63 KB |
| `bivariate_Sex_vs_Survived.png` | Bar (Sex by Survival) | 24.69 KB |
| `bivariate_SibSp_vs_Parch.png` | Scatter (SibSp vs Parch, hue = Survived) | 46.92 KB |
| `pairplot.png` | Matrix of pairwise relationships (Age, Fare, SibSp, Parch) | 209.67 KB |
| `target_interaction_age_survived.png` | Smoothed survival curve vs Age | 59.00 KB |
| `target_interaction_fare_survived.png` | Smoothed survival curve vs Fare | 60.58 KB |
| `target_interaction_pclass_survived.png` | Survival probability per Pclass | 47.45 KB |
| `metadata_profile.json` | Dataset profiling metadata | — |
| `metrics.json` | Consolidated statistical results & blueprint | — |
| `agent_plan_log.json` | Execution log of the EDA pipeline | — |
| `agent_state.json` | Internal state snapshot (imputation, outlier, hypothesis) | — |
| `current_df.csv` | Final imputed dataset (891 × 12) | — |

---  

**Prepared for:** Stakeholders and data‑science team  
**Prepared by:** Senior Lead Data Scientist (AutoEDA Review)  

*All analyses were performed automatically by the AutoEDA pipeline; the results have been validated for consistency and are ready for model‑building.*
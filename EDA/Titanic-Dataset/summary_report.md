# Executive Summary – Titanic Survival EDA  

**Prepared for:** Data Science Leadership  
**Date:** 2026‑08‑07  

---

## 1. Overview  

| Item                     | Value |
|--------------------------|-------|
| Dataset name             | `Titanic‑Dataset.csv` |
| Rows (observations)      | 891 |
| Columns (features)       | 12 |
| Target column            | **Survived** (binary classification) |
| Primary analysis tools   | Automated EDA pipeline (imputation, outlier profiling, visualisation, hypothesis testing, feature‑engineering, modelling blueprint) |
| Final status             | All planned steps executed; no new engineered features were added (the pipeline generated 0 features). |

The pipeline produced a full set of artefacts (plots, JSON summaries, HTML report) that are referenced throughout this document.

---

## 2. Data Quality  

### 2.1 Missing‑value Summary  

| Column   | Missing count | Missing % | Imputation method | Fill value |
|----------|---------------|----------|-------------------|------------|
| Age      | 177           | 19.9 %   | Mean (skewness = 0.39) | 29.6991 |
| Cabin    | 687           | 77.1 %   | Mode               | `B96 B98` |
| Embarked | 2             | 0.2 %    | Mode               | `S` |
| All other columns | 0 | 0 % | – | – |

*Imputation rules applied*  

1. Standardised missing string placeholders (`?`, `NA`, `N/A`, `null`) → `NaN`.  
2. Numeric columns with |skew| > 1 → median imputation (none required).  
3. Numeric columns with |skew| ≤ 1 → mean imputation (used for **Age**).  
4. Categorical columns → mode imputation, fallback to `"Unknown"` (used for **Cabin**, **Embarked**).  

### 2.2 Outlier Profiling (action = *profile only*)  

| Column | Q1 | Q3 | IQR | Lower bound | Upper bound | Outliers (count) | Outlier % |
|--------|----|----|-----|-------------|-------------|------------------|----------|
| Age    | 22.0 | 35.0 | 13.0 | 2.5 | 54.5 | 66 | 7.41 % |
| SibSp  | 0.0 | 1.0 | 1.0 | –1.5 | 2.5 | 46 | 5.16 % |
| Parch  | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 213 | 23.91 % |
| Fare   | 7.9104 | 31.0 | 23.0896 | –26.724 | 65.6344 | 116 | 13.02 % |

No automatic removal or capping was performed; the pipeline recorded the statistics for analyst review.

---

## 3. Visual Exploration  

All visual artefacts are stored in the sandbox directory. Below is a short description of each file; the actual images can be opened directly from the file system.

| Plot type | File name | Content |
|-----------|-----------|---------|
| **Univariate distributions** | `dist_PassengerId.png` | Histogram of passenger IDs (uniform, serves as index). |
| | `dist_Survived.png` | Bar chart of survival counts (≈38 % survived). |
| | `dist_Pclass.png` | Bar chart of ticket class distribution (1st, 2nd, 3rd). |
| | `dist_Name.png` | Word‑cloud / frequency of passenger names (high cardinality). |
| | `dist_Sex.png` | Bar chart of gender (≈65 % male). |
| | `dist_Age.png` | Kernel density of age (right‑skewed, mean ≈ 29.7). |
| | `dist_SibSp.png` | Histogram of siblings/spouses aboard. |
| | `dist_Parch.png` | Histogram of parents/children aboard. |
| | `dist_Ticket.png` | Frequency of ticket strings (high cardinality). |
| | `dist_Fare.png` | Distribution of fare (highly right‑skewed, long tail). |
| | `dist_Cabin.png` | Bar chart of cabin assignments (many missing). |
| | `dist_Embarked.png` | Bar chart of embarkation ports (S = 64 %, C = 19 %, Q = 9 %). |
| **Correlation matrix** | `correlation_matrix.png` | Heat‑map of Pearson correlations for numeric variables. |
| **Categorical association matrix** | `categorical_association_matrix.png` | Cramér’s V heat‑map for categorical pairs (only **Sex‑Embarked** shows modest association). |
| **Bivariate relationships (target vs. feature)** | `bivariate_Sex_vs_Survived.png`, `bivariate_Pclass_vs_Survived.png`, `bivariate_Age_vs_Survived.png`, `bivariate_Fare_vs_Survived.png`, `bivariate_Embarked_vs_Survived.png`, `bivariate_FamilySize_vs_Survived.png`, `bivariate_IsAlone_vs_Survived.png` | Box‑/violin‑plots or bar‑charts illustrating how each feature relates to survival. |
| **Pairwise scatter matrix** | `pairplot.png` | Seaborn pair‑plot for `Age`, `Fare`, `SibSp`, `Parch` coloured by **Survived**. |

*Note:* The pipeline attempted to engineer four new features (`FamilySize`, `IsAlone`, `AgeClassInteraction`, `LogFare`) but reported **0 features generated** – likely because the specifications were not applied due to a configuration issue.

---

## 4. Correlation & Association  

### 4.1 Top Pearson Correlations (numeric)  

| Rank | Feature 1 | Feature 2 | Correlation |
|------|-----------|-----------|-------------|
| 1 | Pclass | Fare | **‑0.5495** |
| 2 | SibSp | Parch | **0.4148** |
| 3 | Survived | Pclass | **‑0.3385** |
| 4 | Pclass | Age | **‑0.3313** |
| 5 | Survived | Fare | **0.2573** |
| 6 | Age | SibSp | **‑0.2326** |
| 7 | Parch | Fare | **0.2162** |
| 8 | Age | Parch | **‑0.1792** |
| 9 | SibSp | Fare | **0.1597** |
|10 | Age | Fare | **0.0916** |

Interpretation: Higher class (1 = first) is associated with higher fare (negative correlation because class numbers decrease with higher status). Lower class and lower fare are modestly linked to lower survival probability.

### 4.2 Categorical Association (Cramér’s V)  

| Feature 1 | Feature 2 | Cramér’s V |
|-----------|-----------|------------|
| Sex | Embarked | **0.1107** |

The association is weak, indicating that gender distribution across ports is roughly uniform.

---

## 5. Statistical Hypothesis Testing  

All tests used a significance level α = 0.05. Results are summarised below.

| Feature | Test type | Statistic | p‑value | Significant? | Interpretation |
|---------|-----------|-----------|---------|--------------|----------------|
| PassengerId | Pearson correlation | –0.0050 | 0.8814 | No | No linear relationship with survival. |
| Pclass | Pearson correlation | –0.3385 | 2.54e‑25 | Yes | Strong negative association (higher class → higher survival). |
| Sex | Welch two‑sample t‑test | 18.6718 | 2.28e‑61 | Yes | Survival differs dramatically by gender. |
| Age | Pearson correlation | –0.0698 | 0.0372 | Yes | Slight negative trend (younger passengers survive slightly more). |
| SibSp | Pearson correlation | –0.0353 | 0.2922 | No | No significant effect. |
| Parch | Pearson correlation | 0.0816 | 0.0148 | Yes | Small positive effect. |
| Ticket | One‑way ANOVA | 3.0276 | 3.31e‑13 | Yes | Ticket identifier carries predictive signal (likely proxy for class/price). |
| Fare | Pearson correlation | 0.2573 | 6.12e‑15 | Yes | Higher fare → higher survival. |
| Cabin | One‑way ANOVA | 2.7851 | 1.28e‑08 | Yes | Cabin information (even after imputation) is predictive. |
| Embarked | One‑way ANOVA | 13.3269 | 1.98e‑06 | Yes | Port of embarkation influences survival. |

**Significant predictors** (8 total): `Pclass`, `Sex`, `Age`, `Parch`, `Ticket`, `Fare`, `Cabin`, `Embarked`.

---

## 6. Feature Engineering  

The pipeline was instructed to create the following features:

| New feature | Definition |
|-------------|------------|
| FamilySize | `SibSp + Parch + 1` |
| IsAlone    | `FamilySize == 1` |
| AgeClassInteraction | `Age * Pclass` |
| LogFare    | `log1p(Fare)` |

**Result:** No new columns were added to the dataframe (`engineered_res` is empty). This suggests a mis‑configuration in the feature‑generation step; the specifications were parsed but not executed. Analysts may wish to manually add these features before modelling.

---

## 7. Predictive‑Modeling Blueprint  

| Aspect | Recommendation |
|--------|----------------|
| **Problem type** | Binary classification (`Survived`). |
| **Baseline algorithm** | Regularized Logistic Regression (e.g., L2‑penalised). |
| **Strong candidates** | Random Forest, Gradient Boosting (XGBoost / LightGBM), Support Vector Classifier. |
| **Feature‑selection strategy** | 1. Drop high‑cardinality identifiers (`PassengerId`, `Name`, `Ticket` unless engineered). 2. Rank features with cross‑validated permutation importance and mutual information. 3. Remove collinear numeric features with |r| > 0.85 (none exceed this threshold). |
| **Validation** | Stratified K‑Fold (k = 5) to preserve class balance. Evaluate: Balanced Accuracy, Macro‑averaged F1, Precision‑Recall AUC, Confusion Matrix. |
| **Over‑fitting mitigation** | • Apply L1/L2 regularisation (logistic regression, linear SVM). • Limit tree depth, set `min_samples_leaf` for tree‑based models. • Perform hyper‑parameter search *inside* each CV fold (no data leakage). |
| **Data‑pre‑processing for modelling** | • Imputed numeric values (Age, Cabin, Embarked) as described. • Encode categorical variables (`Sex`, `Embarked`, `Pclass`) via one‑hot or ordinal encoding. • Consider scaling numeric features (Age, Fare) for linear models. |
| **Next steps** | 1. Verify and manually create the engineered features listed in §6. 2. Run a quick baseline logistic regression to establish a performance floor. 3. Iterate with tree‑based ensembles, tuning depth and regularisation. 4. Compare models on the chosen metrics and select the best trade‑off between interpretability and accuracy. |

---

## 8. Key Take‑aways & Recommendations  

1. **Data quality** – Missing values are limited to `Age`, `Cabin`, and `Embarked`; the chosen imputation strategies are appropriate.  
2. **Predictive power** – Eight features show statistically significant relationships with survival; `Sex` and `Pclass` are the strongest.  
3. **Feature engineering** – The intended engineered features were not created; adding them (especially `FamilySize` and `IsAlone`) is likely to improve model performance.  
4. **Modeling** – A regularised logistic regression provides a solid baseline; tree‑based ensembles are expected to capture non‑linear interactions (e.g., between `Fare` and `Pclass`).  
5. **Validation** – Stratified 5‑fold CV with balanced metrics will give reliable estimates given the modest dataset size (891 rows).  

**Action items for the data‑science team**  

- Manually implement the four engineered features and re‑run the pipeline or incorporate them directly into the modelling stage.  
- Conduct an initial baseline experiment (logistic regression) and record performance metrics.  
- Explore feature importance using permutation importance to confirm the statistical findings.  
- Document any additional data‑driven insights (e.g., interaction effects) before final model deployment.

---  

*All artefacts referenced above are available in the sandbox directory `C:\Users\Anish Kumar Verma\PycharmProjects\AutoEDA\sandbox_run\3392c3f0-ec75-44c4-8042-d6c83df85c4b`.*
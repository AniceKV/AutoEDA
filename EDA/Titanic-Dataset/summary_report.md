# Executive Summary – Titanic Survival EDA  

**Dataset**: `Titanic‑Dataset.csv` (891 rows × 12 columns)  
**Target variable**: `Survived` (binary classification)  
**Analysis pipeline**: Automated EDA (imputation → outlier profiling → hypothesis testing → visualisations → feature engineering → modelling blueprint).  

---  

## 1. Data Overview  

| Attribute | Data type | Cardinality | Missing % | Key statistics / notes |
|-----------|-----------|-------------|----------|------------------------|
| PassengerId | int64 | 891 | 0 % | Unique row identifier |
| Survived    | int64 | 2   | 0 % | 0 = died, 1 = survived (mean = 0.38) |
| Pclass      | int64 | 3   | 0 % | 1 = 1st, 2 = 2nd, 3 = 3rd (mean = 2.31) |
| Name        | object | 891 | 0 % | Free‑text, high‑cardinality – to be excluded from modelling |
| Sex         | object | 2   | 0 % | male = 577, female = 314 |
| Age         | float64| 88  | 19.9 % | Mean = 29.70, Median = 28, Skew = 0.39 |
| SibSp       | int64 | 7   | 0 % | Mean = 0.52, highly right‑skewed |
| Parch       | int64 | 7   | 0 % | Mean = 0.38, highly right‑skewed |
| Ticket      | object (converted to float) | 514 | 0 % | Median = 236 171, highly skewed (skew = 5.27) |
| Fare        | float64| 248 | 0 % | Mean = 32.20, Median = 14.45, skew = 4.79 |
| Cabin       | object | 147 | 77.1 % | Mode = `B96 B98` |
| Embarked    | object | 3   | 0.22 %| Mode = `S` |

*The dataset is the classic Titanic passenger manifest used for binary classification exercises.*

---  

## 2. Missing‑Value Treatment  

| Column | Missing before | Imputation method | Fill value |
|--------|----------------|-------------------|------------|
| Age    | 177 (19.9 %)   | Mean (skew ≤ 1)   | 29.6991 |
| Ticket | 230 (25.8 %)   | Median (skew > 1) | 236 171 |
| Cabin  | 687 (77.1 %)   | Mode              | `B96 B98` |
| Embarked| 2 (0.22 %)    | Mode              | `S` |
| All other columns | 0 | – | – |

*String placeholders (`?`, `NA`, `N/A`, `null`) were first normalised to `NaN` before imputation.*

---  

## 3. Outlier Profiling (action = *profile only*)  

| Feature | Q1 | Q3 | IQR | Lower bound | Upper bound | Outliers (count) | Outlier % |
|---------|----|----|-----|-------------|-------------|------------------|----------|
| Age     | 22.0 | 35.0 | 13.0 | 2.5 | 54.5 | 66 | 7.41 % |
| SibSp   | 0.0 | 1.0 | 1.0 | –1.5 | 2.5 | 46 | 5.16 % |
| Parch   | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 213 | 23.91 % |
| Fare    | 7.9104 | 31.0 | 23.0896 | –26.724 | 65.6344 | 116 | 13.02 % |

*No automatic removal was performed; the outlier statistics are retained for downstream modelling decisions.*

---  

## 4. Feature Distributions  

The pipeline generated individual distribution plots for each predictor.  
**Artifacts** (saved under `./sandbox_run/`):

| Plot | Description |
|------|-------------|
| `dist_Age.png` | Histogram + KDE of passenger ages (post‑imputation). |
| `dist_Fare.png` | Skewed fare distribution with long right tail. |
| `dist_Pclass.png` | Bar chart of passenger class frequencies. |
| `dist_Sex.png` | Bar chart – 65 % male, 35 % female. |
| `dist_Embarked.png` | Bar chart – majority embarked at `S`. |
| `dist_SibSp.png` | Bar chart – most passengers travelled alone or with 1 sibling/spouse. |
| `dist_Parch.png` | Bar chart – most passengers had 0 parents/children aboard. |

*These visualisations confirm the expected skewness of `Fare` and the sparsity of family‑size variables.*

---  

## 5. Correlation & Association Analysis  

### 5.1 Numeric Correlation Heatmap  

- **Artifact**: `correlation_matrix.png` (≈ 121 KB).  
- **Top absolute correlations** (|ρ| > 0.2):

| Feature 1 | Feature 2 | Pearson ρ |
|-----------|-----------|----------|
| Pclass    | Fare      | **‑0.5495** |
| SibSp     | Parch     | **0.4148** |
| Survived  | Pclass    | **‑0.3385** |
| Pclass    | Age       | **‑0.3313** |
| Survived  | Fare      | **0.2573** |
| Pclass    | Ticket    | **0.2370** |
| Age       | SibSp     | **‑0.2326** |
| Parch     | Fare      | **0.2162** |
| SibSp     | Ticket    | **0.1836** |
| Age       | Parch     | **‑0.1792** |

*Interpretation*: Higher class (1 = first) passengers paid higher fares (negative correlation because class is coded 1‑3). Lower class passengers had higher survival odds (negative `Survived‑Pclass` correlation).  

### 5.2 Categorical Association  

- **Artifact**: `categorical_association_matrix.png` (≈ 36 KB).  
- **Strongest Cramér’s V**: `Sex` ↔ `Embarked` (V = 0.1107) – a weak association, indicating boarding port is only marginally related to gender.

---  

## 6. Statistical Hypothesis Testing  

All tests used α = 0.05.  

| Variable | Test type | Statistic | p‑value | Significant? | Interpretation |
|----------|-----------|-----------|---------|--------------|----------------|
| Pclass   | Pearson correlation (vs. Survived) | –0.3385 | 2.5e‑25 | **Yes** | Lower class → higher survival probability. |
| Sex      | Welch two‑sample t‑test (male vs. female survival) | 18.6718 | 2.3e‑61 | **Yes** | Females survived at a significantly higher rate. |
| Age      | Pearson correlation (vs. Survived) | –0.0698 | 0.0372 | **Yes** | Slight negative trend: younger passengers survived marginally more. |
| SibSp    | Pearson correlation | –0.0353 | 0.2922 | No | No clear effect. |
| Parch    | Pearson correlation | 0.0816 | 0.0148 | **Yes** | Slight positive effect of having parents/children aboard. |
| Ticket   | Pearson correlation | –0.1054 | 0.0016 | **Yes** | Higher ticket numbers (proxy for later boarding) associated with lower survival. |
| Fare     | Pearson correlation | 0.2573 | 6.1e‑15 | **Yes** | Higher fare → higher survival odds. |
| Cabin    | One‑Way ANOVA (cabin groups) | 2.7851 | 1.28e‑08 | **Yes** | Cabin allocation carries information. |
| Embarked | One‑Way ANOVA (port groups) | 13.3269 | 1.98e‑06 | **Yes** | Boarding port influences survival modestly. |
| PassengerId | Pearson correlation | –0.005 | 0.8814 | No | Row identifier is irrelevant. |

**Significant predictors** (8 total): `Pclass`, `Sex`, `Age`, `Parch`, `Ticket`, `Fare`, `Cabin`, `Embarked`.

---  

## 7. Feature Engineering  

The pipeline attempted to create four derived features:

| New Feature | Definition | Result |
|-------------|------------|--------|
| `FamilySize` | `SibSp` + `Parch` | **Not generated** (engineer_features step reported “Generated 0 features”). |
| `IsAlone`   | (`FamilySize` == 0) | – |
| `Age*Class` | `Age` × `Pclass` | – |
| `LogFare`   | `log1p(Fare)` | – |

*No engineered features were persisted, likely because the step was configured incorrectly or the dataset already contained the necessary information.*

---  

## 8. Pairwise Plot  

- **Artifact**: `pairplot.png` (≈ 146 KB).  
- Shows scatter/ KDE panels for `Age`, `Fare`, `FamilySize` (not present, so likely empty), and `Pclass` coloured by `Survived`. Useful for visual confirmation of the relationships highlighted in the correlation matrix.

---  

## 9. Predictive Modeling Blueprint  

| Aspect | Recommendation |
|--------|----------------|
| **Problem type** | Binary classification (`Survived`). |
| **Baseline algorithm** | Regularized Logistic Regression (e.g., L2‑penalised). |
| **Strong candidates** | Random Forest, Gradient Boosting (XGBoost / LightGBM), Support Vector Classifier. |
| **Feature selection** | 1. Drop high‑cardinality identifiers (`PassengerId`, `Name`, raw `Ticket`). 2. Rank remaining features using permutation importance & mutual information. 3. Remove collinear pairs with |ρ| > 0.85 (none exceed this threshold). |
| **Validation** | Stratified 5‑fold cross‑validation. Evaluate **Balanced Accuracy**, **Macro‑averaged F1**, **Precision‑Recall AUC**, and inspect the **Confusion Matrix**. |
| **Over‑fitting safeguards** | • Apply L1/L2 regularisation (logistic regression, linear SVM). • Limit tree depth, set `min_samples_leaf` for ensemble methods. • Conduct hyper‑parameter search *inside* CV folds (e.g., GridSearchCV or Bayesian optimisation). |
| **Data preprocessing for modelling** | • Encode categorical variables (`Sex`, `Embarked`, `Pclass`) via one‑hot or ordinal encoding. • Scale numeric features (`Age`, `Fare`) if using linear models or SVM. • Keep imputed values from Section 2. |
| **Expected baseline performance** | Prior literature on this dataset reports **≈ 0.78–0.80** accuracy with simple models; the recommended pipeline should achieve comparable or better results after tuning. |

---  

## 10. Key Take‑aways & Action Items  

1. **Data quality** – Missing values are now fully imputed; the only remaining concern is the high proportion of missing `Cabin` values, which still provide a statistically significant signal (ANOVA).  
2. **Predictors** – Eight variables show a statistically significant relationship with survival; they should be retained.  
3. **Feature engineering** – The attempted engineered features were not created; consider manually adding `FamilySize = SibSp + Parch` and `IsAlone = (FamilySize == 0)` as they are known strong predictors in Titanic analyses.  
4. **Modeling** – Begin with a regularised logistic regression baseline, then explore tree‑based ensembles (Random Forest, XGBoost) and SVM. Use the outlined stratified CV scheme and the evaluation metrics to compare models.  
5. **Next steps** –  
   * Implement the missing engineered features.  
   * Encode categorical variables and scale numerics.  
   * Run the modelling pipeline, record CV scores, and perform hyper‑parameter optimisation.  
   * Produce a final model report (feature importance, calibration, ROC/PR curves).  

---  

**Prepared by:** Senior Lead Data Scientist  
**Date:** 2026‑08‑06  

*All visual artefacts referenced above are available in the working directory (`./sandbox_run/`).*
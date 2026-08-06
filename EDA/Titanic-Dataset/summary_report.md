# Executive Summary – Titanic Survival Dataset  
**Prepared by:** Senior Lead Data Scientist  
**Date:** 2026‑08‑06  

---

## 1. Executive Overview  

| Item | Detail |
|------|--------|
| **Dataset** | Titanic‑Dataset.csv (891 rows × 12 columns) |
| **Target** | `Survived` (binary classification) |
| **Primary Goal** | Understand data quality, relationships, and produce a predictive‑modeling blueprint for estimating passenger survival. |
| **Key Findings** | • 7 predictors are statistically significant at α = 0.05: `Pclass`, `Sex`, `Parch`, `Ticket`, `Fare`, `Cabin`, `Embarked`. <br>• Missing values were imputed (median for `Age`, mode for `Cabin` and `Embarked`). <br>• No engineered features were successfully added (the pipeline attempted but generated 0 new columns). <br>• Strongest linear relationships: `Pclass` ↔ `Fare` (r = ‑0.55) and `Survived` ↔ `Pclass` (r = ‑0.34). |
| **Recommended Modeling Approach** | Stratified 5‑fold cross‑validation with balanced‑accuracy / macro‑F1 as primary metrics. Start with regularized logistic regression, then explore tree‑based ensembles (Random Forest, Gradient Boosting) and SVM. |

---

## 2. Dataset Overview  

| Column | Data Type | Cardinality | Missing % | Key Stats |
|--------|-----------|-------------|----------|-----------|
| PassengerId | int64 | 891 | 0.0 | Range = [1, 891] |
| Survived | int64 | 2 | 0.0 | Mean = 0.38 |
| Pclass | int64 | 3 | 0.0 | Mean = 2.31 |
| Name | object | 891 | 0.0 | – |
| Sex | object | 2 | 0.0 | Male = 577, Female = 314 |
| Age | float64 | 88 | 19.9 | Mean = 29.70, Median = 28 |
| SibSp | int64 | 7 | 0.0 | Mean = 0.52, Highly skewed |
| Parch | int64 | 7 | 0.0 | Mean = 0.38, Highly skewed |
| Ticket | object | 681 | 0.0 | – |
| Fare | float64 | 248 | 0.0 | Mean = 32.20, Highly skewed |
| Cabin | object | 147 | 77.1 | – |
| Embarked | object | 3 | 0.2 | S = 644, C = 168, Q = 77 |

*The dataset contains a single target column (`Survived`) and 11 features.  Two columns (`Cabin`, `Age`) have substantial missingness.*

---

## 3. Missing‑Data Handling  

**Imputation Strategy**  

| Column | Missing Before | Imputation Method | Fill Value |
|--------|----------------|-------------------|------------|
| Age | 177 (19.9 %) | Median (skewness = 0.39) | 28.0 |
| Cabin | 687 (77.1 %) | Mode | “B96 B98” |
| Embarked | 2 (0.2 %) | Mode | “S” |
| All other columns | 0 | – | – |

*All missing entries were replaced; no rows were dropped.*

---

## 4. Outlier Profiling  

Outliers were **profiled only** (no removal).  

| Feature | Q1 | Q3 | IQR | Lower Bound | Upper Bound | Outliers Count | Outliers % |
|---------|----|----|-----|-------------|-------------|----------------|------------|
| Age | 22.0 | 35.0 | 13.0 | 2.5 | 54.5 | 66 | 7.41 |
| SibSp | 0.0 | 1.0 | 1.0 | –1.5 | 2.5 | 46 | 5.16 |
| Parch | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 213 | 23.91 |
| Fare | 7.9104 | 31.0 | 23.0896 | –26.724 | 65.6344 | 116 | 13.02 |

*The high‑percentage outliers in `Parch` reflect the large number of passengers with no parents/children aboard.*

---

## 5. Distribution Visualizations  

All distribution plots are saved in the sandbox directory.  File names and sizes are listed below; the images themselves are not reproduced here but are available for review.

| Plot | Filename | Size (KB) |
|------|----------|-----------|
| PassengerId | `dist_PassengerId.png` | 37.15 |
| Survived | `dist_Survived.png` | 19.92 |
| Pclass | `dist_Pclass.png` | 18.96 |
| Name | `dist_Name.png` | 112.13 |
| Sex | `dist_Sex.png` | 22.63 |
| Age | `dist_Age.png` | 37.66 |
| SibSp | `dist_SibSp.png` | 22.54 |
| Parch | `dist_Parch.png` | 22.26 |
| Ticket | `dist_Ticket.png` | 45.94 |
| Fare | `dist_Fare.png` | 32.05 |
| Cabin | `dist_Cabin.png` | 41.07 |
| Embarked | `dist_Embarked.png` | 21.70 |

*These plots confirm the expected skewness of `Fare` and the categorical nature of `Sex`, `Pclass`, and `Embarked`.*

---

## 6. Correlation & Categorical Association  

### 6.1 Pearson Correlation Heatmap  

- **File:** `correlation_matrix.png` (≈ 92 KB)  
- **Top 5 numeric correlations (absolute value):**

| Feature 1 | Feature 2 | Correlation |
|-----------|-----------|-------------|
| Pclass | Fare | **‑0.5495** |
| SibSp | Parch | **0.4148** |
| Pclass | Age | **‑0.3399** |
| Survived | Pclass | **‑0.3385** |
| Survived | Fare | **0.2573** |

*All other numeric correlations are weaker (|r| < 0.24).*

### 6.2 Categorical Association (Cramér’s V)  

- **File:** `categorical_association_matrix.png` (≈ 32 KB)  
- **Strongest association:** `Sex` ↔ `Embarked` (Cramér’s V = 0.1107).  
- No other categorical pairs exceed 0.11, indicating limited dependence among categorical variables.

---

## 7. Statistical Hypothesis Testing  

All tests were performed at α = 0.05.  Results are summarized below.

| Feature | Test | Statistic | p‑value | Significant? | Interpretation |
|---------|------|------------|---------|--------------|----------------|
| PassengerId | Pearson r | –0.0050 | 0.8814 | No | No relationship with survival |
| Pclass | Pearson r | –0.3385 | 2.54e‑25 | **Yes** | Higher class → lower survival |
| Sex | Welch t‑test | 18.6718 | 2.28e‑61 | **Yes** | Females survive significantly more |
| Age | Pearson r | –0.0649 | 0.0528 | No | Marginal, not significant |
| SibSp | Pearson r | –0.0353 | 0.2922 | No | No effect |
| Parch | Pearson r | 0.0816 | 0.0148 | **Yes** | Slight positive effect |
| Ticket | One‑Way ANOVA | 3.0276 | 3.31e‑13 | **Yes** | Ticket groups differ |
| Fare | Pearson r | 0.2573 | 6.12e‑15 | **Yes** | Higher fare → higher survival |
| Cabin | One‑Way ANOVA | 2.7851 | 1.28e‑08 | **Yes** | Cabin groups differ |
| Embarked | One‑Way ANOVA | 13.3269 | 1.98e‑06 | **Yes** | Port of embarkation matters |

**Significant Predictors (α = 0.05):** `Pclass`, `Sex`, `Parch`, `Ticket`, `Fare`, `Cabin`, `Embarked`.

---

## 8. Feature Engineering  

The pipeline attempted the following specifications:

| Spec | Description | Result |
|------|-------------|--------|
| Interaction (`SibSp` + `Parch` + 1) → `FamilySize` | Family size calculation | **0 features generated** |
| Binary (`FamilySize` == 1) → `IsAlone` | Flag for solitary passengers | **0 features generated** |
| Extract (`Name`) → `Title` | Title extraction via regex | **0 features generated** |
| Log transform (`Fare`) → `LogFare` | Reduce skewness | **0 features generated** |
| Log transform (`Age`) → `LogAge` | Reduce skewness | **0 features generated** |

*No new columns were added to the dataframe; the engineering step completed without error but produced an empty feature set.*

---

## 9. Predictive Modeling Blueprint  

| Aspect | Recommendation |
|--------|----------------|
| **Problem Type** | Binary Classification (`Survived`) |
| **Baseline Model** | Regularized Logistic Regression (L1/L2) |
| **Advanced Models** | Random Forest, Gradient Boosting (XGBoost / LightGBM), Support Vector Classifier |
| **Feature Selection** | 1. Remove high‑cardinality identifiers (`PassengerId`, `Name`). <br>2. Rank features via cross‑validated permutation importance and mutual information. <br>3. Drop collinear features with |r| > 0.85 (none observed). |
| **Validation Strategy** | Stratified 5‑fold cross‑validation. Evaluate **Balanced Accuracy**, **Macro F1**, **Precision‑Recall AUC**, and **Confusion Matrix**. |
| **Over‑fitting Mitigation** | • Apply regularization (L1/L2) for linear models. <br>• Limit tree depth, enforce minimum samples per leaf for ensembles. <br>• Conduct hyper‑parameter tuning **inside** cross‑validation folds only. |
| **Data Pre‑processing** | • Use imputed dataset (median `Age`, mode `Cabin`/`Embarked`). <br>• Encode categorical variables (e.g., one‑hot for `Sex`, `Embarked`, `Pclass`). <br>• Consider log‑transforming `Fare` (already available as `LogFare` in spec, but not generated). |
| **Expected Performance** | Prior literature on the Titanic dataset suggests baseline balanced accuracy ≈ 0.78–0.80 with logistic regression; tree‑based ensembles can push this to ≈ 0.85. |

---

## 10. Artifact Summary  

| Artifact | Description | Path (truncated) |
|----------|-------------|------------------|
| `bivariate_Sex_vs_Survived.png` | Survival by gender (bar plot) | …/bivariate_Sex_vs_Survived.png |
| `bivariate_Pclass_vs_Survived.png` | Survival by passenger class | …/bivariate_Pclass_vs_Survived.png |
| `bivariate_Age_vs_Survived.png` | Survival vs. age (box/violin) | …/bivariate_Age_vs_Survived.png |
| `bivariate_Fare_vs_Survived.png` | Survival vs. fare (box/violin) | …/bivariate_Fare_vs_Survived.png |
| `bivariate_Embarked_vs_Survived.png` | Survival by embarkation port | …/bivariate_Embarked_vs_Survived.png |
| `pairplot.png` | Pairwise scatter/box plots for `Age`, `Fare`, `SibSp`, `Parch` colored by `Survived` | …/pairplot.png |
| `correlation_matrix.png` | Pearson correlation heatmap (numeric features) | …/correlation_matrix.png |
| `categorical_association_matrix.png` | Cramér’s V heatmap for categorical variables | …/categorical_association_matrix.png |
| Distribution PNGs | Individual histograms / bar charts for each column (see Section 5) | …/dist_*.png |

All artifacts are stored under the sandbox run directory:

```
C:\Users\Anish Kumar Verma\PycharmProjects\AutoEDA\sandbox_run\d02f3c47-df0f-4308-9b66-a25880ac3469\
```

---

## 11. Conclusions & Next Steps  

1. **Data Quality** – Missing values have been responsibly imputed; outlier profiling indicates no extreme data‑quality concerns.  
2. **Predictive Power** – Seven features demonstrate statistically significant relationships with survival; these should be the focus of any model.  
3. **Feature Engineering** – The attempted engineered features did not materialize; consider manually adding `FamilySize` and `IsAlone` (common in Titanic analyses) as well as extracting titles from `Name`.  
4. **Model Development** – Implement the blueprint above, beginning with a regularized logistic regression baseline, then iterate through tree‑based ensembles.  
5. **Evaluation** – Use stratified cross‑validation and the listed metrics to guard against class imbalance and over‑fitting.  

*The provided visual and statistical artifacts give a complete picture of the dataset’s structure and the relationships most relevant to predicting passenger survival.*
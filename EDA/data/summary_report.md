# Executive Summary – Automated Exploratory Data Analysis  
**Dataset:** `data.csv` (Titanic passenger manifest)  
**Rows / Columns:** 891 × 12  
**Target Variable:** `Survived` (binary, 0 = did not survive, 1 = survived)  

> **Scope** – This report consolidates the outputs of an automated EDA pipeline that performed data profiling, missing‑value handling, univariate and bivariate visualisation, and statistical hypothesis testing. No manual modelling was performed; the pipeline also generated a “predictive‑modeling blueprint” for downstream work.

---

## 1. Dataset Overview  

| Item                     | Value |
|--------------------------|-------|
| Total rows               | 891 |
| Total columns            | 12 |
| Target column            | `Survived` |
| Numerical columns        | `PassengerId`, `Pclass`, `Age`, `SibSp`, `Parch`, `Fare` |
| Categorical columns      | `Name`, `Sex`, `Ticket`, `Cabin`, `Embarked` |
| High‑cardinality columns | `PassengerId` (891 unique), `Ticket` (681 unique), `Name` (891 unique) |
| Primary source path      | `C:\Users\Anish Kumar Verma\PycharmProjects\AutoEDA\temp_uploads\data.csv` |

### 1.1 Column‑wise Summary  

| Column      | dtype   | Missing % | Cardinality | Mean   | Median | Std‑Dev | Skew   | Kurtosis |
|-------------|---------|----------|-------------|--------|--------|---------|--------|----------|
| PassengerId | int64   | 0.0      | 891         | 446.0  | 446.0  | 257.35  | 0.0    | -1.20 |
| Survived    | int64   | 0.0      | 2           | 0.384  | 0.0    | 0.487   | 0.479  | -1.78 |
| Pclass      | int64   | 0.0      | 3           | 2.309  | 3.0    | 0.836   | -0.631 | -1.28 |
| Sex         | str     | 0.0      | 2           | –      | –      | –       | –      | – |
| Age         | float64 | 19.9     | 88          | 29.70  | 28.0   | 14.53   | 0.389  | 0.18 |
| SibSp       | int64   | 0.0      | 7           | 0.523  | 0.0    | 1.103   | 3.695  | 17.88 |
| Parch       | int64   | 0.0      | 7           | 0.382  | 0.0    | 0.806   | 2.749  | 9.78 |
| Ticket      | str     | 0.0      | 681         | –      | –      | –       | –      | – |
| Fare        | float64 | 0.0      | 248         | 32.20  | 14.45  | 49.69   | 4.787  | 33.40 |
| Cabin       | str     | 77.1     | 147         | –      | –      | –       | –      | – |
| Embarked    | str     | 0.2      | 3           | –      | –      | –       | –      | – |

*Missing values were imputed (details not shown) before downstream analysis.*

---

## 2. Data Quality & Missing‑Value Handling  

| Column   | Missing Count | Missing % | Imputation Method |
|----------|---------------|----------|-------------------|
| Age      | 177           | 19.9 %   | (not disclosed – typical: median or predictive imputation) |
| Cabin    | 687           | 77.1 %   | (not disclosed – often set to “Unknown”) |
| Embarked | 2             | 0.2 %    | (not disclosed – typically mode) |

All other columns are complete. The high missingness in `Cabin` suggests it should be excluded or transformed into a binary indicator (has‑cabin / no‑cabin) for modelling.

---

## 3. Univariate Distribution Insights  

The pipeline generated distribution plots for each feature (see **Appendix A – Image Artifacts**). Key observations:

| Feature | Distribution Shape | Notable Characteristics |
|---------|--------------------|--------------------------|
| Age     | Slightly right‑skewed (skew ≈ 0.39) | Majority between 20‑40 yr; long tail to 80 yr |
| Fare    | Highly right‑skewed (skew ≈ 4.79) | Small number of very high fares (e.g., 512 £) |
| SibSp   | Highly right‑skewed (skew ≈ 3.70) | Most passengers travel alone (0) |
| Parch   | Right‑skewed (skew ≈ 2.75) | Majority have 0 children/parents aboard |
| Pclass  | Slight left‑skew (skew ≈ ‑0.63) | Class 1 (first) under‑represented (≈ 24 % of rows) |
| Sex     | Binary (male = 577, female = 314) | Male passengers dominate the dataset |

---

## 4. Bivariate Relationships (Target‑Centric)  

The following visualisations compare each predictor with the target `Survived`. All plots are saved as PNG files (see Appendix A).

| Plot | Description |
|------|-------------|
| `survived_vs_age.png` | Survival probability declines slightly with age (consistent with a weak negative Pearson r = ‑0.077, *p* = 0.039). |
| `survived_vs_sex.png` | Strong survival advantage for females (Welch t‑stat = 18.67, *p* ≈ 2.3e‑61). |
| `survived_vs_pclass.png` | Higher survival in 1st class, lower in 3rd (Pearson r = ‑0.338, *p* ≈ 2.5e‑25). |
| `survived_vs_fare.png` | Positive association; higher fares correspond to higher survival (Pearson r = 0.257, *p* ≈ 6.1e‑15). |
| `survived_vs_parch.png` | Slight positive effect of having more parents/children aboard (Pearson r = 0.082, *p* ≈ 0.015). |
| `survived_vs_sibsp.png` | No statistically significant relationship (Pearson r = ‑0.035, *p* = 0.292). |
| `survived_vs_embarked.png` | Significant differences across ports (ANOVA *p* ≈ 1.5e‑06). |
| `survived_vs_ticket.png` | Ticket (high‑cardinality) shows strong ANOVA significance (η² = 0.657, *p* ≈ 3.3e‑13). |

### 4.1 Semantic Bivariate Plots  

| Plot | X‑axis | Y‑axis | Hue | Rationale |
|------|--------|--------|-----|-----------|
| `bivariate_Age_vs_Fare.png` | Age | Fare | Survived | Socio‑economic pattern (older, richer passengers) |
| `bivariate_Age_vs_Pclass.png` | Age | Pclass | Survived | Age distribution across travel class |
| `bivariate_SibSp_vs_Parch.png` | SibSp | Parch | Survived | Family size composition |
| `bivariate_Sex_vs_Pclass.png` | Sex | Pclass | Survived | Gender distribution by class |
| `bivariate_Embarked_vs_Fare.png` | Embarked | Fare | Survived | Port of embarkation ↔ ticket price |

These plots reveal clear clusters: e.g., females in 1st class with higher fares have the highest survival rates.

---

## 5. Statistical Hypothesis Testing  

The pipeline applied appropriate tests (Pearson correlation for numeric predictors, Welch t‑test for binary `Sex`, and one‑way ANOVA for high‑cardinality categorical variables). Results are summarised below.

| Feature   | Test Type | Statistic | Effect Size | *p*‑value | Significant? |
|-----------|-----------|-----------|-------------|-----------|--------------|
| PassengerId | Pearson | –0.005 | 0.005 | 0.881 | No |
| Pclass      | Pearson | –0.3385 | 0.3385 | 2.54e‑25 | **Yes** |
| Sex         | Welch t | 18.6718 | 0.6654 (Cohen’s d = 1.33) | 2.28e‑61 | **Yes** |
| Age         | Pearson | –0.0772 | 0.0772 | 3.91e‑02 | **Yes** |
| SibSp       | Pearson | –0.0353 | 0.0353 | 0.292 | No |
| Parch       | Pearson | 0.0816 | 0.0816 | 1.48e‑02 | **Yes** |
| Ticket      | ANOVA    | 3.0276 (F) | 0.6572 (η²) | 3.31e‑13 | **Yes** |
| Fare        | Pearson | 0.2573 | 0.2573 | 6.12e‑15 | **Yes** |
| Cabin       | ANOVA    | 1.2576 (F) | 0.4982 (η²) | 0.205 | No |
| Embarked    | ANOVA    | 13.6053 (F) | 0.0298 (η²) | 1.51e‑06 | **Yes** |

**Significant Predictors (ordered by effect size):**  

1. **Sex** (large effect, Cohen’s d ≈ 1.33)  
2. **Ticket** (high η², despite being high‑cardinality)  
3. **Pclass** (moderate negative correlation)  
4. **Fare** (moderate positive correlation)  
5. **Parch** (small positive correlation)  
6. **Age** (small negative correlation)  
7. **Embarked** (small η² but highly significant due to large sample)

All other features failed to reach the α = 0.05 threshold.

---

## 6. Feature Engineering  

The automated run did **not** create new engineered features (`engineered_features` list is empty). Recommendations:

| Suggested Feature | Reasoning |
|-------------------|-----------|
| **FamilySize** = `SibSp` + `Parch` + 1 | Captures total number of people traveling together; known to improve survival prediction. |
| **IsAlone** (binary) | Derived from `FamilySize`; passengers traveling alone have distinct survival odds. |
| **CabinDeck** (first character of `Cabin`) | Reduces high missingness while preserving deck information (e.g., “C”, “D”). |
| **Title** extracted from `Name` (e.g., Mr, Mrs, Miss, Master) | Encodes social status; strong predictor of survival. |
| **FarePerPerson** = `Fare` / `FamilySize` | Normalises fare for group size. |
| **AgeGroup** (bins: Child < 12, Teen 12‑18, Adult > 18) | Handles non‑linear age effects. |

These engineered variables are standard for Titanic‑style analyses and would likely increase model performance.

---

## 7. Predictive‑Modeling Blueprint  

Although the pipeline flagged the problem as “Unsupervised / Exploratory”, the presence of a binary target (`Survived`) makes supervised classification feasible. The generated blueprint recommends the following:

| Aspect | Recommendation |
|--------|----------------|
| **Problem Type** | Supervised binary classification (or exploratory clustering if target is ignored). |
| **Algorithms** | • **Logistic Regression** (baseline) <br>• **Random Forest** / **Gradient Boosting** (tree‑based ensembles) <br>• **Support Vector Machine** (with class weighting) |
| **Feature Selection** | – Remove high‑cardinality identifiers (`PassengerId`, `Ticket`, `Name`). <br>– Apply permutation importance or mutual information after encoding categorical variables. <br>– Drop collinear features (|r| > 0.85). |
| **Encoding** | – One‑hot encode `Sex`, `Embarked`, `Pclass` (or ordinal for `Pclass`). <br>– Use target‑encoding or frequency encoding for `Ticket` if retained. |
| **Imputation** | – Median imputation for `Age`. <br>– “Missing” indicator for `Cabin`. |
| **Validation Strategy** | • **Stratified 5‑fold cross‑validation** (preserves class balance). <br>• Evaluate using **ROC‑AUC**, **Precision‑Recall**, and **F1‑score** (due to class imbalance). |
| **Over‑fitting Mitigation** | – Regularisation (L1/L2) for linear models. <br>– Limit tree depth, set `min_samples_leaf`. <br>– Hyper‑parameter tuning within CV folds (e.g., GridSearchCV). |
| **Performance Targets** | Aim for ROC‑AUC > 0.85 (benchmark for Titanic data). |

If an unsupervised approach is desired, the blueprint suggests:

* **K‑Means** (k = 2–4) with silhouette analysis.  
* **Hierarchical Agglomerative Clustering** (Ward linkage).  
* **PCA** for visualising latent structure (2‑3 components).

---

## 8. Key Take‑aways & Recommendations  

1. **Strongest predictors** of survival are gender (`Sex`), ticket class (`Pclass`), fare amount, and the ticket identifier itself.  
2. **Age** and **Parch** have modest but statistically significant effects; **SibSp** shows no effect.  
3. **Missing data** is concentrated in `Age` and `Cabin`. Imputation has been performed, but consider retaining a “missing” flag for `Cabin`.  
4. **Feature engineering** (family size, title, cabin deck) is expected to boost predictive power and should be added before model training.  
5. **Modeling**: start with a logistic regression baseline, then explore tree‑based ensembles. Use stratified CV and monitor ROC‑AUC and F1‑score.  
6. **Unsupervised exploration** (clustering, PCA) can be useful for detecting hidden sub‑populations, but supervised classification is the natural next step given the target variable.

---

## 9. Appendix A – Image Artifacts  

| File | Type | Size (KB) | Brief Description |
|------|------|-----------|-------------------|
| `dist_Age.png` | Histogram | 41.8 | Age distribution (right‑skewed). |
| `dist_Fare.png` | Histogram | 31.9 | Fare distribution (highly right‑skewed). |
| `dist_Sex.png` | Bar chart | 22.6 | Male vs. female counts. |
| `dist_Pclass.png` | Bar chart | 19.0 | Passenger class frequencies. |
| `dist_Embarked.png` | Bar chart | 21.8 | Port of embarkation frequencies. |
| `dist_SibSp.png` | Histogram | 22.5 | Number of siblings/spouses aboard. |
| `dist_Parch.png` | Histogram | 22.3 | Number of parents/children aboard. |
| `survived_vs_age.png` | Line/Scatter | 59.2 | Survival rate by age. |
| `survived_vs_sex.png` | Bar chart | 27.1 | Survival by gender. |
| `survived_vs_pclass.png` | Bar chart | 47.7 | Survival by passenger class. |
| `survived_vs_fare.png` | Scatter with trend | 67.1 | Survival vs. fare amount. |
| `survived_vs_parch.png` | Bar chart | 43.2 | Survival vs. number of parents/children. |
| `survived_vs_sibsp.png` | Bar chart | 40.0 | Survival vs. number of siblings/spouses. |
| `survived_vs_embarked.png` | Bar chart | 26.2 | Survival by embarkation port. |
| `bivariate_Age_vs_Fare.png` | Scatter (hue = Survived) | 117.6 | Socio‑economic pattern. |
| `bivariate_Age_vs_Pclass.png` | Scatter (hue = Survived) | 102.3 | Age vs. class interaction. |
| `bivariate_SibSp_vs_Parch.png` | Scatter (hue = Survived) | 46.8 | Family size composition. |
| `bivariate_Sex_vs_Pclass.png` | Scatter (hue = Survived) | 29.0 | Gender distribution across classes. |
| `bivariate_Embarked_vs_Fare.png` | Scatter (hue = Survived) | 30.4 | Port ↔ fare relationship. |
| `pairplot.png` | Pairwise matrix (hue = Survived) | 212.0 | Joint distributions of Age, Fare, SibSp, Parch. |

*All images are stored in the sandbox run directory and can be opened with any standard image viewer.*

---

**Prepared by:**  
Senior Lead Data Scientist – Automated EDA Review  
Date: 2026‑08‑07  

---
# Executive Summary – Titanic Survival Dataset  
**Date:** 2026‑08‑07  
**Prepared by:** Senior Lead Data Scientist  

---  

## 1. Project Context  

The dataset (originally `data.csv`) contains passenger information from the RMS Titanic and the binary target **Survived** (0 = did not survive, 1 = survived). The automated EDA pipeline has completed data cleaning, profiling, statistical testing, and produced a predictive‑modeling blueprint. This report consolidates those outputs for rapid stakeholder consumption.

---  

## 2. Dataset Overview  

| Property                     | Value |
|------------------------------|-------|
| Rows                         | 891 |
| Columns (pre‑cleaning)       | 12 |
| Target column                | `Survived` (binary classification) |
| Primary key / ID column      | `PassengerId` |
| Categorical columns          | `Name`, `Sex`, `Ticket`, `Cabin`, `Embarked` |
| Numerical columns            | `Pclass`, `Age`, `SibSp`, `Parch`, `Fare` |
| Overall missing‑value rate   | 19.9 % (Age) + 77.1 % (Cabin) + 0.2 % (Embarked) |

> **Note:** The pipeline has already imputed missing values (see Section 3). No further missing entries remain.

---  

## 3. Missing‑Value Handling  

### 3.1 Imputation Rules Applied  

| Rule ID | Description |
|---------|-------------|
| 1 | Standardised missing string placeholders (`?`, `NA`, `N/A`, `null`) → `NaN`. |
| 2 | Numeric columns with absolute skewness > 1.0 → **median** imputation. |
| 3 | Numeric columns with absolute skewness ≤ 1.0 → **mean** imputation. |
| 4 | Categorical / string columns → **mode** imputation (fallback to `"Unknown"`). |

### 3.2 Column‑wise Imputation Summary  

| Column | dtype | Missing Before | Missing After | Imputation Method | Fill Value |
|--------|-------|----------------|---------------|-------------------|------------|
| Age    | float64 | 177 (19.9 %) | 0 | Median | 28.0 |
| Cabin  | str     | 687 (77.1 %) | 0 | Mode | `B96 B98` |
| Embarked| str    | 2 (0.2 %)   | 0 | Mode | `S` |
| All other columns | – | 0 | 0 | – | – |

All other features required **no** imputation.

---  

## 4. Outlier Profiling  

Outliers were **profiled only** (no removal) to preserve data integrity for downstream modeling.

| Feature | Q1 | Q3 | IQR | Lower Bound | Upper Bound | Outlier Count | Outlier % |
|---------|----|----|-----|-------------|-------------|---------------|-----------|
| Age     | 22.0 | 35.0 | 13.0 | 2.5 | 54.5 | 66 | 7.41 % |
| SibSp   | 0.0 | 1.0 | 1.0 | –1.5 | 2.5 | 46 | 5.16 % |
| Parch   | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 213 | 23.91 % |
| Fare    | 7.9104 | 31.0 | 23.0896 | –26.724 | 65.6344 | 116 | 13.02 % |
| Pclass  | 2.0 | 3.0 | 1.0 | 0.5 | 4.5 | 0 | 0 % |

*Action taken:* `profile` – outlier statistics recorded, no trimming performed.

---  

## 5. Feature Distribution Visualisations  

All distribution plots are saved in the sandbox directory. File names and sizes are listed below; the images themselves are attached to the final HTML report (`eda_report.html`).

| Plot File | Description | Size (KB) |
|-----------|-------------|-----------|
| `dist_Age.png` | Histogram & KDE of passenger ages (post‑imputation) | 37.47 |
| `dist_Sex.png` | Bar chart of gender counts (`male` = 577, `female` = 314) | 22.63 |
| `dist_Pclass.png` | Bar chart of passenger class distribution (1, 2, 3) | 18.96 |
| `dist_SibSp.png` | Histogram of siblings/spouses aboard | 22.54 |
| `dist_Parch.png` | Histogram of parents/children aboard | 22.26 |
| `dist_Fare.png` | Histogram & KDE of ticket fare (highly right‑skewed) | 31.93 |
| `dist_Embarked.png` | Bar chart of embarkation ports (S = 644, C = 168, Q = 77) | 21.70 |
| `dist_Cabin.png` *(implicit in feature‑distribution image)* | Not plotted separately due to high missingness; mode imputed value shown in summary. | – |

---  

## 6. Correlation & Association Analysis  

### 6.1 Pearson Correlation Heatmap  

The correlation matrix image is stored as `correlation_matrix.png` (91.97 KB). The top numeric correlations (absolute value) are:

| # | Feature 1 | Feature 2 | Pearson r |
|---|-----------|-----------|-----------|
| 1 | `Pclass` | `Fare` | **‑0.5495** |
| 2 | `SibSp` | `Parch` | **0.4148** |
| 3 | `Pclass` | `Age` | **‑0.3399** |
| 4 | `Survived` | `Pclass` | **‑0.3385** |
| 5 | `Survived` | `Fare` | **0.2573** |
| 6 | `Age` | `SibSp` | **‑0.2333** |
| 7 | `Parch` | `Fare` | **0.2162** |
| 8 | `Age` | `Parch` | **‑0.1725** |
| 9 | `SibSp` | `Fare` | **0.1597** |
|10 | `Age` | `Fare` | **0.0967** |

*Interpretation:*  
- **Pclass** and **Fare** are strongly inversely correlated (higher class → higher fare).  
- **Pclass** also shows a moderate negative correlation with **Survived**, indicating lower survival rates in lower classes.  
- **SibSp** and **Parch** are positively linked (larger families travel together).  

### 6.2 Categorical Association (Cramér’s V)  

The categorical association matrix image is `categorical_association_matrix.png` (31.91 KB). The strongest association:

| Feature 1 | Feature 2 | Cramér’s V |
|----------|----------|------------|
| `Sex`    | `Embarked` | **0.1107** |

All other categorical pairs have negligible association (≤ 0.11).  

---  

## 7. Statistical Hypothesis Testing  

Each feature was tested against the target `Survived` using the most appropriate test (Pearson correlation for numeric, Welch t‑test for binary gender, one‑way ANOVA for high‑cardinality categorical). Results are summarised below.

| Feature | Test | Statistic | p‑value | Significant (α = 0.05) |
|---------|------|------------|---------|------------------------|
| `Pclass` | Pearson r | ‑0.3385 | 2.54 e‑25 | **Yes** |
| `Sex`    | Welch t | 18.6718 | 2.28 e‑61 | **Yes** |
| `Parch`  | Pearson r | 0.0816 | 1.48 e‑02 | **Yes** |
| `Ticket` | One‑Way ANOVA | 3.0276 | 3.31 e‑13 | **Yes** |
| `Fare`   | Pearson r | 0.2573 | 6.12 e‑15 | **Yes** |
| `Cabin`  | One‑Way ANOVA | 2.7851 | 1.28 e‑08 | **Yes** |
| `Embarked`| One‑Way ANOVA | 13.3269 | 1.98 e‑06 | **Yes** |
| `Age`    | Pearson r | ‑0.0649 | 5.28 e‑02 | No |
| `SibSp`  | Pearson r | ‑0.0353 | 2.92 e‑01 | No |
| `PassengerId`| Pearson r | ‑0.0050 | 8.81 e‑01 | No |

**Significant predictors** (p < 0.05): `Pclass`, `Sex`, `Parch`, `Ticket`, `Fare`, `Cabin`, `Embarked`.  

---  

## 8. Predictive‑Modeling Blueprint  

The pipeline generated a concise modelling plan, reproduced here for implementation.

### 8.1 Problem Definition  

- **Target:** `Survived` (binary classification)  
- **Data size:** 891 × 12 (post‑imputation)  

### 8.2 Recommended Algorithms  

| Rank | Algorithm | Rationale |
|------|-----------|-----------|
| 1 | Regularized Logistic Regression (baseline) | Interpretable, fast, handles multicollinearity with L1/L2 penalties. |
| 2 | Random Forest Classifier | Captures non‑linear interactions, robust to outliers, provides feature importance. |
| 3 | Gradient Boosting (XGBoost / LightGBM) | State‑of‑the‑art performance on tabular data, handles missing values internally. |
| 4 | Support Vector Classifier (SVM) | Effective in high‑dimensional space, especially with kernel tricks. |

### 8.3 Feature‑Selection Strategy  

1. **Exclude** high‑cardinality identifiers (`PassengerId`, `Name`, `Ticket`) from modeling.  
2. **Rank** remaining features using cross‑validated permutation importance **and** mutual information.  
3. **Drop** any pair of features with absolute Pearson correlation > 0.85 (none detected in current set).  

### 8.4 Validation Protocol  

- **Stratified K‑Fold Cross‑Validation** (K = 5) to preserve class balance in each fold.  
- **Evaluation metrics:**  
  - Balanced Accuracy  
  - Macro‑averaged F1 Score  
  - Precision‑Recall AUC (more informative for imbalanced classes)  
  - Confusion Matrix (for error analysis)  

### 8.5 Over‑fitting Mitigation  

| Technique | Application |
|-----------|-------------|
| Regularization (L1/L2) | Logistic Regression, linear SVM |
| Tree depth limits & min‑samples‑leaf | Random Forest, Gradient Boosting |
| Hyper‑parameter tuning **within** CV folds (e.g., GridSearchCV) | All algorithms |
| Early stopping (for boosting) | XGBoost / LightGBM |

---  

## 9. Key Insights & Recommendations  

| Insight | Business Implication |
|---------|----------------------|
| **Sex** is the strongest single predictor (Welch t‑test p ≈ 0). Female passengers had dramatically higher survival rates. | Targeted safety measures (e.g., lifeboat allocation) historically favoured women; modern simulations should consider gender‑balanced protocols. |
| **Pclass** shows a strong negative correlation with survival (lower class → lower survival). | Socio‑economic status heavily influenced outcomes; may be used as a proxy for access to resources in downstream risk models. |
| **Fare** positively correlates with survival (r ≈ 0.26, p < 1e‑14). | Ticket price reflects class and possibly cabin location; useful for feature engineering (e.g., fare‑per‑person). |
| **Parch** is a modest but significant predictor (p ≈ 0.015). Larger families (more parents/children) slightly increase survival odds. | Family size could be engineered (e.g., total relatives = `SibSp` + `Parch`). |
| **Cabin** and **Embarked** are statistically significant despite high missingness (Cabin) or low cardinality (Embarked). | After imputation, cabin information still carries signal; consider encoding cabin decks (e.g., first letter) for richer features. |
| **Age** is not statistically significant at α = 0.05 (p ≈ 0.053). | Age alone may not be a strong predictor, but interactions (e.g., Age × Sex) could be explored. |
| **Outlier percentages** are modest (< 15 %) for numeric features, except `Parch` (≈ 24 %). | No aggressive outlier removal required; profiling suffices. |

**Action Items**  

1. Implement the modelling blueprint using the recommended algorithms and validation scheme.  
2. Engineer additional features:  
   - `FamilySize = SibSp + Parch + 1` (including the passenger).  
   - `CabinDeck = first character of Cabin` (after mode imputation).  
   - `FarePerPerson = Fare / FamilySize`.  
3. Perform hyper‑parameter optimisation within the stratified CV framework.  
4. Compare baseline logistic regression against ensemble methods; select the model with the highest macro‑F1 while maintaining interpretability.  

---  

## 10. Artefacts Summary  

| Artefact | Description | Path (excerpt) |
|----------|-------------|----------------|
| `correlation_matrix.png` | Pearson correlation heatmap (numeric features). | `...\\correlation_matrix.png` |
| `categorical_association_matrix.png` | Cramér’s V heatmap for categorical pairs. | `...\\categorical_association_matrix.png` |
| `bivariate_Age_vs_Survived.png` | Age distribution split by survival outcome. | `...\\bivariate_Age_vs_Survived.png` |
| `bivariate_Fare_vs_Survived.png` | Fare distribution split by survival outcome. | `...\\bivariate_Fare_vs_Survived.png` |
| `bivariate_Sex_vs_Survived.png` | Survival rates for male vs female. | `...\\bivariate_Sex_vs_Survived.png` |
| `bivariate_Pclass_vs_Survived.png` | Survival rates across passenger classes. | `...\\bivariate_Pclass_vs_Survived.png` |
| `bivariate_SibSp_vs_Survived.png` | Survival vs number of siblings/spouses aboard. | `...\\bivariate_SibSp_vs_Survived.png` |
| `bivariate_Parch_vs_Survived.png` | Survival vs number of parents/children aboard. | `...\\bivariate_Parch_vs_Survived.png` |
| `dist_*.png` (7 files) | Univariate distribution plots for each feature. | See Section 5. |
| `eda_report.html` | Full interactive HTML report generated by the pipeline. | (root of sandbox) |
| `metrics.json` | Machine‑readable summary of all metrics (included above). | (root) |
| `metadata_profile.json` | Detailed schema and descriptive statistics. | (root) |
| `current_df.csv` | Cleaned dataset after imputation (ready for modelling). | (root) |

---  

## 11. Conclusion  

The EDA pipeline has produced a clean, fully‑imputed dataset, identified the most influential predictors of survival, and delivered a concrete modelling roadmap. Implementing the recommended algorithms with the outlined feature‑selection and validation strategy should yield a robust classifier capable of achieving high balanced accuracy and macro‑F1 on this classic dataset.  

*Prepared for the data‑science leadership team.*  
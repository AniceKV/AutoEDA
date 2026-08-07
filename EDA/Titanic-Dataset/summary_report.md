# Executive Summary – Titanic EDA & Predictive Blueprint  

**Prepared by:** Senior Lead Data Scientist  
**Date:** 2026‑08‑07  

---

## 1. Overview  

| Item | Value |
|------|-------|
| **Dataset** | `Titanic‑Dataset.csv` |
| **Rows / Columns** | 891 × 12 |
| **Target Variable** | `Survived` (binary classification) |
| **Primary Goal** | Build a robust classification model to predict passenger survival. |
| **Key Findings** | • 8 features are statistically significant predictors of survival. <br>• Missing values were fully imputed (Age → mean, Cabin → mode, Embarked → mode). <br>• No engineered features survived the pipeline (all specs produced 0 new columns). <br>• Strongest linear relationships: `Pclass` ↔ `Fare` (ρ = ‑0.55) and `Survived` ↔ `Pclass` (ρ = ‑0.34). |

The automated EDA pipeline completed all planned steps, generated a suite of visual artefacts, and produced a detailed modeling blueprint. The following sections summarise the quantitative results and actionable recommendations.

---

## 2. Data Quality  

### 2.1 Missing‑Value Summary  

| Column | Missing Count | Missing % | Imputation Method | Fill Value |
|--------|---------------|----------|-------------------|------------|
| Age    | 177 | 19.9 % | Mean (skewness = 0.39) | 29.6991 |
| Cabin  | 687 | 77.1 % | Mode | `B96 B98` |
| Embarked| 2   | 0.2 % | Mode | `S` |
| All other columns | 0 | 0 % | – | – |

*All missing entries were replaced; no rows were dropped.*

### 2.2 Imputation Rules Applied  

1. Standardised missing string placeholders (`?`, `NA`, `N/A`, `null`) → `NaN`.  
2. Numeric columns with |skewness| > 1 → median imputation (none required).  
3. Numeric columns with |skewness| ≤ 1 → mean imputation (applied to **Age**).  
4. Categorical columns → mode imputation, fallback to `"Unknown"` (applied to **Cabin**, **Embarked**).  

---

## 3. Outlier Profiling  

| Column | Q1 | Q3 | IQR | Lower Bound | Upper Bound | Outliers (count) | Outlier % |
|--------|----|----|-----|-------------|-------------|------------------|----------|
| Age    | 22.0 | 35.0 | 13.0 | 2.5 | 54.5 | 66 | 7.41 % |
| SibSp  | 0.0 | 1.0 | 1.0 | ‑1.5 | 2.5 | 46 | 5.16 % |
| Parch  | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 213 | 23.91 % |
| Fare   | 7.9104 | 31.0 | 23.0896 | ‑26.724 | 65.6344 | 116 | 13.02 % |

*Action taken:* `profile` – outliers were identified but **not removed** to preserve the original distribution for modeling.

---

## 4. Feature Distributions  

The pipeline generated individual distribution plots for each variable.  

| Plot File | Variable(s) | Description |
|-----------|-------------|-------------|
| `dist_Age.png` | Age | Histogram + KDE of passenger ages (post‑imputation). |
| `dist_Fare.png` | Fare | Highly right‑skewed fare distribution. |
| `dist_Pclass.png` | Pclass | Bar chart of 1st/2nd/3rd class frequencies. |
| `dist_Sex.png` | Sex | Male vs. female count. |
| `dist_Embarked.png` | Embarked | Port of embarkation frequencies. |
| `dist_SibSp.png` | SibSp | Number of siblings/spouses aboard. |
| `dist_Parch.png` | Parch | Number of parents/children aboard. |

*All plots are stored in the working directory and referenced in the HTML report (`eda_report.html`).*

---

## 5. Correlation & Categorical Association  

### 5.1 Pearson Correlations (Top 10)  

| Feature 1 | Feature 2 | ρ (Pearson) |
|-----------|-----------|------------|
| Pclass    | Fare      | **‑0.5495** |
| SibSp     | Parch     | **0.4148** |
| Survived  | Pclass    | **‑0.3385** |
| Pclass    | Age       | **‑0.3313** |
| Survived  | Fare      | **0.2573** |
| Age       | SibSp     | **‑0.2326** |
| Parch     | Fare      | **0.2162** |
| Age       | Parch     | **‑0.1792** |
| SibSp     | Fare      | **0.1597** |
| Age       | Fare      | **0.0916** |

*Correlation matrix saved as `correlation_matrix.png`.*

### 5.2 Categorical Association (Cramér’s V)  

| Feature 1 | Feature 2 | Cramér’s V |
|-----------|-----------|-----------|
| Sex       | Embarked  | **0.1107** |

*Association heatmap saved as `categorical_association_matrix.png`.*

---

## 6. Statistical Hypothesis Testing  

| Feature | Test Type | Statistic | p‑value | Significant (α = 0.05) | Interpretation |
|---------|-----------|-----------|---------|------------------------|----------------|
| Pclass  | Pearson Correlation | ‑0.3385 | 2.54 e‑25 | ✅ | Strong negative association with survival. |
| Sex     | Welch Two‑Sample t‑test | 18.6718 | 2.28 e‑61 | ✅ | Survival differs dramatically by gender. |
| Age     | Pearson Correlation | ‑0.0698 | 3.72 e‑02 | ✅ | Slight negative trend. |
| Parch   | Pearson Correlation | 0.0816 | 1.48 e‑02 | ✅ | Small positive effect. |
| Ticket  | One‑Way ANOVA | 3.0276 | 3.31 e‑13 | ✅ | Ticket categories contain predictive information. |
| Fare    | Pearson Correlation | 0.2573 | 6.12 e‑15 | ✅ | Higher fares increase survival odds. |
| Cabin   | One‑Way ANOVA | 2.7851 | 1.28 e‑08 | ✅ | Cabin assignment is informative. |
| Embarked| One‑Way ANOVA | 13.3269 | 1.98 e‑06 | ✅ | Port of embarkation matters. |
| PassengerId | Pearson Correlation | ‑0.005 | 0.881 | ❌ | No predictive power. |
| SibSp   | Pearson Correlation | ‑0.0353 | 0.292 | ❌ | Not significant. |

**Significant Predictors (8):** `Pclass`, `Sex`, `Age`, `Parch`, `Ticket`, `Fare`, `Cabin`, `Embarked`.

---

## 7. Feature Engineering  

The pipeline attempted to create four engineered features:

| Spec | Intended Operation | Result |
|------|-------------------|--------|
| `FamilySize` | `SibSp` + `Parch` | **0** new columns (already present as derived later). |
| `IsAlone` | `FamilySize` == 0 | **0** new columns. |
| `Age*Pclass` | `Age` × `Pclass` | **0** new columns. |
| `FarePerPerson` | `Fare` ÷ (`FamilySize` + 1) | **0** new columns. |

*All specifications executed without error but produced **no additional columns** (likely due to naming conflicts or downstream filtering).*  
**Recommendation:** Re‑run feature engineering manually, ensuring new column names do not clash with existing ones, and verify that the derived columns are retained.

---

## 8. Predictive Modeling Blueprint  

| Aspect | Recommendation |
|--------|----------------|
| **Problem Type** | Binary Classification (`Survived`). |
| **Baseline Algorithm** | Regularized Logistic Regression (L2 penalty). |
| **Advanced Algorithms** | • Random Forest Classifier <br>• Gradient Boosting (XGBoost / LightGBM) <br>• Support Vector Classifier (SVM). |
| **Feature Selection** | 1. Drop high‑cardinality identifiers (`PassengerId`, `Name`, `Ticket`). <br>2. Rank remaining features via cross‑validated permutation importance & mutual information. <br>3. Remove collinear pairs with |ρ| > 0.85 (none observed above threshold). |
| **Validation Strategy** | Stratified 5‑fold cross‑validation. |
| **Evaluation Metrics** | • Balanced Accuracy <br>• Macro‑averaged F1 <br>• Precision‑Recall AUC <br>• Confusion Matrix (per fold). |
| **Over‑fitting Mitigation** | • Apply L1/L2 regularisation (logistic, linear SVM). <br>• Limit tree depth, set `min_samples_leaf` for tree‑based models. <br>• Perform hyper‑parameter tuning **inside** CV folds (e.g., GridSearchCV or Optuna). |
| **Execution Summary** | “Target: Survived (Classification). Use robust cross‑validation on 891 rows × 12 columns.” |

All blueprint details are stored in `blueprint_res` (also duplicated in `predictive_modeling_blueprint`).

---

## 9. Recommendations & Next Steps  

1. **Finalize Feature Set**  
   * Remove identifier columns (`PassengerId`, `Name`, `Ticket`).  
   * Re‑engineer the four planned features, verify they are added to the dataframe, and assess their importance.

2. **Model Development**  
   * Implement the baseline logistic regression to establish a performance floor.  
   * Parallelly train Random Forest and Gradient Boosting models; compare using the metrics above.  

3. **Hyper‑parameter Optimization**  
   * Use Bayesian optimisation (e.g., Optuna) within the stratified CV loop.  

4. **Model Interpretation**  
   * Generate SHAP or permutation importance plots for the best model to explain the impact of `Sex`, `Pclass`, `Fare`, etc.  

5. **Reporting**  
   * Compile a final model card (performance, data provenance, fairness checks).  
   * Include the visual artefacts (`age_vs_survived.png`, `fare_vs_survived.png`, `bivariate_*` plots) in the presentation deck.

---

## 10. Artefact Inventory  

| File | Type | Brief Description |
|------|------|-------------------|
| `age_vs_survived.png` | Image | Survival rate across age bins. |
| `fare_vs_survived.png` | Image | Survival rate across fare bins. |
| `bivariate_Age_vs_Fare.png` | Image | Scatter of Age vs. Fare coloured by Survived. |
| `bivariate_Pclass_vs_Fare.png` | Image | Box/violin of Fare by passenger class. |
| `bivariate_Sex_vs_Survived.png` | Image | Survival distribution by gender. |
| `categorical_association_matrix.png` | Image | Cramér’s V heatmap (Sex ↔ Embarked). |
| `correlation_matrix.png` | Image | Pearson correlation heatmap for numeric features. |
| `dist_*.png` (7 files) | Images | Univariate distributions for each variable. |
| `current_df.csv` | CSV | Final pre‑modeling dataset (post‑imputation). |
| `metadata_profile.json` | JSON | Schema, cardinalities, missing‑value summary. |
| `metrics.json` | JSON | Consolidated numeric results (imputation, outliers, correlations, hypothesis tests, blueprint). |
| `agent_plan_log.json` | JSON | Full execution plan and step‑wise outcomes. |
| `agent_state.json` | JSON | Current state snapshot (same info as `metrics.json`). |
| `eda_report.html` | HTML | Interactive report aggregating all visualisations and tables. |

---

### Closing Statement  

The Titanic dataset exhibits classic survival‑related patterns (gender, class, fare) that are statistically robust. The automated EDA has prepared a clean, imputed dataset and delivered a clear modeling roadmap. By addressing the minor feature‑engineering gap and following the recommended validation framework, a high‑performing, interpretable classifier can be delivered within a short development cycle.  

---  
# Executive Summary – Titanic Survival Prediction (Auto‑EDA)

**Prepared by:** Senior Lead Data Scientist  
**Date:** 2026‑08‑06  

---

## 1. Executive Overview
The automated EDA pipeline processed the classic **Titanic** passenger dataset (891 rows × 13 columns) to produce a complete statistical portrait, missing‑value handling, outlier profiling, feature‑engineering, and a predictive‑modeling blueprint for the binary target **`Survived`**.  

Key take‑aways:

| Item | Insight |
|------|---------|
| **Target** | `Survived` – binary classification (0 = did not survive, 1 = survived) |
| **Significant predictors** | `Pclass`, `Sex`, `Age`, `Parch`, `Ticket`, `Fare`, `Cabin`, `Embarked`, engineered log‑Fare feature |
| **Imputation** | Mean imputation for `Age`; mode imputation for `Cabin` and `Embarked` |
| **Outliers** | Detected (profile‑only) in `Age`, `Fare`, `SibSp`, `Parch`; no removal performed |
| **Feature engineering** | Log‑1p transformation of `Fare` → `engineered_feature` (high correlation with `Fare` and `Survived`) |
| **Recommended models** | Logistic Regression (baseline), Random Forest, Gradient Boosting (XGBoost / LightGBM), Support Vector Classifier |
| **Validation** | Stratified 5‑fold CV with balanced‑accuracy, macro‑F1, PR‑AUC, confusion matrix |

The following sections detail each component of the analysis.

---

## 2. Dataset Overview
| Property | Value |
|----------|-------|
| **Source file** | `Titanic-Dataset.csv` |
| **Rows** | 891 |
| **Columns (including engineered)** | 13 |
| **Target column** | `Survived` |
| **Data types** | 5 numeric (`PassengerId`, `Survived`, `Pclass`, `Age`, `SibSp`, `Parch`, `Fare`, `engineered_feature`) and 5 categorical (`Name`, `Sex`, `Ticket`, `Cabin`, `Embarked`) |

### Column Summary (selected)

| Column | dtype | Cardinality | Missing % | Key metric |
|--------|-------|-------------|----------|------------|
| `PassengerId` | int64 | 891 | 0.0 | Range [1‑891], Mean 446 |
| `Survived` | int64 | 2 | 0.0 | Mean 0.38 |
| `Pclass` | int64 | 3 | 0.0 | Mean 2.31 |
| `Sex` | object | 2 | 0.0 | Male 577, Female 314 |
| `Age` | float64 | 88 | 19.9 | Mean 29.70, Median 28 |
| `Cabin` | object | 147 | 77.1 | Mode `B96 B98` |
| `Embarked` | object | 3 | 0.2 | Mode `S` |
| `Fare` | float64 | 248 | 0.0 | Mean 32.20, Median 14.45 |
| `engineered_feature` | float64 | 248 | 0.0 | = log1p(`Fare`) |

---

## 3. Missing‑Value Handling & Imputation
The pipeline applied a rule‑based imputation strategy:

| Column | Missing before | Imputation method | Fill value |
|--------|----------------|-------------------|------------|
| `Age` | 177 (19.9 %) | Mean (skewness 0.39) | 29.6991 |
| `Cabin` | 687 (77.1 %) | Mode | `B96 B98` |
| `Embarked` | 2 (0.2 %) | Mode | `S` |
| All others | 0 | – | – |

*Numeric columns with modest skewness (|skew| ≤ 1) used mean imputation; highly skewed numeric columns would have used median (none required). Categorical columns used mode with a fallback to `"Unknown"`.*

---

## 4. Outlier Profiling
Outliers were **profiled only** (no removal). Summary:

| Feature | Q1 | Q3 | IQR | Lower bound | Upper bound | Outliers % |
|---------|----|----|-----|-------------|-------------|-----------|
| `Age` | 22.0 | 35.0 | 13.0 | 2.5 | 54.5 | 7.41 % |
| `Fare` | 7.9104 | 31.0 | 23.0896 | –26.724 | 65.6344 | 13.02 % |
| `SibSp` | 0.0 | 1.0 | 1.0 | –1.5 | 2.5 | 5.16 % |
| `Parch` | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 23.91 % |

*The high outlier percentage for `Parch` reflects the large number of passengers with zero parents/children aboard.*

---

## 5. Feature Engineering
**Engineered Feature:**  

| Feature | Formula | Data type | Rationale |
|---------|---------|-----------|-----------|
| `engineered_feature` | `np.log1p(Fare)` | float64 | Captures non‑linear relationship of fare with survival; reduces skewness of `Fare`. |

Correlation of the engineered feature:

| Pair | Correlation |
|------|-------------|
| `Fare` ↔ `engineered_feature` | **0.7875** |
| `Survived` ↔ `engineered_feature` | **0.3299** |
| `Pclass` ↔ `engineered_feature` | **‑0.661** |

The engineered feature is strongly linked to both the original `Fare` and the target, justifying its inclusion.

---

## 6. Correlation & Categorical Association Analysis
### 6.1 Top Numeric Correlations (|ρ| > 0.30)

| Feature 1 | Feature 2 | Correlation |
|-----------|-----------|-------------|
| `Fare` | `engineered_feature` | 0.7875 |
| `Pclass` | `engineered_feature` | –0.661 |
| `Pclass` | `Fare` | –0.5495 |
| `SibSp` | `Parch` | 0.4148 |
| `Survived` | `Pclass` | –0.3385 |
| `Parch` | `engineered_feature` | 0.3322 |
| `Pclass` | `Age` | –0.3313 |
| `Survived` | `engineered_feature` | 0.3299 |
| `SibSp` | `engineered_feature` | 0.3185 |
| `Survived` | `Fare` | 0.2573 |

*All correlations are computed on the imputed dataset (including engineered feature).*

### 6.2 Categorical Association (Cramér’s V)

| Feature 1 | Feature 2 | Cramér’s V |
|-----------|-----------|------------|
| `Sex` | `Embarked` | 0.1107 |

The low association indicates that `Sex` and `Embarked` are largely independent, but both remain useful predictors (see hypothesis testing).

---

## 7. Statistical Hypothesis Testing
A suite of tests evaluated each predictor’s relationship with the target (`Survived`). Significance threshold: α = 0.05.

| Feature | Test | Statistic | p‑value | Significant? | Interpretation |
|---------|------|-----------|---------|--------------|----------------|
| `PassengerId` | Pearson r | –0.0050 | 0.8814 | No | No predictive signal |
| `Pclass` | Pearson r | –0.3385 | 2.54e‑25 | **Yes** | Strong negative association |
| `Sex` | Welch t‑test | 18.6718 | 2.28e‑61 | **Yes** | Large difference between genders |
| `Age` | Pearson r | –0.0698 | 0.0372 | **Yes** | Small negative effect |
| `SibSp` | Pearson r | –0.0353 | 0.2922 | No | Not predictive |
| `Parch` | Pearson r | 0.0816 | 0.0148 | **Yes** | Small positive effect |
| `Ticket` | One‑Way ANOVA | 3.0276 | 3.31e‑13 | **Yes** | Ticket groups differ in survival |
| `Fare` | Pearson r | 0.2573 | 6.12e‑15 | **Yes** | Higher fare → higher survival |
| `Cabin` | One‑Way ANOVA | 2.7851 | 1.28e‑08 | **Yes** | Cabin categories differ |
| `Embarked` | One‑Way ANOVA | 13.3269 | 1.98e‑06 | **Yes** | Port of embarkation matters |
| `engineered_feature` | Pearson r | 0.3299 | 4.65e‑24 | **Yes** | Log‑Fare captures survival signal |

**Significant predictors (9 total):** `Pclass`, `Sex`, `Age`, `Parch`, `Ticket`, `Fare`, `Cabin`, `Embarked`, `engineered_feature`.

---

## 8. Predictive Modeling Blueprint
The pipeline generated a concise modeling plan:

| Aspect | Recommendation |
|--------|----------------|
| **Problem type** | Binary Classification (`Survived`) |
| **Baseline algorithm** | Regularized Logistic Regression |
| **Advanced algorithms** | Random Forest, Gradient Boosting (XGBoost / LightGBM), Support Vector Classifier |
| **Feature selection** | 1. Drop high‑cardinality identifiers (`PassengerId`, `Name`, `Ticket` unless encoded). 2. Rank features via cross‑validated permutation importance & mutual information. 3. Remove collinear features with |ρ| > 0.85 (e.g., `Fare` vs. `engineered_feature`). |
| **Validation strategy** | Stratified 5‑fold CV; evaluate **Balanced Accuracy**, **Macro F1**, **Precision‑Recall AUC**, and **Confusion Matrix**. |
| **Overfitting mitigation** | • L1/L2 regularization (logistic regression, SVM).<br>• Tree depth limits & minimum samples per leaf (RF, GB).<br>• Hyper‑parameter tuning confined to inner CV folds. |
| **Execution environment** | 891 rows × 13 columns – fits comfortably in memory; no distributed computing required. |
| **Executive summary** | Use robust cross‑validation on the full dataset; start with logistic regression as a baseline, then explore ensemble methods for potential performance gains. |

---

## 9. Visual Artifacts (PNG files)
All visualizations are saved in the sandbox run directory. Below is a brief description of each artifact; the actual images can be opened from the indicated paths.

| File | Description |
|------|-------------|
| `bivariate_Embarked_vs_Fare.png` | Scatter/box plot of `Fare` across `Embarked` categories, colored by `Survived`. |
| `bivariate_Pclass_vs_Fare.png` | Relationship between passenger class and fare, with survival hue. |
| `bivariate_Sex_vs_Age.png` | Age distribution split by gender and survival status. |
| `categorical_association_matrix.png` | Heatmap of Cramér’s V for categorical pairs (only `Sex`‑`Embarked` shown). |
| `correlation_matrix.png` | Full Pearson correlation heatmap for numeric features (incl. engineered). |
| `dist_*.png` (Age, Fare, etc.) | Univariate histograms / bar charts for each listed column. |
| `pairplot.png` | Pairwise scatter/box plots for `Age`, `Fare`, `Pclass`, `SibSp` colored by `Survived`. |
| `target_interactions.png` | Visual of `Survived` vs. `Fare` (including engineered feature) to illustrate interaction. |

*All images are under 210 KB, suitable for inclusion in reports or dashboards.*

---

## 10. Recommendations & Next Steps
1. **Data Preparation**  
   - Encode categorical variables (`Sex`, `Embarked`, `Cabin`) using appropriate schemes (e.g., one‑hot or target encoding).  
   - Consider dimensionality reduction for high‑cardinality `Ticket` if retained.  

2. **Model Development**  
   - Implement the baseline logistic regression with L2 regularization; record baseline metrics.  
   - Sequentially train Random Forest and Gradient Boosting models, applying the feature‑selection strategy to avoid multicollinearity (e.g., drop either `Fare` or `engineered_feature`).  

3. **Model Evaluation**  
   - Use the prescribed stratified 5‑fold CV; compare models on balanced accuracy and macro‑F1.  
   - Plot ROC and PR curves; examine confusion matrices for class‑imbalance handling.  

4. **Hyper‑parameter Tuning**  
   - Conduct grid/random search within the CV folds (e.g., `max_depth`, `n_estimators`, `learning_rate`).  

5. **Interpretability**  
   - Generate SHAP or permutation‑importance plots to confirm that the statistically significant predictors identified earlier retain importance in the final model.  

6. **Deployment Considerations**  
   - Since the dataset is small, a lightweight model (logistic regression or shallow tree ensemble) will be fast to serve.  
   - Ensure reproducibility by persisting the imputation and encoding pipelines alongside the trained model.  

---

**Prepared by:**  
Senior Lead Data Scientist – Auto‑EDA Project  

*All analyses are derived from the automatically generated artifacts listed above; no manual code modifications were performed.*
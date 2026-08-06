# Executive Summary – Fertility Dataset (Diagnosis Prediction)

**Dataset**: `fertility.csv` (100 rows × 10 columns)  
**Target**: `Diagnosis` – binary (Normal = 88 %, Altered = 12 %)  
**Problem Type**: Supervised **classification**  

---

## 1. Data Overview  

| Attribute | Data Type | Cardinality | Missing % | Key Summary |
|----------|-----------|------------|----------|-------------|
| Season | object | 4 | 0 % | spring (37), fall (31), winter (28) |
| Age | int64 | 10 | 0 % | Mean = 30.11, Median = 30, Std = 2.25, Range = 27‑36, Skew = 0.67 |
| Childish diseases | object | 2 | 0 % | yes (87), no (13) |
| Accident or serious trauma | object | 2 | 0 % | yes (44), no (56) |
| Surgical intervention | object | 2 | 0 % | yes (51), no (49) |
| High fevers in the last year | object | 3 | 0 % | > 3 months ago (63), no (28), < 3 months (9) |
| Frequency of alcohol consumption | object | 5 | 0 % | hardly ever (40), once a week (39), several times (19) |
| Smoking habit | object | 3 | 0 % | never (56), occasional (23), daily (21) |
| Number of hours spent sitting per day | int64 | 14 | 0 % | Mean = 10.80, Median = 7, Std = 33.62, Range = 1‑342, Skew = 9.85 |
| Diagnosis (target) | object | 2 | 0 % | Normal (88), Altered (12) |

*No missing values were detected; the imputation module therefore performed no actions.*

---

## 2. Outlier Detection & Treatment  

| Numeric Feature | Q1 | Q3 | IQR | Lower Bound | Upper Bound | Outliers (Count %) | Action |
|-----------------|----|----|-----|-------------|-------------|-------------------|--------|
| Age | 28 | 32 | 4 | 22 | 38 | 0 (0 %) | Capped (no effect) |
| Hours sitting | 5 | 9 | 4 | –1 | 15 | 5 (5 %) | Values > 15 capped to 15 |

The capping of the five extreme “hours‑sitting” records reduces the maximum from 342 h to 15 h, mitigating the heavy right‑tail while preserving the bulk of the distribution.

---

## 3. Feature Engineering  

The automated plan attempted two engineered features:

| Spec | Source | Result |
|------|--------|--------|
| `log1p` on *Hours sitting* → `log_hours` | `Number of hours spent sitting per day` | **0 features generated** |
| Interaction → `age_loghours_interaction` | `Age` × `log_hours` | **0 features generated** |

No new columns were added to the final dataframe. Consequently, the modeling stage will rely on the original ten variables.

---

## 4. Univariate & Bivariate Visualisations  

All plots are stored in the `sandbox_run` directory (file sizes shown for reference).

| Plot | Description | Size (KB) |
|------|-------------|-----------|
| `dist_Age.png` | Histogram + KDE of Age | 39.78 |
| `dist_Number_of_hours_spent_sitting_per_day.png` | Highly right‑skewed distribution (post‑capping) | 37.85 |
| `dist_Season.png` | Bar chart of seasonal counts | 32.40 |
| `dist_Smoking_habit.png` | Bar chart of smoking frequency | 31.87 |
| `dist_Frequency_of_alcohol_consumption.png` | Bar chart of alcohol consumption | 52.88 |
| `bivariate_Age_vs_Diagnosis.png` | Age vs. Diagnosis (box/violin) | 29.24 |
| `bivariate_Number_of_hours_spent_sitting_per_day_vs_Diagnosis.png` | Hours sitting vs. Diagnosis (box/violin) | 44.91 |
| `bivariate_Season_vs_Diagnosis.png` | Diagnosis rates per season | 39.57 |
| `bivariate_Smoking_habit_vs_Diagnosis.png` | Diagnosis vs. smoking habit | 39.72 |
| `bivariate_Frequency_of_alcohol_consumption_vs_Diagnosis.png` | Diagnosis vs. alcohol frequency | 63.48 |
| `pairplot.png` | Pairwise scatter/box plots coloured by Diagnosis | 79.26 |
| `correlation_matrix.png` | Pearson correlation heatmap (numeric) | 40.30 |
| `categorical_association_matrix.png` | Cramér’s V heatmap (categorical) | 140.25 |
| `age_vs_diagnosis.png` | Target interaction – Age | 31.52 |
| `hours_vs_diagnosis.png` | Target interaction – Hours sitting | 43.30 |

*Interpretation*: Visual inspection confirms the lack of strong separation between classes for any single variable; the class imbalance (12 % Altered) is evident.

---

## 5. Correlation & Association Analysis  

### 5.1 Numeric Correlations  

Only one pair shows a measurable relationship:

| Feature 1 | Feature 2 | Pearson r |
|-----------|-----------|-----------|
| Age | Hours sitting | **‑0.047** |

The magnitude is negligible, indicating virtually independent numeric predictors.

### 5.2 Categorical Associations (Cramér’s V)  

| Feature 1 | Feature 2 | Cramér’s V |
|-----------|-----------|------------|
| Surgical intervention | High fevers in the last year | **0.244** |
| Season | Accident or serious trauma | **0.225** |
| Frequency of alcohol consumption | Smoking habit | **0.210** |
| Season | High fevers in the last year | **0.203** |
| Accident or serious trauma | Frequency of alcohol consumption | **0.177** |
| … | … | (remaining values ≤ 0.13) |

The strongest association (≈ 0.24) is still modest, suggesting limited predictive power from any single categorical pair.

---

## 6. Statistical Hypothesis Testing  

All tests compare each predictor against the binary target `Diagnosis`. Significance threshold α = 0.05.

| Predictor | Test | Statistic | p‑value | Significant? | Interpretation |
|-----------|------|-----------|---------|--------------|----------------|
| Season | Chi‑square (df = 3) | 4.1613 | 0.245 | No | No evidence of dependence |
| Age | Welch t‑test | 1.0435 | 0.313 | No | Mean ages similar across classes |
| Childish diseases | Chi‑square (df = 1) | 0.0000 | 1.000 | No | No association |
| Accident or serious trauma | Chi‑square (df = 1) | 1.2177 | 0.270 | No | No association |
| Surgical intervention | Chi‑square (df = 1) | 0.0547 | 0.815 | No | No association |
| High fevers (last year) | Chi‑square (df = 2) | 1.5452 | 0.462 | No | No association |
| Frequency of alcohol consumption | Chi‑square (df = 4) | 4.0263 | 0.402 | No | No association |
| Smoking habit | Chi‑square (df = 2) | 0.2153 | 0.898 | No | No association |
| Hours sitting | Welch t‑test | ‑0.9024 | 0.369 | No | No difference in means |

**Result**: *No predictor reached statistical significance.* Consequently, the `significant_predictors` list is empty.

---

## 7. Predictive‑Modeling Blueprint  

| Aspect | Recommendation |
|--------|----------------|
| **Target** | `Diagnosis` (binary classification) |
| **Algorithms** (baseline → advanced) | 1. Regularized Logistic Regression  <br>2. Random Forest Classifier  <br>3. Gradient Boosting (XGBoost / LightGBM)  <br>4. Support Vector Classifier |
| **Feature Selection** | • Remove any high‑cardinality identifiers (none present) <br>• Rank features via cross‑validated permutation importance and mutual information <br>• Drop collinear numeric features with |r| > 0.85 (none detected) |
| **Validation Strategy** | Stratified 5‑fold cross‑validation (preserves 12 % Altered class) |
| **Evaluation Metrics** | Balanced Accuracy, Macro‑averaged F1, Precision‑Recall AUC, Confusion Matrix (focus on minority class) |
| **Over‑fitting Mitigation** | • L1/L2 regularization (logistic, linear SVM) <br>• Tree depth limits, minimum samples per leaf (RF, GB) <br>• Hyper‑parameter tuning confined to inner CV loops |
| **Class Imbalance Handling** | Consider SMOTE, class‑weighting, or focal loss in model training. |
| **Execution Summary** | “Target: Diagnosis (Classification). Use robust cross‑validation on 100 rows × 10 columns.” |

---

## 8. Key Insights & Recommendations  

| Insight | Actionable Recommendation |
|---------|---------------------------|
| **Class imbalance (12 % Altered)** | Apply resampling or cost‑sensitive learning; report per‑class metrics. |
| **No statistically significant predictors** | Expect modest baseline performance; consider collecting additional predictive variables (e.g., clinical measurements). |
| **Highly skewed “hours sitting” variable** | The capping reduced extreme values; a log‑transform (already attempted) may still be useful if engineered correctly. |
| **Weak numeric correlation** | Linear models may struggle; tree‑based ensembles could capture subtle interactions. |
| **Modest categorical associations** | Encode categoricals with target‑aware techniques (e.g., CatBoost encoding) to potentially improve signal. |
| **Feature engineering not realized** | Re‑run the engineering step ensuring the new columns are added; interaction terms may help. |
| **Small sample size (n = 100)** | Use repeated stratified CV or bootstrap to obtain stable performance estimates. |

---

## 9. Appendices – Artifact Inventory  

| File | Type | Brief Description |
|------|------|-------------------|
| `age_vs_diagnosis.png` | Target interaction plot (Age) | Visualizes distribution of Age across Diagnosis |
| `hours_vs_diagnosis.png` | Target interaction plot (Hours sitting) | Visualizes distribution of capped Hours across Diagnosis |
| `bivariate_*.png` (5 files) | Bivariate plots | Box/violin plots of each predictor vs. Diagnosis |
| `dist_*.png` (5 files) | Univariate distribution | Histograms / bar charts for each variable |
| `pairplot.png` | Pairwise scatter/box matrix | All numeric variables coloured by Diagnosis |
| `correlation_matrix.png` | Pearson correlation heatmap (numeric) |
| `categorical_association_matrix.png` | Cramér’s V heatmap (categorical) |
| `eda_report.html` | Full HTML EDA report (generated by pipeline) |
| `metrics.json` | Structured summary of all quantitative results |
| `metadata_profile.json` | Schema and descriptive statistics |
| `agent_plan_log.json` | Log of automated analysis steps |
| `agent_state.json` | Consolidated state (imputation, outliers, etc.) |
| `current_df.csv` | Final pre‑processed dataset (capped outliers) |

*All image files reside under `./sandbox_run/` as indicated in the logs.*

---

### Closing Remark  

The automated EDA reveals a clean but limited dataset: no missing data, modest variability, and no single feature that discriminates the target. A carefully tuned ensemble model with appropriate class‑imbalance handling, combined with robust cross‑validation, is the recommended path forward. Further data collection or richer clinical features will likely be required to achieve high predictive performance.
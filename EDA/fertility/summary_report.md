# Executive Summary – Fertility Dataset (Auto‑EDA)

**Dataset:** `fertility.csv`  
**Rows / Columns:** 100 × 11 (including engineered feature `age_hours_ratio`)  
**Target:** **Diagnosis** (binary – *Normal* / *Altered*)  
**Problem Type:** Classification  

---

## 1. Data Profile  

| Column | Data Type | Cardinality | Missing % | Key Summary |
|--------|-----------|------------|----------|-------------|
| Season | object | 4 | 0 % | spring (37), fall (31), winter (28) |
| Age | int64 | 10 | 0 % | 27 – 36 yr, mean 30.11, median 30 |
| Childish diseases | object | 2 | 0 % | yes (87), no (13) |
| Accident or serious trauma | object | 2 | 0 % | yes (44), no (56) |
| Surgical intervention | object | 2 | 0 % | yes (51), no (49) |
| High fevers in the last year | object | 3 | 0 % | > 3 months ago (63), no (28), < 3 months ago (9) |
| Frequency of alcohol consumption | object | 5 | 0 % | hardly ever/never (40), once a week (39), several times a week (19) |
| Smoking habit | object | 3 | 0 % | never (56), occasional (23), daily (21) |
| Number of hours spent sitting per day | int64 | 14 | 0 % | 1 – 342 h, mean 10.8, median 7, **highly skewed** (skew ≈ 9.85) |
| Diagnosis (target) | object | 2 | 0 % | Normal (88), Altered (12) |
| **age_hours_ratio** (engineered) | float64 | 47 | 0 % | Age ÷ (Hours + ε) – intended high‑signal feature |

*No missing values were detected; the imputation step therefore performed no fills.*

---

## 2. Distribution Visualizations  

All visual artifacts are stored in the working directory; they can be opened directly for visual inspection.

| Image | Description |
|-------|-------------|
| `dist_Season.png` | Bar chart of seasonal counts (spring, fall, winter). |
| `dist_Age.png` | Histogram of age (27‑36 yr) with mean line. |
| `dist_Childish_diseases.png` | Binary bar plot (yes / no). |
| `dist_Accident_or_serious_trauma.png` | Binary bar plot (yes / no). |
| `dist_Surgical_intervention.png` | Binary bar plot (yes / no). |
| `dist_High_fevers_in_the_last_year.png` | Three‑category bar plot. |
| `dist_Frequency_of_alcohol_consumption.png` | Five‑category bar plot. |
| `dist_Smoking_habit.png` | Three‑category bar plot. |
| `dist_Number_of_hours_spent_sitting_per_day.png` | Highly right‑skewed histogram (range 1‑342 h). |
| `dist_Diagnosis.png` | Target distribution (Normal 88 %, Altered 12 %). |
| `pairplot.png` | Pairwise scatter/box plots for all numeric variables (Age, Hours, age_hours_ratio). |
| `target_interactions.png` | Visualisation of each feature vs. the target (stacked bar / box). |
| `correlation_matrix.png` | Heat‑map of Pearson correlations among numeric features. |
| `bivariate_Age_vs_Diagnosis.png` | Age distribution split by Diagnosis. |
| `bivariate_Number_of_hours_spent_sitting_per_day_vs_Diagnosis.png` | Hours distribution split by Diagnosis. |
| `bivariate_Smoking_habit_vs_Diagnosis.png` | Smoking habit vs. Diagnosis. |
| `bivariate_Frequency_of_alcohol_consumption_vs_Diagnosis.png` | Alcohol consumption vs. Diagnosis. |

*All images are ≤ 70 KB, suitable for rapid review.*

---

## 3. Correlation & Interaction Analysis  

### 3.1 Pearson Correlation (numeric subset)

| Feature Pair | Correlation |
|--------------|-------------|
| Age ↔ age_hours_ratio | **+0.34** |
| Hours ↔ age_hours_ratio | **‑0.18** |
| Age ↔ Hours | **‑0.05** |

The correlation matrix (saved as `correlation_matrix.png`) shows no pair exceeding |0.85|, indicating low multicollinearity among numeric predictors.

### 3.2 Bivariate Visual Checks  

- **Age vs. Diagnosis** – Overlap of age distributions; no clear separation.  
- **Hours vs. Diagnosis** – Slightly higher median hours for *Altered* cases, but visual overlap is large.  
- **Smoking habit / Alcohol consumption** – Distribution of categories is similar across target classes.

Overall, visual and numeric interactions suggest weak linear relationships with the target.

---

## 4. Outlier & Distribution Diagnostics  

| Variable | IQR‑based Bounds | Outliers (count, %) |
|----------|------------------|----------------------|
| Age | 22 – 38 | 0 (0 %) |
| Hours (sitting) | –1 – 15 | 5 (5 %) |

The five hour‑related outliers lie above the upper bound (hours > 15). No action was taken beyond profiling, as the dataset is small and the outliers may carry signal.

---

## 5. Statistical Hypothesis Testing  

| Feature | Test | Statistic | p‑value | Significant? | Interpretation |
|---------|------|-----------|---------|--------------|----------------|
| Season | Chi‑Square (independence) | 4.1613 | 0.245 | No | No evidence of association with Diagnosis. |
| Age | Welch t‑test | 1.0435 | 0.313 | No | Mean ages of the two classes are not different. |
| Childish diseases | Chi‑Square | 0.0 | 1.0 | No | Perfectly balanced; no predictive power. |
| Accident/trauma | Chi‑Square | 1.2177 | 0.270 | No | No association. |
| Surgical intervention | Chi‑Square | 0.0547 | 0.815 | No | No association. |
| High fevers | Chi‑Square | 1.5452 | 0.462 | No | No association. |
| Alcohol consumption | Chi‑Square | 4.0263 | 0.402 | No | No association. |
| Smoking habit | Chi‑Square | 0.2153 | 0.898 | No | No association. |
| Hours (sitting) | Welch t‑test | –0.9024 | 0.369 | No | No difference between classes. |
| age_hours_ratio | Welch t‑test | 0.4771 | 0.642 | No | Engineered feature does not separate classes. |

**Result:** No feature reached conventional statistical significance (α = 0.05). Consequently, the `significant_predictors` list is empty.

---

## 6. Feature Engineering  

- **`age_hours_ratio`** = Age ÷ (Hours + ε)  
  - Rationale: Combine age and sedentary behavior into a single intensity metric.  
  - Correlation with target: Not significant (p = 0.64).  
  - Correlation with other numeric features: modest positive with Age (0.34) and negative with Hours (‑0.18).  

No other engineered features were added.

---

## 7. Predictive Modeling Blueprint  

| Aspect | Recommendation |
|--------|----------------|
| **Problem** | Binary classification (`Diagnosis`). |
| **Baseline** | Regularized Logistic Regression (L2 penalty). |
| **Advanced Models** | Random Forest, Gradient Boosting (XGBoost / LightGBM), Support Vector Classifier. |
| **Feature Selection** | <ul><li>Exclude any high‑cardinality identifiers (none present).</li><li>Rank features via cross‑validated permutation importance and mutual information.</li><li>Drop collinear features with |ρ| > 0.85 (none identified).</li></ul> |
| **Encoding** | One‑Hot encode all categorical variables (Season, Childish diseases, Accident, Surgical, High fevers, Alcohol, Smoking). |
| **Scaling** | StandardScaler for numeric columns (Age, Hours, age_hours_ratio). |
| **Validation** | Stratified 5‑fold cross‑validation. |
| **Metrics** | Balanced Accuracy, Macro F1, Precision‑Recall AUC, Confusion Matrix. |
| **Over‑fitting Controls** | <ul><li>L1/L2 regularization (logistic, linear SVM).</li><li>Tree depth ≤ 5, min samples leaf ≥ 5 (RF/GBM).</li><li>Hyper‑parameter search confined to inner CV folds.</li></ul> |
| **Expected Difficulty** | Small sample size (100 rows) → high variance; robust CV essential. |

**Executive Note:** Given the lack of statistically significant predictors, model performance will likely be modest. Emphasis should be placed on robust validation and possibly augmenting the dataset (e.g., collecting more records or external features).

---

## 8. Key Take‑aways & Recommendations  

1. **Data Quality** – No missing values; all columns are clean.  
2. **Target Imbalance** – 88 % Normal, 12 % Altered – consider class‑weighting or resampling techniques.  
3. **Predictive Signal** – Individual features show weak or no association with the target; the engineered ratio does not improve separation.  
4. **Modeling Strategy** – Start with a regularized logistic baseline; explore tree‑based ensembles for potential non‑linear interactions.  
5. **Future Work** –  
   - Acquire additional samples to reduce variance.  
   - Investigate external health indicators (e.g., BMI, hormonal levels).  
   - Perform feature‑level interaction mining (e.g., Age × Smoking) to uncover hidden patterns.  

---

*All visual assets referenced above are available in the current directory and can be embedded directly into a report or notebook for stakeholder review.*
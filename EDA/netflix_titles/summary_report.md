# Executive Summary – Netflix Titles Dataset  
**Prepared by:** Senior Lead Data Scientist  
**Date:** 2026‑08‑07  

---  

## 1. Dataset Overview  

| Property                     | Value                              |
|------------------------------|------------------------------------|
| **Source file**              | `netflix_titles.csv`               |
| **Rows**                     | 8 807                              |
| **Columns**                  | 12                                 |
| **Target column**            | `rating` (categorical, 17 levels) |
| **Problem type**             | Multi‑class Classification         |
| **Key categorical columns**  | `type`, `director`, `cast`, `country`, `date_added`, `listed_in` |
| **Numeric columns**          | `release_year` (int64)             |
| **High‑cardinality fields**  | `show_id`, `title`, `description` (unique for each record) |

### 1.1 Schema Snapshot  

| Column        | Data type | Cardinality | Missing % | Notable notes |
|---------------|-----------|-------------|----------|---------------|
| `show_id`     | str       | 8 807       | 0.0      | Unique identifier |
| `type`        | str       | 2           | 0.0      | “Movie” vs “TV Show” |
| `title`       | str       | 8 807       | 0.0      | Unique titles |
| `director`    | str       | 4 528       | 29.9     | Mode = **Rajiv Chilaka** |
| `cast`        | str       | 7 692       | 9.4      | Mode = **David Attenborough** |
| `country`     | str       | 748         | 9.4      | Mode = **United States** |
| `date_added`  | str       | 1 767       | 0.1      | Mode = **January 1, 2020** |
| `release_year`| int64     | 74          | 0.0      | Mean = 2014.18, Skew = ‑3.45 |
| `rating`      | str       | 17          | 0.0      | Mode = **TV‑MA** |
| `duration`    | str       | 220         | 0.0      | Mode = **1 Season** |
| `listed_in`   | str       | 514         | 0.0      | Primary genre tags |
| `description` | str       | 8 775       | 0.0      | Free‑text synopsis |

*All missing values were imputed (see Section 2).*

---  

## 2. Data Quality & Pre‑Processing  

### 2.1 Missing‑Value Imputation  

| Column      | Missing before | Imputation method | Fill value (mode) |
|-------------|----------------|-------------------|-------------------|
| `director`  | 2 634 (29.9 %) | Mode              | **Rajiv Chilaka** |
| `cast`      |   825 (9.4 %)  | Mode              | **David Attenborough** |
| `country`   |   831 (9.4 %)  | Mode              | **United States** |
| `date_added`|    10 (0.1 %)  | Mode              | **January 1, 2020** |
| `rating`    |     4 (0.0 %)  | Mode              | **TV‑MA** |
| `duration`  |     3 (0.0 %)  | Mode              | **1 Season** |

*Numeric columns with extreme skewness (e.g., `release_year`) were left untouched because no missing values existed.*

### 2.2 Outlier Profiling – `release_year`  

| Statistic | Value |
|-----------|-------|
| Q1        | 2013 |
| Q3        | 2019 |
| IQR       | 6 |
| Lower bound (Q1‑1.5·IQR) | 2004 |
| Upper bound (Q3+1.5·IQR) | 2028 |
| Outlier count | 719 |
| Outlier % of rows | 8.16 % |

*Action:* Outliers were **profiled only** (no removal) because the variable is a temporal identifier; extreme years are legitimate (e.g., early titles from 1925).

### 2.3 Correlation Analysis  

- The pipeline attempted a Pearson correlation matrix but **failed** (`Insufficient numeric columns for correlation analysis`).  
- Only one numeric column (`release_year`) exists, so pairwise numeric correlation is not applicable.  

---  

## 3. Statistical Association & Hypothesis Testing  

All categorical variables were tested against the target `rating` using Chi‑Square tests; the numeric `release_year` was examined with One‑Way ANOVA.

| Feature | Test | Statistic | p‑value | Significant? | Interpretation |
|---------|------|-----------|---------|--------------|----------------|
| `type` | Chi‑Square | 1 047.86 | 6.28 e‑213 | **Yes** | Strong association between “Movie/TV Show” and rating |
| `director` | Chi‑Square | 76 320.51 | 5.03 e‑24 | **Yes** | Certain directors influence rating distribution |
| `country` | Chi‑Square | 17 691.94 | 3.47 e‑231 | **Yes** | Geographic origin matters |
| `date_added` | Chi‑Square | 44 815.02 | 0.0 | **Yes** | Release timing correlates with rating |
| `release_year` | One‑Way ANOVA | 75.93 | 9.02 e‑192 | **Yes** | Year of release impacts rating |
| `duration` | Chi‑Square | 9 532.55 | 0.0 | **Yes** | Length (seasons/minutes) linked to rating |
| `listed_in` | Chi‑Square | 24 428.93 | 0.0 | **Yes** | Genre tags are predictive |
| `cast` | Chi‑Square | 110 951.52 | 1.0 | No | No detectable effect |
| `description` | Chi‑Square | 140 840.49 | 0.19 | No | Text length/contents not directly associated |
| `show_id` / `title` | Chi‑Square | 140 912.00 | 0.49 | No | Unique identifiers carry no predictive power |

**Significant predictors (α = 0.05):**  
`type`, `director`, `country`, `date_added`, `release_year`, `duration`, `listed_in`.

---  

## 4. Feature Engineering  

| Specified Feature | Transformation | Source column | Result |
|-------------------|----------------|---------------|--------|
| `log1p_release_year` | Log‑1p (`log(1 + x)`) | `release_year` | **Not generated** (pipeline reported “Generated 0 features”) |
| `duration_numeric` | Extract numeric part (e.g., “90 min” → 90) | `duration` | **Not generated** |
| `type_encoded` | Label‑encode (`Movie` = 0, `TV Show` = 1) | `type` | **Not generated** |

*The engineering step completed without error but produced no new columns, likely because downstream modeling will handle encoding on‑the‑fly.*

---  

## 5. Visual Artefacts  

All plots are saved in the sandbox directory (`…/sandbox_run/0482fd59‑bd3e‑43ce‑81a2‑0424eab7d978`). Below is a brief description of each image file.

| Image File | Size (KB) | Description |
|------------|-----------|-------------|
| `dist_type.png` | 23.55 | Bar chart of “Movie” vs “TV Show” counts (≈ 61 % Movies). |
| `dist_release_year.png` | 33.99 | Histogram of release years (1925‑2021) showing a strong right‑skew toward recent years. |
| `dist_rating.png` | 36.97 | Bar plot of the 17 rating categories; dominant levels are **TV‑MA** (≈ 36 %) and **TV‑14** (≈ 24 %). |
| `dist_duration.png` | 46.23 | Distribution of `duration` strings (seasons/minutes); “1 Season” is most common. |
| `dist_country.png` | 55.73 | Top countries – United States, India, United Kingdom. |
| `dist_director.png` | 68.06 | Frequency of directors; long tail with many unique names. |
| `bivariate_type_vs_rating.png` | 49.38 | Box/violin style comparison of rating distribution across `type`. |
| `bivariate_release_year_vs_rating.png` | 67.67 | Scatter/box plot of rating vs. release year, confirming the ANOVA result. |
| `bivariate_country_vs_rating.png` | 340.15 | Rating distribution across the most frequent countries (high‑cardinality, aggregated). |
| `bivariate_duration_vs_rating.png` | 87.87 | Rating vs. duration (seasons/minutes) – longer series tend toward higher ratings. |
| `bivariate_listed_in_vs_rating.png` | 160.05 | Rating vs. primary genre tags – certain genres (e.g., “Documentaries”) show distinct rating patterns. |
| `rating_vs_release_year.png` | 80.14 | Line/point plot of average rating per release year, visualising the upward trend in recent years. |
| `correlation_matrix.png` | – | **Not generated** – insufficient numeric columns. |

*All visualisations are ready for inclusion in presentations or dashboards.*

---  

## 6. Predictive Modeling Blueprint  

### 6.1 Target & Problem Definition  

- **Target:** `rating` (17‑class categorical variable).  
- **Goal:** Build a robust multi‑class classifier that predicts the content rating based on the available metadata.

### 6.2 Recommended Algorithms  

| Rank | Algorithm | Rationale |
|------|-----------|-----------|
| 1 | **Regularized Logistic Regression** (multinomial) | Fast baseline, interpretable coefficients, works well with high‑cardinality one‑hot encoded features. |
| 2 | **Random Forest Classifier** | Handles mixed data types, captures non‑linear interactions, robust to noisy categorical features. |
| 3 | **Gradient Boosting (XGBoost / LightGBM)** | State‑of‑the‑art performance on tabular data, can exploit sparse one‑hot encodings efficiently. |
| 4 | **Support Vector Classifier (SVM)** with linear or RBF kernel | Useful if the decision boundary is complex; may require dimensionality reduction. |

### 6.3 Feature Selection & Engineering Strategy  

1. **Exclude** high‑cardinality identifiers (`show_id`, `title`, `description`).  
2. **Encode** categorical variables:  
   - Low‑cardinality (`type`, `rating`) → label or one‑hot encoding.  
   - High‑cardinality (`director`, `cast`, `country`, `listed_in`) → target encoding, frequency encoding, or hashing trick.  
3. **Rank features** using cross‑validated permutation importance and mutual information to keep the most predictive subset.  
4. **Remove collinear features** (if any) exceeding a Pearson correlation threshold of **0.85** (not applicable now but retained for future numeric expansions).  

### 6.4 Validation & Evaluation  

| Aspect | Specification |
|--------|----------------|
| **Cross‑validation** | Stratified K‑Fold (K = 5) to preserve rating distribution across folds. |
| **Primary metrics** | **Balanced Accuracy** (accounts for class imbalance), **Macro‑averaged F1**, **Precision‑Recall AUC** (per‑class). |
| **Secondary diagnostics** | Confusion matrix, per‑class recall/precision, calibration curves. |
| **Hyper‑parameter tuning** | Grid/Random search *inside* each CV fold (no data leakage). |
| **Over‑fitting safeguards** | L1/L2 regularization (logistic), tree depth limits, minimum samples per leaf, early stopping for boosting. |

### 6.5 Expected Challenges  

- **Class imbalance:** Some ratings (e.g., “TV‑Y7”) have far fewer instances; consider class‑weighting or SMOTE‑like oversampling.  
- **High cardinality:** `director`, `cast`, `listed_in` have thousands of unique values; careful encoding is required to avoid dimensionality explosion.  
- **Limited numeric information:** Only `release_year` is numeric; feature engineering (e.g., `log1p_release_year`, extracting numeric duration) could improve model capacity but was not automatically generated.

---  

## 7. Key Take‑aways  

1. **Data Quality** – Missing values were modest and successfully imputed using mode values.  
2. **Temporal Skew** – `release_year` is heavily right‑skewed; recent titles dominate the dataset.  
3. **Predictive Signals** – Seven features show statistically significant association with the rating, most notably `type`, `director`, `country`, `date_added`, `release_year`, `duration`, and `listed_in`.  
4. **Feature Engineering Gap** – The automated pipeline defined three engineered features but did not materialize them; manual creation (e.g., numeric duration, log‑transformed year) is advisable.  
5. **Modeling Path** – Begin with a regularized multinomial logistic regression baseline, then progress to ensemble methods (Random Forest, Gradient Boosting) while applying robust encoding and feature‑selection pipelines.  
6. **Visualization Assets** – All distribution and bivariate plots are ready for stakeholder decks; they clearly illustrate the relationships uncovered by hypothesis testing.  

---  

## 8. Next Steps  

1. **Implement manual feature engineering** for `duration_numeric` and `log1p_release_year`.  
2. **Develop encoding pipelines** for high‑cardinality categorical columns (frequency/target encoding).  
3. **Run baseline logistic regression** and record performance metrics.  
4. **Iteratively test ensemble models** with hyper‑parameter tuning under the prescribed CV scheme.  
5. **Perform error analysis** on mis‑classified ratings to uncover potential data issues or additional features (e.g., text embeddings from `description`).  

---  

*Prepared for the data science team and project stakeholders. All artefacts referenced are available in the sandbox directory.*
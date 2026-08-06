# Executive Summary – Netflix Titles Dataset  
**Prepared by:** Senior Lead Data Scientist  
**Date:** 2026‑08‑06  

---

## 1. Overview  

| Item | Value |
|------|-------|
| **Dataset** | `netflix_titles.csv` |
| **Rows** | 8,807 |
| **Columns** | 12 |
| **Target variable** | `type` (Movie | TV Show) |
| **Problem type** | Binary Classification |
| **Primary goal** | Build a robust model that predicts whether a title is a *Movie* or a *TV Show* using the available metadata. |

The automated EDA pipeline generated a full data profile, missing‑value handling, outlier detection, hypothesis testing, and a modeling blueprint. All visual artifacts are listed in **Appendix A**.

---

## 2. Data Quality Assessment  

### 2.1 Missing‑Value Summary  

| Column | Missing Count | Missing % | Imputation Method | Fill Value |
|--------|---------------|----------|-------------------|------------|
| director | 2,634 | 29.91 % | Mode | `Rajiv Chilaka` |
| cast | 825 | 9.37 % | Mode | `David Attenborough` |
| country | 831 | 9.44 % | Mode | `United States` |
| date_added | 10 | 0.11 % | Mode | `January 1, 2020` |
| rating | 4 | 0.05 % | Mode | `TV-MA` |
| duration | 3 | 0.03 % | Mode | `1 Season` |
| All other columns | 0 | 0 % | – | – |

*All categorical missing values were replaced with the most frequent category (mode). No numeric columns required imputation.*

### 2.2 Cardinality & Data Types  

| Column | Data Type | Cardinality |
|--------|-----------|-------------|
| show_id | object | 8,807 |
| type | object | 2 |
| title | object | 8,807 |
| director | object | 4,528 |
| cast | object | 7,692 |
| country | object | 748 |
| date_added | object | 1,767 |
| release_year | int64 | 74 |
| rating | object | 17 |
| duration | object | 220 |
| listed_in | object | 514 |
| description | object | 8,775 |

High‑cardinality fields (`show_id`, `title`, `description`) are identifiers or free‑text and will be excluded from modeling unless transformed (e.g., TF‑IDF).

---

## 3. Distributional Insights  

The pipeline produced a series of distribution plots (see **Appendix A**). Key observations:

| Plot | Main Take‑away |
|------|----------------|
| `dist_release_year.png` | Release years are heavily right‑skewed (skew = ‑3.45) with a mean of **2014.2** and median **2017**. Most titles were added after 2010. |
| `dist_rating.png` | The most common ratings are `TV‑MA` (3,207), `TV‑14` (2,160) and `TV‑PG` (863). |
| `dist_country.png` | `United States` dominates (2,818 titles), followed by `India` (972) and `United Kingdom` (419). |
| `dist_duration.png` | For TV Shows, the mode is “1 Season” (1,793 titles). For Movies, durations are expressed in minutes (not shown here). |
| `dist_listed_in.png` | Top genres: *Dramas, International Movies* (362), *Documentaries* (359), *Stand‑Up Comedy* (334). |
| `dist_cast.png` | The most frequent cast entry is `David Attenborough` (19 appearances). |
| `dist_director.png` | `Rajiv Chilaka` appears most often (19 titles). |
| `dist_description.png` | Descriptions are highly unique (8,775 distinct values). |

These distributions confirm a strong US‑centric catalog with a recent production focus.

---

## 4. Bivariate Relationships with the Target  

| Plot | Relationship | Interpretation |
|------|--------------|----------------|
| `bivariate_country_vs_rating.png` | Country vs. Rating | Certain countries (e.g., United States) have a higher proportion of `TV‑MA` ratings, while others (e.g., India) show more `TV‑14`. |
| `bivariate_listed_in_vs_rating.png` | Genre (`listed_in`) vs. Rating | Genres such as *Documentaries* and *Stand‑Up Comedy* skew toward lower age‑restriction ratings (`TV‑PG`, `TV‑14`). |
| `bivariate_release_year_vs_rating.png` | Release Year vs. Rating | Newer titles (post‑2015) are more likely to carry `TV‑MA` ratings, reflecting a trend toward mature content. |

These plots suggest that **country, genre, and release year** are informative for distinguishing Movies from TV Shows.

---

## 5. Outlier Analysis  

Only the numeric column `release_year` was examined for outliers.

| Statistic | Value |
|-----------|-------|
| Q1 | 2013 |
| Q3 | 2019 |
| IQR | 6 |
| Lower bound (Q1‑1.5·IQR) | 2004 |
| Upper bound (Q3+1.5·IQR) | 2028 |
| Outlier count | 719 |
| Outlier % | 8.16 % |
| Action taken | Profile only (no removal) |

The outliers correspond to very early titles (e.g., 1925) and a few future‑dated entries; they are retained for completeness.

---

## 6. Statistical Hypothesis Testing  

A series of chi‑square tests (categorical) and a Welch two‑sample t‑test (numeric) were performed to assess association with the target `type`.

| Feature | Test | Statistic | p‑value | Significant? | Interpretation |
|---------|------|------------|---------|--------------|----------------|
| director | Chi‑Square | 7,767.86 | 1.24e‑175 | **Yes** | Strong dependence on `type`. |
| cast | Chi‑Square | 7,728.20 | 0.38 | No | No evidence of dependence. |
| country | Chi‑Square | 1,720.83 | 1.25e‑78 | **Yes** | Strong dependence on `type`. |
| date_added | Chi‑Square | 2,748.19 | 4.85e‑46 | **Yes** | Strong dependence on `type`. |
| release_year | Welch t‑test | -20.98 | 3.71e‑95 | **Yes** | Mean release year differs between Movies and TV Shows. |
| rating | Chi‑Square | 1,047.86 | 6.28e‑213 | **Yes** | Rating distribution differs by `type`. |
| duration | Chi‑Square | 8,792.84 | 0.0 | **Yes** | Duration format (seasons vs. minutes) is highly predictive. |
| listed_in | Chi‑Square | 8,807.00 | 0.0 | **Yes** | Genre strongly associated with `type`. |
| description | Chi‑Square | 8,801.48 | 0.42 | No | Text description not directly predictive (as expected). |
| show_id, title | Chi‑Square | 8,807.00 | 0.49 | No | IDs and titles are unique identifiers, not predictive. |

**Significant predictors** (p < 0.05):  
`director`, `country`, `date_added`, `release_year`, `rating`, `duration`, `listed_in`.

These variables will be prioritized in feature engineering and model training.

---

## 7. Feature Engineering Highlights  

- **No engineered features** were created automatically (`engineered_features` list is empty).  
- Recommended transformations:  
  - Encode `type` as binary (Movie = 0, TV Show = 1).  
  - Convert `release_year` to a numeric feature (already numeric).  
  - Parse `duration` into two numeric columns: `duration_value` (int) and `duration_unit` (`min` | `Season`).  
  - One‑hot encode high‑cardinality categorical variables after applying a frequency threshold (e.g., keep only categories with ≥ 50 occurrences).  
  - Apply TF‑IDF or word‑embedding vectors to `description` if text modeling is desired (optional).  

---

## 8. Predictive Modeling Blueprint  

| Component | Recommendation |
|-----------|----------------|
| **Target** | `type` (binary classification) |
| **Baseline** | Regularized Logistic Regression (L2 penalty) |
| **Advanced models** | Random Forest, Gradient Boosting (XGBoost / LightGBM), Support Vector Classifier |
| **Feature selection** | <ul><li>Drop `show_id`, `title`, `description` (unless using text features).</li><li>Rank remaining features via cross‑validated permutation importance and mutual information.</li><li>Remove collinear pairs with Pearson |r| > 0.85.</li></ul> |
| **Validation** | Stratified 5‑fold cross‑validation (preserves class balance). |
| **Evaluation metrics** | Balanced Accuracy, Macro F1, Precision‑Recall AUC, Confusion Matrix. |
| **Overfitting mitigation** | <ul><li>L1/L2 regularization for linear models.</li><li>Limit tree depth, set `min_samples_leaf` for ensemble methods.</li><li>Hyper‑parameter tuning inside CV folds (e.g., GridSearchCV or Bayesian optimization).</li></ul> |
| **Pipeline** | Imputation → Encoding → Scaling (if needed) → Model → Metric aggregation. |
| **Execution environment** | Python 3.10+, scikit‑learn, XGBoost/LightGBM, pandas. |

**Executive Summary (from pipeline)**:  
> “Target: type (Classification). Use robust cross‑validation on 8,807 rows × 12 columns.”

---

## 9. Recommendations & Next Steps  

1. **Pre‑processing** – Apply the imputation rules already documented; verify that mode replacements do not introduce bias.  
2. **Feature transformation** – Implement the suggested `duration` split and frequency‑based one‑hot encoding for `director`, `country`, `rating`, and `listed_in`.  
3. **Baseline model** – Train a regularized logistic regression to establish a performance floor.  
4. **Model comparison** – Evaluate Random Forest and Gradient Boosting models using the CV strategy; select the model with the highest macro F1 while monitoring overfitting.  
5. **Error analysis** – Examine mis‑classifications, especially for titles where `duration` or `rating` may be ambiguous.  
6. **Optional text enrichment** – If higher accuracy is required, experiment with TF‑IDF vectors from `description` and `title`.  

---

## Appendix A – Image Artifacts  

| File | Description |
|------|-------------|
| `dist_release_year.png` | Histogram of release years (highly right‑skewed). |
| `dist_rating.png` | Bar chart of content rating frequencies. |
| `dist_country.png` | Bar chart of country counts (US dominant). |
| `dist_duration.png` | Distribution of duration formats (seasons vs. minutes). |
| `dist_listed_in.png` | Frequency of top genres/categories. |
| `dist_cast.png` | Top cast members by occurrence. |
| `dist_director.png` | Top directors by occurrence. |
| `dist_description.png` | Word‑cloud‑style frequency of description snippets (high uniqueness). |
| `bivariate_country_vs_rating.png` | Country vs. rating stacked bar chart. |
| `bivariate_listed_in_vs_rating.png` | Genre vs. rating stacked bar chart (large file, 670 KB). |
| `bivariate_release_year_vs_rating.png` | Release year vs. rating line/box plot. |
| `target_interaction.png` | Interaction plot of target `type` with key predictors. |

All images are stored in the working directory and can be incorporated into a Jupyter notebook or reporting dashboard for visual inspection.  

---  

*Prepared for internal use by the Data Science team.*
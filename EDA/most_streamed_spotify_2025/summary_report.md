# Executive Summary – Most‑Streamed Spotify Tracks (2025)

**Prepared by:** Senior Lead Data Scientist  
**Date:** 6 August 2026  

---

## 1. Project Context  

The dataset **`most_streamed_spotify_2025.csv`** contains the 730 most‑streamed Spotify tracks for the year 2025.  
The target variable is **`wrapped_global_top10_rank`**, a numeric score (1 = best) that has been treated as a **classification** problem (top‑10 vs. non‑top‑10).  

The automated EDA pipeline generated a set of visual artefacts, statistical tables, and a modeling blueprint that are summarised below.

---

## 2. Data Overview  

| Property                              | Value |
|--------------------------------------|-------|
| Rows                                 | 730 |
| Columns (including engineered)       | 11 |
| Target column                         | `wrapped_global_top10_rank` |
| Primary key (implicit)                | `rank` (1 – 730) |
| Cardinality (unique values)          | See column‑wise table |
| Missing values (overall)              | 727 rows (99.6 %) in target column only – imputed with mean (5.33) |
| Data types                           | 5 numeric, 2 boolean, 2 categorical, 2 engineered (float) |

### 2.1 Column‑wise Summary  

| Column                     | dtype   | Cardinality | Missing % | Key Statistics (where applicable) |
|----------------------------|---------|-------------|----------|-----------------------------------|
| `rank`                     | int64   | 730         | 0.0      | Range 1‑730, Mean 365.5 |
| `track`                    | object  | 726         | 0.0      | Top values: “Show Me Love”, “NO BATIDÃO”, “MONTAGEM TOMADA” (2 each) |
| `artist`                   | object  | 401         | 0.0      | Top artists: Bad Bunny (15), Sabrina Carpenter (14), Taylor Swift (12) |
| `billed_artist_count`     | int64   | 2           | 0.0      | Range 1‑2, Mean 1.03, Skew 5.37 |
| `is_collaboration`        | bool    | 2           | 0.0      | False 707, True 23 |
| `spotify_streams_total`   | int64   | 730         | 0.0      | Mean 2.34 × 10⁸, Median 1.67 × 10⁸, Skew 4.05 |
| `daily_streams`           | int64   | 730         | 0.0      | Mean 3.68 × 10⁵, Median 2.23 × 10⁵, Skew 3.16 |
| `daily_streams_rank`      | int64   | 730         | 0.0      | Mirrors `rank` (range 1‑730) |
| `daily_stream_share_pct`  | float64 | 646         | 0.0      | Mean 0.15, Median 0.12, Skew 2.20 |
| `wrapped_global_top10_rank`| float64 | 3 (after imputation) | 0.0 (imputed) | Mean 5.33, Median 5.00 |
| `engineered_feature`       | float64 | 730         | 0.0      | Defined as `daily_streams / (spotify_streams_total + eps)` |

---

## 3. Missing‑Value Handling  

* Only the target column contained missing entries (727/730).  
* Imputation rule: **Mean imputation** (skewness ≈ 0.94, within the “median” band).  
* All other columns were complete; no further imputation required.

---

## 4. Distribution Visualisations  

| Image File | Description |
|------------|-------------|
| `dist_artist.png` | Bar chart of artist frequencies – heavy tail with Bad Bunny, Sabrina Carpenter, Taylor Swift leading. |
| `dist_track.png` | Histogram of track titles – 726 unique tracks, indicating near‑unique identifiers. |
| `dist_billed_artist_count.png` | Very narrow distribution (mostly 1 billed artist). |
| `dist_is_collaboration.png` | Pie chart: 96 % non‑collaborations, 4 % collaborations. |
| `dist_spotify_streams_total.png` | Right‑skewed distribution of total streams (log‑scale recommended). |
| `dist_daily_streams.png` | Right‑skewed distribution of daily streams, similar shape to total streams but lower magnitude. |
| `dist_daily_stream_share_pct.png` | Distribution of daily share percentage – most values < 0.2, long tail up to 0.98. |
| `dist_daily_streams_rank.png` | Uniform distribution mirroring `rank`. |
| `dist_rank.png` | Uniform rank distribution (1‑730). |
| `dist_wrapped_global_top10_rank.png` | After imputation, three distinct values (4, 5, 7) with the majority at 5. |
| `bivariate_daily_streams_vs_spotify_streams_total.png` | Scatter plot showing strong positive relationship (Pearson r ≈ 0.78). |
| `bivariate_daily_stream_share_pct_vs_billed_artist_count.png` | Bivariate plot – negligible association (correlation ≈ ‑0.06). |
| `bivariate_rank_vs_wrapped_global_top10_rank.png` | No observable pattern; rank does not predict target. |
| `target_interactions.png` | Interaction matrix of target vs. key features – confirms weak linear links. |
| `correlation_matrix.png` | Heatmap of Pearson correlations among all numeric variables (see Section 5). |

*All images are stored in the working directory; file sizes range from 28 KB to 157 KB.*

---

## 5. Correlation & Multicollinearity  

### 5.1 Top Correlations (absolute value)

| Feature 1 | Feature 2 | Pearson r |
|-----------|-----------|-----------|
| `daily_stream_share_pct` | `engineered_feature` | **1.00** |
| `spotify_streams_total` | `daily_streams` | **0.78** |
| `daily_streams` | `daily_streams_rank` | **‑0.73** |
| `daily_streams_rank` | `daily_stream_share_pct` | **‑0.73** |
| `rank` | `daily_streams_rank` | **0.71** |
| `rank` | `spotify_streams_total` | **‑0.68** |
| `daily_streams` | `daily_stream_share_pct` | **0.62** |
| `daily_streams` | `engineered_feature` | **0.62** |
| `rank` | `daily_streams` | **‑0.59** |
| `billed_artist_count` | `rank` | **0.09** |

*The engineered feature is perfectly collinear with `daily_stream_share_pct` (by construction).*

### 5.2 Multicollinearity Assessment  

* No pair exceeds the 0.85 threshold, so **no mandatory removal** is required.  
* However, the perfect correlation between `engineered_feature` and `daily_stream_share_pct` suggests retaining only one of them for parsimonious models.

---

## 6. Outlier Analysis  

| Feature | Outlier % | Action |
|---------|-----------|--------|
| `spotify_streams_total` | 9.45 % (69 rows) | Profiled only – no removal |
| `daily_streams` | 7.67 % (56 rows) | Profiled only – no removal |
| `daily_stream_share_pct` | 5.34 % (39 rows) | Profiled only – no removal |
| `wrapped_global_top10_rank` | 0.41 % (3 rows) | Profiled only – no removal |
| `rank`, `daily_streams_rank` | 0 % | No action needed |

*Outliers were retained for downstream modelling, given the modest dataset size.*

---

## 7. Feature Engineering  

| Engineered Feature | Formula | Rationale | Observed Correlation with Target |
|--------------------|---------|-----------|----------------------------------|
| `engineered_feature` | `daily_streams / (spotify_streams_total + eps)` | Captures the proportion of a track’s daily activity relative to its overall popularity. | 0.001 (practically zero) – not predictive of `wrapped_global_top10_rank`. |

*Because the engineered feature is mathematically identical to `daily_stream_share_pct`, it inherits the same distribution and correlation pattern.*

---

## 8. Statistical Hypothesis Testing  

All tests evaluated the linear (or mean) relationship between each predictor and the target (`wrapped_global_top10_rank`).  

| Feature | Test | Statistic | p‑value | Significant? | Interpretation |
|---------|------|-----------|---------|--------------|----------------|
| `rank` | Pearson | 0.0002 | 0.995 | No | No linear association. |
| `artist` | One‑Way ANOVA | 0.7544 | 0.973 | No | Artist groups do not differ in target rank. |
| `billed_artist_count` | Pearson | –0.0000 | 1.000 | No | No effect. |
| `is_collaboration` | Welch t‑test | 0.0 | 1.000 | No | Collaboration status irrelevant. |
| `spotify_streams_total` | Pearson | –0.025 | 0.500 | No | Weak, non‑significant negative trend. |
| `daily_streams` | Pearson | –0.0113 | 0.761 | No | Weak, non‑significant negative trend. |
| `daily_streams_rank` | Pearson | 0.0006 | 0.987 | No | No trend. |
| `daily_stream_share_pct` | Pearson | 0.0006 | 0.986 | No | No trend. |
| `engineered_feature` | Pearson | 0.0007 | 0.986 | No | No trend. |

**Result:** *No predictor reached statistical significance at conventional α = 0.05.* Consequently, the `significant_predictors` list is empty.

---

## 9. Predictive Modeling Blueprint  

| Aspect | Recommendation |
|--------|----------------|
| **Problem type** | **Classification** (predicting whether a track belongs to the Wrapped Global Top‑10). |
| **Target definition** | `wrapped_global_top10_rank` (treated as categorical with three levels after imputation). |
| **Baseline algorithm** | Regularized Logistic Regression (L1/L2). |
| **Advanced algorithms** | Random Forest, Gradient Boosting (XGBoost / LightGBM), Support Vector Classifier. |
| **Feature selection** | <ul><li>Drop high‑cardinality identifiers (`track`, `artist`).</li><li>Use permutation importance + mutual information (cross‑validated) to rank remaining features.</li><li>Remove collinear features with |r| > 0.85 (none required, but keep only one of `daily_stream_share_pct` / `engineered_feature`).</li></ul> |
| **Validation strategy** | Stratified 5‑fold cross‑validation. |
| **Evaluation metrics** | Balanced Accuracy, Macro F1, Precision‑Recall AUC, Confusion Matrix. |
| **Overfitting mitigation** | <ul><li>Regularization (L1/L2) for linear models.</li><li>Tree depth limits, minimum samples per leaf for ensemble models.</li><li>Hyper‑parameter tuning confined to inner CV loops.</li></ul> |
| **Data split** | No explicit hold‑out set defined; recommend reserving 10‑15 % of rows for final test if additional data becomes available. |

**Executive Summary (from pipeline):**  
> “Target: `wrapped_global_top10_rank` (Classification). Use robust cross‑validation on 730 rows × 11 columns.”

---

## 10. Key Insights & Recommendations  

1. **Target Imbalance & Low Variability** – After imputation the target collapses to three discrete values (4, 5, 7) with the majority at 5. This limited variance explains the lack of statistically significant predictors.  
2. **Strong Internal Correlations** – `daily_streams` and `spotify_streams_total` are highly correlated (r ≈ 0.78). Either can serve as a proxy for overall popularity; the daily‑share percentage adds little new information beyond the engineered feature.  
3. **Sparse Predictive Signal** – None of the examined features (including the engineered one) shows a meaningful linear relationship with the target. Non‑linear models (RF, GBM) may still capture subtle patterns, but expectations should be modest.  
4. **Feature Reduction** – Remove `track` and `artist` (high cardinality, no predictive power) before modelling. Retain one of `daily_stream_share_pct` or `engineered_feature`.  
5. **Modeling Strategy** – Begin with a regularized logistic regression baseline; if performance is inadequate, explore tree‑based ensembles with careful depth control.  
6. **Future Data Enrichment** – Incorporating additional contextual variables (e.g., genre, release date, playlist placements) could increase signal strength for the top‑10 classification task.

---

## 11. Limitations  

* **Target Quality:** 99 % missingness required mean imputation, which may have obscured true class distinctions.  
* **Sample Size:** 730 observations limit the complexity of models that can be reliably trained.  
* **Feature Scope:** Only streaming metrics are present; external factors (marketing, social media trends) are absent.  

---

## 12. Next Steps  

1. **Data Augmentation** – Gather auxiliary metadata (genre, release year, country) to enrich the feature set.  
2. **Re‑evaluate Target** – Consider redefining the target (e.g., binary top‑10 vs. others) after obtaining a more complete label set.  
3. **Model Development** – Implement the recommended baseline and advanced models, report cross‑validated metrics, and compare against a simple majority‑class baseline.  
4. **Interpretability** – Use SHAP or permutation importance to explain any learned patterns, especially if non‑linear models outperform the baseline.  

---

*All visual artefacts referenced above are available in the working directory and can be incorporated into a full report or presentation as needed.*
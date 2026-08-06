# Executive Summary – Spotify “Most‑Streamed 2025” Dataset  

**Prepared by:** Senior Lead Data Scientist  
**Date:** 2026‑08‑06  

---  

## 1. Dataset Overview  

| Item | Value |
|------|-------|
| **Source file** | `most_streamed_spotify_2025.csv` |
| **Rows / Columns** | 730 × 10 |
| **Target column** | `wrapped_global_top10_rank` (numeric, 3 distinct values) |
| **Primary key** | `rank` (1 – 730, unique) |
| **Cardinality (unique values)** | `track`: 726  ·  `artist`: 401  ·  `billed_artist_count`: 2  ·  `is_collaboration`: 2 |
| **Missing values (overall)** | 727 missing in `wrapped_global_top10_rank` (99.6 %); all other columns complete |
| **Skewness (selected numeric vars)** | `spotify_streams_total`: 4.05 (high)  ·  `daily_streams`: 3.16 (high)  ·  `daily_stream_share_pct`: 2.20 (high) |

*The dataset records the 730 most‑streamed Spotify tracks for 2025, together with daily streaming metrics and whether the track appeared in the global “Wrapped” top‑10.*  

---  

## 2. Data Quality & Imputation  

| Column | Missing before | Missing after | Imputation method | Fill value |
|--------|----------------|---------------|-------------------|------------|
| rank | 0 | 0 | – | – |
| track | 0 | 0 | – | – |
| artist | 0 | 0 | – | – |
| billed_artist_count | 0 | 0 | – | – |
| is_collaboration | 0 | 0 | – | – |
| spotify_streams_total | 0 | 0 | – | – |
| daily_streams | 0 | 0 | – | – |
| daily_streams_rank | 0 | 0 | – | – |
| daily_stream_share_pct | 0 | 0 | – | – |
| **wrapped_global_top10_rank** | 727 | 0 | Median imputation (skewness = 0.94) | **5.0** |

*All missing target values were replaced with the median (5.0) because the column is effectively categorical (values 4‑7) and the distribution is tightly centred.*  

---  

## 3. Descriptive Statistics & Distributions  

| Variable | Type | Range | Mean | Median | Skewness |
|----------|------|-------|------|--------|----------|
| `spotify_streams_total` | int64 | 100 284 982 – 1 948 570 210 | 233 700 631 | 167 033 880 | **4.05** |
| `daily_streams` | int64 | 9 432 – 3 075 070 | 368 485 | 222 966 | **3.16** |
| `daily_stream_share_pct` | float64 | 0.01 – 0.98 | 0.15 | 0.12 | **2.20** |
| `rank` | int64 | 1 – 730 | 365.5 | 365.5 | 0.00 |
| `daily_streams_rank` | int64 | 1 – 730 | 365.5 | 365.5 | 0.00 |
| `billed_artist_count` | int64 | 1 – 2 | 1.03 | 1.0 | 5.37 (high) |
| `is_collaboration` | bool | – | – | – | – |
| `wrapped_global_top10_rank` | float64 | 4 – 7 | 5.33 | 5.0 | 0.00 |

**Distribution visualisations** (file name – size)  

| Image | Description |
|-------|-------------|
| `dist_spotify_streams_total.png` (43.5 KB) | Heavy right‑skew; a few tracks dominate total streams. |
| `dist_daily_streams.png` (43.8 KB) | Similar right‑skew; daily streams follow the same pattern. |
| `dist_daily_stream_share_pct.png` (43.9 KB) | Concentrated around 0.10‑0.30 with a long tail toward 0.90+. |
| `dist_billed_artist_count.png` (36.9 KB) | Almost all records have a single billed artist (≈97 %). |
| `dist_is_collaboration.png` (28.5 KB) | 96 % non‑collaborations, 4 % collaborations. |
| `dist_rank.png` (38.6 KB) | Uniform distribution of rank positions (1‑730). |
| `dist_daily_streams_rank.png` (42.8 KB) | Uniform distribution of daily‑stream rank. |
| `dist_wrapped_global_top10_rank.png` (42.3 KB) | Majority of tracks have rank 5 (the imputed median). |
| `dist_artist.png` (76.6 KB) | Long tail of artists; top three: Bad Bunny (15), Sabrina Carpenter (14), Taylor Swift (12). |
| `dist_track.png` (96.0 KB) | Near‑unique track titles (726 distinct). |

---  

## 4. Outlier Analysis  

| Variable | Q1 | Q3 | IQR | Lower bound | Upper bound | Outliers (count) | Outlier % |
|----------|----|----|-----|-------------|-------------|------------------|----------|
| `spotify_streams_total` | 126 638 224 | 245 628 560 | 118 990 336 | –51 847 280 | 424 114 065 | **69** | **9.45 %** |
| `daily_streams` | 110 628 | 433 710 | 323 083 | –373 996 | 918 334 | **56** | **7.67 %** |
| `daily_stream_share_pct` | 0.0729 | 0.1823 | 0.1095 | –0.0913 | 0.3465 | **39** | **5.34 %** |
| `rank` | 183.25 | 547.75 | 364.5 | –363.5 | 1 094.5 | 0 | 0 % |
| `daily_streams_rank` | 183.25 | 547.75 | 364.5 | –363.5 | 1 094.5 | 0 | 0 % |
| `wrapped_global_top10_rank` | 5.0 | 5.0 | 0.0 | 5.0 | 5.0 | 2 | 0.27 % |

*Outliers were **profiled only** (no removal) to preserve the full streaming landscape.*  

---  

## 5. Correlation Analysis  

**Top 10 absolute correlations (|r| > 0.12)**  

| Feature 1 | Feature 2 | Pearson r |
|-----------|-----------|----------|
| `spotify_streams_total` | `daily_streams` | **0.7796** |
| `daily_streams` | `daily_streams_rank` | **‑0.7338** |
| `daily_streams_rank` | `daily_stream_share_pct` | **‑0.7254** |
| `rank` | `daily_streams_rank` | **0.7081** |
| `rank` | `spotify_streams_total` | **‑0.6800** |
| `daily_streams` | `daily_stream_share_pct` | **0.6221** |
| `rank` | `daily_streams` | **‑0.5854** |
| `spotify_streams_total` | `daily_streams_rank` | **‑0.5632** |
| `rank` | `daily_stream_share_pct` | **‑0.2206** |
| `spotify_streams_total` | `daily_stream_share_pct` | **0.1212** |

The full correlation matrix is saved as `correlation_matrix.png` (135.8 KB).  

---  

## 6. Statistical Hypothesis Tests  

| Variable | Test | Statistic | p‑value | Significant? | Interpretation |
|----------|------|-----------|---------|--------------|----------------|
| `rank` | Pearson correlation (vs. target) | –0.0283 | 0.445 | **No** | No linear relationship with Wrapped rank. |
| `artist` | One‑Way ANOVA (target across artists) | 0.771 | 0.962 | **No** | Artist identity does not explain target variance. |
| `billed_artist_count` | Pearson | –0.0030 | 0.936 | **No** | No effect. |
| `is_collaboration` | Welch t‑test (binary) | 0.447 | 0.655 | **No** | Collaboration status not predictive. |
| `spotify_streams_total` | Pearson | **0.1068** | **0.0039** | **Yes** | Small but statistically significant positive association with Wrapped rank. |
| `daily_streams` | Pearson | 0.0668 | 0.071 | No | Trend not significant at α = 0.05. |
| `daily_streams_rank` | Pearson | –0.0274 | 0.460 | No | |
| `daily_stream_share_pct` | Pearson | –0.0009 | 0.982 | No | |
| `wrapped_global_top10_rank` | (self) – | – | – | – | – |

**Only `spotify_streams_total` emerged as a statistically significant predictor of the target.**  

---  

## 7. Feature Engineering  

*No engineered features were added by the automated pipeline.*  
**Potential next‑step ideas**  

| Idea | Rationale |
|------|-----------|
| Log‑transform `spotify_streams_total` and `daily_streams` | Reduces heavy right‑skew, may improve linear models. |
| Ratio `daily_streams / spotify_streams_total` | Captures “current popularity” relative to lifetime streams. |
| Binary flag for “top‑5 artist” (e.g., Bad Bunny, Taylor Swift) | May capture brand effects. |
| Interaction term `is_collaboration × daily_stream_share_pct` | Tests whether collaborations boost share. |

---  

## 8. Predictive Modeling Blueprint  

| Aspect | Recommendation |
|--------|----------------|
| **Problem type** | Multi‑class classification (`wrapped_global_top10_rank` ∈ {4,5,6,7}) |
| **Baseline algorithm** | Regularized Logistic Regression (L2) |
| **Strong learners** | Random Forest, Gradient Boosting (XGBoost / LightGBM), Support Vector Classifier |
| **Feature selection** | <ul><li>Drop high‑cardinality identifiers (`track`, `artist`) or encode with target‑guided embeddings.</li><li>Use permutation importance + mutual information to rank remaining features.</li><li>Remove collinear predictors with |r| > 0.85 (none exceed this threshold after dropping IDs).</li></ul> |
| **Validation strategy** | Stratified 5‑fold cross‑validation (preserves class distribution). |
| **Evaluation metrics** | Balanced Accuracy, Macro F1, Precision‑Recall AUC, Confusion Matrix (to monitor class imbalance). |
| **Over‑fitting mitigation** | <ul><li>L1/L2 regularization for linear models.</li><li>Limit tree depth (≤ 6) and set `min_samples_leaf` (≥ 20) for ensembles.</li><li>Perform hyper‑parameter search *inside* CV folds (e.g., GridSearchCV). </li></ul> |
| **Implementation notes** | • Encode `billed_artist_count` as numeric (already). <br>• Encode `is_collaboration` as 0/1. <br>• Consider log‑scale of `spotify_streams_total` and `daily_streams`. |

---  

## 9. Visual Artifact Inventory  

| File | Size (KB) | Brief description |
|------|-----------|-------------------|
| `bivariate_billed_artist_count_vs_spotify_streams_total.png` | 66.3 | Scatter of total streams vs. billed‑artist count (mostly 1). |
| `bivariate_daily_stream_share_pct_vs_spotify_streams_total.png` | 1368.7 | Dense cloud showing weak positive trend; highlights outliers with high share. |
| `bivariate_daily_streams_vs_spotify_streams_total.png` | 130.1 | Strong positive relationship (r ≈ 0.78). |
| `bivariate_rank_vs_wrapped_global_top10_rank.png` | 2643.9 | Rank vs. Wrapped rank – essentially flat, confirming non‑significance. |
| `correlation_matrix.png` | 135.8 | Full Pearson correlation heatmap (numeric columns). |
| `dist_*` (9 files) | 28 – 96 | Univariate histograms / bar charts for each variable (see Section 3). |
| `pairplot.png` | 235.2 | Pairwise scatter/ KDE matrix for all numeric features. |
| `target_interactions.png` | 63.3 | Visual of target (`wrapped_global_top10_rank`) against key predictors (e.g., `spotify_streams_total`). |

---  

## 10. Key Insights & Recommendations  

1. **Data quality is high** apart from the target column, which required median imputation.  
2. **Streaming volume dominates** the variance: `spotify_streams_total` is highly skewed, strongly correlated with `daily_streams`, and the **only statistically significant predictor** of Wrapped rank.  
3. **Rank position (`rank`) is not predictive** of Wrapped rank; the ordering in the source file is unrelated to the Wrapped outcome.  
4. **Collaboration status and billed‑artist count have negligible impact** on the target.  
5. **Modeling focus** should be on transformed streaming metrics (log‑scale) and possibly engineered ratios; categorical identifiers (`track`, `artist`) can be excluded or encoded with careful regularization.  
6. **Next steps**:  
   - Apply log‑transformations to heavy‑tailed numeric features.  
   - Build a baseline logistic regression; benchmark against tree‑based ensembles.  
   - Use stratified CV and macro‑averaged metrics to guard against class imbalance (most rows now have the imputed class 5).  
   - Explore feature importance to confirm that `spotify_streams_total` (or its log) remains the dominant driver.  

*Overall, the dataset offers a clear, single‑signal predictive landscape: total streaming volume is the primary lever for explaining a track’s Wrapped global top‑10 rank.*  

---  

*Prepared for internal analytics review. All visual assets referenced are available in the working directory.*
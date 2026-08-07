# Executive Summary Report: Urban Noise Levels Exploratory Data Analysis

## 1. Dataset Overview

| Attribute | Value |
|---|---|
| **Dataset Name** | `urban_noise_levels.csv` |
| **Rows** | 2,000 |
| **Columns** | 26 |
| **Target Variable** | `decibel_level` (continuous) |
| **Missing Values** | None (0% across all columns) |
| **Problem Type** | Unsupervised / Exploratory |

The dataset captures urban noise measurements across New York City coordinates (latitude range 40.50–40.90, longitude range -74.20 to -73.70), with each record representing a sensor reading at a specific timestamp. The target variable, `decibel_level`, ranges from **33.23 dB to 97.43 dB** with a mean of **64.82 dB** and median of **65.02 dB**, indicating a roughly symmetric distribution (skewness = -0.042).

## 2. Data Quality & Completeness

The pipeline reported **zero missing values** across all 26 columns. The dataset is fully populated with no imputation required. Key schema characteristics:

| Column Group | Columns | Cardinality | Notes |
|---|---|---|---|
| Identifiers | `id`, `sensor_id` | 2,000 / 50 | `id` is unique per row; `sensor_id` groups 50 sensors |
| Geospatial | `latitude`, `longitude` | 2,000 each | Continuous coordinates |
| Temporal | `datetime`, `hour`, `day_of_week`, `is_weekend` | 1,998 / 24 / 7 / 2 | `datetime` has 2 duplicate timestamps |
| Weather | `temperature_c`, `humidity_%`, `wind_speed_kmh`, `precipitation_mm` | 2,000 each | `precipitation_mm` is highly skewed (1.93) |
| Traffic | `traffic_density`, `vehicle_count`, `honking_events` | 5 / 29 / 11 | Ordinal and count features |
| Contextual | `near_airport`, `near_highway`, `near_construction`, `park_proximity`, `industrial_zone`, `school_zone`, `public_event`, `holiday`, `noise_complaints` | 2 each (binary) | Several are highly skewed (e.g., `public_event` skew = 3.84) |

## 3. Statistical Hypothesis Testing Results

Pearson correlation tests were conducted for all numeric features against `decibel_level` at α = 0.05. **Only one feature achieved statistical significance:**

| Feature | Correlation (r) | p-value | Effect Size | Significant? |
|---|---|---|---|---|
| **day_of_week** | **-0.0513** | **0.0219** | **0.0513** | **Yes** |
| is_weekend | -0.0379 | 0.0900 | 0.0379 | No |
| traffic_density | 0.0387 | 0.0834 | 0.0387 | No |
| population_density | -0.0416 | 0.0629 | 0.0416 | No |
| holiday | -0.0409 | 0.0673 | 0.0409 | No |
| noise_complaints | -0.0411 | 0.0659 | 0.0411 | No |
| temperature_c | 0.0317 | 0.1564 | 0.0317 | No |
| wind_speed_kmh | 0.0291 | 0.1936 | 0.0291 | No |
| near_highway | -0.0333 | 0.1366 | 0.0333 | No |
| All others | — | > 0.25 | — | No |

**Interpretation:** The negative correlation with `day_of_week` suggests noise levels tend to decrease slightly as the week progresses (Monday = 0 through Sunday = 6), though the effect size is small (r = -0.05). The `is_weekend` variable approaches significance (p = 0.090) but does not cross the threshold.

## 4. Correlation Analysis

The correlation matrix reveals **generally weak linear relationships** across the dataset. Notable observations:

### Top Correlations (Absolute Value)

| Feature Pair | Correlation |
|---|---|
| `day_of_week` ↔ `is_weekend` | 0.7974 |
| `noise_complaints` ↔ `sensor_id` | -0.0843 |
| `industrial_zone` ↔ `noise_complaints` | -0.0723 |
| `industrial_zone` ↔ `vehicle_count` | -0.0707 |
| `temperature_c` ↔ `noise_complaints` | -0.0619 |
| `temperature_c` ↔ `honking_events` | -0.0578 |
| `holiday` ↔ `noise_complaints` | 0.0544 |
| `near_highway` ↔ `honking_events` | 0.0521 |
| `decibel_level` ↔ `day_of_week` | -0.0513 |

### Key Insights from Correlation Structure

1. **No strong multicollinearity** exists among features (max |r| = 0.797 between `day_of_week` and `is_weekend`, which is expected as they are derived from the same temporal source).
2. **`decibel_level` shows no strong linear association** with any single feature (max |r| = 0.051 with `day_of_week`), suggesting that noise levels are driven by complex, non-linear interactions or unmeasured factors.
3. **`noise_complaints` correlates weakly** with several features (`sensor_id`, `industrial_zone`, `temperature_c`, `holiday`), indicating complaint patterns may be spatially and temporally structured.
4. **`honking_events`** shows weak positive associations with `near_highway` (0.052) and `public_event` (0.049), and a weak negative association with `temperature_c` (-0.058).

## 5. Distribution Analysis

Individual distribution plots were generated for all 21 analyzed columns. Key distribution characteristics:

| Feature | Distribution Shape | Notable Characteristics |
|---|---|---|
| `decibel_level` | Approximately normal | Mean 64.82, SD 10.07, skew -0.042 |
| `hour` | Uniform | Range 0–23, mean 11.61 |
| `day_of_week` | Uniform | Range 0–6, mean 3.06 |
| `temperature_c` | Approximately normal | Range -4.55 to 40.0, mean 17.70 |
| `humidity_%` | Approximately uniform | Range 20.0–89.98, mean 55.18 |
| `wind_speed_kmh` | Approximately uniform | Range 0.01–39.97, mean 20.10 |
| `precipitation_mm` | **Highly right-skewed** | Skew 1.93, median 1.34, max 17.09 |
| `traffic_density` | Uniform (ordinal) | 5 levels, mean 2.93 |
| `vehicle_count` | Slightly right-skewed | Range 7–39, mean 20.11 |
| `honking_events` | Right-skewed | Range 0–10, mean 2.99 |
| `noise_complaints` | Right-skewed | Range 0–5, mean 0.99 |
| Binary flags | Imbalanced | `public_event` (5.7%), `near_airport` (10.1%), `industrial_zone` (14.4%), `holiday` (10.6%), `school_zone` (14.4%) |

## 6. Bivariate Relationship Analysis

The pipeline generated **23 semantic bivariate plots** in the first pass and **4 additional day-of-week focused plots** in the second pass. Key relationships examined:

### Traffic-Related Relationships
- **`traffic_density` vs `decibel_level`**: Weak positive trend (r = 0.039), not statistically significant
- **`vehicle_count` vs `decibel_level`**: Essentially no linear relationship (r = 0.006)
- **`honking_events` vs `decibel_level`**: No linear relationship (r = -0.006)
- **`vehicle_count` vs `traffic_density`**: Weak positive (r = 0.014)
- **`honking_events` vs `vehicle_count`**: Weak negative (r = -0.011)

### Temporal Relationships
- **`hour` vs `decibel_level`**: No significant linear trend (r = -0.012); the target interaction plot (`target_interaction_hour.png`) provides visual detail on hourly patterns
- **`day_of_week` vs `decibel_level`**: Statistically significant weak negative correlation (r = -0.051, p = 0.022)
- **`is_weekend` vs `decibel_level`**: Marginal (r = -0.038, p = 0.090)

### Environmental Relationships
- **`temperature_c` vs `decibel_level`**: Weak positive (r = 0.032)
- **`humidity_%` vs `decibel_level`**: Negligible (r = 0.011)
- **`wind_speed_kmh` vs `decibel_level`**: Weak positive (r = 0.029)
- **`precipitation_mm` vs `decibel_level`**: Negligible (r = 0.003)

### Contextual/Location Relationships
- **`near_highway` vs `decibel_level`**: Weak negative (r = -0.033)
- **`near_airport` vs `decibel_level`**: Negligible (r = 0.001)
- **`near_construction` vs `decibel_level`**: Negligible (r = -0.007)
- **`park_proximity` vs `decibel_level`**: Weak negative (r = -0.016)
- **`industrial_zone` vs `decibel_level`**: Weak negative (r = -0.013)
- **`population_density` vs `decibel_level`**: Weak negative (r = -0.042)

## 7. Generated Visual Artifacts

The pipeline produced **50+ visualization files** organized into four categories:

| Category | Count | Examples |
|---|---|---|
| **Univariate Distributions** | 21 | `dist_decibel_level.png`, `dist_hour.png`, `dist_traffic_density.png` |
| **Bivariate Relationships** | 27 | `bivariate_traffic_density_vs_decibel_level.png`, `bivariate_day_of_week_vs_decibel_level.png`, `bivariate_temperature_c_vs_humidity.png` |
| **Target Interactions** | 3 | `target_interaction_traffic.png`, `target_interaction_hour.png`, `target_interaction_dayofweek.png` |
| **Multivariate** | 2 | `correlation_matrix.png`, `pairplot.png` |

### Notable Visualizations
- **`correlation_matrix.png`** (659 KB): Full 26×26 correlation heatmap
- **`pairplot.png`** (404 KB): Scatter matrix for `decibel_level`, `vehicle_count`, `honking_events`, `temperature_c`
- **`target_interaction_hour.png`** (212 KB): Hourly noise level patterns
- **`bivariate_humidity_vs_decibel_level.png`** (276 KB): Weather-noise relationship
- **`bivariate_wind_speed_kmh_vs_decibel_level.png`** (280 KB): Wind-noise relationship

## 8. Feature Engineering & Preprocessing

- **Imputation**: Completed (though no missing values were detected)
- **Outlier Analysis**: No explicit outlier handling was triggered
- **Engineered Features**: None were created during this run
- **Categorical Associations**: No categorical association tests were reported

## 9. Predictive Modeling Blueprint

The pipeline classified this dataset as **Unsupervised / Exploratory** with the following recommendations:

| Component | Recommendation |
|---|---|
| **Problem Type** | Unsupervised / Exploratory (target undefined for supervised learning) |
| **Recommended Algorithms** | K-Means Clustering, Hierarchical Agglomerative Clustering, PCA for Dimensionality Reduction |
| **Feature Selection** | Exclude high-cardinality ID/text columns; rank features via cross-validated permutation importance and mutual information; remove collinear features above r > 0.85 |
| **Validation Strategy** | Silhouette Score and Inertia elbow curve |
| **Overfitting Mitigation** | Apply L1/L2 regularization; limit tree depth; enforce minimum samples per leaf; hyperparameter tuning strictly within CV folds |

## 10. Key Findings & Conclusions

1. **Weak Linear Signal**: The target variable `decibel_level` exhibits no strong linear correlation with any individual feature. The strongest relationship (r = -0.051 with `day_of_week`) is statistically significant but practically weak, explaining less than 0.3% of variance.

2. **Temporal Patterns Dominate**: The only statistically significant predictor is `day_of_week` (p = 0.022), suggesting that noise levels vary modestly by day of the week. The `is_weekend` variable approaches significance (p = 0.090), reinforcing a potential weekday/weekend effect.

3. **Traffic Metrics Underperform**: Despite intuitive expectations, `traffic_density`, `vehicle_count`, and `honking_events` show negligible linear correlations with noise levels (r = 0.039, 0.006, and -0.006 respectively). This suggests either non-linear relationships or that these proxy variables do not capture the true noise-generating mechanisms.

4. **Data Quality is High**: No missing values, no strong multicollinearity, and well-distributed continuous features provide a clean foundation for further analysis.

5. **Unsupervised Direction Recommended**: Given the weak linear signal, the pipeline recommends clustering approaches (K-Means, Hierarchical) and dimensionality reduction (PCA) to uncover latent structure in the data rather than pursuing supervised regression directly.

## 11. Recommendations for Next Steps

1. **Explore Non-Linear Models**: Given weak linear correlations, consider gradient boosting or random forest models to capture non-linear interactions.
2. **Feature Engineering**: Create interaction terms (e.g., `hour × day_of_week`, `traffic_density × vehicle_count`) and cyclical encodings for temporal features.
3. **Spatial Analysis**: Leverage `latitude`/`longitude` for spatial clustering or geospatial noise mapping.
4. **Cluster Analysis**: Apply K-Means with silhouette scoring to identify distinct noise regimes (e.g., quiet residential vs. busy commercial zones).
5. **Temporal Decomposition**: Investigate hourly and daily patterns more deeply using time-series decomposition techniques.
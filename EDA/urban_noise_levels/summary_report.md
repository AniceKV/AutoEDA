# Executive Summary – Urban Noise Levels EDA  

**Dataset:** `urban_noise_levels.csv`  
**Target variable:** `noise_complaints` (integer count, 0‑5)  
**Rows / Columns:** 2 000 × 27 (including the engineered feature)  

---

## 1. Dataset Overview  

| Item                              | Value |
|-----------------------------------|-------|
| Total rows                        | 2 000 |
| Total columns (pre‑engineered)    | 26 |
| Columns after engineering          | 27 |
| Target column                     | `noise_complaints` |
| Primary data types                | int64, float64, str |
| Unique IDs (`id`)                 | 2 000 (1 – 2 000) |
| Sensor IDs (`sensor_id`)          | 50 distinct sensors (mean = 25.54) |
| Datetime format                   | `YYYY‑MM‑DD HH:MM:SS` (1998 unique timestamps) |

### 1.1 Key Column Statistics  

| Column                | Type    | Mean   | Median | Std. Dev. | Min   | Max   | Cardinality |
|----------------------|---------|--------|--------|-----------|-------|-------|-------------|
| `decibel_level`      | float64 | 64.82  | 65.02  | 10.07     | 33.23 | 97.43 | 2 000 |
| `temperature_c`      | float64 | 17.70  | 17.64  | 7.17      | –4.55 | 40.00 | 2 000 |
| `humidity_%`         | float64 | 55.18  | 55.20  | 19.99     | 20.00 | 89.98 | 2 000 |
| `wind_speed_kmh`     | float64 | 20.10  | 19.82  | 11.68     | 0.01  | 39.97 | 2 000 |
| `precipitation_mm`   | float64 | 2.00   | 1.34   | 2.03      | 0.00  | 17.09 | 2 000 |
| `traffic_density`    | int64   | 2.93   | 3.00   | 1.41      | 1     | 5     | 5 |
| `near_airport`       | int64   | 0.10   | 0      | 0.30      | 0     | 1     | 2 |
| `near_highway`       | int64   | 0.31   | 0      | 0.46      | 0     | 1     | 2 |
| `near_construction`  | int64   | 0.22   | 0      | 0.41      | 0     | 1     | 2 |
| `population_density` | int64   | 15 560 | 15 669 | 8 370     | 1 018 | 29 991| 1 931 |
| `industrial_zone`    | int64   | 0.14   | 0      | 0.35      | 0     | 1     | 2 |
| `vehicle_count`      | int64   | 20.11  | 20.00  | 4.50      | 7     | 39    | 29 |
| `honking_events`     | int64   | 2.99   | 3.00   | 1.71      | 0     | 10    | 11 |
| `public_event`       | int64   | 0.06   | 0      | 0.23      | 0     | 1     | 2 |
| `holiday`            | int64   | 0.11   | 0      | 0.31      | 0     | 1     | 2 |
| `school_zone`        | int64   | 0.14   | 0      | 0.35      | 0     | 1     | 2 |
| `noise_complaints`   | int64   | 0.99   | 1.00   | 0.98      | 0     | 5     | 6 |
| `sensor_id`          | int64   | 25.54  | 26.00  | 14.41     | 1     | 50    | 50 |
| `engineered_feature` | float64 | 0.001 ± 0.003 | 0.000 | – | 0.000 | 0.008 | 1 997 |

*All numeric columns have **0 % missing values**; categorical/string columns were imputed using mode (no missing values were present).*

---

## 2. Data Quality & Imputation  

The automated pipeline applied the following imputation policy:

| Rule | Applied To |
|------|------------|
| Standardise missing placeholders (`?`, `NA`, `N/A`, `null`) → `NaN` | All columns |
| Numeric skew > 1 or < ‑1 → **median** imputation | None (no missing) |
| Numeric skew within [‑1, 1] → **mean** imputation | None (no missing) |
| Categorical/string → **mode** (`'Unknown'` fallback) | None (no missing) |

Result: **No imputation was required**; the dataset is complete.

---

## 3. Outlier Analysis  

Outliers were profiled (no removal) using the IQR method.

| Column               | Q1   | Q3   | IQR   | Lower Bound | Upper Bound | Outliers | % of rows |
|----------------------|------|------|-------|-------------|-------------|----------|-----------|
| `decibel_level`      | 57.91| 71.65| 13.74 | 37.30       | 92.26       | 17       | 0.85 % |
| `temperature_c`      | 12.78| 22.76| 9.98  | –2.20       | 37.73       | 8        | 0.40 % |
| `precipitation_mm`   | 0.54 | 2.84 | 2.30  | –2.91       | 6.30        | 92       | 4.60 % |
| `vehicle_count`      | 17   | 23   | 6     | 8           | 32          | 13       | 0.65 % |
| `honking_events`     | 2    | 4    | 2     | –1          | 7           | 20       | 1.00 % |
| *All other columns*  | –    | –    | –     | –           | –           | 0        | 0 % |

The outlier percentages are low; the pipeline retained all records for downstream modeling.

---

## 4. Feature Engineering  

The pipeline was instructed to create three engineered features, but only **one** was successfully generated:

| Feature Name        | Formula (≈)                              | Type    | Rationale                              | Correlation with Target |
|---------------------|------------------------------------------|---------|----------------------------------------|--------------------------|
| `engineered_feature`| `vehicle_count / (population_density + ε)`| float64 | Ratio captures traffic intensity per capita | **0.022** (p = 0.327) – not significant |

*Planned transformations (log‑1p of `decibel_level`, interaction `near_airport × traffic_density`) were not materialised in the final artifact.*

---

## 5. Correlation Analysis  

A full Pearson correlation matrix was saved as an image:

```
C:\Users\Anish Kumar Verma\PycharmProjects\AutoEDA\sandbox_run\b427831d-1e01-400d-a790-be834174a84f\correlation_matrix.png
```

### 5.1 Top 10 Pairwise Correlations  

| Rank | Feature 1            | Feature 2            | Pearson r |
|------|---------------------|---------------------|-----------|
| 1    | `day_of_week`       | `is_weekend`        | **0.7974** |
| 2    | `population_density`| `engineered_feature`| **‑0.674** |
| 3    | `vehicle_count`     | `engineered_feature`| **0.1604** |
| 4    | `noise_complaints`  | `sensor_id`         | **‑0.0843** |
| 5    | `industrial_zone`   | `noise_complaints`  | **‑0.0723** |
| 6    | `industrial_zone`   | `vehicle_count`     | **‑0.0707** |
| 7    | `temperature_c`     | `noise_complaints`  | **‑0.0619** |
| 8    | `temperature_c`     | `honking_events`    | **‑0.0578** |
| 9    | `near_airport`      | `engineered_feature`| **0.0577** |
|10    | `holiday`           | `noise_complaints`  | **0.0544** |

*All other absolute correlations are ≤ 0.05.*

---

## 6. Statistical Hypothesis Testing  

Pearson correlation tests (α = 0.05) were performed between each feature and the target. Only four predictors reached statistical significance.

| Predictor          | Pearson r | p‑value | Significance |
|--------------------|-----------|---------|--------------|
| `temperature_c`    | ‑0.0619   | 0.0056  | **Significant** |
| `industrial_zone`  | ‑0.0723   | 0.0012  | **Significant** |
| `holiday`          | 0.0544    | 0.0149  | **Significant** |
| `sensor_id`        | ‑0.0843   | 0.00016 | **Significant** |
| *All other features* | – | – | Not significant (p > 0.05) |

These four variables are the strongest candidates for inclusion in a predictive model.

---

## 7. Visual Artifacts  

All plots are stored in the sandbox directory; file sizes are shown for quick reference.

| Plot Type | Filename | Size (KB) | Brief Description |
|-----------|----------|----------|-------------------|
| Distribution – `decibel_level` | `dist_decibel_level.png` | 45.18 | Histogram of sound pressure levels |
| Distribution – `temperature_c` | `dist_temperature_c.png` | 44.27 | Temperature distribution (wide range) |
| Distribution – `precipitation_mm` | `dist_precipitation_mm.png` | 40.50 | Highly skewed precipitation values |
| Distribution – `hour` | `dist_hour.png` | 35.08 | Uniform hourly coverage (0‑23) |
| Distribution – `noise_complaints` | `dist_noise_complaints.png` | 24.26 | Majority of observations have 0‑2 complaints |
| Bivariate – `decibel_level` vs `noise_complaints` | `bivariate_decibel_level_vs_noise_complaints.png` | 59.86 | Scatter with low linear trend |
| Bivariate – `traffic_density` vs `noise_complaints` | `bivariate_traffic_density_vs_noise_complaints.png` | 38.02 | Slight positive association |
| Bivariate – `near_construction` vs `noise_complaints` | `bivariate_near_construction_vs_noise_complaints.png` | 32.11 | No clear pattern |
| Bivariate – `hour` vs `noise_complaints` (hue = `is_weekend`) | `bivariate_hour_vs_noise_complaints.png` | 50.87 | Slight increase in complaints during evening hours |
| Correlation Heatmap | `correlation_matrix.png` | 719.25 | Full 27 × 27 Pearson matrix (see path above) |

All distribution plots are standard histograms or bar charts; bivariate plots are scatter/box‑type visualisations that aid rapid visual inspection.

---

## 8. Predictive Modeling Blueprint  

| Item | Recommendation |
|------|----------------|
| **Problem type** | **Classification** (multi‑class, 0‑5 complaints) |
| **Baseline model** | Regularized Logistic Regression (L2 penalty) |
| **Advanced models** | • Random Forest Classifier  <br>• Gradient Boosting (XGBoost or LightGBM) <br>• Support Vector Classifier (SVM) |
| **Feature selection** | 1. Remove high‑cardinality identifiers (`id`, `sensor_id` if not predictive). <br>2. Rank features by cross‑validated permutation importance and mutual information. <br>3. Drop collinear features with |r| > 0.85 (none observed beyond `day_of_week`/`is_weekend`). |
| **Validation strategy** | Stratified 5‑fold cross‑validation (preserves complaint class distribution). |
| **Evaluation metrics** | Balanced Accuracy, Macro‑averaged F1, Precision‑Recall AUC, Confusion Matrix. |
| **Over‑fitting safeguards** | • Apply L1/L2 regularisation (logistic regression). <br>• Limit tree depth, enforce minimum samples per leaf (RF/GBM). <br>• Conduct hyper‑parameter tuning **inside** cross‑validation folds only. |
| **Executive note** | The dataset is modest (2 k rows) but contains several statistically significant predictors (`temperature_c`, `industrial_zone`, `holiday`, `sensor_id`). A well‑regularised linear model may already achieve respectable performance; tree‑based ensembles can be explored for potential gains, especially if non‑linear interactions (e.g., `near_airport × traffic_density`) are later engineered. |

---

## 9. Key Take‑aways & Next Steps  

1. **Data integrity is high** – no missing values, low outlier rates.  
2. **Four predictors are statistically linked** to noise complaints; they should be prioritized in model building.  
3. **Engineered ratio feature** (`vehicle_count / population_density`) shows negligible correlation with the target; consider alternative transformations (e.g., log‑scale, interaction terms).  
4. **Strong temporal relationship** (`day_of_week` ↔ `is_weekend`) suggests that weekend vs weekday patterns may be captured via a single binary flag.  
5. **Modeling plan** – start with a regularised logistic regression baseline, then evaluate Random Forest and Gradient Boosting models using the stratified CV scheme.  
6. **Further feature work** – implement the planned log‑1p transformation of `decibel_level` and the interaction `near_airport × traffic_density`; assess their impact on model performance.  

---  

*Prepared by the Senior Lead Data Scientist – Automated EDA pipeline output (2026‑08‑06).*
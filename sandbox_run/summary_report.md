# Executive Summary – Mobile Phone Pricing Dataset  
**Prepared by:** Senior Lead Data Scientist  
**Date:** 2026‑08‑06  

---

## 1. Overview  

| Item | Detail |
|------|--------|
| **Dataset** | `train.csv` (2000 rows × 21 columns) |
| **Target variable** | `price_range` (4‑class classification: 0‑3) |
| **Primary goal** | Understand data characteristics, identify predictive features, and outline a modelling blueprint for a robust classifier. |
| **Key findings** | • No missing values after imputation. <br>• Outlier profiling performed (action = *profile*). <br>• Five features are statistically significant predictors of price range: **battery_power, int_memory, px_height, px_width, ram**. <br>• `ram` shows an exceptionally strong linear relationship with the target (Pearson r = 0.917). <br>• No new engineered features were created (the feature‑engineering step generated 0 features). |

---

## 2. Data Quality  

### 2.1 Missing‑value handling  

The automated pipeline applied a uniform strategy:

* Standardised missing string placeholders (`?`, `NA`, `N/A`, `null`) → `NaN`.  
* Numeric columns with skewness > 1.0 or < ‑1.0 → median imputation.  
* Numeric columns with skewness between –1.0 and 1.0 → mean imputation.  
* Categorical/string columns → mode imputation (fallback = “Unknown”).  

**Result:** All 21 columns reported **0 missing values** before and after imputation (see `imputation_res` in `agent_state.json`). No imputation was required.

### 2.2 Outlier profiling  

Outlier detection was performed on every numeric column with the *profile* action (no values were removed). The most extreme case was `fc` (front‑camera count) with 18 outliers (0.9 % of rows). `px_height` had 2 outliers (0.1 %). All other columns reported 0 outliers.

| Feature | Outlier % | Action |
|---------|-----------|--------|
| `fc` | 0.9 % | Profile |
| `px_height` | 0.1 % | Profile |
| *All others* | 0 % | Profile |

---

## 3. Statistical Summaries  

### 3.1 Central tendency & dispersion  

| Feature | dtype | Cardinality | Mean | Median | Std. Dev. | Min | Max |
|---------|-------|-------------|------|--------|-----------|-----|-----|
| battery_power | int64 | 1094 | 1238.52 | 1226.0 | 439.42 | 501 | 1998 |
| blue | int64 | 2 | 0.495 | 0.0 | 0.5001 | 0 | 1 |
| clock_speed | float64 | 26 | 1.522 | 1.5 | 0.8160 | 0.5 | 3.0 |
| dual_sim | int64 | 2 | 0.5095 | 1.0 | 0.5000 | 0 | 1 |
| fc | int64 | 20 | 4.3095 | 3.0 | 4.3414 | 0 | 19 |
| four_g | int64 | 2 | 0.5215 | 1.0 | 0.4997 | 0 | 1 |
| int_memory | int64 | 63 | 32.0465 | 32.0 | 18.1457 | 2 | 64 |
| m_dep | float64 | 10 | 0.5018 | 0.5 | 0.2884 | 0.1 | 1.0 |
| mobile_wt | int64 | 121 | 140.25 | 141.0 | 35.3997 | 80 | 200 |
| n_cores | int64 | 8 | 4.5205 | 4.0 | 2.2878 | 1 | 8 |
| pc | int64 | 21 | 9.9165 | 10.0 | 6.0643 | 0 | 20 |
| px_height | int64 | 1137 | 645.11 | 564.0 | 443.78 | 0 | 1960 |
| px_width | int64 | 1109 | 1251.52 | 1247.0 | 432.20 | 500 | 1998 |
| ram | int64 | 1562 | 2124.21 | 2146.5 | 1084.73 | 256 | 3998 |
| sc_h | int64 | 15 | 12.3065 | 12.0 | 4.2132 | 5 | 19 |
| sc_w | int64 | 19 | 5.7670 | 5.0 | 4.3564 | 0 | 18 |
| talk_time | int64 | 19 | 11.011 | 11.0 | 5.46396 | 2 | 20 |
| three_g | int64 | 2 | 0.7615 | 1.0 | 0.4263 | 0 | 1 |
| touch_screen | int64 | 2 | 0.503 | 1.0 | 0.5001 | 0 | 1 |
| wifi | int64 | 2 | 0.507 | 1.0 | 0.5001 | 0 | 1 |
| price_range | int64 | 4 | 1.5 | 1.5 | 1.1183 | 0 | 3 |

*Skewness and kurtosis values are available in `metadata_profile.json` – all numeric features are close to symmetric except `fc` (skew ≈ 1.02) and `px_height` (skew ≈ 0.67).*

### 3.2 Correlation matrix (top 10 absolute correlations)

| Feature 1 | Feature 2 | Pearson ρ |
|----------|-----------|-----------|
| **ram** | **price_range** | **0.917** |
| **fc** | **pc** | 0.645 |
| **four_g** | **three_g** | 0.584 |
| **px_height** | **px_width** | 0.511 |
| **sc_h** | **sc_w** | 0.506 |
| **battery_power** | **price_range** | 0.202 |
| **px_width** | **price_range** | 0.166 |
| **px_height** | **price_range** | 0.149 |
| **px_height** | **sc_h** | 0.060 |
| **battery_power** | **talk_time** | 0.053 |

*All other pairwise correlations are ≤ 0.05 in magnitude.*

---

## 4. Hypothesis‑Testing Results  

Pearson correlation tests were run for every feature against the target (`price_range`). Significance threshold α = 0.05.

| Feature | Pearson r | p‑value | Significant? |
|---------|-----------|---------|--------------|
| battery_power | 0.2007 | 1.26 e‑19 | **Yes** |
| int_memory | 0.0444 | 4.69 e‑02 | **Yes** |
| px_height | 0.1489 | 2.23 e‑11 | **Yes** |
| px_width | 0.1658 | 8.48 e‑14 | **Yes** |
| ram | 0.9170 | 0.0 | **Yes** |
| all other features | – | – | No |

**Interpretation:** The five listed features have statistically significant linear relationships with the price range and should be prioritized in any predictive model.

---

## 5. Feature Engineering  

The pipeline was instructed to create four engineered features:

| New Feature | Source Columns | Transformation |
|-------------|----------------|----------------|
| `log1p_battery_power` | battery_power | log(1 + x) |
| `log1p_ram` | ram | log(1 + x) |
| `battery_to_weight_ratio` | battery_power, mobile_wt | battery_power / mobile_wt |
| `pc_ram_interaction` | pc, ram | product (pc × ram) |

**Outcome:** The `engineer_features` step reported *“Generated 0 features.”* – none of the proposed transformations were added to the final dataframe (likely due to redundancy or collinearity checks). Consequently, the current feature set remains the original 21 columns.

---

## 6. Visual Artefacts  

All visualisations were saved in the `./sandbox_run` directory. File sizes are shown for reference.

| File | Description | Size (KB) |
|------|-------------|-----------|
| `correlation_matrix.png` | Heat‑map of the full Pearson correlation matrix (21 × 21). | 503.47 |
| `pairplot.png` | Pairwise scatter‑plot matrix for `battery_power`, `ram`, `mobile_wt`, and `price_range` (hue = price_range). | 720.30 |
| `price_vs_ram.png` | Scatter plot of `ram` vs. `price_range` with trend line. | 77.54 |
| `bivariate_battery_power_vs_price_range.png` | Battery power distribution across price classes. | 75.50 |
| `bivariate_mobile_wt_vs_price_range.png` | Mobile weight distribution across price classes. | 67.08 |
| `bivariate_pc_vs_price_range.png` | Primary camera count vs. price classes. | 54.42 |
| `bivariate_ram_vs_price_range.png` | RAM vs. price classes (highly discriminative). | 72.65 |
| `bivariate_three_g_vs_price_range.png` | 3G support vs. price classes. | 34.52 |
| `dist_*.png` (21 files) | Individual univariate histograms / bar‑charts for each column (e.g., `dist_battery_power.png`, `dist_px_height.png`). | 19 – 53 each |

*All images are ready for inclusion in a presentation deck or interactive notebook.*

---

## 7. Predictive‑Modeling Blueprint  

The pipeline generated a concise modelling plan (`blueprint_res`). Key components are reproduced below.

### 7.1 Problem definition  

* **Target:** `price_range`  
* **Task type:** Multi‑class classification (4 classes)  

### 7.2 Recommended algorithms  

| Rank | Algorithm | Rationale |
|------|-----------|-----------|
| 1 | Regularized Logistic Regression (baseline) | Simple, interpretable, fast to train. |
| 2 | Random Forest Classifier | Handles non‑linear interactions, robust to noisy features. |
| 3 | Gradient Boosting (XGBoost / LightGBM) | State‑of‑the‑art performance on tabular data. |
| 4 | Support Vector Classifier (SVM) | Effective when classes are not linearly separable. |

### 7.3 Feature‑selection strategy  

1. **Exclude** any high‑cardinality identifier or textual name columns (none present).  
2. **Rank** features using cross‑validated permutation importance **and** mutual information.  
3. **Remove** collinear features with absolute correlation > 0.85 (e.g., `px_height` vs. `px_width` – 0.511, below threshold, so both may be retained).  

### 7.4 Validation strategy  

* **Stratified K‑Fold CV** – 5 folds (preserves class distribution).  
* **Evaluation metrics:** Balanced Accuracy, Macro‑averaged F1, Precision‑Recall AUC, and Confusion Matrix.  

### 7.5 Over‑fitting mitigation  

* Apply **L1/L2 regularisation** (logistic regression, linear SVM).  
* **Limit tree depth** and enforce **minimum samples per leaf** (Random Forest, Gradient Boosting).  
* Conduct **hyper‑parameter tuning** strictly within CV folds (no leakage).  

### 7.6 Executive summary (blueprint)  

> “Target: `price_range` (Classification). Use robust cross‑validation on 2000 rows × 21 columns.”

---

## 8. Recommendations & Next Steps  

1. **Model Development** – Implement the blueprint using a framework such as `scikit‑learn` or `mlflow`. Begin with logistic regression as a baseline, then iterate through Random Forest and Gradient Boosting.  
2. **Feature Engineering Review** – Re‑evaluate the four proposed engineered features. `log1p_ram` and `battery_to_weight_ratio` may still add value, especially after checking multicollinearity with `ram` and `battery_power`.  
3. **Class Imbalance Check** – Verify the distribution of `price_range` (4 classes) to ensure stratified splits are appropriate; consider class‑weighting or SMOTE if imbalance is severe.  
4. **Model Explainability** – Use SHAP or permutation importance to confirm that the statistically significant predictors (battery_power, int_memory, px_height, px_width, ram) drive model decisions.  
5. **Performance Monitoring** – After deployment, monitor drift in the key predictors (especially `ram` and `px_*` dimensions) and re‑train models quarterly.  

---

**Prepared by:**  
Senior Lead Data Scientist  
*Automated EDA & Modelling Pipeline*  

---  
# Executive Summary – Credit‑Card Fraud Detection (2026)

**Dataset**: `credit_card_fraud_2026.csv`  
**Rows / Columns**: 20 000 × 26  
**Target**: `is_fraud` (binary classification) – fraud rate ≈ 1.7 % (mean = 0.01695)  

The automated EDA pipeline completed data profiling, outlier detection, correlation & categorical association analysis, statistical hypothesis testing, and produced a predictive‑modeling blueprint. No missing values were found; outlier profiling was performed (no removal). Feature‑engineering steps were attempted but did **not** generate new columns (all specifications failed to create features). The following sections synthesize the key quantitative findings and actionable recommendations for model development.

---

## 1. Dataset Overview  

| Item                              | Value |
|-----------------------------------|-------|
| Total rows                        | 20 000 |
| Total columns                     | 26 |
| Target column                     | `is_fraud` |
| Fraud prevalence (mean)           | 0.01695 (≈ 1.7 %) |
| Numeric columns (high‑skew)       | `amount_usd`, `account_balance_usd`, `hours_since_last_txn`, `distance_from_home_km`, `prior_disputes` |
| Categorical columns (cardinality) | `merchant_category` (12), `card_type` (5), `auth_method` (5), `channel` (5), `device_type` (8) |
| Boolean columns                   | 7 (e.g., `is_foreign_transaction`, `used_vpn`) |

### 1.1 Key Numeric Summaries  

| Feature                | Mean   | Median | Std‑Dev | Min   | Max      | Skew   | Kurtosis |
|------------------------|--------|--------|---------|-------|----------|--------|----------|
| `amount_usd`           | 132.42 | 57.51  | 256.96  | 1.00  | 6 872.69 | 7.21   | 87.55    |
| `hours_since_last_txn` | 8.95   | 6.21   | 8.84    | 0.01  | 87.05    | 1.89   | 5.05     |
| `distance_from_home_km`| 22.14  | 15.50  | 22.12   | 0.00  | 216.19   | 2.03   | 6.24     |
| `account_balance_usd`  | 3 316.66| 2 007.68| 4 350.72| 52.05 | 127 125.86| 5.63   | 70.30    |
| `cvv_retry_count`      | 0.18   | 0.00   | 0.42    | 0     | 3        | 2.29   | 5.06     |
| `velocity_score`       | 19.81  | 18.80  | 12.37   | 0.00  | 74.40    | 0.54   | 0.17     |
| `customer_age`         | 49.67  | 50.00  | 18.49   | 18    | 81       | -0.01  | -1.19    |
| `prior_disputes`       | 0.28   | 0.00   | 0.53    | 0     | 4        | 1.88   | 3.48     |

*All categorical and boolean columns have 0 % missing values.*

---

## 2. Data Quality  

### 2.1 Missing‑Value Handling  
- No missing entries detected across any column.  
- Imputation rules were defined (median for highly skewed numerics, mean for near‑normal, mode for categoricals) but were not applied because the dataset is complete.

### 2.2 Outlier Profiling (action = *profile*)  

| Feature                | IQR   | Lower Bound | Upper Bound | Outliers | % of rows |
|------------------------|-------|-------------|-------------|----------|-----------|
| `amount_usd`           | 105.52| –132.04     | 290.03      | 2 061    | 10.3 % |
| `hours_since_last_txn` | 9.84  | –12.17      | 27.20       | 955      | 4.8 % |
| `txn_count_last_24h`   | 2.00  | –1.0        | 7.0         | 336      | 1.7 % |
| `distance_from_home_km`| 24.48 | –30.39      | 67.54       | 904      | 4.5 % |
| `card_age_months`      | 9.00  | 28.5        | 64.5        | 177      | 0.9 % |
| `account_balance_usd`  | 2 924.98| –3 368.33   | 8 331.59    | 1 569    | 7.9 % |
| `cvv_retry_count`      | 0.00  | 0.0         | 0.0         | 3 342    | 16.7 % |
| `velocity_score`       | 16.90 | –14.65      | 52.95       | 213      | 1.1 % |
| `prior_disputes`       | 0.00  | 0.0         | 0.0         | 4 915    | 24.6 % |

*No rows were removed; outliers were retained for modeling.*

---

## 3. Correlation Analysis  

| Feature Pair                     | Pearson r |
|----------------------------------|-----------|
| `txn_count_last_24h` & `velocity_score` | **0.6224** |
| `hours_since_last_txn` & `velocity_score`| **‑0.2655** |
| `cvv_retry_count` & `is_fraud`   | **0.1555** |
| `merchant_risk_score` & `is_fraud`| **0.1077** |
| `velocity_score` & `is_fraud`    | **0.0973** |
| `amount_usd` & `merchant_risk_score`| **0.0971** |
| `txn_count_last_24h` & `is_fraud`| **0.0599** |
| `time_of_day_hour` & `is_fraud`  | **‑0.0308** |
| `hours_since_last_txn` & `is_fraud`| **‑0.0288** |
| `amount_usd` & `is_fraud`        | **0.0250** |

*All correlations are modest; the strongest linear relationship is between transaction count in the last 24 h and the velocity score (r ≈ 0.62).*

---

## 4. Categorical Association (Cramér’s V)  

| Categorical Pair                     | Cramér’s V |
|--------------------------------------|------------|
| `is_foreign_transaction` & `ip_country_mismatch` | **0.2722** |
| `channel` & `billing_shipping_mismatch`          | **0.1605** |
| `channel` & `is_ai_generated_scam_attempt`      | **0.0992** |
| `device_type` & `ip_country_mismatch`           | **0.0251** |
| `merchant_category` & `is_new_merchant`         | **0.0215** |
| `billing_shipping_mismatch` & `is_ai_generated_scam_attempt` | **0.0198** |
| `auth_method` & `is_ai_generated_scam_attempt` | **0.0192** |
| `device_type` & `is_ai_generated_scam_attempt`| **0.0138** |
| `merchant_category` & `is_ai_generated_scam_attempt`| **0.0126** |
| `device_type` & `used_vpn`                     | **0.0112** |

*The strongest association is between foreign‑transaction flag and IP‑country mismatch (V ≈ 0.27).*

---

## 5. Statistical Hypothesis Testing  

All tests were performed at α = 0.05.  

| Feature                | Test Type                | Statistic | p‑value | Significant? |
|------------------------|--------------------------|-----------|---------|--------------|
| `amount_usd`           | Pearson correlation      | 0.0250    | 4.17 e‑4| ✅ |
| `merchant_category`   | One‑Way ANOVA            | 11.0278   | 1.16 e‑20| ✅ |
| `auth_method`          | One‑Way ANOVA            | 33.2641   | 1.07 e‑27| ✅ |
| `is_foreign_transaction`| Welch t‑test (binary)   | –6.6283   | 4.99 e‑11| ✅ |
| `hours_since_last_txn` | Pearson correlation      | –0.0288   | 4.64 e‑5| ✅ |
| `txn_count_last_24h`   | Pearson correlation      | 0.0599    | 2.38 e‑17| ✅ |
| `customer_age`         | Pearson correlation      | 0.0195    | 5.71 e‑3| ✅ |
| `is_new_merchant`      | Welch t‑test (binary)    | –8.2423   | 2.09 e‑16| ✅ |
| `used_vpn`             | Welch t‑test (binary)    | –5.9316   | 3.56 e‑9| ✅ |
| `ip_country_mismatch`  | Welch t‑test (binary)    | –8.2095   | 5.72 e‑16| ✅ |
| `billing_shipping_mismatch`| Welch t‑test (binary)| –6.9905   | 5.38 e‑12| ✅ |
| `cvv_retry_count`      | Pearson correlation      | 0.1555    | 1.79 e‑108| ✅ |
| `velocity_score`       | Pearson correlation      | 0.0973    | 3.10 e‑43| ✅ |
| `time_of_day_hour`     | Pearson correlation      | –0.0308   | 1.31 e‑5| ✅ |
| `is_ai_generated_scam_attempt`| Welch t‑test (binary)| –5.0509 | 6.87 e‑7| ✅ |
| `merchant_risk_score`  | Pearson correlation      | 0.1077    | 1.18 e‑52| ✅ |
| `prior_disputes`       | Pearson correlation      | 0.0239    | 7.23 e‑4| ✅ |

**Non‑significant** (p > 0.05): `transaction_id`, `card_type`, `channel`, `device_type`, `distance_from_home_km`, `card_age_months`, `account_balance_usd`, `day_of_week`.

### 5.1 Consolidated List of Significant Predictors  

```
amount_usd
merchant_category
auth_method
is_foreign_transaction
hours_since_last_txn
txn_count_last_24h
customer_age
is_new_merchant
used_vpn
ip_country_mismatch
billing_shipping_mismatch
cvv_retry_count
velocity_score
time_of_day_hour
is_ai_generated_scam_attempt
merchant_risk_score
prior_disputes
```

These 17 features will be the primary focus for model building.

---

## 6. Feature Engineering  

The pipeline attempted the following specifications:

| New Feature                     | Transformation | Source Columns |
|--------------------------------|----------------|----------------|
| `log_amount_usd`               | log1p          | `amount_usd` |
| `log_account_balance_usd`      | log1p          | `account_balance_usd` |
| `amount_to_balance_ratio`      | ratio (÷)      | `amount_usd`, `account_balance_usd` |
| `hours_distance_interaction`   | product (×)    | `hours_since_last_txn`, `distance_from_home_km` |
| `card_age_times_velocity`      | product (×)    | `card_age_months`, `velocity_score` |

**Result** – No new columns were successfully added (engineered_features list empty). Possible causes: division‑by‑zero safeguards, duplicate column names, or pipeline error. Recommend revisiting the engineering step before model training.

---

## 7. Predictive‑Modeling Blueprint  

| Aspect                     | Recommendation |
|----------------------------|----------------|
| **Problem Type**           | Binary Classification (`is_fraud`) |
| **Baseline Model**         | Regularized Logistic Regression (L1/L2) |
| **Advanced Models**        | • Random Forest Classifier  <br>• Gradient Boosting (XGBoost / LightGBM) <br>• Support Vector Classifier (SVM) |
| **Feature Selection**      | 1. Drop high‑cardinality identifiers (`transaction_id`). <br>2. Rank features using cross‑validated permutation importance **and** mutual information. <br>3. Remove collinear features with |r| > 0.85 (none observed above threshold). |
| **Validation Strategy**    | Stratified 5‑fold cross‑validation (preserves fraud proportion). <br>Metrics: Balanced Accuracy, Macro F1, Precision‑Recall AUC, Confusion Matrix. |
| **Over‑fitting Mitigation**| • Apply regularization (C‑parameter for LR/SVM, L1/L2). <br>• Limit tree depth, set `min_samples_leaf` for ensemble models. <br>• Conduct hyper‑parameter search **inside** CV folds (e.g., GridSearchCV, Optuna). |
| **Implementation Notes**   | • Encode categoricals via target encoding or frequency encoding (avoid one‑hot explosion). <br>• Scale numeric features (standard scaler or robust scaler due to skew). <br>• Consider SMOTE or class‑weighting to address 1.7 % fraud prevalence. |

---

## 8. Visual Artifacts (Generated PNGs)

| File (path)                              | Description |
|------------------------------------------|-------------|
| `correlation_matrix.png`                 | Heat‑map of Pearson correlations for all numeric features (saved in `./sandbox_run`). |
| `categorical_association_matrix.png`    | Heat‑map of Cramér’s V values for all categorical/binary pairs. |
| `dist_*.png` (24 files)                  | Univariate distribution plots for each column (histograms / bar charts). |
| `bivariate_*.png` (10 files)             | Bivariate visualizations of each significant predictor vs. `is_fraud` (box‑plots, violin plots, or stacked bars). |
| `target_interactions.png`                | Interaction plot for `amount_usd` against the target (e.g., partial dependence or SHAP interaction). |
| `eda_report.html`                        | Full HTML report (not reproduced here). |

All images are stored under the sandbox run directory and are ready for inclusion in stakeholder presentations.

---

## 9. Recommendations & Next Steps  

1. **Re‑run Feature Engineering** – Verify transformation logic (handle zeros, log‑transform only positive values) and ensure engineered columns are added to the dataframe.  
2. **Encoding Strategy** – Apply appropriate encoding for the 12‑level `merchant_category`, 5‑level `card_type`, etc. Target or frequency encoding is preferred to keep dimensionality low.  
3. **Class Imbalance Handling** – Experiment with class weighting in loss functions and/or oversampling techniques (SMOTE, ADASYN).  
4. **Model Prototyping** –  
   - Start with a regularized logistic regression baseline (quick to train, interpretable).  
   - Progress to tree‑based ensembles (Random Forest, XGBoost) to capture non‑linear interactions, especially those hinted by the strong `txn_count_last_24h` ↔ `velocity_score` relationship.  
5. **Evaluation** – Use stratified 5‑fold CV; report both ROC‑AUC (for completeness) and PR‑AUC (more informative for rare events).  
6. **Explainability** – Generate SHAP values for the best model to communicate feature impact to non‑technical stakeholders (e.g., `cvv_retry_count`, `used_vpn`, `merchant_risk_score`).  
7. **Production Checklist** – Ensure reproducible preprocessing (scalers, encoders) and version control of the final model artefacts.

---

### Closing Statement  

The dataset is clean, well‑profiled, and contains a rich set of statistically significant predictors for fraud detection. With proper feature engineering, encoding, and a disciplined modeling pipeline (as outlined above), a high‑performing, explainable fraud classifier can be delivered within a short development cycle.
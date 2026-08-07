# Executive EDA & Dataset Summary Report
**Target Directory:** `C:\Users\Anish Kumar Verma\PycharmProjects\AutoEDA\tests\benchmark_sandbox\credit_card_fraud_2026`
**Processed Files:** `metadata_profile.json`, `metrics.json`
**Excluded Files:** `generated_analysis.py` (Script excluded from summary)

---

## 1. Dataset Overview
- **Dataset Identifier:** `credit_card_fraud_2026.csv`
- **Dimensions:** `20000` rows x `26` columns
- **Target Variable:** `Not Specified`
- **Data Quality:** No missing values detected in raw profile.

---

## 1.5 Full Column Statistics
| Column | Type | Missing % | Unique % | Mean | Median | Std | Skew | Kurtosis |
|---|---|---|---|---|---|---|---|---|
| `transaction_id` | `int64` | 0.0% | 100.0% | 10000.5 | 10000.5 | 5773.65 | 0.0 | -1.2 |
| `amount_usd` | `float64` | 0.0% | 63.91% | 132.42 | 57.51 | 256.96 | 7.21 | 87.55 |
| `merchant_category` | `str` | 0.0% | 0.06% | N/A | N/A | N/A | N/A | N/A |
| `card_type` | `str` | 0.0% | 0.03% | N/A | N/A | N/A | N/A | N/A |
| `auth_method` | `str` | 0.0% | 0.03% | N/A | N/A | N/A | N/A | N/A |
| `channel` | `str` | 0.0% | 0.03% | N/A | N/A | N/A | N/A | N/A |
| `device_type` | `str` | 0.0% | 0.04% | N/A | N/A | N/A | N/A | N/A |
| `is_foreign_transaction` | `bool` | 0.0% | 0.01% | N/A | N/A | N/A | N/A | N/A |
| `hours_since_last_txn` | `float64` | 0.0% | 16.48% | 8.95 | 6.21 | 8.84 | 1.89 | 5.05 |
| `txn_count_last_24h` | `int64` | 0.0% | 0.07% | 3.19 | 3.0 | 1.78 | 0.55 | 0.24 |
| `distance_from_home_km` | `float64` | 0.0% | 30.61% | 22.14 | 15.5 | 22.12 | 2.03 | 6.24 |
| `card_age_months` | `int64` | 0.0% | 0.26% | 46.94 | 47.0 | 6.77 | 0.12 | -0.01 |
| `customer_age` | `int64` | 0.0% | 0.32% | 49.67 | 50.0 | 18.49 | -0.01 | -1.19 |
| `account_balance_usd` | `float64` | 0.0% | 98.15% | 3316.66 | 2007.68 | 4350.72 | 5.63 | 70.3 |
| `is_new_merchant` | `bool` | 0.0% | 0.01% | N/A | N/A | N/A | N/A | N/A |
| `used_vpn` | `bool` | 0.0% | 0.01% | N/A | N/A | N/A | N/A | N/A |
| `ip_country_mismatch` | `bool` | 0.0% | 0.01% | N/A | N/A | N/A | N/A | N/A |
| `billing_shipping_mismatch` | `bool` | 0.0% | 0.01% | N/A | N/A | N/A | N/A | N/A |
| `cvv_retry_count` | `int64` | 0.0% | 0.02% | 0.18 | 0.0 | 0.42 | 2.29 | 5.06 |
| `velocity_score` | `float64` | 0.0% | 3.17% | 19.81 | 18.8 | 12.37 | 0.54 | 0.17 |
| `time_of_day_hour` | `int64` | 0.0% | 0.12% | 11.53 | 12.0 | 6.93 | -0.01 | -1.21 |
| `day_of_week` | `int64` | 0.0% | 0.03% | 3.0 | 3.0 | 2.0 | 0.01 | -1.25 |
| `is_ai_generated_scam_attempt` | `bool` | 0.0% | 0.01% | N/A | N/A | N/A | N/A | N/A |
| `merchant_risk_score` | `float64` | 0.0% | 4.7% | 37.4 | 35.7 | 17.06 | 0.57 | 0.44 |
| `prior_disputes` | `int64` | 0.0% | 0.03% | 0.28 | 0.0 | 0.53 | 1.88 | 3.48 |
| `is_fraud` | `int64` | 0.0% | 0.01% | 0.02 | 0.0 | 0.13 | 7.48 | 54.03 |

---

## 2. Data Imputation & Preprocessing
- **status:** Imputation completed

---

## 3. Outlier Analysis (IQR Method)
No numeric outlier statistics reported.

---

## 4. Derived Domain Attributes & Composite Metrics
No custom derived domain metrics synthesized during this run.

---

## 5. Statistical Hypothesis Testing & Key Predictors
No statistically significant predictors identified.

---

## 6. Redundancy & Multicollinearity Analysis
No high-correlation or cross-type redundant feature pairs detected (threshold: |r| or Eta >= 0.85).

---

## 7. Generated Visualizations
No custom chart image assets found in directory (Interactive Plotly visualizations generated directly in HTML report).

---

## 8. Categorical Associations (Cramer's V)
| Feature 1 | Feature 2 | Cramer's V |
|---|---|---|
| `merchant_category` | `card_type` | 0.0 |
| `merchant_category` | `auth_method` | 0.0105 |
| `merchant_category` | `channel` | 0.0 |
| `merchant_category` | `device_type` | 0.0 |
| `merchant_category` | `is_foreign_transaction` | 0.0 |
| `merchant_category` | `is_new_merchant` | 0.0215 |
| `merchant_category` | `used_vpn` | 0.0 |
| `merchant_category` | `ip_country_mismatch` | 0.0 |
| `merchant_category` | `billing_shipping_mismatch` | 0.0 |
| `merchant_category` | `cvv_retry_count` | 0.0 |
| `merchant_category` | `day_of_week` | 0.0 |
| `merchant_category` | `is_ai_generated_scam_attempt` | 0.0126 |
| `merchant_category` | `prior_disputes` | 0.0 |
| `merchant_category` | `is_fraud` | 0.074 |
| `card_type` | `auth_method` | 0.0 |
| `card_type` | `channel` | 0.0072 |
| `card_type` | `device_type` | 0.0 |
| `card_type` | `is_foreign_transaction` | 0.0 |
| `card_type` | `is_new_merchant` | 0.0 |
| `card_type` | `used_vpn` | 0.0 |
| `card_type` | `ip_country_mismatch` | 0.0058 |
| `card_type` | `billing_shipping_mismatch` | 0.0 |
| `card_type` | `cvv_retry_count` | 0.0 |
| `card_type` | `day_of_week` | 0.0057 |
| `card_type` | `is_ai_generated_scam_attempt` | 0.0 |
| `card_type` | `prior_disputes` | 0.0062 |
| `card_type` | `is_fraud` | 0.0048 |
| `auth_method` | `channel` | 0.0 |
| `auth_method` | `device_type` | 0.0 |
| `auth_method` | `is_foreign_transaction` | 0.0 |
| `auth_method` | `is_new_merchant` | 0.0089 |
| `auth_method` | `used_vpn` | 0.0 |
| `auth_method` | `ip_country_mismatch` | 0.0 |
| `auth_method` | `billing_shipping_mismatch` | 0.0 |
| `auth_method` | `cvv_retry_count` | 0.0 |
| `auth_method` | `day_of_week` | 0.0 |
| `auth_method` | `is_ai_generated_scam_attempt` | 0.0192 |
| `auth_method` | `prior_disputes` | 0.0054 |
| `auth_method` | `is_fraud` | 0.0801 |
| `channel` | `device_type` | 0.0022 |
| `channel` | `is_foreign_transaction` | 0.0 |
| `channel` | `is_new_merchant` | 0.0 |
| `channel` | `used_vpn` | 0.0104 |
| `channel` | `ip_country_mismatch` | 0.0079 |
| `channel` | `billing_shipping_mismatch` | 0.1605 |
| `channel` | `cvv_retry_count` | 0.0055 |
| `channel` | `day_of_week` | 0.0103 |
| `channel` | `is_ai_generated_scam_attempt` | 0.0992 |
| `channel` | `prior_disputes` | 0.0 |
| `channel` | `is_fraud` | 0.0105 |
| `device_type` | `is_foreign_transaction` | 0.0 |
| `device_type` | `is_new_merchant` | 0.0 |
| `device_type` | `used_vpn` | 0.0112 |
| `device_type` | `ip_country_mismatch` | 0.0251 |
| `device_type` | `billing_shipping_mismatch` | 0.0 |
| `device_type` | `cvv_retry_count` | 0.0 |
| `device_type` | `day_of_week` | 0.0082 |
| `device_type` | `is_ai_generated_scam_attempt` | 0.0138 |
| `device_type` | `prior_disputes` | 0.0 |
| `device_type` | `is_fraud` | 0.0041 |
| `is_foreign_transaction` | `is_new_merchant` | 0.0 |
| `is_foreign_transaction` | `used_vpn` | 0.0 |
| `is_foreign_transaction` | `ip_country_mismatch` | 0.2722 |
| `is_foreign_transaction` | `billing_shipping_mismatch` | 0.0101 |
| `is_foreign_transaction` | `cvv_retry_count` | 0.0 |
| `is_foreign_transaction` | `day_of_week` | 0.0 |
| `is_foreign_transaction` | `is_ai_generated_scam_attempt` | 0.0075 |
| `is_foreign_transaction` | `prior_disputes` | 0.0198 |
| `is_foreign_transaction` | `is_fraud` | 0.0832 |
| `is_new_merchant` | `used_vpn` | 0.0111 |
| `is_new_merchant` | `ip_country_mismatch` | 0.0098 |
| `is_new_merchant` | `billing_shipping_mismatch` | 0.0091 |
| `is_new_merchant` | `cvv_retry_count` | 0.0146 |
| `is_new_merchant` | `day_of_week` | 0.0 |
| `is_new_merchant` | `is_ai_generated_scam_attempt` | 0.0016 |
| `is_new_merchant` | `prior_disputes` | 0.0 |
| `is_new_merchant` | `is_fraud` | 0.0761 |
| `used_vpn` | `ip_country_mismatch` | 0.0 |
| `used_vpn` | `billing_shipping_mismatch` | 0.0 |
| `used_vpn` | `cvv_retry_count` | 0.0 |
| `used_vpn` | `day_of_week` | 0.0 |
| `used_vpn` | `is_ai_generated_scam_attempt` | 0.0 |
| `used_vpn` | `prior_disputes` | 0.0041 |
| `used_vpn` | `is_fraud` | 0.0641 |
| `ip_country_mismatch` | `billing_shipping_mismatch` | 0.0102 |
| `ip_country_mismatch` | `cvv_retry_count` | 0.0 |
| `ip_country_mismatch` | `day_of_week` | 0.0 |
| `ip_country_mismatch` | `is_ai_generated_scam_attempt` | 0.0 |
| `ip_country_mismatch` | `prior_disputes` | 0.0029 |
| `ip_country_mismatch` | `is_fraud` | 0.1176 |
| `billing_shipping_mismatch` | `cvv_retry_count` | 0.0 |
| `billing_shipping_mismatch` | `day_of_week` | 0.0 |
| `billing_shipping_mismatch` | `is_ai_generated_scam_attempt` | 0.0198 |
| `billing_shipping_mismatch` | `prior_disputes` | 0.0 |
| `billing_shipping_mismatch` | `is_fraud` | 0.1006 |
| `cvv_retry_count` | `day_of_week` | 0.0 |
| `cvv_retry_count` | `is_ai_generated_scam_attempt` | 0.0 |
| `cvv_retry_count` | `prior_disputes` | 0.0 |
| `cvv_retry_count` | `is_fraud` | 0.1657 |
| `day_of_week` | `is_ai_generated_scam_attempt` | 0.0 |
| `day_of_week` | `prior_disputes` | 0.0 |
| `day_of_week` | `is_fraud` | 0.0 |
| `is_ai_generated_scam_attempt` | `prior_disputes` | 0.0 |
| `is_ai_generated_scam_attempt` | `is_fraud` | 0.0786 |
| `prior_disputes` | `is_fraud` | 0.0404 |

---

## 9. Predictive Modeling Strategy Blueprint
- **Target Definition:** Undefined (Unsupervised)
- **Problem Type:** Unsupervised / Exploratory
### Recommended Algorithms
- K-Means Clustering
- Hierarchical Agglomerative Clustering
- Principal Component Analysis (PCA) for Dimensionality Reduction
### Feature Selection Strategy
- Exclude high-cardinality ID or text name columns
- Rank features using cross-validated permutation importance and mutual information
- Remove collinear features exceeding correlation threshold > 0.85
### Validation Strategy
- Evaluate Silhouette Score and Inertia elbow curve
### Overfitting Risk Mitigation
- Apply regularization penalties (L1/L2)
- Limit tree depth and enforce minimum samples per leaf
- Perform hyperparameter tuning strictly within cross-validation folds
- **Executive Summary:** Target: 'Undefined (Unsupervised)' (Unsupervised / Exploratory). Model recommendations and validation strategy tailored for 20000 rows x 26 columns.

---

*Report generated automatically by `summary_generator.py`*
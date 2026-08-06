# Executive Summary Report: Credit Card Fraud Detection EDA

## 1. Dataset Overview

The automated Exploratory Data Analysis (EDA) pipeline has processed a dataset named `credit_card_fraud_2026.csv`, containing **20,000 rows** and **26 columns**. The target variable for classification is `is_fraud`, which indicates whether a transaction is fraudulent (0 = not fraudulent, 1 = fraudulent). The dataset exhibits no missing values across all features, indicating complete data coverage.

### Key Metadata
| Metric                  | Value                     |
|------------------------|---------------------------|
| Total Rows             | 20,000                    |
| Total Columns          | 26                        |
| Target Variable        | is_fraud                  |
| Data Type              | Classification            |
| Imputation Applied     | None (no missing values)  |

---

## 2. Statistical Summary & Feature Distributions

All numeric and categorical features were profiled. Below are key statistical summaries:

### Numerical Features (Summary Statistics)
| Feature               | Mean       | Median    | Skewness | Outlier % | Notes                          |
|-----------------------|------------|-----------|----------|-----------|--------------------------------|
| amount_usd            | 132.42     | 57.51     | 7.21     | 10.3%     | Highly skewed                 |
| hours_since_last_txn  | 8.95       | 6.21      | 1.89     | 4.78%     | Highly skewed                 |
| distance_from_home_km | 22.14      | 15.50     | 2.03     | 4.52%     | Highly skewed                 |
| account_balance_usd   | 3,316.66   | 2,007.68  | 5.63     | 7.85%     | Highly skewed                 |
| velocity_score        | 19.81      | 18.80     | —        | 1.06%     | Low outlier count             |
| txn_count_last_24h    | 3.19       | 3.00      | —        | —         | Low variance                   |

### Categorical Features (Top Values)
| Feature               | Top Value(s)                         | Count   |
|-----------------------|-------------------------------------|---------|
| merchant_category     | Online Retail, Groceries, Restaurants | 3,416 / 3,242 / 2,534 |
| card_type             | Visa, Mastercard, Amex              | 8,417 / 6,629 / 2,135 |
| auth_method           | 3D Secure, OTP, PIN                 | 5,554 / 4,792 / 4,173 |
| channel               | Online, POS, Contactless            | 6,810 / 5,191 / 3,400 |
| device_type           | Android Phone, iPhone, POS Terminal | 5,503 / 4,179 / 3,162 |

---

## 3. Outlier Analysis

Outliers were detected using the IQR method. The following features exhibited significant outlier presence (>4%):

| Feature               | Outlier Count | Outlier % | Action Taken |
|-----------------------|---------------|-----------|--------------|
| amount_usd            | 2,061         | 10.3%     | Profiled     |
| hours_since_last_txn  | 955           | 4.78%     | Profiled     |
| distance_from_home_km | 904           | 4.52%     | Profiled     |
| account_balance_usd   | 1,569         | 7.85%     | Profiled     |
| velocity_score        | 213           | 1.06%     | Profiled     |

> *Note: All outliers were flagged for further investigation but not removed or transformed in this phase.*

---

## 4. Correlation Analysis

A Pearson correlation matrix was computed, revealing strong relationships between certain features. The top correlations are summarized below:

### Top Positive Correlations
| Feature 1           | Feature 2         | Correlation | Significance |
|---------------------|-------------------|-------------|--------------|
| txn_count_last_24h  | velocity_score    | 0.622       | ✅           |
| cvv_retry_count     | is_fraud          | 0.156       | ✅           |
| merchant_risk_score | is_fraud          | 0.108       | ✅           |
| amount_usd          | merchant_risk_score | 0.097     | ✅           |
| velocity_score      | is_fraud          | 0.097       | ✅           |

### Top Negative Correlations
| Feature 1           | Feature 2         | Correlation | Significance |
|---------------------|-------------------|-------------|--------------|
| hours_since_last_txn | velocity_score    | -0.266      | ✅           |
| time_of_day_hour    | is_fraud          | -0.031      | ✅           |

> *Correlation heatmap saved as `correlation_matrix.png`*

---

## 5. Statistical Hypothesis Testing

Statistical significance was tested using Pearson correlation and One-Way ANOVA for categorical features. The following features showed statistically significant relationships with the target (`is_fraud`):

### Statistically Significant Predictors (p < 0.05)
| Feature               | Test Type             | p-value         | Interpretation                                  |
|-----------------------|-----------------------|-----------------|------------------------------------------------|
| amount_usd            | Pearson Correlation  | 4.17e-04        | Strongly associated with fraud                 |
| merchant_category     | One-Way ANOVA        | 1.16e-20        | Category significantly impacts fraud risk      |
| auth_method           | One-Way ANOVA        | 1.07e-27        | Authentication method matters                  |
| is_foreign_transaction| Welch T-Test         | 4.99e-11        | Foreign transactions more likely to be fraud   |
| hours_since_last_txn  | Pearson Correlation  | 4.64e-05        | Shorter time since last txn → higher fraud risk|
| txn_count_last_24h    | Pearson Correlation  | 2.38e-17        | High transaction volume → higher fraud risk    |
| customer_age          | Pearson Correlation  | 5.71e-03        | Older customers less likely to be victims?     |
| is_new_merchant       | Welch T-Test         | 2.09e-16        | New merchants linked to fraud                  |
| used_vpn              | Welch T-Test         | 3.56e-09        | VPN usage correlates with fraud                |
| ip_country_mismatch   | Welch T-Test         | 5.72e-16        | IP mismatch strongly signals fraud             |
| billing_shipping_mismatch | Welch T-Test     | 5.38e-12        | Mismatched addresses indicate fraud            |
| cvv_retry_count       | Pearson Correlation  | 1.79e-108       | Multiple CVV retries → high fraud likelihood   |
| velocity_score        | Pearson Correlation  | 3.10e-43        | Velocity score predicts fraud                  |
| time_of_day_hour      | Pearson Correlation  | 1.31e-05        | Late-night transactions more risky             |
| is_ai_generated_scam_attempt | Welch T-Test   | 6.87e-07        | AI-generated scams highly correlated with fraud|
| merchant_risk_score   | Pearson Correlation  | 1.18e-52        | Higher merchant risk → higher fraud probability|
| prior_disputes        | Pearson Correlation  | 7.23e-04        | Prior disputes increase fraud risk             |

> **Executive Summary of Significant Predictors**:  
> `amount_usd`, `merchant_category`, `auth_method`, `is_foreign_transaction`, `hours_since_last_txn`, `txn_count_last_24h`, `customer_age`, `is_new_merchant`, `used_vpn`, `ip_country_mismatch`, `billing_shipping_mismatch`, `cvv_retry_count`, `velocity_score`, `time_of_day_hour`, `is_ai_generated_scam_attempt`, `merchant_risk_score`, `prior_disputes`

---

## 6. Bivariate Visualizations

Key bivariate plots were generated to explore feature-target interactions:

### Key Plots:
- **`bivariate_amount_usd_vs_merchant_risk_score.png`**: Shows positive trend — higher merchant risk scores correlate with larger transaction amounts.
- **`bivariate_customer_age_vs_velocity_score.png`**: Suggests older customers have lower velocity scores, implying less frequent activity.
- **`bivariate_cvv_retry_count_vs_is_fraud.png`**: Strong association — higher retry counts → higher fraud likelihood.
- **`bivariate_distance_from_home_km_vs_is_foreign_transaction.png`**: Transactions from far distances are more likely to be foreign.

> *All visualizations are stored in the working directory.*

---

## 7. Feature Engineering Highlights

No engineered features were created during this EDA phase. However, the following insights suggest potential future engineering:

- **Velocity Score**: Could be enhanced by incorporating rolling window calculations.
- **Merchant Risk Score**: May benefit from normalization or binning.
- **Time-based Features**: Consider creating “hour-of-day” bins or “weekend vs weekday” flags.
- **Interaction Terms**: Potential for `is_foreign_transaction × merchant_risk_score` or `cvv_retry_count × txn_count_last_24h`.

---

## 8. Predictive Modeling Blueprint

### Problem Definition
- **Target**: `is_fraud` (Binary Classification)
- **Dataset Size**: 20,000 samples × 26 features
- **Goal**: Build a robust fraud detection model with high precision and recall.

### Recommended Algorithms
1. Regularized Logistic Regression (Baseline)
2. Random Forest Classifier
3. Gradient Boosting Classifier (XGBoost / LightGBM)
4. Support Vector Classifier (SVM)

### Feature Selection Strategy
- Exclude ID columns (`transaction_id`) and low-information categorical variables.
- Use cross-validated permutation importance and mutual information to rank features.
- Remove collinear features with correlation > 0.85.

### Validation Strategy
- Stratified K-Fold Cross-Validation (5 folds)
- Metrics: Balanced Accuracy, Macro F1, Precision-Recall AUC, Confusion Matrix

### Overfitting Mitigation
- Apply L1/L2 regularization
- Limit tree depth and enforce minimum samples per leaf
- Hyperparameter tuning within CV folds only

---

## 9. Image Artifact Descriptions

The following visual artifacts were generated and saved:

| Artifact Name                      | Description                                                                 |
|------------------------------------|-----------------------------------------------------------------------------|
| `correlation_matrix.png`           | Heatmap showing pairwise correlations among all features.                   |
| `pairplot.png`                     | Scatterplots of all numerical pairs + histograms on diagonal.               |
| `target_interactions.png`          | Visualization of how each feature interacts with the target (`is_fraud`).   |
| `bivariate_*_vs_is_fraud.png`      | 5 key bivariate scatterplots highlighting strongest feature-target links.   |
| `dist_*_png`                       | Distribution plots for all continuous and binary features.                  |

> *All images are stored locally and can be viewed via file explorer or Jupyter notebook.*

---

## 10. Conclusion & Recommendations

This EDA reveals that the dataset is rich in predictive signals, particularly around transaction behavior, authentication methods, and user/device context. The most powerful predictors include:

- **Behavioral Signals**: `cvv_retry_count`, `velocity_score`, `txn_count_last_24h`
- **Risk Indicators**: `merchant_risk_score`, `ip_country_mismatch`, `used_vpn`
- **Contextual Flags**: `is_foreign_transaction`, `is_new_merchant`, `is_ai_generated_scam_attempt`

### Next Steps:
1. Implement baseline logistic regression model.
2. Perform hyperparameter tuning using GridSearchCV.
3. Engineer new features based on interaction patterns.
4. Deploy ensemble models (Random Forest + XGBoost) for improved performance.
5. Monitor model drift and retrain periodically.

This dataset is well-suited for building a production-grade fraud detection system with high accuracy and interpretability.

--- 

*Generated by Senior Lead Data Scientist — AutoEDA Pipeline Output*
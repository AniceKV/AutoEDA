# Executive EDA & Dataset Summary Report
**Target Directory:** `C:\Users\Anish Kumar Verma\PycharmProjects\AutoEDA\tests\benchmark_sandbox\synthetic_credit_card_customer_behavior_dataset`
**Processed Files:** `metadata_profile.json`, `metrics.json`
**Excluded Files:** `generated_analysis.py` (Script excluded from summary)

---

## 1. Dataset Overview
- **Dataset Identifier:** `synthetic_credit_card_customer_behavior_dataset.csv`
- **Dimensions:** `50000` rows x `30` columns
- **Target Variable:** `Not Specified`
- **Data Quality:** No missing values detected in raw profile.

---

## 1.5 Full Column Statistics
| Column | Type | Missing % | Unique % | Mean | Median | Std | Skew | Kurtosis |
|---|---|---|---|---|---|---|---|---|
| `Customer_ID` | `str` | 0.0% | 100.0% | N/A | N/A | N/A | N/A | N/A |
| `Age` | `int64` | 0.0% | 0.11% | 36.08 | 34.0 | 12.24 | 0.69 | -0.12 |
| `Gender` | `str` | 0.0% | 0.0% | N/A | N/A | N/A | N/A | N/A |
| `Annual_Income` | `int64` | 0.0% | 98.97% | 1479317.91 | 1127629.5 | 1407257.48 | 2.55 | 6.95 |
| `Occupation` | `str` | 0.0% | 0.01% | N/A | N/A | N/A | N/A | N/A |
| `Card_Type` | `str` | 0.0% | 0.01% | N/A | N/A | N/A | N/A | N/A |
| `Credit_Limit` | `float64` | 0.0% | 8.64% | 709677.22 | 396000.0 | 1009854.42 | 3.01 | 9.45 |
| `Card_Age_Months` | `int64` | 0.0% | 0.45% | 53.63 | 48.0 | 31.2 | 1.14 | 1.86 |
| `Monthly_Spending` | `float64` | 0.0% | 99.8% | 73871.8 | 50022.38 | 84338.27 | 3.0 | 10.56 |
| `Monthly_Transactions` | `int64` | 0.0% | 0.33% | 83.04 | 76.0 | 42.05 | 0.54 | -0.61 |
| `Avg_Transaction_Value` | `float64` | 0.0% | 80.95% | 835.99 | 617.16 | 779.59 | 3.9 | 20.69 |
| `Online_Shopping_Spending` | `float64` | 0.0% | 98.57% | 11238.28 | 6163.21 | 16599.42 | 4.56 | 30.33 |
| `Grocery_Spending` | `float64` | 0.0% | 99.18% | 17794.28 | 10651.41 | 23697.66 | 4.01 | 23.59 |
| `Fuel_Spending` | `float64` | 0.0% | 98.53% | 10212.73 | 5466.36 | 15509.64 | 4.69 | 32.34 |
| `Dining_Spending` | `float64` | 0.0% | 98.27% | 9090.1 | 4847.46 | 13712.97 | 4.64 | 32.86 |
| `Travel_Spending` | `float64` | 0.0% | 97.17% | 6141.73 | 2883.43 | 10685.75 | 5.71 | 51.38 |
| `Entertainment_Spending` | `float64` | 0.0% | 97.87% | 7648.22 | 3829.69 | 12348.37 | 5.01 | 38.47 |
| `Utility_Bill_Spending` | `float64` | 0.0% | 98.72% | 11746.45 | 6574.18 | 16824.44 | 4.35 | 28.83 |
| `Outstanding_Balance` | `float64` | 0.0% | 99.94% | 303420.29 | 145306.33 | 499569.54 | 4.02 | 20.49 |
| `Statement_Balance` | `float64` | 0.0% | 99.96% | 366172.44 | 192358.4 | 559547.63 | 3.8 | 18.1 |
| `Payment_Amount` | `float64` | 0.0% | 99.94% | 279373.95 | 136639.78 | 453396.64 | 4.2 | 23.49 |
| `Payment_Ratio` | `float64` | 0.0% | 0.18% | 0.76 | 0.84 | 0.24 | -1.04 | 0.04 |
| `Credit_Utilization` | `float64` | 0.0% | 0.17% | 0.43 | 0.4 | 0.21 | 0.53 | -0.43 |
| `Cash_Advance_Amount` | `float64` | 0.0% | 29.65% | 23056.76 | 0.0 | 77966.35 | 6.78 | 58.43 |
| `EMI_Count` | `int64` | 0.0% | 0.01% | 2.45 | 2.0 | 1.67 | 0.17 | -0.86 |
| `International_Transactions` | `int64` | 0.0% | 0.03% | 1.53 | 0.0 | 2.6 | 2.11 | 4.61 |
| `Reward_Points_Earned` | `int64` | 0.0% | 23.36% | 3891.41 | 1833.0 | 6445.46 | 3.5 | 13.84 |
| `Reward_Points_Redeemed` | `int64` | 0.0% | 11.03% | 871.99 | 135.0 | 2356.9 | 6.55 | 59.75 |
| `Mobile_App_Login` | `int64` | 0.0% | 0.14% | 33.16 | 31.0 | 16.85 | 0.33 | -0.76 |
| `Credit_Score` | `int64` | 0.0% | 0.8% | 628.98 | 638.0 | 68.44 | -0.58 | -0.05 |

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
**Numeric-Numeric High Correlation Pairs (|r| >= 0.85):**

| Feature 1 | Feature 2 | Correlation (r) | Interpretation |
|---|---|---|---|
| `Credit_Limit` | `Monthly_Spending` | 0.9403 | Strong correlation |
| `Credit_Limit` | `Outstanding_Balance` | 0.8661 | Strong correlation |
| `Credit_Limit` | `Statement_Balance` | 0.8938 | Strong correlation |
| `Credit_Limit` | `Reward_Points_Earned` | 0.9581 | Strong correlation |
| `Monthly_Spending` | `Avg_Transaction_Value` | 0.8822 | Strong correlation |
| `Monthly_Spending` | `Grocery_Spending` | 0.8567 | Strong correlation |
| `Monthly_Spending` | `Statement_Balance` | 0.8553 | Strong correlation |
| `Monthly_Spending` | `Reward_Points_Earned` | 0.9797 | Strong correlation |
| `Avg_Transaction_Value` | `Reward_Points_Earned` | 0.8734 | Strong correlation |
| `Outstanding_Balance` | `Statement_Balance` | 0.9971 | Strong correlation |
| `Outstanding_Balance` | `Payment_Amount` | 0.9324 | Strong correlation |
| `Statement_Balance` | `Payment_Amount` | 0.9359 | Strong correlation |
| `Statement_Balance` | `Reward_Points_Earned` | 0.8665 | Strong correlation |

**Cross-Type Redundant Pairs (categorical vs. its own numeric/ordinal encoding, Eta >= 0.85):**

| Categorical Feature | Numeric Feature | Correlation Ratio (Eta) | Interpretation |
|---|---|---|---|
| `Card_Type` | `Credit_Limit` | 0.8943 | High cross-type redundancy between 'Card_Type' and 'Credit_Limit' (Eta = 0.8943). |
| `Card_Type` | `Reward_Points_Earned` | 0.8748 | High cross-type redundancy between 'Card_Type' and 'Reward_Points_Earned' (Eta = 0.8748). |

_Recommendation: drop one feature from each redundant pair before modeling to avoid multicollinearity._

---

## 7. Generated Visualizations
No custom chart image assets found in directory (Interactive Plotly visualizations generated directly in HTML report).

---

## 8. Categorical Associations (Cramer's V)
| Feature 1 | Feature 2 | Cramer's V |
|---|---|---|
| `Gender` | `Occupation` | 0.0 |
| `Gender` | `Card_Type` | 0.0058 |
| `Gender` | `EMI_Count` | 0.0085 |
| `Occupation` | `Card_Type` | 0.4653 |
| `Occupation` | `EMI_Count` | 0.2566 |
| `Card_Type` | `EMI_Count` | 0.354 |

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
- **Executive Summary:** Target: 'Undefined (Unsupervised)' (Unsupervised / Exploratory). Model recommendations and validation strategy tailored for 50000 rows x 30 columns.

---

*Report generated automatically by `summary_generator.py`*
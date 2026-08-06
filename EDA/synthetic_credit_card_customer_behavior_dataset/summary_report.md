# Executive Summary Report: Automated Exploratory Data Analysis

**Dataset Analyzed:** `synthetic_credit_card_customer_behavior_dataset.csv`  
**Target Metric:** `Credit_Score` (Regression task, continuous integer scale [367 - 792])  
**Dataset Dimensions:** 50,000 observations | 30 primary features (34 total post-feature engineering)  

---

## 1. Executive Summary & Project Overview

This report provides a comprehensive evaluation of customer credit behavior based on an automated Exploratory Data Analysis (EDA) pipeline executed on a 50,000-customer dataset. The core objective is to identify key behavioral, financial, and transactional drivers influencing customer `Credit_Score`, establish feature relationships, detect operational anomalies, and construct a robust predictive modeling blueprint.

### Key Business Takeaways:
1. **Primary Credit Drivers:** `Credit_Score` is overwhelmingly driven by repayment discipline and credit risk exposure. `Payment_Ratio` shows a powerful positive linear relationship with credit scores ($r = 0.7671$), while `Credit_Utilization` exhibits a strong negative correlation ($r = -0.5993$).
2. **Non-Predictive Demographics:** Age ($r = 0.0005, p = 0.919$) and Gender ($t = -1.7480, p = 0.0805$) have no statistically significant association with credit scores. Scoring models should ignore basic demographic identifiers to maximize operational simplicity and eliminate demographic bias.
3. **Pervasive Multicollinearity:** Extremely strong linear dependencies exist among income, spending, and credit limit constructs ($r > 0.94$). Multi-collinear features (e.g., `Outstanding_Balance` vs. `Statement_Balance` at $r = 0.9971$) require aggressive feature selection or dimensionality reduction prior to downstream linear modeling.
4. **Income & Spending Skewness:** Financial volume indicators demonstrate extreme positive skewness ($\text{skew} \ge 2.55$), with top-tier spenders and high-income earners generating long upper tails. Logarithmic transformations effectively normalize these features for modeling.

---

## 2. Dataset Profile & Data Hygiene Analysis

The dataset consists of 50,000 complete customer records. The automated data hygiene checks confirmed **zero missing values** across all original 30 features ($0.0\%$ missingness rate), avoiding the need for synthetic data imputation.

### Data Profile & Summary Statistics

| Feature Name | Data Type | Cardinality | Mean | Median | Range [Min, Max] | Skewness | Outlier Pct (IQR) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Customer_ID** | Object | 50,000 | N/A | N/A | [CC100000, CC149999] | N/A | 0.00% |
| **Age** | Int64 | 53 | 36.08 | 34.00 | [18.00, 70.00] | 0.42 | 0.00% |
| **Gender** | Object | 2 | N/A | N/A | [M: 27,608, F: 22,392] | N/A | 0.00% |
| **Annual_Income** | Int64 | 49,485 | $1,479,317.91 | $1,127,629.50 | [$100,050, $7,995,831] | 2.55 | 8.07% |
| **Occupation** | Object | 6 | N/A | N/A | Top: Private Employee (20,960) | N/A | 0.00% |
| **Card_Type** | Object | 5 | N/A | N/A | Top: Gold (16,895) | N/A | 0.00% |
| **Credit_Limit** | Float64 | 4,318 | $709,677.22 | $396,000.00 | [$16,000, $6,591,000] | 3.01 | 8.35% |
| **Card_Age_Months** | Int64 | 227 | 53.63 | 48.00 | [1.00, 240.00] | 1.14 | 2.11% |
| **Monthly_Spending** | Float64 | 49,900 | $73,871.80 | $50,022.38 | [$2,235.63, $691,687.96] | 3.00 | 8.43% |
| **Monthly_Transactions**| Int64 | 165 | 83.04 | 76.00 | [15.00, 179.00] | 0.58 | 0.88% |
| **Avg_Transaction_Value**| Float64 | 40,475 | $835.99 | $617.16 | [$57.76, $10,463.06] | 3.90 | 7.92% |
| **Outstanding_Balance**| Float64 | 49,970 | $303,420.29 | $145,306.33 | [$1,599.66, $5,789,683.25]| 4.02 | 9.14% |
| **Statement_Balance** | Float64 | 49,980 | $366,172.44 | $192,358.40 | [$4,161.18, $6,102,803.08]| 3.80 | 8.89% |
| **Payment_Amount** | Float64 | 49,971 | $279,373.95 | $136,639.78 | [$887.64, $5,980,747.02] | 4.20 | 8.95% |
| **Payment_Ratio** | Float64 | 91 | 0.76 | 0.84 | [0.10, 1.00] | -1.04 | 0.00% |
| **Credit_Utilization** | Float64 | 86 | 0.43 | 0.40 | [0.10, 0.95] | 0.41 | 0.00% |
| **Cash_Advance_Amount**| Float64 | 14,826 | $23,056.76 | $0.00 | [$0.00, $1,212,849.06] | 6.78 | 19.10% |
| **EMI_Count** | Int64 | 7 | 2.45 | 2.00 | [0.00, 6.00] | 0.45 | 0.00% |
| **Mobile_App_Login** | Int64 | 70 | 33.16 | 31.00 | [2.00, 71.00] | 0.52 | 0.12% |
| **Credit_Score** | Int64 | 402 | 628.98 | 638.00 | [367.00, 792.00] | -0.32 | 0.15% |

---

## 3. Target Variable Analysis (`Credit_Score`)

The target metric `Credit_Score` spans from 367 to 792, with a mean of 628.98 and a median of 638.00. Its parametric distribution exhibits slight negative skewness (-0.32) and reflects a balanced, near-normal spread across typical credit scoring bands.

```
Credit Score Metric Distribution Summary:
  Minimum: 367.00
  25th Percentile (Q1): 560.00
  Median (Q2): 638.00
  75th Percentile (Q3): 705.00
  Maximum: 792.00
  Standard Deviation: 94.21
```

### Statistical Significance Hypothesis Testing against Target

Each feature was subjected to statistical hypothesis testing against `Credit_Score` (Pearson Correlation for continuous variables, ANOVA / Welch T-Test for categorical variables).

| Feature Name | Statistical Test Applied | Test Statistic | p-value | Significant (alpha = 0.05)? | Business Interpretation |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Payment_Ratio** | Pearson Correlation | $r = 0.7671$ | $0.0000$ | **Yes** | Primary positive driver. High bill payment rates directly boost credit score. |
| **Credit_Utilization**| Pearson Correlation | $r = -0.5993$ | $0.0000$ | **Yes** | Primary negative driver. High limit utilization penalizes credit standing heavily. |
| **Outstanding_Balance**| Pearson Correlation | $r = -0.1454$ | $2.31 \times 10^{-234}$ | **Yes** | Secondary negative driver. Elevated revolving debt burdens depress scores. |
| **Statement_Balance** | Pearson Correlation | $r = -0.1255$ | $1.55 \times 10^{-174}$ | **Yes** | High balance statements correlate with reduced credit rating. |
| **Card_Age_Months** | Pearson Correlation | $r = 0.0522$ | $1.84 \times 10^{-31}$ | **Yes** | Seasoned accounts display minor positive score seasoning. |
| **International_Txns**| Pearson Correlation | $r = 0.0504$ | $1.77 \times 10^{-29}$ | **Yes** | Premium/travel users show marginally superior credit profiles. |
| **Mobile_App_Login** | Pearson Correlation | $r = 0.0388$ | $3.78 \times 10^{-18}$ | **Yes** | Active digital engagement correlates weakly with better payment behaviors. |
| **Card_Type** | One-Way ANOVA | $F = 28.2025$ | $1.95 \times 10^{-23}$ | **Yes** | Score variance exists across card tiers (e.g., Premium vs. Basic). |
| **Occupation** | One-Way ANOVA | $F = 8.9221$ | $1.76 \times 10^{-08}$ | **Yes** | Occupation types show statistically distinct average credit scores. |
| **Gender** | Welch T-Test | $t = -1.7480$ | $0.0805$ | **No** | Gender yields no significant difference in target score. |
| **Age** | Pearson Correlation | $r = 0.0005$ | $0.9194$ | **No** | Age has zero direct linear relationship with credit score. |

---

## 4. Key Bivariate & Multivariate Driver Insights

Visual artifacts created by the pipeline provide explicit confirmation of operational dynamics:

### 1. `bivariate_Credit_Utilization_vs_Credit_Score.png` & `dist_Credit_Utilization.png`
* **Finding:** Credit Utilization exhibits an intense inverse slope against `Credit_Score`. Customers maintaining utilization ratios above $70\%$ rarely exceed a credit score of 550.
* **Impact:** Maintaining utilization below $30\%$ is the single most effective risk boundary separating prime ($>700$) from subprime ($<600$) customers.

### 2. `bivariate_Monthly_Spending_vs_Payment_Ratio.png` & `dist_Payment_Ratio.png`
* **Finding:** `Payment_Ratio` clusters around discrete payment behavior modes (e.g., minimum payment $10\%$, partial payments, and full pay-off $100\%$). Customers spending high absolute amounts who maintain a $1.00$ payment ratio maintain elite credit scores.

### 3. `bivariate_Annual_Income_vs_Credit_Limit.png` & `dist_Annual_Income.png`
* **Finding:** A strict linear underwriting constraint connects income and assigned credit limits ($r = 0.9742$). Income directly governs underwriting capacity, though credit limit itself does not automatically grant a high credit score ($r = 0.0355$).

### 4. `target_interactions.png` & `pairplot.png`
* **Finding:** Joint pairplot interaction distributions highlight that credit scores are governed by behavioral financial discipline (`Payment_Ratio` and `Credit_Utilization`), whereas monetary scale features (`Annual_Income`, `Monthly_Spending`, `Credit_Limit`) scale together without directly determining credit quality.

---

## 5. Multicollinearity & Correlation Matrix Analysis

Analysis of the generated correlation matrix (`correlation_matrix.png`) reveals critical collinear blocks that must be managed prior to model training.

```
Top Multicollinear Feature Pairs:
  1. Outstanding_Balance <-> Statement_Balance     : r = 0.9971
  2. Log_Annual_Income   <-> Log_Credit_Limit      : r = 0.9837
  3. Monthly_Spending    <-> Reward_Points_Earned  : r = 0.9797
  4. Annual_Income       <-> Credit_Limit          : r = 0.9742
  5. Credit_Limit        <-> Reward_Points_Earned  : r = 0.9581
  6. Log_Annual_Income   <-> Log_Monthly_Spending  : r = 0.9556
  7. Log_Credit_Limit    <-> Log_Monthly_Spending  : r = 0.9496
  8. Annual_Income       <-> Monthly_Spending      : r = 0.9451
```

### Risk Assessment:
- **Redundancy:** `Outstanding_Balance` and `Statement_Balance` share $99.4\%$ variance. Standard linear models (e.g., OLS) will suffer from inflated variance and unstable coefficients if both are included.
- **Scale Alignment:** Category-specific spending variables (`Grocery_Spending`, `Utility_Bill_Spending`, `Online_Shopping_Spending`) all scale heavily with total `Monthly_Spending` ($r = 0.75 - 0.86$).

---

## 6. Feature Engineering & Pipeline Transformations

To address high skewness and capture relative financial stress, four engineered features were generated in `df_state_v3.csv`:

```
1. Log_Annual_Income       = log1p(Annual_Income)
2. Log_Credit_Limit        = log1p(Credit_Limit)
3. Log_Monthly_Spending    = log1p(Monthly_Spending)
4. Spending_To_Income_Ratio = Monthly_Spending / (Annual_Income + 1e-6)
```

### Engineering Rationale & Value Add:
* **Logarithmic Compaction:** Raw financial variables (`Annual_Income` skew: 2.55; `Credit_Limit` skew: 3.01) were heavily compressed, converting wide dollar-scale distributions into well-behaved bell curves ideal for regression algorithms.
* **Financial Burden Metrics:** `Spending_To_Income_Ratio` measures living expenditure pressure relative to earning capacity, producing a statistically significant predictor ($p = 0.00226$) that captures individual living leverage independent of absolute income level.

---

## 7. Predictive Modeling Blueprint & Actionable Recommendations

### Algorithm Selection Strategy
Given the continuous regression target (`Credit_Score`), 50,000 instances, and non-linear interactions, the following model hierarchy is recommended:

1. **Primary Model - Gradient Boosting (XGBoost / LightGBM):**
   * *Rationale:* Best handles non-linear relationships, scale-invariant to remaining skewed spend features, and naturally robust to linear collinearity.
2. **Secondary Model - Regularized Ridge / Lasso Regression:**
   * *Rationale:* Excellent baseline for regulatory interpretability. L1/L2 penalties effectively shrink collinear coefficients (e.g., `Outstanding_Balance` vs. `Statement_Balance`).
3. **Benchmark - Random Forest Regressor:**
   * *Rationale:* Strong non-parametric performance for capturing non-linear interactions between `Payment_Ratio` and `Credit_Utilization`.

### Model Pipeline Blueprint

```
[Raw Dataset: 50k x 30]
       |
       v
[Data Preprocessing]
  |-- Drop: Customer_ID (Key metric, high cardinality)
  |-- Drop: Demographics with p > 0.05 (Age, Gender)
  |-- Categorical Encoding: Target Encoding / One-Hot for Card_Type & Occupation
  +-- Numeric Normalization: RobustScaler on skewed spend columns
       |
       v
[Feature Selection & De-duplication]
  |-- Remove Variance Inflation: Drop Statement_Balance (keep Outstanding_Balance)
  |-- Remove Reward_Points_Earned (keep Monthly_Spending)
  +-- Retain Engineered Features (Log_Annual_Income, Spending_To_Income_Ratio)
       |
       v
[Validation & Evaluation]
  |-- Strategy: 5-Fold Stratified Cross-Validation (binned by Credit_Score deciles)
  |-- Primary Metrics: Root Mean Squared Error (RMSE), Mean Absolute Error (MAE)
  +-- Secondary Metric: R-Squared (R2 > 0.85 target threshold)
```

### Strategic Business Recommendations

1. **Automated Credit Line Adjustments:**
   * Automatically decrease credit limits for users whose `Credit_Utilization` breaches $70\%$ and whose `Payment_Ratio` falls below $0.30$. This preemptively limits default risk before scores drop into subprime territory.
2. **Targeted Financial Health Interventions:**
   * Prompt customers maintaining low `Payment_Ratio` levels ($< 0.50$) via mobile app alerts (`Mobile_App_Login`), offering structured EMI conversions. Increased EMI structured plans stabilize payment ratios and reduce balance risk.
3. **Streamlined Underwriting Inputs:**
   * Streamline initial credit application forms by removing non-predictive demographic questions (e.g., age, gender). Focus risk assessment strictly on verified income, existing debt balances, and payment history ratios.
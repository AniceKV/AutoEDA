# Executive EDA & Dataset Summary Report
**Target Directory:** `C:\Users\Anish Kumar Verma\PycharmProjects\AutoEDA\sandbox_run\9578e6f7-0726-45fa-8b92-b6db747a1311`
**Processed Files:** `agent_plan_log.json`, `agent_state.json`, `current_df.csv`, `metadata_profile.json`, `metrics.json`
**Excluded Files:** `generated_analysis.py` (Script excluded from summary)

---

## 1. Dataset Overview
- **Dataset Identifier:** `bdi_and_screen_items.csv`
- **Dimensions:** `4810` rows x `31` columns
- **Target Variable:** `Not Specified`
- **Data Quality:** No missing values detected in raw profile.

---

## 1.5 Full Column Statistics
| Column | Type | Missing % | Unique % | Mean | Median | Std | Skew | Kurtosis |
|---|---|---|---|---|---|---|---|---|
| `subject_id` | `int64` | 0.0% | 100.0% | 2405.5 | 2405.5 | 1388.67 | 0.0 | -1.2 |
| `screen_normal_day_1to6` | `int64` | 0.0% | 0.12% | 2.96 | 3.0 | 1.08 | 0.56 | 0.42 |
| `screen_weekday_1to6` | `int64` | 0.0% | 0.12% | 2.98 | 3.0 | 1.18 | 0.9 | 0.38 |
| `screen_weekend_1to6` | `int64` | 0.0% | 0.12% | 3.71 | 4.0 | 1.43 | 0.14 | -0.93 |
| `sqi_fall_asleep_1to6` | `int64` | 0.0% | 0.12% | 2.37 | 2.0 | 1.35 | 1.08 | 0.42 |
| `sqi_repeated_awake_1to6` | `int64` | 0.0% | 0.12% | 1.73 | 1.0 | 1.06 | 1.81 | 3.34 |
| `sqi_disturbed_1to6` | `int64` | 0.0% | 0.12% | 1.9 | 2.0 | 1.09 | 1.48 | 2.23 |
| `sqi_early_awake_1to6` | `int64` | 0.0% | 0.12% | 1.61 | 1.0 | 1.01 | 2.11 | 4.73 |
| `bdi_item_01` | `int64` | 0.0% | 0.08% | 0.28 | 0.0 | 0.5 | 1.73 | 2.9 |
| `bdi_item_02` | `int64` | 0.0% | 0.08% | 0.43 | 0.0 | 0.63 | 1.39 | 1.62 |
| `bdi_item_03` | `int64` | 0.0% | 0.08% | 0.36 | 0.0 | 0.64 | 1.77 | 2.73 |
| `bdi_item_04` | `int64` | 0.0% | 0.08% | 0.25 | 0.0 | 0.53 | 2.18 | 4.6 |
| `bdi_item_05` | `int64` | 0.0% | 0.08% | 0.45 | 0.0 | 0.72 | 1.62 | 2.1 |
| `bdi_item_06` | `int64` | 0.0% | 0.08% | 0.17 | 0.0 | 0.48 | 3.29 | 11.82 |
| `bdi_item_07` | `int64` | 0.0% | 0.08% | 0.32 | 0.0 | 0.66 | 2.33 | 5.35 |
| `bdi_item_08` | `int64` | 0.0% | 0.08% | 0.37 | 0.0 | 0.67 | 1.83 | 2.72 |
| `bdi_item_09` | `int64` | 0.0% | 0.08% | 0.14 | 0.0 | 0.39 | 2.95 | 9.98 |
| `bdi_item_10` | `int64` | 0.0% | 0.08% | 0.32 | 0.0 | 0.73 | 2.51 | 5.65 |
| `bdi_item_11` | `int64` | 0.0% | 0.08% | 0.31 | 0.0 | 0.61 | 2.17 | 4.85 |
| `bdi_item_12` | `int64` | 0.0% | 0.08% | 0.27 | 0.0 | 0.55 | 2.24 | 5.51 |
| `bdi_item_13` | `int64` | 0.0% | 0.08% | 0.33 | 0.0 | 0.69 | 2.38 | 5.59 |
| `bdi_item_14` | `int64` | 0.0% | 0.08% | 0.22 | 0.0 | 0.56 | 2.68 | 6.82 |
| `bdi_item_15` | `int64` | 0.0% | 0.08% | 0.42 | 0.0 | 0.61 | 1.25 | 0.89 |
| `bdi_item_16` | `int64` | 0.0% | 0.08% | 0.71 | 1.0 | 0.66 | 0.72 | 0.71 |
| `bdi_item_17` | `int64` | 0.0% | 0.08% | 0.44 | 0.0 | 0.67 | 1.56 | 2.32 |
| `bdi_item_18` | `int64` | 0.0% | 0.08% | 0.47 | 0.0 | 0.7 | 1.6 | 2.54 |
| `bdi_item_19` | `int64` | 0.0% | 0.08% | 0.43 | 0.0 | 0.72 | 1.58 | 1.65 |
| `bdi_item_20` | `int64` | 0.0% | 0.08% | 0.47 | 0.0 | 0.64 | 1.36 | 2.02 |
| `bdi_item_21` | `int64` | 0.0% | 0.08% | 0.13 | 0.0 | 0.5 | 4.24 | 18.42 |

---

## 2. Data Imputation & Preprocessing
- **status:** Imputation completed

---

## 3. Outlier Analysis (IQR Method)
No numeric outlier statistics reported.

---

## 4. Derived Domain Attributes & Composite Metrics
- **`bdi_total_score`**: Formula: `bdi_item_01 + bdi_item_02 + bdi_item_03 + bdi_item_04 + bdi_item_05 + bdi_item_06 + bdi_item_07 + bdi_item_08 + bdi_item_09 + bdi_item_10 + bdi_item_11 + bdi_item_12 + bdi_item_13 + bdi_item_14 + bdi_item_15 + bdi_item_16 + bdi_item_17 + bdi_item_18 + bdi_item_19 + bdi_item_20 + bdi_item_21` | Purpose: Sum of all BDI items to create a global depression severity index for EDA.
- **`sqi_total_disturb`**: Formula: `sqi_fall_asleep_1to6 + sqi_repeated_awake_1to6 + sqi_disturbed_1to6 + sqi_early_awake_1to6` | Purpose: Composite score of sleep quality issues.

---

## 5. Statistical Hypothesis Testing & Key Predictors
No statistically significant predictors identified.

---

## 6. Redundancy & Multicollinearity Analysis
No high-correlation or cross-type redundant feature pairs detected (threshold: |r| or Eta >= 0.85).

---

## 7. Generated Visual Artifacts
No PNG/SVG image assets found in directory.

---

## 8. Categorical Associations (Cramer's V)
| Feature 1 | Feature 2 | Cramer's V |
|---|---|---|
| `screen_normal_day_1to6` | `screen_weekday_1to6` | 0.2904 |
| `screen_normal_day_1to6` | `screen_weekend_1to6` | 0.253 |
| `screen_normal_day_1to6` | `sqi_fall_asleep_1to6` | 0.0498 |
| `screen_normal_day_1to6` | `sqi_repeated_awake_1to6` | 0.0382 |
| `screen_normal_day_1to6` | `sqi_disturbed_1to6` | 0.0158 |
| `screen_normal_day_1to6` | `sqi_early_awake_1to6` | 0.0457 |
| `screen_normal_day_1to6` | `bdi_item_01` | 0.0585 |
| `screen_normal_day_1to6` | `bdi_item_02` | 0.0549 |
| `screen_normal_day_1to6` | `bdi_item_03` | 0.0796 |
| `screen_normal_day_1to6` | `bdi_item_04` | 0.0485 |
| `screen_normal_day_1to6` | `bdi_item_05` | 0.0566 |
| `screen_normal_day_1to6` | `bdi_item_06` | 0.0275 |
| `screen_normal_day_1to6` | `bdi_item_07` | 0.0568 |
| `screen_normal_day_1to6` | `bdi_item_08` | 0.0613 |
| `screen_normal_day_1to6` | `bdi_item_09` | 0.0485 |
| `screen_normal_day_1to6` | `bdi_item_10` | 0.0474 |
| `screen_normal_day_1to6` | `bdi_item_11` | 0.0395 |
| `screen_normal_day_1to6` | `bdi_item_12` | 0.0516 |
| `screen_normal_day_1to6` | `bdi_item_13` | 0.0564 |
| `screen_normal_day_1to6` | `bdi_item_14` | 0.0585 |
| `screen_normal_day_1to6` | `bdi_item_15` | 0.0575 |
| `screen_normal_day_1to6` | `bdi_item_16` | 0.048 |
| `screen_normal_day_1to6` | `bdi_item_17` | 0.0558 |
| `screen_normal_day_1to6` | `bdi_item_18` | 0.0484 |
| `screen_normal_day_1to6` | `bdi_item_19` | 0.0563 |
| `screen_normal_day_1to6` | `bdi_item_20` | 0.0436 |
| `screen_normal_day_1to6` | `bdi_item_21` | 0.0156 |
| `screen_weekday_1to6` | `screen_weekend_1to6` | 0.5144 |
| `screen_weekday_1to6` | `sqi_fall_asleep_1to6` | 0.0324 |
| `screen_weekday_1to6` | `sqi_repeated_awake_1to6` | 0.0287 |
| `screen_weekday_1to6` | `sqi_disturbed_1to6` | 0.0318 |
| `screen_weekday_1to6` | `sqi_early_awake_1to6` | 0.0158 |
| `screen_weekday_1to6` | `bdi_item_01` | 0.0454 |
| `screen_weekday_1to6` | `bdi_item_02` | 0.0449 |
| `screen_weekday_1to6` | `bdi_item_03` | 0.0553 |
| `screen_weekday_1to6` | `bdi_item_04` | 0.0322 |
| `screen_weekday_1to6` | `bdi_item_05` | 0.0493 |
| `screen_weekday_1to6` | `bdi_item_06` | 0.0263 |
| `screen_weekday_1to6` | `bdi_item_07` | 0.0411 |
| `screen_weekday_1to6` | `bdi_item_08` | 0.0312 |
| `screen_weekday_1to6` | `bdi_item_09` | 0.0295 |
| `screen_weekday_1to6` | `bdi_item_10` | 0.0203 |
| `screen_weekday_1to6` | `bdi_item_11` | 0.0302 |
| `screen_weekday_1to6` | `bdi_item_12` | 0.0434 |
| `screen_weekday_1to6` | `bdi_item_13` | 0.0364 |
| `screen_weekday_1to6` | `bdi_item_14` | 0.0363 |
| `screen_weekday_1to6` | `bdi_item_15` | 0.0426 |
| `screen_weekday_1to6` | `bdi_item_16` | 0.036 |
| `screen_weekday_1to6` | `bdi_item_17` | 0.0439 |
| `screen_weekday_1to6` | `bdi_item_18` | 0.0472 |
| `screen_weekday_1to6` | `bdi_item_19` | 0.0557 |
| `screen_weekday_1to6` | `bdi_item_20` | 0.0326 |
| `screen_weekday_1to6` | `bdi_item_21` | 0.0286 |
| `screen_weekend_1to6` | `sqi_fall_asleep_1to6` | 0.0506 |
| `screen_weekend_1to6` | `sqi_repeated_awake_1to6` | 0.0343 |
| `screen_weekend_1to6` | `sqi_disturbed_1to6` | 0.0 |
| `screen_weekend_1to6` | `sqi_early_awake_1to6` | 0.034 |
| `screen_weekend_1to6` | `bdi_item_01` | 0.0385 |
| `screen_weekend_1to6` | `bdi_item_02` | 0.0404 |
| `screen_weekend_1to6` | `bdi_item_03` | 0.0502 |
| `screen_weekend_1to6` | `bdi_item_04` | 0.0287 |
| `screen_weekend_1to6` | `bdi_item_05` | 0.0334 |
| `screen_weekend_1to6` | `bdi_item_06` | 0.0 |
| `screen_weekend_1to6` | `bdi_item_07` | 0.0404 |
| `screen_weekend_1to6` | `bdi_item_08` | 0.0299 |
| `screen_weekend_1to6` | `bdi_item_09` | 0.0369 |
| `screen_weekend_1to6` | `bdi_item_10` | 0.0257 |
| `screen_weekend_1to6` | `bdi_item_11` | 0.0413 |
| `screen_weekend_1to6` | `bdi_item_12` | 0.0311 |
| `screen_weekend_1to6` | `bdi_item_13` | 0.0156 |
| `screen_weekend_1to6` | `bdi_item_14` | 0.0318 |
| `screen_weekend_1to6` | `bdi_item_15` | 0.0417 |
| `screen_weekend_1to6` | `bdi_item_16` | 0.0283 |
| `screen_weekend_1to6` | `bdi_item_17` | 0.0424 |
| `screen_weekend_1to6` | `bdi_item_18` | 0.0294 |
| `screen_weekend_1to6` | `bdi_item_19` | 0.05 |
| `screen_weekend_1to6` | `bdi_item_20` | 0.0376 |
| `screen_weekend_1to6` | `bdi_item_21` | 0.0368 |
| `sqi_fall_asleep_1to6` | `sqi_repeated_awake_1to6` | 0.2422 |
| `sqi_fall_asleep_1to6` | `sqi_disturbed_1to6` | 0.1379 |
| `sqi_fall_asleep_1to6` | `sqi_early_awake_1to6` | 0.2253 |
| `sqi_fall_asleep_1to6` | `bdi_item_01` | 0.1528 |
| `sqi_fall_asleep_1to6` | `bdi_item_02` | 0.1176 |
| `sqi_fall_asleep_1to6` | `bdi_item_03` | 0.1709 |
| `sqi_fall_asleep_1to6` | `bdi_item_04` | 0.1266 |
| `sqi_fall_asleep_1to6` | `bdi_item_05` | 0.1176 |
| `sqi_fall_asleep_1to6` | `bdi_item_06` | 0.0904 |
| `sqi_fall_asleep_1to6` | `bdi_item_07` | 0.1525 |
| `sqi_fall_asleep_1to6` | `bdi_item_08` | 0.1205 |
| `sqi_fall_asleep_1to6` | `bdi_item_09` | 0.1077 |
| `sqi_fall_asleep_1to6` | `bdi_item_10` | 0.1183 |
| `sqi_fall_asleep_1to6` | `bdi_item_11` | 0.1299 |
| `sqi_fall_asleep_1to6` | `bdi_item_12` | 0.131 |
| `sqi_fall_asleep_1to6` | `bdi_item_13` | 0.1406 |
| `sqi_fall_asleep_1to6` | `bdi_item_14` | 0.143 |
| `sqi_fall_asleep_1to6` | `bdi_item_15` | 0.1721 |
| `sqi_fall_asleep_1to6` | `bdi_item_16` | 0.1627 |
| `sqi_fall_asleep_1to6` | `bdi_item_17` | 0.1644 |
| `sqi_fall_asleep_1to6` | `bdi_item_18` | 0.1156 |
| `sqi_fall_asleep_1to6` | `bdi_item_19` | 0.161 |
| `sqi_fall_asleep_1to6` | `bdi_item_20` | 0.1734 |
| `sqi_fall_asleep_1to6` | `bdi_item_21` | 0.0492 |
| `sqi_repeated_awake_1to6` | `sqi_disturbed_1to6` | 0.253 |
| `sqi_repeated_awake_1to6` | `sqi_early_awake_1to6` | 0.3129 |
| `sqi_repeated_awake_1to6` | `bdi_item_01` | 0.1002 |
| `sqi_repeated_awake_1to6` | `bdi_item_02` | 0.1184 |
| `sqi_repeated_awake_1to6` | `bdi_item_03` | 0.1223 |
| `sqi_repeated_awake_1to6` | `bdi_item_04` | 0.0984 |
| `sqi_repeated_awake_1to6` | `bdi_item_05` | 0.1179 |
| `sqi_repeated_awake_1to6` | `bdi_item_06` | 0.08 |
| `sqi_repeated_awake_1to6` | `bdi_item_07` | 0.096 |
| `sqi_repeated_awake_1to6` | `bdi_item_08` | 0.0812 |
| `sqi_repeated_awake_1to6` | `bdi_item_09` | 0.0835 |
| `sqi_repeated_awake_1to6` | `bdi_item_10` | 0.0924 |
| `sqi_repeated_awake_1to6` | `bdi_item_11` | 0.0884 |
| `sqi_repeated_awake_1to6` | `bdi_item_12` | 0.0925 |
| `sqi_repeated_awake_1to6` | `bdi_item_13` | 0.1145 |
| `sqi_repeated_awake_1to6` | `bdi_item_14` | 0.108 |
| `sqi_repeated_awake_1to6` | `bdi_item_15` | 0.1088 |
| `sqi_repeated_awake_1to6` | `bdi_item_16` | 0.1435 |
| `sqi_repeated_awake_1to6` | `bdi_item_17` | 0.1183 |
| `sqi_repeated_awake_1to6` | `bdi_item_18` | 0.0971 |
| `sqi_repeated_awake_1to6` | `bdi_item_19` | 0.1272 |
| `sqi_repeated_awake_1to6` | `bdi_item_20` | 0.134 |
| `sqi_repeated_awake_1to6` | `bdi_item_21` | 0.0498 |
| `sqi_disturbed_1to6` | `sqi_early_awake_1to6` | 0.2149 |
| `sqi_disturbed_1to6` | `bdi_item_01` | 0.0765 |
| `sqi_disturbed_1to6` | `bdi_item_02` | 0.0561 |
| `sqi_disturbed_1to6` | `bdi_item_03` | 0.0722 |
| `sqi_disturbed_1to6` | `bdi_item_04` | 0.0446 |
| `sqi_disturbed_1to6` | `bdi_item_05` | 0.0735 |
| `sqi_disturbed_1to6` | `bdi_item_06` | 0.0532 |
| `sqi_disturbed_1to6` | `bdi_item_07` | 0.0494 |
| `sqi_disturbed_1to6` | `bdi_item_08` | 0.0439 |
| `sqi_disturbed_1to6` | `bdi_item_09` | 0.0672 |
| `sqi_disturbed_1to6` | `bdi_item_10` | 0.0628 |
| `sqi_disturbed_1to6` | `bdi_item_11` | 0.068 |
| `sqi_disturbed_1to6` | `bdi_item_12` | 0.0733 |
| `sqi_disturbed_1to6` | `bdi_item_13` | 0.0803 |
| `sqi_disturbed_1to6` | `bdi_item_14` | 0.0619 |
| `sqi_disturbed_1to6` | `bdi_item_15` | 0.0558 |
| `sqi_disturbed_1to6` | `bdi_item_16` | 0.0903 |
| `sqi_disturbed_1to6` | `bdi_item_17` | 0.0706 |
| `sqi_disturbed_1to6` | `bdi_item_18` | 0.0676 |
| `sqi_disturbed_1to6` | `bdi_item_19` | 0.0675 |
| `sqi_disturbed_1to6` | `bdi_item_20` | 0.0586 |
| `sqi_disturbed_1to6` | `bdi_item_21` | 0.0448 |
| `sqi_early_awake_1to6` | `bdi_item_01` | 0.1393 |
| `sqi_early_awake_1to6` | `bdi_item_02` | 0.1089 |
| `sqi_early_awake_1to6` | `bdi_item_03` | 0.1282 |
| `sqi_early_awake_1to6` | `bdi_item_04` | 0.1019 |
| `sqi_early_awake_1to6` | `bdi_item_05` | 0.1455 |
| `sqi_early_awake_1to6` | `bdi_item_06` | 0.0862 |
| `sqi_early_awake_1to6` | `bdi_item_07` | 0.1228 |
| `sqi_early_awake_1to6` | `bdi_item_08` | 0.1012 |
| `sqi_early_awake_1to6` | `bdi_item_09` | 0.1032 |
| `sqi_early_awake_1to6` | `bdi_item_10` | 0.124 |
| `sqi_early_awake_1to6` | `bdi_item_11` | 0.1131 |
| `sqi_early_awake_1to6` | `bdi_item_12` | 0.1008 |
| `sqi_early_awake_1to6` | `bdi_item_13` | 0.1461 |
| `sqi_early_awake_1to6` | `bdi_item_14` | 0.1276 |
| `sqi_early_awake_1to6` | `bdi_item_15` | 0.1169 |
| `sqi_early_awake_1to6` | `bdi_item_16` | 0.1171 |
| `sqi_early_awake_1to6` | `bdi_item_17` | 0.1354 |
| `sqi_early_awake_1to6` | `bdi_item_18` | 0.1051 |
| `sqi_early_awake_1to6` | `bdi_item_19` | 0.1465 |
| `sqi_early_awake_1to6` | `bdi_item_20` | 0.1246 |
| `sqi_early_awake_1to6` | `bdi_item_21` | 0.0435 |
| `bdi_item_01` | `bdi_item_02` | 0.2321 |
| `bdi_item_01` | `bdi_item_03` | 0.3435 |
| `bdi_item_01` | `bdi_item_04` | 0.3306 |
| `bdi_item_01` | `bdi_item_05` | 0.2329 |
| `bdi_item_01` | `bdi_item_06` | 0.1601 |
| `bdi_item_01` | `bdi_item_07` | 0.325 |
| `bdi_item_01` | `bdi_item_08` | 0.2781 |
| `bdi_item_01` | `bdi_item_09` | 0.2501 |
| `bdi_item_01` | `bdi_item_10` | 0.306 |
| `bdi_item_01` | `bdi_item_11` | 0.2058 |
| `bdi_item_01` | `bdi_item_12` | 0.2979 |
| `bdi_item_01` | `bdi_item_13` | 0.2402 |
| `bdi_item_01` | `bdi_item_14` | 0.3449 |
| `bdi_item_01` | `bdi_item_15` | 0.2956 |
| `bdi_item_01` | `bdi_item_16` | 0.1526 |
| `bdi_item_01` | `bdi_item_17` | 0.2347 |
| `bdi_item_01` | `bdi_item_18` | 0.2052 |
| `bdi_item_01` | `bdi_item_19` | 0.2122 |
| `bdi_item_01` | `bdi_item_20` | 0.252 |
| `bdi_item_01` | `bdi_item_21` | 0.0852 |
| `bdi_item_02` | `bdi_item_03` | 0.2875 |
| `bdi_item_02` | `bdi_item_04` | 0.2169 |
| `bdi_item_02` | `bdi_item_05` | 0.198 |
| `bdi_item_02` | `bdi_item_06` | 0.1447 |
| `bdi_item_02` | `bdi_item_07` | 0.2318 |
| `bdi_item_02` | `bdi_item_08` | 0.2289 |
| `bdi_item_02` | `bdi_item_09` | 0.2093 |
| `bdi_item_02` | `bdi_item_10` | 0.1659 |
| `bdi_item_02` | `bdi_item_11` | 0.1524 |
| `bdi_item_02` | `bdi_item_12` | 0.2134 |
| `bdi_item_02` | `bdi_item_13` | 0.1971 |
| `bdi_item_02` | `bdi_item_14` | 0.2816 |
| `bdi_item_02` | `bdi_item_15` | 0.2064 |
| `bdi_item_02` | `bdi_item_16` | 0.1392 |
| `bdi_item_02` | `bdi_item_17` | 0.1886 |
| `bdi_item_02` | `bdi_item_18` | 0.1495 |
| `bdi_item_02` | `bdi_item_19` | 0.2076 |
| `bdi_item_02` | `bdi_item_20` | 0.1993 |
| `bdi_item_02` | `bdi_item_21` | 0.0927 |
| `bdi_item_03` | `bdi_item_04` | 0.2776 |
| `bdi_item_03` | `bdi_item_05` | 0.2431 |
| `bdi_item_03` | `bdi_item_06` | 0.1848 |
| `bdi_item_03` | `bdi_item_07` | 0.3751 |
| `bdi_item_03` | `bdi_item_08` | 0.338 |
| `bdi_item_03` | `bdi_item_09` | 0.2701 |
| `bdi_item_03` | `bdi_item_10` | 0.2552 |
| `bdi_item_03` | `bdi_item_11` | 0.1848 |
| `bdi_item_03` | `bdi_item_12` | 0.2771 |
| `bdi_item_03` | `bdi_item_13` | 0.2193 |
| `bdi_item_03` | `bdi_item_14` | 0.4553 |
| `bdi_item_03` | `bdi_item_15` | 0.2725 |
| `bdi_item_03` | `bdi_item_16` | 0.1595 |
| `bdi_item_03` | `bdi_item_17` | 0.2149 |
| `bdi_item_03` | `bdi_item_18` | 0.1896 |
| `bdi_item_03` | `bdi_item_19` | 0.2498 |
| `bdi_item_03` | `bdi_item_20` | 0.2471 |
| `bdi_item_03` | `bdi_item_21` | 0.1123 |
| `bdi_item_04` | `bdi_item_05` | 0.2045 |
| `bdi_item_04` | `bdi_item_06` | 0.1451 |
| `bdi_item_04` | `bdi_item_07` | 0.2718 |
| `bdi_item_04` | `bdi_item_08` | 0.2465 |
| `bdi_item_04` | `bdi_item_09` | 0.2455 |
| `bdi_item_04` | `bdi_item_10` | 0.2293 |
| `bdi_item_04` | `bdi_item_11` | 0.1767 |
| `bdi_item_04` | `bdi_item_12` | 0.3689 |
| `bdi_item_04` | `bdi_item_13` | 0.2112 |
| `bdi_item_04` | `bdi_item_14` | 0.2842 |
| `bdi_item_04` | `bdi_item_15` | 0.2968 |
| `bdi_item_04` | `bdi_item_16` | 0.1683 |
| `bdi_item_04` | `bdi_item_17` | 0.2228 |
| `bdi_item_04` | `bdi_item_18` | 0.1884 |
| `bdi_item_04` | `bdi_item_19` | 0.181 |
| `bdi_item_04` | `bdi_item_20` | 0.2465 |
| `bdi_item_04` | `bdi_item_21` | 0.0883 |
| `bdi_item_05` | `bdi_item_06` | 0.1769 |
| `bdi_item_05` | `bdi_item_07` | 0.2304 |
| `bdi_item_05` | `bdi_item_08` | 0.2487 |
| `bdi_item_05` | `bdi_item_09` | 0.1909 |
| `bdi_item_05` | `bdi_item_10` | 0.1796 |
| `bdi_item_05` | `bdi_item_11` | 0.1746 |
| `bdi_item_05` | `bdi_item_12` | 0.1672 |
| `bdi_item_05` | `bdi_item_13` | 0.2415 |
| `bdi_item_05` | `bdi_item_14` | 0.2369 |
| `bdi_item_05` | `bdi_item_15` | 0.1968 |
| `bdi_item_05` | `bdi_item_16` | 0.135 |
| `bdi_item_05` | `bdi_item_17` | 0.1939 |
| `bdi_item_05` | `bdi_item_18` | 0.1571 |
| `bdi_item_05` | `bdi_item_19` | 0.1921 |
| `bdi_item_05` | `bdi_item_20` | 0.1915 |
| `bdi_item_05` | `bdi_item_21` | 0.0733 |
| `bdi_item_06` | `bdi_item_07` | 0.1628 |
| `bdi_item_06` | `bdi_item_08` | 0.1564 |
| `bdi_item_06` | `bdi_item_09` | 0.139 |
| `bdi_item_06` | `bdi_item_10` | 0.1205 |
| `bdi_item_06` | `bdi_item_11` | 0.1329 |
| `bdi_item_06` | `bdi_item_12` | 0.1489 |
| `bdi_item_06` | `bdi_item_13` | 0.1497 |
| `bdi_item_06` | `bdi_item_14` | 0.1691 |
| `bdi_item_06` | `bdi_item_15` | 0.1456 |
| `bdi_item_06` | `bdi_item_16` | 0.1007 |
| `bdi_item_06` | `bdi_item_17` | 0.1484 |
| `bdi_item_06` | `bdi_item_18` | 0.1209 |
| `bdi_item_06` | `bdi_item_19` | 0.141 |
| `bdi_item_06` | `bdi_item_20` | 0.1348 |
| `bdi_item_06` | `bdi_item_21` | 0.0652 |
| `bdi_item_07` | `bdi_item_08` | 0.3854 |
| `bdi_item_07` | `bdi_item_09` | 0.2186 |
| `bdi_item_07` | `bdi_item_10` | 0.2792 |
| `bdi_item_07` | `bdi_item_11` | 0.1818 |
| `bdi_item_07` | `bdi_item_12` | 0.2433 |
| `bdi_item_07` | `bdi_item_13` | 0.262 |
| `bdi_item_07` | `bdi_item_14` | 0.4312 |
| `bdi_item_07` | `bdi_item_15` | 0.2694 |
| `bdi_item_07` | `bdi_item_16` | 0.1496 |
| `bdi_item_07` | `bdi_item_17` | 0.2302 |
| `bdi_item_07` | `bdi_item_18` | 0.1897 |
| `bdi_item_07` | `bdi_item_19` | 0.2272 |
| `bdi_item_07` | `bdi_item_20` | 0.2307 |
| `bdi_item_07` | `bdi_item_21` | 0.0797 |
| `bdi_item_08` | `bdi_item_09` | 0.2248 |
| `bdi_item_08` | `bdi_item_10` | 0.2473 |
| `bdi_item_08` | `bdi_item_11` | 0.1786 |
| `bdi_item_08` | `bdi_item_12` | 0.2295 |
| `bdi_item_08` | `bdi_item_13` | 0.245 |
| `bdi_item_08` | `bdi_item_14` | 0.3834 |
| `bdi_item_08` | `bdi_item_15` | 0.2187 |
| `bdi_item_08` | `bdi_item_16` | 0.1533 |
| `bdi_item_08` | `bdi_item_17` | 0.2219 |
| `bdi_item_08` | `bdi_item_18` | 0.1851 |
| `bdi_item_08` | `bdi_item_19` | 0.227 |
| `bdi_item_08` | `bdi_item_20` | 0.2143 |
| `bdi_item_08` | `bdi_item_21` | 0.0914 |
| `bdi_item_09` | `bdi_item_10` | 0.2011 |
| `bdi_item_09` | `bdi_item_11` | 0.1435 |
| `bdi_item_09` | `bdi_item_12` | 0.233 |
| `bdi_item_09` | `bdi_item_13` | 0.1594 |
| `bdi_item_09` | `bdi_item_14` | 0.3011 |
| `bdi_item_09` | `bdi_item_15` | 0.1832 |
| `bdi_item_09` | `bdi_item_16` | 0.1345 |
| `bdi_item_09` | `bdi_item_17` | 0.1353 |
| `bdi_item_09` | `bdi_item_18` | 0.1427 |
| `bdi_item_09` | `bdi_item_19` | 0.1519 |
| `bdi_item_09` | `bdi_item_20` | 0.1665 |
| `bdi_item_09` | `bdi_item_21` | 0.0997 |
| `bdi_item_10` | `bdi_item_11` | 0.158 |
| `bdi_item_10` | `bdi_item_12` | 0.2123 |
| `bdi_item_10` | `bdi_item_13` | 0.2037 |
| `bdi_item_10` | `bdi_item_14` | 0.2844 |
| `bdi_item_10` | `bdi_item_15` | 0.2075 |
| `bdi_item_10` | `bdi_item_16` | 0.141 |
| `bdi_item_10` | `bdi_item_17` | 0.236 |
| `bdi_item_10` | `bdi_item_18` | 0.196 |
| `bdi_item_10` | `bdi_item_19` | 0.203 |
| `bdi_item_10` | `bdi_item_20` | 0.2078 |
| `bdi_item_10` | `bdi_item_21` | 0.0713 |
| `bdi_item_11` | `bdi_item_12` | 0.1694 |
| `bdi_item_11` | `bdi_item_13` | 0.1946 |
| `bdi_item_11` | `bdi_item_14` | 0.1723 |
| `bdi_item_11` | `bdi_item_15` | 0.1736 |
| `bdi_item_11` | `bdi_item_16` | 0.1537 |
| `bdi_item_11` | `bdi_item_17` | 0.2434 |
| `bdi_item_11` | `bdi_item_18` | 0.1595 |
| `bdi_item_11` | `bdi_item_19` | 0.3083 |
| `bdi_item_11` | `bdi_item_20` | 0.222 |
| `bdi_item_11` | `bdi_item_21` | 0.0479 |
| `bdi_item_12` | `bdi_item_13` | 0.2177 |
| `bdi_item_12` | `bdi_item_14` | 0.2798 |
| `bdi_item_12` | `bdi_item_15` | 0.2794 |
| `bdi_item_12` | `bdi_item_16` | 0.1628 |
| `bdi_item_12` | `bdi_item_17` | 0.2202 |
| `bdi_item_12` | `bdi_item_18` | 0.1702 |
| `bdi_item_12` | `bdi_item_19` | 0.2163 |
| `bdi_item_12` | `bdi_item_20` | 0.2578 |
| `bdi_item_12` | `bdi_item_21` | 0.1085 |
| `bdi_item_13` | `bdi_item_14` | 0.2671 |
| `bdi_item_13` | `bdi_item_15` | 0.214 |
| `bdi_item_13` | `bdi_item_16` | 0.1514 |
| `bdi_item_13` | `bdi_item_17` | 0.2328 |
| `bdi_item_13` | `bdi_item_18` | 0.1482 |
| `bdi_item_13` | `bdi_item_19` | 0.2374 |
| `bdi_item_13` | `bdi_item_20` | 0.2128 |
| `bdi_item_13` | `bdi_item_21` | 0.0764 |
| `bdi_item_14` | `bdi_item_15` | 0.2517 |
| `bdi_item_14` | `bdi_item_16` | 0.1464 |
| `bdi_item_14` | `bdi_item_17` | 0.2214 |
| `bdi_item_14` | `bdi_item_18` | 0.1888 |
| `bdi_item_14` | `bdi_item_19` | 0.2381 |
| `bdi_item_14` | `bdi_item_20` | 0.2189 |
| `bdi_item_14` | `bdi_item_21` | 0.0754 |
| `bdi_item_15` | `bdi_item_16` | 0.1936 |
| `bdi_item_15` | `bdi_item_17` | 0.2409 |
| `bdi_item_15` | `bdi_item_18` | 0.166 |
| `bdi_item_15` | `bdi_item_19` | 0.229 |
| `bdi_item_15` | `bdi_item_20` | 0.4091 |
| `bdi_item_15` | `bdi_item_21` | 0.0889 |
| `bdi_item_16` | `bdi_item_17` | 0.179 |
| `bdi_item_16` | `bdi_item_18` | 0.1907 |
| `bdi_item_16` | `bdi_item_19` | 0.1685 |
| `bdi_item_16` | `bdi_item_20` | 0.2511 |
| `bdi_item_16` | `bdi_item_21` | 0.0639 |
| `bdi_item_17` | `bdi_item_18` | 0.1921 |
| `bdi_item_17` | `bdi_item_19` | 0.2641 |
| `bdi_item_17` | `bdi_item_20` | 0.2415 |
| `bdi_item_17` | `bdi_item_21` | 0.0549 |
| `bdi_item_18` | `bdi_item_19` | 0.1574 |
| `bdi_item_18` | `bdi_item_20` | 0.1705 |
| `bdi_item_18` | `bdi_item_21` | 0.0871 |
| `bdi_item_19` | `bdi_item_20` | 0.2633 |
| `bdi_item_19` | `bdi_item_21` | 0.0734 |
| `bdi_item_20` | `bdi_item_21` | 0.1084 |

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
- **Executive Summary:** Target: 'Undefined (Unsupervised)' (Unsupervised / Exploratory). Model recommendations and validation strategy tailored for 4810 rows x 29 columns.

---

*Report generated automatically by `summary_generator.py`*
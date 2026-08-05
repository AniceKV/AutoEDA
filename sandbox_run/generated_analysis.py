DATA_FILEPATH = r'C:\Users\Anish Kumar Verma\PycharmProjects\AutoEDA\test_data\dataset_2191_sleep.csv'

import json
import os
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")

if "DATA_FILEPATH" not in globals():
    raise NameError("DATA_FILEPATH must be defined before running this script.")

PROFILE_FILEPATH = "metadata_profile.json"
METRICS_FILEPATH = "metrics.json"
TARGET_COLUMN = "total_sleep"
ALPHA = 0.05

if not os.path.exists(DATA_FILEPATH):
    raise FileNotFoundError(f"Dataset not found: {DATA_FILEPATH}")

if not os.path.exists(PROFILE_FILEPATH):
    raise FileNotFoundError(f"Metadata profile not found: {PROFILE_FILEPATH}")

with open(PROFILE_FILEPATH, "r", encoding="utf-8") as profile_file:
    profile = json.load(profile_file)

df_raw = pd.read_csv(DATA_FILEPATH)
df = df_raw.copy()

missing_tokens = ["?", "", "NA", "N/A", "null", "None", "nan"]
profile_schema = profile.get("schema", {})
profile_missing_summary = profile.get("missing_values_summary", {})


def json_safe(value):
    if isinstance(value, dict):
        return {str(key): json_safe(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, float):
        return None if not np.isfinite(value) else value
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value


def force_numeric_series(series):
    """
    Convert a column to a regular float64 pandas Series.
    This explicitly avoids Arrow large_string arithmetic errors.
    """
    converted_values = pd.to_numeric(
        series.astype("string"),
        errors="coerce"
    ).to_numpy(dtype="float64")

    return pd.Series(
        converted_values,
        index=series.index,
        name=series.name,
        dtype="float64"
    )


# Replace common textual missing-value markers before type conversion.
df = df.replace(missing_tokens, np.nan)

# Convert numeric-looking string and Arrow string columns to float64.
# The explicit conversion handles pandas Arrow large_string columns safely.
for column in df.columns:
    if not pd.api.types.is_numeric_dtype(df[column]):
        numeric_candidate = force_numeric_series(df[column])
        non_missing_count = int(df[column].notna().sum())
        convertible_count = int(numeric_candidate.notna().sum())

        if (
            non_missing_count > 0
            and convertible_count / non_missing_count >= 0.80
        ):
            df[column] = numeric_candidate

# The target is expected to be numeric after parsing.
if TARGET_COLUMN not in df.columns:
    raise ValueError(f"Required target column '{TARGET_COLUMN}' was not found.")

if not pd.api.types.is_numeric_dtype(df[TARGET_COLUMN]):
    df[TARGET_COLUMN] = force_numeric_series(df[TARGET_COLUMN])

if df[TARGET_COLUMN].notna().sum() == 0:
    raise ValueError(f"Target column '{TARGET_COLUMN}' contains no numeric values.")

# ---------------------------------------------------------------------
# Smart, type-safe imputation
# ---------------------------------------------------------------------
imputation_summary = {
    "rules_applied": {
        "numeric_skewed": "Median imputation when skewness > 1 or skewness < -1.",
        "numeric_symmetric": "Mean imputation when -1 <= skewness <= 1.",
        "categorical": "Mode imputation, or Unknown if no mode exists.",
        "missing_tokens_replaced": missing_tokens
    },
    "columns": {}
}

for column in df.columns:
    missing_before = int(df[column].isna().sum())
    dtype_before = str(df[column].dtype)

    if pd.api.types.is_numeric_dtype(df[column]):
        column_skewness = df[column].skew()

        if not np.isfinite(column_skewness):
            column_skewness = 0.0

        if column_skewness > 1 or column_skewness < -1:
            method = "median"
            fill_value = df[column].median()
        else:
            method = "mean"
            fill_value = df[column].mean()

        if pd.isna(fill_value):
            fill_value = 0.0

        df[column] = df[column].fillna(float(fill_value))

        imputation_summary["columns"][column] = {
            "dtype_before": dtype_before,
            "dtype_after": str(df[column].dtype),
            "missing_count_before": missing_before,
            "missing_count_after": int(df[column].isna().sum()),
            "skewness": float(column_skewness),
            "method": method,
            "fill_value": float(fill_value)
        }
    else:
        mode_values = df[column].mode(dropna=True)

        if len(mode_values) > 0:
            fill_value = mode_values.iloc[0]
            method = "mode"
        else:
            fill_value = "Unknown"
            method = "placeholder"

        df[column] = df[column].fillna(fill_value)

        imputation_summary["columns"][column] = {
            "dtype_before": dtype_before,
            "dtype_after": str(df[column].dtype),
            "missing_count_before": missing_before,
            "missing_count_after": int(df[column].isna().sum()),
            "method": method,
            "fill_value": str(fill_value)
        }

# ---------------------------------------------------------------------
# Domain-specific feature engineering
# ---------------------------------------------------------------------
engineered_features = []

if {"body_weight", "brain_weight"}.issubset(df.columns):
    body_values = pd.to_numeric(
        df["body_weight"], errors="coerce"
    ).to_numpy(dtype="float64")
    brain_values = pd.to_numeric(
        df["brain_weight"], errors="coerce"
    ).to_numpy(dtype="float64")

    body_denominator = np.where(body_values == 0, np.nan, body_values)
    ratio_values = np.divide(
        brain_values,
        body_denominator,
        out=np.zeros_like(brain_values, dtype="float64"),
        where=np.isfinite(body_denominator)
    )

    df["brain_body_ratio"] = pd.Series(
        ratio_values,
        index=df.index,
        dtype="float64"
    ).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    engineered_features.append({
        "feature_name": "brain_body_ratio",
        "formula_method": "brain_weight / body_weight, with zero denominators safely handled",
        "data_type": str(df["brain_body_ratio"].dtype),
        "rationale_purpose": "Measures relative brain investment independently of absolute body size."
    })

if "body_weight" in df.columns:
    body_values = pd.to_numeric(
        df["body_weight"], errors="coerce"
    ).to_numpy(dtype="float64")

    df["log_body_weight"] = pd.Series(
        np.log1p(np.clip(body_values, 0, None)),
        index=df.index,
        dtype="float64"
    )

    engineered_features.append({
        "feature_name": "log_body_weight",
        "formula_method": "log1p(body_weight)",
        "data_type": str(df["log_body_weight"].dtype),
        "rationale_purpose": "Reduces extreme right skew and limits the influence of very large species."
    })

if "brain_weight" in df.columns:
    brain_values = pd.to_numeric(
        df["brain_weight"], errors="coerce"
    ).to_numpy(dtype="float64")

    df["log_brain_weight"] = pd.Series(
        np.log1p(np.clip(brain_values, 0, None)),
        index=df.index,
        dtype="float64"
    )

    engineered_features.append({
        "feature_name": "log_brain_weight",
        "formula_method": "log1p(brain_weight)",
        "data_type": str(df["log_brain_weight"].dtype),
        "rationale_purpose": "Stabilizes the heavily skewed brain-weight distribution."
    })

if {"gestation_time", "total_sleep"}.issubset(df.columns):
    # Explicit numeric arrays prevent division of Arrow large_string values.
    gestation_values = pd.to_numeric(
        df["gestation_time"], errors="coerce"
    ).to_numpy(dtype="float64")
    sleep_values = pd.to_numeric(
        df["total_sleep"], errors="coerce"
    ).to_numpy(dtype="float64")

    sleep_denominator = np.where(sleep_values == 0, np.nan, sleep_values)
    ratio_values = np.divide(
        gestation_values,
        sleep_denominator,
        out=np.zeros_like(gestation_values, dtype="float64"),
        where=np.isfinite(sleep_denominator)
    )

    df["gestation_sleep_ratio"] = pd.Series(
        ratio_values,
        index=df.index,
        dtype="float64"
    ).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    engineered_features.append({
        "feature_name": "gestation_sleep_ratio",
        "formula_method": "gestation_time / total_sleep using explicitly parsed float64 arrays",
        "data_type": str(df["gestation_sleep_ratio"].dtype),
        "rationale_purpose": "Captures the relative relationship between gestation and sleep duration."
    })

# Impute any missing values created during feature engineering.
for feature_record in engineered_features:
    feature_name = feature_record["feature_name"]
    if df[feature_name].isna().any():
        df[feature_name] = df[feature_name].fillna(df[feature_name].median())

# ---------------------------------------------------------------------
# IQR outlier analysis
# ---------------------------------------------------------------------
numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()

outlier_analysis = {
    "method": "IQR rule: below Q1 - 1.5*IQR or above Q3 + 1.5*IQR",
    "numeric_columns": {}
}

for column in numeric_columns:
    q1 = float(df[column].quantile(0.25))
    q3 = float(df[column].quantile(0.75))
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    outlier_mask = (
        (df[column] < lower_bound)
        | (df[column] > upper_bound)
    )

    outlier_count = int(outlier_mask.sum())
    outlier_percentage = float(outlier_count / len(df) * 100)

    outlier_analysis["numeric_columns"][column] = {
        "Q1": q1,
        "Q3": q3,
        "IQR": float(iqr),
        "lower_bound": float(lower_bound),
        "upper_bound": float(upper_bound),
        "outlier_count": outlier_count,
        "outlier_percentage": outlier_percentage
    }

    print(
        f"Outliers in {column}: "
        f"{outlier_count}/{len(df)} ({outlier_percentage:.2f}%)"
    )

# ---------------------------------------------------------------------
# Pearson correlation analysis and heatmap
# ---------------------------------------------------------------------
numeric_df = df.select_dtypes(include=[np.number]).copy()

correlation_analysis = {
    "top_positive_correlations": [],
    "top_negative_correlations": [],
    "target_correlations": {},
    "correlation_matrix_text": ""
}

if numeric_df.shape[1] >= 2:
    correlation_matrix = numeric_df.corr(method="pearson")

    correlation_analysis["correlation_matrix_text"] = correlation_matrix.to_string(
        float_format=lambda value: f"{value:.6f}"
    )

    upper_triangle = correlation_matrix.where(
        np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool)
    )

    correlation_pairs = (
        upper_triangle
        .stack()
        .reset_index()
    )
    correlation_pairs.columns = [
        "feature_1",
        "feature_2",
        "correlation"
    ]

    positive_pairs = correlation_pairs.sort_values(
        "correlation",
        ascending=False
    ).head(10)

    negative_pairs = correlation_pairs.sort_values(
        "correlation",
        ascending=True
    ).head(10)

    correlation_analysis["top_positive_correlations"] = [
        {
            "feature_1": row["feature_1"],
            "feature_2": row["feature_2"],
            "correlation": float(row["correlation"])
        }
        for _, row in positive_pairs.iterrows()
    ]

    correlation_analysis["top_negative_correlations"] = [
        {
            "feature_1": row["feature_1"],
            "feature_2": row["feature_2"],
            "correlation": float(row["correlation"])
        }
        for _, row in negative_pairs.iterrows()
    ]

    if TARGET_COLUMN in correlation_matrix.columns:
        target_correlations = correlation_matrix[TARGET_COLUMN].drop(
            labels=[TARGET_COLUMN],
            errors="ignore"
        ).sort_values(
            key=lambda values: values.abs(),
            ascending=False
        )

        correlation_analysis["target_correlations"] = {
            str(feature): float(value)
            for feature, value in target_correlations.items()
        }

    plt.figure(figsize=(12, 9))
    sns.heatmap(
        correlation_matrix,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        square=True,
        linewidths=0.5
    )
    plt.title("Pearson Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(
        "correlation_matrix.png",
        dpi=200,
        bbox_inches="tight"
    )
    plt.close()
else:
    correlation_analysis["correlation_matrix_text"] = (
        "Insufficient numeric columns for correlation analysis."
    )

# Add target correlation to engineered feature metadata.
for feature_record in engineered_features:
    feature_name = feature_record["feature_name"]

    if (
        feature_name in df.columns
        and TARGET_COLUMN in df.columns
        and df[feature_name].nunique() > 1
        and df[TARGET_COLUMN].nunique() > 1
    ):
        feature_record["correlation_with_target"] = float(
            df[feature_name].corr(df[TARGET_COLUMN])
        )
    else:
        feature_record["correlation_with_target"] = None

# ---------------------------------------------------------------------
# Hypothesis testing
# ---------------------------------------------------------------------
statistical_tests = {}
significant_predictors = []

target_series = df[TARGET_COLUMN]

for column in df.columns:
    if column == TARGET_COLUMN:
        continue

    predictor = df[column]

    if pd.api.types.is_numeric_dtype(predictor):
        if (
            predictor.nunique() > 1
            and target_series.nunique() > 1
            and len(df) >= 3
        ):
            valid_mask = predictor.notna() & target_series.notna()

            test_result = stats.pearsonr(
                predictor.loc[valid_mask],
                target_series.loc[valid_mask]
            )

            statistic_value = float(test_result.statistic)
            p_value = float(test_result.pvalue)

            interpretation = (
                "A statistically significant linear association was detected."
                if p_value < ALPHA
                else "No statistically significant linear association was detected."
            )

            statistical_tests[column] = {
                "test_name": "Pearson correlation test",
                "statistic_value": statistic_value,
                "p_value": p_value,
                "degrees_of_freedom": None,
                "is_statistically_significant": bool(p_value < ALPHA),
                "interpretation_summary": interpretation
            }

            if p_value < ALPHA:
                significant_predictors.append(column)

    else:
        grouped_values = [
            group[TARGET_COLUMN].dropna()
            for _, group in df.groupby(column, dropna=False)
        ]
        grouped_values = [
            group for group in grouped_values if len(group) >= 2
        ]

        if len(grouped_values) == 2:
            test_result = stats.ttest_ind(
                grouped_values[0],
                grouped_values[1],
                equal_var=False,
                nan_policy="omit"
            )
            test_name = "Welch independent-samples t-test"
            degrees_of_freedom = None
        elif len(grouped_values) > 2:
            test_result = stats.f_oneway(*grouped_values)
            test_name = "One-way ANOVA"
            degrees_of_freedom = None
        else:
            continue

        statistic_value = float(test_result.statistic)
        p_value = float(test_result.pvalue)

        interpretation = (
            "Target means differ significantly across predictor groups."
            if p_value < ALPHA
            else "No statistically significant difference in target means was detected across groups."
        )

        statistical_tests[column] = {
            "test_name": test_name,
            "statistic_value": statistic_value,
            "p_value": p_value,
            "degrees_of_freedom": degrees_of_freedom,
            "is_statistically_significant": bool(p_value < ALPHA),
            "interpretation_summary": interpretation
        }

        if p_value < ALPHA:
            significant_predictors.append(column)

print("Statistical hypothesis test results:")
for feature_name, test_details in statistical_tests.items():
    print(
        f"{feature_name}: "
        f"{test_details['test_name']}, "
        f"p-value={test_details['p_value']:.12g}"
    )

# ---------------------------------------------------------------------
# Target interaction visualization
# ---------------------------------------------------------------------
interaction_feature = None

for candidate in [
    "body_weight",
    "brain_weight",
    "log_body_weight",
    "log_brain_weight"
]:
    if candidate in df.columns:
        interaction_feature = candidate
        break

if interaction_feature is not None:
    target_median = df[TARGET_COLUMN].median()

    interaction_data = pd.DataFrame({
        "feature": df[interaction_feature].astype("float64"),
        "target_segment": np.where(
            df[TARGET_COLUMN] <= target_median,
            "Lower target",
            "Higher target"
        )
    })

    plt.figure(figsize=(10, 6))
    sns.boxplot(
        data=interaction_data,
        x="target_segment",
        y="feature",
        color="#6baed6"
    )
    plt.yscale("symlog", linthresh=1)
    plt.title(f"{interaction_feature} Distribution by Target Segment")
    plt.xlabel("Target segment based on median total sleep")
    plt.ylabel(interaction_feature)
    plt.tight_layout()
    plt.savefig(
        "target_interactions.png",
        dpi=200,
        bbox_inches="tight"
    )
    plt.close()
else:
    plt.figure(figsize=(8, 5))
    plt.text(
        0.5,
        0.5,
        "Target interaction visualization unavailable",
        ha="center",
        va="center"
    )
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(
        "target_interactions.png",
        dpi=200,
        bbox_inches="tight"
    )
    plt.close()

# ---------------------------------------------------------------------
# Dataset overview
# ---------------------------------------------------------------------
column_summary = {}

for column in df_raw.columns:
    raw_missing_tokens = df_raw[column].astype("string").isin(missing_tokens).sum()

    column_summary[column] = {
        "dtype_raw": str(df_raw[column].dtype),
        "dtype_processed": str(df[column].dtype),
        "missing_count_raw": int(
            df_raw[column].isna().sum() + raw_missing_tokens
        ),
        "missing_count_after_processing": int(df[column].isna().sum()),
        "cardinality_raw": int(df_raw[column].nunique(dropna=True)),
        "cardinality_processed": int(df[column].nunique(dropna=True)),
        "profile_metadata": profile_schema.get(column, {})
    }

dataset_overview = {
    "shape": {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1])
    },
    "target_column": TARGET_COLUMN,
    "target_definition": "Continuous total sleep duration parsed from total_sleep.",
    "column_summary": column_summary,
    "profile_dimensions": profile.get("dimensions"),
    "profile_missing_values_summary": profile_missing_summary
}

# ---------------------------------------------------------------------
# Extracted insights
# ---------------------------------------------------------------------
target_correlations = correlation_analysis["target_correlations"]

ranked_target_drivers = sorted(
    target_correlations.items(),
    key=lambda item: abs(item[1]),
    reverse=True
)

key_findings = []
data_quality_issues = []

for column in ["body_weight", "brain_weight"]:
    if column in df.columns and abs(df[column].skew()) > 1:
        key_findings.append(
            f"{column} is highly skewed; logarithmic transformation is recommended."
        )

if ranked_target_drivers:
    strongest_feature, strongest_correlation = ranked_target_drivers[0]
    key_findings.append(
        f"The strongest absolute target correlation is {strongest_feature} "
        f"with correlation {strongest_correlation:.4f}."
    )

if significant_predictors:
    key_findings.append(
        "Statistically significant predictors: "
        + ", ".join(sorted(set(significant_predictors)))
        + "."
    )
else:
    key_findings.append(
        "No predictors met the unadjusted 0.05 significance threshold."
    )

for column, details in outlier_analysis["numeric_columns"].items():
    if details["outlier_percentage"] >= 10:
        data_quality_issues.append(
            f"{column} contains {details['outlier_percentage']:.2f}% IQR-defined outliers."
        )

for column, details in imputation_summary["columns"].items():
    if details["missing_count_before"] > 0:
        data_quality_issues.append(
            f"{column} required {details['missing_count_before']} missing-value replacements "
            f"using {details['method']} imputation."
        )

data_quality_issues.extend([
    "The sample size is small, so statistical tests and validation estimates may be unstable.",
    "Several columns contain placeholder strings or numeric-looking string values.",
    "Extreme body-weight and brain-weight values may strongly influence correlations and models.",
    "Univariate statistical significance does not establish causation.",
    "Multiple hypothesis tests should be adjusted using an FDR procedure for formal inference."
])

top_key_feature_drivers = [
    {
        "feature": feature,
        "target_correlation": float(correlation),
        "absolute_target_correlation": float(abs(correlation))
    }
    for feature, correlation in ranked_target_drivers[:10]
]

extracted_insights = {
    "key_findings": key_findings,
    "data_quality_issues": data_quality_issues,
    "key_feature_drivers": top_key_feature_drivers
}

# ---------------------------------------------------------------------
# Predictive modeling blueprint
# ---------------------------------------------------------------------
predictive_modeling_blueprint = {
    "target_definition": {
        "column": TARGET_COLUMN,
        "problem_type": "Supervised regression",
        "description": "Predict continuous total sleep duration."
    },
    "recommended_algorithms": [
        "Median baseline regressor",
        "Regularized linear regression",
        "Random Forest Regressor",
        "Gradient Boosting Regressor",
        "HistGradientBoostingRegressor"
    ],
    "feature_selection_strategy": [
        "Use domain features such as brain_body_ratio and logarithmic weight features.",
        "Remove identifiers and leakage-prone fields if present.",
        "Inspect redundancy using the correlation matrix.",
        "Use repeated cross-validated permutation importance.",
        "Prefer compact feature sets because of the small sample size."
    ],
    "validation_strategy": [
        "Use repeated 5-fold cross-validation.",
        "Place imputation, transformations, scaling, and encoding inside a Pipeline.",
        "Report MAE, RMSE, and R-squared with variability estimates.",
        "Compare all models against the median-target baseline."
    ],
    "preprocessing_steps": [
        "Replace '?' and other placeholder strings with missing values.",
        "Parse numeric-looking strings explicitly to float64.",
        "Use median imputation for skewed numeric fields.",
        "Use mean imputation for symmetric numeric fields.",
        "Use mode or Unknown for categorical fields.",
        "Apply log1p transformations to heavily skewed positive variables.",
        "Standardize predictors for regularized linear models.",
        "One-hot encode remaining categorical predictors."
    ],
    "overfitting_risk_mitigation": [
        "Use regularization and constrained tree depth.",
        "Use minimum leaf-size constraints for tree ensembles.",
        "Avoid repeated tuning against final evaluation data.",
        "Use repeated cross-validation because there are only 62 rows.",
        "Investigate influential observations and robust alternatives.",
        "Prefer stable performance over marginal improvements."
    ],
    "overall_executive_modeling_strategy_summary": (
        "Start with a transparent median baseline and regularized regression using "
        "log-transformed size variables and biologically motivated ratios. Compare "
        "against constrained tree ensembles using repeated cross-validation. Prioritize "
        "MAE and stability because the dataset is small and contains extreme values."
    )
}

# ---------------------------------------------------------------------
# Save complete metrics JSON
# ---------------------------------------------------------------------
metrics_dict = {
    "dataset_overview": dataset_overview,
    "imputation_summary": imputation_summary,
    "outlier_analysis": outlier_analysis,
    "engineered_features": engineered_features,
    "correlation_analysis": correlation_analysis,
    "statistical_hypothesis_tests": {
        "tests_by_feature": statistical_tests,
        "significance_level": ALPHA,
        "significant_predictors": sorted(set(significant_predictors))
    },
    "extracted_insights": extracted_insights,
    "predictive_modeling_blueprint": predictive_modeling_blueprint
}

with open(METRICS_FILEPATH, "w", encoding="utf-8") as metrics_file:
    json.dump(
        json_safe(metrics_dict),
        metrics_file,
        indent=2
    )

print("EDA completed successfully.")
print("Saved visualization: correlation_matrix.png")
print("Saved visualization: target_interactions.png")
print("Saved metrics: metrics.json")
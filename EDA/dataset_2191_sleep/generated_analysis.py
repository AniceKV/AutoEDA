DATA_FILEPATH = r'C:\Users\Anish Kumar Verma\PycharmProjects\AutoEDA\test_data\dataset_2191_sleep.csv'

import os
import re
import json
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from scipy import stats
from scipy.stats import f_oneway, pearsonr, chi2_contingency

warnings.filterwarnings("ignore")

# DATA_FILEPATH must be defined globally by the execution environment.
if "DATA_FILEPATH" not in globals():
    raise NameError("DATA_FILEPATH must be defined as a global variable.")

PROFILE_FILEPATH = "metadata_profile.json"
METRICS_FILEPATH = "metrics.json"
TARGET_COLUMN = "total_sleep"

# ---------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------

def make_json_safe(value):
    """Convert numpy/pandas/scipy values into JSON-serializable values."""
    if isinstance(value, dict):
        return {str(k): make_json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [make_json_safe(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        if np.isnan(value) or np.isinf(value):
            return None
        return float(value)
    if isinstance(value, float):
        if np.isnan(value) or np.isinf(value):
            return None
        return value
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if pd.isna(value):
        return None
    return value


def safe_float(value):
    try:
        value = float(value)
        return None if not np.isfinite(value) else value
    except Exception:
        return None


def infer_numeric_columns(dataframe, threshold=0.80):
    """
    Convert object columns to numeric when at least threshold of nonmissing
    values can be parsed numerically.
    """
    converted_columns = []
    for column in dataframe.columns:
        if dataframe[column].dtype == "object":
            cleaned = dataframe[column].replace(
                ["?", "NA", "N/A", "na", "null", "None", ""],
                np.nan
            )
            numeric_version = pd.to_numeric(cleaned, errors="coerce")
            nonmissing_count = cleaned.notna().sum()

            if nonmissing_count > 0:
                conversion_rate = numeric_version.notna().sum() / nonmissing_count
                if conversion_rate >= threshold:
                    dataframe[column] = numeric_version
                    converted_columns.append(column)
                else:
                    dataframe[column] = cleaned.astype("object")
    return converted_columns


def column_summary(dataframe):
    summary = {}
    for column in dataframe.columns:
        summary[column] = {
            "dtype": str(dataframe[column].dtype),
            "missing_count": int(dataframe[column].isna().sum()),
            "missing_percentage": float(dataframe[column].isna().mean() * 100),
            "cardinality": int(dataframe[column].nunique(dropna=True))
        }
    return summary


def select_target(dataframe):
    if TARGET_COLUMN in dataframe.columns:
        return TARGET_COLUMN

    candidate_names = [
        "target", "label", "y", "outcome", "response",
        "survived", "price", "sale_price"
    ]
    for candidate in candidate_names:
        if candidate in dataframe.columns:
            return candidate

    numeric_columns = dataframe.select_dtypes(include=np.number).columns.tolist()
    if numeric_columns:
        return numeric_columns[-1]

    return dataframe.columns[-1]


def interpret_p_value(p_value, alpha=0.05):
    if p_value is None:
        return "Test could not be completed."
    if p_value < alpha:
        return "Evidence suggests a statistically significant association with the target."
    return "Insufficient evidence of a statistically significant association with the target."


def calculate_correlation_records(correlation_matrix, target_column):
    if target_column not in correlation_matrix.columns:
        return [], [], []

    target_correlations = correlation_matrix[target_column].drop(labels=[target_column])
    target_correlations = target_correlations.dropna().sort_values(ascending=False)

    positive = [
        {
            "feature": str(feature),
            "correlation": safe_float(value)
        }
        for feature, value in target_correlations.head(10).items()
        if value > 0
    ]

    negative = [
        {
            "feature": str(feature),
            "correlation": safe_float(value)
        }
        for feature, value in target_correlations.sort_values().head(10).items()
        if value < 0
    ]

    target_records = [
        {
            "feature": str(feature),
            "correlation": safe_float(value)
        }
        for feature, value in target_correlations.items()
    ]

    return positive, negative, target_records


# ---------------------------------------------------------------------
# Load data and metadata profile
# ---------------------------------------------------------------------

with open(PROFILE_FILEPATH, "r", encoding="utf-8") as profile_file:
    metadata_profile = json.load(profile_file)

dataframe = pd.read_csv(DATA_FILEPATH)
raw_dataframe = dataframe.copy()

profile_schema = metadata_profile.get("schema", {})
original_missing_counts = dataframe.isna().sum().to_dict()

# Treat common textual missing-value markers as missing.
missing_markers = ["?", "NA", "N/A", "na", "null", "None", ""]
dataframe = dataframe.replace(missing_markers, np.nan)

converted_numeric_columns = infer_numeric_columns(dataframe)
target_column = select_target(dataframe)

# ---------------------------------------------------------------------
# Smart type-safe imputation
# ---------------------------------------------------------------------

imputation_summary = {
    "rules": {
        "numeric_highly_skewed": "Median imputation when absolute skewness is greater than 1.",
        "numeric_symmetric": "Mean imputation when absolute skewness is less than or equal to 1.",
        "categorical": "Mode imputation, or Unknown when no mode exists.",
        "target": "Target values are not imputed for statistical testing or modeling."
    },
    "columns": {}
}

for column in dataframe.columns:
    missing_before = int(dataframe[column].isna().sum())
    dtype_before = str(dataframe[column].dtype)

    if missing_before == 0:
        imputation_summary["columns"][column] = {
            "dtype": dtype_before,
            "missing_before": 0,
            "filled_count": 0,
            "method": "none",
            "replacement_value": None,
            "skewness": safe_float(dataframe[column].skew())
        }
        continue

    if column == target_column:
        imputation_summary["columns"][column] = {
            "dtype": dtype_before,
            "missing_before": missing_before,
            "filled_count": 0,
            "method": "not_imputed_target",
            "replacement_value": None,
            "skewness": safe_float(
                dataframe[column].dropna().skew()
                if pd.api.types.is_numeric_dtype(dataframe[column])
                else np.nan
            )
        }
        continue

    if pd.api.types.is_numeric_dtype(dataframe[column]):
        skewness = dataframe[column].dropna().skew()
        if pd.isna(skewness):
            skewness = 0.0

        if abs(skewness) > 1:
            replacement_value = dataframe[column].median()
            method = "median"
        else:
            replacement_value = dataframe[column].mean()
            method = "mean"

        if pd.isna(replacement_value):
            replacement_value = 0.0

        dataframe[column] = dataframe[column].fillna(replacement_value)
    else:
        mode_values = dataframe[column].mode(dropna=True)
        if len(mode_values) > 0:
            replacement_value = mode_values.iloc[0]
            method = "mode"
        else:
            replacement_value = "Unknown"
            method = "placeholder_unknown"

        dataframe[column] = dataframe[column].fillna(replacement_value)
        skewness = None

    filled_count = missing_before - int(dataframe[column].isna().sum())

    imputation_summary["columns"][column] = {
        "dtype": dtype_before,
        "missing_before": missing_before,
        "filled_count": int(filled_count),
        "method": method,
        "replacement_value": make_json_safe(replacement_value),
        "skewness": safe_float(skewness)
    }

# ---------------------------------------------------------------------
# Target preparation
# ---------------------------------------------------------------------

target_numeric = pd.to_numeric(dataframe[target_column], errors="coerce")
valid_target_mask = target_numeric.notna()
dataframe[target_column] = target_numeric

analysis_dataframe = dataframe.loc[valid_target_mask].copy()

# ---------------------------------------------------------------------
# Domain-specific feature engineering
# ---------------------------------------------------------------------

engineered_feature_records = []

if {"brain_weight", "body_weight"}.issubset(analysis_dataframe.columns):
    denominator = analysis_dataframe["body_weight"].replace(0, np.nan)
    analysis_dataframe["brain_body_ratio"] = (
        analysis_dataframe["brain_weight"] / denominator
    )
    analysis_dataframe["brain_body_ratio"] = (
        analysis_dataframe["brain_body_ratio"].replace([np.inf, -np.inf], np.nan)
    )
    ratio_fill = analysis_dataframe["brain_body_ratio"].median()
    analysis_dataframe["brain_body_ratio"] = (
        analysis_dataframe["brain_body_ratio"].fillna(ratio_fill)
    )

    engineered_feature_records.append({
        "feature_name": "brain_body_ratio",
        "formula_method": "brain_weight / body_weight",
        "data_type": str(analysis_dataframe["brain_body_ratio"].dtype),
        "rationale_purpose": "Approximates relative brain investment while reducing raw body-size scale effects.",
        "correlation_with_target": None
    })

if "body_weight" in analysis_dataframe.columns:
    body_values = analysis_dataframe["body_weight"].clip(lower=0)
    analysis_dataframe["log_body_weight"] = np.log1p(body_values)

    engineered_feature_records.append({
        "feature_name": "log_body_weight",
        "formula_method": "log1p(body_weight)",
        "data_type": str(analysis_dataframe["log_body_weight"].dtype),
        "rationale_purpose": "Compresses extreme body-weight values and improves linear-model stability under severe right skew.",
        "correlation_with_target": None
    })

if {"predation_index", "danger_index"}.issubset(analysis_dataframe.columns):
    analysis_dataframe["predation_danger_interaction"] = (
        analysis_dataframe["predation_index"] *
        analysis_dataframe["danger_index"]
    )

    engineered_feature_records.append({
        "feature_name": "predation_danger_interaction",
        "formula_method": "predation_index * danger_index",
        "data_type": str(analysis_dataframe["predation_danger_interaction"].dtype),
        "rationale_purpose": "Captures combined ecological risk that may not be represented by either index independently.",
        "correlation_with_target": None
    })

# Impute engineered numeric features safely.
for column in analysis_dataframe.columns:
    if column not in imputation_summary["columns"] and analysis_dataframe[column].isna().any():
        if pd.api.types.is_numeric_dtype(analysis_dataframe[column]):
            replacement = analysis_dataframe[column].median()
            analysis_dataframe[column] = analysis_dataframe[column].fillna(replacement)
            imputation_summary["columns"][column] = {
                "dtype": str(analysis_dataframe[column].dtype),
                "missing_before": int(analysis_dataframe[column].isna().sum()),
                "filled_count": int(analysis_dataframe[column].isna().sum()),
                "method": "median_engineered_feature",
                "replacement_value": safe_float(replacement),
                "skewness": safe_float(analysis_dataframe[column].skew())
            }

# ---------------------------------------------------------------------
# Outlier profiling using IQR
# ---------------------------------------------------------------------

outlier_analysis = {}
numeric_predictors = analysis_dataframe.select_dtypes(include=np.number).columns.tolist()
numeric_predictors = [
    column for column in numeric_predictors if column != target_column
]

for column in numeric_predictors:
    series = analysis_dataframe[column].dropna()

    if len(series) == 0:
        continue

    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    outlier_mask = (series < lower_bound) | (series > upper_bound)
    outlier_count = int(outlier_mask.sum())

    outlier_analysis[column] = {
        "Q1": safe_float(q1),
        "Q3": safe_float(q3),
        "IQR": safe_float(iqr),
        "lower_bound": safe_float(lower_bound),
        "upper_bound": safe_float(upper_bound),
        "outlier_count": outlier_count,
        "outlier_percentage": safe_float(outlier_count / len(series) * 100)
    }

    print(
        "Outliers in {}: {} of {} ({:.2f}%)".format(
            column,
            outlier_count,
            len(series),
            outlier_count / len(series) * 100
        )
    )

# ---------------------------------------------------------------------
# Correlation analysis
# ---------------------------------------------------------------------

numeric_analysis_dataframe = analysis_dataframe.select_dtypes(
    include=np.number
)

correlation_matrix = numeric_analysis_dataframe.corr(method="pearson")
positive_correlations, negative_correlations, target_correlations = (
    calculate_correlation_records(correlation_matrix, target_column)
)

for record in engineered_feature_records:
    feature_name = record["feature_name"]
    if feature_name in correlation_matrix.columns:
        record["correlation_with_target"] = safe_float(
            correlation_matrix.loc[feature_name, target_column]
        )

# Save Pearson heatmap.
plt.figure(figsize=(12, 9))
sns.heatmap(
    correlation_matrix,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    center=0,
    linewidths=0.5,
    square=False
)
plt.title("Pearson Correlation Matrix")
plt.tight_layout()
plt.savefig("correlation_matrix.png", dpi=200, bbox_inches="tight")
plt.close()

# ---------------------------------------------------------------------
# Target interaction visualization
# ---------------------------------------------------------------------

interaction_feature = None
for preferred_feature in [
    "body_weight",
    "brain_weight",
    "log_body_weight",
    "brain_body_ratio"
]:
    if preferred_feature in analysis_dataframe.columns:
        interaction_feature = preferred_feature
        break

if interaction_feature is not None and target_column in analysis_dataframe.columns:
    interaction_dataframe = analysis_dataframe[
        [interaction_feature, target_column]
    ].dropna()

    if len(interaction_dataframe) > 0:
        target_unique_count = interaction_dataframe[target_column].nunique()

        if target_unique_count >= 2:
            try:
                interaction_dataframe["target_segment"] = pd.qcut(
                    interaction_dataframe[target_column],
                    q=min(4, target_unique_count),
                    duplicates="drop"
                )
            except Exception:
                interaction_dataframe["target_segment"] = pd.cut(
                    interaction_dataframe[target_column],
                    bins=3,
                    duplicates="drop"
                )

            plt.figure(figsize=(11, 7))
            sns.boxplot(
                data=interaction_dataframe,
                x="target_segment",
                y=interaction_feature,
                color="#6baed6"
            )
            plt.xticks(rotation=30, ha="right")
            plt.title(
                "{} distribution segmented by {}".format(
                    interaction_feature,
                    target_column
                )
            )
            plt.xlabel("Target segment")
            plt.ylabel(interaction_feature)
            plt.tight_layout()
            plt.savefig("target_interactions.png", dpi=200, bbox_inches="tight")
            plt.close()

# ---------------------------------------------------------------------
# Statistical hypothesis testing
# ---------------------------------------------------------------------

statistical_hypothesis_tests = {}
significant_predictors = []
alpha = 0.05

feature_columns = [
    column for column in analysis_dataframe.columns
    if column != target_column
]

for feature in feature_columns:
    test_dataframe = analysis_dataframe[[feature, target_column]].dropna()

    if len(test_dataframe) < 3:
        statistical_hypothesis_tests[feature] = {
            "test_name": "not_run_insufficient_data",
            "statistic_value": None,
            "p_value": None,
            "degrees_of_freedom": None,
            "is_statistically_significant": False,
            "interpretation_summary": "Insufficient valid observations."
        }
        continue

    feature_series = test_dataframe[feature]
    target_series = test_dataframe[target_column]

    if pd.api.types.is_numeric_dtype(feature_series):
        if feature_series.nunique() >= 3 and target_series.nunique() >= 3:
            statistic_value, p_value = pearsonr(
                feature_series.astype(float),
                target_series.astype(float)
            )

            test_name = "Pearson correlation significance test"
            degrees_of_freedom = len(test_dataframe) - 2
            interpretation = interpret_p_value(p_value)

        else:
            grouped_values = [
                group[target_column].values
                for _, group in test_dataframe.groupby(feature)
                if len(group) >= 2
            ]

            if len(grouped_values) >= 2:
                statistic_value, p_value = f_oneway(*grouped_values)
                test_name = "One-way ANOVA across feature levels"
                degrees_of_freedom = [len(grouped_values) - 1, len(test_dataframe) - len(grouped_values)]
                interpretation = interpret_p_value(p_value)
            else:
                statistic_value = None
                p_value = None
                test_name = "not_run_insufficient_groups"
                degrees_of_freedom = None
                interpretation = "Insufficient groups for testing."
    else:
        target_bins = pd.qcut(
            target_series,
            q=min(3, target_series.nunique()),
            duplicates="drop"
        )
        contingency_table = pd.crosstab(feature_series, target_bins)

        if contingency_table.shape[0] >= 2 and contingency_table.shape[1] >= 2:
            chi2, p_value, dof, expected = chi2_contingency(
                contingency_table
            )
            statistic_value = chi2
            test_name = "Chi-square independence test"
            degrees_of_freedom = dof
            interpretation = interpret_p_value(p_value)
        else:
            statistic_value = None
            p_value = None
            test_name = "not_run_insufficient_contingency_table"
            degrees_of_freedom = None
            interpretation = "Insufficient category variation for testing."

    is_significant = p_value is not None and p_value < alpha

    statistical_hypothesis_tests[feature] = {
        "test_name": test_name,
        "statistic_value": safe_float(statistic_value),
        "p_value": safe_float(p_value),
        "degrees_of_freedom": make_json_safe(degrees_of_freedom),
        "is_statistically_significant": bool(is_significant),
        "interpretation_summary": interpretation
    }

    print(
        "Hypothesis test for {}: {} | p-value = {}".format(
            feature,
            test_name,
            "None" if p_value is None else "{:.8g}".format(p_value)
        )
    )

    if is_significant:
        significant_predictors.append(feature)

# ---------------------------------------------------------------------
# Insights and modeling blueprint
# ---------------------------------------------------------------------

data_quality_issues = []

for column, details in imputation_summary["columns"].items():
    if details["missing_before"] > 0:
        data_quality_issues.append(
            "{} contained {} missing values and was handled using {}.".format(
                column,
                details["missing_before"],
                details["method"]
            )
        )

for column, details in outlier_analysis.items():
    if details["outlier_percentage"] is not None and details["outlier_percentage"] > 5:
        data_quality_issues.append(
            "{} has {:.2f}% IQR-defined outliers; robust scaling or log transformation should be considered.".format(
                column,
                details["outlier_percentage"]
            )
        )

for column in raw_dataframe.columns:
    if raw_dataframe[column].dtype == "object":
        marker_count = int(raw_dataframe[column].isin(missing_markers).sum())
        if marker_count > 0:
            data_quality_issues.append(
                "{} uses textual missing-value markers in {} records.".format(
                    column,
                    marker_count
                )
            )

key_findings = []

if target_column in correlation_matrix.columns:
    absolute_target_correlations = (
        correlation_matrix[target_column]
        .drop(labels=[target_column])
        .abs()
        .sort_values(ascending=False)
    )

    if len(absolute_target_correlations) > 0:
        strongest_feature = absolute_target_correlations.index[0]
        strongest_value = correlation_matrix.loc[strongest_feature, target_column]
        key_findings.append(
            "{} has the strongest linear association with {}: r = {:.3f}.".format(
                strongest_feature,
                target_column,
                strongest_value
            )
        )

if significant_predictors:
    key_findings.append(
        "Statistically significant predictors at alpha = 0.05 include: {}.".format(
            ", ".join(significant_predictors)
        )
    )
else:
    key_findings.append(
        "No predictors met the alpha = 0.05 significance threshold under the selected tests."
    )

if "body_weight" in analysis_dataframe.columns:
    body_skewness = analysis_dataframe["body_weight"].skew()
    if abs(body_skewness) > 1:
        key_findings.append(
            "body_weight is strongly skewed; logarithmic transformation is recommended."
        )

top_key_feature_drivers = [
    record["feature"]
    for record in target_correlations[:10]
    if record["correlation"] is not None
]

if not top_key_feature_drivers:
    top_key_feature_drivers = significant_predictors[:10]

predictive_modeling_blueprint = {
    "target_definition": {
        "target_column": target_column,
        "target_type": str(analysis_dataframe[target_column].dtype),
        "target_missing_values_excluded": int(dataframe[target_column].isna().sum())
    },
    "problem_type": "Regression",
    "recommended_algorithms": [
        "Regularized linear regression with log-transformed skewed predictors",
        "Random Forest Regressor",
        "Gradient Boosting Regressor",
        "HistGradientBoostingRegressor",
        "Elastic Net regression"
    ],
    "feature_selection_strategy": [
        "Use domain-informed features including brain_body_ratio and log_body_weight.",
        "Rank features using cross-validated permutation importance and mutual information.",
        "Inspect pairwise correlation and remove redundant raw/derived variables when appropriate.",
        "Retain statistically significant predictors as candidates, but validate selection within each training fold."
    ],
    "validation_strategy": [
        "Use repeated K-fold cross-validation because the dataset contains only 62 rows.",
        "Use 5 folds where sample size permits, with repeated random seeds.",
        "Report MAE, RMSE, R-squared, and cross-validation confidence intervals.",
        "Keep a final untouched holdout only if sufficient observations remain."
    ],
    "preprocessing_steps": [
        "Replace textual missing markers with NaN.",
        "Apply type-safe numeric imputation using median for highly skewed columns and mean otherwise.",
        "Apply mode or Unknown imputation to categorical columns.",
        "Use log1p transformation for heavily right-skewed positive variables.",
        "Use RobustScaler for models sensitive to outliers.",
        "Encode categorical variables using one-hot encoding."
    ],
    "overfitting_risk_mitigation": [
        "Avoid complex models without cross-validation because of the very small sample size.",
        "Tune hyperparameters conservatively.",
        "Use regularization and shallow tree depth.",
        "Perform all imputation, transformation, and feature selection inside cross-validation pipelines.",
        "Use bootstrap uncertainty intervals and inspect residual stability."
    ],
    "overall_executive_modeling_strategy": (
        "Begin with a regularized linear regression baseline using log-transformed body and brain "
        "measurements plus ecological indices. Compare it with constrained tree ensembles using "
        "repeated cross-validation. Prefer the simplest model with stable out-of-sample error, "
        "interpretable feature effects, and robust performance under extreme-value sensitivity."
    )
}

# ---------------------------------------------------------------------
# Dataset overview and final metrics dictionary
# ---------------------------------------------------------------------

dataset_overview = {
    "shape": {
        "rows": int(raw_dataframe.shape[0]),
        "columns": int(raw_dataframe.shape[1])
    },
    "target_column": target_column,
    "profile_dimensions": metadata_profile.get("dimensions"),
    "profile_schema": profile_schema,
    "column_summary": column_summary(dataframe),
    "numeric_columns_inferred_from_text": converted_numeric_columns,
    "original_missing_counts": {
        str(column): int(count)
        for column, count in original_missing_counts.items()
    }
}

correlation_text_lines = [
    "Pearson correlation matrix:",
    correlation_matrix.round(4).to_string()
]
correlation_text = "\n".join(correlation_text_lines)

correlation_analysis = {
    "top_positive_correlations": positive_correlations,
    "top_negative_correlations": negative_correlations,
    "target_correlations": target_correlations,
    "correlation_matrix_plain_text": correlation_text
}

extracted_insights = {
    "key_findings": key_findings,
    "data_quality_issues": data_quality_issues,
    "data_quality_issues_caveats": [
        "The dataset is small, so p-values and model estimates may be unstable.",
        "IQR outlier flags indicate unusual observations, not necessarily data errors.",
        "Correlation does not establish causation.",
        "Feature selection and preprocessing must occur inside cross-validation to avoid leakage."
    ],
    "top_key_feature_drivers": top_key_feature_drivers
}

metrics_dict = {
    "dataset_overview": dataset_overview,
    "imputation_summary": imputation_summary,
    "outlier_analysis": outlier_analysis,
    "engineered_features": engineered_feature_records,
    "correlation_analysis": correlation_analysis,
    "statistical_hypothesis_tests": {
        "tests_by_feature": statistical_hypothesis_tests,
        "significant_predictors": significant_predictors,
        "alpha": alpha
    },
    "extracted_insights": extracted_insights,
    "predictive_modeling_blueprint": predictive_modeling_blueprint
}

with open(METRICS_FILEPATH, "w", encoding="utf-8") as metrics_file:
    json.dump(make_json_safe(metrics_dict), metrics_file, indent=2)

print("EDA completed successfully.")
print("Target column: {}".format(target_column))
print("Correlation heatmap saved to correlation_matrix.png")
print("Target interaction plot saved to target_interactions.png")
print("Comprehensive metrics saved to metrics.json")
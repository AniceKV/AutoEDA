DATA_FILEPATH = r'C:\Users\Anish Kumar Verma\PycharmProjects\AutoEDA\test_data\zoo_data-1.csv'

import os
import re
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")

PROFILE_PATH = "metadata_profile.json"

if "DATA_FILEPATH" not in globals():
    raise NameError("DATA_FILEPATH must be defined as a global variable.")

if not os.path.exists(DATA_FILEPATH):
    raise FileNotFoundError("Dataset not found at: " + str(DATA_FILEPATH))

if not os.path.exists(PROFILE_PATH):
    raise FileNotFoundError("metadata_profile.json was not found in the active directory.")

with open(PROFILE_PATH, "r", encoding="utf-8") as profile_file:
    profile = json.load(profile_file)

df = pd.read_csv(DATA_FILEPATH)
original_df = df.copy()
n_rows, n_columns = df.shape

def json_safe(value):
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        if np.isnan(value) or np.isinf(value):
            return None
        return float(value)
    if isinstance(value, np.ndarray):
        return [json_safe(v) for v in value.tolist()]
    if pd.isna(value):
        return None
    return value

def parse_profile_skewness(column_name):
    schema_item = profile.get("schema", {}).get(column_name, {})
    text = str(schema_item.get("key_metric", ""))
    skew_match = re.search(r"Skewed\s*\((-?\d+(?:\.\d+)?)\)", text)
    if skew_match:
        return float(skew_match.group(1))
    return None

def profile_type(column_name):
    schema_item = profile.get("schema", {}).get(column_name, {})
    return str(schema_item.get("type", ""))

def is_numeric_column(series):
    return pd.api.types.is_numeric_dtype(series)

def identify_target(dataframe):
    preferred_targets = [
        "target", "class_type", "class", "label", "survived",
        "species", "category", "outcome", "y"
    ]
    for candidate in preferred_targets:
        if candidate in dataframe.columns:
            return candidate

    excluded = {"animal_name", "name", "id", "index"}
    candidates = []
    for column in dataframe.columns:
        if column.lower() in excluded:
            continue
        cardinality = dataframe[column].nunique(dropna=True)
        if 2 <= cardinality <= 10:
            candidates.append((column, cardinality))

    if len(candidates) == 1:
        return candidates[0][0]

    return None

target_column = identify_target(df)

# ---------------------------------------------------------------------
# Smart type-safe imputation
# ---------------------------------------------------------------------
imputation_summary = {
    "rules": [
        "Numeric columns with absolute profile skewness greater than 1 use median imputation.",
        "Numeric columns with absolute profile skewness less than or equal to 1 use mean imputation.",
        "Categorical or string columns use mode imputation, with Unknown as a fallback.",
        "No imputation is performed when a column has no missing values."
    ],
    "columns": {}
}

for column in df.columns:
    missing_before = int(df[column].isna().sum())
    column_type = str(df[column].dtype)
    profile_skew = parse_profile_skewness(column)

    if missing_before == 0:
        imputation_summary["columns"][column] = {
            "dtype": column_type,
            "missing_before": 0,
            "missing_after": 0,
            "method": "none",
            "filled_value_or_statistic": None,
            "filled_count": 0,
            "profile_skewness": profile_skew
        }
        continue

    if is_numeric_column(df[column]):
        if profile_skew is not None and abs(profile_skew) > 1:
            fill_value = df[column].median()
            method = "median"
        else:
            fill_value = df[column].mean()
            method = "mean"

        df[column] = df[column].fillna(fill_value)
        fill_description = float(fill_value)
    else:
        mode_values = df[column].mode(dropna=True)
        if len(mode_values) > 0:
            fill_value = mode_values.iloc[0]
            method = "mode"
            df[column] = df[column].fillna(fill_value)
            fill_description = str(fill_value)
        else:
            fill_value = "Unknown"
            method = "placeholder"
            df[column] = df[column].fillna(fill_value)
            fill_description = fill_value

    imputation_summary["columns"][column] = {
        "dtype": column_type,
        "missing_before": missing_before,
        "missing_after": int(df[column].isna().sum()),
        "method": method,
        "filled_value_or_statistic": fill_description,
        "filled_count": missing_before,
        "profile_skewness": profile_skew
    }

# ---------------------------------------------------------------------
# Domain-specific feature engineering
# ---------------------------------------------------------------------
engineered_features = []

if {"hair", "feathers", "milk", "backbone", "breathes"}.issubset(df.columns):
    df["terrestrial_mammalian_score"] = (
        df["hair"] + df["milk"] + df["backbone"] + df["breathes"]
    )
    engineered_features.append({
        "feature_name": "terrestrial_mammalian_score",
        "formula_or_method": "hair + milk + backbone + breathes",
        "data_type": str(df["terrestrial_mammalian_score"].dtype),
        "rationale_purpose": "Summarizes traits associated with mammalian and terrestrial biological structure.",
        "correlation_with_target": None
    })

if {"aquatic", "fins", "airborne"}.issubset(df.columns):
    df["environmental_adaptation_score"] = (
        df["aquatic"] + df["fins"] + df["airborne"]
    )
    engineered_features.append({
        "feature_name": "environmental_adaptation_score",
        "formula_or_method": "aquatic + fins + airborne",
        "data_type": str(df["environmental_adaptation_score"].dtype),
        "rationale_purpose": "Captures adaptation to aquatic and aerial environments.",
        "correlation_with_target": None
    })

if {"legs", "tail", "backbone"}.issubset(df.columns):
    df["locomotion_structure_score"] = (
        df["legs"] / max(float(df["legs"].max()), 1.0)
        + df["tail"]
        + df["backbone"]
    )
    engineered_features.append({
        "feature_name": "locomotion_structure_score",
        "formula_or_method": "normalized legs + tail + backbone",
        "data_type": str(df["locomotion_structure_score"].dtype),
        "rationale_purpose": "Represents structural traits related to movement and body support.",
        "correlation_with_target": None
    })

# ---------------------------------------------------------------------
# Outlier profiling using IQR
# ---------------------------------------------------------------------
numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
outlier_analysis = {}

for column in numeric_columns:
    series = df[column].dropna()
    q1 = float(series.quantile(0.25))
    q3 = float(series.quantile(0.75))
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    outlier_mask = (df[column] < lower_bound) | (df[column] > upper_bound)
    outlier_count = int(outlier_mask.sum())
    outlier_percentage = float(outlier_count / max(len(df), 1) * 100)

    outlier_analysis[column] = {
        "Q1": q1,
        "Q3": q3,
        "IQR": float(iqr),
        "lower_bound": float(lower_bound),
        "upper_bound": float(upper_bound),
        "outlier_count": outlier_count,
        "outlier_percentage": outlier_percentage
    }

# ---------------------------------------------------------------------
# Target correlations
# ---------------------------------------------------------------------
correlation_matrix = df[numeric_columns].corr()
target_correlations = {}

if target_column is not None and target_column in correlation_matrix.columns:
    target_correlations = {
        column: float(correlation_matrix.loc[column, target_column])
        for column in correlation_matrix.columns
        if column != target_column and pd.notna(correlation_matrix.loc[column, target_column])
    }

for item in engineered_features:
    feature_name = item["feature_name"]
    if target_column is not None and target_column in df.columns:
        if is_numeric_column(df[target_column]) and feature_name in df.columns:
            item["correlation_with_target"] = (
                float(df[[feature_name, target_column]].corr().iloc[0, 1])
                if df[[feature_name, target_column]].corr().iloc[0, 1] == df[[feature_name, target_column]].corr().iloc[0, 1]
                else None
            )

correlation_pairs = []
for left_column in correlation_matrix.columns:
    for right_column in correlation_matrix.columns:
        if left_column >= right_column:
            continue
        value = correlation_matrix.loc[left_column, right_column]
        if pd.notna(value):
            correlation_pairs.append({
                "feature_1": left_column,
                "feature_2": right_column,
                "correlation": float(value)
            })

top_positive_correlations = sorted(
    correlation_pairs, key=lambda item: item["correlation"], reverse=True
)[:10]

top_negative_correlations = sorted(
    correlation_pairs, key=lambda item: item["correlation"]
)[:10]

correlation_text = correlation_matrix.round(4).to_string()

# ---------------------------------------------------------------------
# Statistical hypothesis testing
# ---------------------------------------------------------------------
statistical_hypothesis_tests = {}
significant_predictors = []

if target_column is not None and target_column in df.columns:
    target_values = df[target_column].dropna()
    target_cardinality = target_values.nunique()

    predictor_columns = [
        column for column in df.columns
        if column != target_column and column not in {"animal_name"}
    ]

    for feature in predictor_columns:
        valid_data = df[[feature, target_column]].dropna()
        if valid_data.empty or valid_data[feature].nunique() < 2:
            continue

        try:
            if target_cardinality == 2:
                contingency_table = pd.crosstab(valid_data[feature], valid_data[target_column])
                if contingency_table.shape[0] >= 2 and contingency_table.shape[1] >= 2:
                    chi2, p_val, dof, expected = stats.chi2_contingency(contingency_table)
                    is_significant = bool(p_val < 0.05)
                    interpretation = (
                        "Evidence of an association with the target at alpha=0.05."
                        if is_significant
                        else
                        "No statistically significant association with the target at alpha=0.05."
                    )
                    statistical_hypothesis_tests[feature] = {
                        "test_name": "Chi-Square test of independence",
                        "statistic_value": float(chi2),
                        "p_value": float(p_val),
                        "degrees_of_freedom": int(dof),
                        "is_statistically_significant": is_significant,
                        "interpretation_summary": interpretation
                    }
                    if is_significant:
                        significant_predictors.append(feature)
            elif is_numeric_column(df[feature]):
                groups = [
                    group[feature].values
                    for _, group in valid_data.groupby(target_column)
                    if len(group) >= 2
                ]
                if len(groups) >= 2:
                    if len(groups) == 2:
                        statistic_value, p_val = stats.ttest_ind(
                            groups[0], groups[1], equal_var=False
                        )
                        test_name = "Welch independent-samples t-test"
                        degrees_of_freedom = None
                    else:
                        statistic_value, p_val = stats.f_oneway(*groups)
                        test_name = "One-way ANOVA"
                        degrees_of_freedom = None

                    is_significant = bool(p_val < 0.05)
                    interpretation = (
                        "Evidence that feature distributions differ across target groups at alpha=0.05."
                        if is_significant
                        else
                        "No statistically significant distributional difference across target groups at alpha=0.05."
                    )
                    statistical_hypothesis_tests[feature] = {
                        "test_name": test_name,
                        "statistic_value": float(statistic_value),
                        "p_value": float(p_val),
                        "degrees_of_freedom": degrees_of_freedom,
                        "is_statistically_significant": is_significant,
                        "interpretation_summary": interpretation
                    }
                    if is_significant:
                        significant_predictors.append(feature)
        except Exception as test_error:
            statistical_hypothesis_tests[feature] = {
                "test_name": "Test unavailable",
                "statistic_value": None,
                "p_value": None,
                "degrees_of_freedom": None,
                "is_statistically_significant": False,
                "interpretation_summary": "Testing failed: " + str(test_error)
            }
else:
    statistical_hypothesis_tests = {
        "dataset_level_note": {
            "test_name": "Not applicable",
            "statistic_value": None,
            "p_value": None,
            "degrees_of_freedom": None,
            "is_statistically_significant": False,
            "interpretation_summary": (
                "No explicit target column was detected. The dataset appears to contain "
                "animal attributes without a supervised outcome column."
            )
        }
    }

# ---------------------------------------------------------------------
# Visualizations
# ---------------------------------------------------------------------
plt.figure(figsize=(14, 11))
sns.heatmap(
    correlation_matrix,
    cmap="coolwarm",
    center=0,
    annot=True,
    fmt=".2f",
    linewidths=0.4,
    square=True
)
plt.title("Pearson Correlation Matrix")
plt.tight_layout()
plt.savefig("correlation_matrix.png", dpi=200, bbox_inches="tight")
plt.close()

interaction_feature = "legs" if "legs" in df.columns else numeric_columns[0]
interaction_segment = target_column

if interaction_segment is None:
    interaction_segment = "backbone" if "backbone" in df.columns else None

plt.figure(figsize=(9, 6))
if interaction_segment is not None and interaction_segment in df.columns:
    sns.boxplot(data=df, x=interaction_segment, y=interaction_feature)
    plt.title(
        interaction_feature + " distribution segmented by " + interaction_segment
        + (" (proxy segment; no explicit target found)" if target_column is None else "")
    )
    plt.xlabel(interaction_segment)
    plt.ylabel(interaction_feature)
else:
    sns.histplot(data=df, x=interaction_feature, bins=12, kde=True)
    plt.title(interaction_feature + " distribution")
    plt.xlabel(interaction_feature)
    plt.ylabel("Count")

plt.tight_layout()
plt.savefig("target_interactions.png", dpi=200, bbox_inches="tight")
plt.close()

# ---------------------------------------------------------------------
# Extracted insights
# ---------------------------------------------------------------------
key_findings = []
data_quality_issues = []
top_key_feature_drivers = []

if target_column is None:
    key_findings.append(
        "No explicit supervised target was detected; statistical target-based inference is unavailable."
    )
    data_quality_issues.append(
        "The dataset contains animal attributes but no clear outcome or class label column."
    )
else:
    key_findings.append("Detected target column: " + target_column)
    if significant_predictors:
        key_findings.append(
            "Statistically significant predictors were identified using alpha=0.05."
        )

if len(df) != 101 or len(df.columns) != 17:
    data_quality_issues.append(
        "Observed dimensions differ from the supplied metadata profile."
    )

if len(df["animal_name"].unique()) < len(df) if "animal_name" in df.columns else False:
    data_quality_issues.append(
        "animal_name is not fully unique and should not automatically be treated as a primary key."
    )

if target_correlations:
    top_key_feature_drivers = [
        item[0] for item in sorted(
            target_correlations.items(),
            key=lambda pair: abs(pair[1]),
            reverse=True
        )[:10]
    ]
else:
    top_key_feature_drivers = [
        item["feature_1"] + " and " + item["feature_2"]
        for item in top_positive_correlations[:5]
    ]

key_findings.append(
    "The feature space is dominated by binary biological attributes, making tree-based "
    "classifiers and interpretable rule-based models appropriate."
)
key_findings.append(
    "Highly skewed binary traits are expected because several biological characteristics "
    "are uncommon in the animal population."
)

# ---------------------------------------------------------------------
# Predictive modeling blueprint
# ---------------------------------------------------------------------
if target_column is not None:
    target_problem_type = (
        "binary classification"
        if df[target_column].nunique(dropna=True) == 2
        else "multiclass classification"
    )
else:
    target_problem_type = "Undefined until a target label is supplied"

predictive_modeling_blueprint = {
    "target_definition": (
        target_column
        if target_column is not None
        else "No explicit target detected; provide a target such as class_type."
    ),
    "problem_type": target_problem_type,
    "recommended_algorithms": [
        "Regularized logistic regression as an interpretable baseline",
        "Decision tree with constrained depth",
        "Random forest with class weighting where appropriate",
        "Gradient boosting or XGBoost-style boosting for nonlinear interactions",
        "Linear or kernel SVM for small-sample comparison"
    ],
    "feature_selection_strategy": [
        "Exclude animal_name unless entity identity is explicitly meaningful.",
        "Use domain-engineered scores alongside original attributes.",
        "Rank features using cross-validated permutation importance and mutual information.",
        "Remove redundant features only after checking model stability and domain interpretability.",
        "Do not select features using the full dataset before cross-validation."
    ],
    "validation_strategy": [
        "Use stratified k-fold cross-validation for classification.",
        "Use repeated stratified cross-validation because the sample size is small.",
        "Report balanced accuracy, macro F1, per-class recall, ROC-AUC where applicable, and a confusion matrix.",
        "Reserve a final holdout set only if enough observations remain after training."
    ],
    "preprocessing_steps": [
        "Apply type-safe imputation inside each training fold.",
        "Treat binary biological indicators as numeric or categorical consistently.",
        "Scale continuous engineered scores for linear models and SVMs.",
        "Avoid scaling requirements for tree-based models.",
        "Encode categorical variables with one-hot encoding if additional categorical predictors are added."
    ],
    "overfitting_risk_mitigation": [
        "Use shallow trees, minimum leaf sizes, regularization, and early stopping.",
        "Avoid high-cardinality animal_name encoding.",
        "Use nested or repeated cross-validation for hyperparameter selection.",
        "Prefer simple models when performance is statistically indistinguishable.",
        "Monitor train-validation performance gaps and feature-importance instability."
    ],
    "overall_executive_modeling_strategy_summary": (
        "Establish a regularized interpretable baseline, then compare constrained tree ensembles "
        "using leakage-safe cross-validation. Emphasize biological interpretability, robust metrics, "
        "and uncertainty because the dataset is small and largely binary. A supervised target must be "
        "confirmed before production modeling."
    )
}

if target_column is not None and target_column in correlation_matrix.columns:
    target_corr_items = sorted(
        target_correlations.items(), key=lambda pair: pair[1], reverse=True
    )
    top_positive_target = [
        {"feature": feature, "correlation": correlation}
        for feature, correlation in target_corr_items[:10]
    ]
    top_negative_target = [
        {"feature": feature, "correlation": correlation}
        for feature, correlation in sorted(
            target_correlations.items(), key=lambda pair: pair[1]
        )[:10]
    ]
else:
    top_positive_target = []
    top_negative_target = []

metrics_dict = {
    "dataset_overview": {
        "dataset_path": str(DATA_FILEPATH),
        "shape": {"rows": int(n_rows), "columns": int(n_columns)},
        "target_column": target_column,
        "profile_dimensions": profile.get("dimensions"),
        "column_summary": {
            column: {
                "dtype": str(df[column].dtype),
                "missing_count": int(df[column].isna().sum()),
                "cardinality": int(df[column].nunique(dropna=True)),
                "profile_type": profile_type(column),
                "profile_key_metric": profile.get("schema", {}).get(column, {}).get("key_metric")
            }
            for column in df.columns
        }
    },
    "imputation_summary": imputation_summary,
    "outlier_analysis": outlier_analysis,
    "engineered_features": engineered_features,
    "correlation_analysis": {
        "top_positive_correlations": top_positive_correlations,
        "top_negative_correlations": top_negative_correlations,
        "target_correlations": {
            "top_positive": top_positive_target,
            "top_negative": top_negative_target,
            "all_target_correlations": target_correlations
        },
        "correlation_matrix_plain_text": correlation_text
    },
    "statistical_hypothesis_tests": {
        "tests_by_feature": statistical_hypothesis_tests,
        "significant_predictors": sorted(set(significant_predictors)),
        "significance_threshold": 0.05
    },
    "extracted_insights": {
        "key_findings": key_findings,
        "data_quality_issues": data_quality_issues,
        "key_feature_drivers": top_key_feature_drivers
    },
    "predictive_modeling_blueprint": predictive_modeling_blueprint,
    "generated_visualizations": [
        "correlation_matrix.png",
        "target_interactions.png"
    ]
}

with open("metrics.json", "w", encoding="utf-8") as metrics_file:
    json.dump(json_safe(metrics_dict), metrics_file, indent=2)

print("EDA completed successfully.")
print("Dataset shape:", df.shape)
print("Detected target:", target_column if target_column is not None else "None")
print("Numeric columns analyzed for outliers:", len(numeric_columns))
print("Engineered features created:", len(engineered_features))
print("Significant predictors:", len(significant_predictors))
print("Saved visualization: correlation_matrix.png")
print("Saved visualization: target_interactions.png")
print("Saved metrics: metrics.json")
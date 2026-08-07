import os
import sys
import json
import pytest
import shutil

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from autoeda_core.summary_generator import (
    scan_and_load_files,
    _caption_for_image,
    _fallback_importance_blurb,
    generate_column_importance_blurbs,
    generate_template_summary,
    extract_dataset_name,
    create_summary,
)


def test_caption_for_image():
    assert _caption_for_image("correlation_matrix.png") == "Pearson correlation heatmap across numeric features."
    assert _caption_for_image("categorical_association_matrix.jpg") == "Cramer's V association heatmap across categorical features."
    assert _caption_for_image("pairplot.png") == "Pairwise scatter/distribution grid across key numeric features, colored by target."
    assert _caption_for_image("target_interactions.svg") == "Overview of how the top features interact with the target variable."
    assert "Relationship between `math score` and `reading score`." in _caption_for_image("bivariate_math_score_vs_reading_score.png")
    assert "Distribution of `age`." in _caption_for_image("dist_age.png")
    assert _caption_for_image("unknown_plot.png") == "Generated analysis artifact."


def test_fallback_importance_blurb():
    blurb1 = _fallback_importance_blurb("ANOVA", "Large effect", "math_score", "passed_exam")
    assert "`math_score` strongly differentiates `passed_exam` outcomes (ANOVA, Large effect)." in blurb1

    blurb2 = _fallback_importance_blurb("Chi-Square", "Medium effect", "gender", "target")
    assert "`gender` moderately differentiates `target` outcomes (Chi-Square, Medium effect)." in blurb2

    blurb3 = _fallback_importance_blurb("T-Test", "Small effect", "age", "target")
    assert "`age` weakly differentiates `target` outcomes (T-Test, Small effect)." in blurb3


def test_scan_and_load_files(tmp_path):
    target_dir = tmp_path / "test_run"
    target_dir.mkdir()

    # Create dummy files
    (target_dir / "generated_analysis.py").write_text("print('hello')", encoding="utf-8")
    (target_dir / "summary_report.md").write_text("old summary", encoding="utf-8")
    (target_dir / "metrics.json").write_text(json.dumps({"test": 123}), encoding="utf-8")
    (target_dir / "plot.png").write_bytes(b"dummy_png_bytes")
    (target_dir / "notes.txt").write_text("some notes", encoding="utf-8")

    data = scan_and_load_files(str(target_dir))

    assert data["target_dir"] == os.path.abspath(str(target_dir))
    assert "generated_analysis.py" not in data["files_scanned"]
    assert "summary_report.md" not in data["files_scanned"]
    assert "metrics.json" in data["files_scanned"]
    assert "plot.png" in data["files_scanned"]
    assert "notes.txt" in data["files_scanned"]

    assert data["contents"]["metrics.json"] == {"test": 123}
    assert data["contents"]["plot.png"]["type"] == "image_visualization"
    assert data["contents"]["notes.txt"] == "some notes"


def test_scan_and_load_files_missing_dir():
    with pytest.raises(FileNotFoundError):
        scan_and_load_files("./non_existent_directory_12345")


def test_generate_column_importance_blurbs_no_llm():
    ranked_details = [
        {
            "feature": "reading_score",
            "test": "Pearson Correlation",
            "effect_size": 0.95,
            "effect_size_label": "Large effect",
            "p_value": 1.2e-5,
        },
        {
            "feature": "lunch",
            "test": "Chi-Square",
            "effect_size": 0.25,
            "effect_size_label": "Medium effect",
            "p_value": 0.001,
        },
    ]
    blurbs = generate_column_importance_blurbs(
        ranked_details, target_col="math_score", dataset_name="StudentPerformance", use_llm=False
    )
    assert len(blurbs) == 2
    assert "reading_score" in blurbs
    assert "strongly differentiates" in blurbs["reading_score"]
    assert "lunch" in blurbs
    assert "moderately differentiates" in blurbs["lunch"]


def test_generate_template_summary():
    mock_data = {
        "target_dir": "C:/fake/path",
        "files_scanned": ["metrics.json", "metadata_profile.json", "correlation_matrix.png"],
        "excluded_files": ["generated_analysis.py"],
        "contents": {
            "metrics.json": {
                "dataset_overview": {
                    "shape": {"rows": 1000, "columns": 5},
                    "target_column": "math_score",
                    "dataset_path": "student_data.csv",
                },
                "imputation_summary": {
                    "rules_applied": ["Numeric NaN -> Median"],
                    "columns": {
                        "reading_score": {
                            "missing_before": 10,
                            "method": "median",
                            "fill_value": 69.0,
                        }
                    },
                },
                "outlier_analysis": {
                    "writing_score": {
                        "outlier_count": 5,
                        "outlier_percentage": 0.5,
                        "lower_bound": 25.0,
                        "upper_bound": 100.0,
                    }
                },
                "engineered_features": [
                    {
                        "feature_name": "total_score",
                        "formula": "math_score + reading_score",
                        "rationale": "Composite metric",
                    }
                ],
                "statistical_hypothesis_tests": {
                    "ranked_significant_details": [
                        {
                            "feature": "reading_score",
                            "test": "Pearson Correlation",
                            "effect_size": 0.95,
                            "effect_size_label": "Large effect",
                            "p_value": 1.2e-5,
                        }
                    ],
                    "target_col": "math_score",
                },
                "correlation_analysis": {
                    "high_correlation_pairs": [
                        {
                            "feature_1": "reading_score",
                            "feature_2": "writing_score",
                            "correlation": 0.95,
                            "interpretation": "Extremely high collinearity",
                        }
                    ],
                    "cross_type_redundant_pairs": [],
                },
                "categorical_associations": [
                    {"feature_1": "gender", "feature_2": "lunch", "cramers_v": 0.12}
                ],
                "predictive_modeling_blueprint": {
                    "recommended_models": ["RandomForestRegressor", "LightGBM"],
                    "evaluation_metric": "RMSE",
                },
            },
            "metadata_profile.json": {
                "dataset_name": "student_data",
                "dimensions": {"rows": 1000, "columns": 5},
                "missing_values_summary": {"reading_score": {"missing_count": 10, "missing_pct": 1.0}},
                "columns": [
                    {
                        "column": "math_score",
                        "dtype": "int64",
                        "missing_pct": 0.0,
                        "unique_pct": 8.5,
                        "mean": 66.08,
                        "median": 66.0,
                        "std": 15.16,
                        "skew": -0.25,
                        "kurtosis": 0.27,
                    }
                ],
            },
            "correlation_matrix.png": {"type": "image_visualization", "size_kb": 50.5},
        },
    }

    report = generate_template_summary(mock_data, use_llm_for_importance=False)

    assert "# Executive EDA & Dataset Summary Report" in report
    assert "`1000` rows x `5` columns" in report
    assert "math_score" in report
    assert "Full Column Statistics" in report
    assert "Data Imputation & Preprocessing" in report
    assert "Numeric NaN -> Median" in report
    assert "Outlier Analysis (IQR Method)" in report
    assert "Derived Domain Attributes & Composite Metrics" in report
    assert "total_score" in report
    assert "Statistical Hypothesis Testing & Key Predictors" in report
    assert "Pearson correlation heatmap across numeric features." in report
    assert "Redundancy & Multicollinearity Analysis" in report
    assert "Predictive Modeling Strategy Blueprint" in report


def test_extract_dataset_name(tmp_path):
    target_dir = tmp_path / "test_extract"
    target_dir.mkdir()

    script = target_dir / "generated_analysis.py"
    script.write_text("DATA_FILEPATH = r'C:\\data\\housing_prices.csv'\n", encoding="utf-8")

    ds_name = extract_dataset_name(str(target_dir))
    assert ds_name == "housing_prices"


def test_create_summary(tmp_path):
    sandbox_dir = tmp_path / "sandbox_run"
    sandbox_dir.mkdir()

    (sandbox_dir / "metrics.json").write_text(
        json.dumps({
            "dataset_overview": {"shape": {"rows": 50, "columns": 2}, "target_column": "y"}
        }),
        encoding="utf-8"
    )

    report_md = create_summary(
        directory_path=str(sandbox_dir),
        output_filename="summary_report.md",
        use_llm=False,
        use_llm_for_importance=False,
        dataset_name="test_dataset"
    )

    assert "# Executive EDA & Dataset Summary Report" in report_md
    assert os.path.exists(sandbox_dir / "summary_report.md")
    assert os.path.exists(os.path.join("EDA", "test_dataset", "summary_report.md"))

    # Cleanup created test output in EDA directory if created
    eda_test_dir = os.path.join("EDA", "test_dataset")
    if os.path.exists(eda_test_dir):
        shutil.rmtree(eda_test_dir)

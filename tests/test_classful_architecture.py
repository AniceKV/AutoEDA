import os
import json
import pytest
import pandas as pd
import numpy as np

from autoeda_core import (
    AutoEDAEngine,
    DataProfiler,
    StatefulDataStore,
    DataImputer,
    OutlierAnalyzer,
    FeatureEngineer,
    HypothesisTester,
    DataVisualizer,
    PredictiveBlueprinter,
    ReportValidator,
    MetricsCompiler,
    ExecutiveSummaryGenerator,
    HTMLReportCompiler,
    AutoEDAAgent,
    CodeExecutorSandbox,
    run_and_save_profile,
    run_statistical_hypothesis_tests,
    plot_feature_distributions,
    create_summary,
    generate_html_report,
)


@pytest.fixture
def sample_df():
    np.random.seed(42)
    return pd.DataFrame({
        "age": [22, 38, 26, 35, 35, np.nan, 54, 2, 27, 14],
        "fare": [7.25, 71.28, 7.92, 53.1, 8.05, 8.46, 51.86, 21.07, 11.13, 30.07],
        "sex": ["male", "female", "female", "female", "male", "male", "male", "male", "female", "female"],
        "survived": [0, 1, 1, 1, 0, 0, 0, 0, 1, 1]
    })


@pytest.fixture
def temp_csv_path(tmp_path, sample_df):
    csv_file = tmp_path / "iris_sample.csv"
    sample_df.to_csv(csv_file, index=False)
    return str(csv_file)


def test_data_profiler(sample_df, temp_csv_path, tmp_path):
    profiler = DataProfiler()
    assert profiler.is_non_distributional_column("passengerid") is True
    assert profiler.is_non_distributional_column("age") is False

    stats = profiler.calculate_column_stats(sample_df)
    assert len(stats) == 4
    col_names = [s["column"] for s in stats]
    assert "age" in col_names and "fare" in col_names

    output_dir = str(tmp_path / "profile_out")
    llm_context = profiler.run_and_save_profile(temp_csv_path, output_dir)
    assert "dimensions" in llm_context
    assert os.path.exists(os.path.join(output_dir, "metadata_profile.json"))


def test_data_imputer(sample_df):
    imputer = DataImputer()
    df_imputed, report = imputer.impute_missing_data(sample_df)
    assert df_imputed["age"].isnull().sum() == 0
    assert "columns" in report


def test_outlier_analyzer(sample_df):
    analyzer = OutlierAnalyzer()
    df_out, report = analyzer.detect_and_handle_outliers(sample_df, columns=["fare"], action="profile")
    assert "fare" in report
    assert "outlier_count" in report["fare"]


def test_feature_engineer(sample_df):
    engineer = FeatureEngineer()
    df_feat, summary = engineer.engineer_features(
        sample_df,
        feature_specs=[{"name": "log_fare", "type": "log1p", "source_col": "fare", "rationale": "Log fare"}],
        target_col="survived"
    )
    assert "log_fare" in df_feat.columns
    assert len(summary) == 1
    assert summary[0]["feature_name"] == "log_fare"


def test_hypothesis_tester(sample_df):
    tester = HypothesisTester()
    res = tester.run_statistical_hypothesis_tests(sample_df, target_col="survived")
    assert "target_col" in res
    assert res["target_col"] == "survived"
    assert "significant_predictors" in res
    assert "ranked_significant_details" in res


def test_data_visualizer(sample_df):
    visualizer = DataVisualizer()
    corr_res = visualizer.plot_correlation_matrix(sample_df)
    assert "high_correlation_pairs" in corr_res

    dist_res = visualizer.plot_feature_distributions(sample_df)
    assert "visual_distributions" in dist_res

    biv_res = visualizer.plot_semantic_bivariate_relationships(sample_df)
    assert "bivariate_data" in biv_res

    pair_res = visualizer.plot_pairplot(sample_df)
    assert "pairplot_matrix" in pair_res


def test_predictive_blueprinter(sample_df):
    blueprinter = PredictiveBlueprinter()
    bp = blueprinter.generate_predictive_blueprint(sample_df, target_col="survived")
    assert bp["target_definition"] == "survived"
    assert bp["problem_type"] == "Binary Classification"


def test_report_validator_and_metrics_compiler(sample_df, temp_csv_path, tmp_path):
    output_dir = str(tmp_path / "metrics_out")
    compiler = MetricsCompiler()
    metrics_path = compiler.compile_and_save_metrics(
        df=sample_df,
        dataset_path=temp_csv_path,
        target_col="survived",
        output_dir=output_dir
    )
    assert os.path.exists(metrics_path)

    with open(metrics_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert data["dataset_overview"]["target_column"] == "survived"


def test_executive_summary_generator(tmp_path, temp_csv_path, sample_df):
    workspace = str(tmp_path / "summary_ws")
    os.makedirs(workspace, exist_ok=True)
    profiler = DataProfiler()
    profiler.run_and_save_profile(temp_csv_path, workspace)

    compiler = MetricsCompiler()
    compiler.compile_and_save_metrics(sample_df, temp_csv_path, target_col="survived", output_dir=workspace)

    gen = ExecutiveSummaryGenerator()
    report_md = gen.create_summary(directory_path=workspace, use_llm=False)
    assert "# Executive EDA & Dataset Summary Report" in report_md


def test_html_report_compiler(tmp_path, temp_csv_path, sample_df):
    workspace = str(tmp_path / "html_ws")
    os.makedirs(workspace, exist_ok=True)
    profiler = DataProfiler()
    profiler.run_and_save_profile(temp_csv_path, workspace)

    compiler = MetricsCompiler()
    compiler.compile_and_save_metrics(sample_df, temp_csv_path, target_col="survived", output_dir=workspace)

    html_compiler = HTMLReportCompiler()
    html_out = html_compiler.generate_html_report(workspace_dir=workspace)
    assert "<title>AutoEDA - Interactive Profile Report" in html_out


def test_auto_eda_engine_facade(temp_csv_path, tmp_path):
    engine = AutoEDAEngine()
    workspace = str(tmp_path / "engine_ws")
    llm_context = engine.profile_file(temp_csv_path, output_dir=workspace)
    assert "dimensions" in llm_context
    assert os.path.exists(os.path.join(workspace, "metadata_profile.json"))


def test_backward_compatibility_wrappers(sample_df, temp_csv_path, tmp_path):
    out_dir = str(tmp_path / "compat_out")
    prof = run_and_save_profile(temp_csv_path, out_dir)
    assert "dimensions" in prof

    hyp = run_statistical_hypothesis_tests(sample_df, target_col="survived")
    assert hyp["target_col"] == "survived"

    dist = plot_feature_distributions(sample_df)
    assert "visual_distributions" in dist



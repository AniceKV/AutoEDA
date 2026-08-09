"""
AutoEDA Core Engine Package
Classful Architecture & Unified Engine Facade
"""

__version__ = "0.1.1"

from typing import Dict, Any, Optional, List

from .profiler import DataProfiler, is_non_distributional_column, calculate_column_stats, run_and_save_profile
from .tools import (
    StatefulDataStore,
    DataImputer,
    OutlierAnalyzer,
    FeatureEngineer,
    HypothesisTester,
    DataVisualizer,
    PredictiveBlueprinter,
    ReportValidator,
    MetricsCompiler,
    impute_missing_data,
    detect_and_handle_outliers,
    engineer_features,
    run_statistical_hypothesis_tests,
    plot_correlation_matrix,
    plot_target_interaction,
    plot_feature_distributions,
    plot_semantic_bivariate_relationships,
    infer_llm_bivariate_pairs,
    compute_bivariate_union,
    plot_pairplot,
    generate_predictive_blueprint,
    validate_report_consistency,
    compile_and_save_metrics,
)
from .executor import CodeExecutorSandbox, execute_code
from .summary_generator import (
    ExecutiveSummaryGenerator,
    scan_and_load_files,
    generate_column_importance_blurbs,
    generate_template_summary,
    generate_llm_summary,
    extract_dataset_name,
    create_summary,
)
from .html_report_generator import (
    HTMLReportCompiler,
    compute_alerts,
    build_variable_chart,
    render_markdown_to_html,
    generate_html_report,
)
from .agent_loop import (
    AutoEDAAgent,
    parse_llm_json_plan,
    validate_tool_plan,
    run_tool_based_eda,
)


class AutoEDAEngine:
    """
    Unified High-Level Engine Facade for AutoEDA.
    Provides single-point access to profiling, agent execution, statistical analysis,
    summary compilation, and interactive HTML report generation.
    """
    def __init__(
        self,
        profiler: Optional[DataProfiler] = None,
        agent: Optional[AutoEDAAgent] = None,
        summary_generator: Optional[ExecutiveSummaryGenerator] = None,
        html_compiler: Optional[HTMLReportCompiler] = None,
    ):
        self.profiler = profiler or DataProfiler()
        self.agent = agent or AutoEDAAgent()
        self.summary_generator = summary_generator or ExecutiveSummaryGenerator()
        self.html_compiler = html_compiler or HTMLReportCompiler()

    def analyze(
        self,
        data_path: str,
        user_request: str = "Perform complete exploratory data analysis.",
        workspace_dir: str = "./sandbox_run",
        generate_summary: bool = True,
        conversation_history: Optional[List[Dict[str, Any]]] = None,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
        answer_fn: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Runs full end-to-end tool-based EDA agent analysis on the target dataset."""
        if answer_fn is None:
            answer_fn = lambda q: "infer it yourself"
            
        return self.agent.run_tool_based_eda(
            data_path=data_path,
            user_request=user_request,
            workspace_dir=workspace_dir,
            generate_summary=generate_summary,
            conversation_history=conversation_history,
            api_key=api_key,
            model_name=model_name,
            answer_fn=answer_fn,
        )

    def profile_file(self, data_path: str, output_dir: str = "./sandbox_run") -> Dict[str, Any]:
        """Profiles dataset and saves metadata JSON."""
        return self.profiler.run_and_save_profile(data_path, output_dir)

    def compile_html(self, workspace_dir: str = "./sandbox_run", output_path: Optional[str] = None) -> str:
        """Compiles interactive HTML report from workspace artifacts."""
        return self.html_compiler.generate_html_report(workspace_dir=workspace_dir, output_path=output_path)


__all__ = [
    # Unified Facade Engine
    "AutoEDAEngine",
    "__version__",
    # Core Classful Components
    "DataProfiler",
    "StatefulDataStore",
    "DataImputer",
    "OutlierAnalyzer",
    "FeatureEngineer",
    "HypothesisTester",
    "DataVisualizer",
    "PredictiveBlueprinter",
    "ReportValidator",
    "MetricsCompiler",
    "ExecutiveSummaryGenerator",
    "HTMLReportCompiler",
    "AutoEDAAgent",
    "CodeExecutorSandbox",
    # Delegate Functions (Backward Compatibility)
    "run_tool_based_eda",
    "calculate_column_stats",
    "run_and_save_profile",
    "is_non_distributional_column",
    "impute_missing_data",
    "detect_and_handle_outliers",
    "engineer_features",
    "run_statistical_hypothesis_tests",
    "plot_correlation_matrix",
    "plot_target_interaction",
    "plot_feature_distributions",
    "plot_semantic_bivariate_relationships",
    "infer_llm_bivariate_pairs",
    "compute_bivariate_union",
    "plot_pairplot",
    "generate_predictive_blueprint",
    "validate_report_consistency",
    "compile_and_save_metrics",
    "execute_code",
    "scan_and_load_files",
    "generate_column_importance_blurbs",
    "generate_template_summary",
    "generate_llm_summary",
    "extract_dataset_name",
    "create_summary",
    "compute_alerts",
    "build_variable_chart",
    "render_markdown_to_html",
    "generate_html_report",
    "parse_llm_json_plan",
    "validate_tool_plan",
]

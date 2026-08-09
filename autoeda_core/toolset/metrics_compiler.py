import os
import json
import re
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.style as mplstyle
mplstyle.use("fast")  # Apply performance-oriented fast style sheet globally
import matplotlib.pyplot as plt
import seaborn as sns
import PIL.Image
PIL.Image.MAX_IMAGE_PIXELS = None  # Disable DecompressionBombWarning for large EDA visual plots
from typing import Dict, Any, List, Optional, Tuple
from pydantic import BaseModel, Field

from ..profiler import is_non_distributional_column
from ..llm_config import get_api_key, get_model, get_base_url

sns.set_theme(style="whitegrid")


from .utils import _sanitize_col_name, _safe_float, _is_numeric_col
from .data_visualizer import DataVisualizer, default_visualizer
from .predictive_blueprinter import PredictiveBlueprinter, default_predictive_blueprinter
from .report_validator import ReportValidator, default_report_validator


class MetricsCompiler:
    """
    Compiles all analysis outputs into the canonical metrics.json format.
    """
    def __init__(
        self,
        visualizer: Optional[DataVisualizer] = None,
        blueprinter: Optional[PredictiveBlueprinter] = None,
        validator: Optional[ReportValidator] = None
    ):
        self.visualizer = visualizer or DataVisualizer()
        self.blueprinter = blueprinter or PredictiveBlueprinter()
        self.validator = validator or ReportValidator()

    def compile_and_save_metrics(
        self,
        df: pd.DataFrame,
        dataset_path: str,
        target_col: Optional[str] = None,
        imputation_res: Optional[Dict[str, Any]] = None,
        outlier_res: Optional[Dict[str, Any]] = None,
        engineered_res: Optional[List[Dict[str, Any]]] = None,
        corr_res: Optional[Dict[str, Any]] = None,
        hypothesis_res: Optional[Dict[str, Any]] = None,
        blueprint_res: Optional[Dict[str, Any]] = None,
        output_dir: str = "./sandbox_run"
    ) -> str:
        os.makedirs(output_dir, exist_ok=True)

        num_rows, num_cols = df.shape

        raw_num_cols = num_cols
        profile_path = os.path.join(output_dir, "metadata_profile.json")
        if os.path.exists(profile_path):
            try:
                with open(profile_path, "r", encoding="utf-8") as f:
                    prof_data = json.load(f)
                    if isinstance(prof_data.get("dimensions"), dict) and "columns" in prof_data["dimensions"]:
                        raw_num_cols = prof_data["dimensions"]["columns"]
            except Exception:
                pass

        column_summary = {}
        for col in df.columns:
            column_summary[col] = {
                "dtype": str(df[col].dtype),
                "missing_count": int(df[col].isnull().sum()),
                "cardinality": int(df[col].nunique())
            }

        dist_res = self.visualizer.plot_feature_distributions(df)
        corr_matrix_res = corr_res or self.visualizer.plot_correlation_matrix(df)
        bivariate_res = self.visualizer.plot_semantic_bivariate_relationships(df)
        bivariate_union_res = self.visualizer.compute_bivariate_union(df, target_col=target_col, dataset_name=os.path.basename(dataset_path))
        pairplot_res = self.visualizer.plot_pairplot(df)
        target_inter_res = self.visualizer.plot_target_interaction(df, target_col=target_col, top_n=10)

        metrics_dict = {
            "dataset_overview": {
                "dataset_path": os.path.abspath(dataset_path),
                "shape": {"rows": num_rows, "columns": raw_num_cols},
                "raw_shape": {"rows": num_rows, "columns": raw_num_cols},
                "modified_shape": {"rows": num_rows, "columns": num_cols},
                "target_column": target_col,
                "column_summary": column_summary
            },
            "imputation_summary": imputation_res or {"status": "Imputation completed"},
            "outlier_analysis": outlier_res or {},
            "engineered_features": engineered_res or [],
            "correlation_analysis": corr_matrix_res,
            "categorical_associations": corr_matrix_res.get("categorical_associations", []),
            "statistical_hypothesis_tests": hypothesis_res or {},
            "predictive_modeling_blueprint": blueprint_res or self.blueprinter.generate_predictive_blueprint(df, target_col),
            "visual_distributions": dist_res.get("visual_distributions", {}),
            "correlation_data": corr_matrix_res,
            "bivariate_data": bivariate_res.get("bivariate_data", []),
            "bivariate_union": bivariate_union_res,
            "pairplot_data": pairplot_res,
            "target_interaction_data": target_inter_res.get("interaction_data", {}),
            "target_interactions": target_inter_res.get("target_interactions", []),
            "extracted_insights": {
                "key_findings": [
                    f"Dataset contains {num_rows} rows and {num_cols} columns.",
                    f"Processed missing values and computed statistical distributions."
                ],
                "statistically_significant_predictors": (hypothesis_res or {}).get("significant_predictors", [])
            }
        }

        metrics_dict["pipeline_validation"] = self.validator.validate_report_consistency(metrics_dict)

        metrics_path = os.path.join(output_dir, "metrics.json")
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics_dict, f, indent=2)

        print(f"[tools] Canonical metrics.json successfully saved to: {os.path.abspath(metrics_path)}")
        return metrics_path


default_metrics_compiler = MetricsCompiler(
    visualizer=default_visualizer,
    blueprinter=default_predictive_blueprinter,
    validator=default_report_validator
)


class FinishAnalysisArgs(BaseModel):
    pass


def compile_and_save_metrics(df: pd.DataFrame, dataset_path: str, target_col=None, imputation_res=None, outlier_res=None, engineered_res=None, corr_res=None, hypothesis_res=None, blueprint_res=None, output_dir="./sandbox_run"):
    return default_metrics_compiler.compile_and_save_metrics(df, dataset_path=dataset_path, target_col=target_col, imputation_res=imputation_res, outlier_res=outlier_res, engineered_res=engineered_res, corr_res=corr_res, hypothesis_res=hypothesis_res, blueprint_res=blueprint_res, output_dir=output_dir)


def finish_analysis(df: pd.DataFrame, **kwargs) -> Dict[str, Any]:
    return {"status": "finished"}


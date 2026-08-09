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
from .data_imputer import impute_missing_data, ImputeMissingDataArgs, default_imputer
from .outlier_analyzer import detect_and_handle_outliers, DetectAndHandleOutliersArgs, default_outlier_analyzer
from .feature_engineer import engineer_features, EngineerFeaturesArgs, default_feature_engineer
from .hypothesis_tester import run_statistical_hypothesis_tests, RunStatisticalHypothesisTestsArgs, default_hypothesis_tester
from .data_visualizer import plot_correlation_matrix, PlotCorrelationMatrixArgs, plot_target_interaction, PlotTargetInteractionArgs, plot_feature_distributions, PlotFeatureDistributionsArgs, plot_semantic_bivariate_relationships, PlotSemanticBivariateRelationshipsArgs, plot_pairplot, PlotPairplotArgs, default_visualizer
from .predictive_blueprinter import generate_predictive_blueprint, GeneratePredictiveBlueprintArgs, default_predictive_blueprinter
from .metrics_compiler import finish_analysis, FinishAnalysisArgs, default_metrics_compiler
from .agent_tools import ask_clarifying_question, AskClarifyingQuestionArgs


TOOL_REGISTRY = {
    "impute_missing_data": {
        "function": default_imputer.impute_missing_data,
        "description": "Imputes missing values using type-safe strategy.",
        "model": ImputeMissingDataArgs
    },
    "detect_and_handle_outliers": {
        "function": default_outlier_analyzer.detect_and_handle_outliers,
        "description": "Detects outliers via IQR method.",
        "model": DetectAndHandleOutliersArgs
    },
    "engineer_features": {
        "function": default_feature_engineer.engineer_features,
        "description": "Creates high-signal feature transformations (log1p, ratios, interactions).",
        "model": EngineerFeaturesArgs
    },
    "run_statistical_hypothesis_tests": {
        "function": default_hypothesis_tester.run_statistical_hypothesis_tests,
        "description": "Calculates statistical significance against target variable (T-Test, ANOVA, Chi-Square, Pearson).",
        "model": RunStatisticalHypothesisTestsArgs
    },
    "plot_correlation_matrix": {
        "function": default_visualizer.plot_correlation_matrix,
        "description": "Generates Pearson correlation matrix heatmap asset.",
        "model": PlotCorrelationMatrixArgs
    },
    "plot_feature_distributions": {
        "function": default_visualizer.plot_feature_distributions,
        "description": "Plots univariate histograms or countplots for important non-identifier features.",
        "model": PlotFeatureDistributionsArgs
    },
    "plot_target_interaction": {
        "function": default_visualizer.plot_target_interaction,
        "description": "Plots target variable interaction chart.",
        "model": PlotTargetInteractionArgs
    },
    "plot_semantic_bivariate_relationships": {
        "function": default_visualizer.plot_semantic_bivariate_relationships,
        "description": "Plots semantically meaningful bivariate relationship charts.",
        "model": PlotSemanticBivariateRelationshipsArgs
    },
    "plot_pairplot": {
        "function": default_visualizer.plot_pairplot,
        "description": "Generates a concise pairplot visualizing pairwise distributions.",
        "model": PlotPairplotArgs
    },
    "generate_predictive_blueprint": {
        "function": default_predictive_blueprinter.generate_predictive_blueprint,
        "description": "Compiles machine learning modeling blueprint and cross-validation strategy.",
        "model": GeneratePredictiveBlueprintArgs
    },
    "ask_clarifying_question": {
        "function": ask_clarifying_question,
        "description": "Ask the user a clarifying question when requirements or target columns are ambiguous.",
        "model": AskClarifyingQuestionArgs
    },
    "finish_analysis": {
        "function": finish_analysis,
        "description": "Call this tool when you have finished all necessary exploratory data analysis.",
        "model": FinishAnalysisArgs
    }
}


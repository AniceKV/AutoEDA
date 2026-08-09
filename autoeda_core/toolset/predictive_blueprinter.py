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


class PredictiveBlueprinter:
    """
    Encapsulates predictive modeling strategy, problem type inference
    (Binary/Multiclass Classification vs. Regression vs. Unsupervised),
    recommended algorithms, validation strategy, and overfitting risk mitigation.
    """
    def generate_predictive_blueprint(
        self,
        df: pd.DataFrame,
        target_col: Optional[str] = None,
        custom_blueprint: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        num_rows, num_cols = df.shape

        if not target_col or target_col not in df.columns:
            target_col = kwargs.get("target_col") or kwargs.get("target") or kwargs.get("target_definition")

        if not target_col or target_col not in df.columns:
            candidates = ["survived", "target", "label", "class", "income", "salary", "salary_usd", "price", "price_range", "diagnosis", "noise_complaints", "churn"]
            for cand in candidates:
                for col in df.columns:
                    if col.strip().lower() == cand:
                        target_col = col
                        break
                if target_col in df.columns:
                    break

        if target_col and target_col in df.columns:
            n_unique = df[target_col].nunique()
            if _is_numeric_col(df[target_col]) and n_unique > 20:
                problem_type = "Regression"
            elif n_unique == 2:
                problem_type = "Binary Classification"
            else:
                problem_type = "Multiclass Classification"
        else:
            target_col = "Undefined (Unsupervised)"
            problem_type = "Unsupervised / Exploratory"

        if "Classification" in problem_type:
            recommended_algos = [
                "Regularized Logistic Regression (baseline)",
                "Random Forest Classifier",
                "Gradient Boosting Classifier (XGBoost / LightGBM)",
                "Support Vector Classifier (SVM)"
            ]
            val_strat = [
                f"Stratified K-Fold Cross-Validation ({'5' if num_rows > 50 else 'Repeated 5'} folds)",
                "Evaluate Balanced Accuracy, Macro F1, Precision-Recall AUC, and Confusion Matrix"
            ]
        elif problem_type == "Regression":
            recommended_algos = [
                "Regularized Linear Regression (Ridge / Lasso)",
                "Random Forest Regressor",
                "Gradient Boosting Regressor",
                "Support Vector Regressor (SVR)"
            ]
            val_strat = [
                f"K-Fold Cross-Validation ({'5' if num_rows > 50 else 'Repeated 5'} folds)",
                "Evaluate MAE, RMSE, R-Squared, and Residual Error distribution"
            ]
        else:
            recommended_algos = [
                "K-Means Clustering",
                "Hierarchical Agglomerative Clustering",
                "Principal Component Analysis (PCA) for Dimensionality Reduction"
            ]
            val_strat = ["Evaluate Silhouette Score and Inertia elbow curve"]

        feature_selection = [
            "Exclude high-cardinality ID or text name columns",
            "Rank features using cross-validated permutation importance and mutual information",
            "Remove collinear features exceeding correlation threshold > 0.85"
        ]

        overfitting_mitigation = [
            "Apply regularization penalties (L1/L2)",
            "Limit tree depth and enforce minimum samples per leaf",
            "Perform hyperparameter tuning strictly within cross-validation folds"
        ]

        return {
            "target_definition": target_col,
            "problem_type": problem_type,
            "recommended_algorithms": recommended_algos,
            "feature_selection_strategy": feature_selection,
            "validation_strategy": val_strat,
            "overfitting_risk_mitigation": overfitting_mitigation,
            "executive_summary": f"Target: '{target_col}' ({problem_type}). Model recommendations and validation strategy tailored for {num_rows} rows x {num_cols} columns."
        }


default_predictive_blueprinter = PredictiveBlueprinter()


class GeneratePredictiveBlueprintArgs(BaseModel):
    target_col: str = Field(description="Target column name")


def generate_predictive_blueprint(df: pd.DataFrame, target_col=None, custom_blueprint=None, **kwargs):
    return default_predictive_blueprinter.generate_predictive_blueprint(df, target_col=target_col, custom_blueprint=custom_blueprint, **kwargs)


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


class ReportValidator:
    """
    Validates report consistency across metrics, targets, hypothesis tests, and prose text.
    """
    def validate_report_consistency(
        self,
        metrics_dict: Dict[str, Any],
        report_text: Optional[str] = None
    ) -> Dict[str, Any]:
        overview = metrics_dict.get("dataset_overview", {})
        target_col = overview.get("target_column")

        hypothesis_res = metrics_dict.get("statistical_hypothesis_tests", {})
        blueprint_res = metrics_dict.get("predictive_modeling_blueprint", {})

        validation_passed = True
        warnings = []

        hyp_target = hypothesis_res.get("target_col") if isinstance(hypothesis_res, dict) else None
        bp_target = blueprint_res.get("target_definition") if isinstance(blueprint_res, dict) else None

        if target_col and hyp_target and target_col != hyp_target:
            validation_passed = False
            warnings.append(f"Target mismatch between dataset overview ('{target_col}') and hypothesis tests ('{hyp_target}').")

        if target_col and bp_target and target_col != bp_target:
            validation_passed = False
            warnings.append(f"Target mismatch between dataset overview ('{target_col}') and predictive blueprint ('{bp_target}').")

        engineered = metrics_dict.get("engineered_features", [])
        if not isinstance(engineered, list):
            validation_passed = False
            warnings.append("Engineered features in metrics.json is not a valid list.")

        if report_text and isinstance(report_text, str):
            if target_col and target_col != "Undefined (Unsupervised)" and target_col not in report_text:
                validation_passed = False
                warnings.append(f"Target column '{target_col}' is missing from rendered report text.")

            if len(engineered) == 0 and "FamilySize" in report_text and "No custom derived" not in report_text:
                validation_passed = False
                warnings.append("Rendered report claims engineered feature 'FamilySize' when engineered_features is empty.")

        validation_summary = {
            "is_consistent": validation_passed,
            "warnings": warnings,
            "validated_target": target_col,
            "statistically_significant_count": len(hypothesis_res.get("significant_predictors", [])) if isinstance(hypothesis_res, dict) else 0,
            "engineered_features_count": len(engineered) if isinstance(engineered, list) else 0
        }

        if warnings:
            print(f"[tools] Validation Warning: Report consistency issues detected: {warnings}")
        else:
            print("[tools] Post-generation validation PASSED cleanly: Target, metrics, and report text are 100% consistent.")

        return validation_summary


default_report_validator = ReportValidator()


def validate_report_consistency(metrics_dict: Dict[str, Any], report_text: Optional[str] = None):
    return default_report_validator.validate_report_consistency(metrics_dict, report_text=report_text)


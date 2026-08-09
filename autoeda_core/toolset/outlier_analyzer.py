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


class OutlierAnalyzer:
    """
    Encapsulates outlier detection and capping algorithms (IQR method).
    """
    def detect_and_handle_outliers(
        self,
        df: pd.DataFrame,
        columns: Optional[List[str]] = None,
        method: str = "iqr",
        action: str = "profile",
        iqr_multiplier: float = 1.5,
        **kwargs
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        df_out = df.copy()

        action = str(action).lower() if action in ["profile", "cap"] else "profile"
        try:
            iqr_mult = max(0.5, min(5.0, float(iqr_multiplier)))
        except (ValueError, TypeError):
            iqr_mult = 1.5

        raw_cols = columns or [c for c in df_out.columns if _is_numeric_col(df_out[c])]
        target_cols = [c for c in raw_cols if c in df_out.columns and _is_numeric_col(df_out[c])]

        outlier_report = {}

        for col in target_cols:
            if df_out[col].nunique() <= 2:
                continue

            q1 = float(df_out[col].quantile(0.25))
            q3 = float(df_out[col].quantile(0.75))
            iqr = q3 - q1

            lower_bound = q1 - iqr_mult * iqr
            upper_bound = q3 + iqr_mult * iqr

            is_outlier = (df_out[col] < lower_bound) | (df_out[col] > upper_bound)
            outlier_count = int(is_outlier.sum())
            outlier_pct = round((outlier_count / len(df_out)) * 100, 2) if len(df_out) > 0 else 0.0

            if action == "cap" and outlier_count > 0:
                df_out[col] = df_out[col].clip(lower=lower_bound, upper=upper_bound)

            outlier_report[col] = {
                "q1": round(q1, 4),
                "q3": round(q3, 4),
                "iqr": round(iqr, 4),
                "lower_bound": round(lower_bound, 4),
                "upper_bound": round(upper_bound, 4),
                "outlier_count": outlier_count,
                "outlier_percentage": outlier_pct,
                "action_taken": action
            }

        return df_out, outlier_report


default_outlier_analyzer = OutlierAnalyzer()


class DetectAndHandleOutliersArgs(BaseModel):
    columns: List[str] = Field(description="List of numeric cols")
    action: str = Field(description="'profile' or 'cap'")


def detect_and_handle_outliers(df: pd.DataFrame, columns=None, method="iqr", action="profile", iqr_multiplier=1.5, **kwargs):
    return default_outlier_analyzer.detect_and_handle_outliers(df, columns=columns, method=method, action=action, iqr_multiplier=iqr_multiplier, **kwargs)


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


def _json_safe(obj: Any) -> Any:
    """Recursively convert numpy/pandas/scalar objects into JSON-serializable Python values."""
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return [_json_safe(v) for v in obj.tolist()]
    if isinstance(obj, np.generic):
        return _json_safe(obj.item())
    if isinstance(obj, (pd.Timestamp,)):
        return obj.isoformat()
    if isinstance(obj, (pd.Timedelta,)):
        return str(obj)
    if isinstance(obj, (pd.Series, pd.Index)):
        return [_json_safe(v) for v in obj.tolist()]
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    if hasattr(obj, "model_dump"):
        return _json_safe(obj.model_dump())
    if hasattr(obj, "to_dict"):
        return _json_safe(obj.to_dict())
    return str(obj)


def _sanitize_col_name(col: str) -> str:
    """Sanitize a column name into a filesystem-safe string."""
    return re.sub(r'\W+', '_', col).strip('_')


def _safe_float(val: Any, default: float = 0.0) -> float:
    """Safely converts any value (string, float, int, numpy value) to float without raising exceptions."""
    if val is None or pd.isna(val):
        return default
    try:
        return float(val)
    except (ValueError, TypeError):
        try:
            cleaned = re.sub(r'[^\d\.\-]+', '', str(val))
            return float(cleaned) if cleaned else default
        except Exception:
            return default


def _is_numeric_col(s: pd.Series) -> bool:
    """Helper to check if a Series has numeric dtype and is not boolean."""
    return pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s)


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


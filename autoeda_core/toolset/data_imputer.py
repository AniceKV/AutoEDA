import re
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
from pydantic import BaseModel, Field

from ..profiler import is_non_distributional_column
from ..llm_config import get_api_key, get_model, get_base_url

sns.set_theme(style="whitegrid")


from .utils import _sanitize_col_name, _safe_float, _is_numeric_col


class DataImputer:
    r"""
    Encapsulates type-safe missing value imputation strategies.
    
                          strategy_map
                              │
                              ▼
                  Is column in strategy_map?
                       /              \
                     YES               NO
                      │                 │
                      ▼                 ▼
              User's strategy        "auto"
          (mean / median / mode)       │
                      │                ▼
                      │        Is column numeric?
                      │           /          \
                      │         YES           NO
                      │          │             │
                      │          ▼             ▼
                      │    Calculate skew      MODE
                      │          │             │
                      │          ▼             │
                      │    Is |skew| > 1?      │
                      │       /      \         │
                      │     YES       NO       │
                      │      │         │       │
                      │      ▼         ▼       │
                      │    MEDIAN     MEAN     │
                      │      │         │       │
                      └──────┴─────────┴───────┘
                                      │
                                      ▼
                               Fill missing values
        """
    def impute_missing_data(self, df: pd.DataFrame, strategy_map: Optional[Dict[str, str]] = None,
                            numeric_skew_threshold: float = 1.0, **kwargs) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        df_imputed = df.copy()

        try:
            numeric_skew_threshold = abs(float(numeric_skew_threshold)) # we treat positive or negative skew as the same
        except (ValueError, TypeError):
            numeric_skew_threshold = 1.0

        strategy_map = strategy_map if isinstance(strategy_map, dict) else {}

        missing_tokens = ["?", "NA", "N/A", "null", "None", "nan", "NaN", ""]
        df_imputed = df_imputed.replace(missing_tokens, np.nan)

        rules_applied = [
            "Standardized missing string placeholders ('?', 'NA', 'N/A', 'null') to NaN.",
            f"Numeric columns with skewness > {numeric_skew_threshold} or < -{numeric_skew_threshold} use median imputation.",
            f"Numeric columns with skewness between -{numeric_skew_threshold} and {numeric_skew_threshold} use mean imputation.",
            "Categorical/String columns use mode imputation with 'Unknown' fallback."
        ]

        col_summary = {}
        dataset_size = df_imputed.shape[0]

        for col in df_imputed.columns:
            if df_imputed[col].dtype == object:
                converted = pd.to_numeric(df_imputed[col], errors="coerce") #if the column cant be converted converted as Nan
                if converted.notnull().sum() > 0.5 * dataset_size: #if 50% of values can be treated as numbers then its a number
                    df_imputed[col] = converted

            missing_before = df_imputed[col].isnull().sum()
            dtype_str = str(df_imputed[col].dtype)

            if missing_before == 0:
                col_summary[col] = {"dtype": dtype_str, "missing_before": 0, "missing_after": 0 ,"method": "none", "fill_value": None}
                continue

            strategy = strategy_map.get(col, "auto").lower()

            if _is_numeric_col(df_imputed[col]):
                skew_val = float(df_imputed[col].skew()) if missing_before / dataset_size < 0.5 else None

                if strategy == "median" or (strategy == "auto" and skew_val is not None and abs(skew_val) > numeric_skew_threshold): #if the stratergy recommended is median or the data is highly skewed use median
                    fill_val = float(df_imputed[col].median()) if df_imputed[col].notnull().any() else 0.0
                    method = "median"
                else:
                    fill_val = float(df_imputed[col].mean()) if df_imputed[col].notnull().any() else 0.0
                    method = "mean"

                df_imputed[col] = df_imputed[col].fillna(fill_val)

            else:
                skew_val = None

                if strategy in ["mean", "median"]:
                    print(f"[tools] Parameter Clamping: Clamped invalid strategy '{strategy}' to 'mode' for non-numeric column '{col}'.")
                    strategy = "mode"

                if strategy in ["mode", "auto"]:
                    mode_res = df_imputed[col].mode()
                    fill_val = str(mode_res.iloc[0]) if not mode_res.empty else "Unknown"
                    method = "mode"
                else:
                    fill_val = "Unknown"
                    method = "constant"

                df_imputed[col] = df_imputed[col].fillna(fill_val)

            col_summary[col] = {
                "dtype": dtype_str,
                "missing_before": missing_before,
                "missing_after": int(df_imputed[col].isnull().sum()),
                "method": method,
                "skewness": round(skew_val, 2) if skew_val is not None else None,
                "fill_value": fill_val
            }

        imputation_report = {
            "rules_applied": rules_applied,
            "columns": col_summary
        }

        return df_imputed, imputation_report


default_imputer = DataImputer()


class ImputeMissingDataArgs(BaseModel):
    strategy_map: Optional[Dict[str, str]] = Field(default=None, description="Optional dict of {col: strategy}")
    numeric_skew_threshold: Optional[float] = Field(default=1.0)


def impute_missing_data(df: pd.DataFrame, strategy_map=None, numeric_skew_threshold=1.0, **kwargs):
    return default_imputer.impute_missing_data(df, strategy_map=strategy_map, numeric_skew_threshold=numeric_skew_threshold, **kwargs)


import re
import pandas as pd
import seaborn as sns
from typing import Dict, Any, List, Optional, Tuple
from pydantic import BaseModel, Field

from ..profiler import is_non_distributional_column
from ..llm_config import get_api_key, get_model, get_base_url

sns.set_theme(style="whitegrid")


from .utils import _sanitize_col_name, _safe_float, _is_numeric_col


class FeatureEngineer:
    """
    Encapsulates dynamic feature engineering, mathematical formula evaluation,
    log transforms, ratios, interaction terms, and target correlation computation.
    """
    def _find_matching_col(self, requested: Optional[str], df_columns: List[str]) -> Optional[str]:
        if not requested or not isinstance(requested, str):
            return None
        if requested in df_columns:
            return requested

        clean_req = re.sub(r'[\s_]+', '', requested.strip().lower())
        for col in df_columns:
            clean_col = re.sub(r'[\s_]+', '', col.strip().lower())
            if clean_req == clean_col:
                return col

        for col in df_columns:
            clean_col = re.sub(r'[\s_]+', '', col.strip().lower())
            if clean_req in clean_col or clean_col in clean_req:
                return col

        return None

    def _evaluate_feature_formula(self, df: pd.DataFrame, formula_str: str) -> Optional[pd.Series]:
        try:
            clean_expr = formula_str.replace("np.log1p", "log1p").replace("np.log", "log").replace("np.abs", "abs")

            if "log1p(" in clean_expr:
                m = re.search(r"log1p\(([^)]+)\)", clean_expr)
                if m:
                    col = m.group(1).strip()
                    if col in df.columns:
                        return np.log1p(np.maximum(0, pd.to_numeric(df[col], errors="coerce").fillna(0)))
            elif "log(" in clean_expr:
                m = re.search(r"log\(([^)]+)\)", clean_expr)
                if m:
                    col = m.group(1).strip()
                    if col in df.columns:
                        return np.log(np.maximum(1e-5, pd.to_numeric(df[col], errors="coerce").fillna(0)))

            res = df.eval(clean_expr)
            if isinstance(res, (pd.Series, np.ndarray)):
                return pd.Series(res, index=df.index)
        except Exception:
            pass

        try:
            sorted_cols = sorted(df.columns, key=len, reverse=True)
            mapped_formula = formula_str
            local_dict = {"np": np, "pd": pd}
            col_to_idx = {str(col): i for i, col in enumerate(df.columns)}
            
            for col, i in col_to_idx.items():
                local_dict[f"__COL_{i}__"] = df[col]
            
            for col in sorted_cols:
                col_str = str(col)
                i = col_to_idx[col_str]
                escaped_col = re.escape(col_str)
                pattern = r'(?<![a-zA-Z0-9_])' + escaped_col + r'(?![a-zA-Z0-9_])'
                mapped_formula = re.sub(pattern, f"__COL_{i}__", mapped_formula)

            res = eval(mapped_formula, {"__builtins__": None}, local_dict)
            if isinstance(res, (pd.Series, np.ndarray, list)):
                return pd.Series(res, index=df.index)
        except Exception:
            pass

        return None

    def engineer_features(
        self,
        df: pd.DataFrame,
        feature_specs: Optional[List[Dict[str, Any]]] = None,
        target_col: Optional[str] = None,
        **kwargs
    ) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
        df_feat = df.copy()
        engineered_summary = []

        raw_specs = feature_specs or kwargs.get("specs") or kwargs.get("features") or kwargs.get("engineered_features") or kwargs.get("feature_list") or kwargs.get("feature_specs") or []
        if isinstance(raw_specs, dict):
            specs = [raw_specs]
        elif isinstance(raw_specs, list):
            specs = raw_specs
        else:
            specs = []

        if not specs:
            numeric_cols = [c for c in df_feat.columns if _is_numeric_col(df_feat[c]) and c != target_col and df_feat[c].nunique() > 2]

            for col in numeric_cols:
                skew_val = df_feat[col].skew()
                if skew_val > 1.5 and (df_feat[col] >= 0).all():
                    specs.append({
                        "name": f"log_{col}",
                        "type": "log1p",
                        "source_col": col,
                        "rationale": f"Log transform to normalize highly right-skewed variable ({round(skew_val, 2)})"
                    })

            if len(numeric_cols) >= 2:
                c1, c2 = numeric_cols[0], numeric_cols[1]
                specs.append({
                    "name": f"{c1}_{c2}_interaction",
                    "type": "product",
                    "source_cols": [c1, c2],
                    "rationale": f"Multiplicative interaction feature between key numerical attributes {c1} and {c2}"
                })

        for spec in specs:
            if not isinstance(spec, dict):
                continue

            ftype = str(spec.get("type") or spec.get("transformation") or spec.get("operation") or "custom").lower()
            rationale = spec.get("rationale") or spec.get("description") or "High-signal feature engineering transformation"
            raw_formula = spec.get("formula") or spec.get("expression")

            raw_cols = spec.get("columns") or spec.get("source_cols") or spec.get("source_columns") or spec.get("features") or spec.get("input_cols") or spec.get("input_columns") or spec.get("cols") or []
            if isinstance(raw_cols, str):
                raw_cols = [c.strip() for c in raw_cols.split(",") if c.strip()]

            raw_scol = spec.get("source_col") or spec.get("column") or (raw_cols[0] if raw_cols else None)

            df_cols_list = list(df_feat.columns)
            cols = []
            for c in raw_cols:
                matched = self._find_matching_col(c, df_cols_list)
                if matched and matched not in cols:
                    cols.append(matched)

            scol = self._find_matching_col(raw_scol, df_cols_list) if raw_scol else (cols[0] if cols else None)

            fname = spec.get("name") or spec.get("feature_name") or spec.get("target")
            if not fname or fname == "engineered_feature":
                if raw_formula:
                    fname = "derived_formula_metric"
                elif ftype in ["sum", "add", "addition", "total", "plus", "aggregate", "combine", "cumulative"]:
                    fname = f"total_{'_'.join(cols[:3])}" if cols else "composite_total"
                elif ftype in ["ratio", "division", "divide", "div"]:
                    num_raw = spec.get("numerator") or (raw_cols[0] if len(raw_cols) >= 1 else None)
                    den_raw = spec.get("denominator") or (raw_cols[1] if len(raw_cols) >= 2 else None)
                    num = self._find_matching_col(num_raw, df_cols_list) if num_raw else None
                    den = self._find_matching_col(den_raw, df_cols_list) if den_raw else None
                    fname = f"{num}_per_{den}" if (num and den) else "composite_ratio"
                elif ftype in ["product", "interaction", "multiply", "mult", "multiplication", "times"]:
                    fname = f"{cols[0]}_x_{cols[1]}" if len(cols) >= 2 else "composite_interaction"
                elif ftype in ["log1p", "log", "logarithm", "log_transform"]:
                    fname = f"log_{scol}" if scol else "log_transformed_feature"
                elif ftype in ["difference", "subtraction", "subtract", "minus", "sub"]:
                    fname = f"{cols[0]}_minus_{cols[1]}" if len(cols) >= 2 else "composite_diff"
                elif ftype in ["mean", "average", "avg"]:
                    fname = f"avg_{'_'.join(cols[:3])}" if cols else "composite_average"
                else:
                    fname = "derived_domain_metric"

            try:
                if raw_formula:
                    evaluated_series = self._evaluate_feature_formula(df_feat, raw_formula)
                    if evaluated_series is not None:
                        df_feat[fname] = evaluated_series
                        formula = raw_formula
                    else:
                        continue
                elif ftype in ["log1p", "log", "logarithm", "log_transform"]:
                    if scol and scol in df_feat.columns:
                        df_feat[fname] = np.log1p(np.maximum(0, pd.to_numeric(df_feat[scol], errors="coerce").fillna(0)))
                        formula = f"np.log1p({scol})"
                    else:
                        continue

                elif ftype in ["ratio", "division", "divide", "div"]:
                    num_raw = spec.get("numerator") or (raw_cols[0] if len(raw_cols) >= 1 else None)
                    den_raw = spec.get("denominator") or (raw_cols[1] if len(raw_cols) >= 2 else None)
                    num = self._find_matching_col(num_raw, df_cols_list) if num_raw else None
                    den = self._find_matching_col(den_raw, df_cols_list) if den_raw else None

                    if num and den and num in df_feat.columns and den in df_feat.columns:
                        den_series = pd.to_numeric(df_feat[den], errors="coerce").fillna(0)
                        num_series = pd.to_numeric(df_feat[num], errors="coerce").fillna(0)
                        df_feat[fname] = num_series / (den_series.abs() + 1e-5)
                        formula = f"{num} / ({den} + eps)"
                    else:
                        continue

                elif ftype in ["product", "interaction", "multiply", "mult", "multiplication", "times"]:
                    if len(cols) >= 2 and all(c in df_feat.columns for c in cols[:2]):
                        c1_series = pd.to_numeric(df_feat[cols[0]], errors="coerce").fillna(0)
                        c2_series = pd.to_numeric(df_feat[cols[1]], errors="coerce").fillna(0)
                        df_feat[fname] = c1_series * c2_series
                        formula = f"{cols[0]} * {cols[1]}"
                    else:
                        continue

                elif ftype in ["sum", "add", "addition", "total", "plus", "aggregate", "combine", "cumulative"]:
                    valid_cols = [c for c in cols if c in df_feat.columns]
                    if valid_cols:
                        df_feat[fname] = df_feat[valid_cols].apply(pd.to_numeric, errors="coerce").fillna(0).sum(axis=1)
                        formula = f"sum({', '.join(valid_cols)})"
                    else:
                        continue

                elif ftype in ["difference", "subtraction", "subtract", "minus", "sub"]:
                    if len(cols) >= 2 and all(c in df_feat.columns for c in cols[:2]):
                        c1_series = pd.to_numeric(df_feat[cols[0]], errors="coerce").fillna(0)
                        c2_series = pd.to_numeric(df_feat[cols[1]], errors="coerce").fillna(0)
                        df_feat[fname] = c1_series - c2_series
                        formula = f"{cols[0]} - {cols[1]}"
                    else:
                        continue

                elif ftype in ["mean", "average", "avg"]:
                    valid_cols = [c for c in cols if c in df_feat.columns]
                    if valid_cols:
                        df_feat[fname] = df_feat[valid_cols].apply(pd.to_numeric, errors="coerce").fillna(0).mean(axis=1)
                        formula = f"mean({', '.join(valid_cols)})"
                    else:
                        continue
                else:
                    continue

                corr_with_target = None
                if target_col and target_col in df_feat.columns and _is_numeric_col(df_feat[target_col]):
                    corr_val = df_feat[fname].corr(df_feat[target_col])
                    corr_with_target = round(float(corr_val), 4) if pd.notnull(corr_val) else None

                engineered_summary.append({
                    "feature_name": fname,
                    "formula": formula,
                    "data_type": str(df_feat[fname].dtype),
                    "rationale": rationale,
                    "correlation_with_target": corr_with_target
                })
            except Exception as e:
                print(f"[tools] Error engineering feature '{fname}': {e}")

        return df_feat, engineered_summary


default_feature_engineer = FeatureEngineer()


class EngineerFeaturesArgs(BaseModel):
    feature_specs: List[Dict[str, Any]] = Field(
        description="List of dicts defining features. Accepts arbitrary formulas or standard types."
    )


def engineer_features(df: pd.DataFrame, feature_specs=None, target_col=None, **kwargs):
    return default_feature_engineer.engineer_features(df, feature_specs=feature_specs, target_col=target_col, **kwargs)


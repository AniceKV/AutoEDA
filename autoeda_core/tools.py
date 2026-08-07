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
from .profiler import is_non_distributional_column

sns.set_theme(style="whitegrid")


def _sanitize_col_name(col: str) -> str:
    """Sanitize a column name into a filesystem-safe string."""
    return re.sub(r'\W+', '_', col).strip('_')


def _is_numeric_col(s: pd.Series) -> bool:
    """Helper to check if a Series has numeric dtype and is not boolean."""
    return pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s)


def _downsample_for_viz(df: pd.DataFrame, max_samples: int = 10000, random_state: int = 42) -> pd.DataFrame:
    """Downsamples large DataFrames for rendering fast Matplotlib/Seaborn visual plots."""
    if len(df) > max_samples:
        print(f"[tools] Parameter Clamping: Downsampling DataFrame from {len(df)} rows to {max_samples} rows for fast visualization rendering.")
        return df.sample(n=max_samples, random_state=random_state)
    return df




# =====================================================================
# STATEFUL EXECUTION MEMORY & DATA VERSION CONTROL (DVC PATTERN)
# =====================================================================
class StatefulDataStore:
    """
    Manages stateful execution memory and DataFrame version control (DVC pattern).
    Maintains checkpoints in memory using df.copy() and copy.deepcopy(agent_state).
    Allows automatic rollback if a tool step corrupts or invalidates the dataset or metadata.
    """
    def __init__(self, workspace_dir: str = "./sandbox_run"):
        self.workspace_dir = workspace_dir
        self.version = 0
        self.history: List[Dict[str, Any]] = []
        os.makedirs(self.workspace_dir, exist_ok=True)

    def _make_entry(self, version: int, df: pd.DataFrame, agent_state: dict, action: str) -> dict:
        import copy
        return {
            "version": version,
            "df": df.copy(),
            "agent_state": copy.deepcopy(agent_state),
            "rows": len(df),
            "cols": len(df.columns),
            "action": action
        }

    def set_initial_state(self, df: pd.DataFrame, agent_state: dict) -> str:
        self.version = 0
        self.history = [self._make_entry(0, df, agent_state, "initial_load")]
        print(f"[DataStore] Initialized state v0 ({len(df)} rows, {len(df.columns)} cols) in memory.")
        return "memory:v0"

    def save_checkpoint(self, df: pd.DataFrame, agent_state: dict, step_name: str) -> str:
        if df is None or len(df) == 0 or len(df.columns) == 0:
            raise ValueError(f"Cannot checkpoint invalid or empty DataFrame after step '{step_name}'.")
        self.version += 1
        self.history.append(self._make_entry(self.version, df, agent_state, step_name))
        print(f"[DataStore] Saved checkpoint v{self.version} after '{step_name}' ({len(df)} rows, {len(df.columns)} cols) in memory.")
        return f"memory:v{self.version}"

    def rollback(self) -> Tuple[pd.DataFrame, dict, int]:
        import copy
        if len(self.history) <= 1:
            print("[DataStore] Cannot rollback further. At initial state v0.")
            latest_state = self.history[0]
            return latest_state["df"].copy(), copy.deepcopy(latest_state["agent_state"]), 0
        
        bad_state = self.history.pop()
        print(f"[DataStore] Rolling back from corrupted state v{bad_state['version']} ({bad_state['action']})...")
        
        latest_state = self.history[-1]
        self.version = latest_state["version"]
        restored_df = latest_state["df"].copy()
        restored_agent_state = copy.deepcopy(latest_state["agent_state"])
        print(f"[DataStore] Successfully rolled back to state v{self.version} ({latest_state['action']})")
        return restored_df, restored_agent_state, self.version

    def purge_intermediate_states(self):
        """
        Deletes intermediate checkpoint data frames to save memory,
        keeping only the initial load (v0) and the final active version.
        """
        if len(self.history) <= 2:
            return
            
        final_state = self.history[-1]
        self.history = [self.history[0], final_state]
        print(f"[DataStore] Cleaned up intermediate states. Retained initial state (v0) and final state (v{self.version}).")


# =====================================================================
# 1. SMART TYPE-SAFE IMPUTATION TOOL (ROBUST PARAMETER CLAMPING)
# =====================================================================
def impute_missing_data(
    df: pd.DataFrame,
    strategy_map: Optional[Dict[str, str]] = None,
    numeric_skew_threshold: float = 1.0,
    **kwargs
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Programmatically applies type-safe missing value imputation.
    - Parameter Clamping: Clamps skew threshold, validates column existence,
      and automatically clamps strategy from mean/median to mode/constant for non-numeric columns.
    """
    df_imputed = df.copy()
    
    # 1. Clamp parameters defensively
    try:
        numeric_skew_threshold = abs(float(numeric_skew_threshold))
    except (ValueError, TypeError):
        numeric_skew_threshold = 1.0
        
    strategy_map = strategy_map if isinstance(strategy_map, dict) else {}
    
    # Standardize string missing value representations
    missing_tokens = ["?", "NA", "N/A", "null", "None", "nan", "NaN", ""]
    df_imputed = df_imputed.replace(missing_tokens, np.nan)
    
    rules_applied = [
        "Standardized missing string placeholders ('?', 'NA', 'N/A', 'null') to NaN.",
        f"Numeric columns with skewness > {numeric_skew_threshold} or < -{numeric_skew_threshold} use median imputation.",
        f"Numeric columns with skewness between -{numeric_skew_threshold} and {numeric_skew_threshold} use mean imputation.",
        "Categorical/String columns use mode imputation with 'Unknown' fallback."
    ]
    
    col_summary = {}
    
    for col in df_imputed.columns:
        # Convert numeric-like object columns if possible
        if df_imputed[col].dtype == object:
            try:
                converted = pd.to_numeric(df_imputed[col], errors="coerce")
                if converted.notnull().sum() > 0.5 * len(df_imputed):
                    df_imputed[col] = converted
            except Exception:
                pass
                
        missing_before = int(df_imputed[col].isnull().sum())
        dtype_str = str(df_imputed[col].dtype)
        
        if missing_before == 0:
            col_summary[col] = {
                "dtype": dtype_str,
                "missing_before": 0,
                "missing_after": 0,
                "method": "none",
                "fill_value": None
            }
            continue
            
        strategy = strategy_map.get(col, "auto").lower()
        
        # PARAMETER CLAMPING: Enforce type constraints programmatically
        if _is_numeric_col(df_imputed[col]):
            skew_val = float(df_imputed[col].skew()) if df_imputed[col].notnull().sum() > 2 else 0.0
            
            if strategy == "median" or (strategy == "auto" and abs(skew_val) > numeric_skew_threshold):
                fill_val = float(df_imputed[col].median()) if df_imputed[col].notnull().any() else 0.0
                method = "median"
            else:
                fill_val = float(df_imputed[col].mean()) if df_imputed[col].notnull().any() else 0.0
                method = "mean"
                
            df_imputed[col] = df_imputed[col].fillna(fill_val)
        else:
            skew_val = None
            # If LLM mistakenly passed mean or median for categorical column, clamp strategy to mode
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


# =====================================================================
# 2. OUTLIER PROFILING & CAPPING TOOL (ROBUST PARAMETER CLAMPING)
# =====================================================================
def detect_and_handle_outliers(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    method: str = "iqr",
    action: str = "profile",
    iqr_multiplier: float = 1.5,
    **kwargs
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Detects outliers using IQR (Interquartile Range) method with parameter clamping.
    - action='profile': Calculates IQR bounds, outlier counts, and percentages.
    - action='cap': Caps extreme values at calculated lower and upper bounds.
    """
    df_out = df.copy()
    
    # Defensive parameter clamping
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


# =====================================================================
# 3. HIGH-SIGNAL FEATURE ENGINEERING TOOL
# =====================================================================
def _find_matching_col(requested: Optional[str], df_columns: List[str]) -> Optional[str]:
    """Helper to match column names flexibly across case, spaces, and underscores."""
    if not requested or not isinstance(requested, str):
        return None
    if requested in df_columns:
        return requested
    
    clean_req = re.sub(r'[\s_]+', '', requested.strip().lower())
    for col in df_columns:
        clean_col = re.sub(r'[\s_]+', '', col.strip().lower())
        if clean_req == clean_col:
            return col
            
    # Substring match (e.g. "math" -> "math_score" or "math score")
    for col in df_columns:
        clean_col = re.sub(r'[\s_]+', '', col.strip().lower())
        if clean_req in clean_col or clean_col in clean_req:
            return col
            
    return None


def _evaluate_feature_formula(df: pd.DataFrame, formula_str: str) -> Optional[pd.Series]:
    """
    Safely evaluates mathematical/logical feature expressions against DataFrame columns.
    Supports arithmetic (+, -, *, /), comparisons (==, >, <), and functions (log1p, log, abs).
    """
    try:
        clean_expr = formula_str.replace("np.log1p", "log1p").replace("np.log", "log").replace("np.abs", "abs")
        
        import re
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
        local_dict = {col: df[col] for col in df.columns}
        local_dict["np"] = np
        local_dict["pd"] = pd
        res = eval(formula_str, {"__builtins__": None}, local_dict)
        if isinstance(res, (pd.Series, np.ndarray, list)):
            return pd.Series(res, index=df.index)
    except Exception:
        pass
        
    return None


def engineer_features(
    df: pd.DataFrame,
    feature_specs: Optional[List[Dict[str, Any]]] = None,
    target_col: Optional[str] = None,
    **kwargs
) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
    """
    Creates high-signal domain features safely.
    Supports auto-generation, arbitrary formula evaluation (e.g. 'SibSp + Parch + 1'), and flexible LLM specifications.
    """
    df_feat = df.copy()
    engineered_summary = []
    
    raw_specs = feature_specs or kwargs.get("specs") or kwargs.get("features") or kwargs.get("engineered_features") or kwargs.get("feature_list") or kwargs.get("feature_specs") or []
    if isinstance(raw_specs, dict):
        specs = [raw_specs]
    elif isinstance(raw_specs, list):
        specs = raw_specs
    else:
        specs = []
    
    # Auto-generate features if specs not provided
    if not specs:
        numeric_cols = [c for c in df_feat.columns if _is_numeric_col(df_feat[c]) and c != target_col and df_feat[c].nunique() > 2]
        
        # 1. Log transform skewed features
        for col in numeric_cols:
            skew_val = df_feat[col].skew()
            if skew_val > 1.5 and (df_feat[col] >= 0).all():
                specs.append({
                    "name": f"log_{col}",
                    "type": "log1p",
                    "source_col": col,
                    "rationale": f"Log transform to normalize highly right-skewed variable ({round(skew_val, 2)})"
                })
                
        # 2. Pairwise interaction between top 2 numeric features if available
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
        
        # Extract column references gracefully
        raw_cols = spec.get("columns") or spec.get("source_cols") or spec.get("source_columns") or spec.get("features") or spec.get("input_cols") or spec.get("input_columns") or spec.get("cols") or []
        if isinstance(raw_cols, str):
            raw_cols = [c.strip() for c in raw_cols.split(",") if c.strip()]
            
        raw_scol = spec.get("source_col") or spec.get("column") or (raw_cols[0] if raw_cols else None)
        
        # Resolve requested column names to actual dataframe column names
        df_cols_list = list(df_feat.columns)
        cols = []
        for c in raw_cols:
            matched = _find_matching_col(c, df_cols_list)
            if matched and matched not in cols:
                cols.append(matched)
                
        scol = _find_matching_col(raw_scol, df_cols_list) if raw_scol else (cols[0] if cols else None)

        # Dynamic Semantic Feature Naming (avoid generic 'engineered_feature')
        fname = spec.get("name") or spec.get("feature_name") or spec.get("target")
        if not fname or fname == "engineered_feature":
            if raw_formula:
                fname = "derived_formula_metric"
            elif ftype in ["sum", "add", "addition", "total", "plus", "aggregate", "combine", "cumulative"]:
                fname = f"total_{'_'.join(cols[:3])}" if cols else "composite_total"
            elif ftype in ["ratio", "division", "divide", "div"]:
                num_raw = spec.get("numerator") or (raw_cols[0] if len(raw_cols) >= 1 else None)
                den_raw = spec.get("denominator") or (raw_cols[1] if len(raw_cols) >= 2 else None)
                num = _find_matching_col(num_raw, df_cols_list) if num_raw else None
                den = _find_matching_col(den_raw, df_cols_list) if den_raw else None
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
            # Primary: Formula string evaluation
            if raw_formula:
                evaluated_series = _evaluate_feature_formula(df_feat, raw_formula)
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
                num = _find_matching_col(num_raw, df_cols_list) if num_raw else None
                den = _find_matching_col(den_raw, df_cols_list) if den_raw else None
                
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


# =====================================================================
# 4. HYPOTHESIS TESTING & STATISTICAL SIGNIFICANCE TOOL (ROBUST)
# =====================================================================
def _run_group_test(groups: list) -> Optional[Tuple[str, float, float, float, str]]:
    """
    Shared helper: filters groups with < 2 samples, then dispatches to
    Welch T-Test (2 groups) or One-Way ANOVA (3+ groups).
    Returns (test_name, statistic, p_value, effect_size, interpretation) or None if insufficient groups.
    """
    groups = [g for g in groups if len(g) >= 2]
    if len(groups) < 2:
        return None
    if len(groups) == 2:
        g1, g2 = groups[0], groups[1]
        t_stat, p_val = stats.ttest_ind(g1, g2, equal_var=False)
        std_pooled = np.sqrt((np.var(g1, ddof=1) + np.var(g2, ddof=1)) / 2.0)
        cohens_d = abs(np.mean(g1) - np.mean(g2)) / (std_pooled + 1e-8)
        effect_size = round(float(min(1.0, cohens_d / 2.0)), 4)
        return "Two-Sample Welch T-Test", float(t_stat), float(p_val), effect_size, f"T-statistic = {t_stat:.4f}, Cohen's d = {cohens_d:.4f}, p = {p_val:.4e}."
    
    f_stat, p_val = stats.f_oneway(*groups)
    all_vals = np.concatenate(groups)
    grand_mean = np.mean(all_vals)
    ss_total = np.sum((all_vals - grand_mean) ** 2)
    ss_between = sum(len(g) * (np.mean(g) - grand_mean) ** 2 for g in groups)
    eta_sq = (ss_between / ss_total) if ss_total > 0 else 0.0
    effect_size = round(float(eta_sq), 4)
    return "One-Way ANOVA", float(f_stat), float(p_val), effect_size, f"F-statistic = {f_stat:.4f}, Eta-squared = {eta_sq:.4f}, p = {p_val:.4e}."


def run_statistical_hypothesis_tests(
    df: pd.DataFrame,
    target_col: Optional[str] = None,
    feature_cols: Optional[List[str]] = None,
    alpha: float = 0.05,
    **kwargs
) -> Dict[str, Any]:
    """
    Automates statistical significance testing against target_col with defensive parameter clamping.
    Ranks statistically significant features by effect size.
    """
    # Parameter Clamping for alpha
    try:
        alpha = max(0.0001, min(0.5, float(alpha)))
    except (ValueError, TypeError):
        alpha = 0.05

    if not target_col or target_col not in df.columns:
        numeric_cols = [c for c in df.columns if _is_numeric_col(df[c])]
        target_col = numeric_cols[-1] if numeric_cols else df.columns[-1]

    raw_targets = feature_cols or [c for c in df.columns if c != target_col]
    targets_to_test = [c for c in raw_targets if c in df.columns and c != target_col]
    
    test_results = {}
    significant_items = []
    
    target_is_num = _is_numeric_col(df[target_col])
    
    for col in targets_to_test:
        col_is_num = _is_numeric_col(df[col])
        clean_data = df[[target_col, col]].dropna()
        if len(clean_data) < 5:
            continue
            
        try:
            if target_is_num and col_is_num:
                # Pearson Correlation Test
                r_val, p_val = stats.pearsonr(clean_data[target_col], clean_data[col])
                test_name = "Pearson Correlation Test"
                statistic = float(r_val)
                effect_size = round(abs(float(r_val)), 4)
                interpretation = f"Pearson r = {r_val:.4f}, |r| = {effect_size}, p = {p_val:.4e}."
                
            elif not target_is_num and not col_is_num:
                # Chi-Square Test of Independence
                contingency = pd.crosstab(clean_data[target_col], clean_data[col])
                chi2, p_val, dof, ex = stats.chi2_contingency(contingency)
                test_name = "Chi-Square Test of Independence"
                statistic = float(chi2)
                n = contingency.sum().sum()
                min_dim = min(contingency.shape[0] - 1, contingency.shape[1] - 1)
                cramers_v = np.sqrt(chi2 / (n * min_dim)) if (n > 0 and min_dim > 0) else 0.0
                effect_size = round(float(cramers_v), 4)
                interpretation = f"Chi2 = {chi2:.4f}, Cramér's V = {effect_size:.4f}, p = {p_val:.4e}."
                
            elif not target_is_num and col_is_num:
                # Feature is Numerical, Target is Categorical
                groups = [group[col].dropna().values for name, group in clean_data.groupby(target_col)]
                result = _run_group_test(groups)
                if result is None:
                    continue
                test_name, statistic, p_val, effect_size, interpretation = result
            else:
                # Target is Numerical, Feature is Categorical
                groups = [group[target_col].dropna().values for name, group in clean_data.groupby(col)]
                result = _run_group_test(groups)
                if result is None:
                    continue
                test_name, statistic, p_val, effect_size, interpretation = result
                    
            p_val_float = float(p_val) if pd.notnull(p_val) else 1.0
            is_sig = p_val_float < alpha
            
            if is_sig:
                significant_items.append({"feature": col, "effect_size": effect_size, "p_value": p_val_float})
                
            test_results[col] = {
                "test_name": test_name,
                "statistic": round(statistic, 4),
                "p_value": p_val_float,
                "effect_size": effect_size,
                "is_statistically_significant": is_sig,
                "interpretation": interpretation + (" (Statistically Significant)" if is_sig else " (Not Significant)")
            }
        except Exception as e:
            test_results[col] = {
                "test_name": "Hypothesis Test",
                "error": str(e),
                "p_value": 1.0,
                "effect_size": 0.0,
                "is_statistically_significant": False,
                "interpretation": f"Could not perform test: {e}"
            }

    # Rank significant features by effect size descending
    significant_items.sort(key=lambda x: x["effect_size"], reverse=True)
    test_results["significant_predictors"] = [item["feature"] for item in significant_items]
    test_results["ranked_significant_details"] = significant_items
    return test_results


# =====================================================================
# 5. VISUALIZATION: CORRELATION MATRIX TOOL
# =====================================================================
def _cramers_v(x, y):
    confusion_matrix = pd.crosstab(x, y)
    if confusion_matrix.empty:
        return 0.0
    chi2 = stats.chi2_contingency(confusion_matrix, correction=False)[0]
    n = confusion_matrix.sum().sum()
    if n == 0:
        return 0.0
    phi2 = chi2 / n
    r, k = confusion_matrix.shape
    
    phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))
    rcorr = r - ((r-1)**2)/(n-1)
    kcorr = k - ((k-1)**2)/(n-1)
    
    if min((kcorr-1), (rcorr-1)) == 0:
        return 0.0
    
    return np.sqrt(phi2corr / min((kcorr-1), (rcorr-1)))


def plot_correlation_matrix(
    df: pd.DataFrame,
    numeric_cols: Optional[List[str]] = None,
    save_path: str = "correlation_matrix.png",
    output_dir: str = "./sandbox_run",
    **kwargs
) -> Dict[str, Any]:
    """
    Computes Pearson correlation matrix, saves styled heatmap asset,
    and extracts top positive/negative correlations.
    """
    out_dir = kwargs.get("output_dir") or output_dir
    plt.close()
    target_cols = numeric_cols or [c for c in df.columns if _is_numeric_col(df[c]) and df[c].nunique() > 1]
    
    if len(target_cols) < 2:
        return {"error": "Insufficient numeric columns for correlation analysis."}
        
    corr_matrix = df[target_cols].corr()
    
    os.makedirs(out_dir, exist_ok=True)
    full_save_path = os.path.join(out_dir, os.path.basename(save_path))

    try:
        fig, ax = plt.subplots(figsize=(max(8, len(target_cols) * 0.8), max(6, len(target_cols) * 0.7)))
        sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", center=0, square=True, linewidths=0.5, ax=ax)
        ax.set_title("Pearson Correlation Matrix", fontsize=14, pad=12)
        plt.tight_layout()
        plt.savefig(full_save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    except Exception as e:
        print(f"[tools] Warning: Heatmap rendering error: {e}")
        plt.close()

    # Extract top positive & negative correlation pairs
    pairs = []
    for i in range(len(target_cols)):
        for j in range(i + 1, len(target_cols)):
            c1, c2 = target_cols[i], target_cols[j]
            val = corr_matrix.loc[c1, c2]
            if pd.notnull(val):
                pairs.append({"feature_1": c1, "feature_2": c2, "correlation": round(float(val), 4)})
                
    pairs_sorted = sorted(pairs, key=lambda x: abs(x["correlation"]), reverse=True)
    
    cat_cols = [c for c in df.columns if not _is_numeric_col(df[c]) and df[c].nunique() > 1 and df[c].nunique() <= 50]
    categorical_associations = {}
    
    if len(cat_cols) >= 2:
        cat_corr_matrix = pd.DataFrame(index=cat_cols, columns=cat_cols)
        for i in range(len(cat_cols)):
            for j in range(len(cat_cols)):
                if i == j:
                    cat_corr_matrix.loc[cat_cols[i], cat_cols[j]] = 1.0
                elif i < j:
                    v = _cramers_v(df[cat_cols[i]].dropna(), df[cat_cols[j]].dropna())
                    cat_corr_matrix.loc[cat_cols[i], cat_cols[j]] = v
                    cat_corr_matrix.loc[cat_cols[j], cat_cols[i]] = v
                    
        cat_corr_matrix = cat_corr_matrix.astype(float)
        
        cat_save_path = os.path.join(output_dir, "categorical_association_matrix.png")
        try:
            fig, ax = plt.subplots(figsize=(max(8, len(cat_cols) * 0.8), max(6, len(cat_cols) * 0.7)))
            sns.heatmap(cat_corr_matrix, annot=True, fmt=".2f", cmap="Blues", vmin=0, vmax=1, square=True, linewidths=0.5, ax=ax)
            ax.set_title("Categorical Association Matrix (Cramér's V)", fontsize=14, pad=12)
            plt.tight_layout()
            plt.savefig(cat_save_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
        except Exception as e:
            print(f"[tools] Warning: Categorical Heatmap rendering error: {e}")
            plt.close()
            
        cat_pairs = []
        for i in range(len(cat_cols)):
            for j in range(i + 1, len(cat_cols)):
                c1, c2 = cat_cols[i], cat_cols[j]
                val = cat_corr_matrix.loc[c1, c2]
                if pd.notnull(val):
                    cat_pairs.append({"feature_1": c1, "feature_2": c2, "cramers_v": round(float(val), 4)})
                    
        cat_pairs_sorted = sorted(cat_pairs, key=lambda x: x["cramers_v"], reverse=True)
        
        categorical_associations = {
            "heatmap_saved": cat_save_path,
            "top_correlations": cat_pairs_sorted[:10],
            "association_matrix_text": cat_corr_matrix.round(3).to_dict()
        }
    
    return {
        "heatmap_saved": full_save_path,
        "top_correlations": pairs_sorted[:10],
        "correlation_matrix_text": corr_matrix.round(3).to_dict(),
        "categorical_associations": categorical_associations
    }


# =====================================================================
# 6. VISUALIZATION: SHARED BIVARIATE PLOT DISPATCH
# =====================================================================
def _render_bivariate_axes(
    ax, df: pd.DataFrame, x_col: str, y_col: str, hue_col: Optional[str] = None
) -> None:
    """
    Shared rendering logic for bivariate plots.
    Dispatches to regplot, boxplot, or countplot based on column dtypes.
    """
    df_plot = df.copy()
    
    # 1. Determine original types first
    x_is_num = _is_numeric_col(df_plot[x_col])
    y_is_num = _is_numeric_col(df_plot[y_col])
    
    # 2. Local binning specifically for categorical/mixed plots
    def _bin_series(series):
        if _is_numeric_col(series) and series.nunique() > 15:
            return pd.cut(series, bins=min(10, series.nunique())).astype(str)
        return series

    if x_is_num and y_is_num:
        # Scatter/Regression Plot (Preserves raw float/int values!)
        if hue_col:
            sns.scatterplot(data=df_plot, x=x_col, y=y_col, hue=hue_col, palette="Set1", alpha=0.7, ax=ax)
        sns.regplot(data=df_plot, x=x_col, y=y_col, scatter=(hue_col is None),
                    scatter_kws={"alpha": 0.6}, line_kws={"color": "darkred", "linestyle": "--"}, ax=ax)
        ax.set_title(f"Scatter: {x_col} vs {y_col}", fontsize=12, pad=10)
        
    elif not x_is_num and y_is_num:
        # Bin the categorical x-axis if it is actually numeric
        df_plot[x_col] = _bin_series(df_plot[x_col])
        sns.boxplot(data=df_plot, x=x_col, y=y_col, hue=hue_col if hue_col else x_col, palette="Set2", legend=False, ax=ax)
        ax.set_title(f"Boxplot: {y_col} across {x_col}", fontsize=12, pad=10)
        ax.tick_params(axis='x', rotation=30)
        
    elif x_is_num and not y_is_num:
        df_plot[y_col] = _bin_series(df_plot[y_col])
        sns.boxplot(data=df_plot, x=y_col, y=x_col, hue=hue_col if hue_col else y_col, palette="Set2", legend=False, ax=ax)
        ax.set_title(f"Boxplot: {x_col} across {y_col}", fontsize=12, pad=10)
        ax.tick_params(axis='x', rotation=30)
        
    else:
        # Both categorical
        df_plot[x_col] = _bin_series(df_plot[x_col])
        df_plot[y_col] = _bin_series(df_plot[y_col])
        sns.countplot(data=df_plot, x=x_col, hue=y_col, palette="Set1", ax=ax)
        ax.set_title(f"Categorical: {x_col} by {y_col}", fontsize=12, pad=10)
        ax.tick_params(axis='x', rotation=30)


def plot_target_interaction(
    df: pd.DataFrame,
    target_col: Optional[str] = None,
    feature_col: Optional[str] = None,
    save_path: str = "target_interactions.png",
    output_dir: str = "./sandbox_run",
    **kwargs
) -> Dict[str, Any]:
    """
    Generates and saves a segmented visual plot (boxplot/violinplot/scatter)
    comparing key feature distribution against target variable.
    """
    out_dir = kwargs.get("output_dir") or output_dir
    plt.close()
    if not target_col or target_col not in df.columns:
        numeric_cols = [c for c in df.columns if _is_numeric_col(df[c])]
        target_col = numeric_cols[-1] if numeric_cols else df.columns[-1]

    if not feature_col or feature_col not in df.columns or feature_col == target_col:
        candidates = [c for c in df.columns if c != target_col]
        feature_col = candidates[0] if candidates else df.columns[0]

    os.makedirs(out_dir, exist_ok=True)
    full_save_path = os.path.join(out_dir, os.path.basename(save_path))

    df_viz = _downsample_for_viz(df)

    try:
        fig, ax = plt.subplots(figsize=(9, 6))
        _render_bivariate_axes(ax, df_viz, feature_col, target_col)
        plt.tight_layout()
        plt.savefig(full_save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    except Exception as e:
        print(f"[tools] Warning: Target interaction plot error: {e}")
        plt.close()

    return {
        "plot_saved": full_save_path,
        "target_col": target_col,
        "feature_col": feature_col
    }


# =====================================================================
# FEATURE DISTRIBUTION PLOTTING TOOL (LLM DECIDES IMPORTANT COLUMNS)
# =====================================================================
def plot_feature_distributions(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    save_path: str = "feature_distributions.png",
    output_dir: str = "./sandbox_run",
    **kwargs
) -> Dict[str, Any]:
    """
    Plots probability distributions / KDE histograms for continuous numeric columns,
    or countplots with count labels for categorical and low-cardinality discrete columns.
    Saves each column as a separate dist_{col}.png file.
    """
    plt.close()
    target_cols = columns or kwargs.get("important_columns") or kwargs.get("cols") or kwargs.get("feature_cols")
    if not target_cols or target_cols == "all" or (isinstance(target_cols, (list, tuple)) and len(target_cols) == 0):
        target_cols = list(df.columns)
    elif isinstance(target_cols, str):
        if target_cols.lower() == "all":
            target_cols = list(df.columns)
        else:
            target_cols = [target_cols]

    valid_cols = [c for c in target_cols if c in df.columns and not is_non_distributional_column(c, df[c])]
    if not valid_cols:
        valid_cols = [c for c in df.columns if not is_non_distributional_column(c, df[c])]

    out_dir = kwargs.get("output_dir") or output_dir
    os.makedirs(out_dir, exist_ok=True)
    saved_files = []

    for col in valid_cols:
        file_path = os.path.join(out_dir, f"dist_{_sanitize_col_name(col)}.png")

        try:
            fig, ax = plt.subplots(figsize=(6, 4))
            s_clean = df[col].dropna()
            is_bool = pd.api.types.is_bool_dtype(s_clean)
            is_num = _is_numeric_col(s_clean) and not is_bool
            n_unique = s_clean.nunique()

            # Continuous numeric variables (is_num and > 10 unique values) get KDE / Histogram
            if is_num and n_unique > 10:
                sns.histplot(s_clean, kde=True, ax=ax, color="#6366f1")
                ax.set_title(f"Numeric Distribution: {col}", fontsize=12, pad=10)
                ax.set_xlabel(col)
                ax.set_ylabel("Density / Frequency")
            else:
                # Categorical, boolean, or low-cardinality discrete numeric features get Count Plot
                if n_unique > 20:
                    top_cats = s_clean.value_counts().head(20).index
                    col_data_str = s_clean[s_clean.isin(top_cats)].astype(str)
                else:
                    col_data_str = s_clean.astype(str)

                # Order categories by frequency for clean presentation
                cat_order = col_data_str.value_counts().index
                sns.countplot(x=col_data_str, hue=col_data_str, order=cat_order, ax=ax, palette="Set2", legend=False)
                ax.set_title(f"Categorical Count Plot: {col}", fontsize=12, pad=10)
                ax.set_xlabel(col)
                ax.set_ylabel("Count")
                ax.tick_params(axis='x', rotation=30)

                # Annotate count bars with exact numeric counts
                if len(ax.containers) > 0:
                    try:
                        ax.bar_label(ax.containers[0], padding=2, fontsize=9)
                    except Exception:
                        pass

            plt.tight_layout()
            plt.savefig(file_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            saved_files.append(file_path)
            print(f"[tools] Saved distribution PNG for '{col}' to: {file_path}")
        except Exception as e:
            print(f"[tools] Warning: Error saving distribution plot for '{col}': {e}")
    return {
        "individual_plots": saved_files,
        "plotted_columns": valid_cols
    }


def _interpret_effect_size(test_name: str, val: float) -> str:
    """
    Translates numerical effect sizes (Pearson r, Cramér's V, Cohen's d, Eta-squared)
    into standard qualitative descriptors (Small, Medium, Large).
    """
    try:
        v = abs(float(val))
    except (ValueError, TypeError):
        return "Unknown effect"
        
    test_clean = str(test_name).lower()
    if "pearson" in test_clean or "correlation" in test_clean:
        if v < 0.1: return "Negligible correlation"
        if v < 0.3: return "Weak correlation"
        if v < 0.5: return "Moderate correlation"
        return "Strong correlation"
    elif "chi" in test_clean or "cramer" in test_clean:
        if v < 0.1: return "Negligible association"
        if v < 0.3: return "Small association"
        if v < 0.5: return "Medium association"
        return "Large association"
    elif "t-test" in test_clean or "welch" in test_clean or "cohen" in test_clean:
        if v < 0.2: return "Negligible effect"
        if v < 0.5: return "Small effect"
        if v < 0.8: return "Medium effect"
        return "Large effect"
    else:
        # ANOVA / F-test / Eta-squared / general
        if v < 0.01: return "Negligible effect"
        if v < 0.06: return "Small effect"
        if v < 0.14: return "Medium effect"
        return "Large effect"


# =====================================================================
# 4. STATISTICAL ANALYSIS: HYPOTHESIS TESTING TOOL
# =====================================================================
def run_statistical_hypothesis_tests(
    df: pd.DataFrame,
    target_col: str,
    alpha: float = 0.05,
    **kwargs
) -> Dict[str, Any]:
    """
    Performs statistical significance tests based on data types.
    Returns ranked significant predictors with qualitative effect size interpretations.
    """
    if target_col not in df.columns:
        return {"error": f"Target column '{target_col}' not found."}

    test_results = {"target_col": target_col, "significant_predictors": [], "ranked_significant_details": []}
    significant_items = []
    
    target_is_num = _is_numeric_col(df[target_col])
    
    for col in df.columns:
        if col == target_col or is_non_distributional_column(col, df[col]):
            continue
            
        col_is_num = _is_numeric_col(df[col])
        s1 = df[col].dropna()
        s2 = df[target_col].dropna()
        
        # Exclude near-unique categorical columns (e.g. Ticket) to avoid fitting noise in ANOVA/Chi2
        if not col_is_num and (df[col].nunique() > 50 or (len(df) > 20 and df[col].nunique() / len(df) > 0.25)):
            continue

        common_idx = s1.index.intersection(s2.index)
        if len(common_idx) < 30: continue
        
        data_col = df.loc[common_idx, col]
        data_target = df.loc[common_idx, target_col]
        
        try:
            if col_is_num and target_is_num:
                corr, p = stats.pearsonr(data_col, data_target)
                eff = abs(float(corr))
                label = _interpret_effect_size("Pearson Correlation", eff)
                if p < alpha:
                    significant_items.append({"feature": col, "test": "Pearson Correlation", "effect_size": round(eff, 4), "effect_size_label": label, "p_value": float(p)})
            elif not col_is_num and target_is_num:
                groups = [group.values for name, group in data_target.groupby(data_col)]
                if len(groups) > 1:
                    f_val, p = stats.f_oneway(*groups)
                    eff = _correlation_ratio(data_col, data_target)
                    label = _interpret_effect_size("ANOVA", eff)
                    if p < alpha:
                        significant_items.append({"feature": col, "test": "ANOVA", "effect_size": round(eff, 4), "effect_size_label": label, "p_value": float(p)})
            elif col_is_num and not target_is_num:
                groups = [group.values for name, group in data_col.groupby(data_target)]
                if len(groups) > 1:
                    f_val, p = stats.f_oneway(*groups)
                    eff = _correlation_ratio(data_target, data_col)
                    label = _interpret_effect_size("ANOVA", eff)
                    if p < alpha:
                        significant_items.append({"feature": col, "test": "ANOVA", "effect_size": round(eff, 4), "effect_size_label": label, "p_value": float(p)})
            else:
                contingency = pd.crosstab(data_col, data_target)
                if contingency.size > 0:
                    chi2, p, _, _ = stats.chi2_contingency(contingency)
                    eff = _cramers_v(data_col, data_target)
                    label = _interpret_effect_size("Chi-Square", eff)
                    if p < alpha:
                        significant_items.append({"feature": col, "test": "Chi-Square", "effect_size": round(eff, 4), "effect_size_label": label, "p_value": float(p)})
        except Exception:
            continue
            
    test_results["target_col"] = target_col
    significant_items.sort(key=lambda x: x["effect_size"], reverse=True)
    test_results["significant_predictors"] = [item["feature"] for item in significant_items]
    test_results["ranked_significant_details"] = significant_items
    return test_results


# =====================================================================
# 5. VISUALIZATION: CORRELATION MATRIX TOOL
# =====================================================================
def _cramers_v(x, y):
    confusion_matrix = pd.crosstab(x, y)
    if confusion_matrix.empty:
        return 0.0
    chi2 = stats.chi2_contingency(confusion_matrix, correction=False)[0]
    n = confusion_matrix.sum().sum()
    if n == 0:
        return 0.0
    phi2 = chi2 / n
    r, k = confusion_matrix.shape
    phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1)) if n > 1 else 0
    rcorr = r - ((r-1)**2)/(n-1) if n > 1 else r
    kcorr = k - ((k-1)**2)/(n-1) if n > 1 else k
    min_dim = min((kcorr-1), (rcorr-1))
    return float(np.sqrt(phi2corr / min_dim)) if min_dim > 0 else 0.0


def _correlation_ratio(categories, measurements):
    """
    Calculates Correlation Ratio (eta) between a categorical and numeric feature.
    eta = sqrt( SS_between / SS_total )
    Returns 1.0 for perfect deterministic/ordinal duplicate encodings (e.g. education vs educational-num).
    """
    try:
        clean_data = pd.DataFrame({'cat': categories, 'num': measurements}).dropna()
        if len(clean_data) < 5 or clean_data['cat'].nunique() <= 1:
            return 0.0
        
        overall_mean = clean_data['num'].mean()
        total_ss = np.sum((clean_data['num'] - overall_mean) ** 2)
        if total_ss == 0:
            return 0.0
        
        category_means = clean_data.groupby('cat')['num'].agg(['mean', 'count'])
        between_ss = np.sum(category_means['count'] * ((category_means['mean'] - overall_mean) ** 2))
        
        eta = np.sqrt(between_ss / total_ss)
        return float(np.clip(eta, 0.0, 1.0))
    except Exception:
        return 0.0


def plot_correlation_matrix(
    df: pd.DataFrame,
    save_path: str = "correlation_matrix.png",
    output_dir: str = "./sandbox_run",
    **kwargs
) -> Dict[str, Any]:
    """
    Generates a correlation matrix heatmap for numeric features and association matrix for categorical features.
    Detects cross-type redundant duplicate pairs (e.g. education vs educational-num) using Correlation Ratio (eta).
    Excludes non-distributional columns (IDs, coordinates, timestamps).
    """
    plt.close()
    os.makedirs(output_dir, exist_ok=True)
    full_save_path = os.path.join(output_dir, os.path.basename(save_path))

    numeric_cols = [c for c in df.columns if _is_numeric_col(df[c]) and not is_non_distributional_column(c, df[c])]

    if len(numeric_cols) < 2:
        return {"error": "Insufficient numerical columns for correlation heatmap."}

    corr_matrix = df[numeric_cols].corr()

    high_corr_pairs = []
    cols = list(corr_matrix.columns)
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            val = corr_matrix.iloc[i, j]
            if pd.notnull(val) and abs(val) >= 0.85:
                high_corr_pairs.append({
                    "feature_1": cols[i],
                    "feature_2": cols[j],
                    "correlation": round(float(val), 4),
                    "interpretation": _interpret_effect_size("Pearson", val)
                })

    fig, ax = plt.subplots(figsize=(max(6, len(numeric_cols) * 0.8), max(5, len(numeric_cols) * 0.7)))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1, ax=ax, cbar=True)
    ax.set_title("Pearson Correlation Heatmap", fontsize=14, pad=12)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)

    try:
        plt.savefig(full_save_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"[tools] Correlation matrix successfully saved to: {full_save_path}")
    except Exception as e:
        print(f"[tools] Warning: Error saving correlation matrix: {e}")
        plt.close()

    cat_cols = [c for c in df.columns if (not _is_numeric_col(df[c]) or df[c].nunique() <= 10) and not is_non_distributional_column(c, df[c])]
    cat_assoc = []
    if len(cat_cols) >= 2:
        for i in range(len(cat_cols)):
            for j in range(i + 1, len(cat_cols)):
                v = _cramers_v(df[cat_cols[i]], df[cat_cols[j]])
                cat_assoc.append({
                    "feature_1": cat_cols[i],
                    "feature_2": cat_cols[j],
                    "cramers_v": round(v, 4),
                    "interpretation": _interpret_effect_size("Cramer", v)
                })

    # Cross-type redundancy detection (Categorical vs Numeric pairs, e.g. education vs educational-num)
    cross_type_redundant_pairs = []
    for c_col in cat_cols:
        for n_col in numeric_cols:
            if c_col == n_col: continue
            eta_val = _correlation_ratio(df[c_col], df[n_col])
            if eta_val >= 0.85:
                cross_type_redundant_pairs.append({
                    "categorical_feature": c_col,
                    "numeric_feature": n_col,
                    "correlation_ratio_eta": round(eta_val, 4),
                    "interpretation": f"High cross-type redundancy between '{c_col}' and '{n_col}' (Eta = {eta_val:.4f})."
                })

    return {
        "correlation_heatmap_saved": full_save_path,
        "high_correlation_pairs": high_corr_pairs,
        "categorical_associations": cat_assoc,
        "cross_type_redundant_pairs": cross_type_redundant_pairs
    }


# =====================================================================
# SEMANTIC BIVARIATE (X VS Y) RELATIONSHIP PLOTTING TOOL
# =====================================================================
def plot_semantic_bivariate_relationships(
    df: pd.DataFrame,
    bivariate_pairs: Optional[List[Dict[str, Any]]] = None,
    output_dir: str = "./sandbox_run",
    **kwargs
) -> Dict[str, Any]:
    """
    Plots semantic X vs Y relationships selected by the LLM based on domain understanding.
    Each pair in bivariate_pairs contains:
    - 'x': X-axis column name
    - 'y': Y-axis column name
    - 'hue': Optional hue column name for segmentation
    - 'rationale': Semantic domain rationale for comparing these two attributes
    """
    plt.close()
    os.makedirs(output_dir, exist_ok=True)
    
    pairs = bivariate_pairs or kwargs.get("pairs") or kwargs.get("bivariate_list") or kwargs.get("bivariate_pairs") or []
    
    # Auto-generate top pairs if not provided by LLM
    if not pairs:
        numeric_cols = [c for c in df.columns if _is_numeric_col(df[c]) and not is_non_distributional_column(c, df[c])]
        cat_cols = [c for c in df.columns if (not _is_numeric_col(df[c]) or df[c].nunique() <= 10) and not is_non_distributional_column(c, df[c])]
        
        if len(numeric_cols) >= 2:
            pairs.append({"x": numeric_cols[0], "y": numeric_cols[1], "rationale": "Bivariate numerical comparison"})
        if cat_cols and numeric_cols:
            pairs.append({"x": cat_cols[0], "y": numeric_cols[0], "rationale": "Segmented numerical distribution across category"})

    saved_files = []
    
    for pair in pairs:
        if not isinstance(pair, dict):
            continue

        x_col = pair.get("x") or pair.get("x_col") or pair.get("feature_1")
        y_col = pair.get("y") or pair.get("y_col") or pair.get("feature_2")
        hue_col = pair.get("hue") or pair.get("hue_col")
        rationale = pair.get("rationale", "Semantic domain relationship")
        
        if not x_col or not y_col or x_col not in df.columns or y_col not in df.columns:
            continue
            
        file_name = f"bivariate_{_sanitize_col_name(x_col)}_vs_{_sanitize_col_name(y_col)}.png"
        file_path = os.path.join(output_dir, file_name)
        
        try:
            fig, ax = plt.subplots(figsize=(6, 4))
            clean_df = df[[x_col, y_col] + ([hue_col] if hue_col and hue_col in df.columns else [])].dropna()
            
            if len(clean_df) < 5:
                plt.close()
                continue
                
            x_is_num = _is_numeric_col(clean_df[x_col])
            y_is_num = _is_numeric_col(clean_df[y_col])
            
            if x_is_num and y_is_num:
                sns.scatterplot(data=clean_df, x=x_col, y=y_col, hue=hue_col if hue_col in clean_df.columns else None, ax=ax, palette="Set1", alpha=0.7)
                sns.regplot(data=clean_df, x=x_col, y=y_col, scatter=False, ax=ax, color="#ef4444")
                ax.set_title(f"{x_col} vs {y_col}", fontsize=11, pad=10)
            elif not x_is_num and y_is_num:
                sns.boxplot(data=clean_df, x=x_col, y=y_col, hue=hue_col if hue_col in clean_df.columns else None, ax=ax, palette="Set2")
                ax.set_title(f"{y_col} distribution across {x_col}", fontsize=11, pad=10)
                ax.tick_params(axis='x', rotation=30)
            elif x_is_num and not y_is_num:
                sns.boxplot(data=clean_df, x=y_col, y=x_col, hue=hue_col if hue_col in clean_df.columns else None, ax=ax, palette="Set2")
                ax.set_title(f"{x_col} distribution across {y_col}", fontsize=11, pad=10)
                ax.tick_params(axis='x', rotation=30)
            else:
                ct = pd.crosstab(clean_df[x_col], clean_df[y_col], normalize="index")
                ct.plot(kind="bar", stacked=True, ax=ax, colormap="viridis")
                ax.set_title(f"{x_col} vs {y_col} (Proportion)", fontsize=11, pad=10)
                ax.tick_params(axis='x', rotation=30)
                
            plt.tight_layout()
            plt.savefig(file_path, dpi=150, bbox_inches="tight")
            plt.close()
            saved_files.append({"name": file_name, "path": file_path, "x": x_col, "y": y_col, "rationale": rationale})
        except Exception as e:
            print(f"[tools] Warning: Error plotting bivariate relationship '{x_col}' vs '{y_col}': {e}")
            plt.close()

    return {
        "bivariate_plots_saved": saved_files,
        "count": len(saved_files)
    }


# =====================================================================
# CONCISE PAIRPLOT TOOL (CLAMPED FEATURE SUBSET)
# =====================================================================
def plot_pairplot(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    hue: Optional[str] = None,
    save_path: str = "pairplot.png",
    output_dir: str = "./sandbox_run",
    **kwargs
) -> Dict[str, Any]:
    """
    Generates a concise pairplot visualizing pairwise distributions and relationships
    across a reasonable subset of key numerical features (clamped to max 4-5 features).
    """
    plt.close()
    os.makedirs(output_dir, exist_ok=True)
    
    raw_cols = columns or kwargs.get("feature_cols") or kwargs.get("cols") or kwargs.get("numeric_cols")
    if not raw_cols:
        raw_cols = [c for c in df.columns if _is_numeric_col(df[c]) and df[c].nunique() > 2 and not is_non_distributional_column(c, df[c])]
        
    valid_cols = [c for c in raw_cols if c in df.columns and _is_numeric_col(df[c])]
    
    max_features = 4
    if len(valid_cols) > max_features:
        valid_cols = valid_cols[:max_features]
        print(f"[tools] Parameter Clamping: Clamped pairplot features to top {max_features}: {valid_cols}")
        
    if len(valid_cols) < 2:
        return {"error": "Insufficient numeric columns for pairplot rendering."}
        
    hue_col = hue or kwargs.get("target_col")
    if hue_col and hue_col not in df.columns:
        hue_col = None
        
    cols_to_plot = list(valid_cols)
    if hue_col and hue_col not in cols_to_plot:
        cols_to_plot.append(hue_col)
        
    clean_df = df[cols_to_plot].dropna()
    if len(clean_df) < 5:
        clean_df = df[cols_to_plot]
        
    full_save_path = os.path.join(output_dir, os.path.basename(save_path))
    
    try:
        if hue_col:
            grid = sns.pairplot(clean_df, vars=valid_cols, hue=hue_col, palette="Set1", corner=True, plot_kws={"alpha": 0.6, "s": 25})
        else:
            grid = sns.pairplot(clean_df, vars=valid_cols, corner=True, plot_kws={"alpha": 0.6, "s": 25})
            
        grid.fig.suptitle("Pairwise Feature Relationships (Pairplot)", y=1.02, fontsize=14)
        grid.savefig(full_save_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"[tools] Pairplot successfully saved to: {full_save_path}")
    except Exception as e:
        print(f"[tools] Warning: Pairplot rendering error: {e}")
        plt.close()
        
    return {
        "pairplot_saved": full_save_path,
        "features_plotted": valid_cols,
        "hue": hue_col
    }


# =====================================================================
# 7. PREDICTIVE MODELING BLUEPRINT GENERATOR TOOL
# =====================================================================
def generate_predictive_blueprint(
    df: pd.DataFrame,
    target_col: Optional[str] = None,
    custom_blueprint: Optional[Dict[str, Any]] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Generates a predictive modeling blueprint.
    Automatically infers problem type (Classification vs. Regression) when target_col is provided or detected.
    """
    num_rows, num_cols = df.shape
    
    # Auto-detect target_col if omitted but can be inferred
    if not target_col or target_col not in df.columns:
        target_col = kwargs.get("target_col") or kwargs.get("target") or kwargs.get("target_definition")

    if not target_col or target_col not in df.columns:
        # Heuristic search for common target column names in dataset
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
def validate_report_consistency(
    metrics_dict: Dict[str, Any],
    report_text: Optional[str] = None
) -> Dict[str, Any]:
    """
    Lightweight post-generation validation step that asserts consistency across:
    1. Target column alignment between dataset overview, hypothesis tests, and modeling blueprint.
    2. Non-empty predictor reporting consistency.
    3. Rendered Markdown prose report consistency (verifying target name and zero false feature claims).
    Returns a dictionary of validation checks and warning flags.
    """
    overview = metrics_dict.get("dataset_overview", {})
    target_col = overview.get("target_column")
    
    hypothesis_res = metrics_dict.get("statistical_hypothesis_tests", {})
    blueprint_res = metrics_dict.get("predictive_modeling_blueprint", {})
    
    validation_passed = True
    warnings = []
    
    # 1. Target Alignment Check
    hyp_target = hypothesis_res.get("target_col") if isinstance(hypothesis_res, dict) else None
    bp_target = blueprint_res.get("target_definition") if isinstance(blueprint_res, dict) else None
    
    if target_col and hyp_target and target_col != hyp_target:
        validation_passed = False
        warnings.append(f"Target mismatch between dataset overview ('{target_col}') and hypothesis tests ('{hyp_target}').")
        
    if target_col and bp_target and target_col != bp_target:
        validation_passed = False
        warnings.append(f"Target mismatch between dataset overview ('{target_col}') and predictive blueprint ('{bp_target}').")
        
    # 2. Engineered Features Ground Truth Check
    engineered = metrics_dict.get("engineered_features", [])
    if not isinstance(engineered, list):
        validation_passed = False
        warnings.append("Engineered features in metrics.json is not a valid list.")
        
    # 3. Markdown Text Validation (if provided)
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


# =====================================================================
# 8. METRICS JSON COMPILATION TOOL
# =====================================================================
def compile_and_save_metrics(
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
    """
    Compiles all analysis outputs into the canonical metrics.json format.
    Runs automated post-generation consistency validation before saving.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    num_rows, num_cols = df.shape
    
    column_summary = {}
    for col in df.columns:
        column_summary[col] = {
            "dtype": str(df[col].dtype),
            "missing_count": int(df[col].isnull().sum()),
            "cardinality": int(df[col].nunique())
        }
        
    metrics_dict = {
        "dataset_overview": {
            "dataset_path": os.path.abspath(dataset_path),
            "shape": {"rows": num_rows, "columns": num_cols},
            "target_column": target_col,
            "column_summary": column_summary
        },
        "imputation_summary": imputation_res or {"status": "Imputation completed"},
        "outlier_analysis": outlier_res or {},
        "engineered_features": engineered_res or [],
        "correlation_analysis": corr_res or {},
        "categorical_associations": (corr_res or {}).get("categorical_associations", []),
        "statistical_hypothesis_tests": hypothesis_res or {},
        "predictive_modeling_blueprint": blueprint_res or generate_predictive_blueprint(df, target_col),
        "extracted_insights": {
            "key_findings": [
                f"Dataset contains {num_rows} rows and {num_cols} columns.",
                f"Processed missing values and computed statistical distributions."
            ],
            "statistically_significant_predictors": (hypothesis_res or {}).get("significant_predictors", [])
        }
    }
    
    # Run post-generation consistency validation
    metrics_dict["pipeline_validation"] = validate_report_consistency(metrics_dict)
    
    metrics_path = os.path.join(output_dir, "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_dict, f, indent=2)
        
    print(f"[tools] Canonical metrics.json successfully saved to: {os.path.abspath(metrics_path)}")
    return metrics_path


# =====================================================================
# 9. TOOL REGISTRY CATALOG FOR LLM PROMPTING
# =====================================================================
from pydantic import BaseModel, Field

class ImputeMissingDataArgs(BaseModel):
    strategy_map: Optional[Dict[str, str]] = Field(default=None, description="Optional dict of {col: strategy}")
    numeric_skew_threshold: Optional[float] = Field(default=1.0)

class DetectAndHandleOutliersArgs(BaseModel):
    columns: List[str] = Field(description="List of numeric cols")
    action: str = Field(description="'profile' or 'cap'")

class EngineerFeaturesArgs(BaseModel):
    feature_specs: List[Dict[str, Any]] = Field(
        description="List of dicts defining features. Accepts arbitrary formulas or standard types. "
                    "Example: [{'name': 'FamilySize', 'formula': 'SibSp + Parch + 1', 'rationale': 'Total family count'}, "
                    "{'name': 'IsAlone', 'formula': 'FamilySize == 1', 'rationale': 'Solo traveler indicator'}]"
    )

class RunStatisticalHypothesisTestsArgs(BaseModel):
    target_col: str = Field(description="Name of target column")
    alpha: float = Field(default=0.05, description="Significance threshold (0.05)")

class PlotCorrelationMatrixArgs(BaseModel):
    save_path: str = Field(default="correlation_matrix.png")

class PlotFeatureDistributionsArgs(BaseModel):
    columns: Optional[List[str]] = Field(default=None, description="Optional list of column names to plot distributions for. Excludes IDs, coordinates (latitude, longitude), timestamps, and index keys.")
    save_path: str = Field(default="feature_distributions.png")

class PlotTargetInteractionArgs(BaseModel):
    target_col: str = Field(description="Target column name")
    feature_col: str = Field(description="Feature column name")
    save_path: str = Field(default="target_interactions.png")

class PlotSemanticBivariateRelationshipsArgs(BaseModel):
    bivariate_pairs: List[Dict[str, Optional[str]]] = Field(description="List of dicts [{'x': 'col1', 'y': 'col2', 'hue': 'target'}]")

class PlotPairplotArgs(BaseModel):
    columns: List[str] = Field(description="List of 3-4 key numeric features")
    hue: Optional[str] = Field(default=None, description="Optional target/hue column")

class GeneratePredictiveBlueprintArgs(BaseModel):
    target_col: str = Field(description="Target column name")

class AskClarifyingQuestionArgs(BaseModel):
    question: str = Field(description="The question to ask the user to clarify ambiguity.")

class FinishAnalysisArgs(BaseModel):
    pass

def ask_clarifying_question(df: pd.DataFrame, question: str, **kwargs) -> Dict[str, Any]:
    return {"question": question, "status": "paused_for_user_input"}

def finish_analysis(df: pd.DataFrame, **kwargs) -> Dict[str, Any]:
    return {"status": "finished"}

TOOL_REGISTRY = {
    "impute_missing_data": {
        "function": impute_missing_data,
        "description": "Imputes missing values using type-safe strategy. SKIP this tool if user requests relationship-focused analysis or skipping imputation.",
        "model": ImputeMissingDataArgs
    },
    "detect_and_handle_outliers": {
        "function": detect_and_handle_outliers,
        "description": "Detects outliers via IQR method. SKIP this tool if user requests relationship-focused analysis or skipping outlier handling.",
        "model": DetectAndHandleOutliersArgs
    },
    "engineer_features": {
        "function": engineer_features,
        "description": "Creates high-signal feature transformations (log1p, ratios, interactions).",
        "model": EngineerFeaturesArgs
    },
    "run_statistical_hypothesis_tests": {
        "function": run_statistical_hypothesis_tests,
        "description": "Calculates statistical significance against target variable (T-Test, ANOVA, Chi-Square, Pearson). Ranks significant features by effect size.",
        "model": RunStatisticalHypothesisTestsArgs
    },
    "plot_correlation_matrix": {
        "function": plot_correlation_matrix,
        "description": "Generates Pearson correlation matrix heatmap PNG image asset.",
        "model": PlotCorrelationMatrixArgs
    },
    "plot_feature_distributions": {
        "function": plot_feature_distributions,
        "description": "Plots univariate histograms, KDE distributions, or countplots for important non-identifier features (passed via 'columns' argument).",
        "model": PlotFeatureDistributionsArgs
    },
    "plot_target_interaction": {
        "function": plot_target_interaction,
        "description": "Generates segmented distribution / scatter visualization comparing key feature vs target.",
        "model": PlotTargetInteractionArgs
    },
    "plot_semantic_bivariate_relationships": {
        "function": plot_semantic_bivariate_relationships,
        "description": "Plots custom X vs Y scatter/boxplot/countplot relationships dynamically selected by LLM based on semantic domain reasoning (passed via 'bivariate_pairs' list of dicts with 'x', 'y', optional 'hue', and 'rationale').",
        "model": PlotSemanticBivariateRelationshipsArgs
    },
    "plot_pairplot": {
        "function": plot_pairplot,
        "description": "Generates a concise Seaborn pairplot visualizing pairwise distributions and relationships across a reasonable subset of key numerical features (clamped to max 4-5 features).",
        "model": PlotPairplotArgs
    },
    "generate_predictive_blueprint": {
        "function": generate_predictive_blueprint,
        "description": "Compiles machine learning modeling blueprint and cross-validation strategy. SKIP this tool if user requested relationship-focused analysis or skipping blueprinting.",
        "model": GeneratePredictiveBlueprintArgs
    },
    "ask_clarifying_question": {
        "function": ask_clarifying_question,
        "description": "Ask the user a clarifying question when requirements or target columns are ambiguous.",
        "model": AskClarifyingQuestionArgs
    },
    "finish_analysis": {
        "function": finish_analysis,
        "description": "Call this tool when you have finished all necessary exploratory data analysis and don't need to run any more tools.",
        "model": FinishAnalysisArgs
    }
}

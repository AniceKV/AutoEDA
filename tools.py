import os
import json
import math
import re
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List, Optional, Tuple

sns.set_theme(style="whitegrid")


# =====================================================================
# STATEFUL EXECUTION MEMORY & DATA VERSION CONTROL (DVC PATTERN)
# =====================================================================
class StatefulDataStore:
    """
    Manages stateful execution memory and DataFrame version control (DVC pattern).
    Maintains checkpoints (df_state_v0.csv, df_state_v1.csv, ...) inside workspace_dir.
    Allows automatic rollback if a tool step corrupts or invalidates the dataset.
    """
    def __init__(self, workspace_dir: str = "./sandbox_run"):
        self.workspace_dir = workspace_dir
        self.version = 0
        self.history: List[Dict[str, Any]] = []
        os.makedirs(self.workspace_dir, exist_ok=True)

    def set_initial_state(self, df: pd.DataFrame) -> str:
        self.version = 0
        self.history = []
        filename = f"df_state_v{self.version}.csv"
        path = os.path.join(self.workspace_dir, filename)
        df.to_csv(path, index=False)
        self.history.append({"version": 0, "path": path, "rows": len(df), "cols": len(df.columns), "action": "initial_load"})
        print(f"[DataStore] Initialized state v0 ({len(df)} rows, {len(df.columns)} cols) -> {path}")
        return path

    def save_checkpoint(self, df: pd.DataFrame, step_name: str) -> str:
        if df is None or len(df) == 0 or len(df.columns) == 0:
            raise ValueError(f"Cannot checkpoint invalid or empty DataFrame after step '{step_name}'.")
        self.version += 1
        filename = f"df_state_v{self.version}.csv"
        path = os.path.join(self.workspace_dir, filename)
        df.to_csv(path, index=False)
        self.history.append({"version": self.version, "path": path, "rows": len(df), "cols": len(df.columns), "action": step_name})
        print(f"[DataStore] Saved checkpoint v{self.version} after '{step_name}' ({len(df)} rows, {len(df.columns)} cols) -> {path}")
        return path

    def rollback(self) -> Tuple[pd.DataFrame, int]:
        if len(self.history) <= 1:
            print("[DataStore] Cannot rollback further. At initial state v0.")
            v0_path = self.history[0]["path"]
            return pd.read_csv(v0_path), 0
        
        bad_state = self.history.pop()
        print(f"[DataStore] Rolling back from corrupted state v{bad_state['version']} ({bad_state['action']})...")
        
        latest_state = self.history[-1]
        self.version = latest_state["version"]
        restored_df = pd.read_csv(latest_state["path"])
        print(f"[DataStore] Successfully rolled back to state v{self.version} ({latest_state['action']}) from {latest_state['path']}")
        return restored_df, self.version

    def purge_intermediate_states(self):
        """
        Deletes intermediate checkpoint CSV files to save disk space,
        keeping only the initial load (v0) and the final active version.
        """
        if len(self.history) <= 2:
            return
            
        final_state = self.history[-1]
        for state in self.history[1:-1]:
            path = state.get("path")
            if path and os.path.exists(path):
                try:
                    os.remove(path)
                except Exception as e:
                    print(f"[DataStore] Warning: Could not purge file {path}: {e}")
                    
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
        if pd.api.types.is_numeric_dtype(df_imputed[col]):
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

    raw_cols = columns or [c for c in df_out.columns if pd.api.types.is_numeric_dtype(df_out[c])]
    target_cols = [c for c in raw_cols if c in df_out.columns and pd.api.types.is_numeric_dtype(df_out[c])]
    
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
def engineer_features(
    df: pd.DataFrame,
    feature_specs: Optional[List[Dict[str, Any]]] = None,
    target_col: Optional[str] = None,
    **kwargs
) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
    """
    Creates high-signal domain features safely.
    Auto-detects common transformations if feature_specs is empty:
    - Log transforms for right-skewed variables (skew > 1.5)
    - Interaction features for strongly correlated pairs
    - Ratio features
    """
    df_feat = df.copy()
    engineered_summary = []
    
    specs = feature_specs or []
    
    # Auto-generate features if specs not provided
    if not specs:
        numeric_cols = [c for c in df_feat.columns if pd.api.types.is_numeric_dtype(df_feat[c]) and c != target_col and df_feat[c].nunique() > 2]
        
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
        fname = spec.get("name", "engineered_feature")
        ftype = spec.get("type", "custom").lower()
        rationale = spec.get("rationale", "High-signal feature engineering transformation")
        
        # Extract column references gracefully regardless of key names used by LLM
        cols = spec.get("columns") or spec.get("source_cols") or []
        scol = spec.get("source_col") or spec.get("column") or (cols[0] if cols else None)
        
        try:
            if ftype in ["log1p", "log"]:
                if scol and scol in df_feat.columns:
                    df_feat[fname] = np.log1p(np.maximum(0, pd.to_numeric(df_feat[scol], errors="coerce").fillna(0)))
                    formula = f"np.log1p({scol})"
                else:
                    continue
                
            elif ftype == "ratio":
                num = spec.get("numerator") or (cols[0] if len(cols) >= 1 else None)
                den = spec.get("denominator") or (cols[1] if len(cols) >= 2 else None)
                if num and den and num in df_feat.columns and den in df_feat.columns:
                    den_series = pd.to_numeric(df_feat[den], errors="coerce").fillna(0)
                    num_series = pd.to_numeric(df_feat[num], errors="coerce").fillna(0)
                    df_feat[fname] = num_series / (den_series.abs() + 1e-5)
                    formula = f"{num} / ({den} + eps)"
                else:
                    continue
                    
            elif ftype in ["product", "interaction", "multiply"]:
                if len(cols) >= 2 and all(c in df_feat.columns for c in cols[:2]):
                    c1_series = pd.to_numeric(df_feat[cols[0]], errors="coerce").fillna(0)
                    c2_series = pd.to_numeric(df_feat[cols[1]], errors="coerce").fillna(0)
                    df_feat[fname] = c1_series * c2_series
                    formula = f"{cols[0]} * {cols[1]}"
                else:
                    continue
                    
            elif ftype == "sum":
                valid_cols = [c for c in cols if c in df_feat.columns]
                if valid_cols:
                    df_feat[fname] = df_feat[valid_cols].apply(pd.to_numeric, errors="coerce").fillna(0).sum(axis=1)
                    formula = f"sum({', '.join(valid_cols)})"
                else:
                    continue
                
            else:
                continue
                
            corr_with_target = None
            if target_col and target_col in df_feat.columns and pd.api.types.is_numeric_dtype(df_feat[target_col]):
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
def run_statistical_hypothesis_tests(
    df: pd.DataFrame,
    target_col: Optional[str] = None,
    feature_cols: Optional[List[str]] = None,
    alpha: float = 0.05,
    **kwargs
) -> Dict[str, Any]:
    """
    Automates statistical significance testing against target_col with defensive parameter clamping.
    - Numerical Target + Numerical Feature: Pearson Correlation test
    - Categorical Target + Categorical Feature: Chi-Square Test
    - Binary Target + Numerical Feature: Two-Sample T-test / Mann-Whitney U
    - Multiclass Target + Numerical Feature: One-way ANOVA
    """
    # Parameter Clamping for alpha
    try:
        alpha = max(0.0001, min(0.5, float(alpha)))
    except (ValueError, TypeError):
        alpha = 0.05

    if not target_col or target_col not in df.columns:
        numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        target_col = numeric_cols[-1] if numeric_cols else df.columns[-1]

    raw_targets = feature_cols or [c for c in df.columns if c != target_col]
    targets_to_test = [c for c in raw_targets if c in df.columns and c != target_col]
    
    test_results = {}
    significant_predictors = []
    
    target_is_num = pd.api.types.is_numeric_dtype(df[target_col])
    target_cardinality = df[target_col].nunique()
    
    for col in targets_to_test:
        col_is_num = pd.api.types.is_numeric_dtype(df[col])
        clean_data = df[[target_col, col]].dropna()
        if len(clean_data) < 5:
            continue
            
        try:
            if target_is_num and col_is_num:
                # Pearson Correlation Test
                r_val, p_val = stats.pearsonr(clean_data[target_col], clean_data[col])
                test_name = "Pearson Correlation Test"
                statistic = float(r_val)
                interpretation = f"Pearson r = {r_val:.4f}, p = {p_val:.4e}."
                
            elif not target_is_num and not col_is_num:
                # Chi-Square Test of Independence
                contingency = pd.crosstab(clean_data[target_col], clean_data[col])
                chi2, p_val, dof, ex = stats.chi2_contingency(contingency)
                test_name = "Chi-Square Test of Independence"
                statistic = float(chi2)
                interpretation = f"Chi2 = {chi2:.4f}, dof = {dof}, p = {p_val:.4e}."
                
            elif not target_is_num and col_is_num:
                # Feature is Numerical, Target is Categorical
                groups = [group[col].dropna().values for name, group in clean_data.groupby(target_col)]
                groups = [g for g in groups if len(g) >= 2]
                
                if len(groups) == 2:
                    t_stat, p_val = stats.ttest_ind(groups[0], groups[1], equal_var=False)
                    test_name = "Two-Sample Welch T-Test"
                    statistic = float(t_stat)
                    interpretation = f"T-statistic = {t_stat:.4f}, p = {p_val:.4e}."
                elif len(groups) > 2:
                    f_stat, p_val = stats.f_oneway(*groups)
                    test_name = "One-Way ANOVA"
                    statistic = float(f_stat)
                    interpretation = f"F-statistic = {f_stat:.4f}, p = {p_val:.4e}."
                else:
                    continue
            else:
                # Target is Numerical, Feature is Categorical
                groups = [group[target_col].dropna().values for name, group in clean_data.groupby(col)]
                groups = [g for g in groups if len(g) >= 2]
                
                if len(groups) == 2:
                    t_stat, p_val = stats.ttest_ind(groups[0], groups[1], equal_var=False)
                    test_name = "Two-Sample Welch T-Test"
                    statistic = float(t_stat)
                    interpretation = f"T-statistic = {t_stat:.4f}, p = {p_val:.4e}."
                elif len(groups) > 2:
                    f_stat, p_val = stats.f_oneway(*groups)
                    test_name = "One-Way ANOVA"
                    statistic = float(f_stat)
                    interpretation = f"F-statistic = {f_stat:.4f}, p = {p_val:.4e}."
                else:
                    continue
                    
            p_val_float = float(p_val) if pd.notnull(p_val) else 1.0
            is_sig = p_val_float < alpha
            
            if is_sig:
                significant_predictors.append(col)
                
            test_results[col] = {
                "test_name": test_name,
                "statistic": round(statistic, 4),
                "p_value": p_val_float,
                "is_statistically_significant": is_sig,
                "interpretation": interpretation + (" (Statistically Significant)" if is_sig else " (Not Significant)")
            }
        except Exception as e:
            test_results[col] = {
                "test_name": "Hypothesis Test",
                "error": str(e),
                "p_value": 1.0,
                "is_statistically_significant": False,
                "interpretation": f"Could not perform test: {e}"
            }

    test_results["significant_predictors"] = significant_predictors
    return test_results


# =====================================================================
# 5. VISUALIZATION: CORRELATION MATRIX TOOL
# =====================================================================
def plot_correlation_matrix(
    df: pd.DataFrame,
    numeric_cols: Optional[List[str]] = None,
    save_path: str = "correlation_matrix.png",
    output_dir: str = "./sandbox_run"
) -> Dict[str, Any]:
    """
    Computes Pearson correlation matrix, saves styled heatmap asset,
    and extracts top positive/negative correlations.
    """
    plt.close("all")
    target_cols = numeric_cols or [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and df[c].nunique() > 1]
    
    if len(target_cols) < 2:
        return {"error": "Insufficient numeric columns for correlation analysis."}
        
    corr_matrix = df[target_cols].corr()
    
    os.makedirs(output_dir, exist_ok=True)
    full_save_path = os.path.join(output_dir, os.path.basename(save_path))

    try:
        fig, ax = plt.subplots(figsize=(max(8, len(target_cols) * 0.8), max(6, len(target_cols) * 0.7)))
        sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", center=0, square=True, linewidths=0.5, ax=ax)
        ax.set_title("Pearson Correlation Matrix", fontsize=14, pad=12)
        plt.tight_layout()
        plt.savefig(full_save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    except Exception as e:
        print(f"[tools] Warning: Heatmap rendering error: {e}")
        plt.close("all")

    # Extract top positive & negative correlation pairs
    pairs = []
    for i in range(len(target_cols)):
        for j in range(i + 1, len(target_cols)):
            c1, c2 = target_cols[i], target_cols[j]
            val = corr_matrix.loc[c1, c2]
            if pd.notnull(val):
                pairs.append({"feature_1": c1, "feature_2": c2, "correlation": round(float(val), 4)})
                
    pairs_sorted = sorted(pairs, key=lambda x: abs(x["correlation"]), reverse=True)
    
    return {
        "heatmap_saved": full_save_path,
        "top_correlations": pairs_sorted[:10],
        "correlation_matrix_text": corr_matrix.round(3).to_dict()
    }


# =====================================================================
# 6. VISUALIZATION: TARGET INTERACTION TOOL
# =====================================================================
def plot_target_interaction(
    df: pd.DataFrame,
    target_col: Optional[str] = None,
    feature_col: Optional[str] = None,
    save_path: str = "target_interactions.png",
    output_dir: str = "./sandbox_run"
) -> Dict[str, Any]:
    """
    Generates and saves a segmented visual plot (boxplot/violinplot/scatter)
    comparing key feature distribution against target variable.
    """
    plt.close("all")
    if not target_col or target_col not in df.columns:
        numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        target_col = numeric_cols[-1] if numeric_cols else df.columns[-1]

    if not feature_col or feature_col not in df.columns or feature_col == target_col:
        candidates = [c for c in df.columns if c != target_col]
        feature_col = candidates[0] if candidates else df.columns[0]
        
    os.makedirs(output_dir, exist_ok=True)
    full_save_path = os.path.join(output_dir, os.path.basename(save_path))

    try:
        fig, ax = plt.subplots(figsize=(9, 6))
        target_is_num = pd.api.types.is_numeric_dtype(df[target_col])
        feat_is_num = pd.api.types.is_numeric_dtype(df[feature_col])
        
        if target_is_num and feat_is_num:
            sns.regplot(data=df, x=feature_col, y=target_col, scatter_kws={"alpha": 0.6}, line_kws={"color": "red"}, ax=ax)
            ax.set_title(f"Scatter Interaction: {feature_col} vs {target_col}", fontsize=13)
        elif not target_is_num and feat_is_num:
            sns.boxplot(data=df, x=target_col, y=feature_col, palette="Set2", ax=ax)
            ax.set_title(f"Boxplot Segmented by Target: {feature_col} vs {target_col}", fontsize=13)
        elif target_is_num and not feat_is_num:
            # FIX: Plot Numerical Target vs Categorical Feature using boxplots
            sns.boxplot(data=df, x=feature_col, y=target_col, palette="Set2", ax=ax)
            ax.set_title(f"Boxplot Segmented by Feature: {target_col} vs {feature_col}", fontsize=13)
            ax.tick_params(axis='x', rotation=30)
        else: # Both are categorical
            sns.countplot(data=df, x=feature_col, hue=target_col, palette="Set1", ax=ax)
            ax.set_title(f"Categorical Interaction: {feature_col} by {target_col}", fontsize=13)
            ax.tick_params(axis='x', rotation=30)
            
        plt.tight_layout()
        plt.savefig(full_save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    except Exception as e:
        print(f"[tools] Warning: Target interaction plot error: {e}")
        plt.close("all")
    
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
    Plots probability distributions / KDE histograms or countplots for key important columns.
    The list of important columns is decided dynamically by the LLM and sent via kwargs/columns.
    """
    plt.close("all")
    target_cols = columns or kwargs.get("important_columns") or kwargs.get("cols") or kwargs.get("feature_cols")
    if not target_cols:
        target_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])][:6]

    if isinstance(target_cols, str):
        target_cols = [target_cols]

    valid_cols = [c for c in target_cols if c in df.columns]
    if not valid_cols:
        valid_cols = list(df.columns[:min(6, len(df.columns))])

    num_plots = len(valid_cols)
    ncols = min(3, num_plots)
    nrows = math.ceil(num_plots / ncols)

    os.makedirs(output_dir, exist_ok=True)
    saved_files = []

    for col in valid_cols:
        col_clean = re.sub(r'\W+', '_', col).strip('_')
        file_name = f"dist_{col_clean}.png"
        file_path = os.path.join(output_dir, file_name)

        try:
            fig, ax = plt.subplots(figsize=(6, 4))
            if pd.api.types.is_numeric_dtype(df[col]):
                sns.histplot(df[col].dropna(), kde=True, ax=ax, color="teal")
                ax.set_title(f"Distribution: {col}", fontsize=12, pad=10)
                ax.set_xlabel(col)
                ax.set_ylabel("Density / Frequency")
            else:
                sns.countplot(x=df[col].dropna(), hue=df[col].dropna(), ax=ax, palette="Set2", legend=False)
                ax.set_title(f"Distribution / Counts: {col}", fontsize=12, pad=10)
                ax.set_xlabel(col)
                ax.set_ylabel("Count")
                ax.tick_params(axis='x', rotation=30)

            plt.tight_layout()
            plt.savefig(file_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            saved_files.append(file_path)
            print(f"[tools] Saved separate distribution PNG for '{col}' to: {file_path}")
        except Exception as e:
            print(f"[tools] Warning: Error saving distribution plot for '{col}': {e}")
            plt.close("all")

    # Also maintain a combined grid plot for backward-compatibility if save_path requested
    full_save_path = os.path.join(output_dir, os.path.basename(save_path))
    try:
        num_plots = len(valid_cols)
        ncols = min(3, num_plots)
        nrows = math.ceil(num_plots / ncols)
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(4.5 * ncols, 3.5 * nrows))
        axes_list = [axes] if num_plots == 1 else axes.flatten()

        for idx, col in enumerate(valid_cols):
            ax = axes_list[idx]
            if pd.api.types.is_numeric_dtype(df[col]):
                sns.histplot(df[col].dropna(), kde=True, ax=ax, color="teal")
                ax.set_title(f"Distribution: {col}", fontsize=10)
            else:
                sns.countplot(x=df[col].dropna(), hue=df[col].dropna(), ax=ax, palette="Set2", legend=False)
                ax.set_title(f"Counts: {col}", fontsize=10)
                ax.tick_params(axis='x', rotation=30)

        for idx in range(num_plots, len(axes_list)):
            fig.delaxes(axes_list[idx])

        plt.tight_layout()
        plt.savefig(full_save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    except Exception as e:
        print(f"[tools] Warning: Error rendering combined distribution grid plot: {e}")
        plt.close("all")

    return {
        "plot_saved": full_save_path,
        "individual_plots": saved_files,
        "plotted_columns": valid_cols
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
    plt.close("all")
    os.makedirs(output_dir, exist_ok=True)
    
    pairs = bivariate_pairs or kwargs.get("pairs") or kwargs.get("bivariate_list") or kwargs.get("bivariate_pairs") or []
    
    # Auto-generate top pairs if not provided by LLM
    if not pairs:
        numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        cat_cols = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c]) and df[c].nunique() <= 10]
        
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
            
        if hue_col and hue_col not in df.columns:
            hue_col = None
            
        x_clean = re.sub(r'\W+', '_', x_col).strip('_')
        y_clean = re.sub(r'\W+', '_', y_col).strip('_')
        file_name = f"bivariate_{x_clean}_vs_{y_clean}.png"
        file_path = os.path.join(output_dir, file_name)
        
        try:
            fig, ax = plt.subplots(figsize=(7, 5))
            x_is_num = pd.api.types.is_numeric_dtype(df[x_col])
            y_is_num = pd.api.types.is_numeric_dtype(df[y_col])
            
            if x_is_num and y_is_num:
                sns.scatterplot(data=df, x=x_col, y=y_col, hue=hue_col, palette="Set1" if hue_col else None, alpha=0.7, ax=ax)
                sns.regplot(data=df, x=x_col, y=y_col, scatter=False, line_kws={"color": "darkred", "linestyle": "--"}, ax=ax)
                ax.set_title(f"Semantic Bivariate: {x_col} vs {y_col}", fontsize=12, pad=10)
            elif not x_is_num and y_is_num:
                sns.boxplot(data=df, x=x_col, y=y_col, hue=hue_col, palette="Set2", ax=ax)
                ax.set_title(f"Segmented Boxplot: {y_col} across {x_col}", fontsize=12, pad=10)
                ax.tick_params(axis='x', rotation=30)
            elif x_is_num and not y_is_num:
                sns.boxplot(data=df, x=y_col, y=x_col, hue=hue_col, palette="Set2", ax=ax)
                ax.set_title(f"Segmented Boxplot: {x_col} across {y_col}", fontsize=12, pad=10)
                ax.tick_params(axis='x', rotation=30)
            else:
                sns.countplot(data=df, x=x_col, hue=y_col, palette="Set1", ax=ax)
                ax.set_title(f"Categorical Cross-Tabulation: {x_col} by {y_col}", fontsize=12, pad=10)
                ax.tick_params(axis='x', rotation=30)
                
            plt.tight_layout()
            plt.savefig(file_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            saved_files.append({"x": x_col, "y": y_col, "saved_path": file_path, "rationale": rationale})
            print(f"[tools] Saved semantic bivariate plot '{x_col}' vs '{y_col}' to: {file_path}")
        except Exception as e:
            print(f"[tools] Warning: Error plotting bivariate '{x_col}' vs '{y_col}': {e}")
            plt.close("all")
            
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
    plt.close("all")
    os.makedirs(output_dir, exist_ok=True)
    
    # Select reasonable subset of numerical columns (max 4 to 5)
    raw_cols = columns or kwargs.get("feature_cols") or kwargs.get("cols") or kwargs.get("numeric_cols")
    if not raw_cols:
        raw_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and df[c].nunique() > 2]
        
    valid_cols = [c for c in raw_cols if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]
    
    # Clamp number of features to maximum 4-5 for clean, uncluttered visual rendering
    max_features = 4
    if len(valid_cols) > max_features:
        valid_cols = valid_cols[:max_features]
        print(f"[tools] Parameter Clamping: Clamped pairplot features to top {max_features}: {valid_cols}")
        
    if len(valid_cols) < 2:
        return {"error": "Insufficient numeric columns for pairplot rendering."}
        
    # Check hue column
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
            grid = sns.pairplot(clean_df, vars=valid_cols, color="teal", corner=True, plot_kws={"alpha": 0.6, "s": 25})
            
        grid.fig.suptitle("Pairwise Feature Relationships (Pairplot)", y=1.02, fontsize=14)
        grid.savefig(full_save_path, dpi=150, bbox_inches="tight")
        plt.close("all")
        print(f"[tools] Pairplot successfully saved to: {full_save_path}")
    except Exception as e:
        print(f"[tools] Warning: Pairplot rendering error: {e}")
        plt.close("all")
        
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
    Generates a predictive modeling blueprint. If custom_blueprint or kwargs are provided
    by the LLM, it uses the LLM's dynamic domain strategy.
    """
    if custom_blueprint and isinstance(custom_blueprint, dict):
        return custom_blueprint

    if kwargs and ("recommended_algorithms" in kwargs or "validation_strategy" in kwargs):
        return {
            "target_definition": target_col or kwargs.get("target_definition", "Target"),
            "problem_type": kwargs.get("problem_type", "Regression/Classification"),
            "recommended_algorithms": kwargs.get("recommended_algorithms", []),
            "feature_selection_strategy": kwargs.get("feature_selection_strategy", []),
            "validation_strategy": kwargs.get("validation_strategy", []),
            "overfitting_risk_mitigation": kwargs.get("overfitting_risk_mitigation", []),
            "executive_summary": kwargs.get("executive_summary", "Custom LLM predictive modeling blueprint.")
        }

    num_rows, num_cols = df.shape
    
    if target_col and target_col in df.columns:
        if pd.api.types.is_numeric_dtype(df[target_col]) and df[target_col].nunique() > 10:
            problem_type = "Regression"
        else:
            problem_type = "Classification"
    else:
        target_col = "Undefined (Unsupervised)"
        problem_type = "Unsupervised / Exploratory"

    if problem_type == "Classification":
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
        "executive_summary": f"Target: {target_col} ({problem_type}). Use robust cross-validation on {num_rows} rows x {num_cols} columns."
    }


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
    
    metrics_path = os.path.join(output_dir, "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_dict, f, indent=2)
        
    print(f"[tools] Canonical metrics.json successfully saved to: {os.path.abspath(metrics_path)}")
    return metrics_path


# =====================================================================
# 9. TOOL REGISTRY CATALOG FOR LLM PROMPTING
# =====================================================================
TOOL_REGISTRY = {
    "impute_missing_data": {
        "function": impute_missing_data,
        "description": "Imputes missing values using type-safe strategy (median for skewed numeric, mean for symmetric, mode for categorical).",
        "parameters": {"strategy_map": "Optional dict of {col: strategy}"}
    },
    "detect_and_handle_outliers": {
        "function": detect_and_handle_outliers,
        "description": "Detects outliers via IQR method and optionally caps extreme values.",
        "parameters": {"columns": "List of numeric cols", "action": "'profile' or 'cap'"}
    },
    "engineer_features": {
        "function": engineer_features,
        "description": "Creates high-signal feature transformations (log1p, ratios, interactions).",
        "parameters": {"feature_specs": "List of dicts defining features"}
    },
    "run_statistical_hypothesis_tests": {
        "function": run_statistical_hypothesis_tests,
        "description": "Calculates statistical significance against target variable (T-Test, ANOVA, Chi-Square, Pearson).",
        "parameters": {"target_col": "Name of target column", "alpha": "Significance threshold (0.05)"}
    },
    "plot_correlation_matrix": {
        "function": plot_correlation_matrix,
        "description": "Generates Pearson correlation matrix heatmap PNG image asset.",
        "parameters": {"save_path": "'correlation_matrix.png'"}
    },
    "plot_feature_distributions": {
        "function": plot_feature_distributions,
        "description": "Plots histograms, KDE distributions, or countplots for important columns identified by the LLM (passed via 'columns' argument).",
        "parameters": {"columns": "List of key/important column names to plot distributions for", "save_path": "'feature_distributions.png'"}
    },
    "plot_target_interaction": {
        "function": plot_target_interaction,
        "description": "Generates segmented distribution / scatter visualization comparing key feature vs target.",
        "parameters": {"target_col": "Target column name", "feature_col": "Feature column name", "save_path": "'target_interactions.png'"}
    },
    "plot_semantic_bivariate_relationships": {
        "function": plot_semantic_bivariate_relationships,
        "description": "Plots custom X vs Y scatter/boxplot/countplot relationships dynamically selected by LLM based on semantic domain reasoning (passed via 'bivariate_pairs' list of dicts with 'x', 'y', and optional 'hue').",
        "parameters": {"bivariate_pairs": "List of dicts [{'x': 'col1', 'y': 'col2', 'hue': 'target'}]"}
    },
    "plot_pairplot": {
        "function": plot_pairplot,
        "description": "Generates a concise Seaborn pairplot visualizing pairwise distributions and relationships across a reasonable subset of key numerical features (clamped to max 4-5 features).",
        "parameters": {"columns": "List of 3-4 key numeric features", "hue": "Optional target/hue column"}
    },
    "generate_predictive_blueprint": {
        "function": generate_predictive_blueprint,
        "description": "Compiles machine learning modeling blueprint and cross-validation strategy.",
        "parameters": {"target_col": "Target column name"}
    }
}

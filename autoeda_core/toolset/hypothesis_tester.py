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


class HypothesisTester:
    """
    Encapsulates statistical significance testing, Welch T-Test, ANOVA, Chi-Square,
    Pearson correlation, Cramér's V, and qualitative effect size interpretations.
    """
    def interpret_effect_size(self, test_name: str, val: float) -> str:
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
            if v < 0.01: return "Negligible effect"
            if v < 0.06: return "Small effect"
            if v < 0.14: return "Medium effect"
            return "Large effect"

    def cramers_v(self, x: pd.Series, y: pd.Series) -> float:
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

    def correlation_ratio(self, categories: pd.Series, measurements: pd.Series) -> float:
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

    def run_statistical_hypothesis_tests(
        self,
        df: pd.DataFrame,
        target_col: Optional[str] = None,
        feature_cols: Optional[List[str]] = None,
        alpha: float = 0.05,
        **kwargs
    ) -> Dict[str, Any]:
        try:
            alpha = max(0.0001, min(0.5, float(alpha)))
        except (ValueError, TypeError):
            alpha = 0.05

        if not target_col or target_col not in df.columns:
            numeric_cols = [c for c in df.columns if _is_numeric_col(df[c])]
            target_col = numeric_cols[-1] if numeric_cols else (df.columns[-1] if len(df.columns) > 0 else "")

        if not target_col or target_col not in df.columns:
            return {"error": "No valid target column found in dataset."}

        test_results = {"target_col": target_col, "significant_predictors": [], "ranked_significant_details": []}
        significant_items = []

        target_is_num = _is_numeric_col(df[target_col])

        for col in df.columns:
            if col == target_col or is_non_distributional_column(col, df[col]):
                continue

            col_is_num = _is_numeric_col(df[col])
            s1 = df[col].dropna()
            s2 = df[target_col].dropna()

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
                    label = self.interpret_effect_size("Pearson Correlation", eff)
                    if p < alpha:
                        significant_items.append({"feature": col, "test": "Pearson Correlation", "effect_size": round(eff, 4), "effect_size_label": label, "p_value": float(p)})
                elif not col_is_num and target_is_num:
                    groups = [group.values for name, group in data_target.groupby(data_col)]
                    if len(groups) > 1:
                        f_val, p = stats.f_oneway(*groups)
                        eff = self.correlation_ratio(data_col, data_target)
                        label = self.interpret_effect_size("ANOVA", eff)
                        if p < alpha:
                            significant_items.append({"feature": col, "test": "ANOVA", "effect_size": round(eff, 4), "effect_size_label": label, "p_value": float(p)})
                elif col_is_num and not target_is_num:
                    groups = [group.values for name, group in data_col.groupby(data_target)]
                    if len(groups) > 1:
                        f_val, p = stats.f_oneway(*groups)
                        eff = self.correlation_ratio(data_target, data_col)
                        label = self.interpret_effect_size("ANOVA", eff)
                        if p < alpha:
                            significant_items.append({"feature": col, "test": "ANOVA", "effect_size": round(eff, 4), "effect_size_label": label, "p_value": float(p)})
                else:
                    contingency = pd.crosstab(data_col, data_target)
                    if contingency.size > 0:
                        chi2, p, _, _ = stats.chi2_contingency(contingency)
                        eff = self.cramers_v(data_col, data_target)
                        label = self.interpret_effect_size("Chi-Square", eff)
                        if p < alpha:
                            significant_items.append({"feature": col, "test": "Chi-Square", "effect_size": round(eff, 4), "effect_size_label": label, "p_value": float(p)})
            except Exception:
                continue

        test_results["target_col"] = target_col
        significant_items.sort(key=lambda x: x["effect_size"], reverse=True)
        test_results["significant_predictors"] = [item["feature"] for item in significant_items]
        test_results["ranked_significant_details"] = significant_items
        return test_results


default_hypothesis_tester = HypothesisTester()


class RunStatisticalHypothesisTestsArgs(BaseModel):
    target_col: str = Field(description="Name of target column")
    alpha: float = Field(default=0.05, description="Significance threshold (0.05)")


def run_statistical_hypothesis_tests(df: pd.DataFrame, target_col=None, feature_cols=None, alpha=0.05, **kwargs):
    return default_hypothesis_tester.run_statistical_hypothesis_tests(df, target_col=target_col, feature_cols=feature_cols, alpha=alpha, **kwargs)


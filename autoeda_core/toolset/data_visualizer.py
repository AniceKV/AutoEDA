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
from .hypothesis_tester import HypothesisTester, default_hypothesis_tester


class DataVisualizer:
    """
    Encapsulates exploratory visualization calculations, correlation matrices,
    univariate distribution telemetry, bivariate interaction models, pairplots, and LLM bivariate inference.
    """
    def __init__(self, hypothesis_tester: Optional[HypothesisTester] = None):
        self.hypothesis_tester = hypothesis_tester or HypothesisTester()

    def plot_correlation_matrix(
        self,
        df: pd.DataFrame,
        save_path: str = "correlation_matrix.png",
        output_dir: str = "./sandbox_run",
        **kwargs
    ) -> Dict[str, Any]:
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
                        "interpretation": self.hypothesis_tester.interpret_effect_size("Pearson", val)
                    })

        cat_cols = [c for c in df.columns if (not _is_numeric_col(df[c]) or df[c].nunique() <= 10) and not is_non_distributional_column(c, df[c])]
        cat_assoc = []
        if len(cat_cols) >= 2:
            for i in range(len(cat_cols)):
                for j in range(i + 1, len(cat_cols)):
                    v = self.hypothesis_tester.cramers_v(df[cat_cols[i]], df[cat_cols[j]])
                    cat_assoc.append({
                        "feature_1": cat_cols[i],
                        "feature_2": cat_cols[j],
                        "cramers_v": round(v, 4),
                        "interpretation": self.hypothesis_tester.interpret_effect_size("Cramer", v)
                    })

        cross_type_redundant_pairs = []
        for c_col in cat_cols:
            for n_col in numeric_cols:
                if c_col == n_col: continue
                eta_val = self.hypothesis_tester.correlation_ratio(df[c_col], df[n_col])
                if eta_val >= 0.85:
                    cross_type_redundant_pairs.append({
                        "categorical_feature": c_col,
                        "numeric_feature": n_col,
                        "correlation_ratio_eta": round(eta_val, 4),
                        "interpretation": f"High cross-type redundancy between '{c_col}' and '{n_col}' (Eta = {eta_val:.4f})."
                    })

        z_matrix = corr_matrix.fillna(0).round(4).values.tolist()

        return {
            "correlation_heatmap_saved": "interactive_client_side",
            "z_matrix": z_matrix,
            "x_labels": cols,
            "y_labels": cols,
            "high_correlation_pairs": high_corr_pairs,
            "categorical_associations": cat_assoc,
            "cross_type_redundant_pairs": cross_type_redundant_pairs
        }

    def plot_target_interaction(
        self,
        df: pd.DataFrame,
        target_col: Optional[str] = None,
        feature_col: Optional[str] = None,
        save_path: str = "target_interactions.png",
        output_dir: str = "./sandbox_run",
        **kwargs
    ) -> Dict[str, Any]:
        if not target_col or target_col not in df.columns or is_non_distributional_column(target_col, df[target_col]):
            valid_targets = [c for c in df.columns if not is_non_distributional_column(c, df[c])]
            target_col = valid_targets[-1] if valid_targets else df.columns[-1]

        if not feature_col or feature_col not in df.columns or feature_col == target_col or is_non_distributional_column(feature_col, df[feature_col]):
            candidates = [c for c in df.columns if c != target_col and not is_non_distributional_column(c, df[c])]
            feature_col = candidates[0] if candidates else [c for c in df.columns if c != target_col][0]

        clean_df = df[[feature_col, target_col]].dropna()
        feat_is_num = _is_numeric_col(clean_df[feature_col])
        target_is_num = _is_numeric_col(clean_df[target_col])

        feat_nunique = clean_df[feature_col].nunique()
        target_nunique = clean_df[target_col].nunique()

        feat_is_discrete = (not feat_is_num) or (feat_nunique <= 10)
        target_is_discrete = (not target_is_num) or (target_nunique <= 10)

        interaction_data = {
            "target_col": target_col,
            "feature_col": feature_col,
            "feature_is_numeric": feat_is_num,
            "target_is_numeric": target_is_num,
            "feature_is_discrete": feat_is_discrete,
            "target_is_discrete": target_is_discrete,
        }

        if feat_is_discrete and target_is_discrete:
            # Limit to top 15 categories to prevent massive charts
            top_f = clean_df[feature_col].value_counts().nlargest(15).index
            top_t = clean_df[target_col].value_counts().nlargest(15).index
            sub_df = clean_df[clean_df[feature_col].isin(top_f) & clean_df[target_col].isin(top_t)]

            sf = sub_df[feature_col].astype(str)
            st = sub_df[target_col].astype(str)
            ct = pd.crosstab(sf, st)
            interaction_data["grouped_counts"] = ct.to_dict()
            ct_norm = pd.crosstab(sf, st, normalize="index")
            interaction_data["crosstab"] = ct_norm.round(4).to_dict()
        elif feat_is_discrete and (target_is_num or target_nunique > 10):
            groups = {}
            for cat, group in clean_df.groupby(feature_col):
                vals = pd.to_numeric(group[target_col], errors='coerce').dropna()
                if len(vals) > 0:
                    groups[str(cat)] = {
                        "min": round(_safe_float(vals.min()), 4),
                        "q1": round(_safe_float(vals.quantile(0.25)), 4),
                        "median": round(_safe_float(vals.median()), 4),
                        "q3": round(_safe_float(vals.quantile(0.75)), 4),
                        "max": round(_safe_float(vals.max()), 4),
                    }
            interaction_data["groups"] = groups
        elif target_is_discrete and (feat_is_num or feat_nunique > 10):
            groups = {}
            for cat, group in clean_df.groupby(target_col):
                vals = pd.to_numeric(group[feature_col], errors='coerce').dropna()
                if len(vals) > 0:
                    groups[str(cat)] = {
                        "min": round(_safe_float(vals.min()), 4),
                        "q1": round(_safe_float(vals.quantile(0.25)), 4),
                        "median": round(_safe_float(vals.median()), 4),
                        "q3": round(_safe_float(vals.quantile(0.75)), 4),
                        "max": round(_safe_float(vals.max()), 4),
                    }
            interaction_data["groups"] = groups
        else:
            sample_df = clean_df.sample(n=min(500, len(clean_df)), random_state=42) if len(clean_df) > 500 else clean_df
            points = []
            for _, r in sample_df.iterrows():
                points.append({"x": round(_safe_float(r[feature_col]), 4), "y": round(_safe_float(r[target_col]), 4)})
            interaction_data["points"] = points

        return {
            "plot_saved": "interactive_client_side",
            "target_col": target_col,
            "feature_col": feature_col,
            "interaction_data": interaction_data
        }

    def plot_feature_distributions(
        self,
        df: pd.DataFrame,
        columns: Optional[List[str]] = None,
        save_path: str = "feature_distributions.png",
        output_dir: str = "./sandbox_run",
        **kwargs
    ) -> Dict[str, Any]:
        target_cols = columns or kwargs.get("important_columns") or kwargs.get("cols") or kwargs.get("feature_cols")
        if not target_cols or target_cols == "all" or (isinstance(target_cols, (list, tuple)) and len(target_cols) == 0):
            target_cols = list(df.columns)
        elif isinstance(target_cols, str):
            if target_cols.lower() == "all":
                target_cols = list(df.columns)
            else:
                target_cols = [target_cols]

        if columns:
            valid_cols = [c for c in target_cols if c in df.columns]
        else:
            valid_cols = [c for c in target_cols if c in df.columns and not is_non_distributional_column(c, df[c])]
            if not valid_cols:
                valid_cols = [c for c in df.columns if not is_non_distributional_column(c, df[c])]
        if not valid_cols:
            valid_cols = list(df.columns)

        distributions = {}

        for col in valid_cols:
            s_clean = df[col].dropna()
            if len(s_clean) == 0:
                continue

            is_bool = pd.api.types.is_bool_dtype(s_clean)
            is_num = _is_numeric_col(s_clean) and not is_bool
            n_unique = s_clean.nunique()

            if is_num and n_unique > 10:
                s_num = pd.to_numeric(s_clean, errors='coerce').dropna()
                if len(s_num) > 0:
                    counts, bin_edges = np.histogram(s_num, bins=min(20, max(5, s_num.nunique() // 2)))
                    bin_centers = [(bin_edges[i] + bin_edges[i+1]) / 2.0 for i in range(len(counts))]
                    distributions[col] = {
                        "type": "numeric",
                        "counts": counts.tolist(),
                        "bin_edges": [round(_safe_float(b), 4) for b in bin_edges],
                        "bin_centers": [round(_safe_float(b), 4) for b in bin_centers],
                        "mean": round(_safe_float(s_num.mean()), 4),
                        "median": round(_safe_float(s_num.median()), 4),
                        "std": round(_safe_float(s_num.std()), 4) if len(s_num) > 1 else 0.0
                    }
                else:
                    vc = s_clean.value_counts().head(20)
                    distributions[col] = {
                        "type": "categorical",
                        "labels": [str(idx) for idx in vc.index],
                        "counts": [int(v) for v in vc.values]
                    }
            else:
                vc = s_clean.value_counts().head(20)
                distributions[col] = {
                    "type": "categorical",
                    "labels": [str(idx) for idx in vc.index],
                    "counts": [int(v) for v in vc.values]
                }

        return {
            "individual_plots": ["interactive_client_side"],
            "plotted_columns": valid_cols,
            "visual_distributions": distributions
        }

    def plot_semantic_bivariate_relationships(
        self,
        df: pd.DataFrame,
        bivariate_pairs: Optional[List[Dict[str, Any]]] = None,
        output_dir: str = "./sandbox_run",
        **kwargs
    ) -> Dict[str, Any]:
        raw_pairs = bivariate_pairs or kwargs.get("pairs") or kwargs.get("bivariate_list") or kwargs.get("bivariate_pairs") or []

        valid_cols = [c for c in df.columns if not is_non_distributional_column(c, df[c])]
        numeric_cols = [c for c in valid_cols if _is_numeric_col(df[c]) and df[c].nunique() > 10]
        discrete_cols = [c for c in valid_cols if df[c].nunique() <= 10 or not _is_numeric_col(df[c])]

        pairs = []
        for p in raw_pairs:
            if isinstance(p, dict):
                x = p.get("x") or p.get("x_col") or p.get("feature_1")
                y = p.get("y") or p.get("y_col") or p.get("feature_2")
                if x and y and x in df.columns and y in df.columns and x != y and not is_non_distributional_column(x, df[x]) and not is_non_distributional_column(y, df[y]):
                    pairs.append({"x": x, "y": y, "rationale": p.get("rationale", "Semantic relationship")})

        if not pairs:
            if len(discrete_cols) >= 2:
                pairs.append({"x": discrete_cols[0], "y": discrete_cols[1], "rationale": f"Categorical breakdown of {discrete_cols[0]} by {discrete_cols[1]}"})
            if discrete_cols and numeric_cols:
                pairs.append({"x": discrete_cols[0], "y": numeric_cols[0], "rationale": f"Segmented distribution of {numeric_cols[0]} across {discrete_cols[0]}"})
            if len(numeric_cols) >= 2:
                pairs.append({"x": numeric_cols[0], "y": numeric_cols[1], "rationale": f"Bivariate scatter comparison of {numeric_cols[0]} vs {numeric_cols[1]}"})

        bivariate_data_list = []
        seen_pairs = set()

        for pair in pairs:
            x_col = pair["x"]
            y_col = pair["y"]
            pair_key = tuple(sorted([x_col, y_col]))
            if pair_key in seen_pairs:
                continue
            seen_pairs.add(pair_key)

            clean_df = df[[x_col, y_col]].dropna()
            if len(clean_df) < 5:
                continue

            x_is_num = _is_numeric_col(clean_df[x_col])
            y_is_num = _is_numeric_col(clean_df[y_col])
            x_nunique = clean_df[x_col].nunique()
            y_nunique = clean_df[y_col].nunique()

            x_is_discrete = (not x_is_num) or (x_nunique <= 10)
            y_is_discrete = (not y_is_num) or (y_nunique <= 10)

            pair_entry = {
                "x": x_col,
                "y": y_col,
                "rationale": pair.get("rationale", "Semantic relationship"),
                "x_is_numeric": x_is_num,
                "y_is_numeric": y_is_num,
                "x_is_discrete": x_is_discrete,
                "y_is_discrete": y_is_discrete,
            }

            if x_is_discrete and y_is_discrete:
                # Limit to top 15 categories to prevent massive charts
                top_x = clean_df[x_col].value_counts().nlargest(15).index
                top_y = clean_df[y_col].value_counts().nlargest(15).index
                sub_df = clean_df[clean_df[x_col].isin(top_x) & clean_df[y_col].isin(top_y)]

                sx = sub_df[x_col].astype(str)
                sy = sub_df[y_col].astype(str)
                ct = pd.crosstab(sx, sy)
                pair_entry["grouped_counts"] = ct.to_dict()
                ct_norm = pd.crosstab(sx, sy, normalize="index")
                pair_entry["crosstab"] = ct_norm.round(4).to_dict()
            elif x_is_discrete and (y_is_num or y_nunique > 10):
                groups = {}
                for cat, group in clean_df.groupby(x_col):
                    vals = pd.to_numeric(group[y_col], errors='coerce').dropna()
                    if len(vals) > 0:
                        groups[str(cat)] = {
                            "min": round(_safe_float(vals.min()), 4),
                            "q1": round(_safe_float(vals.quantile(0.25)), 4),
                            "median": round(_safe_float(vals.median()), 4),
                            "q3": round(_safe_float(vals.quantile(0.75)), 4),
                            "max": round(_safe_float(vals.max()), 4)
                        }
                pair_entry["groups"] = groups
            elif y_is_discrete and (x_is_num or x_nunique > 10):
                groups = {}
                for cat, group in clean_df.groupby(y_col):
                    vals = pd.to_numeric(group[x_col], errors='coerce').dropna()
                    if len(vals) > 0:
                        groups[str(cat)] = {
                            "min": round(_safe_float(vals.min()), 4),
                            "q1": round(_safe_float(vals.quantile(0.25)), 4),
                            "median": round(_safe_float(vals.median()), 4),
                            "q3": round(_safe_float(vals.quantile(0.75)), 4),
                            "max": round(_safe_float(vals.max()), 4)
                        }
                pair_entry["groups"] = groups
            else:
                sample_df = clean_df.sample(n=min(500, len(clean_df)), random_state=42) if len(clean_df) > 500 else clean_df
                points = []
                for _, r in sample_df.iterrows():
                    points.append({"x": round(_safe_float(r[x_col]), 4), "y": round(_safe_float(r[y_col]), 4)})
                pair_entry["points"] = points

            bivariate_data_list.append(pair_entry)

        return {
            "bivariate_plots_saved": ["interactive_client_side"],
            "count": len(bivariate_data_list),
            "bivariate_data": bivariate_data_list
        }

    def infer_llm_bivariate_pairs(
        self,
        df: pd.DataFrame,
        dataset_name: str = "",
        target_col: Optional[str] = None,
        top_n: int = 5,
        **kwargs
    ) -> List[Dict[str, Any]]:
        valid_cols = [c for c in df.columns if not is_non_distributional_column(c, df[c])]
        if len(valid_cols) < 2:
            return []

        col_summary = []
        for c in valid_cols[:25]:
            s_clean = df[c].dropna()
            dtype = str(df[c].dtype)
            nunique = int(s_clean.nunique())
            sample_vals = [str(v) for v in s_clean.head(3).tolist()]
            col_summary.append(f"- {c} ({dtype}, {nunique} unique values, samples: {sample_vals})")

        cols_text = "\n".join(col_summary)

        api_key = None
        try:
            api_key = get_api_key()
        except ValueError:
            pass

        if api_key:
            try:
                from openai import OpenAI
                client = OpenAI(base_url=get_base_url(), api_key=api_key)
                prompt = (
                    f"You are an expert data scientist performing EDA on dataset '{dataset_name}'.\n"
                    f"Target column: '{target_col or 'None'}'\n"
                    f"Dataset features:\n{cols_text}\n\n"
                    f"Propose up to {top_n} semantically interesting or domain-relevant bivariate feature pairs (feature_1, feature_2) "
                    f"that should be analyzed together for domain insights, feature interaction, or segmentation.\n"
                    f"Requirements:\n"
                    f"- Exclude identical columns (feature_1 != feature_2)\n"
                    f"- Both features MUST exist in the feature list provided above.\n"
                    f"- Provide a concise 1-sentence domain rationale for each pair.\n"
                    f"Return ONLY a valid JSON array of objects with keys 'feature_1', 'feature_2', 'rationale'."
                )
                response = client.chat.completions.create(
                    model=get_model(),
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.2,
                    max_tokens=500
                )
                raw = response.choices[0].message.content
                match = re.search(r'\[.*\]', raw, re.DOTALL)
                if match:
                    parsed = json.loads(match.group(0))
                    llm_pairs = []
                    for p in parsed:
                        f1 = p.get("feature_1") or p.get("x")
                        f2 = p.get("feature_2") or p.get("y")
                        rat = p.get("rationale", "LLM-inferred domain relationship")
                        if f1 and f2 and f1 in df.columns and f2 in df.columns and f1 != f2:
                            llm_pairs.append({"feature_1": f1, "feature_2": f2, "rationale": rat, "source": "llm_inferred"})
                    if llm_pairs:
                        return llm_pairs[:top_n]
            except Exception as e:
                print(f"[tools] LLM bivariate pair inference warning: {e}")

        fallback_pairs = []
        num_cols = [c for c in valid_cols if _is_numeric_col(df[c])]
        cat_cols = [c for c in valid_cols if not _is_numeric_col(df[c])]

        if target_col and target_col in valid_cols:
            for c in valid_cols:
                if c != target_col:
                    fallback_pairs.append({
                        "feature_1": c,
                        "feature_2": target_col,
                        "rationale": f"Domain interaction of '{c}' against primary target '{target_col}'",
                        "source": "llm_inferred"
                    })
                    if len(fallback_pairs) >= top_n:
                        break

        if len(fallback_pairs) < top_n and len(num_cols) >= 2:
            fallback_pairs.append({
                "feature_1": num_cols[0],
                "feature_2": num_cols[1],
                "rationale": f"Numeric scale comparison and scatter relationship of '{num_cols[0]}' vs '{num_cols[1]}'",
                "source": "llm_inferred"
            })
        if len(fallback_pairs) < top_n and cat_cols and num_cols:
            fallback_pairs.append({
                "feature_1": cat_cols[0],
                "feature_2": num_cols[0],
                "rationale": f"Segmented behavior of numerical '{num_cols[0]}' across categorical '{cat_cols[0]}'",
                "source": "llm_inferred"
            })

        return fallback_pairs[:top_n]

    def compute_bivariate_union(
        self,
        df: pd.DataFrame,
        target_col: Optional[str] = None,
        top_n_algo: int = 10,
        top_n_llm: int = 5,
        dataset_name: str = "",
        **kwargs
    ) -> Dict[str, Any]:
        valid_cols = [c for c in df.columns if not is_non_distributional_column(c, df[c])]
        if len(valid_cols) < 2:
            return {"union_pairs": [], "algorithmic_count": 0, "llm_count": 0, "both_count": 0, "union_count": 0}

        algo_pairs_dict = {}

        num_cols = [c for c in valid_cols if _is_numeric_col(df[c]) and df[c].nunique() > 1]
        if len(num_cols) >= 2:
            corr_matrix = df[num_cols].corr()
            for i in range(len(num_cols)):
                for j in range(i + 1, len(num_cols)):
                    c1, c2 = num_cols[i], num_cols[j]
                    val = corr_matrix.loc[c1, c2]
                    if pd.notnull(val) and abs(val) >= 0.25:
                        pk = tuple(sorted([c1, c2]))
                        algo_pairs_dict[pk] = {
                            "feature_1": c1,
                            "feature_2": c2,
                            "rationale": f"Algorithmic Pearson correlation (|r| = {abs(val):.2f})",
                            "correlation": round(float(val), 4)
                        }

        cat_cols = [c for c in valid_cols if not _is_numeric_col(df[c]) and df[c].nunique() > 1 and df[c].nunique() <= 50]
        if len(cat_cols) >= 2:
            for i in range(len(cat_cols)):
                for j in range(i + 1, len(cat_cols)):
                    c1, c2 = cat_cols[i], cat_cols[j]
                    v = self.hypothesis_tester.cramers_v(df[c1].dropna(), df[c2].dropna())
                    if v >= 0.2:
                        pk = tuple(sorted([c1, c2]))
                        if pk not in algo_pairs_dict:
                            algo_pairs_dict[pk] = {
                                "feature_1": c1,
                                "feature_2": c2,
                                "rationale": f"Algorithmic Cramér's V association (V = {v:.2f})",
                                "cramers_v": round(float(v), 4)
                            }

        if target_col and target_col in valid_cols:
            for c in valid_cols:
                if c != target_col:
                    pk = tuple(sorted([c, target_col]))
                    if pk not in algo_pairs_dict:
                        algo_pairs_dict[pk] = {
                            "feature_1": c,
                            "feature_2": target_col,
                            "rationale": f"Statistical hypothesis predictor against target '{target_col}'"
                        }

        llm_pairs = self.infer_llm_bivariate_pairs(df, dataset_name=dataset_name, target_col=target_col, top_n=top_n_llm)
        llm_pairs_dict = {}
        for item in llm_pairs:
            f1, f2 = item["feature_1"], item["feature_2"]
            pk = tuple(sorted([f1, f2]))
            llm_pairs_dict[pk] = item

        all_keys = set(algo_pairs_dict.keys()) | set(llm_pairs_dict.keys())
        union_list = []
        algo_count = 0
        llm_count = 0
        both_count = 0

        for pk in sorted(all_keys):
            in_algo = pk in algo_pairs_dict
            in_llm = pk in llm_pairs_dict

            if in_algo and in_llm:
                source = "both"
                both_count += 1
                algo_count += 1
                llm_count += 1
                rat = f"{llm_pairs_dict[pk].get('rationale', '')} (Also identified statistically by algorithms)"
            elif in_algo:
                source = "algorithmic"
                algo_count += 1
                rat = algo_pairs_dict[pk].get("rationale", "Algorithmic bivariate relationship")
            else:
                source = "llm_inferred"
                llm_count += 1
                rat = llm_pairs_dict[pk].get("rationale", "LLM-inferred domain interaction")

            f1, f2 = pk[0], pk[1]

            clean_df = df[[f1, f2]].dropna()
            f1_is_num = _is_numeric_col(clean_df[f1])
            f2_is_num = _is_numeric_col(clean_df[f2])
            f1_nunique = clean_df[f1].nunique()
            f2_nunique = clean_df[f2].nunique()

            f1_is_discrete = (not f1_is_num) or (f1_nunique <= 10)
            f2_is_discrete = (not f2_is_num) or (f2_nunique <= 10)

            entry = {
                "feature_1": f1,
                "feature_2": f2,
                "rationale": rat.strip(),
                "source": source,
                "feature_1_is_numeric": f1_is_num,
                "feature_2_is_numeric": f2_is_num,
                "feature_1_is_discrete": f1_is_discrete,
                "feature_2_is_discrete": f2_is_discrete,
            }

            if f1_is_discrete and f2_is_discrete and f1_nunique <= 15 and f2_nunique <= 15:
                s1 = clean_df[f1].astype(str)
                s2 = clean_df[f2].astype(str)
                ct = pd.crosstab(s1, s2)
                entry["grouped_counts"] = ct.to_dict()
                ct_norm = pd.crosstab(s1, s2, normalize="index")
                entry["crosstab"] = ct_norm.round(4).to_dict()
            elif f1_is_discrete and (f2_is_num or f2_nunique > 10):
                groups = {}
                for cat, group in clean_df.groupby(f1):
                    vals = pd.to_numeric(group[f2], errors='coerce').dropna()
                    if len(vals) > 0:
                        groups[str(cat)] = {
                            "min": round(_safe_float(vals.min()), 4),
                            "q1": round(_safe_float(vals.quantile(0.25)), 4),
                            "median": round(_safe_float(vals.median()), 4),
                            "q3": round(_safe_float(vals.quantile(0.75)), 4),
                            "max": round(_safe_float(vals.max()), 4),
                        }
                entry["groups"] = groups
            elif f2_is_discrete and (f1_is_num or f1_nunique > 10):
                groups = {}
                for cat, group in clean_df.groupby(f2):
                    vals = pd.to_numeric(group[f1], errors='coerce').dropna()
                    if len(vals) > 0:
                        groups[str(cat)] = {
                            "min": round(_safe_float(vals.min()), 4),
                            "q1": round(_safe_float(vals.quantile(0.25)), 4),
                            "median": round(_safe_float(vals.median()), 4),
                            "q3": round(_safe_float(vals.quantile(0.75)), 4),
                            "max": round(_safe_float(vals.max()), 4),
                        }
                entry["groups"] = groups
            else:
                sample_df = clean_df.sample(n=min(400, len(clean_df)), random_state=42) if len(clean_df) > 400 else clean_df
                points = []
                for _, r in sample_df.iterrows():
                    points.append({"x": round(_safe_float(r[f1]), 4), "y": round(_safe_float(r[f2]), 4)})
                entry["points"] = points

            union_list.append(entry)

        return {
            "union_pairs": union_list,
            "algorithmic_count": len(algo_pairs_dict),
            "llm_count": len(llm_pairs_dict),
            "both_count": both_count,
            "union_count": len(union_list)
        }

    def plot_pairplot(
        self,
        df: pd.DataFrame,
        columns: Optional[List[str]] = None,
        hue: Optional[str] = None,
        save_path: str = "pairplot.png",
        output_dir: str = "./sandbox_run",
        **kwargs
    ) -> Dict[str, Any]:
        raw_cols = columns or kwargs.get("feature_cols") or kwargs.get("cols") or kwargs.get("numeric_cols")
        if not raw_cols:
            raw_cols = [c for c in df.columns if _is_numeric_col(df[c]) and df[c].nunique() > 2 and not is_non_distributional_column(c, df[c])]

        valid_cols = []
        if raw_cols:
            for c in raw_cols:
                if c in df.columns:
                    s_num = pd.to_numeric(df[c], errors='coerce').dropna()
                    if len(s_num) > 0 and _is_numeric_col(df[c]):
                        valid_cols.append(c)

        max_features = 4
        if len(valid_cols) > max_features:
            valid_cols = valid_cols[:max_features]

        if len(valid_cols) < 2:
            return {"error": "Insufficient numeric columns for pairplot rendering."}

        hue_col = hue or kwargs.get("target_col")
        if hue_col and hue_col not in df.columns:
            hue_col = None

        clean_df = df[valid_cols + ([hue_col] if hue_col else [])].dropna()
        sample_df = clean_df.sample(n=min(300, len(clean_df)), random_state=42) if len(clean_df) > 300 else clean_df

        pairplot_matrix = []
        for c1 in valid_cols:
            row_entry = []
            for c2 in valid_cols:
                if c1 == c2:
                    s_num = pd.to_numeric(clean_df[c1], errors='coerce').dropna()
                    if len(s_num) > 0:
                        counts, bin_edges = np.histogram(s_num, bins=10)
                        bin_centers = [(bin_edges[i] + bin_edges[i+1]) / 2.0 for i in range(len(counts))]
                        row_entry.append({
                            "type": "diag",
                            "feature": c1,
                            "bin_centers": [round(_safe_float(b), 4) for b in bin_centers],
                            "counts": counts.tolist()
                        })
                else:
                    points = []
                    for _, r in sample_df.iterrows():
                        points.append({"x": round(_safe_float(r[c2]), 4), "y": round(_safe_float(r[c1]), 4)})
                    row_entry.append({
                        "type": "scatter",
                        "x_feature": c2,
                        "y_feature": c1,
                        "points": points
                    })
            pairplot_matrix.append(row_entry)

        return {
            "pairplot_saved": "interactive_client_side",
            "features_plotted": valid_cols,
            "hue": hue_col,
            "pairplot_matrix": pairplot_matrix
        }


default_visualizer = DataVisualizer(hypothesis_tester=default_hypothesis_tester)


class PlotCorrelationMatrixArgs(BaseModel):
    save_path: str = Field(default="correlation_matrix.png")


class PlotFeatureDistributionsArgs(BaseModel):
    columns: Optional[List[str]] = Field(default=None, description="Optional list of column names to plot distributions for.")
    save_path: str = Field(default="feature_distributions.png")


class PlotTargetInteractionArgs(BaseModel):
    target_col: str = Field(description="Target column name")
    feature_col: str = Field(description="Feature column name")
    save_path: str = Field(default="target_interactions.png")


class PlotSemanticBivariateRelationshipsArgs(BaseModel):
    bivariate_pairs: List[Dict[str, Optional[str]]] = Field(
        description="List of dicts [{'x': 'col1', 'y': 'col2', 'rationale': 'Domain rationale'}]."
    )


class PlotPairplotArgs(BaseModel):
    columns: List[str] = Field(description="List of 3-4 key numeric features")
    hue: Optional[str] = Field(default=None, description="Optional target/hue column")


def plot_correlation_matrix(df: pd.DataFrame, save_path="correlation_matrix.png", output_dir="./sandbox_run", **kwargs):
    return default_visualizer.plot_correlation_matrix(df, save_path=save_path, output_dir=output_dir, **kwargs)


def plot_target_interaction(df: pd.DataFrame, target_col=None, feature_col=None, save_path="target_interactions.png", output_dir="./sandbox_run", **kwargs):
    return default_visualizer.plot_target_interaction(df, target_col=target_col, feature_col=feature_col, save_path=save_path, output_dir=output_dir, **kwargs)


def plot_feature_distributions(df: pd.DataFrame, columns=None, save_path="feature_distributions.png", output_dir="./sandbox_run", **kwargs):
    return default_visualizer.plot_feature_distributions(df, columns=columns, save_path=save_path, output_dir=output_dir, **kwargs)


def plot_semantic_bivariate_relationships(df: pd.DataFrame, bivariate_pairs=None, output_dir="./sandbox_run", **kwargs):
    return default_visualizer.plot_semantic_bivariate_relationships(df, bivariate_pairs=bivariate_pairs, output_dir=output_dir, **kwargs)


def infer_llm_bivariate_pairs(df: pd.DataFrame, dataset_name="", target_col=None, top_n=5, **kwargs):
    return default_visualizer.infer_llm_bivariate_pairs(df, dataset_name=dataset_name, target_col=target_col, top_n=top_n, **kwargs)


def compute_bivariate_union(df: pd.DataFrame, target_col=None, top_n_algo=10, top_n_llm=5, dataset_name="", **kwargs):
    return default_visualizer.compute_bivariate_union(df, target_col=target_col, top_n_algo=top_n_algo, top_n_llm=top_n_llm, dataset_name=dataset_name, **kwargs)


def plot_pairplot(df: pd.DataFrame, columns=None, hue=None, save_path="pairplot.png", output_dir="./sandbox_run", **kwargs):
    return default_visualizer.plot_pairplot(df, columns=columns, hue=hue, save_path=save_path, output_dir=output_dir, **kwargs)


import os
import json
import re
import numpy as np
import pandas as pd
import seaborn as sns
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
        ## for numeric columns we correlate them with pearson correlation 
        numeric_cols = [c for c in df.columns if _is_numeric_col(df[c]) and not is_non_distributional_column(c, df[c])]

        if len(numeric_cols) < 2:
            return {"error": "Insufficient numeric columns for correlation matrix rendering."}

        corr = df[numeric_cols].apply(pd.to_numeric, errors='coerce').corr(method="pearson").round(4)
        corr_matrix = corr.fillna(0).to_dict()

        key_correlations = []
        for i in range(len(numeric_cols)):
            for j in range(i + 1, len(numeric_cols)):
                col1, col2 = numeric_cols[i], numeric_cols[j]
                val = corr.loc[col1, col2]
                if not np.isnan(val) and abs(val) >= 0.3:
                    key_correlations.append({
                        "feature_1": col1,
                        "feature_2": col2,
                        "correlation": float(val),
                        "interpretation": self.hypothesis_tester.interpret_effect_size("Pearson", val)
                    })

        #for categorical columns we relate them with cramerV correlation
        cat_cols = [c for c in df.columns if (not _is_numeric_col(df[c]) or df[c].nunique() <= 10) and not is_non_distributional_column(c, df[c])]
        cat_assoc = []
        if len(cat_cols) >= 2:
            for i in range(len(cat_cols)):
                for j in range(i + 1, len(cat_cols)):
                    c1, c2 = cat_cols[i], cat_cols[j]
                    v = self.hypothesis_tester.cramers_v(df[c1], df[c2])
                    if v >= 0.3:
                        cat_assoc.append({
                            "feature_1": c1,
                            "feature_2": c2,
                            "cramers_v": float(v),
                            "interpretation": self.hypothesis_tester.interpret_effect_size("Cramer", v)
                        })

        ## two columns may have very high correlation maybe due to being same  olumn with different names
        ## or slightly different linear transformation we can pair them by below code

        cross_type_redundant_pairs = []
        for c_col in cat_cols:
            for n_col in numeric_cols:
                eta = self.hypothesis_tester.correlation_ratio(df[c_col], df[n_col])
                if eta >= 0.8:
                    cross_type_redundant_pairs.append({
                        "categorical_col": c_col,
                        "numeric_col": n_col,
                        "eta": float(eta),
                        "interpretation": self.hypothesis_tester.interpret_effect_size("Correlation Ratio", eta)
                    })

        return {
            "plot_saved": "interactive_client_side",
            "correlation_matrix": corr_matrix,
            "key_correlations": key_correlations,
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
        """
                                    DataFrame
                                        │
                                        ▼
                            Choose target column
                                        │
                                        ▼
                            Choose feature column
                                        │
                                        ▼
                            Remove rows with NaN
                                        │
                                        ▼
                        Determine numeric/categorical
                                        │
                                        ▼
                            Determine discrete/continuous
                                        │
                            ┌─────────┴─────────┐
                            │                   │
                        Both discrete      At least one
                            │               continuous
                            ▼                   │
                        Crosstab                │
                            │            ┌──────┴──────┐
                            │            │             │
                            │       Feature discrete  Target discrete
                            │            │             │
                            │            ▼             ▼
                            │        Group stats   Group stats
                            │
                            └──────────────┬──────────────┘
                                            │
                                            ▼
                                    Both continuous
                                            │
                                            ▼
                                    Sample ≤ 500 rows
                                            │
                                            ▼
                                        x/y points
        """
        if not target_col or target_col not in df.columns or is_non_distributional_column(target_col, df[target_col]):
            valid_targets = [c for c in df.columns if not is_non_distributional_column(c, df[c])]
            target_col = valid_targets[-1] if valid_targets else df.columns[-1]

        top_n = kwargs.get("top_n", None)

        if feature_col and feature_col in df.columns and feature_col != target_col and not is_non_distributional_column(feature_col, df[feature_col]):
            candidate_features = [feature_col]
            if top_n:
                others = [c for c in df.columns if c != target_col and c != feature_col and not is_non_distributional_column(c, df[c])]
                candidate_features.extend(others[:max(0, top_n - 1)])
        else:
            candidate_features = [c for c in df.columns if c != target_col and not is_non_distributional_column(c, df[c])]
            if not candidate_features:
                candidate_features = [c for c in df.columns if c != target_col]
            if top_n:
                candidate_features = candidate_features[:top_n]

        feature_col = candidate_features[0] if candidate_features else None

        target_interactions = []
        for f_col in candidate_features:
            clean_df = df[[f_col, target_col]].dropna()
            if clean_df.empty:
                continue

            feat_is_num = _is_numeric_col(clean_df[f_col])
            target_is_num = _is_numeric_col(clean_df[target_col])

            feat_nunique = clean_df[f_col].nunique()
            target_nunique = clean_df[target_col].nunique()

            feat_is_discrete = (not feat_is_num) or (feat_nunique <= 10)
            target_is_discrete = (not target_is_num) or (target_nunique <= 10)

            inter_data = {
                "target_col": target_col,
                "feature_col": f_col,
                "feature_is_numeric": feat_is_num,
                "target_is_numeric": target_is_num,
                "feature_is_discrete": feat_is_discrete,
                "target_is_discrete": target_is_discrete,
            }

            if feat_is_discrete and target_is_discrete:
                top_f = clean_df[f_col].value_counts().nlargest(15).index
                top_t = clean_df[target_col].value_counts().nlargest(15).index
                sub_df = clean_df[clean_df[f_col].isin(top_f) & clean_df[target_col].isin(top_t)]

                sf = sub_df[f_col].astype(str)
                st = sub_df[target_col].astype(str)
                ct = pd.crosstab(sf, st)
                inter_data["grouped_counts"] = ct.to_dict()
                ct_norm = pd.crosstab(sf, st, normalize="index")
                inter_data["crosstab"] = ct_norm.round(4).to_dict()
            elif feat_is_discrete and (target_is_num or target_nunique > 10):
                groups = {}
                for cat, group in clean_df.groupby(f_col):
                    vals = pd.to_numeric(group[target_col], errors='coerce').dropna()
                    if len(vals) > 0:
                        groups[str(cat)] = {
                            "min": round(_safe_float(vals.min()), 4),
                            "q1": round(_safe_float(vals.quantile(0.25)), 4),
                            "median": round(_safe_float(vals.median()), 4),
                            "q3": round(_safe_float(vals.quantile(0.75)), 4),
                            "max": round(_safe_float(vals.max()), 4),
                        }
                inter_data["groups"] = groups
            elif target_is_discrete and (feat_is_num or feat_nunique > 10):
                groups = {}
                for cat, group in clean_df.groupby(target_col):
                    vals = pd.to_numeric(group[f_col], errors='coerce').dropna()
                    if len(vals) > 0:
                        groups[str(cat)] = {
                            "min": round(_safe_float(vals.min()), 4),
                            "q1": round(_safe_float(vals.quantile(0.25)), 4),
                            "median": round(_safe_float(vals.median()), 4),
                            "q3": round(_safe_float(vals.quantile(0.75)), 4),
                            "max": round(_safe_float(vals.max()), 4),
                        }
                inter_data["groups"] = groups
            else:
                sample_df = clean_df.sample(n=min(500, len(clean_df)), random_state=42) if len(clean_df) > 500 else clean_df
                points = []
                for _, r in sample_df.iterrows():
                    points.append({"x": round(_safe_float(r[f_col]), 4), "y": round(_safe_float(r[target_col]), 4)})
                inter_data["points"] = points

            target_interactions.append(inter_data)

        primary_interaction = target_interactions[0] if target_interactions else {}

        return {
            "plot_saved": "interactive_client_side",
            "target_col": target_col,
            "feature_col": feature_col,
            "top_features": candidate_features,
            "interaction_data": primary_interaction,
            "target_interactions": target_interactions
        }

    def plot_feature_distributions(
        self,
        df: pd.DataFrame,
        columns: Optional[List[str]] = None,
        save_path: str = "feature_distributions.png",
        output_dir: str = "./sandbox_run",
        **kwargs
    ) -> Dict[str, Any]:
        """
                                                DataFrame
                                                    │
                                                    ▼
                                            Determine target_cols
                                                    │
                                        ┌────────────┴────────────┐
                                        │                         │
                                    Columns given            No columns
                                        │                         │
                                        ▼                         ▼
                                    Use those             Use important/feature
                                                            columns or all columns
                                                    │
                                                    ▼
                                            Filter valid columns
                                                    │
                                                    ▼
                                            Loop through each column
                                                    │
                                                    ▼
                                                Drop NaN
                                                    │
                                                    ▼
                                            Is column numeric?
                                                /           \
                                            YES            NO
                                                │              │
                                                ▼              ▼
                                        Is unique count     Frequency counts
                                            > 10?           top 20 categories
                                        /       \
                                        YES        NO
                                        │          │
                                        ▼          ▼
                                    Histogram    Frequency
                                    data         counts
                                        │          │
                                        └────┬─────┘
                                            ▼
                                    Store in distributions
                                            │
                                            ▼
                                    Return visualization data
        """
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

    def plot_semantic_bivariate_relationships(self,df: pd.DataFrame,bivariate_pairs: Optional[List[Dict[str, Any]]] = None,output_dir: str = "./sandbox_run",**kwargs) -> Dict[str, Any]:
        """
                                 DataFrame
                                    │
                                    ▼
                            Get requested pairs
                                    │
                        ┌───────────┴───────────┐
                        │                       │
                Valid pairs supplied      No valid pairs
                        │                       │
                        │                       ▼
                        │                Automatically choose
                        │                useful variable pairs
                        │
                        └───────────┬───────────┘
                                    ▼
                            Remove duplicates
                                    │
                                    ▼
                        Process each pair
                                    │
                                    ▼
                            Drop missing rows
                                    │
                                    ▼
                        Determine variable types
                                    │
                    ┌─────────────┴─────────────┐
                    │                           │
                Both categorical             At least one
                                            numeric
                    │                           │
                    ▼                    ┌──────┴──────┐
                Crosstab                 │             │
                                    Cat + Numeric   Numeric + Numeric
                                        │             │
                                        ▼             ▼
                                    Group stats    Scatter points
                                        │             │
                                        └──────┬──────┘
                                                ▼
                                    Return bivariate data
        """
        raw_pairs = bivariate_pairs or kwargs.get("pairs") or kwargs.get("bivariate_list") or kwargs.get("bivariate_pairs") or []

        valid_cols = [c for c in df.columns if not is_non_distributional_column(c, df[c])]
        numeric_cols = [c for c in valid_cols if _is_numeric_col(df[c]) and df[c].nunique() > 10]
        discrete_cols = [c for c in valid_cols if df[c].nunique() <= 10 or not _is_numeric_col(df[c])]

        pairs = []
        for p in raw_pairs:
            if isinstance(p, dict):
                x = p.get("x") or p.get("x_col") or p.get("feature_1") ## diiferent sources of paring give different naming
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


    def infer_llm_bivariate_pairs(self,df: pd.DataFrame,dataset_name: str = "",target_col: Optional[str] = None,top_n: int = 5,**kwargs) -> List[Dict[str, Any]]:

        valid_cols = [
            c for c in df.columns
            if not is_non_distributional_column(c, df[c])
        ]

        if len(valid_cols) < 2:
            return []

        col_summary = []

        for c in valid_cols[:25]:
            s_clean = df[c].dropna()
            dtype = str(df[c].dtype)
            nunique = int(s_clean.nunique())
            sample_vals = [str(v) for v in s_clean.head(3).tolist()]

            col_summary.append(
                f"- {c} ({dtype}, {nunique} unique values, samples: {sample_vals})"
            )

        cols_text = "\n".join(col_summary)

        api_key = None

        try:
            api_key = get_api_key()
        except ValueError:
            pass

        if not api_key:
            return []

        try:
            from openai import OpenAI

            client = OpenAI(
                base_url=get_base_url(),
                api_key=api_key
            )

            prompt = (
                f"You are an expert data scientist performing EDA on dataset '{dataset_name}'.\n"
                f"Target column: '{target_col or 'None'}'\n"
                f"Dataset features:\n{cols_text}\n\n"
                f"Propose up to {top_n} semantically interesting or domain-relevant "
                f"bivariate feature pairs (feature_1, feature_2) that should be "
                f"analyzed together for domain insights, feature interaction, or segmentation.\n"
                f"Requirements:\n"
                f"- Exclude identical columns (feature_1 != feature_2)\n"
                f"- Both features MUST exist in the feature list provided above.\n"
                f"- Provide a concise 1-sentence domain rationale for each pair.\n"
                f"Return ONLY a valid JSON array of objects with keys "
                f"'feature_1', 'feature_2', 'rationale'."
            )

            response = client.chat.completions.create(
                model=get_model(),
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=1000
            )

            raw = response.choices[0].message.content

            match = re.search(r'\[.*\]', raw, re.DOTALL)

            if not match:
                return []

            parsed = json.loads(match.group(0))
            llm_pairs = []

            for p in parsed:
                f1 = p.get("feature_1") or p.get("x")
                f2 = p.get("feature_2") or p.get("y")
                rat = p.get(
                    "rationale",
                    "LLM-inferred domain relationship"
                )

                if (
                    f1
                    and f2
                    and f1 in df.columns
                    and f2 in df.columns
                    and f1 != f2
                ):
                    llm_pairs.append({
                        "feature_1": f1,
                        "feature_2": f2,
                        "rationale": rat,
                        "source": "llm_inferred"
                    })

            return llm_pairs[:top_n]

        except Exception as e:
            print(f"[tools] LLM bivariate pair inference warning: {e}")
            return []




    def compute_algorithmic_bivariate_pairs(   self, df: pd.DataFrame, target_col: Optional[str] = None, **kwargs ) -> List[Dict[str, Any]]:
        r"""
                    START
                      │
                      ▼
              Input DataFrame
                      │
                      ▼
             Find valid columns
                      │
                      ▼
              At least 2 columns?
                 /          \
               NO            YES
               │              │
               ▼              ▼
          Return []     Separate columns
                         /          \
                        /            \
                       ▼              ▼
                Numeric columns   Categorical
                (>1 unique)       columns
                       │           (2–50 unique)
                       │              │
                       ▼              ▼
                Pearson r         Cramér's V
                       │              │
                 |r| ≥ 0.25?      V ≥ 0.20?
                  /     \           /     \
                NO       YES      NO       YES
                │         │       │          │
                │         ▼       │          ▼
                │    Add pair     │      Add pair
                │         │       │          │
                └─────────┴───────┴──────────┘
                              │
                              ▼
                     Is target provided?
                         /          \
                       NO            YES
                       │              │
                       │              ▼
                       │       Pair every valid
                       │       feature with target
                       │              │
                       └──────┬───────┘
                              ▼
                       Remove duplicates
                              │
                              ▼
                       Return pair list
        """
        valid_cols = [
            c for c in df.columns
            if not is_non_distributional_column(c, df[c])
        ]

        if len(valid_cols) < 2:
            return []

        pairs = {}

        # Numeric ↔ Numeric: Pearson correlation
        num_cols = [
            c for c in valid_cols
            if _is_numeric_col(df[c]) and df[c].nunique() > 1
        ]

        if len(num_cols) >= 2:
            corr_matrix = df[num_cols].corr()

            for i in range(len(num_cols)):
                for j in range(i + 1, len(num_cols)):
                    c1, c2 = num_cols[i], num_cols[j]
                    val = corr_matrix.loc[c1, c2]

                    if pd.notnull(val) and abs(val) >= 0.25:
                        pk = tuple(sorted([c1, c2]))
                        pairs[pk] = {
                            "feature_1": c1,
                            "feature_2": c2,
                            "rationale": f"Algorithmic Pearson correlation (|r| = {abs(val):.2f})",
                            "correlation": round(float(val), 4)
                        }

        # Categorical ↔ Categorical: Cramér's V
        cat_cols = [
            c for c in valid_cols
            if not _is_numeric_col(df[c])
            and df[c].nunique() > 1 ##ignore with only 1 class
            and df[c].nunique() <= 50 ##ignore with more than 50 unique values
        ]

        if len(cat_cols) >= 2:
            for i in range(len(cat_cols)):
                for j in range(i + 1, len(cat_cols)):
                    c1, c2 = cat_cols[i], cat_cols[j]

                    v = self.hypothesis_tester.cramers_v(
                        df[c1].dropna(),
                        df[c2].dropna()
                    )

                    if v >= 0.2:
                        pk = tuple(sorted([c1, c2]))

                        if pk not in pairs:
                            pairs[pk] = {
                                "feature_1": c1,
                                "feature_2": c2,
                                "rationale": f"Algorithmic Cramér's V association (V = {v:.2f})",
                                "cramers_v": round(float(v), 4)
                            }

        # Every feature can be inspected against the target
        if target_col and target_col in valid_cols:
            for c in valid_cols:
                if c != target_col:
                    pk = tuple(sorted([c, target_col]))

                    if pk not in pairs:
                        pairs[pk] = {
                            "feature_1": c,
                            "feature_2": target_col,
                            "rationale": f"Statistical hypothesis predictor against target '{target_col}'"
                        }

        return list(pairs.values())

    def compute_bivariate_union(self,df: pd.DataFrame,target_col: Optional[str] = None,top_n_algo: int = 10,top_n_llm: int = 5,dataset_name: str = "",**kwargs) -> Dict[str, Any]:
        """
                        START
                           │
                           ▼
                    Input DataFrame
                           │
                           ▼
          ┌────────────────────────────────┐
          │ Algorithmic bivariate pairs    │
          │                                │
          │ Pearson + Cramér's V + Target  │
          └───────────────┬────────────────┘
                          │
                          ▼
                 Algorithmic pairs
                          │
                          │
                          ├──────────────────┐
                          │                  │
                          ▼                  ▼
                 LLM bivariate pairs   Build dictionaries
                          │             for fast lookup
                          ▼                  │
                    LLM pairs               │
                          │                  │
                          └────────┬─────────┘
                                   ▼
                         Take SET UNION
                                   │
                                   ▼
                    ┌─────────────────────────┐
                    │ For every unique pair   │
                    └────────────┬────────────┘
                                 ▼
                    ┌─────────────────────────┐
                    │ Where was pair found?   │
                    └────────────┬────────────┘
                                 │
                 ┌───────────────┼───────────────┐
                 ▼               ▼               ▼
              Algorithm        Both             LLM
                 │               │               │
                 ▼               ▼               ▼
           source=          source=both     source=
           algorithmic                      llm_inferred
                 │               │               │
                 └───────────────┼───────────────┘
                                 ▼
                       Drop missing values
                                 │
                                 ▼
                       Determine variable types
                                 │
              ┌──────────────────┼──────────────────┐
              ▼                  ▼                  ▼
        Cat ↔ Cat          Cat ↔ Numeric       Num ↔ Num
              │                  │                  │
              ▼                  ▼                  ▼
          Crosstab          Group statistics     Sample points
          + normalized       min/Q1/median/Q3/    max 400
                              max
              │                  │                  │
              └──────────────────┼──────────────────┘
                                 ▼
                         Add to union_list
                                 │
                                 ▼
                        Return final results
        """
        algo_pairs = self.compute_algorithmic_bivariate_pairs(
            df,
            target_col=target_col
        )

        llm_pairs = self.infer_llm_bivariate_pairs(
            df,
            dataset_name=dataset_name,
            target_col=target_col,
            top_n=top_n_llm
        )

        algo_pairs_dict = {
            tuple(sorted([p["feature_1"], p["feature_2"]])): p
            for p in algo_pairs
        }

        llm_pairs_dict = {
            tuple(sorted([p["feature_1"], p["feature_2"]])): p
            for p in llm_pairs
        }

        all_keys = set(algo_pairs_dict) | set(llm_pairs_dict)

        union_list = []
        both_count = 0

        for pk in sorted(all_keys):
            in_algo = pk in algo_pairs_dict
            in_llm = pk in llm_pairs_dict

            if in_algo and in_llm:
                source = "both"
                both_count += 1
                rat = (
                    f"{llm_pairs_dict[pk].get('rationale', '')} "
                    "(Also identified statistically by algorithms)"
                )
            elif in_algo:
                source = "algorithmic"
                rat = algo_pairs_dict[pk].get(
                    "rationale",
                    "Algorithmic bivariate relationship"
                )
            else:
                source = "llm_inferred"
                rat = llm_pairs_dict[pk].get(
                    "rationale",
                    "LLM-inferred domain interaction"
                )

            f1, f2 = pk

            clean_df = df[[f1, f2]].dropna()

            if len(clean_df) < 5:
                continue

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

            # Categorical ↔ Categorical
            if f1_is_discrete and f2_is_discrete and f1_nunique <= 15 and f2_nunique <= 15:
                s1 = clean_df[f1].astype(str)
                s2 = clean_df[f2].astype(str)

                ct = pd.crosstab(s1, s2)
                entry["grouped_counts"] = ct.to_dict()

                ct_norm = pd.crosstab(s1, s2, normalize="index")
                entry["crosstab"] = ct_norm.round(4).to_dict()

            # Categorical ↔ Numeric
            elif f1_is_discrete and (f2_is_num or f2_nunique > 10):
                groups = {}

                for cat, group in clean_df.groupby(f1):
                    vals = pd.to_numeric(group[f2], errors="coerce").dropna()

                    if len(vals) > 0:
                        groups[str(cat)] = {
                            "min": round(_safe_float(vals.min()), 4),
                            "q1": round(_safe_float(vals.quantile(0.25)), 4),
                            "median": round(_safe_float(vals.median()), 4),
                            "q3": round(_safe_float(vals.quantile(0.75)), 4),
                            "max": round(_safe_float(vals.max()), 4),
                        }

                entry["groups"] = groups

            # Numeric ↔ Categorical
            elif f2_is_discrete and (f1_is_num or f1_nunique > 10):
                groups = {}

                for cat, group in clean_df.groupby(f2):
                    vals = pd.to_numeric(group[f1], errors="coerce").dropna()

                    if len(vals) > 0:
                        groups[str(cat)] = {
                            "min": round(_safe_float(vals.min()), 4),
                            "q1": round(_safe_float(vals.quantile(0.25)), 4),
                            "median": round(_safe_float(vals.median()), 4),
                            "q3": round(_safe_float(vals.quantile(0.75)), 4),
                            "max": round(_safe_float(vals.max()), 4),
                        }

                entry["groups"] = groups

            # Numeric ↔ Numeric
            else:
                sample_df = (
                    clean_df.sample(n=min(400, len(clean_df)), random_state=42)
                    if len(clean_df) > 400
                    else clean_df
                )

                entry["points"] = [
                    {
                        "x": round(_safe_float(r[f1]), 4),
                        "y": round(_safe_float(r[f2]), 4)
                    }
                    for _, r in sample_df.iterrows()
                ]

            union_list.append(entry)

        return {
            "union_pairs": union_list,
            "algorithmic_count": len(algo_pairs_dict),
            "llm_count": len(llm_pairs_dict),
            "both_count": both_count,
            "union_count": len(union_list)
        }

    def plot_pairplot(self,df: pd.DataFrame,columns: Optional[List[str]] = None,hue: Optional[str] = None,save_path: str = "pairplot.png",output_dir: str = "./sandbox_run",**kwargs) -> Dict[str, Any]:
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


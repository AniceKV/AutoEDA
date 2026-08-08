"""
summary_generator.py
Classful Executive Summary Generator for AutoEDA.
"""

import os
import sys
import json
import re
from typing import Dict, Any, List, Optional
from dotenv import load_dotenv

# Try importing OpenAI client module for OpenRouter API synthesis
try:
    from openai import OpenAI
    HAS_OPENROUTER = True
except ImportError:
    HAS_OPENROUTER = False

load_dotenv(override=True)

# Files to explicitly exclude from the summary generation input
EXCLUDED_FILES = {"generated_analysis.py", "summary_report.md", "executive_summary.md"}
EXCLUDED_EXTENSIONS = {".pyc", ".pyo"}


class ExecutiveSummaryGenerator:
    """
    Classful Executive Summary Generator for scanning pipeline artifacts,
    synthesizing structured Markdown reports, and annotating statistical insights.
    """
    def __init__(self, use_llm_for_importance: bool = True):
        self.use_llm_for_importance = use_llm_for_importance

    def scan_and_load_files(self, target_dir: str) -> Dict[str, Any]:
        """
        Scans the given target directory for created analysis files.
        EXCLUDES 'generated_analysis.py' and compiles structured content from remaining files.
        """
        if not os.path.exists(target_dir):
            raise FileNotFoundError(f"Target directory '{target_dir}' does not exist.")

        files_found = []
        parsed_contents = {}

        for entry in sorted(os.listdir(target_dir)):
            full_path = os.path.join(target_dir, entry)

            if os.path.isdir(full_path) or entry.startswith("."):
                continue

            if entry.lower() in EXCLUDED_FILES:
                continue

            _, ext = os.path.splitext(entry)
            if ext.lower() in EXCLUDED_EXTENSIONS:
                continue

            file_size = os.path.getsize(full_path)
            files_found.append({
                "name": entry,
                "size_bytes": file_size,
                "size_kb": round(file_size / 1024, 2),
                "ext": ext.lower()
            })

            if ext.lower() == ".json":
                try:
                    with open(full_path, "r", encoding="utf-8") as f:
                        parsed_contents[entry] = json.load(f)
                except Exception as e:
                    parsed_contents[entry] = {"error": f"Failed to parse JSON: {str(e)}"}

            elif ext.lower() in [".png", ".jpg", ".jpeg", ".svg"]:
                parsed_contents[entry] = {
                    "type": "image_visualization",
                    "filename": entry,
                    "size_kb": round(file_size / 1024, 2)
                }

            elif ext.lower() in [".txt", ".md", ".csv", ".log"]:
                try:
                    with open(full_path, "r", encoding="utf-8", errors="replace") as f:
                        content = f.read(4000)
                        parsed_contents[entry] = content
                except Exception as e:
                    parsed_contents[entry] = f"Error reading text file: {str(e)}"

        return {
            "target_dir": os.path.abspath(target_dir),
            "files_scanned": [f["name"] for f in files_found],
            "excluded_files": ["generated_analysis.py"],
            "file_details": files_found,
            "contents": parsed_contents
        }

    def caption_for_image(self, filename: str) -> str:
        """Pattern-matched caption for a generated plot filename."""
        name = re.sub(r"\.(png|jpg|jpeg|svg)$", "", filename, flags=re.IGNORECASE)
        lname = name.lower()

        if lname == "correlation_matrix":
            return "Pearson correlation heatmap across numeric features."
        if lname == "categorical_association_matrix":
            return "Cramer's V association heatmap across categorical features."
        if lname == "pairplot":
            return "Pairwise scatter/distribution grid across key numeric features, colored by target."
        if lname == "target_interactions":
            return "Overview of how the top features interact with the target variable."

        m = re.match(r"bivariate_(.+?)_vs_(.+)", name, re.IGNORECASE)
        if m:
            x, y = m.group(1).replace("_", " "), m.group(2).replace("_", " ")
            return f"Relationship between `{x}` and `{y}`."

        m = re.match(r"dist_(.+)", name, re.IGNORECASE)
        if m:
            col = m.group(1).replace("_", " ")
            return f"Distribution of `{col}`."

        return "Generated analysis artifact."

    def fallback_importance_blurb(self, test_name: str, effect_label: str, feature: str, target_col: str) -> str:
        """Deterministic, template-based 'why it matters' sentence."""
        label = (effect_label or "").lower()
        if "large" in label or "strong" in label:
            strength = "strongly"
        elif "medium" in label or "moderate" in label:
            strength = "moderately"
        elif "small" in label or "weak" in label:
            strength = "weakly"
        else:
            strength = "measurably"
        test_display = test_name or "a statistical test"
        return f"`{feature}` {strength} differentiates `{target_col}` outcomes ({test_display}, {effect_label})."

    def generate_column_importance_blurbs(
        self,
        ranked_details: List[Dict[str, Any]],
        target_col: str,
        dataset_name: str,
        use_llm: bool = True
    ) -> Dict[str, str]:
        """Produces short explanations for each statistically significant predictor."""
        blurbs: Dict[str, str] = {}
        api_key = os.getenv("OPENROUTER_API_KEY")

        if use_llm and HAS_OPENROUTER and api_key and ranked_details:
            try:
                client = OpenAI(
                    base_url="https://openrouter.ai/api/v1",
                    api_key=api_key,
                )
                model = os.getenv("SUMMARY_MODEL")

                payload = [
                    {
                        "feature": d.get("feature"),
                        "test": d.get("test"),
                        "effect_size": d.get("effect_size"),
                        "effect_size_label": d.get("effect_size_label"),
                        "p_value": d.get("p_value"),
                    }
                    for d in ranked_details
                ]

                prompt = (
                    "You are annotating an EDA report with short 'why it matters' explanations "
                    f"for statistically significant predictors of the target variable '{target_col}' "
                    f"in the dataset '{dataset_name}'.\n\n"
                    "STRICT RULES (violating any of these makes the output unusable):\n"
                    "1. Do NOT restate, round, alter, or invent any numeric value (effect size, "
                    "p-value, etc). Those numbers are already shown elsewhere in the report by "
                    "deterministic code -- your only job is plain-language real-world relevance.\n"
                    "2. Exactly ONE sentence per feature, under 25 words, no jargon.\n"
                    "3. Describe association/relevance only -- do NOT claim causation.\n"
                    "4. Output STRICT JSON ONLY: an object mapping each feature name to its one-sentence "
                    "string. No markdown, no prose, no code fences outside the JSON object.\n\n"
                    f"FEATURES:\n{json.dumps(payload, indent=2, default=str)}"
                )

                response = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.2
                )
                raw = response.choices[0].message.content or ""

                match = re.search(r"\{.*\}", raw, re.DOTALL)
                parsed = json.loads(match.group(0)) if match else json.loads(raw)

                if isinstance(parsed, dict):
                    for d in ranked_details:
                        feat = d.get("feature")
                        val = parsed.get(feat)
                        if feat and isinstance(val, str) and val.strip():
                            blurbs[feat] = val.strip()

            except Exception as e:
                print(f"[summary_generator] Column-importance LLM synthesis failed ({e}). "
                      f"Falling back to deterministic blurbs for affected features.")

        for d in ranked_details:
            feat = d.get("feature")
            if feat and feat not in blurbs:
                blurbs[feat] = self.fallback_importance_blurb(
                    d.get("test", "Statistical Test"),
                    d.get("effect_size_label", ""),
                    feat,
                    target_col
                )

        return blurbs

    def generate_template_summary(self, data: Dict[str, Any], use_llm_for_importance: bool = True) -> str:
        """Generates a comprehensive Markdown summary report programmatically."""
        target_dir = data["target_dir"]
        files_scanned = data["files_scanned"]
        contents = data["contents"]

        lines = []
        lines.append("# Executive EDA & Dataset Summary Report")
        lines.append(f"**Target Directory:** `{target_dir}`")
        lines.append(f"**Processed Files:** {', '.join([f'`{f}`' for f in files_scanned]) if files_scanned else 'None'}")
        lines.append("**Excluded Files:** `generated_analysis.py` (Script excluded from summary)")
        lines.append("\n---\n")

        metrics = contents.get("metrics.json", {})
        profile = contents.get("metadata_profile.json", {})

        dataset_overview = metrics.get("dataset_overview", {})
        shape = profile.get("dimensions") or dataset_overview.get("raw_shape") or dataset_overview.get("shape", {})
        dataset_name = profile.get("dataset_name") or os.path.basename(dataset_overview.get("dataset_path", "Dataset"))
        target_col = dataset_overview.get("target_column") or "Not Specified"

        lines.append("## 1. Dataset Overview")
        lines.append(f"- **Dataset Identifier:** `{dataset_name}`")
        lines.append(f"- **Dimensions:** `{shape.get('rows', 'N/A')}` rows x `{shape.get('columns', 'N/A')}` columns")
        lines.append(f"- **Target Variable:** `{target_col}`")

        missing_summary = profile.get("missing_values_summary", {})
        if missing_summary:
            lines.append(f"- **Missing Value Columns:** {len(missing_summary)}")
            for col, details in missing_summary.items():
                if isinstance(details, dict):
                    lines.append(f"  - `{col}`: {details.get('missing_count', 0)} missing ({details.get('missing_pct', 0)}%)")
                else:
                    lines.append(f"  - `{col}`: {details}")
        else:
            lines.append("- **Data Quality:** No missing values detected in raw profile.")

        lines.append("\n---\n")

        columns_profile = profile.get("columns", [])
        if columns_profile:
            lines.append("## 1.5 Full Column Statistics")
            lines.append("| Column | Type | Missing % | Unique % | Mean | Median | Std | Skew | Kurtosis |")
            lines.append("|---|---|---|---|---|---|---|---|---|")
            for col in columns_profile:
                c_name = col.get("column", "N/A")
                c_type = col.get("dtype", "N/A")
                c_miss = col.get("missing_pct", "N/A")
                c_uniq = col.get("unique_pct", "N/A")

                def _fmt(val):
                    return round(val, 2) if isinstance(val, (int, float)) else "N/A"

                c_mean = _fmt(col.get("mean"))
                c_med = _fmt(col.get("median"))
                c_std = _fmt(col.get("std"))
                c_skew = _fmt(col.get("skew"))
                c_kurt = _fmt(col.get("kurtosis"))
                lines.append(f"| `{c_name}` | `{c_type}` | {c_miss}% | {c_uniq}% | {c_mean} | {c_med} | {c_std} | {c_skew} | {c_kurt} |")

            lines.append("\n---\n")

        imputation_summary = metrics.get("imputation_summary", {})
        lines.append("## 2. Data Imputation & Preprocessing")
        if isinstance(imputation_summary, dict) and imputation_summary:
            rules = imputation_summary.get("rules_applied")
            col_details = imputation_summary.get("columns")

            if rules and isinstance(rules, list):
                lines.append("**Rules Applied:**")
                for r in rules:
                    lines.append(f"- {r}")
                lines.append("")

            if isinstance(col_details, dict) and col_details:
                imputed_cols = {
                    k: v for k, v in col_details.items()
                    if isinstance(v, dict) and v.get("method") not in (None, "none")
                }
                if imputed_cols:
                    lines.append("| Column | Missing (Before) | Method | Fill Value |")
                    lines.append("|---|---|---|---|")
                    for col, info in imputed_cols.items():
                        miss_before = info.get("missing_before", "N/A")
                        method = info.get("method", "N/A")
                        fill_val = info.get("fill_value", "N/A")
                        lines.append(f"| `{col}` | {miss_before} | {method} | {fill_val} |")
                else:
                    lines.append("No columns required imputation this run.")
            elif not rules:
                for k, v in imputation_summary.items():
                    if not isinstance(v, (dict, list)):
                        lines.append(f"- **{k}:** {v}")
        else:
            lines.append("No explicit imputation actions recorded in metrics.json.")

        lines.append("\n---\n")

        outliers = metrics.get("outlier_analysis", {})
        lines.append("## 3. Outlier Analysis (IQR Method)")
        if isinstance(outliers, dict) and outliers:
            lines.append("| Column | Outlier Count | Outlier Percentage | Bounds (Lower / Upper) |")
            lines.append("|---|---|---|---|")
            for col, info in outliers.items():
                if isinstance(info, dict):
                    count = info.get("outlier_count", info.get("count", "N/A"))
                    pct = info.get("outlier_percentage", info.get("pct", "N/A"))
                    lower = info.get("lower_bound", "N/A")
                    upper = info.get("upper_bound", "N/A")
                    lines.append(f"| `{col}` | {count} | {pct}% | [{lower}, {upper}] |")
                else:
                    lines.append(f"| `{col}` | {info} | N/A | N/A |")
        else:
            lines.append("No numeric outlier statistics reported.")

        lines.append("\n---\n")

        engineered = metrics.get("engineered_features", [])
        lines.append("## 4. Derived Domain Attributes & Composite Metrics")
        if isinstance(engineered, list) and engineered:
            for item in engineered:
                if isinstance(item, dict):
                    fname = item.get("feature_name", "Derived Metric")
                    formula = item.get("formula", item.get("method", "Custom transformation"))
                    purpose = item.get("rationale", item.get("purpose", "Enhance predictive signal"))
                    lines.append(f"- **`{fname}`**: Formula: `{formula}` | Purpose: {purpose}")
                else:
                    lines.append(f"- {item}")
        else:
            lines.append("No custom derived domain metrics synthesized during this run.")

        lines.append("\n---\n")

        stats_tests = metrics.get("statistical_hypothesis_tests", {})
        ranked = stats_tests.get("ranked_significant_details", []) if isinstance(stats_tests, dict) else []
        significant = stats_tests.get("significant_predictors", []) if isinstance(stats_tests, dict) else []
        stats_target = stats_tests.get("target_col", target_col) if isinstance(stats_tests, dict) else target_col

        lines.append("## 5. Statistical Hypothesis Testing & Key Predictors")
        if ranked:
            importance_blurbs = self.generate_column_importance_blurbs(
                ranked, stats_target, dataset_name, use_llm=use_llm_for_importance
            )
            lines.append(
                f"All predictors below were tested against `{stats_target}` and found statistically "
                f"significant (p < 0.05), ranked by effect size."
            )
            lines.append("")
            lines.append("| Feature | Test Type | Effect Size | Label | P-Value | Why It Matters |")
            lines.append("|---|---|---|---|---|---|")
            for d in ranked:
                feat = d.get("feature", "N/A")
                test = d.get("test", "N/A")
                eff = d.get("effect_size", "N/A")
                label = d.get("effect_size_label", "N/A")
                p_val = d.get("p_value", "N/A")
                p_fmt = f"{p_val:.4e}" if isinstance(p_val, (int, float)) else p_val
                why = importance_blurbs.get(feat, "")
                lines.append(f"| `{feat}` | {test} | {eff} | {label} | {p_fmt} | {why} |")
        elif significant:
            lines.append(f"- **Statistically Significant Predictors:** {', '.join([f'`{s}`' for s in significant])}")
            lines.append("_Detailed effect sizes unavailable -- `ranked_significant_details` missing from metrics.json._")
        else:
            lines.append("No statistically significant predictors identified.")

        lines.append("\n---\n")

        corr_analysis = metrics.get("correlation_analysis", {})
        high_corr_pairs = corr_analysis.get("high_correlation_pairs", []) if isinstance(corr_analysis, dict) else []
        cross_redundant = corr_analysis.get("cross_type_redundant_pairs", []) if isinstance(corr_analysis, dict) else []

        lines.append("## 6. Redundancy & Multicollinearity Analysis")
        if high_corr_pairs or cross_redundant:
            if high_corr_pairs:
                lines.append("**Numeric-Numeric High Correlation Pairs (|r| >= 0.85):**")
                lines.append("")
                lines.append("| Feature 1 | Feature 2 | Correlation (r) | Interpretation |")
                lines.append("|---|---|---|---|")
                for pair in high_corr_pairs:
                    if isinstance(pair, dict):
                        lines.append(
                            f"| `{pair.get('feature_1', 'N/A')}` | `{pair.get('feature_2', 'N/A')}` | "
                            f"{pair.get('correlation', 'N/A')} | {pair.get('interpretation', 'N/A')} |"
                        )
                lines.append("")
            if cross_redundant:
                lines.append("**Cross-Type Redundant Pairs (categorical vs. its own numeric/ordinal encoding, Eta >= 0.85):**")
                lines.append("")
                lines.append("| Categorical Feature | Numeric Feature | Correlation Ratio (Eta) | Interpretation |")
                lines.append("|---|---|---|---|")
                for pair in cross_redundant:
                    if isinstance(pair, dict):
                        lines.append(
                            f"| `{pair.get('categorical_feature', 'N/A')}` | `{pair.get('numeric_feature', 'N/A')}` | "
                            f"{pair.get('correlation_ratio_eta', 'N/A')} | {pair.get('interpretation', 'N/A')} |"
                        )
                lines.append("")
                lines.append("_Recommendation: drop one feature from each redundant pair before modeling to avoid multicollinearity._")
        else:
            lines.append("No high-correlation or cross-type redundant feature pairs detected (threshold: |r| or Eta >= 0.85).")

        lines.append("\n---\n")

        images = [name for name, val in contents.items() if isinstance(val, dict) and val.get("type") == "image_visualization"]
        lines.append("## 7. Generated Visualizations")
        if images:
            for img in images:
                kb = contents[img]["size_kb"]
                caption = self.caption_for_image(img)
                lines.append(f"- **`{img}`** ({kb} KB) -- {caption}")
        else:
            lines.append("No custom chart image assets found in directory (Interactive Plotly visualizations generated directly in HTML report).")

        lines.append("\n---\n")

        cat_assoc = metrics.get("categorical_associations", [])
        lines.append("## 8. Categorical Associations (Cramer's V)")
        if isinstance(cat_assoc, dict):
            top_cat = cat_assoc.get("top_correlations", [])
        elif isinstance(cat_assoc, list):
            top_cat = cat_assoc
        else:
            top_cat = []

        if top_cat:
            lines.append("| Feature 1 | Feature 2 | Cramer's V |")
            lines.append("|---|---|---|")
            for pair in top_cat:
                if isinstance(pair, dict):
                    f1 = pair.get("feature_1", "N/A")
                    f2 = pair.get("feature_2", "N/A")
                    v = pair.get("cramers_v", "N/A")
                    lines.append(f"| `{f1}` | `{f2}` | {v} |")
        else:
            lines.append("No categorical associations available.")

        lines.append("\n---\n")

        blueprint = metrics.get("predictive_modeling_blueprint", {})
        lines.append("## 9. Predictive Modeling Strategy Blueprint")
        if isinstance(blueprint, dict) and blueprint:
            for key, val in blueprint.items():
                title = key.replace("_", " ").title()
                if isinstance(val, list):
                    lines.append(f"### {title}")
                    for item in val:
                        lines.append(f"- {item}")
                elif isinstance(val, dict):
                    lines.append(f"### {title}")
                    for subk, subv in val.items():
                        lines.append(f"- **{subk}:** {subv}")
                else:
                    lines.append(f"- **{title}:** {val}")
        else:
            lines.append("No predictive modeling blueprint generated.")

        lines.append("\n---\n")
        lines.append("*Report generated automatically by `summary_generator.py`*")

        return "\n".join(lines)

    def generate_llm_summary(self, data: Dict[str, Any]) -> str:
        """Uses OpenRouter API to synthesize an executive summary report."""
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            print("[summary_generator] OPENROUTER_API_KEY not found. Falling back to template summary.")
            return self.generate_template_summary(data)

        try:
            client = OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=api_key,
            )
            model = os.getenv("SUMMARY_MODEL", "google/gemini-3.6-flash")

            prompt = (
                "You are a Senior Lead Data Scientist summarizing the outputs of an automated Exploratory Data Analysis (EDA) pipeline.\n"
                "Below is the data extracted from all generated artifact files in the working directory.\n"
                "NOTE: The script file `generated_analysis.py` has been explicitly excluded from this context as requested.\n\n"
                "CRITICAL TRUTH & CONSISTENCY RULES:\n"
                "1. FEATURE ENGINEERING CLAIMS: Base feature engineering claims STRICTLY on 'engineered_features' in metrics.json. If 'engineered_features' is empty or 0 features were synthesized, state clearly: 'No custom derived domain metrics synthesized during this run.' Do NOT invent or claim features (like FamilySize, IsAlone, or log transforms) unless they are explicitly present in 'engineered_features'.\n"
                "2. PREDICTIVE BLUEPRINT CONSISTENCY: Align the predictive modeling strategy section strictly with 'predictive_modeling_blueprint' and target_column in metrics.json. Ensure problem type (Classification vs. Regression) and recommended algorithms match the blueprint widget.\n"
                "3. KEY PREDICTORS COMPLETENESS: In the Key Predictors section/table, you MUST include ALL statistically significant predictors listed under 'statistically_significant_predictors' or 'ranked_significant_details' in metrics.json. DO NOT quietly drop significant predictors. If displaying a top-N table, explicitly title it 'Top N Key Predictors (by Effect Size)' and add a note listing all remaining significant predictors.\n"
                "4. REDUNDANCY COMPLETENESS: If 'correlation_analysis.high_correlation_pairs' or 'correlation_analysis.cross_type_redundant_pairs' is non-empty, you MUST include a 'Redundancy & Multicollinearity' section listing every pair. Do not omit any.\n\n"
                f"FILE SCAN METADATA:\n"
                f"Files Scanned: {data['files_scanned']}\n"
                f"Explicitly Excluded: {data['excluded_files']}\n\n"
                f"FILE CONTENTS & METRICS:\n"
                f"{json.dumps(data['contents'], indent=2, default=str)}\n\n"
                "TASK:\n"
                "Write a comprehensive, highly professional Markdown Executive Summary Report based strictly on the above files.\n"
                "Structure the report logically with headers, tables, key statistical findings, feature engineering highlights, visualization descriptions, redundancy analysis, and predictive modeling blueprints.\n"
                "Ensure plain ASCII characters in table formatting and text."
            )

            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2
            )

            return response.choices[0].message.content

        except Exception as e:
            print(f"[summary_generator] LLM synthesis failed ({e}). Falling back to template summary.")
            return self.generate_template_summary(data)

    def extract_dataset_name(self, target_dir: str) -> str:
        """Extracts dataset_name from generated_analysis.py or metadata_profile.json."""
        script_path = os.path.join(target_dir, "generated_analysis.py")
        if os.path.exists(script_path):
            try:
                with open(script_path, "r", encoding="utf-8", errors="ignore") as f:
                    for i, line in enumerate(f):
                        if i >= 15:
                            break
                        match = re.search(r"DATA_FILEPATH\s*=\s*r?['\"]([^'\"]+)['\"]", line)
                        if match:
                            name, _ = os.path.splitext(os.path.basename(match.group(1)))
                            if name:
                                return name
            except Exception as e:
                print(f"[summary_generator] Error parsing dataset name from script: {e}")

        profile_path = os.path.join(target_dir, "metadata_profile.json")
        if os.path.exists(profile_path):
            try:
                with open(profile_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    ds_name = data.get("dataset_name")
                    if ds_name:
                        return os.path.splitext(os.path.basename(ds_name))[0]
            except Exception:
                pass

        return os.path.basename(os.path.abspath(target_dir))

    def create_summary(
        self,
        directory_path: str = "./sandbox_run",
        output_filename: str = "summary_report.md",
        use_llm: bool = False,
        use_llm_for_importance: bool = True,
        dataset_name: Optional[str] = None
    ) -> str:
        """Scans directory files and writes summary_report.md."""
        print(f"\n==================================================")
        print(f"Summary Generator: Scanning directory '{directory_path}'...")
        print(f"==================================================")

        data = self.scan_and_load_files(directory_path)

        print(f"Found {len(data['files_scanned'])} files: {data['files_scanned']}")
        print(f"Explicitly excluded: {data['excluded_files']}")

        if use_llm and HAS_OPENROUTER and os.getenv("OPENROUTER_API_KEY"):
            print("Generating summary using full-report LLM synthesis (legacy mode)...")
            report_md = self.generate_llm_summary(data)
        else:
            print("Generating summary using structured template engine "
                  f"(column-importance LLM blurbs: {'on' if use_llm_for_importance else 'off'})...")
            report_md = self.generate_template_summary(data, use_llm_for_importance=use_llm_for_importance)

        if not dataset_name:
            dataset_name = self.extract_dataset_name(directory_path)

        output_dir = os.path.join("EDA", dataset_name)
        os.makedirs(output_dir, exist_ok=True)
        out_path = os.path.join(output_dir, output_filename)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(report_md)
        print(f"Summary report successfully written to: {os.path.abspath(out_path)}")

        sandbox_path = os.path.join(directory_path, output_filename)
        if os.path.abspath(sandbox_path) != os.path.abspath(out_path):
            with open(sandbox_path, "w", encoding="utf-8") as f:
                f.write(report_md)

        return report_md


default_summary_generator = ExecutiveSummaryGenerator()


def scan_and_load_files(target_dir: str) -> Dict[str, Any]:
    return default_summary_generator.scan_and_load_files(target_dir)

def _caption_for_image(filename: str) -> str:
    return default_summary_generator.caption_for_image(filename)

def _fallback_importance_blurb(test_name: str, effect_label: str, feature: str, target_col: str) -> str:
    return default_summary_generator.fallback_importance_blurb(test_name, effect_label, feature, target_col)

def generate_column_importance_blurbs(ranked_details: List[Dict[str, Any]], target_col: str, dataset_name: str, use_llm: bool = True) -> Dict[str, str]:
    return default_summary_generator.generate_column_importance_blurbs(ranked_details, target_col, dataset_name, use_llm=use_llm)

def generate_template_summary(data: Dict[str, Any], use_llm_for_importance: bool = True) -> str:
    return default_summary_generator.generate_template_summary(data, use_llm_for_importance=use_llm_for_importance)

def generate_llm_summary(data: Dict[str, Any]) -> str:
    return default_summary_generator.generate_llm_summary(data)

def extract_dataset_name(target_dir: str) -> str:
    return default_summary_generator.extract_dataset_name(target_dir)

def create_summary(directory_path: str = "./sandbox_run", output_filename: str = "summary_report.md", use_llm: bool = False, use_llm_for_importance: bool = True, dataset_name: Optional[str] = None) -> str:
    return default_summary_generator.create_summary(directory_path=directory_path, output_filename=output_filename, use_llm=use_llm, use_llm_for_importance=use_llm_for_importance, dataset_name=dataset_name)


if __name__ == "__main__":
    target_directory = sys.argv[1] if len(sys.argv) > 1 else "./sandbox_run"
    use_llm_flag = "--llm" in sys.argv
    use_importance_llm_flag = "--no-importance-llm" not in sys.argv

    if os.path.exists(target_directory):
        summary_text = create_summary(
            target_directory,
            use_llm=use_llm_flag,
            use_llm_for_importance=use_importance_llm_flag
        )
        print("\n--- REPORT PREVIEW ---")
        print(summary_text[:1000] + ("\n..." if len(summary_text) > 1000 else ""))
    else:
        print(f"Error: Directory '{target_directory}' does not exist.")

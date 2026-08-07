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


def scan_and_load_files(target_dir: str) -> Dict[str, Any]:
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

        # Skip directories and hidden files
        if os.path.isdir(full_path) or entry.startswith("."):
            continue

        # CRITICAL CONTRACT: Exclude generated_analysis.py and self-generated summary files
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

        # Process file contents based on extension
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
                    content = f.read(4000) # Preview first 4KB
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


def generate_template_summary(data: Dict[str, Any]) -> str:
    """
    Generates a comprehensive Markdown summary report programmatically
    from the scanned files without relying on external API calls.
    """
    target_dir = data["target_dir"]
    files_scanned = data["files_scanned"]
    contents = data["contents"]

    lines = []
    lines.append("# Executive EDA & Dataset Summary Report")
    lines.append(f"**Target Directory:** `{target_dir}`")
    lines.append(f"**Processed Files:** {', '.join([f'`{f}`' for f in files_scanned]) if files_scanned else 'None'}")
    lines.append("**Excluded Files:** `generated_analysis.py` (Script excluded from summary)")
    lines.append("\n---\n")

    # 1. Check for metrics.json or metadata_profile.json
    metrics = contents.get("metrics.json", {})
    profile = contents.get("metadata_profile.json", {})

    # Dataset Overview
    dataset_overview = metrics.get("dataset_overview", {})
    shape = dataset_overview.get("shape") or profile.get("dimensions", {})
    dataset_name = profile.get("dataset_name") or os.path.basename(dataset_overview.get("dataset_path", "Dataset"))
    target_col = dataset_overview.get("target_column") or "Not Specified"

    lines.append("## 1. Dataset Overview")
    lines.append(f"- **Dataset Identifier:** `{dataset_name}`")
    lines.append(f"- **Dimensions:** `{shape.get('rows', 'N/A')}` rows x `{shape.get('columns', 'N/A')}` columns")
    lines.append(f"- **Target Variable:** `{target_col}`")

    # Missing values summary
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
    
    # 1.5 Full Column Statistics
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

    # 2. Imputation Strategy & Preprocessing
    imputation_summary = metrics.get("imputation_summary", {})
    lines.append("## 2. Data Imputation & Preprocessing")
    if imputation_summary:
        for k, v in imputation_summary.items():
            lines.append(f"- **{k}:** {v}")
    else:
        lines.append("No explicit imputation actions recorded in metrics.json.")

    lines.append("\n---\n")

    # 3. Outlier Profiling
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

    # 4. Derived Domain Attributes & Composite Metrics
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
        lines.append("No custom derived domain metrics recorded.")

    lines.append("\n---\n")

    # 5. Statistical Hypothesis Tests
    stats_tests = metrics.get("statistical_hypothesis_tests", {})
    significant = stats_tests.get("significant_predictors", [])
    lines.append("## 5. Statistical Hypothesis Testing")
    if significant:
        lines.append(f"- **Statistically Significant Predictors:** {', '.join([f'`{s}`' for s in significant])}")
    
    if isinstance(stats_tests, dict):
        for feature, details in stats_tests.items():
            if feature == "significant_predictors" or not isinstance(details, dict):
                continue
            test_name = details.get("test_name", "Hypothesis Test")
            p_val = details.get("p_value", details.get("p_val", "N/A"))
            sig = details.get("is_statistically_significant", False)
            interp = details.get("interpretation", "N/A")
            lines.append(f"- **`{feature}`** ({test_name}): p-value = `{p_val}` | Significant: `{sig}`")
            lines.append(f"  - *Interpretation:* {interp}")

    lines.append("\n---\n")

    # 6. Generated Visualizations
    images = [name for name, val in contents.items() if isinstance(val, dict) and val.get("type") == "image_visualization"]
    lines.append("## 6. Generated Visual Artifacts")
    if images:
        for img in images:
            kb = contents[img]["size_kb"]
            lines.append(f"- **![{img}]({img})** - `{img}` ({kb} KB)")
    else:
        lines.append("No PNG/SVG image assets found in directory.")

    lines.append("\n---\n")

    # Categorical Associations
    cat_assoc = metrics.get("categorical_associations", [])
    if cat_assoc:
        lines.append("## Categorical Associations (Cramér's V)")
        if isinstance(cat_assoc, dict):
            top_cat = cat_assoc.get("top_correlations", [])
        elif isinstance(cat_assoc, list):
            top_cat = cat_assoc
        else:
            top_cat = []

        if top_cat:
            lines.append("| Feature 1 | Feature 2 | Cramér's V |")
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

    # 7. Predictive Modeling Blueprint
    blueprint = metrics.get("predictive_modeling_blueprint", {})
    lines.append("## 7. Predictive Modeling Strategy Blueprint")
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


def generate_llm_summary(data: Dict[str, Any]) -> str:
    """
    Uses OpenRouter API to synthesize an executive summary report
    from all created files (excluding generated_analysis.py).
    """
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print("[summary_generator] OPENROUTER_API_KEY not found. Falling back to template summary.")
        return generate_template_summary(data)

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
            "3. KEY PREDICTORS COMPLETENESS: In the Key Predictors section/table, you MUST include ALL statistically significant predictors listed under 'statistically_significant_predictors' or 'ranked_significant_details' in metrics.json. DO NOT quietly drop significant predictors. If displaying a top-N table, explicitly title it 'Top N Key Predictors (by Effect Size)' and add a note listing all remaining significant predictors.\n\n"
            f"FILE SCAN METADATA:\n"
            f"Files Scanned: {data['files_scanned']}\n"
            f"Explicitly Excluded: {data['excluded_files']}\n\n"
            f"FILE CONTENTS & METRICS:\n"
            f"{json.dumps(data['contents'], indent=2, default=str)}\n\n"
            "TASK:\n"
            "Write a comprehensive, highly professional Markdown Executive Summary Report based strictly on the above files.\n"
            "Structure the report logically with headers, tables, key statistical findings, feature engineering highlights, image artifact descriptions, and predictive modeling blueprints.\n"
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
        return generate_template_summary(data)


def extract_dataset_name(target_dir: str) -> str:
    """
    Extracts dataset_name from the first lines of generated_analysis.py if present:
    e.g. DATA_FILEPATH = r'C:\...\test_data\dataset_2191_sleep.csv' -> 'dataset_2191_sleep'
    """
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

    # Fallback 1: check metadata_profile.json
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

    # Fallback 2: folder name
    return os.path.basename(os.path.abspath(target_dir))


def create_summary(
    directory_path: str = "./sandbox_run",
    output_filename: str = "summary_report.md",
    use_llm: bool = False,
    dataset_name: Optional[str] = None
) -> str:
    """
    Main function to scan directory files (excluding generated_analysis.py),
    generate executive summary report, and write it to {directory_path}/summary_report.md.
    The caller (agent_loop.py) is responsible for copying it to EDA/{dataset_name}/.
    """
    print(f"\n==================================================")
    print(f"Summary Generator: Scanning directory '{directory_path}'...")
    print(f"==================================================")

    data = scan_and_load_files(directory_path)

    print(f"Found {len(data['files_scanned'])} files: {data['files_scanned']}")
    print(f"Explicitly excluded: {data['excluded_files']}")

    if use_llm and HAS_OPENROUTER and os.getenv("OPENROUTER_API_KEY"):
        print("Generating summary using LLM synthesis...")
        report_md = generate_llm_summary(data)
    else:
        print("Generating summary using structured template engine...")
        report_md = generate_template_summary(data)

    # Resolve dataset_name only if caller didn't provide it
    if not dataset_name:
        dataset_name = extract_dataset_name(directory_path)

    # Write report into EDA/{dataset_name}/ for primary consumption
    output_dir = os.path.join("EDA", dataset_name)
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, output_filename)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(report_md)
    print(f"Summary report successfully written to: {os.path.abspath(out_path)}")

    # Also write into sandbox_run so the asset-copy step picks it up
    sandbox_path = os.path.join(directory_path, output_filename)
    if os.path.abspath(sandbox_path) != os.path.abspath(out_path):
        with open(sandbox_path, "w", encoding="utf-8") as f:
            f.write(report_md)

    return report_md


if __name__ == "__main__":
    # Determine target directory from CLI argument or default to ./sandbox_run
    target_directory = sys.argv[1] if len(sys.argv) > 1 else "./sandbox_run"
    
    # Optional flag --no-llm to force non-LLM template mode
    use_llm_flag = "--no-llm" not in sys.argv

    if os.path.exists(target_directory):
        summary_text = create_summary(target_directory, use_llm=use_llm_flag)
        print("\n--- REPORT PREVIEW ---")
        print(summary_text[:1000] + ("\n..." if len(summary_text) > 1000 else ""))
    else:
        print(f"Error: Directory '{target_directory}' does not exist.")

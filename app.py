import os
import glob
import json
import re
import pandas as pd
import numpy as np
import streamlit as st

from agent_loop import run_tool_based_eda
from summary_generator import extract_dataset_name
from profiler import calculate_column_stats

# Page Configuration
st.set_page_config(
    page_title="AutoEDA Pro - Autonomous Data Science Agent",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Modern Cyber-Minimalist Aesthetic & Zero Text Clutter Layout
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');
    
    html, body, [class*="st-"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Global Container Background */
    .main {
        background: #080b11;
        color: #f1f5f9;
    }
    
    .stApp {
        background: radial-gradient(circle at 50% 0%, rgba(99, 102, 241, 0.08) 0%, rgba(8, 11, 17, 1) 75%);
    }

    .main .block-container {
        padding-top: 1rem;
        padding-bottom: 2.5rem;
        max-width: 96%;
    }
    
    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #0b0f17 !important;
        border-right: 1px solid rgba(255, 255, 255, 0.08);
    }
    
    /* Clean Header Banner */
    .openrouter-header {
        background: rgba(13, 17, 26, 0.75);
        border: 1px solid rgba(255, 255, 255, 0.08);
        backdrop-filter: blur(16px);
        border-radius: 12px;
        padding: 1.2rem 1.6rem;
        margin-bottom: 1.2rem;
        box-shadow: 0 12px 30px -10px rgba(0, 0, 0, 0.5);
        position: relative;
        overflow: hidden;
    }
    
    .openrouter-header::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 2px;
        background: linear-gradient(90deg, #6366f1, #a855f7, #ec4899);
    }
    
    .openrouter-title {
        font-size: 1.7rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        background: linear-gradient(135deg, #ffffff 0%, #cbd5e1 50%, #818cf8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
        line-height: 1.25;
    }
    
    .openrouter-subtitle {
        color: #94a3b8;
        font-size: 0.88rem;
        margin-top: 0.25rem;
        font-weight: 400;
    }
    
    /* Responsive Metric Cards with Zero Clipping */
    .or-metric-card {
        background: rgba(15, 23, 42, 0.6);
        border: 1px solid rgba(255, 255, 255, 0.07);
        border-radius: 10px;
        padding: 0.8rem 0.5rem;
        text-align: center;
        backdrop-filter: blur(12px);
        transition: all 0.2s ease;
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.2);
        overflow: hidden;
    }
    
    .or-metric-card:hover {
        border-color: rgba(99, 102, 241, 0.4);
        transform: translateY(-2px);
    }
    
    .or-metric-val {
        font-size: 1.45rem;
        font-weight: 800;
        background: linear-gradient(135deg, #f8fafc 0%, #c7d2fe 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }
    
    .or-metric-lbl {
        font-size: 0.68rem;
        font-weight: 700;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin-top: 0.2rem;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }
    
    /* Custom Badges */
    .or-badge-numeric {
        background: rgba(99, 102, 241, 0.2);
        border: 1px solid rgba(99, 102, 241, 0.4);
        color: #a5b4fc;
        padding: 0.2rem 0.55rem;
        border-radius: 5px;
        font-weight: 700;
        font-size: 0.7rem;
        letter-spacing: 0.05em;
    }
    
    .or-badge-categorical {
        background: rgba(168, 85, 247, 0.2);
        border: 1px solid rgba(168, 85, 247, 0.4);
        color: #e9d5ff;
        padding: 0.2rem 0.55rem;
        border-radius: 5px;
        font-weight: 700;
        font-size: 0.7rem;
        letter-spacing: 0.05em;
    }
    
    .or-badge-missing {
        background: rgba(239, 68, 68, 0.2);
        border: 1px solid rgba(239, 68, 68, 0.4);
        color: #fca5a5;
        padding: 0.2rem 0.55rem;
        border-radius: 5px;
        font-weight: 700;
        font-size: 0.7rem;
        letter-spacing: 0.05em;
    }
    
    /* Responsive Tabs Bar */
    .stTabs [data-baseweb="tab-list"] {
        gap: 6px;
        background: rgba(13, 17, 26, 0.7);
        padding: 5px;
        border-radius: 10px;
        border: 1px solid rgba(255, 255, 255, 0.08);
        overflow-x: auto;
        flex-wrap: nowrap;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 38px;
        padding: 0.4rem 0.85rem;
        white-space: nowrap;
        border-radius: 7px;
        font-weight: 600;
        font-size: 0.82rem;
        color: #94a3b8;
        border: none !important;
        transition: all 0.2s ease;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%) !important;
        color: #ffffff !important;
        font-weight: 700 !important;
        box-shadow: 0 4px 12px rgba(99, 102, 241, 0.35);
    }

    /* Streamlit Primary Button */
    div.stButton > button {
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
        color: white;
        font-weight: 700;
        border: none;
        border-radius: 8px;
        padding: 0.6rem 1.1rem;
        box-shadow: 0 4px 15px rgba(99, 102, 241, 0.4);
        transition: all 0.2s ease;
    }
    
    div.stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(99, 102, 241, 0.6);
        background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%);
    }

    /* Expanders */
    .stExpander {
        background: rgba(15, 23, 42, 0.4) !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        border-radius: 8px !important;
    }
</style>
""", unsafe_allow_html=True)


def main():
    # Top Branding Header
    st.markdown("""
    <div class="openrouter-header">
        <div class="openrouter-title">Autonomous Tool-Based Data Science Platform</div>
        <div class="openrouter-subtitle">Stateful Agent Pipeline • Algorithmic Profiling & Statistical Modeling Blueprint</div>
    </div>
    """, unsafe_allow_html=True)

    # Sidebar Controls
    st.sidebar.header("Agent Execution Parameters")
    data_source = st.sidebar.radio("Data Input Mode", ["Sample Datasets", "Upload Custom CSV"])
    
    selected_csv_path = None
    
    if data_source == "Upload Custom CSV":
        uploaded_file = st.sidebar.file_uploader("Choose a CSV dataset", type=["csv"])
        if uploaded_file is not None:
            os.makedirs("./temp_uploads", exist_ok=True)
            selected_csv_path = os.path.join("./temp_uploads", uploaded_file.name)
            with open(selected_csv_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            st.sidebar.success(f"File uploaded: `{uploaded_file.name}`")
    else:
        sample_files = sorted(glob.glob("./test_data/*.csv"))
        if sample_files:
            chosen_sample = st.sidebar.selectbox("Select Sample Dataset", sample_files, format_func=lambda x: os.path.basename(x))
            selected_csv_path = chosen_sample
        else:
            st.sidebar.warning("No sample datasets found in `./test_data/`.")

    user_task_request = st.sidebar.text_area(
        "Analysis Request / LLM Prompt",
        value="Perform complete exploratory analysis, type-safe missing value imputation, outlier profiling, statistical hypothesis testing, semantic bivariate graphing, and predictive blueprinting.",
        height=95
    )

    generate_summary_toggle = st.sidebar.toggle("Generate Executive Summary Report", value=True, help="Enable to generate the LLM executive summary report after pipeline completion.")

    run_btn = st.sidebar.button("Run AutoEDA Pipeline", type="primary", use_container_width=True)

    if run_btn and selected_csv_path:
        with st.spinner("AutoEDA Agent executing tool plan & stateful versioning..."):
            res = run_tool_based_eda(
                data_path=selected_csv_path,
                user_request=user_task_request,
                workspace_dir="./sandbox_run",
                generate_summary=generate_summary_toggle
            )
            st.session_state["pipeline_ran"] = True
            st.toast("Pipeline run completed successfully!")

    # Resolve active dataset artifacts strictly matching the currently selected dataset
    current_dataset_name = os.path.splitext(os.path.basename(selected_csv_path))[0] if selected_csv_path else None
    
    eda_dir = None
    if current_dataset_name:
        candidate_eda_dir = os.path.join("EDA", current_dataset_name)
        if os.path.exists(candidate_eda_dir):
            eda_dir = candidate_eda_dir
        elif os.path.exists("./sandbox_run"):
            sandbox_ds = extract_dataset_name("./sandbox_run")
            if sandbox_ds == current_dataset_name:
                eda_dir = "./sandbox_run"
    
    # Check if dataset is loaded
    if selected_csv_path and os.path.exists(selected_csv_path):
        df_raw = pd.read_csv(selected_csv_path)
        stats_list = calculate_column_stats(df_raw)
        
        # High-level Clean Dataset Metrics Cards
        num_rows, num_cols = df_raw.shape
        num_numeric = sum(pd.api.types.is_numeric_dtype(df_raw[c]) and not pd.api.types.is_bool_dtype(df_raw[c]) for c in df_raw.columns)
        num_cat = num_cols - num_numeric
        total_cells = num_rows * num_cols
        missing_cells = df_raw.isnull().sum().sum()
        missing_pct = round((missing_cells / total_cells * 100), 2) if total_cells > 0 else 0.0
        
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.markdown(f'<div class="or-metric-card"><div class="or-metric-val">{num_rows:,}</div><div class="or-metric-lbl">Total Rows</div></div>', unsafe_allow_html=True)
        c2.markdown(f'<div class="or-metric-card"><div class="or-metric-val">{num_cols:,}</div><div class="or-metric-lbl">Total Columns</div></div>', unsafe_allow_html=True)
        c3.markdown(f'<div class="or-metric-card"><div class="or-metric-val">{num_numeric}</div><div class="or-metric-lbl">Numeric Variables</div></div>', unsafe_allow_html=True)
        c4.markdown(f'<div class="or-metric-card"><div class="or-metric-val">{num_cat}</div><div class="or-metric-lbl">Categorical Variables</div></div>', unsafe_allow_html=True)
        c5.markdown(f'<div class="or-metric-card"><div class="or-metric-val">{missing_pct}%</div><div class="or-metric-lbl">Missing Cells</div></div>', unsafe_allow_html=True)
        
        st.write("") # spacing

        # Clean Concise Primary Tabs Bar
        t_vars, t_plots, t_summary, t_metrics, t_script, t_dvc = st.tabs([
            "Variable Profiling",
            "Visual Gallery",
            "Executive Summary",
            "Metrics & Blueprint",
            "Production Code",
            "Data Version Control"
        ])

        # -------------------------------------------------------------
        # TAB 1: VARIABLE PROFILING
        # -------------------------------------------------------------
        with t_vars:
            st.subheader("Individual Variable Deep-Dive")
            search_col = st.text_input("Search Variable Name", "")
            
            filtered_stats = [s for s in stats_list if search_col.lower() in s["column"].lower()] if search_col else stats_list
            
            for s in filtered_stats:
                col_name = s["column"]
                dtype_str = s["dtype"]
                is_num = pd.api.types.is_numeric_dtype(df_raw[col_name]) and not pd.api.types.is_bool_dtype(df_raw[col_name])
                missing_cnt = s["missing_count"]
                missing_pct_val = s["missing_pct"]
                cardinality = s["cardinality"]
                
                with st.expander(f"Variable: **{col_name}** | Type: `{dtype_str}` | Distinct: `{cardinality}` | Missing: `{missing_pct_val}%`", expanded=False):
                    v_col1, v_col2 = st.columns([1, 1.2])
                    
                    with v_col1:
                        st.markdown("#### Statistics Overview")
                        if is_num:
                            st.markdown('<span class="or-badge-numeric">NUMERIC</span>', unsafe_allow_html=True)
                            if missing_cnt > 0:
                                st.markdown(f' <span class="or-badge-missing">MISSING ({missing_cnt})</span>', unsafe_allow_html=True)
                            
                            stat_df = pd.DataFrame({
                                "Metric": ["Mean", "Std Dev", "Median", "Min", "Max", "Q1 (25%)", "Q3 (75%)", "IQR", "Skewness"],
                                "Value": [
                                    s.get("mean"), s.get("std"), s.get("median"), s.get("min"), s.get("max"),
                                    s.get("q1"), s.get("q3"), s.get("iqr"), s.get("skew")
                                ]
                            })
                            st.table(stat_df.dropna())
                        else:
                            st.markdown('<span class="or-badge-categorical">CATEGORICAL</span>', unsafe_allow_html=True)
                            if missing_cnt > 0:
                                st.markdown(f' <span class="or-badge-missing">MISSING ({missing_cnt})</span>', unsafe_allow_html=True)
                            
                            top_vals = df_raw[col_name].value_counts().head(5).reset_index()
                            top_vals.columns = [col_name, "Frequency"]
                            top_vals["Percentage"] = (top_vals["Frequency"] / len(df_raw) * 100).round(2).astype(str) + "%"
                            st.table(top_vals)

                    with v_col2:
                        st.markdown("#### Distribution Asset")
                        clean_col = re.sub(r'\W+', '_', col_name).strip('_')
                        dist_img = os.path.join(eda_dir, f"dist_{clean_col}.png") if eda_dir else ""
                            
                        if dist_img and os.path.exists(dist_img):
                            st.image(dist_img, caption=f"Distribution of {col_name}", use_container_width=True)
                        else:
                            st.info(f"No rendered distribution PNG asset found for '{col_name}'. Run the pipeline to generate.")

        # -------------------------------------------------------------
        # TAB 2: VISUAL GALLERY & BIVARIATE RELATIONSHIPS
        # -------------------------------------------------------------
        with t_plots:
            st.subheader("Generated Visualization Assets")
            if eda_dir and os.path.exists(eda_dir):
                all_pngs = sorted(glob.glob(os.path.join(eda_dir, "*.png")))
                
                if all_pngs:
                    sub_tab1, sub_tab2, sub_tab3 = st.tabs(["Semantic Bivariate (X vs Y)", "Pairplot & Correlation Heatmap", "All Distribution PNGs"])
                    
                    with sub_tab1:
                        biv_pngs = [p for p in all_pngs if "bivariate_" in os.path.basename(p)]
                        target_pngs = [p for p in all_pngs if "target_interactions" in os.path.basename(p)]
                        
                        biv_all = biv_pngs + target_pngs
                        if biv_all:
                            cols = st.columns(2)
                            for idx, p in enumerate(biv_all):
                                with cols[idx % 2]:
                                    st.image(p, caption=os.path.basename(p), use_container_width=True)
                        else:
                            st.info("No bivariate interaction plots found.")

                    with sub_tab2:
                        corr_png = os.path.join(eda_dir, "correlation_matrix.png")
                        pair_png = os.path.join(eda_dir, "pairplot.png")
                        
                        g1, g2 = st.columns(2)
                        with g1:
                            st.markdown("#### Pearson Correlation Heatmap")
                            if os.path.exists(corr_png):
                                st.image(corr_png, use_container_width=True)
                            else:
                                st.info("No correlation_matrix.png found.")
                        with g2:
                            st.markdown("#### Clamped Seaborn Pairplot")
                            if os.path.exists(pair_png):
                                st.image(pair_png, use_container_width=True)
                            else:
                                st.info("No pairplot.png found.")

                    with sub_tab3:
                        dist_pngs = [p for p in all_pngs if "dist_" in os.path.basename(p)]
                        if dist_pngs:
                            cols = st.columns(3)
                            for idx, p in enumerate(dist_pngs):
                                with cols[idx % 3]:
                                    st.image(p, caption=os.path.basename(p), use_container_width=True)
                        else:
                            st.info("No distribution PNGs found.")
                else:
                    st.info("No visual PNG assets generated yet. Click 'Run AutoEDA Pipeline' in the sidebar.")
            else:
                st.info("Run the AutoEDA pipeline to export visual assets for this dataset.")

        # -------------------------------------------------------------
        # TAB 3: EXECUTIVE SUMMARY REPORT
        # -------------------------------------------------------------
        with t_summary:
            summary_file = os.path.join(eda_dir, "summary_report.md") if eda_dir else ""
            if summary_file and os.path.exists(summary_file):
                with open(summary_file, "r", encoding="utf-8") as f:
                    st.markdown(f.read(), unsafe_allow_html=True)
            else:
                st.info("Executive Summary Report was not generated for this run. Enable the 'Generate Executive Summary Report' toggle in the sidebar before running the pipeline to generate it.")

        # -------------------------------------------------------------
        # TAB 4: METRICS & MODELING BLUEPRINT
        # -------------------------------------------------------------
        with t_metrics:
            metrics_file = os.path.join(eda_dir, "metrics.json") if eda_dir else ""
            if metrics_file and os.path.exists(metrics_file):
                with open(metrics_file, "r", encoding="utf-8") as f:
                    metrics_json = json.load(f)
                
                m_sub1, m_sub2, m_sub3 = st.tabs(["Predictive Blueprint Strategy", "Statistical Significance Tests", "Full Canonical metrics.json"])
                
                with m_sub1:
                    bp = metrics_json.get("predictive_modeling_blueprint", {})
                    st.markdown(f"### Problem Type: `{bp.get('problem_type', 'N/A')}`")
                    st.markdown(f"**Target Column:** `{bp.get('target_definition', 'N/A')}`")
                    st.info(f"**Executive Summary:** {bp.get('executive_summary', 'N/A')}")
                    
                    b_c1, b_c2 = st.columns(2)
                    with b_c1:
                        st.markdown("#### Recommended Machine Learning Algorithms")
                        for algo in bp.get("recommended_algorithms", []):
                            st.markdown(f"- `{algo}`")
                            
                        st.markdown("#### Cross-Validation & Validation Strategy")
                        for v in bp.get("validation_strategy", []):
                            st.markdown(f"- {v}")
                    with b_c2:
                        st.markdown("#### Feature Selection & Dimensionality Strategy")
                        for fs in bp.get("feature_selection_strategy", []):
                            st.markdown(f"- {fs}")
                            
                        st.markdown("#### Overfitting Risk Mitigation")
                        for om in bp.get("overfitting_risk_mitigation", []):
                            st.markdown(f"- {om}")

                with m_sub2:
                    st.subheader("Hypothesis Testing & Statistically Significant Predictors")
                    ht = metrics_json.get("statistical_hypothesis_tests", {})
                    sig = ht.get("significant_predictors", [])
                    
                    if sig:
                        st.success(f"Statistically Significant Predictors (α = 0.05): {', '.join([f'`{s}`' for s in sig])}")
                    
                    ht_rows = []
                    for f_name, details in ht.items():
                        if f_name == "significant_predictors" or not isinstance(details, dict):
                            continue
                        ht_rows.append({
                            "Feature": f_name,
                            "Statistical Test": details.get("test_name", "N/A"),
                            "Statistic": details.get("statistic", "N/A"),
                            "p-value": details.get("p_value", "N/A"),
                            "Significant": "Yes" if details.get("is_statistically_significant") else "No",
                            "Interpretation": details.get("interpretation", "N/A")
                        })
                    if ht_rows:
                        st.dataframe(pd.DataFrame(ht_rows), use_container_width=True)

                with m_sub3:
                    st.json(metrics_json)
            else:
                st.info("Run the pipeline to generate metrics.json for this dataset.")

        # -------------------------------------------------------------
        # TAB 5: GENERATED PRODUCTION CODE
        # -------------------------------------------------------------
        with t_script:
            script_path = os.path.join(eda_dir, "generated_analysis.py") if eda_dir else ""
            if script_path and os.path.exists(script_path):
                st.subheader("LLM-Coded Production Feature Engineering & Predictive Blueprint Script")
                with open(script_path, "r", encoding="utf-8") as f:
                    code_text = f.read()
                st.code(code_text, language="python")
                st.download_button(
                    label="Download generated_analysis.py",
                    data=code_text,
                    file_name="generated_analysis.py",
                    mime="text/x-python"
                )
            else:
                st.info("Generated analysis script will be rendered here after running the pipeline for this dataset.")

        # -------------------------------------------------------------
        # TAB 6: STATEFUL DATA VERSION CONTROL (DVC)
        # -------------------------------------------------------------
        with t_dvc:
            st.subheader("DVC Stateful Execution Memory Checkpoints")
            dvc_files = sorted(glob.glob(os.path.join(eda_dir, "df_state_*.csv"))) if eda_dir and os.path.exists(eda_dir) else []
            
            if dvc_files:
                for dvc_f in dvc_files:
                    bname = os.path.basename(dvc_f)
                    with st.expander(f"Checkpoint State: `{bname}`", expanded=True):
                        df_s = pd.read_csv(dvc_f)
                        st.markdown(f"Dimensions: `{len(df_s)}` rows x `{len(df_s.columns)}` columns")
                        st.dataframe(df_s.head(10), use_container_width=True)
            else:
                st.info("DVC state files will appear here after running the pipeline for this dataset.")
    else:
        st.info("Upload a dataset or choose a sample in the sidebar to view YData-style profiling.")


if __name__ == "__main__":
    main()

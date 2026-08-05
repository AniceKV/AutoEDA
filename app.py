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
    page_title="AutoEDA - YData-Style Interactive Data Science Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for YData-Profiling aesthetics & modern dark/light card themes
st.markdown("""
<style>
    /* Global Container */
    .main .block-container {
        padding-top: 1.5rem;
        padding-bottom: 3rem;
        max-width: 95%;
    }
    
    /* Header Branding */
    .brand-header {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        padding: 1.5rem 2rem;
        border-radius: 12px;
        color: #ffffff;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
        margin-bottom: 1.8rem;
    }
    .brand-title {
        font-size: 2.2rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        background: linear-gradient(135deg, #38bdf8 0%, #818cf8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
    }
    .brand-subtitle {
        color: #94a3b8;
        font-size: 1rem;
        margin-top: 0.3rem;
    }
    
    /* Metric Cards */
    .metric-box {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 1.2rem;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        transition: transform 0.2s, box-shadow 0.2s;
    }
    .metric-box:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1);
    }
    .metric-val {
        font-size: 1.8rem;
        font-weight: 700;
        color: #0f172a;
    }
    .metric-lbl {
        font-size: 0.85rem;
        font-weight: 600;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    /* Variable Card (YData Profiling Style) */
    .var-card {
        background: #ffffff;
        border: 1px solid #cbd5e1;
        border-left: 5px solid #3b82f6;
        border-radius: 8px;
        padding: 1.2rem;
        margin-bottom: 1.2rem;
    }
    .badge-numeric {
        background: #dbeafe;
        color: #1e40af;
        padding: 0.25rem 0.6rem;
        border-radius: 6px;
        font-weight: 700;
        font-size: 0.75rem;
    }
    .badge-categorical {
        background: #fef3c7;
        color: #92400e;
        padding: 0.25rem 0.6rem;
        border-radius: 6px;
        font-weight: 700;
        font-size: 0.75rem;
    }
    .badge-missing {
        background: #fee2e2;
        color: #991b1b;
        padding: 0.25rem 0.6rem;
        border-radius: 6px;
        font-weight: 700;
        font-size: 0.75rem;
    }
    
    /* Tab Headers */
    .stTabs [data-baseweb="tab-list"] {
        gap: 6px;
        border-bottom: 2px solid #e2e8f0;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 0.6rem 1.2rem;
        font-weight: 600;
        border-radius: 8px 8px 0 0;
    }
    .stTabs [aria-selected="true"] {
        background-color: #3b82f6 !important;
        color: white !important;
    }
</style>
""", unsafe_allow_html=True)


def main():
    # Brand Top Banner
    st.markdown("""
    <div class="brand-header">
        <div class="brand-title">⚡ AutoEDA Pro Dashboard</div>
        <div class="brand-subtitle">Autonomous Tool-Based Data Science Agent | YData-Style Variable Profiling & Statistical Insights</div>
    </div>
    """, unsafe_allow_html=True)

    # Sidebar Controls
    st.sidebar.header("⚙️ Execution Setup")
    data_source = st.sidebar.radio("Data Input Mode", ["Upload Custom CSV", "Sample Datasets"])
    
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
            st.sidebar.warning("No sample datasets in `./test_data/`.")

    user_task_request = st.sidebar.text_area(
        "Analysis Request / LLM Prompt",
        value="Perform complete exploratory analysis, type-safe missing value imputation, outlier profiling, statistical hypothesis testing, semantic bivariate graphing, and predictive blueprinting.",
        height=100
    )

    run_btn = st.sidebar.button("🚀 Run AutoEDA Pipeline", type="primary", use_container_width=True)

    if run_btn and selected_csv_path:
        with st.spinner("🤖 AutoEDA Agent executing tool plan & stateful versioning..."):
            res = run_tool_based_eda(
                data_path=selected_csv_path,
                user_request=user_task_request,
                workspace_dir="./sandbox_run"
            )
            st.session_state["pipeline_ran"] = True
            st.toast("Pipeline run completed successfully!", icon="🎉")

    # Resolve active dataset artifacts
    dataset_name = extract_dataset_name("./sandbox_run") if os.path.exists("./sandbox_run") else None
    eda_dir = os.path.join("EDA", dataset_name) if dataset_name else "./sandbox_run"
    
    # Check if dataset is loaded
    if selected_csv_path and os.path.exists(selected_csv_path):
        df_raw = pd.read_csv(selected_csv_path)
        stats_list = calculate_column_stats(df_raw)
        
        # High-level Dataset Metrics Bar (YData Overview Style)
        num_rows, num_cols = df_raw.shape
        num_numeric = sum(pd.api.types.is_numeric_dtype(df_raw[c]) for c in df_raw.columns)
        num_cat = num_cols - num_numeric
        total_cells = num_rows * num_cols
        missing_cells = df_raw.isnull().sum().sum()
        missing_pct = round((missing_cells / total_cells * 100), 2) if total_cells > 0 else 0.0
        
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.markdown(f'<div class="metric-box"><div class="metric-val">{num_rows:,}</div><div class="metric-lbl">Number of Rows</div></div>', unsafe_allow_html=True)
        c2.markdown(f'<div class="metric-box"><div class="metric-val">{num_cols:,}</div><div class="metric-lbl">Number of Cols</div></div>', unsafe_allow_html=True)
        c3.markdown(f'<div class="metric-box"><div class="metric-val">{num_numeric}</div><div class="metric-lbl">Numeric Variables</div></div>', unsafe_allow_html=True)
        c4.markdown(f'<div class="metric-box"><div class="metric-val">{num_cat}</div><div class="metric-lbl">Categorical Variables</div></div>', unsafe_allow_html=True)
        c5.markdown(f'<div class="metric-box"><div class="metric-val">{missing_pct}%</div><div class="metric-lbl">Missing Cells</div></div>', unsafe_allow_html=True)
        
        st.write("") # spacing

        # Primary Tabs: YData-Profiling Experience
        t_vars, t_plots, t_summary, t_metrics, t_script, t_dvc = st.tabs([
            "🔍 Variable Profiling",
            "🖼️ Visual Gallery & Bivariate",
            "📄 Executive Summary Report",
            "📊 Metrics & Modeling Blueprint",
            "🐍 Production Code (generated_analysis.py)",
            "💾 Stateful Data Version Control"
        ])

        # -------------------------------------------------------------
        # TAB 1: VARIABLE PROFILING (YData Profiling Style Per-Column Cards)
        # -------------------------------------------------------------
        with t_vars:
            st.subheader("📌 Individual Variable Deep-Dive")
            search_col = st.text_input("🔍 Search Variable Name", "")
            
            filtered_stats = [s for s in stats_list if search_col.lower() in s["column"].lower()] if search_col else stats_list
            
            for s in filtered_stats:
                col_name = s["column"]
                dtype_str = s["dtype"]
                is_num = pd.api.types.is_numeric_dtype(df_raw[col_name])
                missing_cnt = s["missing_count"]
                missing_pct_val = s["missing_pct"]
                cardinality = s["cardinality"]
                
                with st.expander(f"Variable: **{col_name}** | Type: `{dtype_str}` | Distinct: `{cardinality}` | Missing: `{missing_pct_val}%`", expanded=False):
                    v_col1, v_col2 = st.columns([1, 1.2])
                    
                    with v_col1:
                        st.markdown("#### Statistics Overview")
                        if is_num:
                            st.markdown('<span class="badge-numeric">NUMERIC</span>', unsafe_allow_html=True)
                            if missing_cnt > 0:
                                st.markdown(f' <span class="badge-missing">MISSING ({missing_cnt})</span>', unsafe_allow_html=True)
                            
                            stat_df = pd.DataFrame({
                                "Metric": ["Mean", "Std Dev", "Median", "Min", "Max", "Q1 (25%)", "Q3 (75%)", "IQR", "Skewness"],
                                "Value": [
                                    s.get("mean"), s.get("std"), s.get("median"), s.get("min"), s.get("max"),
                                    s.get("q1"), s.get("q3"), s.get("iqr"), s.get("skew")
                                ]
                            })
                            st.table(stat_df.dropna())
                        else:
                            st.markdown('<span class="badge-categorical">CATEGORICAL</span>', unsafe_allow_html=True)
                            if missing_cnt > 0:
                                st.markdown(f' <span class="badge-missing">MISSING ({missing_cnt})</span>', unsafe_allow_html=True)
                            
                            top_vals = df_raw[col_name].value_counts().head(5).reset_index()
                            top_vals.columns = [col_name, "Frequency"]
                            top_vals["Percentage"] = (top_vals["Frequency"] / len(df_raw) * 100).round(2).astype(str) + "%"
                            st.table(top_vals)

                    with v_col2:
                        st.markdown("#### Distribution Asset")
                        clean_col = re.sub(r'\W+', '_', col_name).strip('_')
                        dist_img = os.path.join(eda_dir, f"dist_{clean_col}.png")
                        if not os.path.exists(dist_img):
                            dist_img = os.path.join("./sandbox_run", f"dist_{clean_col}.png")
                            
                        if os.path.exists(dist_img):
                            st.image(dist_img, caption=f"Distribution of {col_name}", use_container_width=True)
                        else:
                            st.info(f"No rendered distribution PNG asset found for '{col_name}'. Run the pipeline to generate.")

        # -------------------------------------------------------------
        # TAB 2: VISUAL GALLERY & BIVARIATE RELATIONSHIPS
        # -------------------------------------------------------------
        with t_plots:
            st.subheader("🖼️ Generated Visualization Assets")
            if os.path.exists(eda_dir):
                all_pngs = sorted(glob.glob(os.path.join(eda_dir, "*.png")))
                
                if all_pngs:
                    sub_tab1, sub_tab2, sub_tab3 = st.tabs(["🔀 Semantic Bivariate (X vs Y)", "🎯 Pairplot & Correlation Heatmap", "📈 All Distribution PNGs"])
                    
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
                st.info("Run the AutoEDA pipeline to export visual assets.")

        # -------------------------------------------------------------
        # TAB 3: EXECUTIVE SUMMARY REPORT
        # -------------------------------------------------------------
        with t_summary:
            summary_file = os.path.join(eda_dir, "summary_report.md")
            if os.path.exists(summary_file):
                with open(summary_file, "r", encoding="utf-8") as f:
                    st.markdown(f.read(), unsafe_allow_html=True)
            else:
                st.info("Run the pipeline to generate the markdown executive summary report.")

        # -------------------------------------------------------------
        # TAB 4: METRICS & MODELING BLUEPRINT
        # -------------------------------------------------------------
        with t_metrics:
            metrics_file = os.path.join(eda_dir, "metrics.json")
            if os.path.exists(metrics_file):
                with open(metrics_file, "r", encoding="utf-8") as f:
                    metrics_json = json.load(f)
                
                m_sub1, m_sub2, m_sub3 = st.tabs(["🎯 Predictive Blueprint Strategy", "🧪 Statistical Significance Tests", "📋 Full Canonical metrics.json"])
                
                with m_sub1:
                    bp = metrics_json.get("predictive_modeling_blueprint", {})
                    st.markdown(f"### Problem Type: `{bp.get('problem_type', 'N/A')}`")
                    st.markdown(f"**Target Column:** `{bp.get('target_definition', 'N/A')}`")
                    st.info(f"**Executive Summary:** {bp.get('executive_summary', 'N/A')}")
                    
                    b_c1, b_c2 = st.columns(2)
                    with b_c1:
                        st.markdown("#### Recommended Machine Learning Algorithms")
                        for algo in bp.get("recommended_algorithms", []):
                            st.markdown(f"- 🤖 `{algo}`")
                            
                        st.markdown("#### Cross-Validation & Validation Strategy")
                        for v in bp.get("validation_strategy", []):
                            st.markdown(f"- 🛡️ {v}")
                    with b_c2:
                        st.markdown("#### Feature Selection & Dimensionality Strategy")
                        for fs in bp.get("feature_selection_strategy", []):
                            st.markdown(f"- 🎯 {fs}")
                            
                        st.markdown("#### Overfitting Risk Mitigation")
                        for om in bp.get("overfitting_risk_mitigation", []):
                            st.markdown(f"- ⚠️ {om}")

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
                            "Significant": "✅ Yes" if details.get("is_statistically_significant") else "❌ No",
                            "Interpretation": details.get("interpretation", "N/A")
                        })
                    if ht_rows:
                        st.dataframe(pd.DataFrame(ht_rows), use_container_width=True)

                with m_sub3:
                    st.json(metrics_json)
            else:
                st.info("Run the pipeline to generate metrics.json.")

        # -------------------------------------------------------------
        # TAB 5: GENERATED PRODUCTION CODE
        # -------------------------------------------------------------
        with t_script:
            script_path = os.path.join(eda_dir, "generated_analysis.py")
            if os.path.exists(script_path):
                st.subheader("🐍 LLM-Coded Production Feature Engineering & Predictive Blueprint Script")
                with open(script_path, "r", encoding="utf-8") as f:
                    code_text = f.read()
                st.code(code_text, language="python")
                st.download_button(
                    label="📥 Download generated_analysis.py",
                    data=code_text,
                    file_name="generated_analysis.py",
                    mime="text/x-python"
                )
            else:
                st.info("Generated analysis script will be rendered here after running the pipeline.")

        # -------------------------------------------------------------
        # TAB 6: STATEFUL DATA VERSION CONTROL (DVC)
        # -------------------------------------------------------------
        with t_dvc:
            st.subheader("💾 DVC Stateful Execution Memory Checkpoints")
            dvc_files = sorted(glob.glob(os.path.join(eda_dir, "df_state_*.csv")))
            
            if dvc_files:
                for dvc_f in dvc_files:
                    bname = os.path.basename(dvc_f)
                    with st.expander(f"Checkpoint State: `{bname}`", expanded=True):
                        df_s = pd.read_csv(dvc_f)
                        st.markdown(f"Dimensions: `{len(df_s)}` rows x `{len(df_s.columns)}` columns")
                        st.dataframe(df_s.head(10), use_container_width=True)
            else:
                st.info("DVC state files will appear here.")
    else:
        st.info("👈 Upload a dataset or choose a sample in the sidebar to view YData-style profiling.")


if __name__ == "__main__":
    main()

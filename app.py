import os
import glob
import json
import pandas as pd
import streamlit as st

from agent_loop import run_tool_based_eda
from summary_generator import extract_dataset_name

# Page Configuration
st.set_page_config(
    page_title="AutoEDA - AI Tool-Based Data Science Agent",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.3rem;
        font-weight: 700;
        background: linear-gradient(135deg, #0f766e 0%, #0369a1 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        color: #475569;
        font-size: 1.05rem;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 1rem;
        text-align: center;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 45px;
        white-space: pre-wrap;
        border-radius: 6px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


def main():
    st.markdown('<div class="main-header">⚡ AutoEDA: Tool-Based AI Data Science Agent</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Deterministic Function-Calling Architecture with Stateful Execution Memory & Decoupled Rendering</div>', unsafe_allow_html=True)

    # Sidebar Data Selection & Options
    st.sidebar.header("📁 Dataset Configuration")
    data_source = st.sidebar.radio("Data Input Method", ["Upload Custom CSV", "Sample Datasets"])
    
    selected_csv_path = None
    
    if data_source == "Upload Custom CSV":
        uploaded_file = st.sidebar.file_uploader("Upload CSV File", type=["csv"])
        if uploaded_file is not None:
            os.makedirs("./temp_uploads", exist_ok=True)
            selected_csv_path = os.path.join("./temp_uploads", uploaded_file.name)
            with open(selected_csv_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            st.sidebar.success(f"Loaded: `{uploaded_file.name}` ({round(len(uploaded_file.getbuffer())/1024, 1)} KB)")
    else:
        sample_files = glob.glob("./test_data/*.csv")
        if sample_files:
            chosen_sample = st.sidebar.selectbox("Choose Sample Dataset", sample_files, format_func=lambda x: os.path.basename(x))
            selected_csv_path = chosen_sample
            st.sidebar.info(f"Selected sample: `{os.path.basename(selected_csv_path)}`")
        else:
            st.sidebar.warning("No sample CSV files found in `./test_data/`.")

    user_task_request = st.sidebar.text_area(
        "Custom Analysis Request / Prompt",
        value="Perform full exploratory data analysis, type-safe imputation, outlier profiling, statistical hypothesis testing, semantic bivariate graphing, and predictive blueprinting.",
        height=100
    )

    run_pipeline = st.sidebar.button("🚀 Run AutoEDA Pipeline", type="primary", use_container_width=True)

    # Main Area Tabs
    tab_overview, tab_summary, tab_plots, tab_metrics, tab_script, tab_dvc = st.tabs([
        "📋 Dataset Overview",
        "📄 Executive Summary",
        "🖼️ Visual Gallery",
        "📊 Metrics & Blueprint",
        "🐍 Analysis Script",
        "💾 Execution State (DVC)"
    ])

    if selected_csv_path and os.path.exists(selected_csv_path):
        df_preview = pd.read_csv(selected_csv_path)
        
        with tab_overview:
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Rows", f"{len(df_preview):,}")
            col2.metric("Columns", f"{len(df_preview.columns):,}")
            col3.metric("Numeric Cols", f"{sum(pd.api.types.is_numeric_dtype(df_preview[c]) for c in df_preview.columns):,}")
            col4.metric("Categorical Cols", f"{sum(not pd.api.types.is_numeric_dtype(df_preview[c]) for c in df_preview.columns):,}")

            st.subheader("Data Preview")
            st.dataframe(df_preview.head(10), use_container_width=True)
            
            st.subheader("Column Dtypes & Missingness")
            col_info = pd.DataFrame({
                "DataType": df_preview.dtypes.astype(str),
                "Missing Values": df_preview.isnull().sum(),
                "Missing %": (df_preview.isnull().sum() / len(df_preview) * 100).round(2),
                "Unique Values": df_preview.nunique()
            })
            st.dataframe(col_info.T, use_container_width=True)

    if run_pipeline:
        if not selected_csv_path:
            st.error("Please upload a CSV file or select a sample dataset before running!")
            return
            
        with st.spinner("🤖 Running AI Tool-Based AutoEDA Pipeline..."):
            res = run_tool_based_eda(
                data_path=selected_csv_path,
                user_request=user_task_request,
                workspace_dir="./sandbox_run"
            )
            st.session_state["pipeline_run_done"] = True
            st.session_state["last_res"] = res
            st.toast("AutoEDA Pipeline Completed Successfully!", icon="🎉")

    # Render Pipeline Results if Available
    dataset_name = extract_dataset_name("./sandbox_run") if os.path.exists("./sandbox_run") else None
    eda_folder = os.path.join("EDA", dataset_name) if dataset_name else "./sandbox_run"
    
    if os.path.exists(eda_folder):
        summary_file = os.path.join(eda_folder, "summary_report.md")
        metrics_file = os.path.join(eda_folder, "metrics.json")
        script_file = os.path.join(eda_folder, "generated_analysis.py")

        with tab_summary:
            if os.path.exists(summary_file):
                with open(summary_file, "r", encoding="utf-8") as f:
                    summary_md = f.read()
                st.markdown(summary_md, unsafe_allow_html=True)
            else:
                st.info("Run the AutoEDA pipeline to generate the executive summary report.")

        with tab_plots:
            st.subheader("Generated Visual Assets")
            png_files = sorted(glob.glob(os.path.join(eda_folder, "*.png")))
            
            if png_files:
                dist_pngs = [f for f in png_files if "dist_" in os.path.basename(f)]
                biv_pngs = [f for f in png_files if "bivariate_" in os.path.basename(f)]
                other_pngs = [f for f in png_files if f not in dist_pngs and f not in biv_pngs]

                if dist_pngs:
                    st.markdown("#### 📈 Distribution Plots")
                    cols = st.columns(2)
                    for idx, img_path in enumerate(dist_pngs):
                        with cols[idx % 2]:
                            st.image(img_path, caption=os.path.basename(img_path), use_container_width=True)

                if biv_pngs:
                    st.markdown("#### 🔀 Semantic Bivariate Relationships")
                    cols = st.columns(2)
                    for idx, img_path in enumerate(biv_pngs):
                        with cols[idx % 2]:
                            st.image(img_path, caption=os.path.basename(img_path), use_container_width=True)

                if other_pngs:
                    st.markdown("#### 🎯 Overview & Pairwise Artifacts")
                    cols = st.columns(2)
                    for idx, img_path in enumerate(other_pngs):
                        with cols[idx % 2]:
                            st.image(img_path, caption=os.path.basename(img_path), use_container_width=True)
            else:
                st.info("No visual PNG assets generated yet.")

        with tab_metrics:
            if os.path.exists(metrics_file):
                with open(metrics_file, "r", encoding="utf-8") as f:
                    metrics_data = json.load(f)
                
                m1, m2 = st.tabs(["JSON Explorer", "Predictive Blueprint Summary"])
                with m1:
                    st.json(metrics_data)
                with m2:
                    blueprint = metrics_data.get("predictive_modeling_blueprint", {})
                    st.write("**Problem Type:**", blueprint.get("problem_type", "N/A"))
                    st.write("**Target Definition:**", blueprint.get("target_definition", "N/A"))
                    st.write("**Executive Summary:**", blueprint.get("executive_summary", "N/A"))
                    st.subheader("Recommended Algorithms")
                    for algo in blueprint.get("recommended_algorithms", []):
                        st.markdown(f"- `{algo}`")
            else:
                st.info("No metrics.json found.")

        with tab_script:
            if os.path.exists(script_file):
                st.subheader("Generated Production Analysis Script (`generated_analysis.py`)")
                with open(script_file, "r", encoding="utf-8") as f:
                    script_code = f.read()
                st.code(script_code, language="python")
                st.download_button(
                    label="📥 Download generated_analysis.py",
                    data=script_code,
                    file_name="generated_analysis.py",
                    mime="text/x-python"
                )
            else:
                st.info("Generated analysis script will appear here after running the pipeline.")

        with tab_dvc:
            st.subheader("Data Version Control (DVC) Retained States")
            dvc_files = sorted(glob.glob(os.path.join(eda_folder, "df_state_*.csv")))
            if dvc_files:
                for dvc_path in dvc_files:
                    fname = os.path.basename(dvc_path)
                    st.markdown(f"**State Checkpoint:** `{fname}`")
                    df_state = pd.read_csv(dvc_path)
                    st.dataframe(df_state.head(5), use_container_width=True)
            else:
                st.info("DVC state files will be displayed here.")
    else:
        st.info("👈 Upload a dataset or choose a sample in the sidebar and click **Run AutoEDA Pipeline**.")


if __name__ == "__main__":
    main()

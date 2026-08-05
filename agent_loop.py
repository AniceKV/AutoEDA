import os
import json
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv
from executor import CodeExecutorSandbox
from profiler import run_and_save_profile
import shutil

load_dotenv(override=True)
api_key = os.getenv("OPENROUTER_API_KEY")
if not api_key:
    raise ValueError("OPENROUTER_API_KEY is not set. Please set it in your .env file or environment variables.")

# 1. Initialize the client with the API key from environment
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=api_key,
)


MODEL_NAME = "openai/gpt-5.6-luna-pro"

def run_agentic_eda(data_path: str, user_request: str, max_cycles=5):
    sandbox = CodeExecutorSandbox()
    
    # 1. Run the profiler to save the JSON file in the sandbox, 
    # and get the minimal high-level dictionary back for the prompt
    print("Running algorithmic profiler...")
    llm_metadata_summary = run_and_save_profile(
        data_path=data_path, 
        output_dir=sandbox.workspace_dir  # Save directly to './sandbox_run'
    )
    
    # 2. Structure your prompts
    system_prompt = (
        "You are an expert Lead AI Data Scientist. Your role is restricted to high-level reasoning and strategy.\n\n"
        "CRITICAL CODE CONTRACTS:\n"
        "1. Assume the dataset path is stored in the global variable `DATA_FILEPATH`.\n"
        "2. A programmatic profiler has already executed and saved raw statistics to 'metadata_profile.json' in your active directory. "
        "Use this JSON to plan your strategy.\n"
        "3. TYPE-SAFE IMPUTATION RULE: When imputing missing values, check the column's data type first. "
        "Only calculate mean or median on NUMERIC columns. For categorical/string columns (like 'Cabin' or 'Embarked'), "
        "impute using the mode: `df[col].fillna(df[col].mode()[0])` or a placeholder like 'Unknown'.\n"
        "4. NAMESPACE COLLISION RULE: Do NOT name any dictionary, list, or variable as `stats`. "
        "If you need to store final metrics, name your dictionary `metrics_dict` or `results`. "
        "Keep the variable name `stats` strictly reserved for the imported `scipy.stats` module.\n"
        "5. STATS UNPACKING RULE: When running statistical significance tests (like Chi-Square) from scipy.stats, "
        "either capture all 4 values: `chi2, p_val, dof, ex = stats.chi2_contingency(...)`, or capture the p-value float directly without unpacking variables: `p_val = stats.chi2_contingency(...)`. Never write `_, p_val = stats.chi2_contingency(...)`.\n"
        "6. Save all generated visualizations as PNGs directly in the active directory.\n"
        "7. MANDATORY SAVING RULE: `metrics.json` MUST contain ALL extracted information in order to write summary reports and generate insights,add plain text version of correlation matrix in metrics.json. "
        "Save a single structured dictionary (e.g., `metrics_dict`) to 'metrics.json' using `json.dump(..., indent=2)`. "
        "It MUST contain the following sections:\n"
        "   - 'dataset_overview': dataset shape, target column, column summary (dtypes, missing counts, cardinality).\n"
        "   - 'imputation_summary': imputation rules applied and detailed per-column filled values/stats.\n"
        "   - 'outlier_analysis': IQR bounds (Q1, Q3, IQR, lower/upper bounds), outlier counts, and outlier percentages per numeric column.\n"
        "   - 'engineered_features': list of dicts with feature name, formula/method, data type, rationale/purpose, and correlation with target.\n"
        "   - 'correlation_analysis': top positive correlations, top negative correlations, and target correlations.\n"
        "   - 'statistical_hypothesis_tests': dict mapping each feature to test details (test name, statistic value, p-value, degrees of freedom if applicable, is_statistically_significant boolean, interpretation summary), plus a list of 'significant_predictors'.\n"
        "   - 'extracted_insights': list of key_findings, data_quality_issues/caveats, and top key_feature_drivers.\n"
        "   - 'predictive_modeling_blueprint': target definition, problem type, recommended algorithms, feature selection strategy, validation strategy, preprocessing steps, overfitting/risk mitigation, and an overall executive modeling strategy summary.\n"
        "8. SAFE ENCODING RULE: In print statements, use ONLY plain ASCII characters (e.g. use `->` instead of unicode arrows like `\\u2192` or `->`) to avoid Windows console cp1252 character map encoding crashes.\n"
        "9. Output ONLY executable python code wrapped in ```python ... ``` without conversational preambles."
    )
    
    user_prompt = (
        f"### PRE-COMPUTED DATASET METADATA:\n"
        f"{json.dumps(llm_metadata_summary, indent=2)}\n\n"
        f"### USER TASK:\n"
        f"{user_request}\n\n"
        "Generate a targeted Python script to perform the user's task."
    )
    
    conversation_history = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    
    print("\nStarting the Agentic Loop...")
    
    for cycle in range(1, max_cycles + 1):
        print(f"\n--- Cycle {cycle} of {max_cycles} ---")
        print("Querying OpenRouter...")
        
        # Call the LLM
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=conversation_history,
            temperature=0.1 # Low temperature for consistent, structured code output
        )
        
        llm_response = response.choices[0].message.content
        
        # Append the LLM's response to history
        conversation_history.append({"role": "assistant", "content": llm_response})
        
        # Run the code in the sandbox
        print("Running code in sandbox...")
        run_results = sandbox.execute_code(llm_response, data_path)
        
        if run_results["success"]:
            print("***Code executed successfully!***")
            print(f"STDOUT Output:\n{run_results['stdout']}")
            
            # Read metrics.json if generated
            metrics_path = os.path.join(sandbox.workspace_dir, "metrics.json")
            if os.path.exists(metrics_path):
                with open(metrics_path, "r") as f:
                    print(f"Generated Metrics: {json.load(f)}")
            break
        else:
            print("Execution failed!")
            print(f"Error (stderr):\n{run_results['stderr']}")
            
            if cycle == max_cycles:
                print("Reached maximum correction cycles. Exiting without success.")
                break
                
            # Construct the correction feedback prompt
            feedback_prompt = (
                f"The previous code execution failed with the following traceback/error:\n\n"
                f"{run_results['stderr']}\n\n"
                f"Please fix the error and provide the updated complete Python code block. "
                f"Ensure you adhere to our contract (using `DATA_FILEPATH`, plain ASCII characters in print statements, and outputting only code)."
            )
            conversation_history.append({"role": "user", "content": feedback_prompt})
    
     # Extract the dataset name without the path and extension (e.g., "Titanic-Dataset")
    # Added  to extract the actual string from the tuple returned by splitext

    dataset_name = os.path.splitext(os.path.basename(data_path))[0]
    export_dir = os.path.join("EDA", dataset_name)
    
    print(f"\nExporting sandbox assets to: {export_dir}...")
    
    # Create the target directory if it doesn't exist
    os.makedirs(export_dir, exist_ok=True)
    
    # Check if the sandbox directory has files to copy
    if os.path.exists(sandbox.workspace_dir):
        copied_files = []
        for filename in os.listdir(sandbox.workspace_dir):
            src_file = os.path.join(sandbox.workspace_dir, filename)
            
            # Make sure we only copy files (not nested directories)
            if os.path.isfile(src_file):
                dst_file = os.path.join(export_dir, filename)
                
                # Copy the file along with its metadata (permissions, timestamps)
                shutil.copy2(src_file, dst_file)
                copied_files.append(filename)
                
        print(f"Successfully exported {len(copied_files)} assets to the local folder!")
        print(f"Exported files: {copied_files}")
    else:
        print(f"Warning: Sandbox directory '{sandbox.workspace_dir}' was not found. Nothing exported.")

# Quick test trigger
if __name__ == "__main__":
    # Ensure our dummy test data exists
    dummy_csv = "./test_data/dataset_2191_sleep.csv"
    if os.path.exists(dummy_csv):
        run_agentic_eda(
            data_path=dummy_csv, 
            user_request=(
                "Perform an intensive, professional-grade Exploratory Data Analysis (EDA) "
                "and predictive feature engineering strategy on the dataset.\n\n"
                "An algorithmic pre-profiler has already executed and saved the raw statistics to 'metadata_profile.json' "
                "in your active directory. Load and use this profile as your strategic starting point.\n\n"
                "YOUR SCRIPT MUST PERFORM AND OUTPUT THE FOLLOWING:\n"
                "1. Smart Imputation Strategy: Read 'metadata_profile.json'. Identify columns with missing values. "
                "Programmatically apply median imputation for highly skewed columns (skewness > 1 or < -1) and mean imputation for symmetric columns.\n"
                "2. Outlier Profiling: Code an IQR-based (Interquartile Range) outlier detection routine. "
                "Identify and print the percentage of outliers present in key numerical columns.\n"
                "3. Domain-Specific Feature Engineering: Create at least 2 high-signal engineered features "
                "(e.g., if analyzing the Titanic dataset, calculate 'FamilySize' = SibSp + Parch + 1, or extract titles from 'Name').\n"
                "4. Statistical Hypothesis Testing: Perform statistical significance tests (e.g., Chi-Square for categorical columns "
                "or T-tests/ANOVA for numerical columns) against the main target column (e.g., 'Survived') and print the exact p-values.\n"
                "5. Advanced Visualizations:\n"
                "   - Save a Pearson correlation heatmap of all numeric variables as 'correlation_matrix.png'.\n"
                "   - Save a segmented visualization (like a boxplot or violin plot comparing a key numerical feature "
                "distribution segmented by the target variable, e.g., Age distribution vs Survived) as 'target_interactions.png'.\n"
                "6. Comprehensive Metrics JSON: Save a complete 'metrics.json' containing ALL extracted information so downstream applications can write full summaries and generate insights. Include: dataset_overview, imputation_summary, outlier_analysis (Q1, Q3, IQR, lower/upper bounds, rates), engineered_features, correlation_analysis, statistical_hypothesis_tests (test name, statistic value, p-value, significance flag, interpretation), extracted_insights (key_findings, data_quality_issues, key_feature_drivers), and a detailed predictive_modeling_blueprint."
            )
        )
    
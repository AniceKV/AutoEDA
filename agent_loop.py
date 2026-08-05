import os
import json
import re
import shutil
import pandas as pd
from typing import Dict, Any, List, Optional
from openai import OpenAI
from dotenv import load_dotenv

import tools
from profiler import run_and_save_profile
from summary_generator import create_summary, extract_dataset_name

load_dotenv(override=True)

api_key = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENROUTER_API_KEY or OPENAI_API_KEY is not set. Please set it in your environment or .env file.")

# Initialize OpenAI client for OpenRouter or OpenAI
client = OpenAI(
    base_url="https://openrouter.ai/api/v1" if os.getenv("OPENROUTER_API_KEY") else None,
    api_key=api_key,
)

MODEL_NAME = os.getenv("EDA_MODEL", "openai/gpt-5.6-luna-pro")


def parse_llm_json_plan(raw_text: str) -> List[Dict[str, Any]]:
    """
    Cleans markdown code blocks and parses the JSON array tool plan from LLM output.
    """
    pattern = r"```(?:json)?(.*?)```"
    match = re.search(pattern, raw_text, re.DOTALL)
    cleaned = match.group(1).strip() if match else raw_text.strip()
    
    try:
        plan = json.loads(cleaned)
        if isinstance(plan, list):
            return plan
        elif isinstance(plan, dict) and "plan" in plan:
            return plan["plan"]
        elif isinstance(plan, dict) and "tools" in plan:
            return plan["tools"]
        else:
            return [plan]
    except Exception as e:
        print(f"[agent_loop] Warning: Error parsing JSON tool plan ({e}). Attempting regex recovery...")
        # Fallback to extracting JSON array using bracket search
        arr_match = re.search(r"\[.*\]", cleaned, re.DOTALL)
        if arr_match:
            return json.loads(arr_match.group(0))
        raise ValueError(f"Failed to parse valid JSON tool plan from response: {raw_text[:200]}")


def run_tool_based_eda(data_path: str, user_request: str, workspace_dir: str = "./sandbox_run") -> Dict[str, Any]:
    """
    Tool-Based Orchestrator for AutoEDA:
    1. Runs pre-profiler to produce metadata_profile.json
    2. Prompts LLM for a structured JSON Tool Plan
    3. Executes tools deterministically in tools.py
    4. Compiles metrics.json and executes summary_generator.py
    5. Exports all assets to EDA/{dataset_name}/
    """
    os.makedirs(workspace_dir, exist_ok=True)
    abs_data_path = os.path.abspath(data_path)
    
    print("\n==================================================")
    print(f"Tool-Based AutoEDA: Starting analysis on '{abs_data_path}'...")
    print("==================================================")
    
    # 1. Run algorithmic pre-profiler
    print("1. Running pre-profiler...")
    profile_summary = run_and_save_profile(data_path=abs_data_path, output_dir=workspace_dir)
    
    # 2. Build system prompt with tool registry schemas
    tools_catalog_str = json.dumps({
        tool_name: {
            "description": details["description"],
            "parameters": details["parameters"]
        } for tool_name, details in tools.TOOL_REGISTRY.items()
    }, indent=2)

    system_prompt = (
        "You are a Lead AI Data Scientist and Tool Planner.\n"
        "Your sole task is to generate a structured, executable JSON Tool Plan to perform EDA on the user's dataset.\n\n"
        "AVAILABLE TOOLS IN REGISTRY:\n"
        f"{tools_catalog_str}\n\n"
        "CRITICAL PLAN RULES:\n"
        "1. Output ONLY a valid JSON array of tool call objects wrapped in ```json ... ```.\n"
        "2. Each object in the array MUST contain 'tool' (string tool name) and 'args' (dictionary of parameters).\n"
        "3. Follow this standard EDA sequence:\n"
        "   - 'impute_missing_data'\n"
        "   - 'detect_and_handle_outliers'\n"
        "   - 'engineer_features'\n"
        "   - 'run_statistical_hypothesis_tests'\n"
        "   - 'plot_correlation_matrix'\n"
        "   - 'plot_target_interaction'\n"
        "   - 'generate_predictive_blueprint'\n"
        "4. Do NOT output conversational preambles or Python code scripts."
    )
    
    user_prompt = (
        f"### PRE-COMPUTED DATASET METADATA:\n"
        f"{json.dumps(profile_summary, indent=2)}\n\n"
        f"### USER TASK:\n"
        f"{user_request}\n\n"
        "Generate the JSON Tool Plan now."
    )
    
    # 3. Query LLM for Tool Plan
    print("2. Querying LLM for Tool Plan...")
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.1
    )
    
    llm_output = response.choices[0].message.content
    print("Received raw LLM response. Parsing tool plan...")
    
    try:
        tool_plan = parse_llm_json_plan(llm_output)
    except Exception as e:
        print(f"Tool plan parsing failed ({e}). Using robust default fallback plan.")
        tool_plan = [
            {"tool": "impute_missing_data", "args": {}},
            {"tool": "detect_and_handle_outliers", "args": {"action": "profile"}},
            {"tool": "engineer_features", "args": {}},
            {"tool": "run_statistical_hypothesis_tests", "args": {}},
            {"tool": "plot_correlation_matrix", "args": {"save_path": "correlation_matrix.png"}},
            {"tool": "plot_target_interaction", "args": {"save_path": "target_interactions.png"}},
            {"tool": "generate_predictive_blueprint", "args": {}}
        ]

    print(f"Successfully loaded plan with {len(tool_plan)} tool steps!")

    # 4. Execute tool plan against dataset
    print("\n3. Executing Tool Plan...")
    df = pd.read_csv(abs_data_path)
    
    # Execution state holders
    target_col = None
    imputation_res = None
    outlier_res = None
    engineered_res = None
    corr_res = None
    hypothesis_res = None
    blueprint_res = None
    
    for idx, step in enumerate(tool_plan, start=1):
        tool_name = step.get("tool")
        args = step.get("args", {})
        
        if tool_name not in tools.TOOL_REGISTRY:
            print(f"Step {idx}: Skipping unknown tool '{tool_name}'")
            continue
            
        print(f"Step {idx}/{len(tool_plan)}: Executing tool '{tool_name}' with args {args}...")
        
        try:
            if tool_name == "impute_missing_data":
                df, imputation_res = tools.impute_missing_data(df, **args)
                
            elif tool_name == "detect_and_handle_outliers":
                df, outlier_res = tools.detect_and_handle_outliers(df, **args)
                
            elif tool_name == "engineer_features":
                if "target_col" not in args and target_col:
                    args["target_col"] = target_col
                df, engineered_res = tools.engineer_features(df, **args)
                
            elif tool_name == "run_statistical_hypothesis_tests":
                if args.get("target_col"):
                    target_col = args["target_col"]
                hypothesis_res = tools.run_statistical_hypothesis_tests(df, **args)
                if not target_col and "target_col" in args:
                    target_col = args["target_col"]
                    
            elif tool_name == "plot_correlation_matrix":
                args["output_dir"] = workspace_dir
                corr_res = tools.plot_correlation_matrix(df, **args)
                
            elif tool_name == "plot_target_interaction":
                args["output_dir"] = workspace_dir
                if target_col and "target_col" not in args:
                    args["target_col"] = target_col
                plot_res = tools.plot_target_interaction(df, **args)
                target_col = plot_res.get("target_col") or target_col
                
            elif tool_name == "generate_predictive_blueprint":
                if target_col and "target_col" not in args:
                    args["target_col"] = target_col
                blueprint_res = tools.generate_predictive_blueprint(df, **args)
                
        except Exception as e:
            print(f"Error executing step {idx} ({tool_name}): {e}")

    # 5. Save canonical metrics.json
    print("\n4. Compiling and saving canonical metrics.json...")
    metrics_path = tools.compile_and_save_metrics(
        df=df,
        dataset_path=abs_data_path,
        target_col=target_col,
        imputation_res=imputation_res,
        outlier_res=outlier_res,
        engineered_res=engineered_res,
        corr_res=corr_res,
        hypothesis_res=hypothesis_res,
        blueprint_res=blueprint_res,
        output_dir=workspace_dir
    )

    # Write a generated_analysis.py header stub for backward-compatibility dataset name detection
    script_stub_path = os.path.join(workspace_dir, "generated_analysis.py")
    with open(script_stub_path, "w", encoding="utf-8") as f:
        f.write(f"DATA_FILEPATH = r'{abs_data_path}'\n# Executed via Tool-Based AutoEDA Architecture\n")

    # 6. Generate Summary Report
    print("5. Invoking summary_generator...")
    dataset_name = extract_dataset_name(workspace_dir)
    export_dir = os.path.join("EDA", dataset_name)
    os.makedirs(export_dir, exist_ok=True)
    
    summary_text = create_summary(directory_path=workspace_dir, use_llm=True)
    
    # 7. Copy all assets from sandbox_run to EDA/{dataset_name}/
    print(f"6. Exporting sandbox assets to: {export_dir}...")
    copied_files = []
    for entry in os.listdir(workspace_dir):
        src_file = os.path.join(workspace_dir, entry)
        if os.path.isfile(src_file):
            dst_file = os.path.join(export_dir, entry)
            shutil.copy2(src_file, dst_file)
            copied_files.append(entry)
            
    print(f"Successfully exported {len(copied_files)} assets to '{export_dir}'!")
    print(f"Assets: {copied_files}")
    print("\n==================================================")
    print("Tool-Based AutoEDA Pipeline Completed Successfully!")
    print("==================================================")
    
    return {
        "success": True,
        "dataset_name": dataset_name,
        "export_dir": export_dir,
        "copied_files": copied_files
    }


if __name__ == "__main__":
    test_csv = "./test_data/dataset_2191_sleep.csv"
    if os.path.exists(test_csv):
        run_tool_based_eda(
            data_path=test_csv,
            user_request="Perform full exploratory data analysis, imputation, outlier profiling, statistical hypothesis testing, and predictive blueprinting on dataset_2191_sleep.csv."
        )
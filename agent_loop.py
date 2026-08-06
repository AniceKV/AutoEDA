import os
import json
import re
import shutil
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
from openai import OpenAI
from dotenv import load_dotenv

import tools
from profiler import run_and_save_profile
from summary_generator import create_summary, extract_dataset_name

load_dotenv(override=True)

api_key = os.getenv("OPENROUTER_API_KEY")
if not api_key:
    raise ValueError("OPENROUTER_API_KEY is not set. Please set it in your environment or .env file.")

# Initialize OpenRouter API client
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=api_key,
)

MODEL_NAME = os.getenv("EDA_MODEL", "google/gemini-3.6-flash")


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
        arr_match = re.search(r"\[.*\]", cleaned, re.DOTALL)
        if arr_match:
            return json.loads(arr_match.group(0))
        raise ValueError(f"Failed to parse valid JSON tool plan from response: {str(e)}")


def validate_tool_plan(plan: Any) -> Tuple[bool, Optional[str]]:
    """
    Validates that the JSON tool plan is a non-empty list of valid registered tool calls.
    Returns (is_valid, error_description).
    """
    if not isinstance(plan, list) or len(plan) == 0:
        return False, "Tool plan must be a non-empty JSON list of tool call objects."
    for idx, item in enumerate(plan):
        if not isinstance(item, dict):
            return False, f"Item at index {idx} is not a valid JSON dictionary."
        if "tool" not in item:
            return False, f"Item at index {idx} is missing required 'tool' field."
        tool_name = item["tool"]
        if tool_name not in tools.TOOL_REGISTRY:
            return False, f"Item at index {idx} references unknown tool '{tool_name}'. Available tools: {list(tools.TOOL_REGISTRY.keys())}"
        if "args" in item and not isinstance(item["args"], dict):
            return False, f"Item at index {idx} field 'args' must be a dictionary."
    return True, None


def run_tool_based_eda(data_path: str, user_request: str, workspace_dir: str = "./sandbox_run", generate_summary: bool = True) -> Dict[str, Any]:
    """
    Tool-Based Orchestrator for AutoEDA:
    1. Runs pre-profiler to produce metadata_profile.json
    2. Prompts LLM for a structured JSON Tool Plan with direct error feedback routing
    3. Executes tools deterministically with Stateful Data Version Control & rollback protection
    4. Compiles metrics.json and executes summary_generator.py
    5. Exports all assets to EDA/{dataset_name}/
    """
    # Wipe previous sandbox artifacts before starting a fresh run
    if os.path.exists(workspace_dir):
        shutil.rmtree(workspace_dir)
    os.makedirs(workspace_dir)
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
            "parameters": details["model"].model_json_schema()
        } for tool_name, details in tools.TOOL_REGISTRY.items()
    }, indent=2)

    system_prompt = (
        "You are a Lead AI Data Scientist and Tool Planner.\n"
        "Your task is to generate a structured, executable JSON Tool Plan to perform EDA on the user's dataset.\n\n"
        "AVAILABLE TOOLS IN REGISTRY:\n"
        f"{tools_catalog_str}\n\n"
        "CRITICAL PLAN RULES:\n"
        "1. Output ONLY a valid JSON array of tool call objects wrapped in ```json ... ```.\n"
        "2. Each object in the array MUST contain 'tool' (string tool name) and 'args' (dictionary of parameters).\n"
        "3. Include the following EDA sequence:\n"
        "   - 'impute_missing_data'\n"
        "   - 'detect_and_handle_outliers'\n"
        "   - 'plot_feature_distributions': Pass all dataset columns in 'columns' (or omit 'columns' to plot distributions for ALL columns in the dataset).\n"
        "   - 'engineer_features': Pass custom high-signal domain transformations in 'feature_specs'.\n"
        "   - 'run_statistical_hypothesis_tests'\n"
        "   - 'plot_correlation_matrix'\n"
        "   - 'plot_semantic_bivariate_relationships': Perform semantic domain reasoning to choose 2-4 key X vs Y feature pairs and pass in 'bivariate_pairs' (e.g. [{'x': 'Age', 'y': 'Fare', 'hue': 'Survived', 'rationale': '...'}, ...]).\n"
        "   - 'plot_pairplot': Select a reasonable subset of 3-4 key numerical features in 'columns' and pass target in 'hue'.\n"
        "   - 'plot_target_interaction'\n"
        "   - 'generate_predictive_blueprint': Pass tailored predictive strategy parameters in args.\n"
        "4. Do NOT output conversational preambles."
    )
    
    user_prompt = (
        f"### PRE-COMPUTED DATASET METADATA:\n"
        f"{json.dumps(profile_summary, indent=2)}\n\n"
        f"### USER TASK:\n"
        f"{user_request}\n\n"
        "Generate the JSON Tool Plan now."
    )
    
    # 3. Query LLM for Tool Plan with Direct Error Feedback Routing
    print("2. Querying LLM for Tool Plan (with self-correction loop)...")
    MAX_RETRIES = 3
    retry_count = 0
    feedback_error = None
    tool_plan = None
    total_tokens_used = 0

    while retry_count < MAX_RETRIES:
        current_user_prompt = user_prompt
        if feedback_error:
            current_user_prompt += (
                f"\n\n[DIRECT ERROR FEEDBACK FROM PREVIOUS ATTEMPT]:\n"
                f"Your previous tool plan failed validation with error:\n{feedback_error}\n"
                f"Please fix your JSON structure, adhere strictly to the schema, and output a valid JSON Tool Plan."
            )
            
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": current_user_prompt}
                ],
                temperature=0.1
            )
            
            if hasattr(response, "usage") and response.usage:
                total_tokens_used += response.usage.total_tokens
                print(f"[agent_loop] Tokens used: {response.usage.total_tokens} (Total: {total_tokens_used})")
                if total_tokens_used > 100000:
                    print("[agent_loop] Token limit (100k) exceeded! Aborting to prevent runaway cost.")
                    return {"success": False, "error": "Token limit exceeded"}

            llm_output = response.choices[0].message.content
            raw_plan = parse_llm_json_plan(llm_output)
            is_valid, validation_err = validate_tool_plan(raw_plan)
            
            if is_valid:
                tool_plan = raw_plan
                print(f"[agent_loop] Validated tool plan successfully on attempt {retry_count + 1}!")
                break
            else:
                feedback_error = validation_err
                print(f"[agent_loop] Tool plan validation failed (Attempt {retry_count + 1}): {validation_err}")
        except Exception as e:
            feedback_error = f"Failed to parse JSON tool plan: {str(e)}"
            print(f"[agent_loop] Tool plan parsing error (Attempt {retry_count + 1}): {e}")
            
        retry_count += 1

    if not tool_plan:
        print("[agent_loop] Self-correction retries exhausted. Falling back to default execution plan.")
        tool_plan = [
            {"tool": "impute_missing_data", "args": {}},
            {"tool": "detect_and_handle_outliers", "args": {"action": "profile"}},
            {"tool": "plot_feature_distributions", "args": {"save_path": "feature_distributions.png"}},
            {"tool": "engineer_features", "args": {}},
            {"tool": "run_statistical_hypothesis_tests", "args": {}},
            {"tool": "plot_correlation_matrix", "args": {"save_path": "correlation_matrix.png"}},
            {"tool": "plot_semantic_bivariate_relationships", "args": {}},
            {"tool": "plot_pairplot", "args": {}},
            {"tool": "plot_target_interaction", "args": {"save_path": "target_interactions.png"}},
            {"tool": "generate_predictive_blueprint", "args": {}}
        ]

    print(f"Loaded plan with {len(tool_plan)} tool steps!")

    # 4. Execute tool plan against dataset with Stateful Data Version Control Memory
    print("\n3. Executing Tool Plan with Data Version Control...")
    df = pd.read_csv(abs_data_path)
    
    # Initialize DVC Stateful Execution Memory
    data_store = tools.StatefulDataStore(workspace_dir=workspace_dir)
    data_store.set_initial_state(df)
    
    target_col = None
    imputation_res = None
    outlier_res = None
    engineered_res = None
    dist_res = None
    corr_res = None
    hypothesis_res = None
    blueprint_res = None
    
    # Tools that accept output_dir
    _vis_tools = {"plot_feature_distributions", "plot_correlation_matrix",
                  "plot_semantic_bivariate_relationships", "plot_pairplot", "plot_target_interaction"}

    for idx, step in enumerate(tool_plan, start=1):
        tool_name = step.get("tool")
        args = step.get("args", {})
        
        if tool_name not in tools.TOOL_REGISTRY:
            print(f"Step {idx}: Skipping unknown tool '{tool_name}'")
            continue

        # Centralized output_dir injection for all visualization tools
        if tool_name in _vis_tools:
            args["output_dir"] = workspace_dir

        # Validate with Pydantic model to clamp arguments robustly
        try:
            model_class = tools.TOOL_REGISTRY[tool_name]["model"]
            validated_args = model_class(**{k: v for k, v in args.items() if k != "output_dir"})
            args.update(validated_args.model_dump(exclude_unset=True))
        except Exception as e:
            print(f"Step {idx}: Pydantic validation failed for {tool_name}, falling back to raw args. Error: {e}")

        print(f"Step {idx}/{len(tool_plan)}: Executing tool '{tool_name}' with args {args}...")
        
        try:
            if tool_name == "impute_missing_data":
                df, imputation_res = tools.impute_missing_data(df, **args)
                data_store.save_checkpoint(df, "impute_missing_data")
                
            elif tool_name == "detect_and_handle_outliers":
                df, outlier_res = tools.detect_and_handle_outliers(df, **args)
                data_store.save_checkpoint(df, "detect_and_handle_outliers")
                
            elif tool_name == "engineer_features":
                if "target_col" not in args and target_col:
                    args["target_col"] = target_col
                df, engineered_res = tools.engineer_features(df, **args)
                data_store.save_checkpoint(df, "engineer_features")
                
            elif tool_name == "run_statistical_hypothesis_tests":
                if args.get("target_col"):
                    target_col = args["target_col"]
                hypothesis_res = tools.run_statistical_hypothesis_tests(df, **args)
                
            elif tool_name == "plot_feature_distributions":
                dist_res = tools.plot_feature_distributions(df, **args)

            elif tool_name == "plot_correlation_matrix":
                corr_res = tools.plot_correlation_matrix(df, **args)

            elif tool_name == "plot_semantic_bivariate_relationships":
                tools.plot_semantic_bivariate_relationships(df, **args)

            elif tool_name == "plot_pairplot":
                if target_col and "hue" not in args:
                    args["hue"] = target_col
                tools.plot_pairplot(df, **args)
                
            elif tool_name == "plot_target_interaction":
                if target_col and "target_col" not in args:
                    args["target_col"] = target_col
                plot_res = tools.plot_target_interaction(df, **args)
                target_col = plot_res.get("target_col") or target_col
                
            elif tool_name == "generate_predictive_blueprint":
                if target_col and "target_col" not in args:
                    args["target_col"] = target_col
                blueprint_res = tools.generate_predictive_blueprint(df, **args)
                
        except Exception as e:
            print(f"[agent_loop] Error executing step {idx} ({tool_name}): {e}")
            print(f"[agent_loop] Triggering automatic DVC state rollback...")
            df, _ = data_store.rollback()

    # 5. Generate LLM-coded generated_analysis.py script containing domain feature engineering & predictive blueprint
    print("\n4. Generating LLM-coded generated_analysis.py script...")
    script_content = [
        f"DATA_FILEPATH = r'{abs_data_path}'",
        "# Generated Analysis Script purely coded for domain feature engineering & predictive modeling strategy",
        "import pandas as pd",
        "import numpy as np",
        "import json",
        "",
        "df = pd.read_csv(DATA_FILEPATH)",
        "",
        "# --- 1. LLM-Coded Feature Engineering ---",
        f"# Engineered Features Specs: {json.dumps(engineered_res or [], indent=2)}"
    ]
    for feat in (engineered_res or []):
        name = feat.get("feature_name", "feat")
        formula = feat.get("formula", "custom")
        script_content.append(f"# Feature '{name}': {formula}")
    
    script_content.extend([
        "",
        "# --- 2. LLM-Coded Predictive Modeling Strategy Blueprint ---",
        f"predictive_blueprint = {json.dumps(blueprint_res or {}, indent=2)}",
        "",
        "if __name__ == '__main__':",
        "    print('Generated analysis script executed successfully.')",
        "    print('Predictive Blueprint Summary:', predictive_blueprint.get('executive_summary', 'Blueprint created'))"
    ])
    
    script_path = os.path.join(workspace_dir, "generated_analysis.py")
    with open(script_path, "w", encoding="utf-8") as f:
        f.write("\n".join(script_content))

    # 5. Save canonical metrics.json
    print("\n5. Compiling and saving canonical metrics.json...")
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

    # 6. Clean export target directory first to prevent stale artifact persistence
    dataset_name = extract_dataset_name(workspace_dir)
    export_dir = os.path.join("EDA", dataset_name)
    if os.path.exists(export_dir):
        shutil.rmtree(export_dir)
    os.makedirs(export_dir, exist_ok=True)

    # Generate Summary Report (Optional based on toggle)
    if generate_summary:
        print("6. Invoking summary_generator...")
        summary_text = create_summary(directory_path=workspace_dir, use_llm=True, dataset_name=dataset_name)
    else:
        print("6. Skipping summary_generator (generate_summary toggle is OFF)...")
    
    # 7. Copy all assets from sandbox_run to EDA/{dataset_name}/
    print(f"7. Exporting sandbox assets to: {export_dir}...")
    copied_files = []
    for entry in os.listdir(workspace_dir):
        src_file = os.path.join(workspace_dir, entry)
        if os.path.isfile(src_file):
            dst_file = os.path.join(export_dir, entry)
            shutil.copy2(src_file, dst_file)
            copied_files.append(entry)
            
    # Purge intermediate DVC states to optimize sandbox storage
    data_store.purge_intermediate_states()

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
    test_csv = "./test_data/Titanic-Dataset.csv"
    if os.path.exists(test_csv):
        run_tool_based_eda(
            data_path=test_csv,
            user_request="Perform full exploratory data analysis, type-safe imputation, outlier profiling, statistical hypothesis testing, and predictive blueprinting on Titanic-Dataset.csv."
        )
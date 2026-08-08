import os
import json
import re
import shutil
import time
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
from openai import OpenAI
from dotenv import load_dotenv

from . import tools
from .profiler import run_and_save_profile
from .summary_generator import create_summary, extract_dataset_name
from .html_report_generator import generate_html_report
load_dotenv(override=True)


class AutoEDAAgent:
    """
    Classful Agentic Orchestrator for AutoEDA with Multi-Turn Memory and Refinement Loop.
    """
    def __init__(self, max_loops: int = 5, max_retries: int = 3):
        self.max_loops = max_loops
        self.max_retries = max_retries

    def parse_llm_json_plan(self, raw_text: str) -> List[Dict[str, Any]]:
        """Cleans markdown code blocks and parses the JSON array tool plan from LLM output."""
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

    def validate_tool_plan(self, plan: Any) -> Tuple[bool, Optional[str]]:
        """Validates that the JSON tool plan is a non-empty list of valid registered tool calls."""
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

    def run_tool_based_eda(
        self,
        data_path: str,
        user_request: str,
        workspace_dir: str = "./sandbox_run",
        generate_summary: bool = True,
        conversation_history: Optional[List[Dict[str, Any]]] = None,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
        status_callback: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Agentic Tool-Based Orchestrator for AutoEDA."""
        effective_api_key = os.getenv("OPENROUTER_API_KEY") or api_key
        if not effective_api_key:
            raise ValueError("OPENROUTER_API_KEY is missing. Set it in your environment / .env file.")

        effective_model = model_name or os.getenv("EDA_MODEL")

        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=effective_api_key,
        )
        if conversation_history is None:
            conversation_history = []

        resume = len(conversation_history) > 0
        state_file = os.path.join(workspace_dir, "agent_state.json")
        df_file = os.path.join(workspace_dir, "current_df.csv")
        abs_data_path = os.path.abspath(data_path)

        print("\n==================================================")
        print(f"Tool-Based AutoEDA: Starting analysis on '{abs_data_path}'...")
        print("==================================================")

        if not resume:
            if os.path.exists(workspace_dir):
                shutil.rmtree(workspace_dir)
            os.makedirs(workspace_dir)

            print("1. Running pre-profiler...")
            if status_callback:
                status_callback("Running pre-profiler...")
            profile_summary = run_and_save_profile(data_path=abs_data_path, output_dir=workspace_dir)

            agent_state = {
                "target_col": None,
                "imputation_res": None,
                "outlier_res": None,
                "engineered_res": None,
                "dist_res": None,
                "corr_res": None,
                "hypothesis_res": None,
                "blueprint_res": None
            }

            df = pd.read_csv(abs_data_path)
            data_store = tools.StatefulDataStore(workspace_dir=workspace_dir)
            data_store.set_initial_state(df, agent_state)

            try:
                hyp_res = tools.run_statistical_hypothesis_tests(df, output_dir=workspace_dir)
                agent_state["hypothesis_res"] = hyp_res
                if hyp_res and isinstance(hyp_res, dict) and hyp_res.get("target_col"):
                    agent_state["target_col"] = hyp_res["target_col"]
            except Exception as e:
                print(f"[agent_loop] Warning: Error pre-generating hypothesis tests: {e}")

            try:
                bp_res = tools.generate_predictive_blueprint(df, target_col=agent_state.get("target_col"), output_dir=workspace_dir)
                agent_state["blueprint_res"] = bp_res
            except Exception as e:
                print(f"[agent_loop] Warning: Error pre-generating predictive blueprint: {e}")

            user_prompt = (
                f"### PRE-COMPUTED DATASET METADATA:\n"
                f"{json.dumps(profile_summary, indent=2)}\n\n"
                f"### USER TASK:\n"
                f"{user_request}\n\n"
                "Analyze the dataset based on the request. Focus heavily on generating bivariate graphs, target interactions, and pairwise feature plots "
                "Use tools to engineer features, profile missing values, and run hypothesis tests. If it helps to add domain features, add them! "
                "('plot_semantic_bivariate_relationships', 'plot_target_interaction', 'plot_pairplot') to uncover key feature correlations, trends, and domain insights. "
                "If the request is ambiguous, use 'ask_clarifying_question'. When finished, call 'finish_analysis'."
            )
            conversation_history.append({"role": "user", "content": user_prompt})
        else:
            print("1. Resuming existing agent session...")
            try:
                with open(state_file, "r") as f:
                    agent_state = json.load(f)
            except Exception:
                agent_state = {
                    "target_col": None,
                    "imputation_res": None,
                    "outlier_res": None,
                    "engineered_res": None,
                    "dist_res": None,
                    "corr_res": None,
                    "hypothesis_res": None,
                    "blueprint_res": None
                }

            if os.path.exists(df_file):
                df = pd.read_csv(df_file)
            else:
                df = pd.read_csv(abs_data_path)

            data_store = tools.StatefulDataStore(workspace_dir=workspace_dir)
            data_store.set_initial_state(df, agent_state)

            if user_request:
                conversation_history.append({"role": "user", "content": user_request})

        tools_catalog_str = json.dumps({
            tool_name: {
                "description": details["description"],
                "parameters": details["model"].model_json_schema()
            } for tool_name, details in tools.TOOL_REGISTRY.items()
        }, indent=2)

        system_prompt = (
            "You are an Autonomous AI Data Scientist.\n"
            "Your task is to analyze datasets by executing a sequence of tools.\n\n"
            "AVAILABLE TOOLS IN REGISTRY:\n"
            f"{tools_catalog_str}\n\n"
            "CRITICAL PLAN RULES:\n"
            "1. Output ONLY a valid JSON array of tool call objects wrapped in ```json ... ```.\n"
            "2. Each object in the array MUST contain 'tool' (string tool name) and 'args' (dictionary of parameters).\n"
            "3. You can use 'ask_clarifying_question' if the task is ambiguous or you need user input (e.g. asking for the target column).\n"
            "4. You can execute multiple tools in a single plan.\n"
            "5. After executing tools, you will receive feedback with the results. If you need to explore further, output another tool plan.\n"
            "6. If you have completed the analysis or the user's request, you MUST call 'finish_analysis' to conclude the loop.\n"
            "7. Do NOT output conversational preambles.\n"
            "8. Do NOT request univariate distribution plots ('plot_feature_distributions') for identifier columns (ID, UUID, index, key), spatial coordinates (latitude, longitude), or timestamps, as they lack univariate distribution signal.\n"
            "9. SEMANTIC BIVARIATE GRAPH SELECTION: When calling 'plot_semantic_bivariate_relationships' or 'plot_target_interaction', select domain-relevant, semantically meaningful feature pairs (X vs Y) where X != Y. Never request self-bivariate plots where X == Y. Exclude unique ID, index, key, timestamp, or spatial coordinate columns.\n"
            "10. COMPREHENSIVE VISUAL PROFILING: When performing exploratory data analysis, call 'plot_feature_distributions' for key non-identifier features to analyze univariate distributions, alongside bivariate plotting tools ('plot_semantic_bivariate_relationships', 'plot_target_interaction', 'plot_pairplot') and hypothesis testing ('run_statistical_hypothesis_tests'). Skip imputation, outlier handling, and predictive blueprinting unless explicitly requested or needed for severe distortion."
        )

        print("\n2. Starting Agent Refinement Loop...")
        if status_callback:
            status_callback("Starting Agent Refinement Loop...")
        loop_count = 0
        finished = False
        agent_plan_log = []

        _vis_tools = {"plot_feature_distributions", "plot_correlation_matrix",
                      "plot_semantic_bivariate_relationships", "plot_pairplot", "plot_target_interaction"}

        while loop_count < self.max_loops and not finished:
            print(f"\n--- Agent Loop {loop_count + 1} ---")

            retry_count = 0
            feedback_error = None
            tool_plan = None

            while retry_count < self.max_retries:
                messages = [{"role": "system", "content": system_prompt}] + conversation_history
                if feedback_error:
                    messages.append({"role": "user", "content": f"[DIRECT ERROR FEEDBACK]:\nYour previous tool plan failed validation with error:\n{feedback_error}\nPlease fix your JSON structure."})

                try:
                    api_attempts = 0
                    while api_attempts < 3:
                        try:
                            response = client.chat.completions.create(
                                model=effective_model,
                                messages=messages,
                                temperature=0.1
                            )
                            break
                        except Exception as api_err:
                            api_attempts += 1
                            if api_attempts >= 3:
                                raise api_err
                            print(f"[agent_loop] API call warning ({api_err}), retrying in {2 ** api_attempts}s...")
                            time.sleep(2 ** api_attempts)

                    llm_output = response.choices[0].message.content
                    raw_plan = self.parse_llm_json_plan(llm_output)
                    is_valid, validation_err = self.validate_tool_plan(raw_plan)

                    if is_valid:
                        tool_plan = raw_plan
                        conversation_history.append({"role": "assistant", "content": llm_output})
                        break
                    else:
                        feedback_error = validation_err
                except Exception as e:
                    feedback_error = f"Failed to query model or parse JSON plan: {str(e)}"

                retry_count += 1

            if not tool_plan:
                print("[agent_loop] Self-correction retries exhausted. Breaking loop.")
                break

            print(f"Loaded plan with {len(tool_plan)} tool steps!")

            step_results = []
            for idx, step in enumerate(tool_plan, start=1):
                tool_name = step.get("tool")
                args = step.get("args", {})

                if tool_name == "finish_analysis":
                    finished = True
                    step_results.append("Agent called finish_analysis. Terminating loop.")
                    break

                if tool_name == "ask_clarifying_question":
                    question = args.get("question", "I need some clarification to proceed.")
                    print(f"[agent_loop] Agent asked a question: {question}")

                    with open(state_file, "w") as f:
                        json.dump(agent_state, f)
                    df.to_csv(df_file, index=False)

                    return {
                        "success": True,
                        "status": "question",
                        "question": question,
                        "conversation_history": conversation_history
                    }

                if tool_name in _vis_tools:
                    args["output_dir"] = workspace_dir

                try:
                    model_class = tools.TOOL_REGISTRY[tool_name]["model"]
                    validated_args = model_class(**{k: v for k, v in args.items() if k != "output_dir"})
                    args.update(validated_args.model_dump(exclude_unset=True))
                except Exception as e:
                    print(f"Step {idx}: Pydantic validation failed for {tool_name}. Error: {e}")

                args["output_dir"] = workspace_dir

                print(f"Executing '{tool_name}' with args {args}...")
                if status_callback:
                    status_callback(f"Executing tool '{tool_name}'...")

                try:
                    if tool_name == "impute_missing_data":
                        df, agent_state["imputation_res"] = tools.impute_missing_data(df, **args)
                        data_store.save_checkpoint(df, agent_state, "impute_missing_data")
                        step_results.append(f"impute_missing_data successful. Summary: {agent_state['imputation_res']}")

                    elif tool_name == "detect_and_handle_outliers":
                        df, agent_state["outlier_res"] = tools.detect_and_handle_outliers(df, **args)
                        data_store.save_checkpoint(df, agent_state, "detect_and_handle_outliers")
                        step_results.append(f"detect_and_handle_outliers successful. Outlier stats collected.")

                    elif tool_name == "engineer_features":
                        if "target_col" not in args and agent_state["target_col"]:
                            args["target_col"] = agent_state["target_col"]
                        df, engineered = tools.engineer_features(df, **args)
                        if not agent_state["engineered_res"]:
                            agent_state["engineered_res"] = []
                        agent_state["engineered_res"].extend(engineered)
                        data_store.save_checkpoint(df, agent_state, "engineer_features")
                        step_results.append(f"engineer_features successful. Synthesized {len(engineered)} derived domain metrics.")

                    elif tool_name == "run_statistical_hypothesis_tests":
                        if args.get("target_col"):
                            agent_state["target_col"] = args["target_col"]
                        agent_state["hypothesis_res"] = tools.run_statistical_hypothesis_tests(df, **args)
                        step_results.append(f"run_statistical_hypothesis_tests successful. Significant predictors: {agent_state['hypothesis_res'].get('significant_predictors', [])}")

                    elif tool_name == "plot_feature_distributions":
                        agent_state["dist_res"] = tools.plot_feature_distributions(df, **args)
                        step_results.append(f"plot_feature_distributions successful.")

                    elif tool_name == "plot_correlation_matrix":
                        agent_state["corr_res"] = tools.plot_correlation_matrix(df, **args)
                        step_results.append(f"plot_correlation_matrix successful.")

                    elif tool_name == "plot_semantic_bivariate_relationships":
                        res = tools.plot_semantic_bivariate_relationships(df, **args)
                        step_results.append(f"plot_semantic_bivariate_relationships successful. Count: {res.get('count', 0)}")

                    elif tool_name == "plot_pairplot":
                        if agent_state["target_col"] and "hue" not in args:
                            args["hue"] = agent_state["target_col"]
                        res = tools.plot_pairplot(df, **args)
                        step_results.append(f"plot_pairplot successful.")

                    elif tool_name == "plot_target_interaction":
                        if agent_state["target_col"] and "target_col" not in args:
                            args["target_col"] = agent_state["target_col"]
                        plot_res = tools.plot_target_interaction(df, **args)
                        agent_state["target_col"] = plot_res.get("target_col") or agent_state["target_col"]
                        step_results.append(f"plot_target_interaction successful.")

                    elif tool_name == "generate_predictive_blueprint":
                        if agent_state["target_col"] and "target_col" not in args:
                            args["target_col"] = agent_state["target_col"]
                        agent_state["blueprint_res"] = tools.generate_predictive_blueprint(df, **args)
                        step_results.append(f"generate_predictive_blueprint successful.")

                except Exception as e:
                    print(f"[agent_loop] Error executing step {idx} ({tool_name}): {e}")
                    df, agent_state, _ = data_store.rollback()
                    step_results.append(f"Error in {tool_name}: {e}. State rolled back.")

            agent_plan_log.append({
                "loop": loop_count + 1,
                "llm_output": llm_output,
                "plan": tool_plan,
                "step_results": step_results
            })

            if not finished:
                feedback_msg = "Tool Execution Results:\n" + "\n".join(step_results) + "\n\nAnalyze the results. If you need to run more tools, output a new tool plan. If you are completely finished, output `finish_analysis`."
                conversation_history.append({"role": "user", "content": feedback_msg})

            loop_count += 1

        plan_log_path = os.path.join(workspace_dir, "agent_plan_log.json")
        with open(plan_log_path, "w", encoding="utf-8") as f:
            json.dump(agent_plan_log, f, indent=2)

        with open(state_file, "w") as f:
            json.dump(agent_state, f)
        df.to_csv(df_file, index=False)

        print("\n4. Generating LLM-coded generated_analysis.py script...")
        if status_callback:
            status_callback("Generating analysis script...")
        script_content = [
            f"DATA_FILEPATH = r'{abs_data_path}'",
            "# Generated Analysis Script purely coded for domain feature engineering & predictive modeling strategy",
            "import pandas as pd",
            "import numpy as np",
            "import json",
            "",
            "df = pd.read_csv(DATA_FILEPATH)",
            "",
            "# --- 1. Derived Domain Attributes & Composite Metrics ---",
            f"# Derived Domain Metrics Specs: {json.dumps(agent_state['engineered_res'] or [], indent=2)}"
        ]
        for feat in (agent_state['engineered_res'] or []):
            name = feat.get("feature_name", "feat")
            formula = feat.get("formula", "custom")

            if "log1p" in formula or "log" in formula:
                source_col = re.search(r'\((.*?)\)', formula).group(1) if "(" in formula else None
                if source_col and source_col in df.columns:
                    script_content.append(f"df['{name}'] = np.log1p(df['{source_col}'].clip(lower=0))")
                else:
                    script_content.append(f"# Feature '{name}': {formula}")
            elif "/" in formula:
                parts = formula.split("/")
                if len(parts) == 2:
                    num, den = parts[0].strip(), parts[1].strip()
                    script_content.append(f"df['{name}'] = df['{num}'] / (df['{den}'].abs() + 1e-5)")
                else:
                    script_content.append(f"# Feature '{name}': {formula}")
            elif "*" in formula:
                parts = formula.split("*")
                if len(parts) == 2:
                    c1, c2 = parts[0].strip(), parts[1].strip()
                    script_content.append(f"df['{name}'] = df['{c1}'] * df['{c2}']")
                else:
                    script_content.append(f"# Feature '{name}': {formula}")
            elif "+" in formula:
                parts = formula.split("+")
                if len(parts) >= 2:
                    cols = [f"df['{p.strip()}']" for p in parts if p.strip() in df.columns]
                    if cols:
                        script_content.append(f"df['{name}'] = {' + '.join(cols)}")
                    else:
                        script_content.append(f"# Feature '{name}': {formula}")
                else:
                    script_content.append(f"# Feature '{name}': {formula}")
            elif "sum" in formula:
                source_cols = re.search(r'\((.*?)\)', formula).group(1) if "(" in formula else ""
                cols_list = [c.strip() for c in source_cols.split(",") if c]
                script_content.append(f"df['{name}'] = df[{cols_list}].sum(axis=1)")
            else:
                script_content.append(f"# Custom feature placeholder - '{name}': {formula}")

        script_content.extend([
            "",
            "# --- 2. LLM-Coded Predictive Modeling Strategy Blueprint ---",
            f"predictive_blueprint = {json.dumps(agent_state['blueprint_res'] or {}, indent=2)}",
            "",
            "if __name__ == '__main__':",
            "    print('Generated analysis script executed successfully.')",
            "    print('Predictive Blueprint Summary:', predictive_blueprint.get('executive_summary', 'Blueprint created'))"
        ])

        script_path = os.path.join(workspace_dir, "generated_analysis.py")
        with open(script_path, "w", encoding="utf-8") as f:
            f.write("\n".join(script_content))

        print("\n5. Compiling and saving canonical metrics.json...")
        if status_callback:
            status_callback("Compiling metrics...")
        if agent_state.get("target_col"):
            try:
                agent_state["blueprint_res"] = tools.generate_predictive_blueprint(
                    df, target_col=agent_state["target_col"], output_dir=workspace_dir
                )
            except Exception as e:
                print(f"[agent_loop] Warning: Error synchronizing blueprint: {e}")

        metrics_path = tools.compile_and_save_metrics(
            df=df,
            dataset_path=abs_data_path,
            target_col=agent_state["target_col"],
            imputation_res=agent_state["imputation_res"],
            outlier_res=agent_state["outlier_res"],
            engineered_res=agent_state["engineered_res"],
            corr_res=agent_state["corr_res"],
            hypothesis_res=agent_state["hypothesis_res"],
            blueprint_res=agent_state["blueprint_res"],
            output_dir=workspace_dir
        )

        dataset_name = extract_dataset_name(workspace_dir)
        export_dir = os.path.join("EDA", dataset_name)
        if os.path.exists(export_dir):
            shutil.rmtree(export_dir)
        os.makedirs(export_dir, exist_ok=True)

        if generate_summary:
            print("\n6. Invoking summary_generator to synthesize Executive Summary...")
            if status_callback:
                status_callback("Synthesizing Executive Summary...")
            summary_text = create_summary(directory_path=workspace_dir, use_llm=False, dataset_name=dataset_name)

        print("\n7. Generate Standalone Interactive HTML Report")
        if status_callback:
            status_callback("Generating HTML Report...")
        generate_html_report(workspace_dir=workspace_dir)

        print(f"8. Exporting sandbox assets to: {export_dir}...")
        if status_callback:
            status_callback("Exporting analysis assets...")
        copied_files = []
        for entry in os.listdir(workspace_dir):
            src_file = os.path.join(workspace_dir, entry)
            if os.path.isfile(src_file) and not entry.endswith(".csv"):
                dst_file = os.path.join(export_dir, entry)
                with open(src_file, "rb") as fsrc, open(dst_file, "wb") as fdst:
                    shutil.copyfileobj(fsrc, fdst, length=64 * 1024)
                copied_files.append(entry)

        data_store.purge_intermediate_states()

        print("\n==================================================")
        print("Agentic AutoEDA Pipeline Completed Successfully!")
        print("==================================================")

        return {
            "success": True,
            "status": "finished",
            "dataset_name": dataset_name,
            "export_dir": export_dir,
            "copied_files": copied_files,
            "conversation_history": conversation_history
        }


default_agent = AutoEDAAgent()


def parse_llm_json_plan(raw_text: str) -> List[Dict[str, Any]]:
    return default_agent.parse_llm_json_plan(raw_text)

def validate_tool_plan(plan: Any) -> Tuple[bool, Optional[str]]:
    return default_agent.validate_tool_plan(plan)

def run_tool_based_eda(
    data_path: str,
    user_request: str,
    workspace_dir: str = "./sandbox_run",
    generate_summary: bool = True,
    conversation_history: Optional[List[Dict[str, Any]]] = None,
    api_key: Optional[str] = None,
    model_name: Optional[str] = None,
    status_callback: Optional[Any] = None,
) -> Dict[str, Any]:
    return default_agent.run_tool_based_eda(
        data_path=data_path,
        user_request=user_request,
        workspace_dir=workspace_dir,
        generate_summary=generate_summary,
        conversation_history=conversation_history,
        api_key=api_key,
        model_name=model_name,
        status_callback=status_callback
    )


if __name__ == "__main__":
    test_csv = "./test_data/Titanic-Dataset.csv"
    if os.path.exists(test_csv):
        run_tool_based_eda(
            data_path=test_csv,
            user_request="Perform full exploratory data analysis, type-safe imputation, outlier profiling, statistical hypothesis testing, and predictive blueprinting on Titanic-Dataset.csv."
        )
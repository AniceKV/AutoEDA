import subprocess
import sys
import os
import re
from typing import Dict, Any


class CodeExecutorSandbox:
    """
    Classful Code Execution Sandbox for running generated Python code in an isolated subprocess.
    """
    def __init__(self, workspace_dir: str = "./sandbox_run", timeout_seconds: int = 120):
        self.workspace_dir = workspace_dir
        self.timeout_seconds = timeout_seconds
        os.makedirs(self.workspace_dir, exist_ok=True)

    def clean_llm_code(self, raw_code: str) -> str:
        """Cleans markdown-wrapped python blocks from the LLM output."""
        pattern = r"```python(.*?)```"
        match = re.search(pattern, raw_code, re.DOTALL)
        if match:
            return match.group(1).strip()
        return raw_code.strip()

    def execute_code(self, llm_code: str, data_filepath: str) -> Dict[str, Any]:
        """
        Writes the clean code to a script, executes it in a subprocess,
        and captures stdout, stderr, and any exit code errors.
        """
        cleaned_code = self.clean_llm_code(llm_code)
        script_path = os.path.join(self.workspace_dir, "generated_analysis.py")

        abs_data_filepath = os.path.abspath(data_filepath)
        execution_code = f"DATA_FILEPATH = r'{abs_data_filepath}'\n\n" + cleaned_code

        with open(script_path, "w", encoding="utf-8") as f:
            f.write(execution_code)

        try:
            abs_script_path = os.path.abspath(script_path)
            result = subprocess.run(
                [sys.executable, abs_script_path],
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds,
                cwd=self.workspace_dir
            )

            success = result.returncode == 0
            return {
                "success": success,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "exit_code": result.returncode,
                "code_executed": execution_code
            }

        except subprocess.TimeoutExpired as e:
            return {
                "success": False,
                "stdout": e.stdout if e.stdout else "",
                "stderr": f"Execution Timed Out after {self.timeout_seconds} seconds. Standard Error: {e.stderr if e.stderr else ''}",
                "exit_code": -1,
                "code_executed": execution_code
            }


default_executor = CodeExecutorSandbox()


def execute_code(llm_code: str, data_filepath: str, workspace_dir: str = "./sandbox_run") -> Dict[str, Any]:
    executor = CodeExecutorSandbox(workspace_dir=workspace_dir)
    return executor.execute_code(llm_code, data_filepath)

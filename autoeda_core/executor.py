import subprocess
import sys
import os
import re

class CodeExecutorSandbox:
    def __init__(self, workspace_dir="./sandbox_run"):
        self.workspace_dir = workspace_dir
        os.makedirs(self.workspace_dir, exist_ok=True)
        
    def clean_llm_code(self, raw_code: str) -> str:
        """Cleans markdown-wrapped python blocks from the LLM output."""
        pattern = r"```python(.*?)```" #this regex extracts all python code from a given text
        match = re.search(pattern, raw_code, re.DOTALL)
        if match:
            return match.group(1).strip()
        return raw_code.strip()

    def execute_code(self, llm_code: str, data_filepath: str) -> dict:
        """
        Writes the clean code to a script, executes it in a subprocess,
        and captures stdout, stderr, and any exit code errors.
        """
        cleaned_code = self.clean_llm_code(llm_code)
        script_path = os.path.join(self.workspace_dir, "generated_analysis.py")
        
        #  CONVERT TO ABSOLUTE PATH FIRST
        # This prevents the sandbox's 'cwd' folder from breaking relative paths
        abs_data_filepath = os.path.abspath(data_filepath)
        
        # Inject the absolute path variable so the generated script knows what to read
        execution_code = f"DATA_FILEPATH = r'{abs_data_filepath}'\n\n" + cleaned_code
        
        with open(script_path, "w", encoding="utf-8") as f:
            f.write(execution_code)
            
        try:
            # Convert script_path to an absolute path as well
            abs_script_path = os.path.abspath(script_path)
            
            # Run the script using the current Python environment's interpreter
            result = subprocess.run(
                [sys.executable, abs_script_path],
                capture_output=True,
                text=True,
                timeout=120, # Limit execution time to 2 minutes
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
                "stderr": f"Execution Timed Out after 120 seconds. Standard Error: {e.stderr if e.stderr else ''}",
                "exit_code": -1,
                "code_executed": execution_code
            }

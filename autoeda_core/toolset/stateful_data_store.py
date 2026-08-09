import os
import json
import re
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.style as mplstyle
mplstyle.use("fast")  # Apply performance-oriented fast style sheet globally
import matplotlib.pyplot as plt
import seaborn as sns
import PIL.Image
PIL.Image.MAX_IMAGE_PIXELS = None  # Disable DecompressionBombWarning for large EDA visual plots
from typing import Dict, Any, List, Optional, Tuple
from pydantic import BaseModel, Field

from ..profiler import is_non_distributional_column
from ..llm_config import get_api_key, get_model, get_base_url

sns.set_theme(style="whitegrid")


from .utils import _sanitize_col_name, _safe_float, _is_numeric_col


class StatefulDataStore:
    """
    Manages stateful execution memory and DataFrame version control (DVC pattern).
    Maintains checkpoints in memory using df.copy() and copy.deepcopy(agent_state).
    Allows automatic rollback if a tool step corrupts or invalidates the dataset or metadata.
    """
    def __init__(self, workspace_dir: str = "./sandbox_run"):
        self.workspace_dir = workspace_dir
        self.version = 0
        self.history: List[Dict[str, Any]] = []
        os.makedirs(self.workspace_dir, exist_ok=True)

    def _make_entry(self, version: int, df: pd.DataFrame, agent_state: dict, action: str) -> dict:
        import copy
        return {
            "version": version,
            "df": df.copy(),
            "agent_state": copy.deepcopy(agent_state),
            "rows": len(df),
            "cols": len(df.columns),
            "action": action
        }

    def set_initial_state(self, df: pd.DataFrame, agent_state: dict) -> str:
        self.version = 0
        self.history = [self._make_entry(0, df, agent_state, "initial_load")]
        print(f"[DataStore] Initialized state v0 ({len(df)} rows, {len(df.columns)} cols) in memory.")
        return "memory:v0"

    def save_checkpoint(self, df: pd.DataFrame, agent_state: dict, step_name: str) -> str:
        if df is None or len(df) == 0 or len(df.columns) == 0:
            raise ValueError(f"Cannot checkpoint invalid or empty DataFrame after step '{step_name}'.")
        self.version += 1
        self.history.append(self._make_entry(self.version, df, agent_state, step_name))
        print(f"[DataStore] Saved checkpoint v{self.version} after '{step_name}' ({len(df)} rows, {len(df.columns)} cols) in memory.")
        return f"memory:v{self.version}"

    def rollback(self) -> Tuple[pd.DataFrame, dict, int]:
        import copy
        if len(self.history) <= 1:
            print("[DataStore] Cannot rollback further. At initial state v0.")
            latest_state = self.history[0]
            return latest_state["df"].copy(), copy.deepcopy(latest_state["agent_state"]), 0

        bad_state = self.history.pop()
        print(f"[DataStore] Rolling back from corrupted state v{bad_state['version']} ({bad_state['action']})...")

        latest_state = self.history[-1]
        self.version = latest_state["version"]
        restored_df = latest_state["df"].copy()
        restored_agent_state = copy.deepcopy(latest_state["agent_state"])
        print(f"[DataStore] Successfully rolled back to state v{self.version} ({latest_state['action']})")
        return restored_df, restored_agent_state, self.version

    def purge_intermediate_states(self):
        """
        Deletes intermediate checkpoint data frames to save memory,
        keeping only the initial load (v0) and the final active version.
        """
        if len(self.history) <= 2:
            return

        final_state = self.history[-1]
        self.history = [self.history[0], final_state]
        print(f"[DataStore] Cleaned up intermediate states. Retained initial state (v0) and final state (v{self.version}).")


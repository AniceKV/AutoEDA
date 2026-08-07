"""
AutoEDA Core Engine Package
"""

from .agent_loop import run_tool_based_eda
from .tools import StatefulDataStore
from .profiler import calculate_column_stats, run_and_save_profile, is_non_distributional_column
from .summary_generator import create_summary, extract_dataset_name
from .html_report_generator import generate_html_report

__all__ = [
    "run_tool_based_eda",
    "StatefulDataStore",
    "calculate_column_stats",
    "run_and_save_profile",
    "is_non_distributional_column",
    "create_summary",
    "extract_dataset_name",
    "generate_html_report",
]


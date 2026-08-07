"""
AutoEDA Core Engine Package
"""

from .agent_loop import run_tool_based_eda
from .tools import StatefulDataStore
from .profiler import calculate_column_stats, run_and_save_profile
from .summary_generator import create_summary, extract_dataset_name
from .html_report_generator import generate_html_report
from .parallel_plotter import batch_render_pipeline_plots

__all__ = [
    "run_tool_based_eda",
    "StatefulDataStore",
    "calculate_column_stats",
    "run_and_save_profile",
    "create_summary",
    "extract_dataset_name",
    "generate_html_report",
    "batch_render_pipeline_plots",
]

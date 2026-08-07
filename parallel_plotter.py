import os
import logging
import concurrent.futures
from typing import List, Dict, Any, Tuple, Callable
import numpy as np
import pandas as pd
import warnings

# Suppress deprecation and user warnings in background processes for clean stdout
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Setup non-interactive background rendering globally
import matplotlib
matplotlib.use("Agg")
import matplotlib.style as mplstyle
mplstyle.use("fast")  # Apply Matplotlib's performance optimizations globally

from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
import seaborn as sns

logger = logging.getLogger("autoeda.parallel_plotter")

# =============================================================================
# WORKER PLOTTING FUNCTIONS (Object-Oriented API, Thread-Safe)
# =============================================================================

def render_correlation_matrix(df_numerical: pd.DataFrame, save_path: str, title: str = "Correlation Matrix Heatmap") -> str:
    """
    Renders a high-fidelity correlation heatmap.
    Uses the pure OO API of Matplotlib to ensure complete thread-safety.
    """
    if df_numerical.shape[1] < 2:
        logger.warning("Fewer than 2 numerical columns. Skipping correlation matrix.")
        return save_path

    # Compute correlation
    corr = df_numerical.corr()

    # Create figure and axes explicitly (Thread-safe)
    fig = Figure(figsize=(10, 8), dpi=150)
    canvas = FigureCanvasAgg(fig)
    ax = fig.add_subplot(111)

    # Use Seaborn's heatmap plotting on the explicit axes
    sns.heatmap(
        corr,
        annot=df_numerical.shape[1] <= 15,  # Avoid cluttering for huge column counts
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        center=0,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8},
        ax=ax
    )

    ax.set_title(title, fontsize=14, pad=15, fontweight="bold")
    fig.tight_layout()

    # Save to disk using FigureCanvasAgg directly
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    canvas.print_png(save_path)
    
    # Explicit memory cleanup
    fig.clear()
    return save_path


def render_distribution_plot(series_data: np.ndarray, col_name: str, save_path: str, is_categorical: bool = False) -> str:
    """
    Renders a single feature distribution plot.
    """
    fig = Figure(figsize=(7, 5), dpi=150)
    canvas = FigureCanvasAgg(fig)
    ax = fig.add_subplot(111)

    if is_categorical:
        # Count plot for categorical data
        unique, counts = np.unique(series_data, return_counts=True)
        # Sort by frequency
        sorted_indices = np.argsort(-counts)
        unique = unique[sorted_indices]
        counts = counts[sorted_indices]
        
        # Clip categories to top 15 to avoid massive overlapping plots
        if len(unique) > 15:
            unique = list(unique[:14]) + ["Other"]
            counts = list(counts[:14]) + [sum(counts[14:])]

        ax.bar(unique, counts, color="#6366f1", edgecolor="black", alpha=0.85)
        ax.set_xticks(range(len(unique)))
        ax.set_xticklabels(unique, rotation=45, ha="right")
        ax.set_ylabel("Count")
    else:
        # Histogram with KDE for continuous data
        sns.histplot(series_data, kde=True, color="#6366f1", ax=ax, edgecolor="black", alpha=0.7)
        ax.set_ylabel("Density/Count")

    ax.set_title(f"Distribution of {col_name}", fontsize=12, pad=10, fontweight="bold")
    ax.set_xlabel(col_name)
    fig.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    canvas.print_png(save_path)
    
    fig.clear()
    return save_path


def render_bivariate_plot(
    x_data: np.ndarray, 
    y_data: np.ndarray, 
    x_name: str, 
    y_name: str, 
    save_path: str,
    x_is_num: bool,
    y_is_num: bool
) -> str:
    """
    Renders bivariate interactions between two features based on their variable types.
    """
    fig = Figure(figsize=(8, 6), dpi=150)
    canvas = FigureCanvasAgg(fig)
    ax = fig.add_subplot(111)

    if x_is_num and y_is_num:
        # Numerical vs Numerical -> Scatter/Regression
        sns.scatterplot(x=x_data, y=y_data, color="#6366f1", ax=ax, alpha=0.6, edgecolor="white")
        # Add a trend line if dataset size is reasonable
        if len(x_data) < 50000:
            sns.regplot(x=x_data, y=y_data, scatter=False, ax=ax, color="#ef4444", line_kws={"linewidth": 2})
    elif x_is_num or y_is_num:
        # Numerical vs Categorical -> Boxplot
        num_data = x_data if x_is_num else y_data
        cat_data = y_data if x_is_num else x_data
        num_name = x_name if x_is_num else y_name
        cat_name = y_name if x_is_num else x_name

        # Enforce ordering and assign hue to avoid deprecation warnings
        sns.boxplot(x=cat_data, y=num_data, ax=ax, hue=cat_data, legend=False, palette="Purples")
        ax.tick_params(axis='x', labelrotation=45)
    else:
        # Categorical vs Categorical -> Grouped Countplot
        # Handled via raw pandas pivot bar plots
        unique_x, x_counts = np.unique(x_data, return_inverse=True)
        unique_y, y_counts = np.unique(y_data, return_inverse=True)
        
        # Build a cross-tab count
        cross_tab = pd.crosstab(x_data, y_data)
        cross_tab.plot(kind="bar", stacked=True, ax=ax, colormap="tab20")
        ax.tick_params(axis='x', labelrotation=45)

    ax.set_title(f"{y_name} vs {x_name}", fontsize=12, pad=12, fontweight="bold")
    ax.set_xlabel(x_name)
    ax.set_ylabel(y_name)
    fig.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    canvas.print_png(save_path)
    
    fig.clear()
    return save_path


# =============================================================================
# MAIN MULTI-PROCESSED BATCH DISPATCHER
# =============================================================================

def batch_render_pipeline_plots(
    df: pd.DataFrame,
    target_column: str,
    output_directory: str,
    max_workers: int = None
) -> Dict[str, Any]:
    """
    Executes all plotting operations concurrently across a ProcessPoolExecutor.
    Decoupled directly from numerical computation for maximum Gunicorn & Django performance.
    
    Parameters:
    - df: The fully processed, clean, stateful DataFrame.
    - target_column: The target feature column.
    - output_directory: Filepath target to write compiled PNGs.
    - max_workers: Number of processes. If None, default to OS CPU count.
    
    Returns:
    - A dictionary compiling the successfully generated plots and any failed tasks.
    """
    os.makedirs(output_directory, exist_ok=True)
    
    # 1. Inspect data and dynamically extract types
    num_cols = list(df.select_dtypes(include=[np.number]).columns)
    cat_cols = list(df.select_dtypes(exclude=[np.number]).columns)
    all_cols = num_cols + cat_cols

    # Downsample huge datasets to prevent memory saturation inside the subprocesses
    max_plot_rows = 100000
    if len(df) > max_plot_rows:
        logger.info(f"Downsampling dataset from {len(df)} to {max_plot_rows} rows for plotting optimization.")
        df_plot = df.sample(n=max_plot_rows, random_state=42)
    else:
        df_plot = df

    tasks = []

    # Task A: Numerical Correlation Heatmap
    if len(num_cols) >= 2:
        df_num = df_plot[num_cols].dropna()
        corr_path = os.path.join(output_directory, "correlation_heatmap.png")
        tasks.append((
            render_correlation_matrix, 
            (df_num, corr_path), 
            "correlation_heatmap"
        ))

    # Task B: Feature Distributions (Uni-variate)
    for col in all_cols:
        col_data = df_plot[col].dropna().values
        if len(col_data) == 0:
            continue
        
        is_categorical = col in cat_cols
        # Avoid plotting high-cardinality nominals (like unique IDs or hashes)
        if is_categorical and len(np.unique(col_data)) > 50:
            logger.info(f"Skipping distribution plot for high-cardinality feature: {col}")
            continue
            
        dist_path = os.path.join(output_directory, f"dist_{col}.png")
        tasks.append((
            render_distribution_plot,
            (col_data, col, dist_path, is_categorical),
            f"dist_{col}"
        ))

    # Task C: Bivariate Target Interactions (Target vs Other Features)
    if target_column in df_plot.columns:
        target_is_num = target_column in num_cols
        for col in all_cols:
            if col == target_column:
                continue
                
            # Filter and drop NaNs in tandem for pairwise consistency
            pairwise_df = df_plot[[col, target_column]].dropna()
            if len(pairwise_df) == 0:
                continue
                
            col_is_num = col in num_cols
            x_data = pairwise_df[col].values
            y_data = pairwise_df[target_column].values
            
            # Limit categorical features to avoid visual noise
            if not col_is_num and len(np.unique(x_data)) > 20:
                continue
            if not target_is_num and len(np.unique(y_data)) > 20:
                continue

            biv_path = os.path.join(output_directory, f"bivariate_{col}_vs_{target_column}.png")
            tasks.append((
                render_bivariate_plot,
                (x_data, y_data, col, target_column, biv_path, col_is_num, target_is_num),
                f"bivariate_{col}"
            ))

    # 2. Concurrently Dispatch Plotting Workers
    results = {"success": [], "failed": {}}
    
    # ProcessPoolExecutor naturally isolates variables and resets Matplotlib's memory space per worker
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit tasks as process futures
        future_to_name = {}
        for func, args, name in tasks:
            future = executor.submit(func, *args)
            future_to_name[future] = name

        # Monitor progress and handle individual process exceptions gracefully
        for future in concurrent.futures.as_completed(future_to_name):
            name = future_to_name[future]
            try:
                path = future.result()
                results["success"].append({"plot_name": name, "file_path": path})
            except Exception as e:
                logger.error(f"Worker Exception rendering plot {name}: {str(e)}")
                results["failed"][name] = str(e)

    logger.info(f"Batch plotting complete. Success: {len(results['success'])}, Failed: {len(results['failed'])}")
    return results

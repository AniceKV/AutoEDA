# AutoEDA Pro - Autonomous Data Science Agent

AutoEDA Pro is a state-of-the-art, autonomous, tool-based data science platform. It leverages a stateful agent pipeline to perform complete exploratory data analysis (EDA), type-safe missing value imputation, outlier profiling, statistical hypothesis testing, semantic bivariate graphing, and predictive blueprinting.

## Features

- **Autonomous Tool-Based Pipeline:** Execute a complete EDA workflow with a single click. The agent intelligently selects tools and approaches based on the provided dataset and prompt.
- **Algorithmic Profiling:** Generates comprehensive statistics and distributions for both numerical and categorical variables, including type-safe handling and missing value analysis.
- **Statistical Modeling Blueprint:** Automatically recommends machine learning algorithms, validation strategies, feature selection methods, and overfitting risk mitigation techniques based on the dataset characteristics.
- **Hypothesis Testing:** Identifies statistically significant predictors automatically.
- **Visual Gallery:** Generates and organizes visual assets including distribution plots, semantic bivariate interactions, pairplots, and correlation heatmaps.
- **Executive Summary Generation:** Produces a detailed markdown report summarizing the findings of the exploratory analysis.

## Application Architecture

The platform is structured into several core components:
- **Streamlit Interface (`app.py`):** Provides a modern, responsive, and intuitive web UI for dataset upload, agent configuration, and visualization of results.
- **Agent Loop (`agent_loop.py`):** Orchestrates the execution of the autonomous data science agent and maintains state throughout the analysis process.
- **Tools System (`tools.py`):** Contains the modular functions utilized by the agent to manipulate data, generate plots, and compute statistics.
- **Profiler (`profiler.py`):** Handles deep statistical calculations and algorithmic dataset profiling.
- **Summary Generator (`summary_generator.py`):** Compiles the analysis outputs into a comprehensive executive summary report.
- **Executor (`executor.py`):** Manages the safe and isolated execution of agent-directed code or tasks.

## Installation

1. Clone the repository to your local machine.
2. Create and activate a Python virtual environment.
3. Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

Start the Streamlit application by running:

```bash
streamlit run app.py
```

### Navigating the Interface

1. **Data Input:** Use the sidebar to either select a sample dataset from the `./test_data/` directory or upload a custom CSV file.
2. **Analysis Request:** Provide an optional instruction prompt to guide the agent's focus during the analysis.
3. **Execution:** Click "Run AutoEDA Pipeline" to begin the autonomous analysis.
4. **Results Exploration:** Once complete, use the primary tabs to explore Variable Profiling, Visual Gallery, Executive Summary, and Metrics & Blueprint.

## Directory Structure

- `/EDA`: Contains the generated outputs (images, reports, metrics) categorized by dataset.
- `/sandbox_run`: Temporary directory used during active agent execution.
- `/temp_uploads`: Stores user-uploaded custom CSV files.
- `/test_data`: Contains sample datasets for quick testing and demonstration.
- `.venv`: Python virtual environment.

## Requirements

The project relies on standard data science and web application libraries, including:
- pandas
- numpy
- streamlit

For a complete list of dependencies and their exact versions, please refer to the `requirements.txt` file.

## License

This project is licensed under the MIT License.

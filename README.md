# AutoEDA Pro - Autonomous Data Science & EDA Agent

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30%2B-FF4B4B.svg)](https://streamlit.io/)
[![Plotly](https://img.shields.io/badge/Plotly-Interactive%20Viz-3F4F75.svg)](https://plotly.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**AutoEDA Pro** is a state-of-the-art, autonomous, tool-based data science platform. It leverages a stateful agent execution loop with Data Version Control (DVC pattern), type-safe missing value imputation, outlier profiling, statistical hypothesis testing, semantic bivariate graphing, predictive blueprinting, and interactive single-page HTML report generation.

---

## Key Features

- **Autonomous Agent Pipeline:** Executes a multi-step exploratory data analysis (EDA) pipeline with a single click. The agent autonomously selects tools, plans analysis steps, handles user prompts, and pauses for clarification when target requirements are ambiguous.
- **Stateful Data Store & DVC Pattern:** Tracks DataFrame checkpoints in memory (`v0`, `v1`, ...). If an analysis step or transformation invalidates or corrupts the dataset, AutoEDA Pro automatically rolls back to the last valid checkpoint and purges intermediate states.
- **Defensive Parameter Clamping:** Guarantees runtime stability by clamping parameters programmatically (e.g. automatically converting mean/median imputation requests to mode/constant for non-numeric columns, clamping IQR bounds, downsampling large datasets for fast visualization rendering).
- **Algorithmic Dataset Profiling:** Generates comprehensive statistical profiling for continuous, discrete, boolean, and categorical variables with caching for instant re-runs.
- **Statistical Significance Testing:** Automatically dispatches Pearson Correlation, Chi-Square Independence, Welch T-Test, and One-Way ANOVA tests to identify statistically significant predictors against target variables.
- **Visual Gallery Generation:** Renders styled correlation heatmaps, Cramér's V categorical association matrices, probability distribution plots (KDE & count plots), semantic bivariate interactions, and Seaborn pairplots.
- **Interactive Single-Page HTML Profile Report:** Produces a standalone, dark-themed HTML report powered by Plotly JS with responsive tabs, real-time dataset alerts, embedded visual gallery, and downloadable offline reports.
- **Executive Summary Synthesis:** Compiles all findings into a structured Markdown executive summary report, with optional OpenRouter / LLM synthesis for narrative insights.
- **Predictive Machine Learning Blueprint:** Automatically formulates model recommendations (Regression vs Classification vs Clustering), cross-validation strategies, feature selection criteria, and overfitting risk mitigation.

---

## Comprehensive Statistics Generated

AutoEDA Pro computes granular statistical metrics across the entire dataset:

### 1. Dataset-Level Summary Stats
| Metric | Description |
| :--- | :--- |
| **Dimensions** | Total row count ($N$) and column count ($P$) |
| **Variable Breakdown** | Total numerical vs. categorical vs. boolean features |
| **Cell Integrity** | Total missing cells and overall dataset missing percentage |
| **Automated Alerts** | Flags high missingness (> 20%), zero variance (constant columns), and high collinearity ($\|r\| \ge 0.85$) |

### 2. Numerical Variable Statistics
For continuous and discrete numeric columns:
| Metric | Description |
| :--- | :--- |
| **Central Tendency** | Mean, Median (50th percentile) |
| **Dispersion & Range** | Minimum, Maximum, Standard Deviation ($\sigma$) |
| **Quartiles & IQR** | First Quartile ($Q_1$, 25%), Third Quartile ($Q_3$, 75%), Interquartile Range ($IQR = Q_3 - Q_1$) |
| **Distribution Shape** | Skewness (flags highly skewed features where $\|skew\| > 1.0$), Kurtosis |
| **Completeness & Cardinality** | Missing count & percentage, Cardinality (distinct count), Unique percentage |

### 3. Categorical & Discrete Variable Statistics
For object, string, boolean, and low-cardinality discrete columns:
| Metric | Description |
| :--- | :--- |
| **Cardinality** | Number of unique levels/categories |
| **Frequency Table** | Top 5 most frequent values with exact counts and dataset percentages |
| **Mode & Fallbacks** | Most frequent class token and missing value count & percentage |
| **Categorical Association** | Cramér's V association coefficients for non-numeric variable pairs |

### 4. Outlier Profiling Statistics
Calculated using the Interquartile Range (IQR) method (default $1.5 \times IQR$):
| Metric | Description |
| :--- | :--- |
| **IQR Bounds** | Lower Bound ($Q_1 - 1.5 \times IQR$), Upper Bound ($Q_3 + 1.5 \times IQR$) |
| **Outlier Counts** | Number of rows falling outside the lower/upper bounds |
| **Outlier Percentage** | Percentage of extreme values in the column |
| **Action Modes** | `profile` (statistical audit) or `cap` (clipping values at bounds) |

### 5. Statistical Significance & Hypothesis Testing
Automated hypothesis testing against the target column ($\alpha = 0.05$):
| Combination | Statistical Test Applied | Outputs |
| :--- | :--- | :--- |
| **Numeric Target + Numeric Feature** | Pearson Correlation Test | Correlation coefficient ($r$), $p$-value, Significance flag |
| **Categorical Target + Categorical Feature** | Chi-Square Test of Independence | Chi-square statistic ($\chi^2$), Degrees of Freedom ($dof$), $p$-value |
| **Binary Target + Numeric Feature** | Two-Sample Welch T-Test | $t$-statistic, $p$-value, Significance flag |
| **Multiclass Target + Numeric Feature** | One-Way ANOVA | $F$-statistic, $p$-value, Significance flag |

---

## Tool Registry Catalog

The agent operates by calling modular functions registered in `tools.py`:

1. `impute_missing_data`: Type-safe missing value imputation (mean for symmetric numeric, median for skewed numeric, mode for categorical).
2. `detect_and_handle_outliers`: Detects outliers via IQR and optionally caps extreme values.
3. `engineer_features`: Creates high-signal features (`log1p` for skewed features, multiplicative interaction products, ratios with $\epsilon$, sum aggregations).
4. `run_statistical_hypothesis_tests`: Runs automated statistical significance tests against target variable.
5. `plot_correlation_matrix`: Renders Pearson numeric correlation heatmap and Cramér's V categorical association matrix PNGs.
6. `plot_feature_distributions`: Renders KDE histograms for continuous features and countplots for categorical features.
7. `plot_semantic_bivariate_relationships`: Renders custom X vs Y scatter/boxplot/countplot relationships dynamically selected by LLM domain reasoning.
8. `plot_target_interaction`: Generates segmented feature vs. target interaction visual assets.
9. `plot_pairplot`: Generates a concise Seaborn pairplot across key numerical attributes (clamped to max 4-5 features).
10. `generate_predictive_blueprint`: Compiles machine learning model recommendations, cross-validation strategies, and risk mitigations.
11. `ask_clarifying_question`: Pauses pipeline execution to request input from the user when requirements are ambiguous.
12. `finish_analysis`: Concludes agent loop execution once exploratory analysis is complete.

---

## Architecture & Project Structure

```
AutoEDA/
├── app.py                      # Modern Cyber-Minimalist Streamlit Web UI
├── agent_loop.py               # Autonomous LLM agent execution loop & state machine
├── profiler.py                 # Algorithmic dataset statistical profiler & json caching
├── html_report_generator.py    # Plotly JS interactive dark-mode HTML report generator
├── summary_generator.py        # Executive summary Markdown report synthesizer
├── tools.py                    # Data science tools registry, DVC DataStore, & parameter clamping
├── executor.py                 # Safe task execution interface
├── requirements.txt            # Python dependencies
├── test_data/                  # Sample CSV datasets for testing (e.g. housing, iris, titanic)
├── EDA/                        # Generated output artifacts grouped by dataset
│   └── <dataset_name>/
│       ├── metadata_profile.json
│       ├── metrics.json
│       ├── eda_report.html
│       ├── executive_summary.md
│       ├── correlation_matrix.png
│       └── dist_*.png
└── sandbox_run/                # Active agent working directory
```

---

## Installation & Setup

### Prerequisites

- Python 3.9 or higher
- `pip` package manager

### Steps

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/AniceKV/AutoEDA.git
   cd AutoEDA
   ```

2. **Create & Activate Virtual Environment:**
   - **On Windows (PowerShell):**
     ```powershell
     python -m venv .venv
     .\.venv\Scripts\Activate.ps1
     ```
   - **On macOS/Linux:**
     ```bash
     python3 -m venv .venv
     source .venv/bin/activate
     ```

3. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. *(Optional)* **Set up API Keys:**
   Create a `.env` file in the project root if you want to enable OpenRouter / OpenAI API synthesis for executive summaries:
   ```env
   OPENROUTER_API_KEY=your_openrouter_api_key_here
   ```

---

## How to Use

### 1. Launch the Web Application

Run the Streamlit app:
```bash
streamlit run app.py
```
The application will open in your default browser at `http://localhost:8501`.

### 2. Navigating the Streamlit Interface

1. **Select Data Source (Sidebar):**
   - **Sample Datasets:** Pick any pre-loaded CSV file from `./test_data/` (e.g., California Housing, Titanic).
   - **Upload Custom CSV:** Drag and drop your own dataset CSV file.
2. **Set Prompt / Task Instruction:**
   - Customize the prompt to focus the agent (e.g., *"Focus on predicting House Value and analyze outliers"*), or leave the default instruction.
3. **Execute Pipeline:**
   - Click **"Execute Analysis Pipeline"**. The agent will profile the dataset, run hypothesis tests, generate charts, compile `metrics.json`, build the interactive HTML report, and draft the executive summary.
4. **Explore the Results (Tabs):**
   - **Interactive Report:** Full offline single-page Plotly HTML report preview with dark glassmorphism aesthetic and a one-click download button.
   - **Variable Profiles:** Filterable deep-dive expander views for each column showing complete statistics tables and distribution plots.
   - **Visual Gallery:** Responsive image grid displaying all generated plot assets (heatmaps, pairplots, bivariate charts).
   - **Executive Summary:** Rendered Markdown report summarizing key findings, statistical significance, and modeling strategy.
   - **Metrics & Strategy Blueprint:** Machine learning blueprint detailing recommended models, cross-validation schemes, and risk mitigation strategies.

---

## Generated Output Artifacts

Upon running the AutoEDA pipeline for a dataset (e.g. `housing.csv`), the generated assets are saved to `EDA/<dataset_name>/` and `./sandbox_run/`:

- **`eda_report.html`**: Interactive, standalone Plotly HTML report file ready for sharing with stakeholders.
- **`executive_summary.md`**: Executive summary document highlighting key insights and statistical test results.
- **`metadata_profile.json`**: Complete structured column-by-column raw profiling statistics.
- **`metrics.json`**: Canonical metrics file consolidating dataset summary, imputation log, hypothesis tests, and ML blueprint.
- **`*.png`**: High-resolution visualization image files (`correlation_matrix.png`, `pairplot.png`, `dist_<col>.png`, `bivariate_*.png`).

---

## Technology Stack

- **Frontend & Web UI:** [Streamlit](https://streamlit.io/)
- **Data Wrangling:** [Pandas](https://pandas.pydata.org/), [NumPy](https://numpy.org/)
- **Statistical Computing:** [SciPy](https://scipy.org/)
- **Visualization:** [Plotly](https://plotly.com/), [Seaborn](https://seaborn.pydata.org/), [Matplotlib](https://matplotlib.org/)
- **Report Templating:** [Jinja2](https://jinja.org/)
- **Schema Validation:** [Pydantic](https://docs.pydantic.dev/)

---

## License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

# AutoEDA — Autonomous Exploratory Data Analysis & Data Science Platform

[![Live Demo](https://img.shields.io/badge/Live%20Demo-autoeda--fjgz.onrender.com-46E3B7?style=for-the-badge&logo=render&logoColor=white)](https://autoeda-fjgz.onrender.com/)
[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Django](https://img.shields.io/badge/Django-4.2%2B-092E20?style=for-the-badge&logo=django&logoColor=white)](https://www.djangoproject.com/)
[![Plotly](https://img.shields.io/badge/Plotly-Interactive%20Viz-3F4F75?style=for-the-badge&logo=plotly&logoColor=white)](https://plotly.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)

**Live Application:** [https://autoeda-fjgz.onrender.com/](https://autoeda-fjgz.onrender.com/)

---

## Overview

**AutoEDA** is an autonomous, tool-based exploratory data analysis and statistical modeling platform. Powered by an LLM-guided agent refinement loop, AutoEDA performs statistical dataset profiling, automated hypothesis testing with effect-size ranking, semantic bivariate graph generation, predictive machine learning blueprinting, and interactive standalone HTML report generation.

---

## System Capabilities

- **Autonomous Agent Execution Loop:** Multi-step reasoning pipeline that profiles datasets, dispatches domain tools, conducts hypothesis tests, and handles ambiguous user instructions.
- **Executive Summary Integration:** Synthesizes an executive summary (`summary_report.md`) prior to report generation and embeds it directly as the primary tab within the interactive HTML report (`eda_report.html`).
- **Non-Distributional Feature Filtering:** Automatically detects and excludes sequential identifiers (IDs, UUIDs), spatial coordinates (latitude/longitude), and timestamps from univariate distribution plotting to reduce execution overhead.
- **Statistical Hypothesis Testing with Effect Size Ranking:** Dispatches Pearson Correlation, Chi-Square Independence, Welch T-Test, and One-Way ANOVA tests at $\alpha = 0.05$, ranking significant predictors by effect size (Cohen's $d$, Eta-squared $\eta^2$, Cramér's V, $|r|$).
- **Dynamic Semantic Feature Synthesis:** Generates domain-driven feature names (e.g., `total_math_score_reading_score_writing_score`, `fare_per_class`, `log_income`) with fuzzy column matching and flexible operation aliases (`sum`, `ratio`, `product`, `diff`, `mean`).
- **Invertible Light/Dark UI Theme:** White-to-charcoal monochrome design system with a live toggle switch that inverts page UI elements and embedded Plotly visual assets.
- **Stateful DataStore & Version Control:** In-memory dataset version control (`v0`, `v1`, ...) with automatic state checkpointing and rollback capabilities.

---

## Statistical Profile & Metrics Summary

AutoEDA computes comprehensive statistical profiles across tabular datasets:

### 1. Dataset Integrity & Overview
| Metric | Description |
| :--- | :--- |
| **Dimensions** | Observation count ($N$) and feature count ($P$) |
| **Variable Types** | Numeric, Categorical, Boolean, Datetime classification |
| **Cell Completeness** | Missing value counts and total missing cell percentage |
| **Quality Alerts** | Automated warnings for missingness (> 20%), zero variance, and collinearity ($\|r\| \ge 0.85$) |

### 2. Numerical Feature Profile
| Metric | Description |
| :--- | :--- |
| **Central Tendency** | Mean, Median (50th percentile) |
| **Dispersion & Range** | Minimum, Maximum, Standard Deviation ($\sigma$) |
| **Quartiles & IQR** | $Q_1$ (25%), $Q_3$ (75%), Interquartile Range ($IQR = Q_3 - Q_1$) |
| **Distribution Shape** | Skewness (flags $\|skew\| \ge 1.0$), Kurtosis |
| **Cardinality** | Missing count, Uniqueness percentage, Distinct value count |

### 3. Categorical & Discrete Feature Profile
| Metric | Description |
| :--- | :--- |
| **Cardinality** | Distinct level count |
| **Frequency Table** | Top 5 most frequent levels with counts and percentages |
| **Categorical Associations** | Cramér's V association matrix for categorical feature pairs |

### 4. Hypothesis Testing & Effect Size Ranking ($\alpha = 0.05$)
| Feature Pair | Applied Test | Effect Size Metric |
| :--- | :--- | :--- |
| **Numeric Target + Numeric Feature** | Pearson Correlation Test | Absolute Pearson $\|r\|$ |
| **Categorical Target + Categorical Feature** | Chi-Square Independence Test | Cramér's V ($V$) |
| **Binary Target + Numeric Feature** | Two-Sample Welch T-Test | Cohen's $d$ |
| **Multiclass Target + Numeric Feature** | One-Way ANOVA | Eta-squared ($\eta^2$) |

---

## Core Tool Catalog

The autonomous agent selects and executes functions defined in `autoeda_core/tools.py`:

1. `impute_missing_data`: Imputes missing values using type-safe strategies (median for skewed numeric, mean for symmetric, mode for categorical).
2. `detect_and_handle_outliers`: Detects outliers via IQR bounds and optionally caps extreme values.
3. `engineer_features`: Creates domain feature transformations (`log1p`, ratios, product interactions, sums, differences, averages) with dynamic semantic naming.
4. `run_statistical_hypothesis_tests`: Dispatches hypothesis tests against target columns and ranks significant features by effect size.
5. `plot_correlation_matrix`: Renders Pearson correlation heatmaps and Cramér's V association matrices.
6. `plot_feature_distributions`: Renders univariate KDE histograms and count plots for non-identifier features.
7. `plot_semantic_bivariate_relationships`: Renders domain scatter plots, box plots, and bar charts based on semantic reasoning.
8. `plot_target_interaction`: Generates segmented feature vs. target interaction plots.
9. `plot_pairplot`: Renders Seaborn pairplots across primary numerical attributes.
10. `generate_predictive_blueprint`: Compiles machine learning model recommendations, cross-validation strategies, and feature selection plans.
11. `ask_clarifying_question`: Pauses execution to solicit user clarification when instructions or target columns are ambiguous.
12. `finish_analysis`: Finalizes pipeline execution once exploratory objectives are satisfied.

---

## Project Structure

```
AutoEDA/
├── autoeda_core/               # Core Analytical Engine & Agent Modules
│   ├── agent_loop.py           # Stateful LLM agent execution loop & orchestrator
│   ├── profiler.py             # Algorithmic dataset profiler & non-distributional filter
│   ├── html_report_generator.py# Plotly JS interactive HTML report & markdown renderer
│   ├── summary_generator.py    # Executive summary synthesizer
│   ├── tools.py                # Tool registry, DVC DataStore, & effect size calculators
│   └── executor.py             # Task execution interface
├── django_app/                 # Production Django Web Application
│   ├── autoeda/                # Django project configuration
│   ├── eda_app/                # Main application (views, templates, static assets)
│   │   ├── static/eda_app/
│   │   │   ├── css/main.css    # Invertible dark/light monochrome stylesheet
│   │   │   └── js/app.js       # Live pipeline polling & theme toggle script
│   │   └── templates/eda_app/
│   │       ├── base.html       # Top navigation and sidebar template
│   │       └── index.html      # Multi-tab dashboard & metric views
│   └── manage.py               # Django CLI management script
├── test_data/                  # Sample CSV datasets
├── requirements.txt            # Python dependencies
├── Dockerfile                  # Container deployment specification
└── EDA/                        # Generated output artifacts grouped by dataset name
    └── <dataset_name>/
        ├── metadata_profile.json
        ├── metrics.json
        ├── summary_report.md
        ├── eda_report.html
        └── *.png
```

---

## Local Setup & Development

### 1. Clone Repository
```bash
git clone https://github.com/AniceKV/AutoEDA.git
cd AutoEDA
```

### 2. Environment Setup
- **Windows (PowerShell):**
  ```powershell
  python -m venv .venv
  .\.venv\Scripts\Activate.ps1
  ```
- **macOS / Linux:**
  ```bash
  python3 -m venv .venv
  source .venv/bin/activate
  ```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Environment Variables
Create a `.env` file in the project root:
```env
OPENROUTER_API_KEY=your_openrouter_api_key_here
EDA_MODEL=google/gemini-2.5-flash
```

### 5. Run Web Server
```bash
python django_app/manage.py runserver 8000
```
Open `http://127.0.0.1:8000/` in your browser.

---

## Deployment

Build and run using Docker:

```bash
docker build -t autoeda .
docker run -p 8000:8000 --env-file .env autoeda
```

Production Deployment: [https://autoeda-fjgz.onrender.com/](https://autoeda-fjgz.onrender.com/)

---

## License

This project is licensed under the [MIT License](LICENSE).

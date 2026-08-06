# AutoEDA — Autonomous Exploratory Data Analysis & Data Science Platform

[![Live Demo](https://img.shields.io/badge/Live%20Demo-autoeda.up.railway.app-000000?style=for-the-badge&logo=railway&logoColor=white)](https://autoeda.up.railway.app/)
[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Django](https://img.shields.io/badge/Django-4.2%2B-092E20?style=for-the-badge&logo=django&logoColor=white)](https://www.djangoproject.com/)
[![Plotly](https://img.shields.io/badge/Plotly-Interactive%20Viz-3F4F75?style=for-the-badge&logo=plotly&logoColor=white)](https://plotly.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)

> 🚀 **Live Demo Application:** [https://autoeda.up.railway.app/](https://autoeda.up.railway.app/)

---

**AutoEDA** is a state-of-the-art, autonomous, tool-based data science platform. Powered by an LLM agent execution loop, AutoEDA performs statistical dataset profiling, automated hypothesis testing, dynamic bivariate visual gallery creation, predictive machine learning blueprinting, and interactive standalone HTML report generation.

---

## 🌟 Key Highlights

- 🌐 **Live Web Platform:** Hosted and production-ready at [https://autoeda.up.railway.app/](https://autoeda.up.railway.app/).
- 🤖 **Autonomous Agent Pipeline:** Multi-step reasoning loop that profiles datasets, executes domain tools, runs statistical tests, and pauses for clarification if user instructions are ambiguous.
- 🌓 **Invertible Light / Dark Minimalist UI:** Includes a persistent, white-to-charcoal monochrome design system with a live Light/Dark mode toggle switch that seamlessly inverts both page components and embedded Plotly charts.
- 🔄 **Stateful Data Store & DVC Pattern:** In-memory dataset version control (`v0`, `v1`, ...) with automatic rollback to the last valid checkpoint if a transformation corrupts data integrity.
- 📊 **Automated Hypothesis Testing:** Dispatches Pearson Correlation, Chi-Square Independence, Welch T-Test, and One-Way ANOVA tests at $\alpha = 0.05$ to discover statistically significant predictors.
- 📄 **Interactive Profile Report Generator:** Produces self-contained, Plotly-powered single-page HTML reports (`eda_report.html`) complete with responsive tabs, quality alerts, and offline download support.
- 🧠 **Predictive ML Blueprinting:** Formulates target-aware model recommendations (Classification, Regression, Clustering), cross-validation schemes, feature engineering specs, and overfitting mitigations.

---

## 📊 Comprehensive Statistical Metrics Computed

AutoEDA computes granular statistical metrics across tabular datasets:

### 1. Dataset Integrity & Overview
| Metric | Description |
| :--- | :--- |
| **Dimensions** | Total observation count ($N$) and feature count ($P$) |
| **Variable Breakdown** | Numeric vs. Categorical vs. Boolean vs. Datetime columns |
| **Cell Completeness** | Missing value count and total missing cell percentage |
| **Data Quality Alerts** | Automated warnings for high missingness (> 20%), zero variance (constant columns), and high collinearity ($\|r\| \ge 0.85$) |

### 2. Numerical Variable Statistics
For continuous and discrete numeric features:
| Metric | Description |
| :--- | :--- |
| **Central Tendency** | Mean, Median (50th percentile) |
| **Dispersion & Range** | Minimum, Maximum, Standard Deviation ($\sigma$) |
| **Quartiles & IQR** | $Q_1$ (25%), $Q_3$ (75%), Interquartile Range ($IQR = Q_3 - Q_1$) |
| **Shape Metrics** | Skewness (flags highly skewed distributions where $\|skew\| \ge 1.0$), Kurtosis |
| **Completeness & Uniqueness** | Missing count & percentage, Cardinality (distinct values), Unique percentage |

### 3. Categorical & Discrete Variable Statistics
For object, string, boolean, and low-cardinality features:
| Metric | Description |
| :--- | :--- |
| **Cardinality** | Total distinct level count |
| **Frequency Table** | Top 5 most frequent levels with exact counts and dataset percentages |
| **Categorical Associations** | Cramér's V association coefficients for categorical feature pairs |

### 4. Hypothesis Testing Matrix ($\alpha = 0.05$)
Automated statistical tests dispatched against target variables:
| Target & Feature Pair | Applied Test | Metrics Returned |
| :--- | :--- | :--- |
| **Numeric Target + Numeric Feature** | Pearson Correlation Test | Correlation coefficient ($r$), $p$-value, Significance flag |
| **Categorical Target + Categorical Feature** | Chi-Square Independence Test | Chi-square statistic ($\chi^2$), $dof$, $p$-value |
| **Binary Target + Numeric Feature** | Two-Sample Welch T-Test | $t$-statistic, $p$-value, Significance flag |
| **Multiclass Target + Numeric Feature** | One-Way ANOVA | $F$-statistic, $p$-value, Significance flag |

---

## 🛠️ Tool Registry Catalog

The autonomous agent executes tool functions defined in `tools.py`:

1. `impute_missing_data`: Type-safe missing value imputation (symmetric numeric $\rightarrow$ mean, skewed numeric $\rightarrow$ median, categorical $\rightarrow$ mode).
2. `detect_and_handle_outliers`: Detects outliers using IQR bounds and optionally caps extreme values.
3. `engineer_features`: Creates high-signal domain features (`log1p` for skewed features, interaction products, ratio metrics).
4. `run_statistical_hypothesis_tests`: Dispatches hypothesis tests against target columns.
5. `plot_correlation_matrix`: Renders Pearson correlation heatmaps and Cramér's V categorical association matrices.
6. `plot_feature_distributions`: Renders distribution plots (KDE histograms & countplots).
7. `plot_semantic_bivariate_relationships`: Renders domain scatter plots, box plots, and bar charts.
8. `plot_target_interaction`: Generates feature vs. target interaction plots.
9. `plot_pairplot`: Renders concise Seaborn pairplots across primary numerical attributes.
10. `generate_predictive_blueprint`: Compiles machine learning model strategies and validation plans.
11. `ask_clarifying_question`: Pauses execution to solicit user feedback when instructions are ambiguous.
12. `finish_analysis`: Finalizes pipeline execution once exploratory objectives are met.

---

## 🏗️ Project Architecture

```
AutoEDA/
├── django_app/                 # Full Django Production Web Application
│   ├── autoeda/                # Django project configuration (settings, URLs, WSGI)
│   ├── eda_app/                # Main application app (views, templates, static assets)
│   │   ├── static/eda_app/
│   │   │   ├── css/main.css    # Invertible dark/light minimalist theme stylesheet
│   │   │   └── js/app.js       # Live pipeline polling & theme toggle handler
│   │   └── templates/eda_app/
│   │       ├── base.html       # Top navigation, sidebar, and layout template
│   │       └── index.html      # Multi-tab dashboard & metric views
│   └── manage.py               # Django CLI management script
├── agent_loop.py               # Stateful LLM agent execution loop & orchestrator
├── profiler.py                 # Algorithmic statistical dataset profiler
├── html_report_generator.py    # Plotly JS interactive dark/light HTML report generator
├── summary_generator.py        # Markdown executive summary synthesizer
├── tools.py                    # Tool registry, DVC DataStore, & parameter clamping
├── executor.py                 # Task execution interface
├── requirements.txt            # Python dependencies
├── Dockerfile                  # Production container definition
├── test_data/                  # Sample CSV datasets (e.g. Titanic, Housing, Spotify)
└── EDA/                        # Generated output artifacts grouped by dataset
    └── <dataset_name>/
        ├── metadata_profile.json
        ├── metrics.json
        ├── eda_report.html
        └── *.png
```

---

## ⚙️ Local Installation & Development

### 1. Clone the Repository
```bash
git clone https://github.com/AniceKV/AutoEDA.git
cd AutoEDA
```

### 2. Set Up Virtual Environment
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

### 4. Set Up Environment Variables
Create a `.env` file in the project root:
```env
OPENROUTER_API_KEY=your_openrouter_api_key_here
EDA_MODEL=google/gemini-2.5-flash
```

### 5. Launch the Web Application
Run the Django development server:
```bash
python django_app/manage.py runserver 8000
```
Open `http://127.0.0.1:8000/` in your browser.

---

## 🚢 Deployment

The application is configured for one-click production deployment on **Railway**, **Render**, or **Docker**:

```bash
docker build -t autoeda .
docker run -p 8000:8000 --env-file .env autoeda
```

Live Application: [https://autoeda.up.railway.app/](https://autoeda.up.railway.app/)

---

## 📜 License

This project is open source and available under the [MIT License](LICENSE).

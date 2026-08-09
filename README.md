# AutoEDA — Stateful Agentic Exploratory Data Analysis & Predictive Modeling

[![Live Demo](https://img.shields.io/badge/Live%20Demo-autoeda--fjgz.onrender.com-46E3B7?style=for-the-badge&logo=render&logoColor=white)](https://autoeda-fjgz.onrender.com/)
[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Django](https://img.shields.io/badge/Django-4.2%2B-092E20?style=for-the-badge&logo=django&logoColor=white)](https://www.djangoproject.com/)
[![Plotly](https://img.shields.io/badge/Plotly-Interactive%20Viz-3F4F75?style=for-the-badge&logo=plotly&logoColor=white)](https://plotly.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)

**Live Application:** [AutoEDA](https://autoeda-fjgz.onrender.com/)

---

**Example Report:** [Titanic EDA Report](eda_report_Titanic-Dataset.html)

---

## Configuration

You can configure AutoEDA by setting these environment variables (or putting them in a `.env` file):

| Var | Required | Default | Purpose |
| :--- | :---: | :--- | :--- |
| **`OPENROUTER_API_KEY`** | Yes | -- | Your API key from [OpenRouter](https://openrouter.ai/keys) |
| **`EDA_MODEL`** | Yes | -- | Any valid model ID (use can choose free ones too) (e.g.,`meta-llama/llama-3.3-70b:free`, `anthropic/claude-3.5-sonnet`) |
| **`OPENROUTER_BASE_URL`** | No | `https://openrouter.ai/api/v1` | Override this to point to a local endpoint (e.g. `http://localhost:1234/v1` for LM Studio) or another OpenAI-compatible provider. |

**Examples:**
- **Default (Free tier):** Just set `OPENROUTER_API_KEY="sk-or-..."`. It will automatically use OpenRouter's live free models.
- **Specific Free Model:** Set `EDA_MODEL="meta-llama/llama-3.3-70b:free"`
- **Paid Premium Model:** Set `EDA_MODEL="anthropic/claude-3.5-sonnet"`
- **Custom Local Endpoint (LM Studio/Ollama):** Set `OPENROUTER_BASE_URL="http://localhost:1234/v1"` and `EDA_MODEL="local-model-name"`

## Usage as a Python Library (edanet)

AutoEDA's core engine is packaged as `edanet` for easy integration into your own scripts or notebooks.

```bash
pip install edanet
```

```python
from autoeda_core import AutoEDAEngine

engine = AutoEDAEngine()

results = engine.analyze(
    data_path="path/to/your/dataset.csv",
    user_request="Perform full exploratory data analysis",
)
```

AutoEDA is an low cost agentic pipeline designed to perform end-to-end Exploratory Data Analysis, statistical profiling, and predictive modeling on tabular datasets. Built around a strict **Planner-Executor Decoupling Architecture**, AutoEDA operates with complete safety and determinism. It allows compact local models (such as **Qwen 2.5/3.5**, **Gemma 2**, or **Llama 3**) to construct and execute complex mathematical workflows without the instability of arbitrary code execution.

By shifting from unstable, raw Python script generation to **Structured JSON Tool-Plans** mapped onto parameter-clamped statistical actions, AutoEDA eliminates common LLM failure points: syntax hallucinations, library crashes, and data corruption.

**Cost-Effective**: Thanks to an optimized prompt architecture and intelligent tool routing, the average token cosumption is around 3000 tokens per analysis

---

## Architectural Overview

The core system isolates the LLM's reasoning loop from the raw execution environment, using programmatic checkpoints and client-side charting to achieve production stability and near-zero server-side latency.

```text
                  ┌──────────────────────┐
                  │   User Uploads CSV   │
                  └──────────┬───────────┘
                             ▼
               ┌───────────────────────────┐
               │ 1. Algorithmic Profiler   │ (profiler.py)
               │    - Generates Metadata   │
               └──────────┬───────────┘
                          ▼
            ┌─────────────────────────────┐
            │   2. JSON Tool-Plan Query   │ (agent_loop.py)
            │      - Local or Cloud LLMs  │
            └─────────────┬───────────────┘
                          ▼
               ┌───────────────────────────┐
               │ 3. Stateful Executor      │ (executor.py & tools.py)
               │    - Sandboxed Execution  │
               │    - Automatic Rollback   │ (StatefulDataStore)
               └──────────┬───────────┘
                          ▼
            ┌─────────────────────────────┐
            │ 4. Decoupled Aggregates     │ (metrics.json)
            │    - Compact Math Payload   │
            └─────────────┬───────────────┘
                          ▼
            ┌─────────────────────────────┐
            │ 5. Client-Side Rendering    │ (Plotly.js / Chart.js)
            │    - GPU-Accelerated HTML   │
            └─────────────────────────────┘
```

### Key Design Innovations

1. **Stateful Rollback Safety Net (`StatefulDataStore`)**: Before executing any tool-plan step, the database takes a sequence-labeled checkpoint (`v0`, `v1`, etc.) via **deep-copy isolation**. If a step throws a mathematical or datatype exception, the stateful store instantly rolls back the active dataset and metadata to the last stable version and feeds the error traceback back to the LLM for self-correction.
2. **Decoupled Graphics Engine (Client-Side HTML/JS Charting)**: Rather than running heavy server-side Matplotlib/Seaborn drawing loops that choke the server CPU, spike memory, and output rigid PNGs, the backend processes raw data into compact mathematical summaries (correlation matrices, 1D bin counts, and downsampled coordinates). These are exported as a tiny JSON package (`metrics.json`) and rendered dynamically on the client's web-engine using CDNs like **Plotly.js** or **Chart.js** with full interactive panning, zooming, and hover tooltips.
3. **Algorithmic Pre-Profiler (`profiler.py`)**: Computes initial datatypes, null-value percentages, skews, and cardinality before the agent is triggered. This condenses massive raw CSV files into an ultra-compressed, token-efficient `metadata_profile.json` format, dramatically reducing LLM prompt costs and context usage.

---

## Repository Structure

```directory
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
│   └── manage.py               # Django CLI management script
├── test_data/                  # Sample CSV datasets
├── benchmarks/                 # Scaling benchmark and fault-tolerance testing scripts
│   ├── latency_footprint_benchmark.py
│   └── rollback_recovery_benchmark.py
├── EDA/                        # Archive of generated analysis runs
├── requirements.txt            # Python dependencies
├── Dockerfile                  # Container deployment specification
└── .gitignore                  # Prevents caching of models, private credentials, and environments
```

---

## Performance & Scaling Benchmarks

The decoupled state and client-side charting architecture have been aggressively benchmarked under real-world constraints, displaying exceptional stability and throughput.

### 1. Latency & Memory Footprint Scaling (JSON Aggregation vs. Server-Side PNGs)
The table below documents the transformation of **AutoEDA Pro** when migrating from CPU-bound Matplotlib/Seaborn rasterization to our decoupled client-side rendering model:

| Scale (Rows) | Traditional Server Plotting Latency | Decoupled Client JSON Latency | Speed Multiplier | Output File Size (Old PNGs) | Output File Size (New JSON) | Network Footprint Reduction |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **1,000** | 3.4635s | 0.0258s | **134.2x faster** | 430.0 KB | 44.55 KB | **9.7x smaller** |
| **5,000** | 4.2147s | 0.0203s | **207.6x faster** | 454.8 KB | 45.40 KB | **10.0x smaller** |
| **10,000** | 5.4785s | 0.0311s | **176.1x faster** | 458.6 KB | 45.36 KB | **10.1x smaller** |
| **32,000** | 9.0163s | 0.0456s | **197.7x faster** | 481.8 KB | 45.46 KB | **10.6x smaller** |


*   **Flat Payload Scaling**: Regardless of row depth, the compiled output remains compressed to **~45 KB**, ensuring instantaneous data transmission and fluid browser rendering.

---

### 2. Fault Tolerance & Self-Correction (Rollback Recovery Benchmark)
We evaluated the resilience of **AutoEDA's** rollback recovery engine against naive, stateless AI agents by executing **100 multi-step pipeline runs (200 proposed tasks)** while injecting a severe **40% failure rate** (intentional datatype mismatches, mathematical errors, and division-by-zero to mimic LLM planning hallucinations):

| Performance Dimension | Stateful Rollback Pipeline (AutoEDA) | Traditional Stateless Agent |
| :--- | :--- | :--- |
| **Pristine State Isolation** | Yes (Deep Copy isolation) | No (In-place dataframe modification) |
| **Step-Level Rollbacks** | Yes (Rolls back to $v(N-1)$ on error) | No (Requires complete pipeline restart) |
| **Crash Recovery Rate** | **100.00%** | **0.00%** (Crashes instantly) |
| **End-to-End Run Success Rate** | **100.00%** | **81.50%** |
| **Average Rollback Latency** | **0.0948 ms** | N/A |

*   **Self-Healing**: Standard agents crashed on every injected bug, polluting memory and rendering the session unviable. AutoEDA captured the tracebacks, rolled back the state, and successfully re-routed tasks—achieving a **100% run success rate**.
*   **Sub-Millisecond Restoration**: Discarding memory pollution and restoring the dataset to a pristine version takes just **0.0948 milliseconds**, presenting zero noticeable lag to the user.

---

## Comparative Landscape: AutoEDA vs. Market Alternatives

| Feature Profile | Traditional Auto-EDA (ydata, Sweetviz) | Naive AI Coding Agents | AutoEDA Pro |
| :--- | :--- | :--- | :--- |
| **Adaptability** | None (Static, rigid outputs) | High (Writes custom script files) | **High (Generates customized JSON plans)** |
| **State Protection** | Yes (Read-only operations) | None (In-place dataframe corruption) | **Yes (Deep-copy checkpoint boundaries)** |
| **Error Handling** | Fail-fast (Crashes on exceptions) | Fatal-crash (Stops execution on bugs) | **Self-healing (Automatic rollback & retry)** |
| **Server Latency** | High (Server-side rasterization) | High (Iterative syntax checking) | **Low (Sub-50ms mathematical serialization)** |
| **API Costs** | None (Local execution) | High (Pipes raw data / repeating runs) | **Low (Uses condensed schema summaries)** |
| **Interactivity** | Minimal (Static, flat reports) | Minimal (Printed code outcomes) | **Rich (Hardware-accelerated client charts)** |

---

## Execution & Deployment

### 1. Installation
Clone the repository and install all required python libraries:
```bash
git clone https://github.com/AniceKV/AutoEDA.git
cd AutoEDA
pip install -r requirements.txt
```



### 4. Running Local Offline Models (LM Studio / Ollama)
For a 100% free, low-latency, private offline environment:
1. Fire up **LM Studio** and head to the **Local Server** (double-plug) tab.
2. Load a compact model (e.g. `qwen2.5-coder-7b` or `gemma-2-2b-it`).
3. Toggle GPU Offload on to accelerate generation and start the port (default: `http://localhost:1234`).
4. Point your pipeline base URL in `autoeda_core/agent_loop.py` to your local environment.

### 4. Running the Pipelines & Benchmarks
```bash
# Execute the agentic command-line interface
python autoeda_core/agent_loop.py

# Run the comparative latency/footprint scaling benchmark
python benchmarks/latency_footprint_benchmark.py

# Run the exception-injected rollback recovery benchmark
python benchmarks/rollback_recovery_benchmark.py
```

### 5. Running the Django Web Server
Launch the interactive dashboard to upload data files, inspect active planning, and interact with compiled reports:
```bash
python django_app/manage.py runserver 8000
```
Open `http://127.0.0.1:8000/` in your browser.

### 6. Docker Deployment
Build and run using Docker:

```bash
docker build -t autoeda .
docker run -p 8000:8000 --env-file .env autoeda
```



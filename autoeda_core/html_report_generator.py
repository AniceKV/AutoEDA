import os
import json
import base64
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from jinja2 import Template
from typing import Dict, Any, List, Optional

pio.templates.default = "plotly_dark"

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AutoEDA - Interactive Profile Report - {{ dataset_name }}</title>
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap" rel="stylesheet">
    <script>
        (function() {
            const t = localStorage.getItem('autoeda-theme');
            if (t === 'light') document.documentElement.setAttribute('data-theme', 'light');
        })();
    </script>
    <style>
        :root {
            --bg-dark: #09090b;
            --bg-card: #121215;
            --bg-card-hover: #1c1c20;
            --border-color: #27272a;
            --border-highlight: #ffffff;
            --primary: #ffffff;
            --primary-glow: rgba(255, 255, 255, 0.15);
            --text-main: #fafafa;
            --text-muted: #a1a1aa;
            --text-dim: #71717a;
            --logo-bg: linear-gradient(135deg, #ffffff 0%, #e4e4e7 100%);
            --logo-text: #09090b;
            --btn-active-bg: linear-gradient(135deg, #ffffff 0%, #e4e4e7 100%);
            --btn-active-text: #09090b;
            --toggle-track-off: #27272a;
            --toggle-track-on: linear-gradient(135deg, #ffffff 0%, #e4e4e7 100%);
            --toggle-knob-off: #71717a;
            --toggle-knob-on: #09090b;
        }

        [data-theme="light"] {
            --bg-dark: #fafafa;
            --bg-card: #ffffff;
            --bg-card-hover: #f4f4f5;
            --border-color: #e4e4e7;
            --border-highlight: #18181b;
            --primary: #09090b;
            --primary-glow: rgba(0, 0, 0, 0.12);
            --text-main: #09090b;
            --text-muted: #52525b;
            --text-dim: #71717a;
            --logo-bg: linear-gradient(135deg, #18181b 0%, #09090b 100%);
            --logo-text: #ffffff;
            --btn-active-bg: linear-gradient(135deg, #27272a 0%, #09090b 100%);
            --btn-active-text: #ffffff;
            --toggle-track-off: #e4e4e7;
            --toggle-track-on: linear-gradient(135deg, #27272a 0%, #09090b 100%);
            --toggle-knob-off: #ffffff;
            --toggle-knob-on: #ffffff;
        }

        * { box-sizing: border-box; margin: 0; padding: 0; }
        
        body {
            font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
            background-color: var(--bg-dark);
            color: var(--text-main);
            line-height: 1.5;
            padding-bottom: 40px;
            transition: background 0.2s ease, color 0.2s ease;
        }

        /* Header Banner */
        .header-container {
            background: var(--bg-card);
            backdrop-filter: blur(16px);
            border-bottom: 1px solid var(--border-color);
            position: sticky;
            top: 0;
            z-index: 1000;
            padding: 1rem 2rem;
            display: flex;
            justify-content: space-between;
            align-items: center;
            transition: background 0.2s ease, border-color 0.2s ease;
        }
        
        .header-title-group {
            display: flex;
            align-items: center;
            gap: 12px;
        }
        
        .header-logo {
            font-size: 0.9rem;
            font-weight: 800;
            letter-spacing: 0.08em;
            background: var(--logo-bg);
            color: var(--logo-text);
            padding: 6px 12px;
            border-radius: 8px;
            box-shadow: 0 2px 8px var(--primary-glow);
        }

        .header-text h1 {
            font-size: 1.35rem;
            font-weight: 800;
            letter-spacing: -0.02em;
            color: var(--text-main);
        }

        .header-text p {
            font-size: 0.78rem;
            color: var(--text-muted);
        }

        .header-badge {
            background: var(--bg-card-hover);
            border: 1px solid var(--border-color);
            color: var(--text-main);
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 0.75rem;
            font-weight: 600;
            letter-spacing: 0.05em;
        }

        /* Toggle Switch Component */
        .toggle-switch {
            position: relative;
            width: 36px; height: 20px;
            display: inline-block;
            flex-shrink: 0;
        }

        .toggle-switch input { opacity: 0; width: 0; height: 0; }

        .toggle-track {
            position: absolute;
            inset: 0;
            background: var(--toggle-track-off);
            border: 1px solid var(--border-color);
            border-radius: 10px;
            cursor: pointer;
            transition: background 0.18s ease;
        }

        .toggle-switch input:checked + .toggle-track {
            background: var(--toggle-track-on);
            border-color: var(--border-highlight);
        }

        .toggle-track::after {
            content: '';
            position: absolute;
            width: 14px; height: 14px;
            border-radius: 50%;
            background: var(--toggle-knob-off);
            top: 2px; left: 2px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.3);
            transition: transform 0.18s ease, background 0.18s ease;
        }

        .toggle-switch input:checked + .toggle-track::after {
            transform: translateX(16px);
            background: var(--toggle-knob-on);
        }

        .main-wrapper {
            max-width: 1440px;
            margin: 0 auto;
            padding: 1.5rem 2rem;
        }

        /* Navigation Tabs */
        .nav-tabs {
            display: flex;
            gap: 8px;
            background: var(--bg-card);
            padding: 6px;
            border-radius: 12px;
            border: 1px solid var(--border-color);
            margin-bottom: 1.5rem;
            overflow-x: auto;
        }

        .tab-btn {
            background: transparent;
            border: none;
            color: var(--text-muted);
            padding: 10px 18px;
            font-family: inherit;
            font-weight: 600;
            font-size: 0.88rem;
            border-radius: 8px;
            cursor: pointer;
            transition: all 0.2s ease;
            white-space: nowrap;
            display: flex;
            align-items: center;
            gap: 8px;
        }

        .tab-btn:hover {
            color: var(--text-main);
            background: var(--bg-card-hover);
        }

        .tab-btn.active {
            background: var(--btn-active-bg);
            color: var(--btn-active-text);
            font-weight: 700;
            box-shadow: 0 2px 8px var(--primary-glow);
        }

        .tab-content {
            display: none;
        }

        .tab-content.active {
            display: block;
            animation: fadeIn 0.3s ease;
        }

        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(6px); }
            to { opacity: 1; transform: translateY(0); }
        }

        /* Cards & Grid Layouts */
        .grid-4 {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
            gap: 1rem;
            margin-bottom: 1.5rem;
        }

        .grid-2 {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(480px, 1fr));
            gap: 1.5rem;
            margin-bottom: 1.5rem;
        }

        .metric-card {
            background: var(--bg-card);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            padding: 1.25rem;
            backdrop-filter: blur(12px);
            transition: all 0.2s ease;
        }

        .metric-card:hover {
            border-color: var(--border-highlight);
            transform: translateY(-2px);
        }

        .metric-val {
            font-size: 1.8rem;
            font-weight: 800;
            margin-top: 4px;
            color: var(--text-main);
        }

        .metric-lbl {
            font-size: 0.72rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.06em;
            color: var(--text-dim);
        }

        /* Alert Section */
        .alerts-container {
            background: var(--bg-card);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            padding: 1.5rem;
            margin-bottom: 1.5rem;
        }

        .section-title {
            font-size: 1.1rem;
            font-weight: 700;
            margin-bottom: 1rem;
            display: flex;
            align-items: center;
            gap: 10px;
            color: var(--text-main);
        }

        .alert-item {
            display: flex;
            align-items: center;
            gap: 12px;
            padding: 10px 14px;
            border-radius: 8px;
            margin-bottom: 8px;
            font-size: 0.85rem;
            background: var(--bg-card-hover);
            border: 1px solid var(--border-color);
            color: var(--text-main);
        }

        .alert-badge {
            padding: 2px 8px;
            border-radius: 4px;
            font-weight: 700;
            font-size: 0.7rem;
            text-transform: uppercase;
            background: var(--border-color);
            color: var(--text-main);
        }

        /* Variable Cards */
        .var-search {
            width: 100%;
            background: var(--bg-card);
            border: 1px solid var(--border-color);
            color: var(--text-main);
            padding: 12px 16px;
            border-radius: 10px;
            font-family: inherit;
            font-size: 0.9rem;
            margin-bottom: 1.5rem;
            outline: none;
            transition: border-color 0.2s;
        }

        .var-search:focus {
            border-color: var(--border-highlight);
        }

        .var-card {
            background: var(--bg-card);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            padding: 1.5rem;
            margin-bottom: 1.5rem;
            backdrop-filter: blur(12px);
        }

        .var-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 1rem;
            padding-bottom: 0.75rem;
            border-bottom: 1px solid var(--border-color);
        }

        .var-name {
            font-size: 1.15rem;
            font-weight: 700;
            font-family: 'JetBrains Mono', monospace;
            color: var(--text-main);
        }

        .var-tags {
            display: flex;
            gap: 6px;
        }

        .tag {
            font-size: 0.7rem;
            font-weight: 700;
            padding: 3px 8px;
            border-radius: 6px;
            letter-spacing: 0.04em;
            background: var(--bg-card-hover);
            color: var(--text-main);
            border: 1px solid var(--border-color);
        }

        .var-body {
            display: grid;
            grid-template-columns: 320px 1fr;
            gap: 1.5rem;
        }

        @media (max-width: 900px) {
            .var-body { grid-template-columns: 1fr; }
        }

        .stats-table {
            width: 100%;
            border-collapse: collapse;
            font-size: 0.82rem;
        }

        .stats-table td {
            padding: 6px 8px;
            border-bottom: 1px solid var(--border-color);
        }

        .stats-table tr:last-child td { border-bottom: none; }
        .stats-lbl { color: var(--text-muted); font-weight: 500; }
        .stats-val { text-align: right; font-weight: 700; font-family: 'JetBrains Mono', monospace; color: var(--text-main); }

        /* Agent Reasoning Timeline */
        .timeline {
            position: relative;
            padding-left: 24px;
            margin-top: 1rem;
        }

        .timeline::before {
            content: '';
            position: absolute;
            left: 7px;
            top: 10px;
            bottom: 10px;
            width: 2px;
            background: var(--border-color);
        }

        .timeline-step {
            position: relative;
            background: var(--bg-card);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            padding: 1.25rem;
            margin-bottom: 1.25rem;
        }

        .timeline-step::before {
            content: '';
            position: absolute;
            left: -24px;
            top: 20px;
            width: 12px;
            height: 12px;
            border-radius: 50%;
            background: var(--primary);
        }

        .step-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 8px;
        }

        .step-title {
            font-size: 0.95rem;
            font-weight: 700;
            color: var(--text-main);
            font-family: 'JetBrains Mono', monospace;
        }

        .step-rationale {
            font-size: 0.88rem;
            color: var(--text-muted);
            margin-bottom: 10px;
            background: var(--bg-card-hover);
            padding: 10px 14px;
            border-radius: 8px;
            border-left: 3px solid var(--primary);
        }

        .step-code {
            background: var(--bg-card-hover);
            border: 1px solid var(--border-color);
            border-radius: 8px;
            padding: 10px 12px;
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.78rem;
            color: var(--text-main);
            overflow-x: auto;
            white-space: pre-wrap;
        }

        /* Image Artifact Cards */
        .img-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
            gap: 1.5rem;
        }

        .img-card {
            background: var(--bg-card);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            padding: 1rem;
            overflow: hidden;
        }

        .img-card img {
            width: 100%;
            height: auto;
            border-radius: 8px;
            display: block;
            margin-bottom: 10px;
        }

        .img-caption {
            font-size: 0.82rem;
            color: var(--text-muted);
            font-style: italic;
        }

        /* Blueprint Table & Section */
        .blueprint-box {
            background: var(--bg-card);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            padding: 1.5rem;
            margin-bottom: 1.5rem;
        }

        .blueprint-item {
            margin-bottom: 1rem;
            padding-bottom: 1rem;
            border-bottom: 1px solid var(--border-color);
        }

        .blueprint-item:last-child {
            border-bottom: none;
            margin-bottom: 0;
            padding-bottom: 0;
        }

        .blueprint-key {
            font-weight: 700;
            color: var(--text-main);
            font-size: 0.9rem;
            margin-bottom: 4px;
        }

        .blueprint-val {
            font-size: 0.85rem;
            color: var(--text-muted);
        }
    </style>
</head>
<body>

    <!-- Header Banner -->
    <header class="header-container">
        <div class="header-title-group">
            <div class="header-logo">AutoEDA</div>
            <div class="header-text">
                <h1>{{ dataset_name }}</h1>
                <p>Interactive Analysis Report &bull; Generated: {{ generation_time }}</p>
            </div>
        </div>
        <div style="display: flex; align-items: center; gap: 16px;">
            <div title="Toggle Light / Dark Mode" style="display: flex; align-items: center; gap: 8px;">
                <span style="font-size: 0.75rem; font-weight: 600; color: var(--text-muted); user-select: none;">Light Mode</span>
                <label class="toggle-switch">
                    <input type="checkbox" id="report-theme-checkbox" onchange="toggleReportTheme(this.checked)">
                    <span class="toggle-track"></span>
                </label>
            </div>
            <span class="header-badge">{{ dimensions.rows }} Rows x {{ dimensions.columns }} Cols</span>
        </div>
    </header>

    <div class="main-wrapper">

        <!-- Nav Tabs -->
        <nav class="nav-tabs">
            <button class="tab-btn active" onclick="switchTab('overview')">Overview & Data Quality</button>
            <button class="tab-btn" onclick="switchTab('reasoning')">Agent Plan & Execution Reasoning</button>
            <button class="tab-btn" onclick="switchTab('variables')">Variable Profiles</button>
            <button class="tab-btn" onclick="switchTab('interactions')">Bivariate Relationships</button>
            <button class="tab-btn" onclick="switchTab('correlations')">Correlation Analysis</button>
            <button class="tab-btn" onclick="switchTab('features')">Feature Engineering Specs</button>
            <button class="tab-btn" onclick="switchTab('blueprint')">Predictive Modeling Strategy</button>
        </nav>

        <!-- TAB 1: OVERVIEW -->
        <div id="overview" class="tab-content active">
            <div class="grid-4">
                <div class="metric-card">
                    <div class="metric-lbl">Total Observations</div>
                    <div class="metric-val">{{ dimensions.rows }}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-lbl">Total Features</div>
                    <div class="metric-val">{{ dimensions.columns }}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-lbl">Missing Cells</div>
                    <div class="metric-val">{{ total_missing }} ({{ missing_pct }}%)</div>
                </div>
                <div class="metric-card">
                    <div class="metric-lbl">Target Variable</div>
                    <div class="metric-val" style="font-size: 1.3rem;">{{ target_column if target_column else "None" }}</div>
                </div>
            </div>

            <!-- Alerts Panel -->
            <div class="alerts-container">
                <div class="section-title">Data Quality & Profiling Alerts ({{ alerts|length }})</div>
                {% if alerts %}
                    {% for alert in alerts %}
                    <div class="alert-item alert-{{ alert.level }}">
                        <span class="alert-badge">{{ alert.type }}</span>
                        <div><strong>{{ alert.column }}:</strong> {{ alert.message }}</div>
                    </div>
                    {% endfor %}
                {% else %}
                    <div class="alert-item alert-notice">
                        <span class="alert-badge">Clean</span>
                        <div>No severe quality anomalies or high missingness detected.</div>
                    </div>
                {% endif %}
            </div>

            <!-- Data Types Summary -->
            <div class="grid-2">
                <div class="metric-card">
                    <div class="section-title">Variable Type Breakdown</div>
                    <div id="chart-dtypes" style="height: 260px;"></div>
                </div>
                <div class="metric-card">
                    <div class="section-title">Missing Value Distribution</div>
                    <div id="chart-missing" style="height: 260px;"></div>
                </div>
            </div>
        </div>

        <!-- TAB 2: AGENT REASONING -->
        <div id="reasoning" class="tab-content">
            <div class="alerts-container">
                <div class="section-title">Agent Plan & Tool Execution Trajectory</div>
                <p style="font-size: 0.85rem; color: var(--text-muted); margin-bottom: 1rem;">
                    This section captures the exact reasoning chain, task planning, tool arguments, and output results produced by the LLM agent during its autonomous refinement loop.
                </p>
                <div class="timeline">
                    {% if agent_trajectory %}
                        {% for step in agent_trajectory %}
                        <div class="timeline-step">
                            <div class="step-header">
                                <span class="step-title">Loop {{ step.loop }} &bull; {{ step.tool }}</span>
                                <span class="tag tag-type">{{ step.status }}</span>
                            </div>
                            {% if step.rationale %}
                            <div class="step-rationale">
                                <strong>Agent Rationale:</strong> {{ step.rationale }}
                            </div>
                            {% endif %}
                            <div class="step-code">Arguments: {{ step.args_json }}
Result Summary: {{ step.result }}</div>
                        </div>
                        {% endfor %}
                    {% else %}
                        <div class="timeline-step">
                            <div class="step-header"><span class="step-title">Single Execution Plan</span></div>
                            <div class="step-rationale">Standard automated exploratory pipeline executed.</div>
                        </div>
                    {% endif %}
                </div>
            </div>
        </div>

        <!-- TAB 3: VARIABLES PROFILING -->
        <div id="variables" class="tab-content">
            <input type="text" id="var-search-input" class="var-search" placeholder="Search variables by name or data type..." onkeyup="filterVariables()">

            {% for col in columns_profile %}
            <div class="var-card" data-var-name="{{ col.column|lower }}" data-var-type="{{ col.dtype|lower }}">
                <div class="var-header">
                    <div class="var-name">{{ col.column }}</div>
                    <div class="var-tags">
                        <span class="tag tag-type">{{ col.dtype }}</span>
                        <span class="tag tag-miss">Missing: {{ col.missing_count }} ({{ col.missing_pct }}%)</span>
                        <span class="tag tag-uniq">Unique: {{ col.cardinality }} ({{ col.unique_pct }}%)</span>
                    </div>
                </div>
                <div class="var-body">
                    <div>
                        <table class="stats-table">
                            <tr><td class="stats-lbl">Distinct Values</td><td class="stats-val">{{ col.cardinality }}</td></tr>
                            <tr><td class="stats-lbl">Missing Count</td><td class="stats-val">{{ col.missing_count }}</td></tr>
                            <tr><td class="stats-lbl">Missing Percentage</td><td class="stats-val">{{ col.missing_pct }}%</td></tr>
                            {% if col.mean is not none %}
                            <tr><td class="stats-lbl">Mean</td><td class="stats-val">{{ col.mean }}</td></tr>
                            <tr><td class="stats-lbl">Median</td><td class="stats-val">{{ col.median }}</td></tr>
                            <tr><td class="stats-lbl">Std Deviation</td><td class="stats-val">{{ col.std }}</td></tr>
                            <tr><td class="stats-lbl">Min / Max</td><td class="stats-val">{{ col.min }} / {{ col.max }}</td></tr>
                            <tr><td class="stats-lbl">IQR (Q1 / Q3)</td><td class="stats-val">{{ col.iqr }} ({{ col.q1 }} / {{ col.q3 }})</td></tr>
                            <tr><td class="stats-lbl">Skewness</td><td class="stats-val">{{ col.skew }}</td></tr>
                            <tr><td class="stats-lbl">Kurtosis</td><td class="stats-val">{{ col.kurtosis }}</td></tr>
                            {% endif %}
                        </table>
                    </div>
                    <div>
                        <div id="var-chart-{{ loop.index }}" style="height: 240px; width: 100%;"></div>
                    </div>
                </div>
            </div>
            {% endfor %}
        </div>

        <!-- TAB 4: INTERACTIONS & BIVARIATE -->
        <div id="interactions" class="tab-content">
            <div class="section-title">Target Interactions & Bivariate Relationships</div>
            <div class="img-grid">
                {% if visual_artifacts %}
                    {% for img in visual_artifacts %}
                    <div class="img-card">
                        <div style="font-weight: 700; margin-bottom: 8px; font-family: 'JetBrains Mono', monospace; font-size: 0.85rem;">
                            {{ img.name }}
                        </div>
                        <img src="data:image/png;base64,{{ img.b64 }}" alt="{{ img.name }}">
                        <div class="img-caption">Artifact generated by AutoEDA tool pipeline.</div>
                    </div>
                    {% endfor %}
                {% else %}
                    <div class="metric-card">No static image artifacts generated.</div>
                {% endif %}
            </div>
        </div>

        <!-- TAB 5: CORRELATIONS -->
        <div id="correlations" class="tab-content">
            <div class="metric-card" style="margin-bottom: 1.5rem;">
                <div class="section-title">Feature Correlation Heatmap</div>
                <div id="chart-correlation" style="width: 100%; min-height: 520px;"></div>
            </div>

            {% for img in visual_artifacts %}
                {% if 'correlation' in img.name|lower %}
                <div class="metric-card" style="margin-bottom: 1.5rem;">
                    <div class="section-title">Static Correlation Heatmap Asset: {{ img.name }}</div>
                    <img src="data:image/png;base64,{{ img.b64 }}" style="max-width: 100%; height: auto; border-radius: 8px; display: block;" alt="{{ img.name }}">
                </div>
                {% endif %}
            {% endfor %}

            {% if categorical_associations %}
            <div class="metric-card">
                <div class="section-title">Categorical Associations (Cramér's V)</div>
                <table class="stats-table" style="font-size: 0.85rem;">
                    <thead>
                        <tr style="color: var(--text-muted); text-align: left;">
                            <th>Feature 1</th>
                            <th>Feature 2</th>
                            <th>Cramér's V</th>
                        </tr>
                    </thead>
                    <tbody>
                        {% for row in categorical_associations %}
                        <tr>
                            <td class="stats-val" style="text-align: left;">{{ row.feature_1 }}</td>
                            <td class="stats-val" style="text-align: left;">{{ row.feature_2 }}</td>
                            <td class="stats-val">{{ row.cramers_v }}</td>
                        </tr>
                        {% endfor %}
                    </tbody>
                </table>
            </div>
            {% endif %}
        </div>

        <!-- TAB 6: ENGINEERED FEATURES -->
        <div id="features" class="tab-content">
            <div class="blueprint-box">
                <div class="section-title">LLM-Engineered Feature Specifications</div>
                {% if engineered_features %}
                    {% for feat in engineered_features %}
                    <div class="blueprint-item">
                        <div class="blueprint-key">Feature: <code>{{ feat.feature_name }}</code></div>
                        <div class="blueprint-val"><strong>Formula / Method:</strong> <code>{{ feat.formula or feat.method or 'Custom' }}</code></div>
                        <div class="blueprint-val"><strong>Stated Rationale:</strong> {{ feat.rationale or feat.purpose or 'Enhance predictive signal' }}</div>
                    </div>
                    {% endfor %}
                {% else %}
                    <div class="blueprint-val">No custom domain features engineered during this run.</div>
                {% endif %}
            </div>
        </div>

        <!-- TAB 7: PREDICTIVE BLUEPRINT -->
        <div id="blueprint" class="tab-content">
            <div class="blueprint-box">
                <div class="section-title">Predictive Modeling Strategy Blueprint</div>
                {% if predictive_blueprint %}
                    {% for key, val in predictive_blueprint.items() %}
                    <div class="blueprint-item">
                        <div class="blueprint-key">{{ key|replace('_', ' ')|title }}</div>
                        <div class="blueprint-val">
                            {% if val is string or val is number %}
                                {{ val }}
                            {% elif val is iterable and val is not mapping %}
                                <ul style="margin-left: 20px;">
                                    {% for item in val %}
                                    <li>{{ item }}</li>
                                    {% endfor %}
                                </ul>
                            {% elif val is mapping %}
                                {% for subk, subv in val.items() %}
                                <div><strong>{{ subk }}:</strong> {{ subv }}</div>
                                {% endfor %}
                            {% endif %}
                        </div>
                    </div>
                    {% endfor %}
                {% else %}
                    <div class="blueprint-val">No predictive modeling blueprint generated.</div>
                {% endif %}
            </div>
        </div>

    </div>

    <script>
        function switchTab(tabId) {
            document.querySelectorAll('.tab-btn').forEach(btn => btn.classList.remove('active'));
            document.querySelectorAll('.tab-content').forEach(content => content.classList.remove('active'));
            
            const targetContent = document.getElementById(tabId);
            if (event && event.currentTarget) {
                event.currentTarget.classList.add('active');
            }
            if (targetContent) {
                targetContent.classList.add('active');
            }

            // Force Plotly resize & relayout for all plots inside the newly visible tab
            setTimeout(() => {
                window.dispatchEvent(new Event('resize'));
                if (targetContent) {
                    targetContent.querySelectorAll('.js-plotly-plot').forEach(plotDiv => {
                        Plotly.Plots.resize(plotDiv);
                        Plotly.relayout(plotDiv, {autosize: true});
                    });
                }
            }, 60);
        }

        function filterVariables() {
            const query = document.getElementById('var-search-input').value.toLowerCase();
            document.querySelectorAll('.var-card').forEach(card => {
                const name = card.getAttribute('data-var-name');
                const type = card.getAttribute('data-var-type');
                if (name.includes(query) || type.includes(query)) {
                    card.style.display = 'block';
                } else {
                    card.style.display = 'none';
                }
            });
        }

        function toggleReportTheme(isLight) {
            const themeName = isLight ? 'light' : 'dark';
            if (isLight) {
                document.documentElement.setAttribute('data-theme', 'light');
            } else {
                document.documentElement.removeAttribute('data-theme');
            }
            localStorage.setItem('autoeda-theme', themeName);

            const fontColor = isLight ? '#09090b' : '#fafafa';
            const gridColor = isLight ? '#e4e4e7' : '#27272a';

            document.querySelectorAll('.js-plotly-plot').forEach(plotDiv => {
                try {
                    Plotly.relayout(plotDiv, {
                        'paper_bgcolor': 'rgba(0,0,0,0)',
                        'plot_bgcolor': 'rgba(0,0,0,0)',
                        'font.color': fontColor,
                        'xaxis.gridcolor': gridColor,
                        'yaxis.gridcolor': gridColor,
                        'xaxis.color': fontColor,
                        'yaxis.color': fontColor
                    });
                } catch(e){}
            });
        }

        window.addEventListener('DOMContentLoaded', () => {
            const savedTheme = localStorage.getItem('autoeda-theme');
            if (savedTheme === 'light') {
                const cb = document.getElementById('report-theme-checkbox');
                if (cb) cb.checked = true;
                document.documentElement.setAttribute('data-theme', 'light');
            }

            // Render Overview Dtypes Chart
            const dtypeData = {{ dtype_chart_json|safe }};
            if (dtypeData && dtypeData.data && dtypeData.data.length > 0) {
                Plotly.newPlot('chart-dtypes', dtypeData.data, dtypeData.layout, {responsive: true, displayModeBar: false});
            }

            // Render Missing Values Chart
            const missingData = {{ missing_chart_json|safe }};
            if (missingData && missingData.data && missingData.data.length > 0) {
                Plotly.newPlot('chart-missing', missingData.data, missingData.layout, {responsive: true, displayModeBar: false});
            }

            // Render Correlation Heatmap
            const corrData = {{ corr_chart_json|safe }};
            if (corrData && corrData.data && corrData.data.length > 0) {
                Plotly.newPlot('chart-correlation', corrData.data, corrData.layout, {responsive: true, displaylogo: false});
            }

            // Render Variable Distribution Charts
            const varCharts = {{ var_charts_json|safe }};
            for (const [divId, chartSpec] of Object.entries(varCharts)) {
                if (chartSpec && chartSpec.data && chartSpec.data.length > 0) {
                    Plotly.newPlot(divId, chartSpec.data, chartSpec.layout, {responsive: true, displayModeBar: false});
                }
            }

            if (savedTheme === 'light') {
                toggleReportTheme(true);
            }
        });
    </script>
</body>
</html>
"""


def compute_alerts(profile: Dict[str, Any], corr_matrix: Optional[pd.DataFrame] = None) -> List[Dict[str, Any]]:
    """
    Detects data quality warnings: high missingness, high skewness, constant columns, high cardinality, etc.
    """
    alerts = []
    columns = profile.get("columns", [])
    
    for col in columns:
        name = col.get("column", "Unknown")
        missing_pct = col.get("missing_pct", 0)
        cardinality = col.get("cardinality", 0)
        unique_pct = col.get("unique_pct", 0)
        skew = col.get("skew")
        dtype = col.get("dtype", "")

        # 1. Missingness Alert
        if missing_pct >= 20.0:
            alerts.append({
                "column": name,
                "type": "High Missingness",
                "level": "warning",
                "message": f"Contains {missing_pct}% missing values ({col.get('missing_count')} rows)."
            })
        elif missing_pct > 0.0:
            alerts.append({
                "column": name,
                "type": "Missing Values",
                "level": "info",
                "message": f"{missing_pct}% values missing."
            })

        # 2. Skewness Alert
        if skew is not None and abs(skew) >= 1.5:
            alerts.append({
                "column": name,
                "type": "High Skewness",
                "level": "notice",
                "message": f"Highly skewed distribution with skewness coefficient = {skew:.2f}."
            })

        # 3. Constant Column Alert
        if cardinality == 1:
            alerts.append({
                "column": name,
                "type": "Constant Column",
                "level": "warning",
                "message": "Contains only 1 unique value across all rows (zero variance)."
            })

        # 4. High Cardinality Alert
        if unique_pct > 80.0 and cardinality > 50 and not dtype.startswith("float"):
            alerts.append({
                "column": name,
                "type": "High Cardinality",
                "level": "notice",
                "message": f"Very high cardinality ({cardinality} distinct values, {unique_pct}% unique)."
            })

    # 5. Correlation Alerts
    if corr_matrix is not None and not corr_matrix.empty:
        cols = corr_matrix.columns
        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                val = corr_matrix.iloc[i, j]
                if pd.notnull(val) and abs(val) >= 0.85:
                    alerts.append({
                        "column": f"{cols[i]} & {cols[j]}",
                        "type": "High Correlation",
                        "level": "warning",
                        "message": f"Strong collinearity detected with correlation r = {val:.2f}."
                    })

    return alerts


def build_variable_chart(df: Optional[pd.DataFrame], col_name: str, col_info: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Generates an interactive Plotly figure spec dict for a single column's distribution.
    - Categorical / Discrete variables -> Count Plot (Bar chart with frequency counts & text labels).
    - Continuous Numeric variables -> Numeric Distribution Plot (Histogram + marginal Box Plot).
    - Fallback for missing/empty columns -> Clear annotated chart.
    """
    layout = dict(
        margin=dict(l=30, r=20, t=30, b=30),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#cbd5e1', family='Plus Jakarta Sans'),
        xaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.05)'),
        yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.05)')
    )

    try:
        if df is None or col_name not in df.columns:
            fig = go.Figure()
            fig.add_annotation(
                text=f"No raw data loaded for '{col_name}'",
                xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False,
                font=dict(size=13, color="#94a3b8")
            )
            fig.update_layout(**layout)
            return json.loads(pio.to_json(fig))

        s = df[col_name].dropna()
        if len(s) == 0:
            fig = go.Figure()
            fig.add_annotation(
                text=f"All values missing (NaN) for '{col_name}'",
                xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False,
                font=dict(size=13, color="#ef4444")
            )
            fig.update_layout(**layout)
            return json.loads(pio.to_json(fig))

        is_bool = pd.api.types.is_bool_dtype(s)
        is_num = pd.api.types.is_numeric_dtype(s) and not is_bool
        n_unique = s.nunique()

        # Categorical, boolean, or low-cardinality discrete numeric features (<= 10 unique values) get Count Plot
        is_categorical = (not is_num) or is_bool or (n_unique <= 10)

        if is_categorical:
            counts_series = s.astype(str).value_counts().head(15)
            cat_df = pd.DataFrame({
                "Category": counts_series.index,
                "Count": counts_series.values
            })
            
            fig = px.bar(
                cat_df,
                x="Category",
                y="Count",
                color_discrete_sequence=['#a855f7'],
                text="Count"
            )
            fig.update_traces(
                textposition='outside',
                texttemplate='%{text}',
                marker_line_color='rgba(255,255,255,0.15)',
                marker_line_width=1
            )
            fig.update_layout(
                title=dict(text=f"Categorical Count Plot: {col_name}", font=dict(size=12, color='#e9d5ff')),
                xaxis_title="Category",
                yaxis_title="Count",
                **layout
            )
            return json.loads(pio.to_json(fig))
        else:
            # Continuous Numeric Distribution Plot (Histogram + Box Plot)
            s_clean = s.replace([np.inf, -np.inf], np.nan).dropna()
            fig = px.histogram(
                s_clean, 
                x=col_name, 
                marginal="box",
                color_discrete_sequence=['#6366f1'],
                opacity=0.85
            )
            fig.update_layout(
                title=dict(text=f"Numeric Distribution: {col_name}", font=dict(size=12, color='#a5b4fc')),
                xaxis_title=col_name,
                yaxis_title="Frequency",
                **layout
            )
            return json.loads(pio.to_json(fig))
    except Exception as e:
        print(f"[html_report_generator] Error building chart for '{col_name}': {e}")
        fig = go.Figure()
        fig.add_annotation(
            text=f"Chart Error: {str(e)}",
            xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False,
            font=dict(size=12, color="#ef4444")
        )
        fig.update_layout(**layout)
        return json.loads(pio.to_json(fig))


def generate_html_report(workspace_dir: str = "./sandbox_run", output_path: Optional[str] = None, data_path: Optional[str] = None) -> str:
    """
    Main entry point to read workspace metadata and generate a complete self-contained HTML profile report.
    """
    # 1. Load Profile metadata
    profile_path = os.path.join(workspace_dir, "metadata_profile.json")
    if os.path.exists(profile_path):
        with open(profile_path, "r", encoding="utf-8") as f:
            profile = json.load(f)
    else:
        profile = {}

    # 2. Load Canonical Metrics
    metrics_path = os.path.join(workspace_dir, "metrics.json")
    if os.path.exists(metrics_path):
        with open(metrics_path, "r", encoding="utf-8") as f:
            metrics = json.load(f)
    else:
        metrics = {}

    # 3. Load Agent Execution Trajectory
    plan_log_path = os.path.join(workspace_dir, "agent_plan_log.json")
    if os.path.exists(plan_log_path):
        with open(plan_log_path, "r", encoding="utf-8") as f:
            agent_plan_log = json.load(f)
    else:
        agent_plan_log = []

    # Parse agent trajectory
    agent_trajectory = []
    for entry in agent_plan_log:
        loop_num = entry.get("loop", 1)
        plan_items = entry.get("plan", [])
        llm_out = entry.get("llm_output", "")
        step_res = entry.get("step_results", [])
        
        for idx, step in enumerate(plan_items):
            res_str = step_res[idx] if idx < len(step_res) else "Executed"
            agent_trajectory.append({
                "loop": loop_num,
                "tool": step.get("tool", "tool_call"),
                "args_json": json.dumps(step.get("args", {})),
                "rationale": llm_out[:300] if idx == 0 else None,
                "result": res_str,
                "status": "Success" if "Error" not in str(res_str) else "Warning"
            })

    # 4. Load Dataset DataFrame if available for charts & correlations
    df = None
    if data_path and os.path.exists(data_path):
        try:
            df = pd.read_csv(data_path)
        except Exception:
            pass

    if df is None:
        curr_df_path = os.path.join(workspace_dir, "current_df.csv")
        if os.path.exists(curr_df_path):
            try:
                df = pd.read_csv(curr_df_path)
            except Exception:
                pass

    if df is None and os.path.exists(workspace_dir):
        data_files = [f for f in os.listdir(workspace_dir) if f.endswith(".csv")]
        if data_files:
            try:
                df = pd.read_csv(os.path.join(workspace_dir, data_files[0]))
            except Exception:
                pass

    # Basic stats
    dimensions = profile.get("dimensions", {"rows": 0, "columns": 0})
    columns_profile = profile.get("columns", [])
    dataset_name = profile.get("dataset_name") or os.path.basename(os.path.abspath(workspace_dir))
    target_column = (metrics.get("dataset_overview") or {}).get("target_column")

    # Missing counts
    total_missing = sum(col.get("missing_count", 0) for col in columns_profile)
    total_cells = dimensions.get("rows", 1) * dimensions.get("columns", 1)
    missing_pct = round((total_missing / total_cells) * 100, 2) if total_cells > 0 else 0.0

    # Dtypes breakdown chart
    dtype_counts = pd.Series([col.get("dtype", "unknown") for col in columns_profile]).value_counts()
    fig_dtype = px.pie(
        names=dtype_counts.index, 
        values=dtype_counts.values,
        hole=0.4,
        color_discrete_sequence=['#6366f1', '#a855f7', '#06b6d4', '#10b981', '#f59e0b']
    )
    fig_dtype.update_layout(
        margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#cbd5e1', family='Plus Jakarta Sans')
    )
    dtype_chart_json = pio.to_json(fig_dtype)

    # Missing chart
    missing_series = pd.Series({col.get("column"): col.get("missing_pct", 0) for col in columns_profile if col.get("missing_pct", 0) > 0})
    if not missing_series.empty:
        fig_miss = px.bar(
            x=missing_series.index, 
            y=missing_series.values,
            color_discrete_sequence=['#ef4444'],
            labels={'x': 'Feature', 'y': 'Missing %'}
        )
        fig_miss.update_layout(
            margin=dict(l=10, r=10, t=10, b=10),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#cbd5e1', family='Plus Jakarta Sans')
        )
        missing_chart_json = pio.to_json(fig_miss)
    else:
        missing_chart_json = "{}"

    # 5. Correlation Matrix Calculation (Robust for all datasets including categorical/mixed)
    corr_matrix = None
    corr_chart_json = "{}"
    
    if df is not None and df.shape[1] >= 2:
        df_corr = df.copy()
        # Convert non-numeric/object/categorical features via factorize
        for c in df_corr.columns:
            if not pd.api.types.is_numeric_dtype(df_corr[c]) or pd.api.types.is_bool_dtype(df_corr[c]):
                df_corr[c] = pd.factorize(df_corr[c])[0]
        
        valid_cols = [c for c in df_corr.columns if df_corr[c].nunique() > 1]
        if len(valid_cols) >= 2:
            corr_df = df_corr[valid_cols].corr().fillna(0).round(2)
            corr_matrix = corr_df
            
            cols = corr_df.columns.tolist()
            z_vals = corr_df.values.tolist()
            text_vals = [[f"{val:.2f}" for val in row] for row in z_vals]
            
            fig_corr = go.Figure(data=go.Heatmap(
                z=z_vals,
                x=cols,
                y=cols,
                text=text_vals,
                texttemplate="%{text}",
                textfont={"size": 11, "color": "#ffffff"},
                colorscale="Viridis",
                colorbar=dict(title="Correlation"),
                zmin=-1,
                zmax=1
            ))
            
            chart_height = max(500, len(cols) * 45)
            fig_corr.update_layout(
                height=chart_height,
                margin=dict(l=90, r=40, t=40, b=90),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#cbd5e1', family='Plus Jakarta Sans'),
                xaxis=dict(tickangle=-45, showgrid=False),
                yaxis=dict(autorange='reversed', showgrid=False)
            )
            corr_chart_json = pio.to_json(fig_corr)

    # Compute Alerts
    alerts = compute_alerts(profile, corr_matrix)

    # Build per-variable charts (guaranteed for every column in columns_profile)
    var_charts_json = {}
    for idx, col in enumerate(columns_profile, start=1):
        col_name = col.get("column")
        chart_spec = build_variable_chart(df, col_name, col_info=col)
        if chart_spec:
            var_charts_json[f"var-chart-{idx}"] = chart_spec

    # Image Artifacts (b64 encoded)
    visual_artifacts = []
    for entry in sorted(os.listdir(workspace_dir)):
        if entry.lower().endswith((".png", ".jpg", ".jpeg", ".svg")):
            img_path = os.path.join(workspace_dir, entry)
            try:
                with open(img_path, "rb") as f:
                    b64_str = base64.b64encode(f.read()).decode("utf-8")
                visual_artifacts.append({"name": entry, "b64": b64_str})
            except Exception:
                pass

    # Render Template
    tmpl = Template(HTML_TEMPLATE)
    html_content = tmpl.render(
        dataset_name=dataset_name,
        generation_time=pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
        dimensions=dimensions,
        target_column=target_column,
        total_missing=total_missing,
        missing_pct=missing_pct,
        alerts=alerts,
        columns_profile=columns_profile,
        dtype_chart_json=dtype_chart_json,
        missing_chart_json=missing_chart_json,
        corr_chart_json=corr_chart_json,
        var_charts_json=json.dumps(var_charts_json),
        agent_trajectory=agent_trajectory,
        visual_artifacts=visual_artifacts,
        categorical_associations=metrics.get("categorical_associations", {}).get("top_correlations", []),
        engineered_features=metrics.get("engineered_features", []),
        predictive_blueprint=metrics.get("predictive_modeling_blueprint", {})
    )

    if not output_path:
        output_path = os.path.join(workspace_dir, "eda_report.html")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    print(f"[html_report_generator] Interactive report written to: {os.path.abspath(output_path)}")
    return html_content


if __name__ == "__main__":
    import sys
    target_dir = sys.argv[1] if len(sys.argv) > 1 else "./sandbox_run"
    if os.path.exists(target_dir):
        generate_html_report(target_dir)
    else:
        print(f"Target directory '{target_dir}' does not exist.")

import os
import json
import re
import base64
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from jinja2 import Template
from typing import Dict, Any, List, Optional
from .profiler import is_non_distributional_column

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
            position: relative;
            background: var(--bg-card);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            padding: 1.5rem;
            margin-bottom: 1.5rem;
        }

        .system-badge {
            display: inline-block;
            font-size: 0.68rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            padding: 3px 8px;
            border-radius: 4px;
            background: rgba(99, 102, 241, 0.15);
            color: #818cf8;
            border: 1px solid rgba(99, 102, 241, 0.3);
            margin-bottom: 12px;
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

        /* Executive Summary Section */
        .summary-box {
            background: var(--bg-card);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            padding: 2rem;
            margin-bottom: 1.5rem;
            line-height: 1.7;
        }
        .summary-h1 { font-size: 1.45rem; font-weight: 800; margin-top: 1.2rem; margin-bottom: 1rem; color: var(--text-main); border-bottom: 1px solid var(--border-color); padding-bottom: 0.5rem; }
        .summary-h2 { font-size: 1.2rem; font-weight: 700; margin-top: 1.4rem; margin-bottom: 0.75rem; color: var(--text-main); }
        .summary-h3 { font-size: 1.05rem; font-weight: 600; margin-top: 1.1rem; margin-bottom: 0.5rem; color: var(--text-main); }
        .summary-h4 { font-size: 0.95rem; font-weight: 600; margin-top: 0.9rem; margin-bottom: 0.4rem; color: var(--text-muted); }
        .summary-list { margin-left: 1.5rem; margin-bottom: 1rem; color: var(--text-main); }
        .summary-list li { margin-bottom: 0.4rem; font-size: 0.9rem; }
        .summary-box p { margin-bottom: 1rem; color: var(--text-main); font-size: 0.92rem; }
        .summary-table { width: 100%; border-collapse: collapse; margin-bottom: 1rem; font-size: 0.85rem; }
        .summary-table th, .summary-table td { padding: 8px 12px; border: 1px solid var(--border-color); text-align: left; }
        .summary-table th { background: var(--bg-card-hover); font-weight: 700; }
        .summary-code { background: var(--bg-card-hover); padding: 1rem; border-radius: 8px; overflow-x: auto; margin-bottom: 1rem; }
        .summary-inline-code { background: var(--bg-card-hover); padding: 2px 6px; border-radius: 4px; font-family: monospace; font-size: 0.85rem; }
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
            <button class="tab-btn active" onclick="switchTab('summary')">Executive Summary</button>
            <button class="tab-btn" onclick="switchTab('overview')">Overview & Data Quality</button>
            <button class="tab-btn" onclick="switchTab('reasoning')">Agent Plan & Execution Reasoning</button>
            <button class="tab-btn" onclick="switchTab('variables')">Variable Profiles</button>
            <button class="tab-btn" onclick="switchTab('interactions')">Bivariate Relationships</button>
            <button class="tab-btn" onclick="switchTab('correlations')">Correlation Analysis</button>
            <button class="tab-btn" onclick="switchTab('features')">Derived Domain Metrics</button>
            <button class="tab-btn" onclick="switchTab('blueprint')">Predictive Modeling Strategy</button>
        </nav>

        <!-- TAB 1: EXECUTIVE SUMMARY -->
        <div id="summary" class="tab-content active">
            <div class="summary-box">
                <div class="section-title" style="font-size: 1.2rem; margin-bottom: 1.25rem;">
                    Executive Summary & Synthesis Report
                </div>
                {{ executive_summary_html|safe }}
            </div>
        </div>

        <!-- TAB 2: OVERVIEW -->
        <div id="overview" class="tab-content">
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

        <!-- TAB 3: AGENT REASONING -->
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
                            <div style="font-size: 0.82rem; color: var(--text-muted); margin-bottom: 6px; font-style: italic;">
                                "{{ step.rationale }}"
                            </div>
                            {% endif %}
                            <div class="step-args">Args: {{ step.args_json }}</div>
                            <div style="font-size: 0.8rem; color: var(--text-main); margin-top: 6px; background: var(--bg-card-hover); padding: 8px 10px; border-radius: 6px;">
                                Result: {{ step.result }}
                            </div>
                        </div>
                        {% endfor %}
                    {% else %}
                        <div style="font-size: 0.85rem; color: var(--text-muted);">No agent execution trajectory recorded.</div>
                    {% endif %}
                </div>
            </div>
        </div>

        <!-- TAB 4: VARIABLE PROFILES -->
        <div id="variables" class="tab-content">
            <input type="text" class="var-search" id="varSearchInput" placeholder="Filter variables by name or property..." onkeyup="filterVariables()">
            <div id="variablesContainer">
                {% for col in columns_profile %}
                <div class="var-card" data-var-name="{{ col.column }}">
                    <div class="var-card-header">
                        <div class="var-title">
                            <h3>{{ col.column }}</h3>
                            <span class="var-type-badge">{{ col.dtype }}</span>
                        </div>
                        <div class="var-meta">
                            <span>Missing: {{ col.missing_count }} ({{ col.missing_pct }}%)</span>
                            <span>Cardinality: {{ col.cardinality }} ({{ col.unique_pct }}%)</span>
                        </div>
                    </div>
                    {% if col.mean is not none %}
                    <div class="grid-6" style="margin-bottom: 1rem;">
                        <div style="background: var(--bg-card-hover); padding: 8px; border-radius: 6px; font-size: 0.8rem;">Mean: <strong>{{ col.mean }}</strong></div>
                        <div style="background: var(--bg-card-hover); padding: 8px; border-radius: 6px; font-size: 0.8rem;">Median: <strong>{{ col.median }}</strong></div>
                        <div style="background: var(--bg-card-hover); padding: 8px; border-radius: 6px; font-size: 0.8rem;">Std: <strong>{{ col.std }}</strong></div>
                        <div style="background: var(--bg-card-hover); padding: 8px; border-radius: 6px; font-size: 0.8rem;">Skew: <strong>{{ col.skew }}</strong></div>
                        <div style="background: var(--bg-card-hover); padding: 8px; border-radius: 6px; font-size: 0.8rem;">IQR: <strong>{{ col.iqr }}</strong></div>
                        <div style="background: var(--bg-card-hover); padding: 8px; border-radius: 6px; font-size: 0.8rem;">Kurt: <strong>{{ col.kurtosis }}</strong></div>
                    </div>
                    {% endif %}
                    <div id="var-chart-{{ loop.index }}" class="var-chart"></div>
                </div>
                {% endfor %}
            </div>
        </div>

        <!-- TAB 4: INTERACTIONS & BIVARIATE -->
        <div id="interactions" class="tab-content">
            <div style="display: flex; gap: 8px; margin-bottom: 1.5rem; background: var(--bg-card); padding: 6px; border-radius: 10px; border: 1px solid var(--border-color); flex-wrap: wrap;">
                <button class="report-sub-btn active" onclick="switchSubReportTab('report-sub-target', this)">Target Interaction</button>
                <button class="report-sub-btn" onclick="switchSubReportTab('report-sub-biv', this)">Bivariate Relationships</button>
                <button class="report-sub-btn" onclick="switchSubReportTab('report-sub-pair', this)">Pairplot Matrix</button>
                {% if visual_artifacts %}
                <button class="report-sub-btn" onclick="switchSubReportTab('report-sub-artifacts', this)">Generated Visualizations</button>
                {% endif %}
            </div>

            <!-- Target Interaction Sub-Tab -->
            <div id="report-sub-target" class="report-sub-tab">
                <div class="metric-card" style="padding: 1.5rem;">
                    <div class="section-title">Target Interaction Analysis</div>
                    <div id="report-plotly-target" style="width: 100%; min-height: 380px;"></div>
                </div>
            </div>

            <!-- Bivariate Relationships Sub-Tab -->
            <div id="report-sub-biv" class="report-sub-tab" style="display: none;">
                <div id="report-plotly-biv" class="img-grid"></div>
            </div>

            <!-- Pairplot Matrix Sub-Tab -->
            <div id="report-sub-pair" class="report-sub-tab" style="display: none;">
                <div class="metric-card" style="padding: 1.5rem;">
                    <div class="section-title">Pairplot Relationship Matrix</div>
                    <div id="report-plotly-pair" style="width: 100%; min-height: 400px;"></div>
                </div>
            </div>

            {% if visual_artifacts %}
            <!-- Static Image Artifacts Sub-Tab -->
            <div id="report-sub-artifacts" class="report-sub-tab" style="display: none;">
                <div class="img-grid">
                    {% for img in visual_artifacts %}
                    <div class="img-card">
                        <div style="font-weight: 700; margin-bottom: 8px; font-family: 'JetBrains Mono', monospace; font-size: 0.85rem;">
                            {{ img.name }}
                        </div>
                        <img src="data:image/png;base64,{{ img.b64 }}" alt="{{ img.name }}">
                        <div class="img-caption">Artifact generated by AutoEDA tool pipeline.</div>
                    </div>
                    {% endfor %}
                </div>
            </div>
            {% endif %}
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
                <div class="section-title">Categorical Associations (Cram├⌐r's V)</div>
                <table class="stats-table" style="font-size: 0.85rem;">
                    <thead>
                        <tr style="color: var(--text-muted); text-align: left;">
                            <th>Feature 1</th>
                            <th>Feature 2</th>
                            <th>Cram├⌐r's V</th>
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

        <!-- TAB 7: DERIVED DOMAIN METRICS -->
        <div id="features" class="tab-content">
            <div class="blueprint-box">
                <div class="section-title">Derived Domain Attributes & Composite Metrics</div>
                {% if engineered_features %}
                    {% for feat in engineered_features %}
                    <div class="blueprint-item">
                        <div class="blueprint-key">Derived Metric: <code>{{ feat.feature_name }}</code></div>
                        <div class="blueprint-val"><strong>Formula / Method:</strong> <code>{{ feat.formula or feat.method or 'Custom' }}</code></div>
                        <div class="blueprint-val"><strong>Stated Rationale:</strong> {{ feat.rationale or feat.purpose or 'Enhance predictive signal' }}</div>
                    </div>
                    {% endfor %}
                {% else %}
                    <div class="blueprint-val">No custom derived domain metrics synthesized during this run.</div>
                {% endif %}
            </div>
        </div>

        <!-- TAB 7: PREDICTIVE BLUEPRINT -->
        <div id="blueprint" class="tab-content">
            <div class="blueprint-box">
                <div class="system-badge">Automated Pipeline Asset</div>
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

    <script id="eda-metrics-data" type="application/json">{{ metrics_json|safe }}</script>

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

        function switchSubReportTab(subId, btn) {
            document.querySelectorAll('.report-sub-btn').forEach(b => b.classList.remove('active'));
            document.querySelectorAll('.report-sub-tab').forEach(t => t.style.display = 'none');
            if (btn) btn.classList.add('active');
            const target = document.getElementById(subId);
            if (target) {
                target.style.display = 'block';
                setTimeout(() => {
                    window.dispatchEvent(new Event('resize'));
                    target.querySelectorAll('.js-plotly-plot').forEach(plotDiv => {
                        Plotly.Plots.resize(plotDiv);
                    });
                }, 50);
            }
        }

        function filterVariables() {
            const query = document.getElementById('varSearchInput').value.toLowerCase();
            document.querySelectorAll('.var-card').forEach(card => {
                const name = card.getAttribute('data-var-name').toLowerCase();
                if (name.includes(query)) {
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
                if (chartSpec && chartSpec.data) {
                    Plotly.newPlot(divId, chartSpec.data, chartSpec.layout, {responsive: true, displayModeBar: false});
                }
            }

            const metricsEl = document.getElementById("eda-metrics-data");
            const metricsData = metricsEl ? JSON.parse(metricsEl.textContent || "{}") : {};
            const isLightMode = (document.documentElement.getAttribute('data-theme') === 'light');
            const pFontColor = isLightMode ? '#09090b' : '#fafafa';
            const pPaperBg = 'rgba(0,0,0,0)';

            // Render Target Interaction
            const targetDiv = document.getElementById("report-plotly-target");
            if (targetDiv && metricsData.target_interaction_data) {
                const tData = metricsData.target_interaction_data;
                let traces = [], layout = {};

                if (tData.grouped_counts) {
                    const xCats = Object.keys(tData.grouped_counts);
                    const yCatsSet = new Set();
                    xCats.forEach(xVal => Object.keys(tData.grouped_counts[xVal] || {}).forEach(y => yCatsSet.add(y)));
                    const yCats = Array.from(yCatsSet);
                    const palette = ["#6366f1", "#10b981", "#f59e0b", "#ef4444", "#a855f7", "#06b6d4"];

                    traces = yCats.map((yCat, i) => ({
                        x: xCats,
                        y: xCats.map(xCat => (tData.grouped_counts[xCat] ? tData.grouped_counts[xCat][yCat] || 0 : 0)),
                        name: `${tData.target_col}: ${yCat}`,
                        type: 'bar',
                        marker: { color: palette[i % palette.length] }
                    }));
                    layout = {
                        title: `Target Breakdown: ${tData.target_col} across ${tData.feature_col}`,
                        barmode: 'group',
                        margin: { l: 50, r: 20, t: 40, b: 60 },
                        paper_bgcolor: pPaperBg,
                        plot_bgcolor: pPaperBg,
                        autosize: true,
                        font: { color: pFontColor, family: 'Plus Jakarta Sans', size: 12 },
                        xaxis: { title: tData.feature_col, tickangle: -45 },
                        yaxis: { title: 'Count', showgrid: true, gridcolor: '#27272a' }
                    };
                } else if (tData.groups) {
                    const cats = Object.keys(tData.groups);
                    const medians = cats.map(k => tData.groups[k].median);
                    traces = [{
                        x: cats,
                        y: medians,
                        type: "bar",
                        marker: { color: "#f59e0b", opacity: 0.85 },
                        name: `${tData.feature_col} Median`
                    }];
                    layout = {
                        title: `Segmented Median: ${tData.feature_col} across ${tData.target_col}`,
                        margin: { l: 50, r: 20, t: 40, b: 60 },
                        paper_bgcolor: pPaperBg,
                        plot_bgcolor: pPaperBg,
                        autosize: true,
                        font: { color: pFontColor, family: 'Plus Jakarta Sans', size: 12 },
                        xaxis: { title: tData.target_col, tickangle: -45 },
                        yaxis: { title: tData.feature_col, showgrid: true, gridcolor: '#27272a' }
                    };
                } else if (tData.points) {
                    traces = [{
                        x: tData.points.map(p => p.x),
                        y: tData.points.map(p => p.y),
                        mode: "markers",
                        type: "scatter",
                        marker: { color: "#f59e0b", size: 6, opacity: 0.75 }
                    }];
                    layout = {
                        title: `${tData.feature_col} vs ${tData.target_col}`,
                        margin: { l: 50, r: 20, t: 40, b: 60 },
                        paper_bgcolor: pPaperBg,
                        plot_bgcolor: pPaperBg,
                        autosize: true,
                        font: { color: pFontColor },
                        xaxis: { title: tData.feature_col },
                        yaxis: { title: tData.target_col }
                    };
                }
                if (traces.length) Plotly.newPlot("report-plotly-target", traces, layout, { responsive: true });
            }

            // Render Bivariate Union Relationships
            const bivDiv = document.getElementById("report-plotly-biv");
            const bivUnion = (metricsData.bivariate_union && metricsData.bivariate_union.union_pairs) ? metricsData.bivariate_union.union_pairs : (metricsData.bivariate_data || []);
            if (bivDiv && bivUnion && bivUnion.length > 0) {
                bivDiv.innerHTML = "";
                bivUnion.forEach((biv, idx) => {
                    const card = document.createElement("div");
                    card.className = "img-card";
                    card.style.background = isLightMode ? '#ffffff' : '#121215';
                    card.style.border = isLightMode ? '1px solid #e4e4e7' : '1px solid #27272a';
                    card.style.borderRadius = "12px";
                    card.style.padding = "16px";

                    const f1 = biv.feature_1 || biv.x || 'Feature 1';
                    const f2 = biv.feature_2 || biv.y || 'Feature 2';
                    const src = biv.source || 'algorithmic';
                    let badgeBg = '#818cf8', badgeText = 'Algorithmic';
                    if (src === 'llm_inferred') {
                        badgeBg = '#34d399'; badgeText = 'LLM Inferred';
                    } else if (src === 'both') {
                        badgeBg = '#fbbf24'; badgeText = 'Both (Consensus)';
                    }

                    const chartDivId = `report-biv-chart-${idx}`;
                    card.innerHTML = `
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                            <div style="font-weight: 700; font-size: 0.95rem; color: var(--text-main);">${f1} vs ${f2}</div>
                            <span class="system-badge" style="background: ${badgeBg}22; color: ${badgeBg}; border-color: ${badgeBg}55; margin-bottom: 0;">${badgeText}</span>
                        </div>
                        <div style="font-size: 0.82rem; color: var(--text-muted); margin-bottom: 12px;">${biv.rationale || 'Bivariate interaction'}</div>
                        <div id="${chartDivId}" style="width:100%; height:260px;"></div>
                    `;
                    bivDiv.appendChild(card);

                    let traces = [], layout = {};
                    if (biv.grouped_counts) {
                        const xCats = Object.keys(biv.grouped_counts);
                        const yCatsSet = new Set();
                        xCats.forEach(xVal => Object.keys(biv.grouped_counts[xVal] || {}).forEach(y => yCatsSet.add(y)));
                        const yCats = Array.from(yCatsSet);
                        const palette = ["#6366f1", "#10b981", "#f59e0b", "#ef4444", "#a855f7", "#06b6d4"];

                        traces = yCats.map((yCat, i) => ({
                            x: xCats,
                            y: xCats.map(xCat => (biv.grouped_counts[xCat] ? biv.grouped_counts[xCat][yCat] || 0 : 0)),
                            name: `${f2}: ${yCat}`,
                            type: 'bar',
                            marker: { color: palette[i % palette.length] }
                        }));
                        layout = {
                            barmode: 'group',
                            margin: { l: 45, r: 20, t: 30, b: 60 },
                            paper_bgcolor: pPaperBg,
                            plot_bgcolor: pPaperBg,
                            autosize: true,
                            font: { color: pFontColor, family: 'Plus Jakarta Sans', size: 11 },
                            xaxis: { title: f1, tickangle: -45 },
                            yaxis: { title: 'Count', showgrid: true, gridcolor: '#27272a' }
                        };
                    } else if (biv.groups) {
                        const catLabels = Object.keys(biv.groups);
                        const medians = catLabels.map(k => biv.groups[k].median);
                        traces = [{
                            x: catLabels,
                            y: medians,
                            type: "bar",
                            marker: { color: "#10b981" }
                        }];
                        layout = {
                            margin: { l: 45, r: 20, t: 30, b: 60 },
                            paper_bgcolor: pPaperBg,
                            plot_bgcolor: pPaperBg,
                            autosize: true,
                            font: { color: pFontColor, family: 'Plus Jakarta Sans', size: 11 },
                            xaxis: { title: f1, tickangle: -45 },
                            yaxis: { title: f2, showgrid: true, gridcolor: '#27272a' }
                        };
                    } else if (biv.points) {
                        traces = [{
                            x: biv.points.map(p => p.x),
                            y: biv.points.map(p => p.y),
                            mode: "markers",
                            type: "scatter",
                            marker: { color: "#06b6d4", size: 6, opacity: 0.7 }
                        }];
                        layout = {
                            margin: { l: 45, r: 20, t: 30, b: 60 },
                            paper_bgcolor: pPaperBg,
                            plot_bgcolor: pPaperBg,
                            autosize: true,
                            font: { color: pFontColor, family: 'Plus Jakarta Sans', size: 11 },
                            xaxis: { title: f1, tickangle: -45, showgrid: false },
                            yaxis: { title: f2, showgrid: true, gridcolor: '#27272a' }
                        };
                    }
                    if (traces.length) Plotly.newPlot(chartDivId, traces, layout, { responsive: true, displayModeBar: false });
                });
            }

            // Render Pairplot Matrix
            const pairDiv = document.getElementById("report-plotly-pair");
            if (pairDiv && metricsData.pairplot_data && metricsData.pairplot_data.pairplot_matrix) {
                const pData = metricsData.pairplot_data;
                const features = pData.features_plotted || [];
                const matrix = pData.pairplot_matrix || [];

                if (features.length >= 2 && matrix.length > 0) {
                    const N = features.length;
                    pairDiv.innerHTML = "";

                    const gridDiv = document.createElement("div");
                    gridDiv.style.display = "grid";
                    gridDiv.style.gridTemplateColumns = `repeat(${N}, minmax(130px, 1fr))`;
                    gridDiv.style.gap = "8px";
                    gridDiv.style.width = "100%";
                    pairDiv.appendChild(gridDiv);

                    matrix.forEach((row, rIdx) => {
                        row.forEach((cell, cIdx) => {
                            const cellDiv = document.createElement("div");
                            const cellId = `report-pair-cell-${rIdx}-${cIdx}`;
                            cellDiv.id = cellId;
                            cellDiv.style.width = "100%";
                            cellDiv.style.height = "160px";
                            cellDiv.style.background = isLightMode ? '#f4f4f5' : '#18181b';
                            cellDiv.style.borderRadius = "6px";
                            cellDiv.style.padding = "4px";
                            gridDiv.appendChild(cellDiv);

                            let trace, layout;
                            if (cell.type === "diag") {
                                trace = {
                                    x: cell.bin_centers,
                                    y: cell.counts,
                                    type: "bar",
                                    marker: { color: "#6366f1" }
                                };
                                layout = {
                                    margin: { l: 20, r: 10, t: 25, b: 20 },
                                    paper_bgcolor: pPaperBg,
                                    plot_bgcolor: pPaperBg,
                                    font: { color: pFontColor, size: 9 },
                                    title: { text: cell.feature, font: { size: 10, color: pFontColor } },
                                    xaxis: { showticklabels: false, showgrid: false },
                                    yaxis: { showticklabels: false, showgrid: false }
                                };
                            } else {
                                trace = {
                                    x: (cell.points || []).map(p => p.x),
                                    y: (cell.points || []).map(p => p.y),
                                    mode: "markers",
                                    type: "scatter",
                                    marker: { color: "#06b6d4", size: 3, opacity: 0.6 }
                                };
                                layout = {
                                    margin: { l: 20, r: 10, t: 25, b: 20 },
                                    paper_bgcolor: pPaperBg,
                                    plot_bgcolor: pPaperBg,
                                    font: { color: pFontColor, size: 9 },
                                    title: { text: `${cell.y_feature} vs ${cell.x_feature}`, font: { size: 9, color: pFontColor } },
                                    xaxis: { showticklabels: false, showgrid: false },
                                    yaxis: { showticklabels: false, showgrid: false }
                                };
                            }
                            Plotly.newPlot(cellId, [trace], layout, { responsive: true, displayModeBar: false });
                        });
                    });
                }
            }

        });
    </script>
</body>
</html>
"""
class HTMLReportCompiler:
    """
    Classful HTML Report Compiler for synthesizing interactive Plotly reports,
    Jinja2 HTML rendering, data quality alerts, and markdown summary embedding.
    """
    def compute_alerts(self, profile: Dict[str, Any], corr_matrix: Optional[pd.DataFrame] = None) -> List[Dict[str, Any]]:
        """Detects data quality warnings: high missingness, high skewness, constant columns, high cardinality."""
        alerts = []
        columns = profile.get("columns", [])

        for col in columns:
            name = col.get("column", "Unknown")
            missing_pct = col.get("missing_pct", 0)
            cardinality = col.get("cardinality", 0)
            unique_pct = col.get("unique_pct", 0)
            skew = col.get("skew")
            dtype = col.get("dtype", "")

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

            if skew is not None and abs(skew) >= 1.5:
                alerts.append({
                    "column": name,
                    "type": "High Skewness",
                    "level": "notice",
                    "message": f"Highly skewed distribution with skewness coefficient = {skew:.2f}."
                })

            if cardinality == 1:
                alerts.append({
                    "column": name,
                    "type": "Constant Column",
                    "level": "warning",
                    "message": "Contains only 1 unique value across all rows (zero variance)."
                })

            if (unique_pct >= 25.0 and cardinality > 50 and not dtype.startswith("float")) or is_non_distributional_column(name):
                alerts.append({
                    "column": name,
                    "type": "High Cardinality",
                    "level": "notice",
                    "message": f"High cardinality feature ({cardinality} distinct values, {unique_pct}% unique). Excluded from ANOVA hypothesis testing."
                })

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

    def build_variable_chart(self, df: Optional[pd.DataFrame], col_name: str, col_info: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Generates an interactive Plotly figure spec dict for a single column's distribution."""
        layout_base = dict(
            margin=dict(l=30, r=20, t=30, b=30),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#cbd5e1', family='Plus Jakarta Sans')
        )

        try:
            if df is None or col_name not in df.columns:
                fig = go.Figure()
                fig.add_annotation(
                    text=f"No raw data loaded for '{col_name}'",
                    xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False,
                    font=dict(size=13, color="#94a3b8")
                )
                fig.update_layout(**layout_base)
                return json.loads(pio.to_json(fig))

            s = df[col_name].dropna()
            if len(s) == 0:
                fig = go.Figure()
                fig.add_annotation(
                    text=f"All values missing (NaN) for '{col_name}'",
                    xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False,
                    font=dict(size=13, color="#ef4444")
                )
                fig.update_layout(**layout_base)
                return json.loads(pio.to_json(fig))

            if is_non_distributional_column(col_name, df[col_name]):
                fig = go.Figure()
                fig.add_annotation(
                    text=f"Univariate distribution skipped for identifier / spatial coordinate '{col_name}'",
                    xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False,
                    font=dict(size=12, color="#94a3b8")
                )
                fig.update_layout(**layout_base)
                return json.loads(pio.to_json(fig))

            is_bool = pd.api.types.is_bool_dtype(s)
            is_num = pd.api.types.is_numeric_dtype(s) and not is_bool
            n_unique = s.nunique()

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
                    **layout_base
                )
                fig.update_xaxes(showgrid=True, gridcolor='rgba(255,255,255,0.05)')
                fig.update_yaxes(showgrid=True, gridcolor='rgba(255,255,255,0.05)')
                return json.loads(pio.to_json(fig))
            else:
                s_clean = s.replace([np.inf, -np.inf], np.nan).dropna()
                df_clean = s_clean.to_frame(name=col_name)
                fig = px.histogram(
                    df_clean,
                    x=col_name,
                    marginal="box",
                    color_discrete_sequence=['#6366f1'],
                    opacity=0.85
                )
                fig.update_layout(
                    title=dict(text=f"Numeric Distribution: {col_name}", font=dict(size=12, color='#a5b4fc')),
                    xaxis_title=col_name,
                    yaxis_title="Frequency",
                    **layout_base
                )
                fig.update_xaxes(showgrid=True, gridcolor='rgba(255,255,255,0.05)')
                fig.update_yaxes(showgrid=True, gridcolor='rgba(255,255,255,0.05)')
                return json.loads(pio.to_json(fig))
        except Exception as e:
            print(f"[html_report_generator] Error building chart for '{col_name}': {e}")
            fig = go.Figure()
            fig.add_annotation(
                text=f"Chart Error: {str(e)}",
                xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False,
                font=dict(size=12, color="#ef4444")
            )
            fig.update_layout(**layout_base)
            return json.loads(pio.to_json(fig))

    def render_markdown_to_html(self, md_text: str) -> str:
        """Converts Markdown into styled HTML for embedding in HTML profile report."""
        if not md_text or not md_text.strip():
            return "<p style='color: var(--text-muted); font-style: italic;'>No executive summary report available.</p>"

        import html
        lines = md_text.splitlines()
        html_lines = []
        in_list = False
        in_table = False
        in_code_block = False

        for line in lines:
            stripped = line.strip()

            if stripped.startswith("```"):
                if in_code_block:
                    html_lines.append("</code></pre>")
                    in_code_block = False
                else:
                    html_lines.append("<pre class='summary-code'><code>")
                    in_code_block = True
                continue

            if in_code_block:
                html_lines.append(html.escape(line))
                continue

            if in_list and not (stripped.startswith("- ") or stripped.startswith("* ") or re.match(r'^\d+\.\s', stripped)):
                html_lines.append("</ul>")
                in_list = False

            if in_table and not (stripped.startswith("|") and stripped.endswith("|")):
                html_lines.append("</tbody></table></div>")
                in_table = False

            if not stripped:
                continue

            if stripped.startswith("|") and stripped.endswith("|"):
                cells = [c.strip() for c in stripped.split("|")[1:-1]]
                if all(set(c).issubset({'-', ':', ' '}) for c in cells):
                    continue

                if not in_table:
                    html_lines.append("<div class='table-responsive'><table class='summary-table'><thead><tr>")
                    for cell in cells:
                        html_lines.append(f"<th>{html.escape(cell)}</th>")
                    html_lines.append("</tr></thead><tbody>")
                    in_table = True
                else:
                    html_lines.append("<tr>")
                    for cell in cells:
                        html_lines.append(f"<td>{html.escape(cell)}</td>")
                    html_lines.append("</tr>")
                continue

            if stripped.startswith("# "):
                html_lines.append(f"<h1 class='summary-h1'>{html.escape(stripped[2:])}</h1>")
            elif stripped.startswith("## "):
                html_lines.append(f"<h2 class='summary-h2'>{html.escape(stripped[3:])}</h2>")
            elif stripped.startswith("### "):
                html_lines.append(f"<h3 class='summary-h3'>{html.escape(stripped[4:])}</h3>")
            elif stripped.startswith("#### "):
                html_lines.append(f"<h4 class='summary-h4'>{html.escape(stripped[5:])}</h4>")
            elif stripped.startswith("- ") or stripped.startswith("* "):
                if not in_list:
                    html_lines.append("<ul class='summary-list'>")
                    in_list = True
                html_lines.append(f"<li>{html.escape(stripped[2:])}</li>")
            elif re.match(r'^\d+\.\s', stripped):
                if not in_list:
                    html_lines.append("<ol class='summary-list'>")
                    in_list = True
                txt = re.sub(r'^\d+\.\s', '', stripped)
                html_lines.append(f"<li>{html.escape(txt)}</li>")
            else:
                html_lines.append(f"<p>{html.escape(stripped)}</p>")

        if in_list:
            html_lines.append("</ul>")
        if in_table:
            html_lines.append("</tbody></table></div>")
        if in_code_block:
            html_lines.append("</code></pre>")

        res = "\n".join(html_lines)
        res = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', res)
        res = re.sub(r'\*(.*?)\*', r'<em>\1</em>', res)
        res = re.sub(r'`(.*?)`', r'<code class="summary-inline-code">\1</code>', res)
        return res

    def generate_html_report(self, workspace_dir: str = "./sandbox_run", output_path: Optional[str] = None, data_path: Optional[str] = None) -> str:
        """Main entry point to read workspace metadata and generate a complete self-contained HTML profile report."""
        summary_path = os.path.join(workspace_dir, "summary_report.md")
        executive_summary_md = ""
        if os.path.exists(summary_path):
            try:
                with open(summary_path, "r", encoding="utf-8") as f:
                    executive_summary_md = f.read()
            except Exception as e:
                print(f"[html_report_generator] Warning loading summary_report.md: {e}")

        executive_summary_html = self.render_markdown_to_html(executive_summary_md)

        profile_path = os.path.join(workspace_dir, "metadata_profile.json")
        if os.path.exists(profile_path):
            with open(profile_path, "r", encoding="utf-8") as f:
                profile = json.load(f)
        else:
            profile = {}

        metrics_path = os.path.join(workspace_dir, "metrics.json")
        if os.path.exists(metrics_path):
            with open(metrics_path, "r", encoding="utf-8") as f:
                metrics = json.load(f)
        else:
            metrics = {}

        plan_log_path = os.path.join(workspace_dir, "agent_plan_log.json")
        if os.path.exists(plan_log_path):
            with open(plan_log_path, "r", encoding="utf-8") as f:
                agent_plan_log = json.load(f)
        else:
            agent_plan_log = []

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

        dimensions = profile.get("dimensions") or (metrics.get("dataset_overview") or {}).get("raw_shape") or (metrics.get("dataset_overview") or {}).get("shape", {"rows": 0, "columns": 0})
        columns_profile = profile.get("columns", [])
        dataset_name = profile.get("dataset_name") or os.path.basename(os.path.abspath(workspace_dir))
        target_column = (metrics.get("dataset_overview") or {}).get("target_column")

        total_missing = sum(col.get("missing_count", 0) for col in columns_profile)
        total_cells = dimensions.get("rows", 1) * dimensions.get("columns", 1)
        missing_pct = round((total_missing / total_cells) * 100, 2) if total_cells > 0 else 0.0

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

        corr_matrix = None
        corr_chart_json = "{}"

        if df is not None and df.shape[1] >= 2:
            valid_feature_cols = [c for c in df.columns if not is_non_distributional_column(c, df[c])]
            df_corr = df[valid_feature_cols].copy() if valid_feature_cols else df.copy()

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

        alerts = self.compute_alerts(profile, corr_matrix)

        var_charts_json = {}
        for idx, col in enumerate(columns_profile, start=1):
            col_name = col.get("column")
            chart_spec = self.build_variable_chart(df, col_name, col_info=col)
            if chart_spec:
                var_charts_json[f"var-chart-{idx}"] = chart_spec

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

        tmpl = Template(HTML_TEMPLATE)
        raw_cat = metrics.get("categorical_associations", [])
        if isinstance(raw_cat, dict):
            cat_assoc_list = raw_cat.get("top_correlations", [])
        elif isinstance(raw_cat, list):
            cat_assoc_list = raw_cat
        else:
            cat_assoc_list = []

        html_content = tmpl.render(
            dataset_name=dataset_name,
            generation_time=pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
            dimensions=dimensions,
            executive_summary_html=executive_summary_html,
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
            categorical_associations=cat_assoc_list,
            engineered_features=metrics.get("engineered_features", []),
            predictive_blueprint=metrics.get("predictive_modeling_blueprint", {}),
            metrics_json=json.dumps(metrics, indent=2)
        )

        if not output_path:
            output_path = os.path.join(workspace_dir, "eda_report.html")

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html_content)

        print(f"[html_report_generator] Interactive report written to: {os.path.abspath(output_path)}")
        return html_content


default_html_compiler = HTMLReportCompiler()


def compute_alerts(profile: Dict[str, Any], corr_matrix: Optional[pd.DataFrame] = None) -> List[Dict[str, Any]]:
    return default_html_compiler.compute_alerts(profile, corr_matrix)

def build_variable_chart(df: Optional[pd.DataFrame], col_name: str, col_info: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    return default_html_compiler.build_variable_chart(df, col_name, col_info=col_info)

def render_markdown_to_html(md_text: str) -> str:
    return default_html_compiler.render_markdown_to_html(md_text)

def generate_html_report(workspace_dir: str = "./sandbox_run", output_path: Optional[str] = None, data_path: Optional[str] = None) -> str:
    return default_html_compiler.generate_html_report(workspace_dir=workspace_dir, output_path=output_path, data_path=data_path)


if __name__ == "__main__":
    import sys
    target_dir = sys.argv[1] if len(sys.argv) > 1 else "./sandbox_run"
    if os.path.exists(target_dir):
        generate_html_report(target_dir)
    else:
        print(f"Target directory '{target_dir}' does not exist.")

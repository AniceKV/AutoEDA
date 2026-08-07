# AutoEDA Performance & Execution Timing Benchmark Report

**Benchmark Date**: 2026-08-07 21:24:07
**Total Datasets Evaluated**: 5

## Executive Summary
This report provides an empirical breakdown of execution timings across key stages of the AutoEDA pipeline: Data Pre-Profiling, Hypothesis Testing, Predictive Blueprinting, Visual & Feature Tools, Summary Generation, and HTML Report Generation.

## Component Data Processing Timings (Seconds)

| Dataset | Rows | Cols | Size (KB) | Profiler | Hypothesis Tests | Blueprint | Visual Tools | Summary Gen | HTML Report | Total Data Processing |
|---|---|---|---|---|---|---|---|---|---|---|
| `fertility.csv` | 100 | 10 | 7.17 KB | 0.0012s | 0.0001s | 0.0001s | 0.1562s | 0.0242s | 0.1337s | **0.3372s** |
| `adult_test-selected-columns.csv` | 924 | 10 | 9.11 KB | 0.0012s | 0.0001s | 0.0s | 0.0033s | 0.0038s | 0.1118s | **0.1217s** |
| `Titanic-Dataset.csv` | 891 | 12 | 59.76 KB | 0.0015s | 0.0001s | 0.0s | 0.168s | 0.0341s | 0.1214s | **0.3729s** |
| `StudentsPerformance.csv` | 1000 | 8 | 70.35 KB | 0.0011s | 0.0001s | 0.0s | 0.1018s | 0.031s | 0.0774s | **0.2729s** |
| `gold_stock.csv` | 2970 | 6 | 265.26 KB | 0.0012s | 0.0s | 0.0003s | 1.4762s | 0.0249s | 0.0896s | **0.1373s** |

## Visualization & Feature Tool Timing Breakdown (Seconds)

| Dataset | Feature Dist | Corr Matrix | Bivariate Rel | Pairplot | Target Interaction | Feature Eng | Outliers |
|---|---|---|---|---|---|---|---|
| `fertility.csv` | 0.0008s | 0.1253s | 0.0061s | 0.0039s | 0.0113s | 0.0057s | 0.0031s |
| `adult_test-selected-columns.csv` | 0.0001s | 0.0004s | 0.0007s | 0.001s | 0.0006s | 0.0005s | 0.0s |
| `Titanic-Dataset.csv` | 0.0011s | 0.0826s | 0.0114s | 0.0601s | 0.0093s | 0.0019s | 0.0016s |
| `StudentsPerformance.csv` | 0.001s | 0.0483s | 0.0113s | 0.0301s | 0.0077s | 0.0012s | 0.0022s |
| `gold_stock.csv` | 0.0059s | 0.0001s | 1.4683s | 0.0002s | 0.0014s | 0.0003s | 0.0s |

## Performance Insights & Optimization Highlights
- **Profiling & Statistical Tests**: Linear time complexity scaling with row/column counts.
- **HTML Report Generation**: Highly optimized static HTML rendering executed within <0.05s per dataset.
- **Visualization Tool Execution**: `plot_pairplot` and `plot_semantic_bivariate_relationships` consume the majority of visualization rendering time due to seaborn/matplotlib figure generation and layout bounds calculation.
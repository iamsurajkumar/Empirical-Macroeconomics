# Empirical Macroeconomics

This repository serves as a collection of codes, notebooks, and replications focused on empirical macroeconomics, time series analysis, and machine learning applications in economics.

## Repository Overview

The codebase is organized into four main pillars:

### 1. Machine Learning & Transformers (`Machine_Learning/`)
Explores the intersection of modern machine learning and macroeconomics.
- **Key Concepts:** Double Descent, Causal Transformers, ML Benchmarks.
- **Files:**
  - `DoubleDescent.jl` & `run_experiments.jl`: Julia implementations for Double Descent experiments.
  - `nk_causal_transformer_irf_diagnostics.ipynb`: Analysis of Impulse Response Functions using Causal Transformers.
  - `double_descent_tutorial.tex`: Theoretical background and documentation.
  - **Setup:** See `INSTALLATION.md` and `Project.toml` within this directory for environment setup.

### 2. Time Series & Econometrics (`Time_Series/`)
Standard and advanced econometric implementations.
- **VARs**: Vector Autoregression models.
- **Local Projections**: Tools for estimating impulse responses via local projections.

### 3. Economic Papers & Replications (`Papers/`)
Code and data accompanying specific research papers and classic model replications.
- **Borovička & Shimer (2025)**: Replication materials.
- **Hopenhayn (1992)**: Industry dynamics model implementation.

### 4. General Analysis (`Analysis/`)
Ad-hoc analysis and exploratory notebooks.
- `stock_market_analysis.ipynb`: Analysis of stock market trends and data.

## Getting Started

Please navigate to the specific directories for relevant `README` or technical documentation.
- For Machine Learning experiments, refer to `Machine_Learning/INSTALLATION.md`.

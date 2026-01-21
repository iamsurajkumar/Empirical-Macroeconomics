# Standard Incomplete Markets Model

This directory contains a Python implementation of a standard heterogeneous agent incomplete markets model (Aiyagari-style).

## Overview

The `incomplete_markets_model.ipynb` notebook implements and solves a model where agents face uninsurable idiosyncratic earnings risk and can only self-insure through a risk-free asset.

### Key Features
- **Policy Functions**: Solves for optimal consumption and saving rules ($c(a,s)$ and $a'(a,s)$).
- **Stationary Distribution**: Computes the long-run distribution of assets across the population.
- **Comparative Statics**: Analyzes how aggregate assets ($A$) and the Marginal Propensity to Consume (MPC) vary with the discount factor ($\beta$).

## Figures

The notebook generates several key visualizations:
1. **Net Saving Policy**: Saving behavior as a function of current assets and income state.
2. **Consumption Policy**: Consumption functions.
3. **Aggregate Assets vs. $\beta$**: Relationship between patience and capital accumulation.
4. **MPC vs. $\beta$**: Relationship between patience and the average marginal propensity to consume.
5. **Asset Distribution**: The resulting cross-sectional distribution of wealth.

## Usage

To run the model, execute the Jupyter notebook:

```bash
jupyter notebook incomplete_markets_model.ipynb
```

## Requirements

- Python 3.x
- NumPy
- Matplotlib

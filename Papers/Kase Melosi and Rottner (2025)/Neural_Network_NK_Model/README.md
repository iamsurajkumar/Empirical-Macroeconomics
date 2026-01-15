# Neural Network Solutions for the New Keynesian DSGE Model

This folder contains the materials for the first proof of concept from the paper:
**Kase, Melosi, & Rottner (2025) - *Estimating Nonlinear Heterogeneous Agent Models with Neural Networks***

## Contents

1.  **NK Three Equation Model and Neural Network.ipynb**: The Jupyter Notebook containing the implementation of the neural network solution for the simple three-equation New Keynesian model.
2.  **model_simple_internal_100k.jld2**: Pre-trained model data required by the notebook.
3. **Figures**:
    - `Policy Function Comparison-Output Gap.pdf`
    - `Policy Function Comparison-Inflation.pdf`

## Instructions

- Ensure you have a Julia environment set up with the required packages (Lux.jl, etc.).
- Run the notebook `NK Three Equation Model and Neural Network.ipynb`.
- The notebook is configured to load the local `model_simple_internal_100k.jld2` file.

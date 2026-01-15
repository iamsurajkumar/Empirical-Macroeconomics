# HankNN.jl

Neural Network Solutions for DSGE Models following Kase, Melosi, Rottner (2025).

## Overview

This package implements neural network-based solutions for Dynamic Stochastic General Equilibrium (DSGE) models:

- **NK Model**: New Keynesian 3-equation model (linear)
- **RANK Model**: Representative Agent NK with Zero Lower Bound (nonlinear) [Future]
- **HANK Model**: Heterogeneous Agent NK with ZLB (highly nonlinear) [Future]

## Installation

```julia
# Navigate to the package directory
cd("path/to/Hank-NN-Claude")

# Activate the project
using Pkg
Pkg.activate(".")

# Install dependencies
Pkg.instantiate()

# Load the package
using HankNN
```

## Quick Start

See `examples/simple_nk_example.jl` for a complete working example.

### Basic Usage

```julia
using HankNN

# 1. Define parameters
par = NKParameters(
    β = 0.99, σ = 2.0, η = 1.0, ϕ = 50.0,
    ϕ_pi = 1.5, ϕ_y = 0.5, ρ = 0.9, σ_shock = 0.01
)

# 2. Define ranges for training
ranges = Ranges(
    ζ = (-0.1, 0.1),
    β = (0.95, 0.995),
    σ = (1.0, 3.0),
    # ... other parameters
)

# 3. Create shock configuration
shock_config = Shocks(σ=1.0, antithetic=true)

# 4. Build network
network, ps, st = make_network(par, ranges)

# 5. Train (simple version)
train_state = train_simple!(network, ps, st, ranges, shock_config;
                           num_epochs=1000, batch=100, mc=10, lr=0.001)

# 6. Simulate
results = simulate(network, train_state.parameters, train_state.states,
                  1, ranges, shock_config; num_steps=100)
```

## Package Structure

```
Hank-NN-Claude/
├── src/                         # Core package code
│   ├── HankNN.jl                # Main module
│   ├── 01-structs.jl            # Type definitions
│   ├── 02-economics-utils.jl    # Shared utilities
│   ├── 03-economics-nk.jl       # NK model functions
│   ├── 04-economics-rank.jl     # RANK model functions [Future]
│   ├── 06-deeplearning.jl       # Neural network training
│   ├── 07-particlefilter.jl     # Likelihood evaluation [User implements]
│   └── 08-plotting.jl           # Visualization
├── scripts/                     # Batch/automated scripts
│   ├── train_nk_model.jl        # NK training script
│   ├── train_rank_model.jl      # RANK training script [Future]
│   ├── particle_filter_workflow.jl  # PF pipeline [Future]
│   └── make_figures.jl          # Generate publication plots
├── notebooks/                   # Interactive exploration
│   ├── 01_nk_model_exploration.jl   # NK model testing
│   ├── 02_rank_model_exploration.jl # RANK model exploration [Future]
│   └── README.md                # Notebook usage guide
├── examples/
│   └── simple_nk_example.jl     # Quick start example
├── save/                        # Trained models and checkpoints
│   └── .gitkeep
├── figures/                     # Generated plots
│   └── .gitkeep
├── data/                        # Observed data for estimation
│   └── README.md
├── Project.toml                 # Package dependencies
├── README.md                    # This file
└── REFACTORING_SUMMARY.md      # Detailed change log
```

## Core Features

### ✓ Implemented (NK Model)

- **Type hierarchy**: Abstract types + multiple dispatch for extensibility
- **Parameter batching**: Train on multiple parameter sets simultaneously
- **Vectorized policy evaluation**: Efficient MC × batch operations
- **Training loops**: Simple and advanced (with LR scheduling)
- **Simulation**: Generate time series from trained networks
- **Steady state computation**: Analytical for NK model
- **Shock generation**: With antithetic variates option

### 🔨 User Implements

Functions marked with `[USER IMPLEMENTS]` are skeletons you should complete:

- `policy_analytical()` - Analytical NK policy (method of undetermined coefficients)
- `policy_over_par()` - Policy sensitivity to parameters
- Particle filter functions in `07-particlefilter.jl`

### 🚧 Future Work

- **RANK model**: ZLB constraint implementation
- **HANK model**: Three-network architecture, two-stage training
- Additional plotting functions

## Architecture

### Type System

```julia
# Abstract base type
abstract type AbstractModelParameters{T} end

# Concrete types
struct NKParameters{T} <: AbstractModelParameters{T}
    β::T; σ::T; η::T; ϕ::T
    ϕ_pi::T; ϕ_y::T; ρ::T; σ_shock::T
    κ::Union{T,Nothing}  # Derived
    ω::Union{T,Nothing}  # Derived
end

struct RANKParameters{T} <: AbstractModelParameters{T}
    # All NK fields + ZLB fields
    r_min::T; ϕ_r::T
end
```

### Multiple Dispatch

Generic functions work for all models via dispatch:

```julia
# Shared simulation function
simulate(network, ps, st, batch, ranges, shock_config)

# Model-specific implementations
policy(network, state, par::NKParameters, ps, st)    # NK version
policy(network, state, par::RANKParameters, ps, st)  # RANK version [Future]
```

## Training

### Simple Training

```julia
train_state = train_simple!(network, ps, st, ranges, shock_config;
                           num_epochs=1000,
                           batch=100,
                           mc=10,
                           lr=0.001)
```

### Advanced Training

```julia
train_state, loss_dict = train!(network, ps, st, ranges, shock_config;
                                num_epochs=10000,
                                batch=100,
                                mc=10,
                                lr=0.001,
                                internal=1,
                                print_after=100,
                                par_draw_after=100,
                                num_steps=1,
                                eta_min=1e-10)
```

Features:
- Cosine annealing learning rate schedule
- Parameter redraws during training
- State evolution steps
- Progress tracking
- Loss component tracking

## Plotting

```julia
# Training diagnostics
plot_avg_loss(loss_dict)
plot_loss_components(loss_dict)

# Prior visualization
priors = prior_distribution(ranges)
plot_beta(priors; n=5000)

# Policy comparisons [Requires policy_analytical]
plot_policy_comparison(network, ps, st, par, ranges)
```

## Mathematical Details

### NK Model Equations

**NKPC:**
```
π_t = κ X_t + β E_t[π_{t+1}]
```

**Euler Equation:**
```
X_t = E_t[X_{t+1}] - σ⁻¹(ϕ_π π_t + ϕ_y X_t - E_t[π_{t+1}] - ζ_t)
```

**Shock Process (AR(1)):**
```
ζ_{t+1} = ρ ζ_t + ε_{t+1} · σ_shock · σ · (ρ - 1) · ω
```

where `ε ~ N(0,1)`.

### Neural Network

**Architecture** (default):
- Input: [ζ; β; σ; η; ϕ; ϕ_π; ϕ_y; ρ; σ_ε] (9 dimensions)
- 5 hidden layers × 64 neurons (or 128 for Kase architecture)
- Activation: CELU (or SiLU + Leaky ReLU for Kase)
- Output: [X, π] scaled by factor (default: 1/100)
- Input normalization to [-1, 1]

**Loss Function:**
```
L = λ_X · ||res_X||² + λ_π · ||res_π||²
```

where residuals come from model equations.

## References

Kase, H., Melosi, L., & Rottner, M. (2025). "Estimating Nonlinear Heterogeneous Agent Models with Neural Networks." BIS Working Paper 1241.

## Development Status

- ✅ NK Model: Fully functional
- 🚧 RANK Model: Skeleton implemented, ZLB logic ready
- 📋 HANK Model: Planned (requires three-network architecture)
- 📊 Particle Filter: User implements
- 🎨 Plotting: Basic functions implemented

## Notes

- **Preserves original code**: All working functionality from `hank-nn-claude.jl` is retained
- **Modular design**: Easy to extend with new models
- **Type stability**: Parametric types enable compiler optimizations
- **Zero code duplication**: Generic functions + multiple dispatch

## Next Steps

1. Implement `policy_analytical()` for NK model validation
2. Complete RANK model implementation with ZLB
3. Implement particle filter functions
4. Add HANK model (three-network architecture)

---

**Package Version:** 0.1.0
**Julia Compatibility:** 1.10+
**Based on:** Kase et al. (2025) methodology

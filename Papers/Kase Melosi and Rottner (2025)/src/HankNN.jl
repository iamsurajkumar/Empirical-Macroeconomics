# ═══════════════════════════════════════════════════════════════════════════
# HankNN.jl - Main Module File
# Neural Network Solutions for DSGE Models
# Based on: Kase, Melosi, Rottner (2025)
# ═══════════════════════════════════════════════════════════════════════════

module HankNN

# ───────────────────────────────────────────────────────────────────────────
# Package Imports
# ───────────────────────────────────────────────────────────────────────────

# using ChainRulesCore
# using ComponentArrays
# using CUDA

# using Dates
using Distributions
using JLD2
using Lux
using Optimisers
using LinearAlgebra
using ProgressMeter
using Random
using Zygote

# For training
using Lux.Training: TrainState, single_train_step!
using Optimisers: Adam
using Zygote: gradient
using Lux: Chain, Dense

# Automatic differentiation
import Lux.Training: AutoZygote

# Math Mean
using Statistics

# Plotting
using Plots

# ───────────────────────────────────────────────────────────────────────────
# Include Source Files (in order)
# ───────────────────────────────────────────────────────────────────────────

include("01-structs.jl")
include("02-economics-utils.jl")
include("03-economics-nk.jl")
include("04-economics-rank.jl")
# include("05-economics-hank.jl")  # Future work
include("06-deeplearning.jl")
include("07-particlefilter.jl")
include("08-plotting.jl")

# ───────────────────────────────────────────────────────────────────────────
# Exports: Type Definitions
# ───────────────────────────────────────────────────────────────────────────

export AbstractModelParameters
export NKParameters, RANKParameters
export State, Ranges, Shocks
export TrainingConfig, LossWeights
export NormalizeLayer

# ───────────────────────────────────────────────────────────────────────────
# Exports: Core Economic Functions
# ───────────────────────────────────────────────────────────────────────────

export steady_state
export draw_parameters, prior_distribution
export initialize_state, draw_shocks
export expand
export generate_covariance_matrix

# Model-specific functions (multiple dispatch)
export policy, step, residuals

# Simulation functions
export sim_step, simulate

# NK-specific
export policy_analytical, policy_over_par, policy_over_par_list

# RANK-specific
export apply_zlb_schedule

# ───────────────────────────────────────────────────────────────────────────
# Exports: Neural Network Functions
# ───────────────────────────────────────────────────────────────────────────

export make_network, make_kase_network
export loss_fn, loss_fn_wrapper
export cosine_annealing_lr
export train!, train_simple!

# ───────────────────────────────────────────────────────────────────────────
# Exports: Particle Filter Functions
# ───────────────────────────────────────────────────────────────────────────

export ParticleFilter
export kitagawa_resample, log_prob
export standard_particle_filter!
export filter_dataset!, train_nn_particle_filter!
export log_likelihood_nn
export generate_synthetic_data

# ───────────────────────────────────────────────────────────────────────────
# Exports: Plotting Functions
# ───────────────────────────────────────────────────────────────────────────

export plot_avg_loss, plot_loss_components
export plot_policy_comparison
export plot_par_list
export plot_loss_likelihood
export plot_likelihood_single, plot_likelihood_parameters
export compute_likelihood_results
export plot_simulation
export plot_beta

# ───────────────────────────────────────────────────────────────────────────
# Exports: Utility Functions
# ───────────────────────────────────────────────────────────────────────────

export get_bounds


# Hello World
export helloworld

end # module HankNN

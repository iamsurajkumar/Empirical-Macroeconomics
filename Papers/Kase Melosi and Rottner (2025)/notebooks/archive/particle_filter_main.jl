# Particle Filter Implementation for NK Model
# Converted from notebook: 02-particle-filter-main.ipynb

# ============================================================================
# 1. PACKAGE IMPORTS
# ============================================================================

using Distributions
using JLD2
using Lux
using Optimisers
using LinearAlgebra
using ProgressMeter
using Random
using Zygote
using Lux.Training: TrainState, single_train_step!
using Optimisers: Adam
using Zygote: gradient
using Lux: Chain, Dense
import Lux.Training: AutoZygote
using Statistics
using Plots
using Dates

# ============================================================================
# 2. INCLUDE SOURCE FILES
# ============================================================================

include("../src/01-structs.jl")
include("../src/02-economics-utils.jl")
include("../src/03-economics-nk.jl")
include("../src/04-economics-rank.jl")
include("../src/06-deeplearning.jl")
include("../src/08-plotting.jl")

# ============================================================================
# 3. PARTICLE FILTER STRUCTURES
# ============================================================================

struct ParticleFilter{M <: AbstractModel}
    model::M
    data::Matrix{Float64}  # S × T (Observables × Time)
    R::Matrix{Float64}     # Observables covariance
    R_inv::Matrix{Float64}
    R_det::Float64
end

# Construction function
function ParticleFilter(model::M, data::Matrix{Float64}, R::Matrix{Float64}) where {M <: AbstractModel}
    R_inv = inv(R)
    R_det = det(R)
    return ParticleFilter{M}(model, data, R, R_inv, R_det)
end

# ============================================================================
# 4. PARTICLE FILTER HELPER FUNCTIONS
# ============================================================================

"""
    log_probability(error, R_inv, R_det)

Calculate log probability of observations given model predictions.

Assumes multivariate normal measurement error:
log p(y | ŷ) = -0.5 * [S*log(2π) + log|R| + (y-ŷ)'R⁻¹(y-ŷ)]

Args:
    error: Observation errors (data - model_prediction) [S × P]
    R_inv: Inverse of measurement error covariance
    R_det: Determinant of measurement error covariance

Returns:
    Log probability for each particle [P]
"""
function log_probability(error::Matrix{Float64}, R_inv::Matrix{Float64}, R_det::Float64)
    S = size(error, 1)  # Number of observables
    log_prob = -0.5 * (S * log(2π) .+ log(R_det) .+ sum(error .* (R_inv * error), dims=1))
    return vec(log_prob)
end

"""
    kitagawa_resample(w_norm, P; aux=0.4532)

Systematic resampling using Kitagawa method.

This is more efficient and has lower variance than multinomial resampling.
Instead of P independent draws, it uses a single random starting point
and evenly spaced samples through the cumulative distribution.

Args:
    w_norm: Normalized weights (sum to 1)
    P: Number of particles
    aux: Random starting point offset

Returns:
    Indices of resampled particles
"""
function kitagawa_resample(w_norm::Vector{Float64}, P::Int; aux::Float64=0.4532)
    cum_dist = cumsum(w_norm)
    u = range(aux, step=1.0, length=P) ./ P
    return searchsortedfirst.(Ref(cum_dist), u)
end

# ============================================================================
# 5. MAIN PARTICLE FILTER FUNCTION
# ============================================================================

"""
    filter(pf, P, shock_config; burn=1000, sim=100, par=nothing, seed=nothing)

Run particle filter on observed data.

Args:
    pf: ParticleFilter object containing model and data
    P: Number of particles
    shock_config: Shocks configuration
    burn: Burn-in periods before filtering starts
    sim: Number of periods to filter
    par: Model parameters (defaults to pf.model.parameters)
    seed: Random seed for reproducibility

Returns:
    log_likelihood: Total log likelihood
    filtered_out: NamedTuple with filtered estimates (R, X, π)
"""
function filter(pf::ParticleFilter, P::Int, shock_config::Shocks;
                burn=1000, sim=100, par=nothing, seed=nothing)

    if !isnothing(seed)
        Random.seed!(seed)
    end

    S = size(pf.data, 1)  # Number of observables

    # Use model parameters if not provided
    if isnothing(par)
        par = pf.model.parameters
    end

    # Initialize particles
    par_expanded = expand(par, P)
    ss = steady_state(par_expanded)
    state = initialize_state(par_expanded, P, ss)

    # Burn-in period
    for i in 1:burn
        shocks = draw_shocks(shock_config, 1, P)
        state = step(state, shocks, par_expanded, ss)
    end

    # Storage for results
    errors = Matrix{Float64}(undef, S, P)
    log_likelihood = 0.0
    filtered_R = Vector{Float64}(undef, sim)
    filtered_X = Vector{Float64}(undef, sim)
    filtered_π = Vector{Float64}(undef, sim)

    # Main filtering loop
    for t in 1:sim
        # PREDICT: Compute observables from current state
        sim_out = sim_step(pf.model.network, state, par_expanded, pf.model.ps, pf.model.st)
        pf.model.st = sim_out[:st]

        # UPDATE: Compute prediction errors (observed - predicted)
        errors[1, :] = pf.data[1, t] .- vec(sim_out[:R])
        errors[2, :] = pf.data[2, t] .- vec(sim_out[:X_t])
        errors[3, :] = pf.data[3, t] .- vec(sim_out[:π_t])

        # Compute log probabilities
        log_probs = log_probability(errors, pf.R_inv, pf.R_det)

        # Compute weights with numerical stability
        max_log = maximum(log_probs)
        w = exp.(log_probs .- max_log)
        log_likelihood += log(mean(w)) + max_log

        # Normalize weights
        w_norm = w ./ sum(w)

        # RESAMPLE: Systematic resampling
        idx = kitagawa_resample(w_norm, P)
        state = State(ζ = state.ζ[:, idx])

        # Store filtered estimates (weighted mean)
        filtered_R[t] = mean(sim_out[:R][idx])
        filtered_X[t] = mean(sim_out[:X_t][idx])
        filtered_π[t] = mean(sim_out[:π_t][idx])

        # PROPAGATE: Move state forward
        shocks = draw_shocks(shock_config, 1, P)
        state = step(state, shocks, par_expanded, ss)
    end

    filtered_out = (
        R = filtered_R,
        X = filtered_X,
        π = filtered_π
    )

    return log_likelihood, filtered_out
end

# ============================================================================
# 6. MAIN EXECUTION
# ============================================================================

println("="^70)
println("Particle Filter for NK Model")
println("="^70)

# Define baseline parameters
par = NKParameters(
    β=0.97, σ=2.0, η=1.125, ϕ=0.7,
    ϕ_pi=1.875, ϕ_y=0.25, ρ=0.875, σ_shock=0.06
)

# Parameter ranges for training
ranges = Ranges(
    ζ=(0.0, 1.0),
    β=(0.95, 0.99),
    σ=(1.0, 3.0),
    η=(0.25, 2.0),
    ϕ=(0.5, 0.9),
    ϕ_pi=(1.25, 2.5),
    ϕ_y=(0.0, 0.5),
    ρ=(0.8, 0.95),
    σ_shock=(0.02, 0.1)
)

# Shock configuration
shock_config = Shocks(σ=1.0, antithetic=false)
seed = 1234

println("✓ Parameters and ranges defined")

# Initialize and load trained model
model = NKModel(par, ranges; hidden=64, initial_ζ=0.0, N_states=1,
                N_outputs=2, activation=Lux.celu, scale_factor=1/100)
model = load_weights!(model, "model_simple_internal_100k.jld2")
println("✓ Model loaded")

# Simulate data
data = simulate_model(model, 1, shock_config; burn=1000, num_steps=1000, seed=seed)
println("✓ Data simulated (1000 time steps)")

# Generate measurement error covariance
R = generate_covariance_matrix(data)
noisy_data, R_cov = add_measurement_error(data; error_scale=0.1, seed=seed)
println("✓ Measurement error added")

# Create particle filter object
data_matrix = vcat(noisy_data.R', noisy_data.X', noisy_data.π')  # (3 × T)
R_matrix = Matrix(R_cov)
pf = ParticleFilter(model, data_matrix, R_matrix)
println("✓ Particle filter created")

# Run particle filter
println("\nRunning particle filter...")
ll, filtered = filter(pf, 100, shock_config; burn=100, sim=100, seed=seed)
println("Log-likelihood: $ll")

# ============================================================================
# 7. VISUALIZATION
# ============================================================================

println("\nGenerating plots...")

# Plot observed vs filtered estimates
p1 = plot(1:100, pf.data[1, 1:100], label="Observed R (noisy)",
          alpha=0.5, color=:blue, ylabel="R")
plot!(1:100, filtered.R, label="Filtered R", lw=2, color=:darkblue)

p2 = plot(1:100, pf.data[2, 1:100], label="Observed X (noisy)",
          alpha=0.5, color=:red, ylabel="X")
plot!(1:100, filtered.X, label="Filtered X", lw=2, color=:darkred)

p3 = plot(1:100, pf.data[3, 1:100], label="Observed π (noisy)",
          alpha=0.5, color=:green, ylabel="π")
plot!(1:100, filtered.π, label="Filtered π", lw=2, color=:darkgreen)

p_final = plot(p1, p2, p3, layout=(3, 1), size=(800, 900),
               plot_title="Particle Filter: Observed vs Filtered",
               margin=5Plots.mm)

display(p_final)
println("✓ Plots generated")

println("\n" * "="^70)
println("Particle filter completed successfully!")
println("="^70)

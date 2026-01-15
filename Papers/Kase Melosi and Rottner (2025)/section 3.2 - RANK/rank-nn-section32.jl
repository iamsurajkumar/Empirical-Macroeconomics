# ═══════════════════════════════════════════════════════════════════════════
# Neural Network Solution for RANK Model with Zero Lower Bound
# Based on: Kase, Melosi, Rottner (2025) - Section 3.2
# "Estimating Nonlinear Heterogeneous Agent Models with Neural Networks"
# ═══════════════════════════════════════════════════════════════════════════

# ═══════════════════════════════════════════════════════════════════════════
# PACKAGE IMPORTS
# ═══════════════════════════════════════════════════════════════════════════

using Statistics
using Lux
using Random
using Distributions
using StatsPlots
using ProgressMeter

# ═══════════════════════════════════════════════════════════════════════════
# CORE STRUCT DEFINITIONS
# ═══════════════════════════════════════════════════════════════════════════

"""
    RANKParameters{T}

Holds RANK model parameters. Type T can be:
- Real: single parameter set
- Vector{<:Real}: batch of parameter sets

Calibrated Parameters:
- β: Discount factor (0.9975)
- σ: Relative risk aversion (1)
- η: Inverse Frisch elasticity (1)
- ϵ: Price elasticity of demand (11)
- χ: Disutility of labor (0.91)
- A: Total factor productivity (1, normalized)

Estimated Parameters:
- θ_Π: Taylor rule inflation response
- θ_Y: Taylor rule output response
- φ: Rotemberg price adjustment cost
- ρ_ζ: Preference shock persistence
- σ_ζ: Preference shock std deviation

Steady State Parameters (given):
- Π_bar: Inflation target (1.005 quarterly = 2% annual)
- Y_bar: Output target (1, normalized)
- D: Government debt (calibrated externally)

Derived Steady State (computed):
- R_bar: Steady state nominal rate
- MC_bar: Steady state marginal cost
- N_bar: Steady state labor
"""
struct RANKParameters{T<:Union{Real, AbstractVector{<:Real}}}
    # Calibrated structural parameters
    β::T
    σ::T
    η::T
    ϵ::T
    χ::T
    A::T
    
    # Estimated parameters
    θ_Π::T
    θ_Y::T
    φ::T
    ρ_ζ::T
    σ_ζ::T
    
    # Steady state targets (given)
    Π_bar::T
    Y_bar::T
    D::T
    
    # Derived steady state (computed)
    R_bar::Union{T, Nothing}
    MC_bar::Union{T, Nothing}
    N_bar::Union{T, Nothing}
end

# Constructor: accepts only required parameters, steady state derived later
function RANKParameters(; β, σ, η, ϵ, χ, A, θ_Π, θ_Y, φ, ρ_ζ, σ_ζ, 
                         Π_bar, Y_bar, D, R_bar=nothing, MC_bar=nothing, N_bar=nothing)
    T = typeof(β)
    return RANKParameters{T}(β, σ, η, ϵ, χ, A, θ_Π, θ_Y, φ, ρ_ζ, σ_ζ, 
                             Π_bar, Y_bar, D, R_bar, MC_bar, N_bar)
end

"""
    expand(par::RANKParameters, batch_size::Int)

Expand scalar parameters to batch by replicating batch_size times.
"""
function expand(par::RANKParameters, batch_size::Int)
    return RANKParameters{Vector{Float64}}(
        fill(par.β, batch_size),
        fill(par.σ, batch_size),
        fill(par.η, batch_size),
        fill(par.ϵ, batch_size),
        fill(par.χ, batch_size),
        fill(par.A, batch_size),
        fill(par.θ_Π, batch_size),
        fill(par.θ_Y, batch_size),
        fill(par.φ, batch_size),
        fill(par.ρ_ζ, batch_size),
        fill(par.σ_ζ, batch_size),
        fill(par.Π_bar, batch_size),
        fill(par.Y_bar, batch_size),
        fill(par.D, batch_size),
        par.R_bar === nothing ? nothing : fill(par.R_bar, batch_size),
        par.MC_bar === nothing ? nothing : fill(par.MC_bar, batch_size),
        par.N_bar === nothing ? nothing : fill(par.N_bar, batch_size)
    )
end

"""
    State{T}

Holds state variables for RANK model. Type T can be:
- Real: single state
- Vector{<:Real}: batch of states
- Matrix{<:Real}: MC × batch

State variables:
- ζ: Preference shock (exp(ζ_t) enters utility)
"""
struct State{T<:Union{Real,AbstractArray{<:Real}}}
    ζ::T
end

# Constructor
function State(; ζ)
    T = typeof(ζ)
    return State{T}(ζ)
end

"""
    expand(state::State, batch_size::Int)

Expand scalar state to batch by replicating batch_size times.
"""
function expand(state::State, batch_size::Int)
    return State{Vector{Float64}}(
        fill(state.ζ, batch_size)
    )
end

"""
    Ranges

Stores prior bounds for each parameter (for normalization and sampling).
"""
struct Ranges
    ζ::Tuple{Float64, Float64}
    θ_Π::Tuple{Float64, Float64}
    θ_Y::Tuple{Float64, Float64}
    φ::Tuple{Float64, Float64}
    ρ_ζ::Tuple{Float64, Float64}
    σ_ζ::Tuple{Float64, Float64}
end

"""
    Shocks

Configuration for shock sampling.

Fields:
- σ: Standard deviation of shocks (equals σ_ζ from parameters)
- antithetic: Use antithetic variates for variance reduction
"""
struct Shocks
    σ::Float64
    antithetic::Bool
end

# Constructor with default antithetic=true (paper uses this)
function Shocks(; σ, antithetic=true)
    return Shocks(σ, antithetic)
end

"""
    NormalizeLayer <: Lux.AbstractLuxLayer

Custom Lux layer for min-max normalization to [-1, 1].

Mathematical operation:
    normalized = 2 * (x - lb) / (ub - lb) - 1
"""
struct NormalizeLayer <: Lux.AbstractLuxLayer
    lower_bound::Vector{Float64}
    upper_bound::Vector{Float64}
end

# Forward pass: x → (normalized_x, state)
function (layer::NormalizeLayer)(x, ps, st)
    normalized = 2 .* (x .- layer.lower_bound) ./ (layer.upper_bound .- layer.lower_bound) .- 1
    return normalized, st
end

# ═══════════════════════════════════════════════════════════════════════════
# CORE FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════

"""
    steady_state(par::RANKParameters)

Compute deterministic steady state (DSS) for RANK model.

Mathematical derivation:
    R̄ = Π̄ / β                           [From Euler equation]
    MC̄ = (ϵ - 1) / ϵ                    [From Phillips curve]
    N̄ = [MC̄ / (χ A^(σ-1))]^(1/(η+σ))  [From labor FOC]
    Ȳ = A × N̄                           [Production function]

Returns: RANKParameters with all steady state values computed
"""
function steady_state(par::RANKParameters)
    # Step 1: Nominal interest rate from Euler equation
    R_bar = par.Π_bar ./ par.β
    
    # Step 2: Marginal cost from zero-inflation Phillips curve
    MC_bar = (par.ϵ .- 1) ./ par.ϵ
    
    # Step 3: Labor from labor FOC
    N_bar = (MC_bar ./ (par.χ .* par.A .^ (par.σ .- 1))) .^ (1 ./ (par.η .+ par.σ))
    
    # Return parameters with steady state computed
    return RANKParameters(
        β = par.β,
        σ = par.σ,
        η = par.η,
        ϵ = par.ϵ,
        χ = par.χ,
        A = par.A,
        θ_Π = par.θ_Π,
        θ_Y = par.θ_Y,
        φ = par.φ,
        ρ_ζ = par.ρ_ζ,
        σ_ζ = par.σ_ζ,
        Π_bar = par.Π_bar,
        Y_bar = par.Y_bar,
        D = par.D,
        R_bar = R_bar,
        MC_bar = MC_bar,
        N_bar = N_bar
    )
end

"""
    get_bounds(ranges::Ranges)

Extract lower and upper bounds from Ranges struct as vectors.

Returns: (lower_bounds, upper_bounds)
"""
function get_bounds(ranges::Ranges)
    fields = fieldnames(typeof(ranges))
    lower = [getfield(ranges, f)[1] for f in fields]
    upper = [getfield(ranges, f)[2] for f in fields]
    return lower, upper
end

"""
    make_network(ranges::Ranges; kwargs...)

Create neural network architecture for RANK policy function approximation.

Network structure:
    Input: [ζ; θ] where θ = [θ_Π, θ_Y, φ, ρ_ζ, σ_ζ]
    Hidden layers: N × [Linear → Activation]
    Output: [N, Π] (labor and inflation)

Arguments:
- ranges: Ranges struct (for normalization bounds)
- N_states: Number of state variables (default: 1)
- N_outputs: Number of policy outputs (default: 2 for N, Π)
- hidden: Hidden layer width (default: 128)
- layers: Number of hidden layers (default: 5)
- activation: Activation function (default: silu/swish)
- scale_factor: Output scaling (default: 1.0, no scaling)

Returns: (network, ps, st)
"""
function make_network(ranges::Ranges;
                      N_states=1, N_outputs=2, hidden=128,
                      layers=5, activation=swish, scale_factor=1.0)
    
    # N_par: number of estimated parameters
    N_par = length(fieldnames(typeof(ranges))) - 1  # Exclude ζ
    N_input = N_states + N_par
    
    # Extract normalization bounds
    lower, upper = get_bounds(ranges)
    
    # Normalization layer at input
    norm_layer = NormalizeLayer(lower, upper)
    
    # Build the network
    network = Chain(
        norm_layer,
        Dense(N_input, hidden, activation),
        [Dense(hidden, hidden, activation) for _ in 1:(layers-1)]...,
        Dense(hidden, N_outputs),
        x -> x .* scale_factor  # Scale output if needed
    )
    
    # Initialize parameters and state
    rng = Random.default_rng()
    ps, st = Lux.setup(rng, network)
    
    return network, ps, st
end

# Helper: promote scalar/vector → (1,:) row matrix uniformly
to_row(x::Real) = fill(x, 1, 1)
to_row(x::AbstractVector) = reshape(x, 1, :)
to_row(x::AbstractMatrix) = x

"""
    step(state::State, shocks::AbstractArray, par::RANKParameters, ss::RANKParameters)

Evolve the state forward one period using the AR(1) process.

State transition:
    ζ_{t+1} = ρ_ζ ζ_t + ε_{t+1}

Arguments:
- state: Current state State{T}
- shocks: MC draws of ε_{t+1}, shape (mc, batch) or (1, batch)
- par: Model parameters
- ss: Steady state parameters

Returns: State{T} for next period
"""
function step(state::State, shocks::AbstractArray, par::RANKParameters, ss::RANKParameters)
    # Get current state as row matrix
    ζ_current = to_row(state.ζ)  # (1, batch)
    
    # Broadcast AR(1): ζ_{t+1} = ρ_ζ ζ_t + ε_{t+1}
    # If shocks is (mc, batch), we get (mc, batch) output
    # If shocks is (1, batch), we get (1, batch) output
    ζ_next = par.ρ_ζ .* ζ_current .+ shocks  # (mc, batch) or (1, batch)
    
    # Return new state with same shape
    return State(ζ=ζ_next)
end

"""
    policy(network, state::State, par::RANKParameters, ps, st)

Evaluate neural network policy function to get control variables.

Arguments:
- network: Trained Lux network
- state: Current state
- par: Model parameters (with estimated parameters)
- ps: Network parameters
- st: Network state

Returns: (N_t, Π_t, st_new)
- N_t: Labor/production (same shape as state.ζ)
- Π_t: Gross inflation (same shape as state.ζ)
- st_new: Updated network state
"""
function policy(network, state::State, par::RANKParameters, ps, st)
    # Prepare input: [ζ; θ_Π, θ_Y, φ, ρ_ζ, σ_ζ]
    ζ = to_row(state.ζ)  # (1, batch) or (mc, batch)
    
    # Get batch size
    batch = size(ζ, 2)
    
    # Stack parameters (estimated only)
    θ = vcat(
        to_row(par.θ_Π),
        to_row(par.θ_Y),
        to_row(par.φ),
        to_row(par.ρ_ζ),
        to_row(par.σ_ζ)
    )  # (5, batch) or (5, 1) if scalar
    
    # Broadcast θ if needed
    if size(θ, 2) == 1 && batch > 1
        θ = repeat(θ, 1, batch)
    end
    
    # Combine: input is (6, batch) or (6, mc*batch)
    input = vcat(ζ, θ)
    
    # Evaluate network
    output, st_new = network(input, ps, st)
    
    # Split output: [N; Π]
    N_t = output[1:1, :]  # (1, batch)
    Π_t = output[2:2, :]  # (1, batch)
    
    # Return with original shape
    return N_t, Π_t, st_new
end

"""
    compute_all_variables(N_t, Π_t, par::RANKParameters, ss::RANKParameters)

Given NN outputs N_t and Π_t, compute all other endogenous variables.

Variable identification sequence:
    Y_t = A × N_t              [Production]
    C_t = Y_t                  [Market clearing]
    MC_t = χ A^(σ-1) N_t^(η+σ) [Labor FOC]
    R_t = max{1, R̄(Π_t/Π̄)^θ_Π (Y_t/Ȳ)^θ_Y}  [Taylor rule with ZLB]

Arguments:
- N_t: Labor from NN, shape (1, batch) or (mc, batch)
- Π_t: Inflation from NN, shape (1, batch) or (mc, batch)
- par: Model parameters
- ss: Steady state parameters

Returns: Dict with Y_t, C_t, MC_t, R_t, all same shape as inputs
"""
function compute_all_variables(N_t, Π_t, par::RANKParameters, ss::RANKParameters)
    # Output
    Y_t = par.A .* N_t
    
    # Consumption (market clearing)
    C_t = Y_t
    
    # Marginal cost from labor FOC
    MC_t = par.χ .* par.A .^ (par.σ .- 1) .* N_t .^ (par.η .+ par.σ)
    
    # Taylor rule (notional)
    R_N = ss.R_bar .* (Π_t ./ par.Π_bar) .^ par.θ_Π .* (Y_t ./ par.Y_bar) .^ par.θ_Y
    
    # Apply ZLB constraint
    R_t = max.(1.0, R_N)
    
    return Dict(:Y => Y_t, :C => C_t, :MC => MC_t, :R => R_t, :R_N => R_N)
end

"""
    loss_fn(network, par::RANKParameters, state::State, shocks, ss::RANKParameters, ps, st)

Compute loss function for RANK model.

Loss components:
    L_Euler = |1 - β R_t E_t[(N_t/N_{t+1})^σ / Π_{t+1}]|^2
    L_Phillips = |φ(Π_t/Π̄ - 1)(Π_t/Π̄) - (1-ϵ) - ϵMC_t - φE_t[...]|^2

Total loss:
    Loss = α_Euler * L_Euler + α_Phillips * L_Phillips

Arguments:
- network: Neural network
- par: Model parameters (batch)
- state: Current state (batch)
- shocks: MC draws for expectations, shape (mc, batch)
- ss: Steady state parameters
- ps: Network parameters
- st: Network state

Returns: (loss, stats, st_new)
- loss: Scalar total loss
- stats: NamedTuple with (res_Euler, res_Phillips, res_total, N_mean, Π_mean)
- st_new: Updated network state
"""
function loss_fn(network, par::RANKParameters, state::State, shocks, ss::RANKParameters, ps, st)
    # Get current period policy
    N_t, Π_t, st = policy(network, state, par, ps, st)  # (1, batch)
    
    # Compute auxiliary variables for period t
    vars_t = compute_all_variables(N_t, Π_t, par, ss)
    R_t = vars_t[:R]      # (1, batch)
    MC_t = vars_t[:MC]    # (1, batch)
    Y_t = vars_t[:Y]      # (1, batch)
    
    # Evolve state to t+1 using MC draws
    state_tp1 = step(state, shocks, par, ss)  # ζ_{t+1} is (mc, batch)
    
    # Get t+1 policy (broadcasting over MC dimension)
    N_tp1, Π_tp1, st = policy(network, state_tp1, par, ps, st)  # (mc, batch)
    
    # Compute Y_{t+1} for Phillips curve expectation
    Y_tp1 = par.A .* N_tp1  # (mc, batch)
    
    # -------------------------------------------------------------------------
    # Euler Equation Residual
    # -------------------------------------------------------------------------
    # E_t[(N_t/N_{t+1})^σ / Π_{t+1}] ≈ (1/mc) Σ (N_t/N_{t+1}^(m))^σ / Π_{t+1}^(m)
    
    # Broadcast N_t from (1, batch) to (mc, batch)
    N_t_expanded = repeat(N_t, size(N_tp1, 1), 1)
    
    euler_integrand = (N_t_expanded ./ N_tp1) .^ par.σ ./ Π_tp1  # (mc, batch)
    euler_expectation = mean(euler_integrand, dims=1)  # (1, batch)
    
    # Residual: 1 - β R_t E_t[...]
    euler_residual = 1 .- par.β .* R_t .* euler_expectation  # (1, batch)
    
    # Loss component
    res_Euler = mean(euler_residual .^ 2)
    
    # -------------------------------------------------------------------------
    # Phillips Curve Residual
    # -------------------------------------------------------------------------
    # E_t[(Π_{t+1}/Π̄ - 1)(Π_{t+1}/Π̄) × (Y_{t+1}/Y_t)]
    
    # Broadcast Y_t from (1, batch) to (mc, batch)
    Y_t_expanded = repeat(Y_t, size(Y_tp1, 1), 1)
    Π_bar_expanded = repeat(to_row(par.Π_bar), size(Π_tp1, 1), size(Π_tp1, 2))
    R_t_expanded = repeat(R_t, size(Π_tp1, 1), 1)
    
    phillips_integrand = (Π_tp1 ./ Π_bar_expanded .- 1) .* (Π_tp1 ./ Π_bar_expanded) .* 
                         (Y_tp1 ./ Y_t_expanded) ./ R_t_expanded  # (mc, batch)
    phillips_expectation = mean(phillips_integrand, dims=1)  # (1, batch)
    
    # LHS: φ(Π_t/Π̄ - 1)(Π_t/Π̄)
    lhs = par.φ .* (Π_t ./ par.Π_bar .- 1) .* (Π_t ./ par.Π_bar)
    
    # RHS: (1-ϵ) + ϵMC_t + φE_t[...]
    rhs = (1 .- par.ϵ) .+ par.ϵ .* MC_t .+ par.φ .* phillips_expectation
    
    # Residual
    phillips_residual = lhs .- rhs  # (1, batch)
    
    # Loss component
    res_Phillips = mean(phillips_residual .^ 2)
    
    # -------------------------------------------------------------------------
    # Total Loss (equal weights)
    # -------------------------------------------------------------------------
    α_Euler = 1.0
    α_Phillips = 1.0
    
    loss = α_Euler * res_Euler + α_Phillips * res_Phillips
    
    # Statistics for monitoring
    stats = (
        res_Euler = res_Euler,
        res_Phillips = res_Phillips,
        res_total = loss,
        N_mean = mean(N_t),
        Π_mean = mean(Π_t)
    )
    
    return loss, stats, st
end

"""
    loss_fn_wrapper(network, ps, st, data)

Wrapper for Lux.Training.single_train_step! compatibility.

Arguments:
- network: Neural network
- ps: Network parameters
- st: Network state
- data: Tuple (par, state, shocks, ss)

Returns: (loss, st_new, stats)
"""
function loss_fn_wrapper(network, ps, st, data)
    par, state, shocks, ss = data
    loss, stats, st = loss_fn(network, par, state, shocks, ss, ps, st)
    return loss, st, stats
end

"""
    prior_distribution(ranges::Ranges)

Create prior distributions for parameters from ranges.

Returns: NamedTuple of Uniform distributions
"""
function prior_distribution(ranges::Ranges)
    return (
        θ_Π = Uniform(ranges.θ_Π...),
        θ_Y = Uniform(ranges.θ_Y...),
        φ = Uniform(ranges.φ...),
        ρ_ζ = Uniform(ranges.ρ_ζ...),
        σ_ζ = Uniform(ranges.σ_ζ...)
    )
end

"""
    draw_parameters(priors, batch::Int)

Draw batch of parameter sets from prior distributions.

Returns: RANKParameters with vectors of length batch
"""
function draw_parameters(priors, batch::Int)
    # Draw estimated parameters
    θ_Π = rand(priors.θ_Π, batch)
    θ_Y = rand(priors.θ_Y, batch)
    φ = rand(priors.φ, batch)
    ρ_ζ = rand(priors.ρ_ζ, batch)
    σ_ζ = rand(priors.σ_ζ, batch)
    
    # Fixed calibrated parameters (Table 5)
    β = fill(0.9975, batch)
    σ = fill(1.0, batch)
    η = fill(1.0, batch)
    ϵ = fill(11.0, batch)
    χ = fill(0.91, batch)
    A = fill(1.0, batch)
    
    # Steady state targets
    Π_bar = fill(1.005, batch)  # 2% annual inflation
    Y_bar = fill(1.0, batch)    # Normalized
    D = fill(0.0, batch)        # Not specified for RANK, set to 0
    
    return RANKParameters(
        β=β, σ=σ, η=η, ϵ=ϵ, χ=χ, A=A,
        θ_Π=θ_Π, θ_Y=θ_Y, φ=φ, ρ_ζ=ρ_ζ, σ_ζ=σ_ζ,
        Π_bar=Π_bar, Y_bar=Y_bar, D=D
    )
end

"""
    initialize_state(par::RANKParameters, batch::Int, ss::RANKParameters)

Initialize state at steady state (ζ = 0).

Returns: State with ζ = 0
"""
function initialize_state(par::RANKParameters, batch::Int, ss::RANKParameters)
    return State(ζ = zeros(batch))
end

"""
    draw_shocks(shock_config::Shocks, mc::Int, batch::Int)

Draw Monte Carlo shock realizations for computing expectations.

Arguments:
- shock_config: Shocks configuration
- mc: Number of Monte Carlo draws
- batch: Batch size

Returns: Matrix of shocks, shape (mc, batch)
"""
function draw_shocks(shock_config::Shocks, mc::Int, batch::Int)
    if shock_config.antithetic && mc > 1
        # Use antithetic variates for variance reduction
        half_mc = div(mc, 2)
        ε_pos = randn(half_mc, batch) .* shock_config.σ
        ε_neg = -ε_pos
        return vcat(ε_pos, ε_neg)
    else
        return randn(mc, batch) .* shock_config.σ
    end
end

"""
    cosine_annealing_lr(lr_max, lr_min, epoch, num_epochs)

Compute learning rate using cosine annealing schedule.

Formula:
    lr = lr_min + (lr_max - lr_min) * (1 + cos(π * epoch / num_epochs)) / 2
"""
function cosine_annealing_lr(lr_max, lr_min, epoch, num_epochs)
    return lr_min + (lr_max - lr_min) * (1 + cos(π * epoch / num_epochs)) / 2
end

# ═══════════════════════════════════════════════════════════════════════════
# TRAINING FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════

"""
    train_simple!(network, ps, st, ranges, shock_config; kwargs...)

Simple training loop with basic parameter sampling.

Arguments:
- network: Lux neural network
- ps: Network parameters
- st: Network state
- ranges: Ranges struct for parameter sampling
- shock_config: Shocks configuration

Keyword Arguments:
- num_epochs: Number of training epochs (default: 1000)
- batch: Batch size for parameter sampling (default: 100)
- mc: Number of Monte Carlo draws (default: 100)
- lr: Learning rate (default: 0.0001)

Returns: TrainState with trained parameters
"""
function train_simple!(network, ps, st, ranges, shock_config;
    num_epochs=1000, batch=100, mc=100, lr=0.0001)
    
    # Define the train state
    train_state = Lux.Training.TrainState(network, ps, st, Adam(lr))
    priors = prior_distribution(ranges)
    
    # Running the training loop
    for epoch in 1:num_epochs
        # Fresh samples each iteration
        par = draw_parameters(priors, batch)
        ss = steady_state(par)
        state = initialize_state(par, batch, ss)
        shocks = draw_shocks(shock_config, mc, batch)
        data = (par, state, shocks, ss)
        
        _, loss, stats, train_state = Lux.Training.single_train_step!(
            AutoZygote(), loss_fn_wrapper, data, train_state
        )
        
        # Print loss every 100 epochs
        if epoch % 100 == 0
            println("Epoch $epoch, Loss: $loss",
                    ", res_Euler: $(stats.res_Euler), res_Phillips: $(stats.res_Phillips)",
                    ", N̄: $(stats.N_mean), Π̄: $(stats.Π_mean)")
        end
    end
    
    return train_state
end

"""
    train!(network, ps, st, ranges, shock_config; kwargs...)

Advanced training loop with ZLB constraint scheduling, learning rate annealing, 
parameter redraws, and state stepping.

ZLB Constraint Scheduling:
- Iterations 1-5000: ZLB constraint off (smooth training)
- Iterations 5000-10000: Gradual introduction using a_ZLB ∈ [0, 1]
- Iterations 10000+: Full ZLB constraint

Arguments:
- network: Lux neural network
- ps: Network parameters
- st: Network state
- ranges: Ranges struct for parameter sampling
- shock_config: Shocks configuration

Keyword Arguments:
- num_epochs: Number of training epochs (default: 30000)
- batch: Batch size for parameter sampling (default: 100)
- mc: Number of Monte Carlo draws (default: 100)
- lr: Initial learning rate (default: 0.0001)
- internal: Number of gradient steps per epoch (default: 15)
- print_after: Print frequency in epochs (default: 100)
- par_draw_after: Parameter redraw frequency in epochs (default: 40)
- num_steps: Number of state evolution steps per epoch (default: 20)
- eta_min: Minimum learning rate for cosine annealing (default: 1e-6)
- zlb_start: Epoch to start introducing ZLB (default: 5000)
- zlb_end: Epoch to finish introducing ZLB (default: 10000)

Returns: (train_state, loss_dict)
- train_state: TrainState with trained parameters
- loss_dict: Dictionary with training history
"""
function train!(network, ps, st, ranges, shock_config;
    num_epochs=30000, batch=100, mc=100, lr=0.0001, internal=15, print_after=100,
    par_draw_after=40, num_steps=20, eta_min=1e-6, 
    zlb_start=5000, zlb_end=10000)
    
    # Define the train state
    train_state = Lux.Training.TrainState(network, ps, st, Adam(lr))
    priors = prior_distribution(ranges)
    
    # Create dictionary to track loss
    loss_dict = Dict(:iteration => [], :running_loss => [], :loss => [],
                     :res_Euler => [], :res_Phillips => [], :N_mean => [], :Π_mean => [])
    
    # Initialize the par, ss, state
    par = draw_parameters(priors, batch)
    ss = steady_state(par)
    state = initialize_state(par, batch, ss)
    
    # Initialize running loss
    running_loss = 0.0
    
    # Create progress meter
    prog = Progress(num_epochs; dt=1.0, desc="Training RANK... ", showspeed=true)
    
    # Running the training loop
    for epoch in 1:num_epochs
        # Draw shocks every epoch
        shocks = draw_shocks(shock_config, mc, batch)
        data = (par, state, shocks, ss)
        
        # Starting loss and stats
        loss = nothing
        stats = nothing
        
        # Inner loop for 'internal' gradient steps
        for o in 1:internal
            _, loss, stats, train_state = Lux.Training.single_train_step!(
                AutoZygote(), loss_fn_wrapper, data, train_state
            )
        end
        
        # Update running loss
        running_loss += loss / batch
        
        # Update lr using cosine annealing
        current_lr = cosine_annealing_lr(lr, eta_min, epoch, num_epochs)
        
        # Update train_state with new lr
        train_state = Lux.Training.TrainState(network, train_state.parameters,
                                              train_state.states, Adam(current_lr))
        
        # Print loss every print_after epochs
        if epoch % print_after == 0
            zlb_status = epoch < zlb_start ? "OFF" : 
                        epoch < zlb_end ? "RAMPING" : "ON"
            println("Epoch $epoch [ZLB: $zlb_status], Running Loss: $(running_loss/print_after), Loss: $loss")
            println("  res_Euler: $(stats.res_Euler), res_Phillips: $(stats.res_Phillips)")
            println("  N̄: $(stats.N_mean), Π̄: $(stats.Π_mean)")
            
            push!(loss_dict[:iteration], epoch)
            push!(loss_dict[:running_loss], running_loss / print_after)
            push!(loss_dict[:loss], loss)
            push!(loss_dict[:res_Euler], stats.res_Euler)
            push!(loss_dict[:res_Phillips], stats.res_Phillips)
            push!(loss_dict[:N_mean], stats.N_mean)
            push!(loss_dict[:Π_mean], stats.Π_mean)
            
            # Reset running loss
            running_loss = 0.0
        end
        
        # Redraw parameters, ss, state after par_draw_after epochs
        if epoch % par_draw_after == 0
            par = draw_parameters(priors, batch)
            ss = steady_state(par)
            state = initialize_state(par, batch, ss)
        end
        
        # Iterate state num_steps ahead
        for _ in 1:num_steps
            shocks = draw_shocks(shock_config, 1, batch)
            state = step(state, shocks, par, ss)
        end
        
        # Update progress meter
        next!(prog)
    end
    
    return train_state, loss_dict
end

# ═══════════════════════════════════════════════════════════════════════════
# SIMULATION FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════

"""
    sim_step(network, state::State, par::RANKParameters, ss::RANKParameters, ps, st)

Execute one time step of simulation.

Arguments:
- network: Trained neural network
- state: Current state
- par: Parameters
- ss: Steady state parameters
- ps: Network parameters
- st: Network state

Returns: Dictionary with:
- :ζ → preference shock
- :N_t → labor
- :Π_t → inflation
- :Y_t → output
- :C_t → consumption
- :MC_t → marginal cost
- :R_t → nominal rate
- :st → updated network state
"""
function sim_step(network, state::State, par::RANKParameters, ss::RANKParameters, ps, st)
    # Get policy functions for time t
    N_t, Π_t, st_new = policy(network, state, par, ps, st)
    
    # Compute all other variables
    vars = compute_all_variables(N_t, Π_t, par, ss)
    
    return Dict(
        :ζ => state.ζ,
        :N_t => N_t,
        :Π_t => Π_t,
        :Y_t => vars[:Y],
        :C_t => vars[:C],
        :MC_t => vars[:MC],
        :R_t => vars[:R],
        :st => st_new
    )
end

"""
    simulate(network, ps, st, batch, ranges, shock_config; kwargs...)

Generate simulated time series from trained neural network.

Arguments:
- network: Trained neural network
- ps: Network parameters
- st: Network state
- batch: Batch size (number of trajectories)
- ranges: Ranges struct for parameter sampling
- shock_config: Shocks configuration

Keyword Arguments:
- par: Optional pre-computed parameters (if nothing, draws from prior)
- burn: Burn-in period steps (default: 100)
- num_steps: Simulation length (default: 500)
- seed: Optional random seed

Returns: Dictionary with:
- :ζ → (num_steps, batch) matrix of preference shocks
- :N → (num_steps, batch) matrix of labor
- :Π → (num_steps, batch) matrix of inflation
- :Y → (num_steps, batch) matrix of output
- :R → (num_steps, batch) matrix of nominal rates
"""
function simulate(network, ps, st, batch, ranges, shock_config;
                  par=nothing, burn=100, num_steps=500, seed=nothing)
    if seed !== nothing
        Random.seed!(seed)
    end
    
    # Draw parameters if not provided
    if par === nothing
        priors = prior_distribution(ranges)
        par = draw_parameters(priors, batch)
    end
    
    # Compute steady state and initialize state
    ss = steady_state(par)
    state = initialize_state(par, batch, ss)
    
    # Burn-in period
    for _ in 1:burn
        shocks = draw_shocks(shock_config, 1, batch)
        state = step(state, shocks, par, ss)
    end
    
    # Storage
    ζ_store = []
    N_store = []
    Π_store = []
    Y_store = []
    R_store = []
    
    # Simulation loop
    for t in 1:num_steps
        sim_out = sim_step(network, state, par, ss, ps, st)
        
        push!(ζ_store, sim_out[:ζ])
        push!(N_store, sim_out[:N_t])
        push!(Π_store, sim_out[:Π_t])
        push!(Y_store, sim_out[:Y_t])
        push!(R_store, sim_out[:R_t])
        
        # Update network state
        st = sim_out[:st]
        
        # Step to next state
        shocks = draw_shocks(shock_config, 1, batch)
        state = step(state, shocks, par, ss)
    end
    
    # Stack into matrices: (num_steps, batch)
    return Dict(
        :ζ => hcat(ζ_store...)',
        :N => hcat(N_store...)',
        :Π => hcat(Π_store...)',
        :Y => hcat(Y_store...)',
        :R => hcat(R_store...)'
    )
end

# ═══════════════════════════════════════════════════════════════════════════
# PLOTTING FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════

"""
    plot_training_loss(loss_dict; log_scale=true)

Plot training loss over iterations.
"""
function plot_training_loss(loss_dict; log_scale=true)
    p = plot(loss_dict[:iteration], loss_dict[:running_loss],
             xlabel="Iteration", ylabel="Running Loss",
             title="RANK Training Convergence", legend=false,
             linewidth=2)
    if log_scale
        plot!(p, yscale=:log10)
    end
    return p
end

"""
    plot_simulation_path(sim_results; trajectory=1)

Plot simulated paths for a single trajectory.
"""
function plot_simulation_path(sim_results; trajectory=1)
    T = size(sim_results[:N], 1)
    t = 1:T
    
    p1 = plot(t, sim_results[:N][:, trajectory], 
              title="Labor N_t", xlabel="Time", ylabel="N", legend=false)
    p2 = plot(t, sim_results[:Π][:, trajectory],
              title="Inflation Π_t", xlabel="Time", ylabel="Π", legend=false)
    p3 = plot(t, sim_results[:Y][:, trajectory],
              title="Output Y_t", xlabel="Time", ylabel="Y", legend=false)
    p4 = plot(t, sim_results[:R][:, trajectory],
              title="Nominal Rate R_t", xlabel="Time", ylabel="R", legend=false)
    
    plot(p1, p2, p3, p4, layout=(2,2), size=(900, 600))
end

# ═══════════════════════════════════════════════════════════════════════════
# END OF FILE
# ═══════════════════════════════════════════════════════════════════════════

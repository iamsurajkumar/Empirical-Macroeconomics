# ═══════════════════════════════════════════════════════════════════════════
# SHARED ECONOMIC UTILITIES
# Functions that work across all models
# ═══════════════════════════════════════════════════════════════════════════

# ───────────────────────────────────────────────────────────────────────────
# Steady State Computation
# ───────────────────────────────────────────────────────────────────────────

"""
    steady_state(par::NKParameters)

Compute derived steady-state parameters κ and ω for NK model.

Mathematical formulations:
    κ = (1 - ϕ)(1 - ϕβ)(σ + η) / ϕ  [NKPC slope]
    ω = (1 + η) / (σ + η)            [Natural rate composite]
"""
function steady_state(par::NKParameters)
    # NKPC slope coefficient
    κ = ((1 .- par.ϕ) .* (1 .- par.ϕ .* par.β) .* (par.σ .+ par.η)) ./ par.ϕ

    # Composite parameter for natural rate equation
    ω = (1 .+ par.η) ./ (par.σ .+ par.η)

    return NKParameters(
        β=par.β,
        σ=par.σ,
        η=par.η,
        ϕ=par.ϕ,
        ϕ_pi=par.ϕ_pi,
        ϕ_y=par.ϕ_y,
        ρ=par.ρ,
        σ_shock=par.σ_shock,
        κ=κ,
        ω=ω
    )
end

# ──────────────────────────────────────────────────────────────────────────
# Check this later, I will uncomment it later once I have the RANK Model added
# -─────────────────────────────────────────────────────────────────────────

# """
#     steady_state(par::RANKParameters)

# Compute RANK deterministic steady state.
# Reuses NK logic since ZLB typically doesn't bind in steady state.
# """
# function steady_state(par::RANKParameters)
#     # NKPC slope coefficient
#     κ = ((1 .- par.ϕ) .* (1 .- par.ϕ .* par.β) .* (par.σ .+ par.η)) ./ par.ϕ

#     # Composite parameter for natural rate equation
#     ω = (1 .+ par.η) ./ (par.σ .+ par.η)

#     return RANKParameters(
#         β = par.β,
#         σ = par.σ,
#         η = par.η,
#         ϕ = par.ϕ,
#         ϕ_pi = par.ϕ_pi,
#         ϕ_y = par.ϕ_y,
#         ρ = par.ρ,
#         σ_shock = par.σ_shock,
#         κ = κ,
#         ω = ω,
#         r_min = par.r_min,
#         ϕ_r = par.ϕ_r
#     )
# end

# ───────────────────────────────────────────────────────────────────────────
# Parameter Sampling
# ───────────────────────────────────────────────────────────────────────────

"""
    prior_distribution(ranges::Ranges)

Define uniform prior distributions for all parameters.

Returns: NamedTuple of Uniform distributions
"""
function prior_distribution(ranges::Ranges)
    return (
        ζ=Uniform(ranges.ζ[1], ranges.ζ[2]),
        β=Uniform(ranges.β[1], ranges.β[2]),
        σ=Uniform(ranges.σ[1], ranges.σ[2]),
        η=Uniform(ranges.η[1], ranges.η[2]),
        ϕ=Uniform(ranges.ϕ[1], ranges.ϕ[2]),
        ϕ_pi=Uniform(ranges.ϕ_pi[1], ranges.ϕ_pi[2]),
        ϕ_y=Uniform(ranges.ϕ_y[1], ranges.ϕ_y[2]),
        ρ=Uniform(ranges.ρ[1], ranges.ρ[2]),
        σ_shock=Uniform(ranges.σ_shock[1], ranges.σ_shock[2])
    )
end

"""
    draw_parameters(priors, batch_size::Int)

Draw batch_size parameter sets from prior distributions.

Returns: NKParameters struct with vector fields
"""
function draw_parameters(priors, batch_size::Int)
    return NKParameters(
        β=rand(priors.β, batch_size),
        σ=rand(priors.σ, batch_size),
        η=rand(priors.η, batch_size),
        ϕ=rand(priors.ϕ, batch_size),
        ϕ_pi=rand(priors.ϕ_pi, batch_size),
        ϕ_y=rand(priors.ϕ_y, batch_size),
        ρ=rand(priors.ρ, batch_size),
        σ_shock=rand(priors.σ_shock, batch_size)
    )
end

# ───────────────────────────────────────────────────────────────────────────
# State Initialization
# ───────────────────────────────────────────────────────────────────────────

"""
    initialize_state(par::AbstractModelParameters, batch::Int, ss::AbstractModelParameters)

Initialize state from ergodic distribution of AR(1) process.

Mathematical operation:
    σ_ergodic = σ_shock · σ · (ρ - 1) · ω / √(1 - ρ²)
    ζ_0 ~ N(0, σ_ergodic²)
"""
function initialize_state(par::AbstractModelParameters, batch::Int, ss::AbstractModelParameters)
    rho = par.ρ
    sigma = par.σ_shock .* par.σ .* (par.ρ .- 1) .* ss.ω
    ergodic = sigma ./ sqrt.(1 .- rho .^ 2)
    ζ = randn(batch) .* ergodic

    return State(ζ=ζ)
end

# ───────────────────────────────────────────────────────────────────────────
# Shock Generation
# ───────────────────────────────────────────────────────────────────────────

"""
    draw_shocks(shock_struct::Shocks, mc::Int, batch::Int)

Draw shock innovations for Monte Carlo simulation.

Arguments:
- shock_struct: Shocks configuration
- mc: Number of Monte Carlo draws
- batch: Batch size

Returns: (mc, batch) matrix of shocks

If antithetic=true, uses antithetic variates for variance reduction.
"""
function draw_shocks(shock_struct::Shocks, mc::Int, batch::Int)
    if shock_struct.antithetic
        half_mc = div(mc, 2)
        ϵ_shocks = randn(half_mc, batch)
        shocks = vcat(ϵ_shocks, -ϵ_shocks)
        if isodd(mc)
            extra_shock = randn(1, batch)
            shocks = vcat(shocks, extra_shock)
        end
    else
        shocks = randn(mc, batch)
    end
    return shocks
end

# ───────────────────────────────────────────────────────────────────────────
# Simulation Functions THINK ABOUT IT LATER IF these functions need to be specific to NK or RANK or HANK Model
# ───────────────────────────────────────────────────────────────────────────



"""
    sim_step(network, state::State, par::AbstractModelParameters, ps, st)

Execute one time step of simulation.

Arguments:
- network: Trained neural network
- state: Current state
- par: Parameters
- ps: Network parameters
- st: Network state

Returns: Dictionary with:
- :R → natural rate shock ζ
- :X_t → output gap
- :π_t → inflation
- :st → updated network state
"""
function sim_step(network, state::State, par::AbstractModelParameters, ps, st)
    # Get policy functions for time t
    X_t, π_t, st_new = policy(network, state, par, ps, st)

    return Dict(:R => state.ζ, :X_t => X_t, :π_t => π_t, :st => st_new)
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
- burn: Burn-in period steps (default: 99)
- num_steps: Simulation length (default: 101)
- seed: Optional random seed

Returns: Dictionary with:
- :R → (num_steps, batch) matrix of natural rate shocks
- :X → (num_steps, batch) matrix of output gaps
- :π → (num_steps, batch) matrix of inflation rates
"""
function simulate(network, ps, st, batch, ranges, shock_config;
    par=nothing, burn=99, num_steps=101, seed=nothing)
    if seed != nothing
        Random.seed!(seed)
    end

    # Draw parameters if not provided
    if par === nothing
        par = model.parameters  # Ensure correct parameter struct
    end

    # Compute steady state and initialize state
    ss = steady_state(par)
    state = initialize_state(par, batch, ss)

    # Burn-in period
    for _ in 1:burn
        shocks = draw_shocks(shock_config, 1, batch)  # Single shock per step
        state = step(state, shocks, par, ss)
    end

    # Storage
    R_store = []
    X_store = []
    π_store = []

    # Simulation loop
    for t in 1:num_steps
        sim_out = sim_step(network, state, par, ps, st)

        push!(R_store, sim_out[:R])
        push!(X_store, sim_out[:X_t])
        push!(π_store, sim_out[:π_t])

        # Update network state
        st = sim_out[:st]

        # Step to next state
        shocks = draw_shocks(shock_config, 1, batch)  # Single shock per step
        state = step(state, shocks, par, ss)
    end

    # Stack into matrices: (num_steps, batch)
    return Dict(
        :R => hcat(R_store...)',
        :X => hcat(X_store...)',
        :π => hcat(π_store...)'
    )
end



"""
    generate_covariance_matrix(data::Dict{Symbol, <:AbstractVector{<:Real}})

Computes a diagonal covariance matrix from a dictionary of time series data.

# Arguments
- `data`: A dictionary mapping symbols to vectors of real numbers (the time series).

# Returns
- A `LinearAlgebra.Diagonal` matrix where the diagonal elements are the
  variances of the corresponding time series in `data`.
"""
function generate_covariance_matrix(data::NamedTuple)
    # Ensure the dictionary is not empty to avoid errors.
    @assert !isempty(data) "Input data matrix is empty."

    # Use a comprehension to compute the variance for each vector in the dictionary.
    # `values(data)` returns an iterator over all the time series vectors.
    # `var(v)` computes the sample variance for each vector `v`.
    # The result is a vector of variances.
    vars = [var(data[key]) for key in keys(data)]

    # `LinearAlgebra.Diagonal` creates an efficient diagonal matrix from the vector of variances.
    return LinearAlgebra.Diagonal(vars)
end


function add_measurement_error(data::NamedTuple; error_scale::Float64=0.1, seed=nothing)
    if seed != nothing
        Random.seed!(seed)
    end

    # Generate covariance matrix
    R_cov = generate_covariance_matrix(data) * error_scale

    # Create noisy data by adding measurement error
    # Compute covariance matrix scaled by error_share
    # Generate noise from N(0, √R[i,i]) for each variable in data
    noisy_data = NamedTuple{keys(data)}(
    v .+ randn(size(v)) .* sqrt(R_cov[i,i]) for (i, v) in enumerate(values(data))
    )
    return noisy_data, R_cov
end


# Adding the functions to simulate from the NKModel

function simulate_model(model, batch_size, shock_config;
    par=nothing, burn=99, num_steps=101, seed=nothing)
    if seed != nothing
        Random.seed!(seed)
    end

    # Draw parameters if not provided
    if par === nothing
        par = model.parameters  # Ensure correct parameter struct
    end

    # Compute steady state and initialize state
    ss = steady_state(par)
    state = initialize_state(par, batch_size, ss)

    # Burn-in period
    for _ in 1:burn
        shocks = draw_shocks(shock_config, 1, batch_size)  # Single shock per step
        state = step(state, shocks, par, ss)
    end

    # Storage
    R_store = Matrix{Float64}(undef, num_steps, batch_size)
    X_store = Matrix{Float64}(undef, num_steps, batch_size)
    π_store = Matrix{Float64}(undef, num_steps, batch_size)

    # Simulation loop
    for t in 1:num_steps
        sim_out = sim_step(model.network, state, par, model.ps, model.st)

        R_store[t, :] = vec(sim_out[:R])
        X_store[t, :] = vec(sim_out[:X_t])
        π_store[t, :] = vec(sim_out[:π_t])

        # Update network state
        model.st = sim_out[:st]

        # Step to next state
        shocks = draw_shocks(shock_config, 1, batch_size)  # Single shock per step
        state = step(state, shocks, par, ss)
    end

    # Stack into matrices: (num_steps, batch_size)
    return (R = R_store, X = X_store, π = π_store)
end

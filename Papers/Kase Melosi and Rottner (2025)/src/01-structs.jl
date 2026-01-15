# ═══════════════════════════════════════════════════════════════════════════
# TYPE DEFINITIONS AND CORE STRUCTURES
# ═══════════════════════════════════════════════════════════════════════════

# ───────────────────────────────────────────────────────────────────────────
# Abstract Type Hierarchy
# ───────────────────────────────────────────────────────────────────────────

"""
    AbstractModelParameters{T}

Abstract base type for all model parameters.

Type parameter T can be:
- Real: single parameter set
- Vector{<:Real}: batch of parameter sets
"""
abstract type AbstractModelParameters{T <: Union{Real, AbstractVector{<:Real}}} end

# ───────────────────────────────────────────────────────────────────────────
# Model-Specific Parameter Types
# ───────────────────────────────────────────────────────────────────────────

"""
    NKParameters{T} <: AbstractModelParameters{T}

Parameters for New Keynesian model.

# Parameters
- β: Discount factor
- σ: Inverse EIS (risk aversion)
- η: Inverse Frisch elasticity
- ϕ: Calvo parameter (price stickiness)
- ϕ_pi: Taylor rule inflation response
- ϕ_y: Taylor rule output response
- ρ: Shock persistence
- σ_shock: Shock volatility
- κ: NKPC slope (derived)
- ω: Natural rate composite parameter (derived)
"""
struct NKParameters{T} <: AbstractModelParameters{T}
    β::T
    σ::T
    η::T
    ϕ::T
    ϕ_pi::T
    ϕ_y::T
    ρ::T
    σ_shock::T
    κ::Union{T,Nothing}
    ω::Union{T,Nothing}
end

# Constructor: accepts scalars with keyword arguments
function NKParameters(; β, σ, η, ϕ, ϕ_pi, ϕ_y, ρ, σ_shock, κ=nothing, ω=nothing)
    T = typeof(β)
    return NKParameters{T}(β, σ, η, ϕ, ϕ_pi, ϕ_y, ρ, σ_shock, κ, ω)
end

"""
    expand(par::NKParameters, batch_size::Int)

Expand scalar parameters to batch by replicating batch_size times.
"""
function expand(par::NKParameters, batch_size::Int)
    return NKParameters{Vector{Float64}}(
        fill(par.β, batch_size),
        fill(par.σ, batch_size),
        fill(par.η, batch_size),
        fill(par.ϕ, batch_size),
        fill(par.ϕ_pi, batch_size),
        fill(par.ϕ_y, batch_size),
        fill(par.ρ, batch_size),
        fill(par.σ_shock, batch_size),
        par.κ === nothing ? nothing : fill(par.κ, batch_size),
        par.ω === nothing ? nothing : fill(par.ω, batch_size)
    )
end

"""
A construction function to create a new NKParameters struct with one parameter changed.

"""
function with_param(par::NKParameters, name::Symbol, val)
    NKParameters(
        β = name == :β ? val : par.β,
        σ = name == :σ ? val : par.σ,
        η = name == :η ? val : par.η,
        ϕ = name == :ϕ ? val : par.ϕ,
        ϕ_pi = name == :ϕ_pi ? val : par.ϕ_pi,
        ϕ_y = name == :ϕ_y ? val : par.ϕ_y,
        ρ = name == :ρ ? val : par.ρ,
        σ_shock = name == :σ_shock ? val : par.σ_shock,
        κ = par.κ,
        ω = par.ω
    )
end


"""
    RANKParameters{T} <: AbstractModelParameters{T}

Parameters for RANK model with ZLB.

# Parameters (inherits all NK parameters plus:)
- r_min: ZLB constraint (typically 1.0)
- ϕ_r: Interest rate smoothing parameter
"""
struct RANKParameters{T} <: AbstractModelParameters{T}
    # NK parameters
    β::T
    σ::T
    η::T
    ϕ::T
    ϕ_pi::T
    ϕ_y::T
    ρ::T
    σ_shock::T
    κ::Union{T,Nothing}
    ω::Union{T,Nothing}
    # RANK-specific
    r_min::T
    ϕ_r::T
end

# Constructor
function RANKParameters(; β, σ, η, ϕ, ϕ_pi, ϕ_y, ρ, σ_shock, r_min, ϕ_r, κ=nothing, ω=nothing)
    T = typeof(β)
    return RANKParameters{T}(β, σ, η, ϕ, ϕ_pi, ϕ_y, ρ, σ_shock, κ, ω, r_min, ϕ_r)
end



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

struct RANKParameters{T} <: AbstractModelParameters{T}
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




#----
# Add the HANK Model parameters later
#---

    

# ───────────────────────────────────────────────────────────────────────────
# State Variables
# ───────────────────────────────────────────────────────────────────────────

"""
    State{T}

State variables for NK/RANK models.

Type T can be:
- Real: single state
- Vector{<:Real}: batch of states
- Matrix{<:Real}: MC × batch

# State variables
- ζ: Natural rate shock (R*_t in continuous time)
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

# ───────────────────────────────────────────────────────────────────────────
# Configuration Structures
# ───────────────────────────────────────────────────────────────────────────

"""
    Ranges

Parameter bounds for training (normalization and sampling).

Stores (min, max) tuples for each parameter.
"""
Base.@kwdef struct Ranges
    ζ::Tuple{Float64,Float64}
    β::Tuple{Float64, Float64}
    σ::Tuple{Float64, Float64}
    η::Tuple{Float64, Float64}
    ϕ::Tuple{Float64, Float64}
    ϕ_pi::Tuple{Float64, Float64}
    ϕ_y::Tuple{Float64, Float64}
    ρ::Tuple{Float64, Float64}
    σ_shock::Tuple{Float64, Float64}
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
    Shocks

Configuration for shock sampling.

# Fields
- σ: Standard deviation of shocks
- antithetic: Use antithetic variates for variance reduction
"""
struct Shocks
    σ::Float64
    antithetic::Bool
end

# Constructor with default antithetic=false
function Shocks(; σ, antithetic = false)
    return Shocks(σ, antithetic)
end


## Defining the NKModel Struct

abstract type AbstractModel{P <: AbstractModelParameters} end

# Creatinga a mutalble struct for the NK Model as the network parameters will be updated during training
mutable struct NKModel{P <: AbstractModelParameters, N <: Lux.AbstractLuxLayer} <: AbstractModel{P}
    parameters::P
    state::State
    network::N
    ps::NamedTuple        # Network parameters (weights)
    st::NamedTuple        # Network state
    ranges::Ranges
end



# Update NKModel constructor:
function NKModel(par::P, ranges::Ranges, ::Val{L}; hidden=64, initial_ζ=0.0, 
    N_states=1, N_outputs=2, activation=Lux.celu, scale_factor=1/100) where {P <: NKParameters, L}
    
    state = State(ζ=initial_ζ)
    network, ps, st = make_network_stable(par, ranges, Val(L); 
        hidden=hidden, N_states=N_states, N_outputs=N_outputs, 
        activation=activation, scale_factor=scale_factor)
    
    return NKModel{P, typeof(network)}(par, state, network, ps, st, ranges)
end
# Extract components
get_parameters(m::NKModel) = m.parameters
get_state(m::NKModel) = m.state
get_network(m::NKModel) = (m.network, m.ps, m.st)
get_ranges(m::NKModel) = m.ranges

# Update mutable fields
set_ps!(m::NKModel, ps) = (m.ps = ps)
set_st!(m::NKModel, st) = (m.st = st)
set_state!(m::NKModel, state) = (m.state = state)




# # It will be used later  when I add the HANK Model Part

# """
#     TrainingConfig

# Training hyperparameters configuration.

# # Fields
# - n_iterations: Total number of training iterations
# - batch_size: Number of economies per batch
# - zlb_start_iter: Iteration to start introducing ZLB (for RANK/HANK)
# - zlb_end_iter: Iteration when ZLB fully active
# - param_redraw_freq: How often to redraw parameters
# - initial_sim_periods: Simulation length when drawing new params
# - regular_sim_periods: Simulation length for regular iterations
# """
# struct TrainingConfig
#     n_iterations::Int
#     batch_size::Int
#     zlb_start_iter::Int
#     zlb_end_iter::Int
#     param_redraw_freq::Int
#     initial_sim_periods::Int
#     regular_sim_periods::Int
# end

# """
#     LossWeights

# Weights for different loss function components.

# # Fields
# - euler: Weight for Euler equation residuals
# - phillips_curve: Weight for NKPC
# - bond_market: Weight for bond market clearing
# - goods_market: Weight for goods market clearing
# """
# struct LossWeights
#     euler::Float64
#     phillips_curve::Float64
#     bond_market::Float64
#     goods_market::Float64
# end





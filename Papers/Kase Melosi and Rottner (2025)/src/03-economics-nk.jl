# ═══════════════════════════════════════════════════════════════════════════
# NK MODEL-SPECIFIC FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════

# ───────────────────────────────────────────────────────────────────────────
# Policy Functions
# ───────────────────────────────────────────────────────────────────────────

"""
    policy(network, state::State, par::NKParameters, ps, st)

Evaluate policy π(ζ; θ) using single-pass vectorization over MC × batch.

Mathematical operation:
    For each (i,j) ∈ {1,…,mc} × {1,…,B}, compute
        (X^(i,j), Π^(i,j)) = f_NN([ζ^(i,j), θ^(j)])

    Input tensor shape: (9, mc·B) where each column is [ζ; β; σ; η; ϕ; ϕ_π; ϕ_y; ρ; σ_ε]
    Output tensor shape: (2, mc·B) reshaped to (mc, B) for X and Π

    Expectation: E[X_{t+1} | ζ_t, θ] ≈ (1/mc) Σ_i X^(i,·)

Returns: (X, Π, st_new) where X and Π have shape (mc, B)
"""
function policy(network, state::State, par::NKParameters, ps, st)
    # Handle ζ dimensions: scalar → (1,1), vector → (1,B), matrix → (mc,B)
    ζ_mat = state.ζ isa Real ? fill(state.ζ, 1, 1) :
            state.ζ isa AbstractVector ? reshape(state.ζ, 1, :) :
            state.ζ

    mc, B = size(ζ_mat)  # mc: Monte Carlo draws, B: batch size

    # Reshape ζ from (mc, B) → (1, mc·B) for single forward pass
    ζ_flat = reshape(ζ_mat, 1, mc * B)  # Flatten: column-major stacking

    # Helper: ensure parameter is array-like for repeat()
    to_vec(x) = x isa AbstractArray ? x : [x]

    # Tile each parameter θ_j to cover all mc draws for that batch element
    # Result: each parameter row is (1, B) repeated mc times → (1, mc·B)
    β_flat = reshape(repeat(to_vec(par.β), inner=mc), 1, mc * B)
    σ_flat = reshape(repeat(to_vec(par.σ), inner=mc), 1, mc * B)
    η_flat = reshape(repeat(to_vec(par.η), inner=mc), 1, mc * B)
    ϕ_flat = reshape(repeat(to_vec(par.ϕ), inner=mc), 1, mc * B)
    ϕ_pi_flat = reshape(repeat(to_vec(par.ϕ_pi), inner=mc), 1, mc * B)
    ϕ_y_flat = reshape(repeat(to_vec(par.ϕ_y), inner=mc), 1, mc * B)
    ρ_flat = reshape(repeat(to_vec(par.ρ), inner=mc), 1, mc * B)
    σ_shock_flat = reshape(repeat(to_vec(par.σ_shock), inner=mc), 1, mc * B)

    # Stack input features: (9, mc·B)
    # Each column k ∈ [1, mc·B] is [ζ_k; β_j; σ_j; …] for appropriate j
    input = vcat(ζ_flat, β_flat, σ_flat, η_flat, ϕ_flat,
        ϕ_pi_flat, ϕ_y_flat, ρ_flat, σ_shock_flat)

    # Convert to Float32 for network as otherwise the multiplication can be slow with the weights
    input = Float32.(input)

    # Single forward pass: network maps (9, mc·B) → (2, mc·B)
    output, st_new = network(input, ps, st)

    # Reshape output: (2, mc·B) → (2, mc, B)
    Y = reshape(output, 2, mc, B)

    # Extract outputs: (mc, B) each
    X = Y[1, :, :]  # Output gap
    Π = Y[2, :, :]  # Inflation

    return X, Π, st_new
end



# Helper: promote scalar/vector → (1,:) row matrix uniformly
to_row(x::Real) = fill(x, 1, 1)
to_row(x::AbstractVector) = reshape(x, 1, :)
to_row(x::AbstractMatrix) = x



# ───────────────────────────────────────────────────────────────────────────
# State Transition
# ───────────────────────────────────────────────────────────────────────────

"""
    step(state::State, shocks::AbstractArray, par::NKParameters, ss::NKParameters)

Evolve state forward one period using AR(1) process.

Mathematical operation:
    ζ_{t+1} = ρ ζ_t + ε_{t+1} · σ_shock · σ · (ρ - 1) · ω

where ε ~ N(0,1) is the shock innovation.

Handles:
- Scalar state + scalar shock
- Vector state + vector shocks (batch)
- Vector state + matrix shocks (MC × batch)
"""
function step(state::State, shocks::AbstractArray, par::NKParameters, ss::NKParameters)
    # AR(1): ζ_{t+1} = ρ ζ_t + ε · σ_shock · σ · (ρ - 1) · ω
    # Broadcasting: (1,n) .* (mc,n) → (mc,n)
    new_ζ = to_row(par.ρ) .* to_row(state.ζ) .+
            shocks .* to_row(par.σ_shock) .* to_row(par.σ) .*
            (to_row(par.ρ) .- 1) .* to_row(ss.ω)

    return State(ζ=new_ζ)
end

# ───────────────────────────────────────────────────────────────────────────
# Residuals
# ───────────────────────────────────────────────────────────────────────────

"""
    residuals(network, ζ::State, par::NKParameters, shocks::AbstractArray,
              ss::NKParameters, ps, st)

Compute residuals of model equations (NKPC and Euler equation).

Model equations:
    NKPC: Π_t = κ X_t + β E_t[Π_{t+1}]
    Euler: X_t = E_t[X_{t+1}] - σ⁻¹(ϕ_π Π_t + ϕ_y X_t - E_t[Π_{t+1}] - ζ_t)

Returns: (res_X_sum, res_π_sum) - sum of squared residuals
"""
function residuals(network, ζ::State, par::NKParameters, shocks::AbstractArray,
    ss::NKParameters, ps, st)

    # Get policy functions for time t
    X_t, π_t, st_new = policy(network, ζ, par, ps, st)

    # Convert policy from (mc, B) matrix to (B,) vector
    X = vec(X_t)
    π = vec(π_t)

    # Simulate next state
    next_ζ = step(ζ, shocks, par, ss)

    # Forecast policies one-step ahead
    X_next, π_next, st_new2 = policy(network, next_ζ, par, ps, st_new)

    # Compute expectation over MC draws
    E_X_next = vec(mean(X_next, dims=1))
    E_π_next = vec(mean(π_next, dims=1))

    # Compute residuals
    # Euler equation: X_t = E_t[X_{t+1}] - σ⁻¹(ϕ_π π_t + ϕ_y X_t - E_t[π_{t+1}] - ζ_t)
    res_X = X .- (E_X_next .- (1 ./ par.σ) .* (par.ϕ_pi .* π .+ par.ϕ_y .* X .- E_π_next .- ζ.ζ))

    # NKPC: π_t = β E_t[π_{t+1}] + κ X_t
    res_π = π .- (par.β .* E_π_next .+ ss.κ .* X)

    # Return sum of squared residuals
    res_X_sum = sum(res_X .^ 2)
    res_π_sum = sum(res_π .^ 2)

    return res_X_sum, res_π_sum, st_new2
end

# ───────────────────────────────────────────────────────────────────────────
# Analytical Solutions (User Implementation)
# ───────────────────────────────────────────────────────────────────────────

"""
    policy_analytical(state, par::NKParameters, ss)

Compute analytical policy functions using method of undetermined coefficients.
Returns (X, π) for given state and parameters.
den = (σ*(1 - ρ_A) + θ_Y)(1 - βρ_A) + κ*(θ_Π - ρ_A)
X̂ₜ = ((1 - β*ρ_A)/den) * ζₜ
Π̂ₜ = (κ/den) * ζₜ


"""
function policy_analytical(state, par::NKParameters)

    # Calculates the steady state derived parameters
    # NKPC slope coefficient
    κ = ((1 .- par.ϕ) .* (1 .- par.ϕ .* par.β) .* (par.σ .+ par.η)) ./ par.ϕ
    # println("Calculated κ: ", κ)

    # Composite parameter for natural rate equation
    ω = (1 .+ par.η) ./ (par.σ .+ par.η)
    # println("Calculated ω: ", ω)

    # Analytical Solutions
    # Denominator is common to both X and π
    den = (par.σ * (1 - par.ρ) + par.ϕ_y) * (1 - par.β * par.ρ) + κ * (par.ϕ_pi - par.ρ)
    # println("Calculated den: ", den)

    # Output gap X_t
    X = ((1 - par.β * par.ρ) / den) .* state.ζ

    # Inflation π_t
    π = (κ / den) .* state.ζ
    return X, π

end

"""
    policy_over_par(par::NKParameters, ranges::Ranges, par_name::Symbol; n_points=100)

Evaluate policy functions over a grid of parameter values.
Holds all other parameters fixed, varies par_name from ranges.


ζ_t = ρ_a * ζ_{t-1} + ε_t * σ_a * σ * (ρ_a - 1) * ω

sigma = σ_a * σ * (ρ - 1) * ω
    σ_a: standard deviation of natural rate shock
    σ: intertemporal elasticity of substitution
    ρ : persistence of natural rate shock
    ω : composite parameter (1 + η) / (σ + η)

[USER IMPLEMENTS]
"""
function policy_over_par(par::NKParameters, ranges::Ranges, par_name::Symbol, network, ps, st; n_points=100, shock_std=1.0, analytical=false)

    # Get the parameter range to vary
    param_range = getfield(ranges, par_name)
    param_values = range(param_range[1], param_range[2], length=n_points)

    # Prepare storage for results
    X_list = zeros(n_points)
    π_list = zeros(n_points)

    # Loop over parameter values
    for (i, val) in enumerate(param_values)
        # Create a new parameter struct with the varied parameter
        # (NKParameters is immutable, so we can't use setfield!)
        par_dict = Dict(k => getfield(par, k) for k in fieldnames(NKParameters))
        par_dict[par_name] = val
        par_varied = NKParameters(; par_dict...)

        # Calculate steady state of derived parameters
        par_current = steady_state(par_varied)

        # Set state to specified shock level (in standard deviations)
        sigma = par_current.σ_shock * par_current.σ * (par_current.ρ - 1) * par_current.ω
        state = State(ζ=shock_std * sigma)

        if analytical
            # Use analytical policy function
            X, π = policy_analytical(state, par_current)
        else
            # Use neural network policy function
            X_mat, π_mat, _ = policy(network, state, par_current, ps, st)
            X = mean(X_mat)  # Average over MC draws
            π = mean(π_mat)
        end

        X_list[i] = X
        π_list[i] = π
    end
    return Dict(:param_values => param_values, :X => X_list, :π => π_list)
end




"""
    policy_over_par_list(par::NKParameters, ranges::Ranges, par_names::Vector{Symbol}; n_points=100)

Apply policy_over_par for multiple parameters. Returns dict of results.
"""
function policy_over_par_list(par::NKParameters, ranges::Ranges, par_names::Vector{Symbol}, network, ps, st; n_points=100, shock_std=1.0, analytical=false)
    results = Dict{Symbol,Any}()

    for par_name in par_names
        results[par_name] = policy_over_par(par, ranges, par_name, network, ps, st; n_points=n_points, shock_std=shock_std, analytical=analytical)
    end

    return results
end

# Function to test if the revise is working in the REPL envrionment
function helloworld(name::AbstractString)
    println("Hello $name, You are the best coder! Keep at it. You are good. Keep going!")
end





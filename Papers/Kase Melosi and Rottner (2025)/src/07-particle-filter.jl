# ============================================================================
# Particle Filter for NK Model
# ============================================================================
# Extracted from notebooks/02-particle-filter-main.ipynb
#
# This file implements:
# - NKModel struct and constructors
# - ParticleFilter struct for state estimation
# - Bootstrap particle filter algorithm
# - Likelihood evaluation over parameter grids
# - Dataset generation for training meta-models
# ============================================================================

# ----------------------------------------------------------------------------
# Package Imports
# ----------------------------------------------------------------------------
using Distributions
using JLD2
using Lux
using LinearAlgebra
using Random
using Statistics
using Sobol

# ----------------------------------------------------------------------------
# ParticleFilter Structure
# ----------------------------------------------------------------------------

"""
    ParticleFilter{M <: AbstractModel}

Container for particle filter components.

# Fields
- `model::M`: NKModel with neural network policy
- `data::Matrix{Float64}`: Observed data (S × T) where S = observables, T = time periods
- `R::Matrix{Float64}`: Measurement error covariance (S × S)
- `R_inv::Matrix{Float64}`: Inverse of R (pre-computed for efficiency)
- `R_det::Float64`: Determinant of R (pre-computed for efficiency)

# Data format
data[1, :] = Interest rate (R)
data[2, :] = Output gap (X)
data[3, :] = Inflation (π)
"""
struct ParticleFilter{M <: AbstractModel}
    model::M
    data::Matrix{Float64}     # S × T (Observables × Time)
    R::Matrix{Float64}        # Measurement error covariance
    R_inv::Matrix{Float64}    # Pre-computed inverse
    R_det::Float64            # Pre-computed determinant
end

"""
    ParticleFilter(model::M, data::Matrix{Float64}, R::Matrix{Float64})

Construct ParticleFilter with pre-computed inverse and determinant of R.
"""
function ParticleFilter(model::M, data::Matrix{Float64}, R::Matrix{Float64}) where {M <: AbstractModel}
    R_inv = inv(R)
    R_det = det(R)
    return ParticleFilter{M}(model, data, R, R_inv, R_det)
end

# ----------------------------------------------------------------------------
# Log Probability Function
# ----------------------------------------------------------------------------

"""
    log_probability(error::Matrix{Float64}, R_inv::Matrix{Float64}, R_det::Float64)

Calculate log probability of observations given model predictions.

Assumes multivariate normal measurement error:
    log p(y | ŷ) = -0.5 * [S*log(2π) + log|R| + (y-ŷ)'R⁻¹(y-ŷ)]

# Mathematical Details
The quadratic form (y-ŷ)'R⁻¹(y-ŷ) is computed as:
1. R_inv * error → (S×S) × (S×P) = (S×P)
2. error .* (R_inv * error) → element-wise multiply
3. sum(..., dims=1) → sum over observables, giving Mahalanobis distance per particle

# Arguments
- `error::Matrix{Float64}`: Observation errors (data - model_prediction) [S × P]
- `R_inv::Matrix{Float64}`: Inverse of measurement error covariance [S × S]
- `R_det::Float64`: Determinant of measurement error covariance

# Returns
- `Vector{Float64}`: Log probability for each particle [P]
"""
function log_probability(error::Matrix{Float64}, R_inv::Matrix{Float64}, R_det::Float64)
    S = size(error, 1)  # Number of observables
    # Compute log p(y|ŷ) = -0.5 * [S*log(2π) + log|R| + (y-ŷ)'R⁻¹(y-ŷ)]
    log_prob = -0.5 * (S * log(2π) .+ log(R_det) .+ sum(error .* (R_inv * error), dims=1))
    return vec(log_prob)
end

# ----------------------------------------------------------------------------
# Kitagawa Resampling
# ----------------------------------------------------------------------------

"""
    kitagawa_resample(w_norm::Vector{Float64}, P::Int; aux=0.4532)

Systematic resampling using Kitagawa method.

This is more efficient and has lower variance than multinomial resampling.
Instead of P independent random draws, it uses a single random starting point
and evenly spaced samples through the cumulative distribution.

# Algorithm
1. Compute cumulative distribution: cum_dist = cumsum(w_norm)
2. Generate evenly-spaced thresholds: u = (aux + 0:P-1) / P
3. For each threshold, find which particle's range it falls into

# Intuition (Pie Chart Analogy)
Think of weights as slices of a pie. We throw P evenly-spaced darts at the pie.
Bigger slices (higher weight) get hit by more darts (more copies).

# Arguments
- `w_norm::Vector{Float64}`: Normalized weights (must sum to 1)
- `P::Int`: Number of particles
- `aux::Float64`: Random starting point offset (default: 0.4532)

# Returns
- `Vector{Int}`: Indices of resampled particles (length P)

# Example
```julia
w_norm = [0.5, 0.3, 0.1, 0.05, 0.05]  # 5 particles
idx = kitagawa_resample(w_norm, 5)    # e.g., [1, 1, 1, 2, 3]
# Particle 1 (weight 0.5) gets picked 3 times
# Particles 4,5 (small weights) die
```
"""
function kitagawa_resample(w_norm::Vector{Float64}, P::Int; aux=0.4532)
    # Cumulative sum of weights
    cum_dist = cumsum(w_norm)

    # Evenly-spaced thresholds: (aux, aux+1/P, aux+2/P, ..., aux+(P-1)/P) / P
    u = range(aux, step=1.0, length=P) ./ P

    # For each threshold, find which particle bin it falls into
    return searchsortedfirst.(Ref(cum_dist), u)
end

# ----------------------------------------------------------------------------
# Main Particle Filter
# ----------------------------------------------------------------------------

"""
    filter(pf::ParticleFilter, P::Int, shock_config::Shocks;
           burn=1000, sim=100, par=nothing, seed=nothing)

Bootstrap particle filter for state estimation and likelihood evaluation.

# Algorithm (for each time step t)
1. **Predict**: Compute observables (R, X, π) from current state using neural network
2. **Update**: Compute log p(y_t | ŷ_t) for each particle
3. **Resample**: Use Kitagawa resampling to eliminate low-weight particles
4. **Propagate**: Evolve state forward: ζ_t+1 = ρ·ζ_t + ε_t

# Mathematical Details
- State equation: ζ_t+1 = ρ·ζ_t + σ_shock·σ·(ρ-1)·ω·ε_t
- Measurement equation: y_t = h(ζ_t; θ_network) + v_t, where v_t ~ N(0, R)
- Log-likelihood: log p(y_1:T | θ) = Σ_t log[ (1/P) Σ_p w_p^t ]

# Numerical Stability
To avoid underflow when computing log(mean(exp(log_probs))):
    max_log = max(log_probs)
    w = exp(log_probs - max_log)
    log_lik += log(mean(w)) + max_log

# Arguments
- `pf::ParticleFilter`: Particle filter object (model + data + measurement covariance)
- `P::Int`: Number of particles (higher = more accurate but slower)
- `shock_config::Shocks`: Shock configuration (σ, antithetic)
- `burn::Int`: Burn-in periods before filtering starts (default: 1000)
- `sim::Int`: Number of periods to filter over (default: 100)
- `par::Union{Nothing, NKParameters}`: Parameters to use (default: use model.parameters)
- `seed::Union{Nothing, Int}`: Random seed for reproducibility (default: nothing)

# Returns
- `log_likelihood::Float64`: Total log-likelihood over filtered periods
- `filtered_out::NamedTuple`: Filtered estimates (mean across resampled particles)
    - `R::Vector{Float64}`: Filtered interest rate
    - `X::Vector{Float64}`: Filtered output gap
    - `π::Vector{Float64}`: Filtered inflation

# Example
```julia
pf = ParticleFilter(model, data_matrix, R_matrix)
ll, filtered = filter(pf, 1000, shock_config; sim=100, burn=1000, seed=1234)
plot(filtered.R)  # Smoothed estimate of interest rate
```
"""
function filter(pf::ParticleFilter, P::Int, shock_config::Shocks;
                burn=1000, sim=100, par=nothing, seed=nothing)

    # Set random seed for reproducibility
    if !isnothing(seed)
        Random.seed!(seed)
    end

    S = size(pf.data, 1)  # Number of observables

    # Use model parameters if not specified
    if isnothing(par)
        par = pf.model.parameters
    end

    # Initialize particles
    par_expanded = expand(par, P)
    ss = steady_state(par_expanded)
    state = initialize_state(par_expanded, P, ss)

    # Burn-in: Let particles settle to ergodic distribution
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

    # Main filter loop
    for t in 1:sim
        # ---- PREDICT ----
        # Compute observables from current state using neural network
        sim_out = sim_step(pf.model.network, state, par_expanded, pf.model.ps, pf.model.st)
        pf.model.st = sim_out[:st]

        # ---- UPDATE ----
        # Compute measurement errors: y_t - ŷ_t
        errors[1, :] = pf.data[1, t] .- vec(sim_out[:R])      # Interest rate errors
        errors[2, :] = pf.data[2, t] .- vec(sim_out[:X_t])    # Output gap errors
        errors[3, :] = pf.data[3, t] .- vec(sim_out[:π_t])    # Inflation errors

        # Compute log probabilities: log p(y_t | ŷ_t^p) for each particle p
        log_probs = log_probability(errors, pf.R_inv, pf.R_det)

        # Update log-likelihood with numerical stability
        max_log = maximum(log_probs)
        w = exp.(log_probs .- max_log)
        log_likelihood += log(mean(w)) + max_log

        # Normalize weights
        w_norm = w ./ sum(w)

        # ---- RESAMPLE ----
        # Systematic resampling (Kitagawa method)
        idx = kitagawa_resample(w_norm, P)
        state = State(ζ = state.ζ[:, idx])

        # Store filtered estimates (mean across resampled particles)
        filtered_R[t] = mean(sim_out[:R][idx])
        filtered_X[t] = mean(sim_out[:X_t][idx])
        filtered_π[t] = mean(sim_out[:π_t][idx])

        # ---- PROPAGATE ----
        # Evolve state forward one period
        shocks = draw_shocks(shock_config, 1, P)
        state = step(state, shocks, par_expanded, ss)
    end

    # Return results
    filtered_out = (
        R = filtered_R,
        X = filtered_X,
        π = filtered_π
    )

    return log_likelihood, filtered_out
end

# ----------------------------------------------------------------------------
# Likelihood Surface Estimation
# ----------------------------------------------------------------------------

"""
    filter_grid(pf::ParticleFilter, par_name::Symbol, shock_config::Shocks;
                n=25, sim=100, P=1000, burn=1000)

Evaluate log-likelihood over a grid of parameter values.

Useful for visualizing likelihood surfaces and verifying identification.

# Algorithm
1. Extract parameter bounds from model.ranges
2. Create grid of n evenly-spaced values
3. For each value, run particle filter and record log-likelihood

# Arguments
- `pf::ParticleFilter`: Particle filter object
- `par_name::Symbol`: Parameter to vary (e.g., :β, :σ, :σ_shock)
- `shock_config::Shocks`: Shock configuration
- `n::Int`: Number of grid points (default: 25)
- `sim::Int`: Number of periods to filter (default: 100)
- `P::Int`: Number of particles (default: 1000)
- `burn::Int`: Burn-in periods (default: 1000)

# Returns
- `grid::Vector{Float64}`: Parameter values (length n)
- `log_likelihood::Vector{Float64}`: Log-likelihood at each grid point (length n)

# Example
```julia
# Evaluate likelihood surface for β
grid, ll = filter_grid(pf, :β, shock_config; n=20, sim=100, P=1000)

# Plot likelihood surface
plot(grid, ll, xlabel="β", ylabel="Log-Likelihood")
vline!([true_β], linestyle=:dash, label="True value")
```
"""
function filter_grid(pf::ParticleFilter, par_name::Symbol, shock_config::Shocks;
                     n=25, sim=100, P=1000, burn=1000, random_draws=false)
    # Extract parameter bounds from model ranges
    bounds = getfield(pf.model.ranges, par_name)
    lower, upper = bounds[1], bounds[2]

    # Create grid — random or evenly spaced
    if random_draws
        grid = lower .+ (upper - lower) .* rand(n)
    else
        grid = collect(range(lower, upper, length=n))
    end

    # Storage for log-likelihoods
    log_likelihood = zeros(n)

    # Get base parameters
    base_par = pf.model.parameters

    # Evaluate log-likelihood at each grid point
    for (i, val) in enumerate(grid)
        # Update parameter value
        new_par = with_param(base_par, par_name, val)

        # Run particle filter
        ll, _ = filter(pf, P, shock_config; par=new_par, sim=sim, burn=burn)
        log_likelihood[i] = ll
    end

    return collect(grid), log_likelihood
end

# ----------------------------------------------------------------------------
# Dataset Generation for Meta-Model Training
# ----------------------------------------------------------------------------

"""
    filter_dataset(pf::ParticleFilter, shock_config::Shocks;
                   par_names=nothing, N=1000, sim=100, use_sobol=true,
                   P=1000, burn=1000, seed=nothing)

Generate dataset of (parameters, log-likelihood) pairs for training meta-models.

Uses Sobol sequences for efficient parameter space exploration.

# Algorithm
1. Generate N parameter draws using Sobol sequences (or uniform random)
2. For each parameter draw:
   - Run particle filter
   - Record log-likelihood
3. Return (parameters, log_likelihoods) dataset

# Arguments
- `pf::ParticleFilter`: Particle filter object
- `shock_config::Shocks`: Shock configuration
- `par_names::Union{Nothing, Tuple}`: Parameters to vary (default: all parameters)
- `N::Int`: Number of parameter draws (default: 1000)
- `sim::Int`: Simulation periods per filter run (default: 100)
- `use_sobol::Bool`: Use Sobol sequences vs random sampling (default: true)
- `P::Int`: Number of particles per filter run (default: 1000)
- `burn::Int`: Burn-in periods (default: 1000)
- `seed::Union{Nothing, Int}`: Random seed (default: nothing)

# Returns
- `NamedTuple` with fields:
    - `parameters::Matrix{Float64}`: Parameter draws (N × num_params)
    - `log_likelihoods::Vector{Float64}`: Log-likelihoods (N)

# Example
```julia
# Generate dataset varying β, σ, σ_shock
dataset = filter_dataset(pf, shock_config;
                        par_names=(:β, :σ, :σ_shock),
                        N=5000, use_sobol=true)

# Train meta-model
X_train = dataset.parameters
y_train = dataset.log_likelihoods
```
"""
function filter_dataset(pf::ParticleFilter, shock_config::Shocks;
                       par_names=nothing, N=1000, sim=100, use_sobol=true,
                       P=1000, burn=1000, seed=nothing)

    # Set random seed if provided
    if !isnothing(seed)
        Random.seed!(seed)
    end

    # Get base parameters
    par = pf.model.parameters

    # Determine which parameters to vary
    if isnothing(par_names)
        # Vary all parameters except κ and ω (which are derived)
        param_fields = fieldnames(typeof(par))
        par_names = tuple([f for f in param_fields if f ∉ (:κ, :ω)]...)
    end

    # Filter par_names to only those in ranges
    fieldnames_tuple = fieldnames(typeof(pf.model.ranges))
    par_names = tuple([f for f in par_names if f in fieldnames_tuple]...)

    # Generate parameter draws using Sobol sequences
    if use_sobol
        s = SobolSeq(length(par_names))
        sobol_points = [Sobol.next!(s) for _ in 1:N]
        sobol_matrix = hcat(sobol_points...)'  # N × num_varied_params
    else
        sobol_matrix = rand(N, length(par_names))
    end

    # Extract bounds for varied parameters
    bounds = [getfield(pf.model.ranges, f) for f in par_names]
    lower_bounds = first.(bounds)
    upper_bounds = last.(bounds)

    # Scale Sobol/random draws to parameter bounds
    scaled_params = lower_bounds' .+ sobol_matrix .* (upper_bounds' .- lower_bounds')

    # Create full parameter matrix (including fixed parameters)
    param_fields = fieldnames(typeof(par))
    values_replaced = replace([getfield(par, f) for f in param_fields], nothing => NaN)
    par_row_vector = values_replaced'
    par_matrix = repeat(par_row_vector, N, 1)

    # Update varied parameters
    par_indices = [findfirst(==(name), param_fields) for name in par_names]
    par_matrix[:, par_indices] .= scaled_params

    # Compute log-likelihoods
    log_likelihoods = zeros(N)

    for i in 1:N
        par_draw = NKParameters(par_matrix[i, :]...)
        log_likelihoods[i], _ = filter(pf, P, shock_config; par=par_draw, sim=sim, burn=burn, seed=seed)
    end

    # Return dataset
    return (parameters = par_matrix, log_likelihoods = log_likelihoods)
end

# ----------------------------------------------------------------------------
# Neural Network Surrogate for Log-Likelihood
# ----------------------------------------------------------------------------

"""
    pf_train!(network, ps, st, dataset; num_epochs=50, batch_size=10, lr=0.001,
              print_every=10, eta_min=1e-10, seed=nothing, train_sample_ratio=0.8)

Train neural network to approximate log-likelihood function.

# Mathematical Formulation
Minimize mean squared error between NN predictions and particle filter log-likelihoods:
    ℒ = (1/N) Σᵢ (ℓ̂(θᵢ; ψ) - ℓ_PF(θᵢ))²

where:
- ℓ̂(θ; ψ) is the NN surrogate with weights ψ
- ℓ_PF(θ) is the true particle filter log-likelihood
- θᵢ are parameter draws from Sobol sequence

# Training Details
- Optimizer: AdamW with cosine annealing schedule
- Learning rate: lr → eta_min following cosine schedule
- Data split: train_sample_ratio (default 0.8) for train/validation
- Uses first 8 parameters only (excludes derived κ, ω)

# Arguments
- `network`: Lux neural network architecture
- `ps::NamedTuple`: Initial network parameters
- `st::NamedTuple`: Initial network state
- `dataset::NamedTuple`: Dataset with `.parameters` (N × 10) and `.log_likelihoods` (N)
- `num_epochs::Int`: Number of training epochs (default: 50)
- `batch_size::Int`: Batch size for SGD (default: 10)
- `lr::Float64`: Initial learning rate (default: 0.001)
- `print_every::Int`: Print loss every N epochs (default: 10)
- `eta_min::Float64`: Minimum learning rate for cosine annealing (default: 1e-10)
- `seed::Union{Nothing,Int}`: Random seed for reproducibility
- `train_sample_ratio::Float64`: Fraction of data for training (default: 0.8)

# Returns
- `train_state::Lux.Training.TrainState`: Trained model state
- `loss_matrix::Matrix{Float64}`: Training history (num_epochs × 3)
    - Column 1: Epoch number
    - Column 2: Training loss (MSE)
    - Column 3: Validation loss (MSE)

# Example
```julia
dataset = filter_dataset(pf, shock_config; N=5000, use_sobol=true)
train_state, loss_history = pf_train!(network, ps, st, dataset; num_epochs=100)
plot_loss_evolution(loss_history)
```
"""
function pf_train!(network, ps, st, dataset; num_epochs=50, batch_size=10, lr=0.001,
    print_every=10, eta_min=1e-10, seed=nothing, train_sample_ratio=0.8)

    if !isnothing(seed)
        Random.seed!(seed)
    end

    # Defining the train_state
    train_state = Lux.Training.TrainState(network, ps, st, AdamW(lr))

    # Defing the loss matrix to record, epoch, train loss and validation loss
    loss_matrix = zeros(num_epochs, 3)  # Column 1: Epoch, Column 2: Train Loss, Column 3: Val Loss

    # Shuffle the dataset
    N = size(dataset.parameters, 1)
    shuffled_indices = shuffle(1:N)
    shuffled_dataset = (
        parameters = dataset.parameters[shuffled_indices, :],
        log_likelihoods = dataset.log_likelihoods[shuffled_indices]
    )

    # Split into training and validation sets as per train_sample_ratio
    num_train = Int(floor(train_sample_ratio * N))

    # Converting to Float32 for Lux compatibility and Machine Learning efficiency
    train_data = (
        parameters = Float32.(shuffled_dataset.parameters[1:num_train, 1:8]),  # Using only first 8 parameters, removing κ and ω
        log_likelihoods = Float32.(shuffled_dataset.log_likelihoods[1:num_train])
    )
    val_data = (
        parameters = Float32.(shuffled_dataset.parameters[num_train+1:end, 1:8]), # Using only first 8 parameters, removing κ and ω
        log_likelihoods = Float32.(shuffled_dataset.log_likelihoods[num_train+1:end])
    )

    # Defining the loss function (MSE)
    loss_function = MSELoss()

    # Training loop
    @showprogress for epoch in 1:num_epochs
        # Training Step

        # Randomly shuffle training data at the start of each epoch for batch sampling, could have used dataloader from MLUtils but this is simpler for now
        idx = randperm(num_train)
        epoch_train_loss = 0.0
        for i in 1:batch_size:num_train
            batch_indices = idx[i:min(i+batch_size-1, num_train)]
            x_batch = train_data.parameters[batch_indices, :]' # Transpose as Julia uses column-major order
            y_batch = train_data.log_likelihoods[batch_indices]' # Transpose for consistency
            data = (x_batch, y_batch)
                _, loss, _, train_state = Lux.Training.single_train_step!(
                        AutoZygote(), loss_function, data, train_state

            )
            epoch_train_loss += loss
        end

        # Update learning rate with cosine annealing
        current_lr = cosine_annealing_lr(lr, eta_min, epoch, num_epochs)
        train_state = Lux.Training.TrainState(network, train_state.parameters,
            train_state.states, AdamW(current_lr))

        # Normalizing the training loss
        num_batches = ceil(Int, num_train / batch_size)
        train_loss_normalized = epoch_train_loss / num_batches

        # Calculating the validation loss
        log_likelihoods_pred,_ = Lux.apply(network, val_data.parameters', train_state.parameters, train_state.states)
        val_loss = loss_function(log_likelihoods_pred, val_data.log_likelihoods')
        if epoch % print_every == 0
            println("Epoch $epoch, Train Loss: $train_loss_normalized, Validation Loss: $val_loss")
        end
        # Updating the loss matrix
        loss_matrix[epoch, 1] = epoch
        loss_matrix[epoch, 2] = train_loss_normalized
        loss_matrix[epoch, 3] = val_loss
    end
    return train_state, loss_matrix
end

"""
    predict_log_likelihood(network, ps, st, par_matrix)

Predict log-likelihood using trained neural network surrogate.

Fast alternative to running full particle filter.

# Mathematical Details
For parameter vector θ ∈ ℝ⁸, computes:
    ℓ̂(θ) = f_NN(θ; ψ)

where f_NN is the trained network with weights ψ.

# Arguments
- `network`: Trained Lux network
- `ps::NamedTuple`: Network parameters (weights)
- `st::NamedTuple`: Network state
- `par_matrix::Matrix{Float64}`: Parameter matrix (N × 8 or N × 10)
    Uses only first 8 columns (excludes κ, ω)

# Returns
- `Vector{Float64}`: Predicted log-likelihoods (N × 1)

# Example
```julia
# Predict likelihood for grid of β values
par_matrix, grid = build_param_matrix_fixed(pf, :β, 50)
ll_pred = predict_log_likelihood(network, ps, st, par_matrix)
plot(grid, ll_pred, label="NN Surrogate")
```
"""
function predict_log_likelihood(network, ps, st, par_matrix)
    N = size(par_matrix, 1)
    log_likelihoods_pred, _ = Lux.apply(network, Float32.(par_matrix[:, 1:8])', ps, st)  # Using only first 8 parameters
    return Array(log_likelihoods_pred)'  # Transpose to get (N x 1)
end

"""
    build_param_matrix_fixed(pf::ParticleFilter, par_name::Symbol, n::Int)

Build parameter matrix varying only one parameter.

Creates grid of parameter values holding all others fixed at baseline values.
Used for likelihood slice visualization.

# Algorithm
1. Extract bounds for `par_name` from `pf.model.ranges`
2. Create n evenly-spaced grid points
3. Replicate baseline parameters n times
4. Replace `par_name` column with grid values

# Arguments
- `pf::ParticleFilter`: Particle filter object (contains baseline parameters)
- `par_name::Symbol`: Parameter to vary (e.g., :β, :σ, :σ_shock)
- `n::Int`: Number of grid points

# Returns
- `par_matrix::Matrix{Float64}`: Parameter matrix (n × 8) varying only `par_name`
- `grid::Vector{Float64}`: Grid values for `par_name` (length n)

# Example
```julia
# Create grid varying β from 0.95 to 0.995
par_matrix, β_grid = build_param_matrix_fixed(pf, :β, 50)
# par_matrix[i, :] has β = β_grid[i], all others fixed
```
"""
function build_param_matrix_fixed(pf::ParticleFilter, par_name::Symbol, n::Int)
    bounds = getfield(pf.model.ranges, par_name)
    lower, upper = bounds[1], bounds[2]
    grid = range(lower, upper, length=n)
    N_params = length(fieldnames(typeof(pf.model.parameters))) - 2  # Exclude κ and ω
    par_matrix = zeros(n, N_params)
    base_par = pf.model.parameters
    param_fields = fieldnames(typeof(base_par))
    values_replaced = replace([getfield(base_par,f) for f in param_fields], nothing=>NaN) # if κ and ω are nothing replace with NaN
    par_row_vector = values_replaced[1:N_params]'  # Exclude κ and ω
    par_matrix .= repeat(par_row_vector, n, 1)
    par_index = findfirst(==( par_name), param_fields)
    par_matrix[:, par_index] .= collect(grid)
    return par_matrix, collect(grid)
end

# ============================================================================
# End of Particle Filter Module
# ============================================================================

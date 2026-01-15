# ═══════════════════════════════════════════════════════════════════════════
# NEURAL NETWORK ARCHITECTURE, TRAINING, AND LOSS FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════

# ───────────────────────────────────────────────────────────────────────────
# Custom Normalization Layer
# ───────────────────────────────────────────────────────────────────────────

"""
    NormalizeLayer <: Lux.AbstractLuxLayer

Custom Lux layer for min-max normalization to [-1, 1].

Mathematical operation:
    normalized = 2 * (x - lb) / (ub - lb) - 1
"""
struct NormalizeLayer <: Lux.AbstractLuxLayer
    lower_bound::Vector{Float32}
    upper_bound::Vector{Float32}
end

# Forward pass: x → (normalized_x, state)
#  Mathematical operation:
#     normalized = 2 * (x - lb) / (ub - lb) - 1

function (layer::NormalizeLayer)(x, ps, st)
    normalized = 2f0 .* (x .- layer.lower_bound) ./ (layer.upper_bound .- layer.lower_bound) .- 1f0
    return normalized, st
end

# ───────────────────────────────────────────────────────────────────────────
# Network Architecture
# ───────────────────────────────────────────────────────────────────────────

"""
    make_network(par::AbstractModelParameters, ranges::Ranges; kwargs...)

Create neural network architecture for policy function approximation.

Network structure:
    Input: [ζ; θ] where θ = [β, σ, η, ϕ, ϕ_π, ϕ_y, ρ, σ_ε]
    Hidden layers: N × [Linear → Activation]
    Output: [X, Π] scaled by scale_factor

Arguments:
- par: Parameters struct (to count number of parameters)
- ranges: Ranges struct (for normalization bounds)
- N_states: Number of state variables (default: 1)
- N_outputs: Number of policy outputs (default: 2)
- hidden: Hidden layer width (default: 64)
- layers: Number of hidden layers (default: 5)
- activation: Activation function (default: CELU)
- scale_factor: Output scaling (default: 1/100)

Returns: (network, ps, st)
"""
# function make_network(par::AbstractModelParameters, ranges::Ranges;
#     N_states=1, N_outputs=2, hidden=64,
#     layers=5, activation=Lux.celu, scale_factor=1 / 100.0)

#     # N_par: number of parameters (excluding κ and ω which are derived)
#     N_par = length(fieldnames(typeof(par))) - 2
#     N_input = N_states + N_par

#     # Extract normalization bounds
#     lower, upper = get_bounds(ranges)
#     lower = Float32.(lower)
#     upper = Float32.(upper)

#     # Normalization layer at input
#     norm_layer = NormalizeLayer(lower, upper)

#     # Convert to Float32 for network efficiency
#     scale_factor = Float32(scale_factor)

#     # Build the network
#     network = Chain(
#         norm_layer,
#         Dense(N_input, hidden, activation),
#         [Dense(hidden, hidden, activation) for _ in 1:(layers-1)]...,
#         Dense(hidden, N_outputs),
#         x -> x .* scale_factor  # Scale output
#     )

#     # Initialize parameters and state
#     rng = Random.default_rng()
#     ps, st = Lux.setup(rng, network)

#     return network, ps, st
# end

# NK Simple 3Equation Model 
# Adapting for the case where state variable is excluded from input like in the case of network for the PF with NK DSGE
function make_network(par::AbstractModelParameters, ranges::Ranges;
    N_states=1, N_outputs=2, hidden=64,
    layers=5, activation=Lux.celu, scale_factor=1 / 100.0, include_state::Bool=true)

    # N_par: number of parameters (excluding κ and ω which are derived)
    N_par = length(fieldnames(typeof(par))) - 2

    if include_state
        N_input = N_states + N_par
        # Extract normalization bounds
        lower, upper = get_bounds(ranges)
        lower = Float32.(lower)
        upper = Float32.(upper)
    else
        N_input = N_par
        lower, upper = get_bounds(ranges)
        lower = Float32.(lower[2:end])  # Exclude state variable bounds
        upper = Float32.(upper[2:end])  # Exclude state variable bounds
    end

    # Normalization layer at input
    norm_layer = NormalizeLayer(lower, upper)

    # Convert to Float32 for network efficiency
    scale_factor = Float32(scale_factor)

    # Build the network
    network = Chain(
        norm_layer,
        Dense(N_input, hidden, activation),
        [Dense(hidden, hidden, activation) for _ in 1:(layers-1)]...,
        Dense(hidden, N_outputs),
        x -> x .* scale_factor  # Scale output
    )

    # Initialize parameters and state
    rng = Random.default_rng()
    ps, st = Lux.setup(rng, network)

    return network, ps, st
end



# Type Stable network construction
function make_network_stable(par::AbstractModelParameters, ranges::Ranges, ::Val{L};
    N_states=1, N_outputs=2, hidden=64,
    activation=Lux.celu, scale_factor=1/100.0) where {L}

    N_par = length(fieldnames(typeof(par))) - 2
    N_input = N_states + N_par

    lower, upper = get_bounds(ranges)
    lower = Float32.(lower)
    upper = Float32.(upper)

    norm_layer = NormalizeLayer(lower, upper)
    scale_factor = Float32(scale_factor)

    network = Chain(
        norm_layer,
        Dense(N_input, hidden, activation),
        ntuple(_ -> Dense(hidden, hidden, activation), Val(L-1))...,
        Dense(hidden, N_outputs),
        x -> x .* scale_factor
    )

    rng = Random.default_rng()
    ps, st = Lux.setup(rng, network)

    return network, ps, st
end


"""
    make_kase_network(input_dim::Int, output_dim::Int)

Create neural network following Kase et al. architecture:
- 5 hidden layers with 128 neurons each
- Activation: SiLU (swish) for first 4 layers, Leaky ReLU for layer 5
- Uses Lux.jl framework

Architecture from Kase et al. footnote 29.
"""
function make_kase_network(input_dim::Int, output_dim::Int)
    network = Chain(
        Dense(input_dim, 128, Lux.swish),
        Dense(128, 128, Lux.swish),
        Dense(128, 128, Lux.swish),
        Dense(128, 128, Lux.swish),
        Dense(128, 128, Lux.leakyrelu),
        Dense(128, output_dim)
    )

    # Initialize parameters and state
    rng = Random.default_rng()
    ps, st = Lux.setup(rng, network)

    return network, ps, st
end

# ───────────────────────────────────────────────────────────────────────────
# Loss Functions
# ───────────────────────────────────────────────────────────────────────────

"""
    loss_fn(ps, st, network, par::AbstractModelParameters, state::State,
            shocks::AbstractArray, ss::AbstractModelParameters; λ_X=1.0, λ_π=0.1)

Compute weighted loss from residuals.

Loss = λ_X · ||res_X||² + λ_π · ||res_π||²
"""
function loss_fn(ps, st, network, par::AbstractModelParameters, state::State,
    shocks::AbstractArray, ss::AbstractModelParameters;
    λ_X=1.0, λ_π=0.1)
    # Call residuals function
    res_X_sum, res_π_sum, st_new = residuals(network, state, par, shocks, ss, ps, st)

    # Weighted combination
    loss = λ_X * res_X_sum + λ_π * res_π_sum

    return loss, res_X_sum, res_π_sum, st_new
end

"""
    loss_fn_wrapper(model, ps, st, data)

Wrapper for loss_fn to work with Lux.Training API.

Unpacks data tuple (par, state, shocks, ss, weights) and calls loss_fn.
weights is a NamedTuple with λ_X and λ_π.
Returns loss, new state, and stats named tuple with residual components.
"""
function loss_fn_wrapper(model, ps, st, data)
    par, state, shocks, ss, weights = data
    loss, res_X_sum, res_π_sum, st_new = loss_fn(ps, st, model, par, state, shocks, ss; λ_X=weights.λ_X, λ_π=weights.λ_π)
    return loss, st_new, (res_X=res_X_sum, res_π=res_π_sum)
end

"""
    loss_fn_std_normalized(ps, st, network, par, state, shocks, ss; λ_X=1.0, λ_π=1.0)

Loss with per-component standard deviation normalization.
Normalizes each residual by its own batch standard deviation before weighting.
This ensures gradients are balanced even when residuals have different scales.
"""
function loss_fn_std_normalized(ps, st, network, par::AbstractModelParameters, state::State,
    shocks::AbstractArray, ss::AbstractModelParameters;
    λ_X=1.0, λ_π=1.0, eps=1f-8)

    # Get policy functions for time t
    X_t, π_t, st_new = policy(network, state, par, ps, st)
    X = vec(X_t)
    π = vec(π_t)

    # Simulate next state and get expectations
    next_state = step(state, shocks, par, ss)
    X_next, π_next, st_new2 = policy(network, next_state, par, ps, st_new)
    E_X_next = vec(mean(X_next, dims=1))
    E_π_next = vec(mean(π_next, dims=1))

    # Compute ELEMENTWISE residuals (not summed yet)
    res_X_vec = X .- (E_X_next .- (1 ./ par.σ) .* (par.ϕ_pi .* π .+ par.ϕ_y .* X .- E_π_next .- state.ζ))
    res_π_vec = π .- (par.β .* E_π_next .+ ss.κ .* X)

    # Normalize by standard deviation across batch (z-score normalization)
    std_X = std(res_X_vec) + eps
    std_π = std(res_π_vec) + eps

    res_X_normalized = res_X_vec ./ std_X
    res_π_normalized = res_π_vec ./ std_π

    # Compute loss on normalized residuals
    loss = λ_X * sum(res_X_normalized .^ 2) + λ_π * sum(res_π_normalized .^ 2)

    # Return raw (unnormalized) residuals for tracking
    res_X_sum = sum(res_X_vec .^ 2)
    res_π_sum = sum(res_π_vec .^ 2)

    return loss, res_X_sum, res_π_sum, st_new2
end

"""
    loss_fn_std_normalized_wrapper(model, ps, st, data)

Wrapper for loss_fn_std_normalized to work with Lux.Training API.
"""
function loss_fn_std_normalized_wrapper(model, ps, st, data)
    par, state, shocks, ss, weights = data
    loss, res_X_sum, res_π_sum, st_new = loss_fn_std_normalized(
        ps, st, model, par, state, shocks, ss;
        λ_X=weights.λ_X, λ_π=weights.λ_π
    )
    return loss, st_new, (res_X=res_X_sum, res_π=res_π_sum)
end

# ───────────────────────────────────────────────────────────────────────────
# Learning Rate Scheduling
# ───────────────────────────────────────────────────────────────────────────

"""
    cosine_annealing_lr(initial_lr, eta_min, epoch, num_epochs)

Compute learning rate using cosine annealing schedule.

Mathematical formula:
    lr_t = η_min + (1/2)(lr_0 - η_min)(1 + cos(πt/T))

where:
- lr_0 is the initial learning rate
- η_min is the minimum learning rate
- t is the current epoch
- T is the total number of epochs
"""
function cosine_annealing_lr(initial_lr, eta_min, epoch, num_epochs)
    return eta_min + (initial_lr - eta_min) * (1 + cos(pi * epoch / num_epochs)) / 2
end

# ───────────────────────────────────────────────────────────────────────────
# Training Functions
# ───────────────────────────────────────────────────────────────────────────

"""
    train_simple!(network, ps, st, ranges, shock_config; kwargs...)

Simple training loop for neural network policy approximation.

Arguments:
- network: Lux neural network
- ps: Network parameters
- st: Network state
- ranges: Ranges struct for parameter sampling
- shock_config: Shocks configuration

Keyword Arguments:
- num_epochs: Number of training epochs (default: 1000)
- batch: Batch size for parameter sampling (default: 100)
- mc: Number of Monte Carlo draws (default: 10)
- lr: Learning rate (default: 0.001)
 - λ_X: Weight on Euler/IS residual (default: 1.0)
 - λ_π: Weight on NKPC residual (default: 1.0)

Returns: TrainState with trained parameters
"""
function train_simple!(network, ps, st, ranges, shock_config;
    num_epochs=1000, batch=100, mc=10, lr=0.001, λ_X=1.0, λ_π=0.1)

    # Defining the train state
    train_state = Lux.Training.TrainState(network, ps, st, Adam(lr))
    priors = prior_distribution(ranges)

    # Create weights NamedTuple
    weights = (λ_X=λ_X, λ_π=λ_π)

    # Running the training loop
    for epoch in 1:num_epochs
        # Fresh samples each iteration
        par = draw_parameters(priors, batch)
        ss = steady_state(par)
        state = initialize_state(par, batch, ss)
        shocks = draw_shocks(shock_config, mc, batch)
        data = (par, state, shocks, ss, weights)
        _, loss, stats, train_state = Lux.Training.single_train_step!(
            AutoZygote(), loss_fn_wrapper, data, train_state
        )

        # Print loss every 100 epochs
        if epoch % 100 == 0
            println("Epoch $epoch, Loss: $loss",
                ", res_X: $(stats.res_X), res_π: $(stats.res_π)")
        end

    end

    return train_state
end

"""
    train_simple_internal!(network, ps, st, ranges, shock_config; kwargs...)

Training with fresh samples + internal gradient steps + cosine annealing.
Combines the reliability of train_simple! with the efficiency of multiple gradient steps.

This is the RECOMMENDED approach following paper's configuration:
- Fresh parameter/state samples every epoch (ensures ergodic distribution)
- Multiple gradient steps per epoch (internal=5 from paper)
- Cosine annealing learning rate schedule
- No state stepping (avoids training distribution issues)

Arguments:
- network: Lux neural network
- ps: Network parameters
- st: Network state
- ranges: Ranges struct for parameter sampling
- shock_config: Shocks configuration

Keyword Arguments:
- num_epochs: Number of training epochs (default: 1000)
- batch: Batch size for parameter sampling (default: 100)
- mc: Number of Monte Carlo draws (default: 10)
- lr: Initial learning rate (default: 0.001)
- internal: Number of gradient steps per epoch (default: 5, following paper)
- λ_X: Weight on Euler/IS residual (default: 1.0)
- λ_π: Weight on NKPC residual (default: 1.0)
- print_every: Print frequency in epochs (default: 100)
- eta_min: Minimum learning rate for cosine annealing (default: 1e-10)

Returns: (TrainState, loss_dict) with trained parameters and loss history
"""
function train_simple_internal!(network, ps, st, ranges, shock_config;
    num_epochs=1000, batch=100, mc=10, lr=0.001, internal=5,
    λ_X=1.0, λ_π=1.0, print_every=100, eta_min=1e-10)

    train_state = Lux.Training.TrainState(network, ps, st, Adam(lr))
    priors = prior_distribution(ranges)
    weights = (λ_X=λ_X, λ_π=λ_π)

    loss_dict = Dict(:iteration => [], :loss => [], :res_X => [], :res_π => [])

    # Create progress meter
    prog = Progress(num_epochs; dt=1.0, desc="Training... ", showspeed=true)

    for epoch in 1:num_epochs
        # Fresh samples each iteration (ensures ergodic distribution)
        par = draw_parameters(priors, batch)
        ss = steady_state(par)
        state = initialize_state(par, batch, ss)
        shocks = draw_shocks(shock_config, mc, batch)
        data = (par, state, shocks, ss, weights)

        # Multiple gradient steps per epoch (following paper's internal=5)
        loss = nothing
        stats = nothing
        for _ in 1:internal
            _, loss, stats, train_state = Lux.Training.single_train_step!(
                AutoZygote(), loss_fn_wrapper, data, train_state
            )
        end

        # Update learning rate with cosine annealing
        current_lr = cosine_annealing_lr(lr, eta_min, epoch, num_epochs)
        train_state = Lux.Training.TrainState(network, train_state.parameters,
            train_state.states, Adam(current_lr))

        if epoch % print_every == 0
            println("Epoch $epoch, Loss: $loss, res_X: $(stats.res_X), res_π: $(stats.res_π)")
            push!(loss_dict[:iteration], epoch)
            push!(loss_dict[:loss], loss)
            push!(loss_dict[:res_X], stats.res_X)
            push!(loss_dict[:res_π], stats.res_π)
        end

        # Update progress meter
        next!(prog)
    end

    return train_state, loss_dict
end

"""
    train!(network, ps, st, ranges, shock_config; kwargs...)

Advanced training loop with learning rate scheduling, parameter redraws, and state stepping.

Arguments:
- network: Lux neural network
- ps: Network parameters
- st: Network state
- ranges: Ranges struct for parameter sampling
- shock_config: Shocks configuration

Keyword Arguments:
- num_epochs: Number of training epochs (default: 1000)
- batch: Batch size for parameter sampling (default: 100)
- mc: Number of Monte Carlo draws (default: 10)
- lr: Initial learning rate (default: 0.001)
- internal: Number of gradient steps per epoch (default: 1)
- print_every: Print frequency in epochs (default: 100)
- redraw_every: Parameter redraw frequency in epochs (default: 100)
- num_steps: Number of state evolution steps per epoch (default: 1)
- eta_min: Minimum learning rate for cosine annealing (default: 1e-10)
- λ_X: Weight on Euler/IS residual (default: 1.0)
- λ_π: Weight on NKPC residual (default: 1.0)

Returns: (train_state, loss_dict)
- train_state: TrainState with trained parameters
- loss_dict: Dictionary with training history
"""
function train!(network, ps, st, ranges, shock_config;
    num_epochs=1000, batch=100, mc=10, lr=0.001, internal=1, print_every=100,
    redraw_every=1, num_steps=1, eta_min=1e-10, λ_X=1.0, λ_π=1.0)



    # Defining the train state
    train_state = Lux.Training.TrainState(network, ps, st, Adam(lr))
    priors = prior_distribution(ranges)

    # Creating a dictionary to track loss
    loss_dict = Dict(:iteration => [], :running_loss => [], :loss => [],
        :res_X => [], :res_π => [])

    # Create weights NamedTuple
    weights = (λ_X=λ_X, λ_π=λ_π)

    # Initialize the par, ss, state
    par = draw_parameters(priors, batch)
    ss = steady_state(par)
    state = initialize_state(par, batch, ss)

    # Initializing the running loss
    running_loss = 0.0

    # Create progress meter with iterations/sec display
    prog = Progress(num_epochs; dt=1.0, desc="Training... ", showspeed=true)

    # Running the training loop
    for epoch in 1:num_epochs
        # Redraw parameters FIRST (at epoch start, not end)
        # This ensures data is created with current parameters
        if epoch % redraw_every == 0
            par = draw_parameters(priors, batch)
            ss = steady_state(par)
        end

        # Step state forward BEFORE creating data
        # This ensures data uses the evolved state
        for _ in 1:num_steps
            step_shocks = draw_shocks(shock_config, 1, batch)
            state = step(state, step_shocks, par, ss)
        end

        # NOW create data with current par and state
        shocks = draw_shocks(shock_config, mc, batch)
        data = (par, state, shocks, ss, weights)

        # Starting loss and stats
        loss = nothing
        stats = nothing

        # Inner loop for 'internal' gradient steps
        for o in 1:internal
            _, loss, stats, train_state = Lux.Training.single_train_step!(
                AutoZygote(), loss_fn_wrapper, data, train_state
            )
        end

        # Updating the running loss
        running_loss += loss / batch

        # Updating the lr using cosine annealing
        current_lr = cosine_annealing_lr(lr, eta_min, epoch, num_epochs)

        # Updating the train_state with new lr
        train_state = Lux.Training.TrainState(network, train_state.parameters,
            train_state.states, Adam(current_lr))

        # Print loss every print_every epochs
        if epoch % print_every == 0
            println("Epoch $epoch,  Running Loss: $running_loss, Loss: $loss, res_X: $(stats.res_X), res_π: $(stats.res_π)")
            push!(loss_dict[:iteration], epoch)
            push!(loss_dict[:running_loss], running_loss / print_every)
            push!(loss_dict[:loss], loss)
            push!(loss_dict[:res_X], stats.res_X)
            push!(loss_dict[:res_π], stats.res_π)
            println("Current Weights: λ_X=$(weights.λ_X), λ_π=$(weights.λ_π)")
            # Resetting the running loss
            running_loss = 0.0
        end

        # Update progress meter
        next!(prog)

    end

    return train_state, loss_dict
end

# Defining a function to calculate the gradient

grad_l2(x) = sqrt(_grad_sumsq(x))
function _grad_sumsq(x)
    if x === nothing
        return 0.0
    elseif x isa AbstractArray
        return sum(abs2, x)          # elementwise sum of squares
    elseif x isa Number
        return abs2(x)
    elseif x isa NamedTuple
        return sum(_grad_sumsq, values(x))
    elseif x isa Tuple
        return sum(_grad_sumsq, x)
    elseif x isa Dict
        return sum(_grad_sumsq, values(x))
    else
        return 0.0
    end
end




function train_clipped_GR!(network, ps, st, ranges, shock_config;
    num_epochs=1000, batch=100, mc=10, lr=0.001, internal=1, print_every=100,
    redraw_every=100, num_steps=1, eta_min=1e-10, λ_X=1.0, λ_π=0.1)


    # ✓ FIX 1: Create optimizer WITH gradient clipping
    opt = OptimiserChain(ClipGrad(1.0), Adam(lr))

    # Defining the train state
    train_state = Lux.Training.TrainState(network, ps, st, opt)
    priors = prior_distribution(ranges)

    # Creating a dictionary to track loss and gradient norm
    loss_dict = Dict(:iteration => [], :running_loss => [], :loss => [],
        :res_X => [], :res_π => [], :grad_norm => [])

    # Create weights NamedTuple
    weights = (λ_X=λ_X, λ_π=λ_π)

    # Initialize the par, ss, state
    par = draw_parameters(priors, batch)
    ss = steady_state(par)
    state = initialize_state(par, batch, ss)

    # Initializing the running loss
    running_loss = 0.0

    # Create progress meter with iterations/sec display
    prog = Progress(num_epochs; dt=1.0, desc="Training... ", showspeed=true)

    # Running the training loop
    for epoch in 1:num_epochs
        # Draw shocks every epoch
        shocks = draw_shocks(shock_config, mc, batch)
        data = (par, state, shocks, ss, weights)

        # Starting loss and stats
        loss = nothing
        stats = nothing
        grad_norm = nothing

        # Inner loop for 'internal' gradient steps
        for o in 1:internal
            gs, loss, stats, train_state = Lux.Training.compute_gradients(
                AutoZygote(), loss_fn_wrapper, data, train_state
            )


            # Calculate gradient norm BEFORE clipping (filter out Nothing) as some layers like NormalizeLayer may not have gradients as they don't have parameters so their gradients are Nothing
            grad_norm = grad_l2(gs)

            # Apply optimizer (includes ClipGrad)
            train_state = Lux.Training.apply_gradients!(train_state, gs)

        end

        # Updating the running loss
        running_loss += loss / batch

        # Updating the lr using cosine annealing
        current_lr = cosine_annealing_lr(lr, eta_min, epoch, num_epochs)

        # Updating the train_state with new lr
        new_opt = OptimiserChain(ClipGrad(1.0), Adam(current_lr))
        train_state = Lux.Training.TrainState(network, train_state.parameters,
            train_state.states, new_opt)

        # Print loss every print_every epochs
        if epoch % print_every == 0
            println("Epoch $epoch, Grad Norm: $grad_norm, Running Loss: $running_loss, Loss: $loss",
                ", res_X: $(stats.res_X), res_π: $(stats.res_π)")
            push!(loss_dict[:iteration], epoch)
            push!(loss_dict[:running_loss], running_loss / print_every)
            push!(loss_dict[:loss], loss)
            push!(loss_dict[:res_X], stats.res_X)
            push!(loss_dict[:res_π], stats.res_π)
            push!(loss_dict[:grad_norm], grad_norm)
            # Resetting the running loss
            running_loss = 0.0
        end

        # Redrawing the parameters and ss after redraw_every number of epochs
        # Note: State continues to evolve (NOT reinitialized) following paper's approach
        if epoch % redraw_every == 0
            par = draw_parameters(priors, batch)
            ss = steady_state(par)
        end

        # Iterating state num_steps ahead
        for _ in 1:num_steps
            shocks = draw_shocks(shock_config, 1, batch)
            state = step(state, shocks, par, ss)
        end

        # Update progress meter
        next!(prog)

    end

    return train_state, loss_dict
end

"""
    train_adaptive_weights!(network, ps, st, ranges, shock_config; kwargs...)

Training loop with adaptive loss reweighting based on residual magnitudes.
Automatically balances loss components by tracking exponential moving average of residuals.

Arguments:
- network: Lux neural network
- ps: Network parameters
- st: Network state
- ranges: Ranges struct for parameter sampling
- shock_config: Shocks configuration

Keyword Arguments:
- num_epochs: Number of training epochs (default: 1000)
- batch: Batch size for parameter sampling (default: 100)
- mc: Number of Monte Carlo draws (default: 10)
- lr: Initial learning rate (default: 0.001)
- internal: Number of gradient steps per epoch (default: 1)
- print_every: Print frequency in epochs (default: 100)
- redraw_every: Parameter redraw frequency in epochs (default: 100)
- num_steps: Number of state evolution steps per epoch (default: 1)
- eta_min: Minimum learning rate for cosine annealing (default: 1e-10)
- alpha: EMA smoothing factor (default: 0.9)

Returns: (train_state, loss_dict)
"""
function train_adaptive_weights!(network, ps, st, ranges, shock_config;
    num_epochs=1000, batch=100, mc=10, lr=0.001, internal=1, print_every=100,
    redraw_every=100, num_steps=1, eta_min=1e-10, alpha=0.9)

    opt = OptimiserChain(ClipGrad(1.0), Adam(lr))
    train_state = Lux.Training.TrainState(network, ps, st, opt)
    priors = prior_distribution(ranges)

    loss_dict = Dict(:iteration => [], :running_loss => [], :loss => [],
        :res_X => [], :res_π => [], :res_X_rms => [], :res_π_rms => [],
        :weight_X => [], :weight_π => [])

    # Initialize exponential moving averages for residuals
    ema_res_X = 1.0
    ema_res_π = 1.0

    par = draw_parameters(priors, batch)
    ss = steady_state(par)
    state = initialize_state(par, batch, ss)
    running_loss = 0.0

    prog = Progress(num_epochs; dt=1.0, desc="Training (adaptive)... ", showspeed=true)

    for epoch in 1:num_epochs
        shocks = draw_shocks(shock_config, mc, batch)

        # Compute adaptive weights based on EMA of residuals
        # Invert so larger residuals get smaller weights
        weight_X = ema_res_π / (ema_res_X + ema_res_π)
        weight_π = ema_res_X / (ema_res_X + ema_res_π)

        weights = (λ_X=weight_X, λ_π=weight_π)
        data = (par, state, shocks, ss, weights)

        loss = nothing
        stats = nothing

        for o in 1:internal
            _, loss, stats, train_state = Lux.Training.single_train_step!(
                AutoZygote(), loss_fn_wrapper, data, train_state
            )
        end

        # Update EMA of residuals
        res_X_rms = sqrt(stats.res_X / batch)
        res_π_rms = sqrt(stats.res_π / batch)
        ema_res_X = alpha * ema_res_X + (1 - alpha) * res_X_rms
        ema_res_π = alpha * ema_res_π + (1 - alpha) * res_π_rms

        running_loss += loss / batch

        # Update learning rate
        current_lr = cosine_annealing_lr(lr, eta_min, epoch, num_epochs)
        new_opt = OptimiserChain(ClipGrad(1.0), Adam(current_lr))
        train_state = Lux.Training.TrainState(network, train_state.parameters,
            train_state.states, new_opt)

        if epoch % print_every == 0
            println("Epoch $epoch, Loss: $loss")
            println("  RMS residuals - res_X: $res_X_rms, res_π: $res_π_rms")
            println("  Adaptive weights - λ_X: $weight_X, λ_π: $weight_π")
            println("  EMA residuals - ema_X: $ema_res_X, ema_π: $ema_res_π")

            push!(loss_dict[:iteration], epoch)
            push!(loss_dict[:running_loss], running_loss / print_every)
            push!(loss_dict[:loss], loss)
            push!(loss_dict[:res_X], stats.res_X)
            push!(loss_dict[:res_π], stats.res_π)
            push!(loss_dict[:res_X_rms], res_X_rms)
            push!(loss_dict[:res_π_rms], res_π_rms)
            push!(loss_dict[:weight_X], weight_X)
            push!(loss_dict[:weight_π], weight_π)

            running_loss = 0.0
        end

        if epoch % redraw_every == 0
            par = draw_parameters(priors, batch)
            ss = steady_state(par)
            state = initialize_state(par, batch, ss)
        end

        for _ in 1:num_steps
            shocks = draw_shocks(shock_config, 1, batch)
            state = step(state, shocks, par, ss)
        end

        next!(prog)
    end

    return train_state, loss_dict
end

"""
    train_std_normalized!(network, ps, st, ranges, shock_config; kwargs...)

Training loop with standard deviation normalized loss.
Uses per-component std normalization to balance gradient magnitudes.

Arguments:
- network: Lux neural network
- ps: Network parameters
- st: Network state
- ranges: Ranges struct for parameter sampling
- shock_config: Shocks configuration

Keyword Arguments:
- num_epochs: Number of training epochs (default: 1000)
- batch: Batch size for parameter sampling (default: 100)
- mc: Number of Monte Carlo draws (default: 10)
- lr: Initial learning rate (default: 0.001)
- internal: Number of gradient steps per epoch (default: 1)
- print_every: Print frequency in epochs (default: 100)
- redraw_every: Parameter redraw frequency in epochs (default: 100)
- num_steps: Number of state evolution steps per epoch (default: 1)
- eta_min: Minimum learning rate for cosine annealing (default: 1e-10)
- λ_X: Weight on Euler/IS residual (default: 1.0)
- λ_π: Weight on NKPC residual (default: 1.0)

Returns: (train_state, loss_dict)
"""
function train_std_normalized!(network, ps, st, ranges, shock_config;
    num_epochs=1000, batch=100, mc=10, lr=0.001, internal=1, print_every=100,
    redraw_every=100, num_steps=1, eta_min=1e-10, λ_X=1.0, λ_π=1.0)

    opt = OptimiserChain(ClipGrad(1.0), Adam(lr))
    train_state = Lux.Training.TrainState(network, ps, st, opt)
    priors = prior_distribution(ranges)

    loss_dict = Dict(:iteration => [], :running_loss => [], :loss => [],
        :res_X => [], :res_π => [], :res_X_rms => [], :res_π_rms => [])

    weights = (λ_X=λ_X, λ_π=λ_π)

    par = draw_parameters(priors, batch)
    ss = steady_state(par)
    state = initialize_state(par, batch, ss)
    running_loss = 0.0

    prog = Progress(num_epochs; dt=1.0, desc="Training (std-normalized)... ", showspeed=true)

    for epoch in 1:num_epochs
        shocks = draw_shocks(shock_config, mc, batch)
        data = (par, state, shocks, ss, weights)

        loss = nothing
        stats = nothing

        for o in 1:internal
            _, loss, stats, train_state = Lux.Training.single_train_step!(
                AutoZygote(), loss_fn_std_normalized_wrapper, data, train_state
            )
        end

        running_loss += loss / batch

        # Update learning rate
        current_lr = cosine_annealing_lr(lr, eta_min, epoch, num_epochs)
        new_opt = OptimiserChain(ClipGrad(1.0), Adam(current_lr))
        train_state = Lux.Training.TrainState(network, train_state.parameters,
            train_state.states, new_opt)

        if epoch % print_every == 0
            # Compute RMS residuals for display
            res_X_rms = sqrt(stats.res_X / batch)
            res_π_rms = sqrt(stats.res_π / batch)

            println("Epoch $epoch, Loss: $loss")
            println("  RMS residuals - res_X: $res_X_rms, res_π: $res_π_rms")
            println("  Raw residuals - res_X: $(stats.res_X), res_π: $(stats.res_π)")

            push!(loss_dict[:iteration], epoch)
            push!(loss_dict[:running_loss], running_loss / print_every)
            push!(loss_dict[:loss], loss)
            push!(loss_dict[:res_X], stats.res_X)
            push!(loss_dict[:res_π], stats.res_π)
            push!(loss_dict[:res_X_rms], res_X_rms)
            push!(loss_dict[:res_π_rms], res_π_rms)

            running_loss = 0.0
        end

        # Add diagnostics at epoch 100
        if epoch == 100
            X_debug, π_debug, _ = policy(network, state, par, train_state.parameters, train_state.states)
            println("\n  Policy output statistics at epoch $epoch:")
            println("    X: mean=$(mean(X_debug)), std=$(std(X_debug)), range=[$(minimum(X_debug)), $(maximum(X_debug))]")
            println("    π: mean=$(mean(π_debug)), std=$(std(π_debug)), range=[$(minimum(π_debug)), $(maximum(π_debug))]\n")
        end

        if epoch % redraw_every == 0
            par = draw_parameters(priors, batch)
            ss = steady_state(par)
            state = initialize_state(par, batch, ss)
        end

        for _ in 1:num_steps
            shocks = draw_shocks(shock_config, 1, batch)
            state = step(state, shocks, par, ss)
        end

        next!(prog)
    end

    return train_state, loss_dict
end



"""
    loss_fn_normalized(ps, st, network, par, state, shocks, ss; λ_X=1.0, λ_π=1.0)

Loss function with automatic residual normalization to balance gradient magnitudes.
Divides each residual by sqrt(batch) to get RMS (root mean square) values.
"""
function loss_fn_normalized(ps, st, network, par::AbstractModelParameters, state::State,
    shocks::AbstractArray, ss::AbstractModelParameters;
    λ_X=1.0, λ_π=1.0)

    # Compute residuals (sums of squares)
    res_X_sum, res_π_sum, st_new = residuals(network, state, par, shocks, ss, ps, st)

    # Get batch size
    batch = length(state.ζ)

    # Normalize to RMS (root mean square) - puts them on comparable scale
    res_X_rms = sqrt(res_X_sum / batch)
    res_π_rms = sqrt(res_π_sum / batch)

    # Compute normalized loss (squared RMS values with weights)
    loss = λ_X * res_X_rms^2 + λ_π * res_π_rms^2

    return loss, res_X_sum, res_π_sum, st_new
end

function loss_fn_normalized_wrapper(model, ps, st, data)
    par, state, shocks, ss, weights = data
    loss, res_X_sum, res_π_sum, st_new = loss_fn_normalized(
        ps, st, model, par, state, shocks, ss;
        λ_X=weights.λ_X, λ_π=weights.λ_π
    )
    return loss, st_new, (res_X=res_X_sum, res_π=res_π_sum)
end


function train_normalized!(network, ps, st, ranges, shock_config;
    num_epochs=1000, batch=100, mc=10, lr=0.001, internal=1, print_every=100,
    redraw_every=100, num_steps=1, eta_min=1e-10, λ_X=1.0, λ_π=1.0)

    opt = OptimiserChain(ClipGrad(1.0), Adam(lr))
    train_state = Lux.Training.TrainState(network, ps, st, opt)
    priors = prior_distribution(ranges)

    loss_dict = Dict(:iteration => [], :running_loss => [], :loss => [],
        :res_X => [], :res_π => [], :res_X_rms => [], :res_π_rms => [])

    weights = (λ_X=λ_X, λ_π=λ_π)

    par = draw_parameters(priors, batch)
    ss = steady_state(par)
    state = initialize_state(par, batch, ss)
    running_loss = 0.0

    prog = Progress(num_epochs; dt=1.0, desc="Training... ", showspeed=true)

    for epoch in 1:num_epochs
        shocks = draw_shocks(shock_config, mc, batch)
        data = (par, state, shocks, ss, weights)

        loss = nothing
        stats = nothing

        for o in 1:internal
            _, loss, stats, train_state = Lux.Training.single_train_step!(
                AutoZygote(), loss_fn_normalized_wrapper, data, train_state
            )
        end

        running_loss += loss / batch

        # Update learning rate
        current_lr = cosine_annealing_lr(lr, eta_min, epoch, num_epochs)
        new_opt = OptimiserChain(ClipGrad(1.0), Adam(current_lr))
        train_state = Lux.Training.TrainState(network, train_state.parameters,
            train_state.states, new_opt)

        if epoch % print_every == 0
            # Compute RMS residuals for display
            res_X_rms = sqrt(stats.res_X / batch)
            res_π_rms = sqrt(stats.res_π / batch)

            println("Epoch $epoch, Loss: $loss")
            println("  RMS residuals - res_X: $res_X_rms, res_π: $res_π_rms")
            println("  Raw residuals - res_X: $(stats.res_X), res_π: $(stats.res_π)")

            push!(loss_dict[:iteration], epoch)
            push!(loss_dict[:running_loss], running_loss / print_every)
            push!(loss_dict[:loss], loss)
            push!(loss_dict[:res_X], stats.res_X)
            push!(loss_dict[:res_π], stats.res_π)
            push!(loss_dict[:res_X_rms], res_X_rms)
            push!(loss_dict[:res_π_rms], res_π_rms)

            running_loss = 0.0
        end

        if epoch % redraw_every == 0
            par = draw_parameters(priors, batch)
            ss = steady_state(par)
            state = initialize_state(par, batch, ss)
        end

        for _ in 1:num_steps
            shocks = draw_shocks(shock_config, 1, batch)
            state = step(state, shocks, par, ss)
        end

        next!(prog)
    end

    return train_state, loss_dict
end

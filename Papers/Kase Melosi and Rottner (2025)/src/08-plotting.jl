# ═══════════════════════════════════════════════════════════════════════════
# PLOTTING AND VISUALIZATION FUNCTIONS
# Using multiple dispatch for model-specific plots
# ═══════════════════════════════════════════════════════════════════════════

# ───────────────────────────────────────────────────────────────────────────
# Training Diagnostics
# ───────────────────────────────────────────────────────────────────────────

"""
    plot_avg_loss(loss_history)

Plot average loss vs iterations during training.
Simple line plot showing convergence.
"""
function plot_avg_loss(loss_history)
    iterations = loss_history[:iteration]
    avg_loss = loss_history[:running_loss]

    p = plot(iterations, avg_loss,
             xlabel="Iteration",
             ylabel="Average Loss",
             title="Training Convergence",
             legend=false,
             linewidth=2)

    return p
end

"""
    plot_loss_components(loss_dict)

Plot training loss components:
- Average total loss
- NKPC residuals
- Euler equation residuals
- Bond market clearing
- Goods market clearing

Multi-panel plot showing contribution of each component.
"""
function plot_loss_components(loss_dict)
    iterations = loss_dict[:iteration]

    # Create subplots
    p1 = plot(iterations, loss_dict[:running_loss],
             ylabel="Avg Loss",
             title="Total Loss",
             legend=false,
             linewidth=2)

    p2 = plot(iterations, loss_dict[:res_X],
             ylabel="Euler Residual",
             legend=false,
             linewidth=2)

    p3 = plot(iterations, loss_dict[:res_π],
             xlabel="Iteration",
             ylabel="NKPC Residual",
             legend=false,
             linewidth=2)

    # Combine into single plot
    p = plot(p1, p2, p3, layout=(3,1), size=(800,600))

    return p
end

# ───────────────────────────────────────────────────────────────────────────
# Policy Function Visualization
# ───────────────────────────────────────────────────────────────────────────

"""
    plot_policy_comparison(network, ps, st, params::NKParameters, ranges::Ranges)

Compare analytical vs NN policy functions for NK model.
Uses policy_analytical() and policy() to generate comparison.

[FUTURE IMPLEMENTATION - requires policy_analytical]
"""
function plot_policy_comparison(network, ps, st, params::NKParameters, ranges::Ranges)
    error("plot_policy_comparison not yet implemented - requires policy_analytical")
end

"""
    plot_policy_comparison(network, ps, st, params::RANKParameters, ranges::Ranges)

RANK-specific policy comparison showing ZLB episodes.

[FUTURE IMPLEMENTATION]
"""
function plot_policy_comparison(network, ps, st, params::RANKParameters, ranges::Ranges)
    error("plot_policy_comparison for RANK not yet implemented")
end

"""
    plot_par_list(results_dict)

Visualize policy functions across parameter grid.
Input: results from policy_over_par_list()
Output: Multi-panel plot showing policy variation with each parameter

[FUTURE IMPLEMENTATION - requires policy_over_par]
"""
function plot_par_list(results_dict)
    error("plot_par_list not yet implemented - requires policy_over_par")
end

# ───────────────────────────────────────────────────────────────────────────
# Particle Filter Diagnostics
# ───────────────────────────────────────────────────────────────────────────

"""
    plot_loss_evolution(loss_matrix; ylim=nothing, ma=1, legend_label=nothing)

Plot training and validation loss evolution for NN particle filter.

Shows convergence of likelihood approximation via side-by-side plots.

# Mathematical Context
During training, the network minimizes MSE between predicted and true log-likelihoods:
    ℒ_train = (1/N_train) Σᵢ (ℓ̂(θᵢ) - ℓ_PF(θᵢ))²
    ℒ_val = (1/N_val) Σⱼ (ℓ̂(θⱼ) - ℓ_PF(θⱼ))²

Both losses should decrease and converge. Divergence indicates overfitting.

# Arguments
- `loss_matrix::Matrix{Float64}`: Training history (num_epochs × 3)
    - Column 1: Epoch number
    - Column 2: Training loss (MSE)
    - Column 3: Validation loss (MSE)
- `ylim::Union{Nothing,Tuple}`: Y-axis limits (default: auto)
- `ma::Int`: Moving average window for smoothing (default: 1 = no smoothing)
- `legend_label::Union{Nothing,String}`: Custom legend label (default: none)

# Returns
- Combined plot with training loss (left) and validation loss (right) in log scale

# Example
```julia
train_state, loss_history = pf_train!(network, ps, st, dataset; num_epochs=100)
plot_loss_evolution(loss_history; ma=5)
```
"""
function plot_loss_evolution(loss_matrix; ylim=nothing, ma=1, legend_label=nothing)
    # Extract data from loss_matrix
    epochs = loss_matrix[:, 1]
    train_loss = loss_matrix[:, 2]
    val_loss = loss_matrix[:, 3]

    # Apply moving average if ma > 1
    if ma > 1
        train_loss = [mean(train_loss[max(1, i-ma+1):i]) for i in 1:length(train_loss)]
        val_loss = [mean(val_loss[max(1, i-ma+1):i]) for i in 1:length(val_loss)]
    end

    # Create side-by-side plots
    p1 = plot(epochs, train_loss,
              title="Training Loss",
              xlabel="Epoch",
              ylabel="Mean Squared Error",
              color=:blue,
              label=legend_label,
              yscale=:log10,
            #   ylim=ylim,
              grid=true,
              minorgrid=true,
              gridlinewidth=1.5,
              minorgridalpha=0.2,
              lw=2)

    p2 = plot(epochs, val_loss,
              title="Validation Loss",
              xlabel="Epoch",
              ylabel="Mean Squared Error",
              color=:red,
              label=legend_label,
              yscale=:log10,
            #   ylim=ylim,
              grid=true,
              minorgrid=true,
              gridlinewidth=1.5,
              minorgridalpha=0.2,
              lw=2)

    # Combine plots side by side
    plot(p1, p2, layout=(1, 2), size=(800, 300))
end

"""
    plot_likelihood_single(pf, shock_config, par_name::Symbol, network, ps, st;
                          N=50, P_1=1000, P_noisy=100, N_noisy=250,
                          sim=100, plot_true_value=true)

Plot likelihood slice for single parameter with three comparison methods.

Validates NN surrogate accuracy by comparing:
1. **Finer PF** (blue circles): High-quality particle filter with P_1 particles
2. **Noisy PF** (red diamonds): Lower-quality PF with P_noisy particles (more scatter)
3. **NN Prediction** (green dashed): Neural network surrogate (fast)

# Mathematical Context
For a single parameter θ_j, we plot:
    ℓ(θ_j | y_1:T) = log p(y_1:T | θ_j, θ_{-j}^{true})

holding all other parameters θ_{-j} fixed at their true values.

The NN surrogate should approximate the finer PF curve, showing it can replace
expensive particle filter evaluations during posterior sampling.

# Arguments
- `pf::ParticleFilter`: Particle filter object
- `shock_config::Shocks`: Shock configuration for filter runs
- `par_name::Symbol`: Parameter to vary (e.g., :β, :σ, :σ_shock)
- `network`: Trained Lux neural network
- `ps::NamedTuple`: Network parameters (weights)
- `st::NamedTuple`: Network state
- `N::Int`: Number of grid points for finer PF and NN (default: 50)
- `P_1::Int`: Number of particles for finer PF (default: 1000)
- `P_noisy::Int`: Number of particles for noisy PF (default: 100)
- `N_noisy::Int`: Number of grid points for noisy PF (default: 250)
- `sim::Int`: Number of periods to filter (default: 100)
- `plot_true_value::Bool`: Add vertical line at true parameter value (default: true)

# Returns
- Plot object showing likelihood comparison

# Example
```julia
# Train network first
dataset = filter_dataset(pf, shock_config; N=5000)
train_state, _ = pf_train!(network, ps, st, dataset; num_epochs=100)

# Plot likelihood slice for β
plot_likelihood_single(pf, shock_config, :β, network,
                      train_state.parameters, train_state.states;
                      N=50, P_1=1000)
```
"""
function plot_likelihood_single(pf, shock_config, par_name::Symbol, network, ps, st;
                                N=50, #Number of grid points
                                P_1=1000, # Number of particles for standard (full) particle filter
                                P_noisy = 100, # Number of particles for noisy particle filter
                                N_noisy = 250, # Number of parameter draws for noisy particle filter
                                sim=100, # Number of periods to filter
                                plot_true_value=true)


    # Build parameter matrix with fixed parameters except par_name
    par_matrix, _ = build_param_matrix_fixed(pf, par_name, N)

    # Predict log-likelihoods using trained network
    ll_network = predict_log_likelihood(network, ps, st, par_matrix)

    # Generate likelihood grid using Full Particle Filter
    grid, ll_grid = filter_grid(pf, par_name, shock_config; n=N, P=P_1, sim=sim)


    # Generate likelihood grid using Noisy Particle Filter
    grid_noisy, ll_grid_noisy = filter_grid(pf, par_name, shock_config; n=N_noisy, P=P_noisy, sim=sim)

    if plot_true_value
        true_value = getfield(pf.model.parameters, par_name)
    end

    # println("grid: ", size(grid))
    # println("ll_network: ", size(ll_network))
    # println("grid_noisy: ", size(grid_noisy))


    # Create plot
    p = Plots.plot(grid, ll_grid,
             color=:blue,
             label="Finer PF ($P_1 particles)",
             lw=2,
             xlabel="Parameter: $par_name",
             ylabel="Log Likelihood",
             title="Log Likelihood conditioned on $par_name",
             legend=:bottomright,
             grid=true,
             marker=:circle,
             markersize=3)
    Plots.scatter!(p, grid_noisy, ll_grid_noisy,
          color=:red,
          label="Noisy PF ($P_noisy particles)",
          lw=2,
          marker=:diamond,
          markersize=3)
    Plots.plot!(p, grid, ll_network,
          color=:green,
          label="NN PF Prediction",
          lw=2,
          linestyle=:dash)

    # Add vertical line at true parameter value
    if plot_true_value
        vline!(p, [true_value],
               color=:black,
               linestyle=:dot,
               label="True value",
               lw=1.5)
    end

    return p
end

"""
    compute_likelihood_results(pf::ParticleFilter, par_names::Vector{Symbol};
                              n_points=50, n_particles=1000)

Compute likelihood values for plotting:
- NN likelihood (fast)
- Standard PF likelihood (slow, for validation)

Returns dict with results for each parameter.
Used by plot_likelihood_parameters.

[USER IMPLEMENTS]
"""
function compute_likelihood_results(pf, par_names; n_points=50, n_particles=1000)
    error("compute_likelihood_results not yet implemented - user function")
end

"""
    plot_likelihood_parameters(results_dict)

Multi-panel plot showing likelihood slices for multiple parameters.
Input: results from compute_likelihood_results()
Output: Grid of likelihood comparison plots

[FUTURE IMPLEMENTATION]
"""
function plot_likelihood_parameters(results_dict)
    error("plot_likelihood_parameters not yet implemented")
end

# ───────────────────────────────────────────────────────────────────────────
# Simulation Visualization
# ───────────────────────────────────────────────────────────────────────────

"""
    plot_simulation(states, observables; true_data=nothing)

Plot simulated state trajectories and observables.
Optionally overlay true data for comparison.

[FUTURE IMPLEMENTATION]
"""
function plot_simulation(states, observables; true_data=nothing)
    error("plot_simulation not yet implemented")
end

"""
    plot_beta(priors; n=5000)

Plot histogram of β samples from prior distribution.
"""
function plot_beta(priors; n=5000)
    β_samples = rand(priors.β, n)
    @info "β min/max" minimum(β_samples) maximum(β_samples)
    histogram(β_samples; bins=40, xlabel="β", ylabel="count",
              title="β ~ Uniform(β_low, β_high)", legend=false)
end

#!/usr/bin/env julia
# Train for 50,000 epochs and generate comparison plots

println("="^80)
println("TRAINING WITH train_simple_internal! FOR 50,000 EPOCHS")
println("="^80)

include("../load_hank.jl")
using Printf
using JLD2
using Plots

# Configuration
seed = 1234
Random.seed!(seed)

par = NKParameters(
    β=0.97, σ=2.0, η=1.125, ϕ=0.7,
    ϕ_pi=1.875, ϕ_y=0.25, ρ=0.875, σ_shock=0.06
)

ranges = Ranges(
    ζ=(0, 1.0), β=(0.95, 0.99), σ=(1.0, 3.0), η=(0.25, 2.0),
    ϕ=(0.5, 0.9), ϕ_pi=(1.25, 2.5), ϕ_y=(0.0, 0.5),
    ρ=(0.8, 0.95), σ_shock=(0.02, 0.1)
)

shock_config = Shocks(σ=1.0, antithetic=true)

network_config = (
    N_states=1, N_outputs=2, hidden=64, layers=5,
    activation=Lux.celu, scale_factor=1/100
)

# Create network
println("\n✓ Creating network...")
net, ps, st = make_network(par, ranges; network_config...)

# Train for 50,000 epochs
println("\n" * "="^80)
println("TRAINING...")
println("Configuration: 50,000 epochs, internal=5, batch=100, mc=10")
println("="^80)

time = @elapsed begin
    state_result, loss_dict = train_simple_internal!(net, ps, st, ranges, shock_config;
        num_epochs=50000,
        batch=100,
        mc=10,
        lr=0.001,
        internal=5,
        λ_X=1.0,
        λ_π=1.0,
        print_every=5000,
        eta_min=1e-10)
end

@printf("\n✓ Training completed in %.2f seconds (%.2f minutes)\n", time, time/60)
@printf("  Final loss: %.8f\n", loss_dict[:loss][end])

# Save the model
model_path = "model_simple_internal_50k.jld2"
println("\n✓ Saving model to $model_path...")
jldsave(model_path;
    parameters=state_result.parameters,
    states=state_result.states,
    loss_dict=loss_dict,
    network_config=network_config,
    ranges=ranges,
    seed=seed)
println("  Model saved successfully!")

# Test policy at a single point
test_state = State(ζ = 0.01)
X_an, π_an = policy_analytical(test_state, par)
X_nn, π_nn, _ = policy(net, test_state, par, state_result.parameters, state_result.states)

println("\n" * "="^80)
println("POLICY EVALUATION (ζ = 0.01)")
println("="^80)
@printf("Analytical: X = %.8f, π = %.8f\n", X_an, π_an)
@printf("Neural Net: X = %.8f, π = %.8f\n", X_nn[1], π_nn[1])
@printf("Errors: X = %.2f%%, π = %.2f%%\n",
        100*abs(X_nn[1] - X_an)/abs(X_an),
        100*abs(π_nn[1] - π_an)/abs(π_an))

# Generate comparison plots
println("\n" * "="^80)
println("GENERATING COMPARISON PLOTS")
println("="^80)

# Test state
test_ζ = 0.01

# Parameters to vary
param_names = [:β, :σ, :η, :ϕ, :ϕ_pi, :ϕ_y, :ρ, :σ_shock]
param_labels = ["β (Discount factor)", "σ (Risk aversion)", "η (Frisch elasticity)",
                "ϕ (Price stickiness)", "ϕ_π (Taylor rule π)", "ϕ_y (Taylor rule y)",
                "ρ (Persistence)", "σ_A (Shock volatility)"]

for (i, param_name) in enumerate(param_names)
    println("\nGenerating plots for $param_name...")

    # Get the range for this parameter
    param_range = getfield(ranges, param_name)
    param_values = range(param_range[1], param_range[2], length=50)

    # Arrays to store results
    X_analytical = zeros(length(param_values))
    π_analytical = zeros(length(param_values))
    X_neural = zeros(length(param_values))
    π_neural = zeros(length(param_values))

    # Compute policies for each parameter value
    for (j, param_val) in enumerate(param_values)
        # Create parameter struct with varied parameter
        test_par = NKParameters(
            β = param_name == :β ? param_val : par.β,
            σ = param_name == :σ ? param_val : par.σ,
            η = param_name == :η ? param_val : par.η,
            ϕ = param_name == :ϕ ? param_val : par.ϕ,
            ϕ_pi = param_name == :ϕ_pi ? param_val : par.ϕ_pi,
            ϕ_y = param_name == :ϕ_y ? param_val : par.ϕ_y,
            ρ = param_name == :ρ ? param_val : par.ρ,
            σ_shock = param_name == :σ_shock ? param_val : par.σ_shock
        )

        # Analytical solution
        X_an, π_an = policy_analytical(test_state, test_par)
        X_analytical[j] = X_an
        π_analytical[j] = π_an

        # Neural network solution
        X_nn, π_nn, _ = policy(net, test_state, test_par, state_result.parameters, state_result.states)
        X_neural[j] = X_nn[1]
        π_neural[j] = π_nn[1]
    end

    # Create output gap plot
    p_X = plot(param_values, X_neural,
        label="Neural Network",
        color=:blue,
        linewidth=2,
        xlabel=param_labels[i],
        ylabel="Output Gap X",
        title="Output Gap Response - train_simple_internal! 50k epochs",
        legend=:best,
        size=(800, 600))
    plot!(p_X, param_values, X_analytical,
        label="Analytical",
        color=:red,
        linewidth=2,
        linestyle=:dash)

    output_file_X = "model_simple_internal_50k_$(param_name)_output_gap.pdf"
    savefig(p_X, output_file_X)
    println("  ✓ Saved $output_file_X")

    # Create inflation plot
    p_π = plot(param_values, π_neural,
        label="Neural Network",
        color=:blue,
        linewidth=2,
        xlabel=param_labels[i],
        ylabel="Inflation π",
        title="Inflation Response - train_simple_internal! 50k epochs",
        legend=:best,
        size=(800, 600))
    plot!(p_π, param_values, π_analytical,
        label="Analytical",
        color=:red,
        linewidth=2,
        linestyle=:dash)

    output_file_π = "model_simple_internal_50k_$(param_name)_inflation.pdf"
    savefig(p_π, output_file_π)
    println("  ✓ Saved $output_file_π")
end

println("\n" * "="^80)
println("✅ ALL TASKS COMPLETED")
println("="^80)
println("Summary:")
println("  - Training: 50,000 epochs completed")
println("  - Final loss: $(loss_dict[:loss][end])")
println("  - Model saved: $model_path")
println("  - Plots generated: 16 PDF files (8 parameters × 2 variables)")
println("="^80)

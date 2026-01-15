#!/usr/bin/env julia
# Generate multi-panel comparison figures matching the paper style

println("="^80)
println("GENERATING COMPARISON FIGURES (Paper Style)")
println("="^80)

include("../load_hank.jl")
using Printf
using JLD2
using Plots
using Plots.PlotMeasures

# Load the saved model
model_path = "model_simple_internal_50k.jld2"
println("\n✓ Loading model from $model_path...")
data = load(model_path)
ps_loaded = data["parameters"]
st_loaded = data["states"]
network_config = data["network_config"]
ranges = data["ranges"]

# Recreate network
par = NKParameters(
    β=0.97, σ=2.0, η=1.125, ϕ=0.7,
    ϕ_pi=1.875, ϕ_y=0.25, ρ=0.875, σ_shock=0.06
)
net, _, _ = make_network(par, ranges; network_config...)

# Shock standard deviation (matching paper: -1.0 means -1 std dev of ergodic distribution)
shock_std = -1.0

# Parameters to vary
param_names = [:β, :σ, :η, :ϕ, :ϕ_pi, :ϕ_y, :ρ, :σ_shock]
param_labels = [
    "Discount factor β",
    "Relative risk aversion σ",
    "Inverse Frisch elasticity η",
    "Price duration ϕ",
    "MP inflation response ϕ_π",
    "MP output response ϕ_y",
    "Persistence shock ρ_A",
    "Standard deviation shock σ_A"
]

# Compute all parameter variations
println("\n✓ Computing policy functions across parameter space...")
n_points = 50
results = Dict()

for (i, param_name) in enumerate(param_names)
    println("  Processing $param_name...")

    param_range = getfield(ranges, param_name)
    param_values = range(param_range[1], param_range[2], length=n_points)

    X_analytical = zeros(n_points)
    π_analytical = zeros(n_points)
    X_neural = zeros(n_points)
    π_neural = zeros(n_points)

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

        # Compute steady state for this parameter combination
        test_ss = steady_state(test_par)

        # Compute ergodic standard deviation and create test state
        # Following Python code: sigma = par.sigma_a * par.sigma * (par.rho_a - 1) * ss.omega
        #                        ergodic = sigma / sqrt(1 - rho^2)
        sigma = test_par.σ_shock * test_par.σ * (test_par.ρ - 1) * test_ss.ω
        ergodic = sigma / sqrt(1 - test_par.ρ^2)
        test_state = State(ζ = shock_std * ergodic)

        # Analytical solution
        X_an, π_an = policy_analytical(test_state, test_par)
        X_analytical[j] = X_an
        π_analytical[j] = π_an

        # Neural network solution
        X_nn, π_nn, _ = policy(net, test_state, test_par, ps_loaded, st_loaded)
        X_neural[j] = X_nn[1]
        π_neural[j] = π_nn[1]
    end

    results[param_name] = (
        values = param_values,
        X_analytical = X_analytical,
        π_analytical = π_analytical,
        X_neural = X_neural,
        π_neural = π_neural
    )
end

println("\n✓ Creating multi-panel figures...")

# Figure 1: Output Gap (X̂_t)
println("  Generating output gap comparison figure...")
p_output = plot(layout=(4,2), size=(1200, 1600),
    left_margin=5mm, right_margin=5mm, top_margin=5mm, bottom_margin=5mm)

for (i, param_name) in enumerate(param_names)
    r = results[param_name]

    # Plot neural network (blue solid)
    plot!(p_output[i], r.values, r.X_neural,
        label="Neural network",
        color=:blue,
        linewidth=2,
        xlabel=param_labels[i],
        ylabel=i <= 2 ? "PF X̂ₜ conditioned on $(String(param_name))" : "",
        title="",
        legend=(i == 8 ? :bottomright : false),
        grid=true,
        gridstyle=:dot,
        gridalpha=0.3,
        framestyle=:box)

    # Plot analytical (red dashed)
    plot!(p_output[i], r.values, r.X_analytical,
        label="Analytical solution",
        color=:red,
        linewidth=2,
        linestyle=:dash)
end

# Add overall title
plot!(p_output, plot_title="Accuracy of the NN solution method - Output Gap",
    plot_titlefontsize=14)

output_file_X = "nk_policy_output_gap_comparison_50k.pdf"
savefig(p_output, output_file_X)
println("  ✓ Saved $output_file_X")

# Figure 2: Inflation (Π̂_t)
println("  Generating inflation comparison figure...")
p_inflation = plot(layout=(4,2), size=(1200, 1600),
    left_margin=5mm, right_margin=5mm, top_margin=5mm, bottom_margin=5mm)

for (i, param_name) in enumerate(param_names)
    r = results[param_name]

    # Plot neural network (blue solid)
    plot!(p_inflation[i], r.values, r.π_neural,
        label="Neural network",
        color=:blue,
        linewidth=2,
        xlabel=param_labels[i],
        ylabel=i <= 2 ? "PF Π̂ₜ conditioned on $(String(param_name))" : "",
        title="",
        legend=(i == 8 ? :bottomright : false),
        grid=true,
        gridstyle=:dot,
        gridalpha=0.3,
        framestyle=:box)

    # Plot analytical (red dashed)
    plot!(p_inflation[i], r.values, r.π_analytical,
        label="Analytical solution",
        color=:red,
        linewidth=2,
        linestyle=:dash)
end

# Add overall title
plot!(p_inflation, plot_title="Accuracy of the NN solution method - Inflation",
    plot_titlefontsize=14)

output_file_π = "nk_policy_inflation_comparison_50k.pdf"
savefig(p_inflation, output_file_π)
println("  ✓ Saved $output_file_π")

println("\n" * "="^80)
println("✅ COMPARISON FIGURES GENERATED")
println("="^80)
println("Files created:")
println("  - $output_file_X")
println("  - $output_file_π")
println("="^80)

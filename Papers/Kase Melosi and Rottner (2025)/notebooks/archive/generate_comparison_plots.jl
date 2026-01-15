#!/usr/bin/env julia
# Generate comparison plots for all 4 trained models
# Creates 8 plots total: 4 models × 2 policies (inflation + output gap)

println("="^80)
println("GENERATING COMPARISON PLOTS FOR ALL TRAINED MODELS")
println("="^80)

# Load HankNN
println("\nLoading HankNN...")
include("../load_hank.jl")

using JLD2
using Plots
using Printf

# Define plot_par_list function (from notebook)
function plot_par_list(par::NKParameters, ranges::Ranges, network, ps, st;
                       shock_std::Float64=-1.0,
                       policy::Symbol=:π,
                       par_list::Union{Nothing,Vector{Symbol}}=nothing,
                       n_points::Int=100)

    # Define default parameter list if not provided
    if par_list === nothing
        par_list = [:β, :σ, :η, :ϕ, :ϕ_pi, :ϕ_y, :ρ, :σ_shock]
    end

    # Simple string labels for parameters (using Unicode symbols)
    par_labels = Dict(
        :β => "β",
        :σ => "σ",
        :η => "η",
        :ϕ => "ϕ",
        :ϕ_pi => "ϕ_π",
        :ϕ_y => "ϕ_y",
        :ρ => "ρ",
        :σ_shock => "σ_A"
    )

    # Human-readable labels
    ylabel_dict = Dict(
        :β => "Discount factor",
        :σ => "Relative risk aversion",
        :η => "Inverse Frisch elasticity",
        :ϕ => "Price duration",
        :ϕ_pi => "MP inflation response",
        :ϕ_y => "MP output response",
        :ρ => "Persistence shock",
        :σ_shock => "Standard deviation shock"
    )

    # Title based on policy choice
    if policy == :π
        title_prefix = "π"
    elseif policy == :X
        title_prefix = "X"
    else
        error("policy must be :π or :X")
    end

    # Get analytical and numerical solutions for all parameters
    ana_results = policy_over_par_list(par, ranges, par_list, network, ps, st;
                                       n_points=n_points, shock_std=shock_std, analytical=true)
    num_results = policy_over_par_list(par, ranges, par_list, network, ps, st;
                                       n_points=n_points, shock_std=shock_std, analytical=false)

    # Create 4×2 subplot grid
    n_cols = 2
    n_rows = 4
    plots_array = []

    for (i, param) in enumerate(par_list)
        # Get data for this parameter
        ana_data = ana_results[param]
        num_data = num_results[param]

        # Extract the appropriate policy variable
        if policy == :π
            ana_y = ana_data[:π]
            num_y = num_data[:π]
        else
            ana_y = ana_data[:X]
            num_y = num_data[:X]
        end

        # Create subplot
        p = plot(num_data[:param_values], num_y,
                label="Neural network",
                linewidth=1.5,
                color=:blue,
                xlabel="$(ylabel_dict[param]) ($(par_labels[param]))",
                title="PF $(title_prefix) | $(par_labels[param])",
                titlefontsize=8,
                labelfontsize=7,
                tickfontsize=6,
                legendfontsize=7)

        plot!(p, ana_data[:param_values], ana_y,
             label="Analytical solution",
             linewidth=1.5,
             linestyle=:dash,
             color=:red)

        # Add legend only to last subplot
        if i == length(par_list)
            plot!(p, legend=:bottomright)
        else
            plot!(p, legend=false)
        end

        push!(plots_array, p)
    end

    # Combine all subplots
    fig = plot(plots_array..., layout=(n_rows, n_cols),
              size=(800, 1000),
              margin=5Plots.mm)

    return fig
end

# Helper function to load model
function load_model(filename)
    data = JLD2.load(filename)

    # Extract activation
    act_name = data["activation_name"]
    if contains(act_name, "celu")
        activation_fn = Lux.celu
    elseif contains(act_name, "relu")
        activation_fn = Lux.relu
    else
        activation_fn = Lux.celu
    end

    # Reconstruct network
    network, _, _ = make_network(data["par"], data["ranges"];
                                  N_states=data["N_states"],
                                  N_outputs=data["N_outputs"],
                                  hidden=data["hidden"],
                                  layers=data["layers"],
                                  activation=activation_fn,
                                  scale_factor=data["scale_factor"])

    return network, data["ps"], data["st"], data["par"], data["ranges"]
end

# Define all models
models = [
    (file="model_simple_50k.jld2", name="train_simple!", label="Simple"),
    (file="model_fast_50k.jld2", name="train_simple_fast!", label="Fast"),
    (file="model_full_50k.jld2", name="train!", label="Full-Featured"),
    (file="model_optimized_50k.jld2", name="train_optimized!", label="Optimized")
]

println("\n" * "="^80)
println("LOADING ALL MODELS")
println("="^80)

# Load all models
loaded_models = []
for model_info in models
    println("\nLoading $(model_info.file)...")
    network, ps, st, par, ranges = load_model(model_info.file)
    push!(loaded_models, (
        network=network, ps=ps, st=st, par=par, ranges=ranges,
        name=model_info.name, label=model_info.label,
        file=model_info.file
    ))
    println("  ✓ Loaded $(model_info.name)")
end

println("\n✓ All models loaded successfully")

# Generate plots for each model
println("\n" * "="^80)
println("GENERATING COMPARISON PLOTS")
println("="^80)

for (i, model) in enumerate(loaded_models)
    println("\n[$i/4] Processing $(model.name)...")

    # Generate inflation comparison plot
    println("  Generating inflation comparison plot...")
    fig_inflation = plot_par_list(
        model.par, model.ranges, model.network, model.ps, model.st;
        shock_std=-1.0, policy=:π, n_points=100
    )

    # Save inflation plot
    inflation_filename = replace(model.file, ".jld2" => "_inflation_comparison.pdf")
    savefig(fig_inflation, inflation_filename)
    println("    ✓ Saved: $inflation_filename")

    # Generate output gap comparison plot
    println("  Generating output gap comparison plot...")
    fig_output = plot_par_list(
        model.par, model.ranges, model.network, model.ps, model.st;
        shock_std=-1.0, policy=:X, n_points=100
    )

    # Save output gap plot
    output_filename = replace(model.file, ".jld2" => "_output_gap_comparison.pdf")
    savefig(fig_output, output_filename)
    println("    ✓ Saved: $output_filename")
end

println("\n" * "="^80)
println("ALL PLOTS GENERATED SUCCESSFULLY!")
println("="^80)

println("\n📊 Generated 8 plots total:")
println("\nInflation Policy Comparisons:")
for model in models
    filename = replace(model.file, ".jld2" => "_inflation_comparison.pdf")
    println("  ✓ $filename")
end

println("\nOutput Gap Policy Comparisons:")
for model in models
    filename = replace(model.file, ".jld2" => "_output_gap_comparison.pdf")
    println("  ✓ $filename")
end

println("\n" * "="^80)
println("SUMMARY")
println("="^80)
println("All comparison plots show:")
println("  - Blue solid line: Neural network predictions")
println("  - Red dashed line: Analytical solution")
println("  - 8 parameters: β, σ, η, ϕ, ϕ_π, ϕ_y, ρ, σ_A")
println("  - Each subplot varies one parameter across its range")
println("\nPlots are ready for review!")
println("="^80)

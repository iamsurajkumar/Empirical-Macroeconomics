#!/usr/bin/env julia
# Generate comparison plots for all 4 trained models
# Fixed version: creates fresh scalar parameters for plotting

println("="^80)
println("GENERATING COMPARISON PLOTS FOR ALL TRAINED MODELS")
println("="^80)

# Load HankNN
println("\nLoading HankNN...")
include("../load_hank.jl")

using JLD2
using Plots

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

    # Create FRESH scalar parameters (not the batched ones from training)
    par = NKParameters(
        β=0.97, σ=2.0, η=1.125, ϕ=0.7,
        ϕ_pi=1.875, ϕ_y=0.25, ρ=0.875, σ_shock=0.06
    )

    ranges = Ranges(
        ζ=(0, 1.0), β=(0.95, 0.99), σ=(1.0, 3.0), η=(0.25, 2.0),
        ϕ=(0.5, 0.9), ϕ_pi=(1.25, 2.5), ϕ_y=(0.0, 0.5),
        ρ=(0.8, 0.95), σ_shock=(0.02, 0.1)
    )

    return network, data["ps"], data["st"], par, ranges
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

    # Parameter list
    par_list = [:β, :σ, :η, :ϕ, :ϕ_pi, :ϕ_y, :ρ, :σ_shock]

    # Get results
    println("  Computing analytical solutions...")
    ana_results = policy_over_par_list(model.par, model.ranges, par_list,
                                       model.network, model.ps, model.st;
                                       n_points=100, shock_std=-1.0, analytical=true)

    println("  Computing numerical solutions...")
    num_results = policy_over_par_list(model.par, model.ranges, par_list,
                                       model.network, model.ps, model.st;
                                       n_points=100, shock_std=-1.0, analytical=false)

    # Parameter labels
    par_labels = Dict(
        :β => "β", :σ => "σ", :η => "η", :ϕ => "ϕ",
        :ϕ_pi => "ϕ_π", :ϕ_y => "ϕ_y", :ρ => "ρ", :σ_shock => "σ_A"
    )

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

    # Create inflation plot
    println("  Creating inflation plot...")
    plots_π = []
    for (j, param) in enumerate(par_list)
        ana_data = ana_results[param]
        num_data = num_results[param]

        p = plot(num_data[:param_values], num_data[:π],
                label="Neural network",
                linewidth=1.5, color=:blue,
                xlabel="$(ylabel_dict[param]) ($(par_labels[param]))",
                title="PF π | $(par_labels[param])",
                titlefontsize=8, labelfontsize=7,
                tickfontsize=6, legendfontsize=7)

        plot!(p, ana_data[:param_values], ana_data[:π],
             label="Analytical solution",
             linewidth=1.5, linestyle=:dash, color=:red)

        if j == length(par_list)
            plot!(p, legend=:bottomright)
        else
            plot!(p, legend=false)
        end

        push!(plots_π, p)
    end

    fig_π = plot(plots_π..., layout=(4, 2), size=(800, 1000), margin=5Plots.mm)
    inflation_file = replace(model.file, ".jld2" => "_inflation_comparison.pdf")
    savefig(fig_π, inflation_file)
    println("    ✓ Saved: $inflation_file")

    # Create output gap plot
    println("  Creating output gap plot...")
    plots_X = []
    for (j, param) in enumerate(par_list)
        ana_data = ana_results[param]
        num_data = num_results[param]

        p = plot(num_data[:param_values], num_data[:X],
                label="Neural network",
                linewidth=1.5, color=:blue,
                xlabel="$(ylabel_dict[param]) ($(par_labels[param]))",
                title="PF X | $(par_labels[param])",
                titlefontsize=8, labelfontsize=7,
                tickfontsize=6, legendfontsize=7)

        plot!(p, ana_data[:param_values], ana_data[:X],
             label="Analytical solution",
             linewidth=1.5, linestyle=:dash, color=:red)

        if j == length(par_list)
            plot!(p, legend=:bottomright)
        else
            plot!(p, legend=false)
        end

        push!(plots_X, p)
    end

    fig_X = plot(plots_X..., layout=(4, 2), size=(800, 1000), margin=5Plots.mm)
    output_file = replace(model.file, ".jld2" => "_output_gap_comparison.pdf")
    savefig(fig_X, output_file)
    println("    ✓ Saved: $output_file")
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

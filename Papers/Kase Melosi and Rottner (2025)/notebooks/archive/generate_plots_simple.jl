#!/usr/bin/env julia
# Generate comparison plots one model at a time (more robust)

println("="^80)
println("GENERATING COMPARISON PLOTS")
println("="^80)

include("../load_hank.jl")
using JLD2
using Plots

# Load one model, generate plots, then move to next
models = [
    "model_simple_50k.jld2",
    "model_fast_50k.jld2",
    "model_full_50k.jld2",
    "model_optimized_50k.jld2"
]

for (i, model_file) in enumerate(models)
    println("\n[$i/4] Processing $model_file...")

    try
        # Load model
        data = JLD2.load(model_file)

        # Get activation
        act_name = data["activation_name"]
        activation_fn = contains(act_name, "celu") ? Lux.celu : Lux.relu

        # Reconstruct network
        network, _, _ = make_network(data["par"], data["ranges"];
                                      N_states=data["N_states"],
                                      N_outputs=data["N_outputs"],
                                      hidden=data["hidden"],
                                      layers=data["layers"],
                                      activation=activation_fn,
                                      scale_factor=data["scale_factor"])

        ps = data["ps"]
        st = data["st"]
        par = data["par"]
        ranges = data["ranges"]

        # Generate plots using the notebook's approach
        println("  Generating inflation plot...")

        # Get all parameter results
        par_list = [:β, :σ, :η, :ϕ, :ϕ_pi, :ϕ_y, :ρ, :σ_shock]

        # Get results
        ana_results = policy_over_par_list(par, ranges, par_list, network, ps, st;
                                           n_points=100, shock_std=-1.0, analytical=true)
        num_results = policy_over_par_list(par, ranges, par_list, network, ps, st;
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
        inflation_file = replace(model_file, ".jld2" => "_inflation_comparison.pdf")
        savefig(fig_π, inflation_file)
        println("    ✓ Saved: $inflation_file")

        # Create output gap plot
        println("  Generating output gap plot...")
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
        output_file = replace(model_file, ".jld2" => "_output_gap_comparison.pdf")
        savefig(fig_X, output_file)
        println("    ✓ Saved: $output_file")

    catch e
        println("    ✗ Error: $e")
        println("    Skipping this model...")
        continue
    end
end

println("\n" * "="^80)
println("PLOT GENERATION COMPLETE")
println("="^80)
println("\nGenerated plots:")
for model_file in models
    inflation_file = replace(model_file, ".jld2" => "_inflation_comparison.pdf")
    output_file = replace(model_file, ".jld2" => "_output_gap_comparison.pdf")
    if isfile(inflation_file)
        println("  ✓ $inflation_file")
    end
    if isfile(output_file)
        println("  ✓ $output_file")
    end
end
println("="^80)

#!/usr/bin/env julia
# Generate plots in completely isolated Julia sessions

models = [
    "model_fast_50k.jld2",
    "model_full_50k.jld2",
    "model_optimized_50k.jld2"
]

script_template = """
include("../load_hank.jl")
using JLD2, Plots

model_file = "MODEL_FILE"
println("Processing \$model_file...")

data = JLD2.load(model_file)
act_name = data["activation_name"]
activation_fn = contains(act_name, "celu") ? Lux.celu : Lux.relu

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

par_list = [:β, :σ, :η, :ϕ, :ϕ_pi, :ϕ_y, :ρ, :σ_shock]

println("  Computing analytical solutions...")
ana_results = policy_over_par_list(par, ranges, par_list, network, ps, st;
                                   n_points=100, shock_std=-1.0, analytical=true)
println("  Computing numerical solutions...")
num_results = policy_over_par_list(par, ranges, par_list, network, ps, st;
                                   n_points=100, shock_std=-1.0, analytical=false)

par_labels = Dict(:β => "β", :σ => "σ", :η => "η", :ϕ => "ϕ",
                  :ϕ_pi => "ϕ_π", :ϕ_y => "ϕ_y", :ρ => "ρ", :σ_shock => "σ_A")

ylabel_dict = Dict(
    :β => "Discount factor", :σ => "Relative risk aversion",
    :η => "Inverse Frisch elasticity", :ϕ => "Price duration",
    :ϕ_pi => "MP inflation response", :ϕ_y => "MP output response",
    :ρ => "Persistence shock", :σ_shock => "Standard deviation shock"
)

# Inflation plot
println("  Creating inflation plot...")
plots_π = []
for (j, param) in enumerate(par_list)
    p = plot(num_results[param][:param_values], num_results[param][:π],
            label="Neural network", linewidth=1.5, color=:blue,
            xlabel="\$(ylabel_dict[param]) (\$(par_labels[param]))",
            title="PF π | \$(par_labels[param])",
            titlefontsize=8, labelfontsize=7, tickfontsize=6, legendfontsize=7)
    plot!(p, ana_results[param][:param_values], ana_results[param][:π],
         label="Analytical solution", linewidth=1.5, linestyle=:dash, color=:red)
    j == length(par_list) ? plot!(p, legend=:bottomright) : plot!(p, legend=false)
    push!(plots_π, p)
end

fig_π = plot(plots_π..., layout=(4, 2), size=(800, 1000), margin=5Plots.mm)
savefig(fig_π, replace(model_file, ".jld2" => "_inflation_comparison.pdf"))
println("  ✓ Saved inflation plot")

# Output gap plot
println("  Creating output gap plot...")
plots_X = []
for (j, param) in enumerate(par_list)
    p = plot(num_results[param][:param_values], num_results[param][:X],
            label="Neural network", linewidth=1.5, color=:blue,
            xlabel="\$(ylabel_dict[param]) (\$(par_labels[param]))",
            title="PF X | \$(par_labels[param])",
            titlefontsize=8, labelfontsize=7, tickfontsize=6, legendfontsize=7)
    plot!(p, ana_results[param][:param_values], ana_results[param][:X],
         label="Analytical solution", linewidth=1.5, linestyle=:dash, color=:red)
    j == length(par_list) ? plot!(p, legend=:bottomright) : plot!(p, legend=false)
    push!(plots_X, p)
end

fig_X = plot(plots_X..., layout=(4, 2), size=(800, 1000), margin=5Plots.mm)
savefig(fig_X, replace(model_file, ".jld2" => "_output_gap_comparison.pdf"))
println("  ✓ Saved output gap plot")
"""

cd("/Users/macbox/Library/Mobile Documents/3L68KQB4HG~com~readdle~CommonDocuments/Documents/Boston College PhD/3_Third Semester/01-Empirical Macroeconomics/Hank-NN-Claude/notebooks")

println("="^80)
println("GENERATING PLOTS FOR REMAINING MODELS")
println("="^80)

for (i, model) in enumerate(models)
    println("\n[$i/3] Processing $model...")

    # Create temporary script
    script = replace(script_template, "MODEL_FILE" => model)
    script_file = "temp_plot_$(i).jl"
    write(script_file, script)

    # Run in fresh Julia session
    run(`julia $script_file`)

    # Clean up
    rm(script_file)
end

println("\n" * "="^80)
println("ALL PLOTS GENERATED!")
println("="^80)

# List all generated plots
println("\nGenerated comparison plots:")
for model in ["model_simple_50k.jld2", models...]
    inf_file = replace(model, ".jld2" => "_inflation_comparison.pdf")
    out_file = replace(model, ".jld2" => "_output_gap_comparison.pdf")
    if isfile(inf_file)
        println("  ✓ $inf_file")
    end
    if isfile(out_file)
        println("  ✓ $out_file")
    end
end
println("="^80)

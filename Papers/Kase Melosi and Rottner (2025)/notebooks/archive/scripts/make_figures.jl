# ═══════════════════════════════════════════════════════════════════════════
# Generate Publication Figures
# Creates all plots for paper/presentation
# ═══════════════════════════════════════════════════════════════════════════

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using HankNN
using JLD2
using Plots
using Random

println("="^80)
println("Figure Generation Script")
println("="^80)

# ═══════════════════════════════════════════════════════════════════════════
# Load Trained Models
# ═══════════════════════════════════════════════════════════════════════════

println("\nLoading trained NK model...")
if !isfile("../save/nk_model.jld2")
    error("No trained model found! Run scripts/train_nk_model.jl first.")
end

data = load("../save/nk_model.jld2")
network = data["network"]
ps = data["parameters"]
st = data["state"]
loss_history = data["loss_history"]
println("✓ NK model loaded")

# ═══════════════════════════════════════════════════════════════════════════
# Figure 1: Training Convergence
# ═══════════════════════════════════════════════════════════════════════════

println("\nGenerating Figure 1: Training convergence...")
p1 = plot_avg_loss(loss_history)
savefig(p1, "../figures/fig1_training_loss.pdf")
savefig(p1, "../figures/fig1_training_loss.png")
println("✓ Saved figures/fig1_training_loss.{pdf,png}")

# ═══════════════════════════════════════════════════════════════════════════
# Figure 2: Loss Components
# ═══════════════════════════════════════════════════════════════════════════

println("\nGenerating Figure 2: Loss components...")
p2 = plot_loss_components(loss_history)
savefig(p2, "../figures/fig2_loss_components.pdf")
savefig(p2, "../figures/fig2_loss_components.png")
println("✓ Saved figures/fig2_loss_components.{pdf,png}")

# ═══════════════════════════════════════════════════════════════════════════
# Figure 3: Prior Distribution
# ═══════════════════════════════════════════════════════════════════════════

println("\nGenerating Figure 3: Prior distribution...")
ranges = Ranges(
    ζ = (-0.1, 0.1),
    β = (0.95, 0.995),
    σ = (1.0, 3.0),
    η = (0.5, 2.0),
    ϕ = (30.0, 70.0),
    ϕ_pi = (1.2, 2.0),
    ϕ_y = (0.1, 1.0),
    ρ = (0.5, 0.95),
    σ_shock = (0.005, 0.02)
)
priors = prior_distribution(ranges)
p3 = plot_beta(priors; n=5000)
savefig(p3, "../figures/fig3_prior_beta.pdf")
savefig(p3, "../figures/fig3_prior_beta.png")
println("✓ Saved figures/fig3_prior_beta.{pdf,png}")

# ═══════════════════════════════════════════════════════════════════════════
# Figure 4: Simulated Trajectories
# ═══════════════════════════════════════════════════════════════════════════

println("\nGenerating Figure 4: Simulated trajectories...")

# Baseline parameters
par = NKParameters(
    β = 0.99, σ = 2.0, η = 1.0, ϕ = 50.0,
    ϕ_pi = 1.5, ϕ_y = 0.5, ρ = 0.9, σ_shock = 0.01
)
shock_config = Shocks(σ=1.0, antithetic=true)

# Simulate
Random.seed!(1234)
sim_results = simulate(network, ps, st, 1, ranges, shock_config;
                      par=par, burn=50, num_steps=200, seed=1234)

# Plot trajectories
p4 = plot(
    plot(sim_results[:R][:, 1], ylabel="Shock ζ", title="Natural Rate Shock",
         legend=false, linewidth=2),
    plot(sim_results[:X][:, 1], ylabel="Output Gap", title="Output Gap",
         legend=false, linewidth=2),
    plot(sim_results[:π][:, 1], ylabel="Inflation", title="Inflation",
         xlabel="Time", legend=false, linewidth=2),
    layout=(3,1), size=(800, 600)
)
savefig(p4, "../figures/fig4_simulation.pdf")
savefig(p4, "../figures/fig4_simulation.png")
println("✓ Saved figures/fig4_simulation.{pdf,png}")

# ═══════════════════════════════════════════════════════════════════════════
# Future Figures (Require User Implementation)
# ═══════════════════════════════════════════════════════════════════════════

println("\n" * "="^80)
println("Additional figures require user implementation:")
println("  - Figure 5: Policy comparison (analytical vs NN)")
println("    → Requires policy_analytical() implementation")
println("  - Figure 6: Parameter sensitivity")
println("    → Requires policy_over_par() implementation")
println("  - Figure 7: Likelihood slices")
println("    → Requires particle filter implementation")
println("="^80)

println("\n✓ Figure generation complete!")
println("All figures saved to figures/ directory")

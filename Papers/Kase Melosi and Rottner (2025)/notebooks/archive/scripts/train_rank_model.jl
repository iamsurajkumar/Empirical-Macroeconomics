# ═══════════════════════════════════════════════════════════════════════════
# RANK Model Training Script (with ZLB)
# For batch/automated training runs
# ═══════════════════════════════════════════════════════════════════════════

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using HankNN
using JLD2
using Random

# ═══════════════════════════════════════════════════════════════════════════
# Configuration Flags
# ═══════════════════════════════════════════════════════════════════════════

load_model = false       # Load existing trained model?
seed = 1234             # Random seed for reproducibility

# ═══════════════════════════════════════════════════════════════════════════
# Model Configuration
# ═══════════════════════════════════════════════════════════════════════════

# RANK parameters (includes ZLB)
par = RANKParameters(
    β = 0.99,
    σ = 2.0,
    η = 1.0,
    ϕ = 50.0,
    ϕ_pi = 1.5,
    ϕ_y = 0.5,
    ρ = 0.9,
    σ_shock = 0.01,
    r_min = 1.0,      # ZLB constraint
    ϕ_r = 0.8         # Interest rate smoothing
)

# Training ranges (same as NK for now)
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

# Shock configuration
shock_config = Shocks(σ=1.0, antithetic=true)

# Training configuration (Kase et al. Section C.4)
training_config = TrainingConfig(
    n_iterations = 100_000,
    batch_size = 100,
    zlb_start_iter = 10_000,
    zlb_end_iter = 15_000,
    param_redraw_freq = 100,      # First 40k: every 100, then every 10
    initial_sim_periods = 100,
    regular_sim_periods = 20
)

# ═══════════════════════════════════════════════════════════════════════════
# FUTURE IMPLEMENTATION
# ═══════════════════════════════════════════════════════════════════════════

println("="^80)
println("RANK Model Training Script")
println("="^80)
println()
println("⚠️  RANK model training not yet implemented!")
println()
println("This script is a template for RANK model training.")
println("You need to:")
println("  1. Complete RANK model functions in src/04-economics-rank.jl")
println("  2. Implement policy(), step(), residuals() for RANK")
println("  3. Test ZLB constraint activation")
println("  4. Run this script to train RANK model")
println()
println("Training configuration ready:")
println("  Total iterations: ", training_config.n_iterations)
println("  Batch size: ", training_config.batch_size)
println("  ZLB introduction: iterations ", training_config.zlb_start_iter,
        " - ", training_config.zlb_end_iter)
println("  Parameter redraw: every ", training_config.param_redraw_freq, " iterations")
println()
println("See kase-my-plan-updated.md for implementation details.")
println("="^80)

# Uncomment below when RANK model is implemented:
#=
if load_model
    println("Loading existing RANK model...")
    data = load("../save/rank_model.jld2")
    network = data["network"]
    ps = data["parameters"]
    st = data["state"]
    loss_history = data["loss_history"]
    println("✓ Model loaded")
else
    println("Training RANK model with ZLB...")

    # Create network
    network, ps, st = make_kase_network(6, 2)  # 6 inputs (1 state + 5 params), 2 outputs

    # Train with ZLB scheduling
    # [IMPLEMENT TRAINING LOOP WITH ZLB]

    println("✓ Training complete")
end
=#

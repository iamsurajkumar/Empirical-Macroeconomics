# ═══════════════════════════════════════════════════════════════════════════
# Example Usage: RANK Model Training and Simulation
# ═══════════════════════════════════════════════════════════════════════════

using Random
include("rank-nn-section32.jl")

# ═══════════════════════════════════════════════════════════════════════════
# SETUP
# ═══════════════════════════════════════════════════════════════════════════

# Set random seed for reproducibility
Random.seed!(42)

# Define parameter ranges (from Table 5)
ranges = Ranges(
    ζ = (-0.05, 0.05),              # Preference shock range
    θ_Π = (1.5, 2.5),               # Taylor rule inflation response
    θ_Y = (0.05, 0.5),              # Taylor rule output response
    φ = (700.0, 1300.0),            # Rotemberg pricing (high = sticky)
    ρ_ζ = (0.5, 0.9),               # Shock persistence
    σ_ζ = (0.01, 0.025)             # Shock volatility
)

# Shock configuration (use mean of prior for initial setup)
shock_config = Shocks(σ = 0.02, antithetic = true)

# ═══════════════════════════════════════════════════════════════════════════
# CREATE NEURAL NETWORK
# ═══════════════════════════════════════════════════════════════════════════

println("Creating neural network...")
network, ps, st = make_network(ranges;
    N_states = 1,          # Just ζ
    N_outputs = 2,         # N and Π
    hidden = 128,          # Hidden layer width
    layers = 5,            # Number of hidden layers
    activation = swish,    # SiLU/Swish activation
    scale_factor = 1.0     # No output scaling
)

println("Network created with $(sum(length, ps)) parameters")

# ═══════════════════════════════════════════════════════════════════════════
# QUICK TEST: Single Forward Pass
# ═══════════════════════════════════════════════════════════════════════════

println("\n" * "="^80)
println("Testing forward pass...")

# Create test parameters
priors = prior_distribution(ranges)
test_par = draw_parameters(priors, 5)  # 5 parameter sets
test_ss = steady_state(test_par)

println("Steady state values:")
println("  R_bar: ", test_ss.R_bar[1:3])
println("  MC_bar: ", test_ss.MC_bar[1:3])
println("  N_bar: ", test_ss.N_bar[1:3])

# Create test state
test_state = State(ζ = randn(5))

# Evaluate policy
N_t, Π_t, st_new = policy(network, test_state, test_par, ps, st)
println("\nPolicy outputs:")
println("  N_t: ", N_t)
println("  Π_t: ", Π_t)

# Compute all variables
vars = compute_all_variables(N_t, Π_t, test_par, test_ss)
println("\nDerived variables:")
println("  Y_t: ", vars[:Y])
println("  R_t: ", vars[:R])
println("  MC_t: ", vars[:MC])

# ═══════════════════════════════════════════════════════════════════════════
# OPTION 1: SIMPLE TRAINING (Quick Test)
# ═══════════════════════════════════════════════════════════════════════════

println("\n" * "="^80)
println("OPTION 1: Simple training (1000 epochs)")
println("="^80)

# Quick training run
trained_state_simple = train_simple!(network, ps, st, ranges, shock_config;
    num_epochs = 1000,
    batch = 100,
    mc = 100,
    lr = 0.0001
)

println("\nSimple training complete!")

# ═══════════════════════════════════════════════════════════════════════════
# OPTION 2: ADVANCED TRAINING (Full Paper Specification)
# ═══════════════════════════════════════════════════════════════════════════

println("\n" * "="^80)
println("OPTION 2: Advanced training (30000 epochs)")
println("This will take a while...")
println("="^80)

# Uncomment to run full training:
# trained_state, loss_dict = train!(network, ps, st, ranges, shock_config;
#     num_epochs = 30000,
#     batch = 100,
#     mc = 100,
#     lr = 0.0001,
#     internal = 15,
#     print_after = 100,
#     par_draw_after = 40,
#     num_steps = 20,
#     eta_min = 1e-6,
#     zlb_start = 5000,
#     zlb_end = 10000
# )

# Plot training convergence
# plot_training_loss(loss_dict)
# savefig("rank_training_loss.png")

# ═══════════════════════════════════════════════════════════════════════════
# SIMULATION
# ═══════════════════════════════════════════════════════════════════════════

println("\n" * "="^80)
println("Simulating from trained network...")
println("="^80)

# Use the simple trained network (or swap for trained_state if using advanced)
sim_results = simulate(
    network, 
    trained_state_simple.parameters,  # Use trained parameters
    trained_state_simple.states,      # Use trained states
    10,                                # 10 trajectories
    ranges, 
    shock_config;
    burn = 100,
    num_steps = 500,
    seed = 123
)

println("\nSimulation complete!")
println("Shape of output: ", size(sim_results[:N]))
println("\nSummary statistics:")
println("  N: mean = $(mean(sim_results[:N])), std = $(std(sim_results[:N]))")
println("  Π: mean = $(mean(sim_results[:Π])), std = $(std(sim_results[:Π]))")
println("  Y: mean = $(mean(sim_results[:Y])), std = $(std(sim_results[:Y]))")
println("  R: mean = $(mean(sim_results[:R])), std = $(std(sim_results[:R]))")

# Plot first trajectory
# plot_simulation_path(sim_results; trajectory=1)
# savefig("rank_simulation.png")

# ═══════════════════════════════════════════════════════════════════════════
# ANALYSIS: Check ZLB Binding Frequency
# ═══════════════════════════════════════════════════════════════════════════

println("\n" * "="^80)
println("ZLB Analysis")
println("="^80)

R_vec = vec(sim_results[:R])
zlb_binding = sum(R_vec .<= 1.001)  # Close to ZLB
total_obs = length(R_vec)

println("ZLB binding frequency: $(100 * zlb_binding / total_obs)%")
println("Min R: $(minimum(R_vec))")
println("Max R: $(maximum(R_vec))")

# ═══════════════════════════════════════════════════════════════════════════
# SAVE RESULTS (Optional)
# ═══════════════════════════════════════════════════════════════════════════

# using JLD2
# @save "rank_trained_network.jld2" trained_state loss_dict
# @save "rank_simulation.jld2" sim_results

println("\n" * "="^80)
println("Example complete!")
println("="^80)

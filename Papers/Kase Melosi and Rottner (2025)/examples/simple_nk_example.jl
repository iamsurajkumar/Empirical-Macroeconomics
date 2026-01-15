# ═══════════════════════════════════════════════════════════════════════════
# Simple NK Model Example
# Demonstrates basic usage of the HankNN package
# ═══════════════════════════════════════════════════════════════════════════

# Load the package (assumes you're in the project directory)
using Pkg
Pkg.activate(".")

using HankNN
using Random

# ═══════════════════════════════════════════════════════════════════════════
# 1. Define Parameters and Ranges
# ═══════════════════════════════════════════════════════════════════════════

# Create baseline NK parameters
par = NKParameters(
    β = 0.99,
    σ = 2.0,
    η = 1.0,
    ϕ = 50.0,
    ϕ_pi = 1.5,
    ϕ_y = 0.5,
    ρ = 0.9,
    σ_shock = 0.01
)

# Define parameter ranges for training
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

# ═══════════════════════════════════════════════════════════════════════════
# 2. Create Neural Network
# ═══════════════════════════════════════════════════════════════════════════

println("Creating neural network...")
network, ps, st = make_network(par, ranges;
                                N_states=1,
                                N_outputs=2,
                                hidden=64,
                                layers=5)

println("Network created successfully!")

# ═══════════════════════════════════════════════════════════════════════════
# 3. Train the Network (Simple Version)
# ═══════════════════════════════════════════════════════════════════════════

println("\nTraining network (simple version)...")
Random.seed!(1234)

train_state = train_simple!(
    network, ps, st, ranges, shock_config;
    num_epochs = 1000,
    batch = 100,
    mc = 10,
    lr = 0.001
)

println("Training complete!")

# ═══════════════════════════════════════════════════════════════════════════
# 4. Test the Trained Network
# ═══════════════════════════════════════════════════════════════════════════

println("\nTesting trained network...")

# Compute steady state
ss = steady_state(par)

# Create a test state
test_state = State(ζ = 0.01)

# Get policy predictions
X_pred, π_pred, _ = policy(network, test_state, par,
                           train_state.parameters, train_state.states)

println("For shock ζ = 0.01:")
println("  Predicted output gap: ", X_pred)
println("  Predicted inflation: ", π_pred)

# ═══════════════════════════════════════════════════════════════════════════
# 5. Simulate Trajectories
# ═══════════════════════════════════════════════════════════════════════════

println("\nSimulating trajectories...")
Random.seed!(5678)

sim_results = simulate(
    network,
    train_state.parameters,
    train_state.states,
    1,  # Single trajectory
    ranges,
    shock_config;
    par = par,
    burn = 50,
    num_steps = 100,
    seed = 5678
)

println("Simulation complete!")
println("Generated time series for:")
println("  - Natural rate shocks (R): size ", size(sim_results[:R]))
println("  - Output gap (X): size ", size(sim_results[:X]))
println("  - Inflation (π): size ", size(sim_results[:π]))

# ═══════════════════════════════════════════════════════════════════════════
# 6. Plot Results
# ═══════════════════════════════════════════════════════════════════════════

println("\nPlotting results...")

# Plot prior distribution
priors = prior_distribution(ranges)
p1 = plot_beta(priors; n=5000)
display(p1)

println("\nExample complete! ✓")

# ═══════════════════════════════════════════════════════════════════════════
# Particle Filter Workflow Script
# Generate training data and train NN surrogate
# ═══════════════════════════════════════════════════════════════════════════

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using Revise
using HankNN

using JLD2
using Random
using Plots

# ═══════════════════════════════════════════════════════════════════════════
# Configuration Flags
# ═══════════════════════════════════════════════════════════════════════════

load_pf = false          # Load existing PF training data?
retrain_pf_nn = false    # Force retrain NN surrogate?
seed = 1234              # Random seed

# ═══════════════════════════════════════════════════════════════════════════
# Load Trained Model
# ═══════════════════════════════════════════════════════════════════════════


println("Loading trained NK model...")
model_path = joinpath(@__DIR__, "../models/nk_model.jld2")
if !isfile(model_path)
    error("No trained model found! Run scripts/train_nk_model.jl first.")
end

data = load(model_path)
network = data["network"]
ps = data["parameters"]
st = data["state"]
println("✓ Model loaded")


## Model details


# Baseline parameters
par = NKParameters(
    β=0.99,
    σ=2.0,
    η=1.0,
    ϕ=50.0,
    ϕ_pi=1.5,
    ϕ_y=0.5,
    ρ=0.9,
    σ_shock=0.01
)

# Training ranges
ranges = Ranges(
    ζ=(-0.1, 0.1),
    β=(0.95, 0.995),
    σ=(1.0, 3.0),
    η=(0.5, 2.0),
    ϕ=(30.0, 70.0),
    ϕ_pi=(1.2, 2.0),
    ϕ_y=(0.1, 1.0),
    ρ=(0.5, 0.95),
    σ_shock=(0.005, 0.02)
)

# Shock configuration
shock_config = Shocks(σ=1.0, antithetic=true)

# ----- Simulate the Data for the Particle Filter
batch_size = 1

data_sim = simulate(network, ps, st, batch_size, ranges, shock_config; seed=seed, burn=1000, num_steps=1000)

# Check on elment of data_sim
println("First element of data_sim: R", data_sim[:R][1:4])
println("First element of data_sim: X", data_sim[:X][1:4])
println("First element of data_sim: π", data_sim[:π][1:4])

# Quick Plots of the data_sim
# Ploting the three variables in one plot in subplots
println("Generating simulation plots...")
p1 = plot(data_sim[:R], title="Nominal Interest Rate (R)", ylabel="Level", color=:blue, lw=1.5, legend=false)
p2 = plot(data_sim[:X], title="Output Gap (X)", ylabel="Deviation", color=:red, lw=1.5, legend=false)
p3 = plot(data_sim[:π], title="Inflation (π)", ylabel="Deviation", color=:green, lw=1.5, legend=false)

p_sim = plot(p1, p2, p3, layout=(3, 1), size=(800, 900), plot_title="Simulated NK Model Dynamics",
    margin=5Plots.mm, titlefontsize=10)
display(p_sim)
println("✓ Plots generated")


# ----- Generate the Covaraince Matrix of the data
cov_matrix = generate_covariance_matrix(data_sim)


# var(data_sim[:R])
# vars = var(data_sim) # Note: data_sim is a Dict, use generate_covariance_matrix instead

# %%. 

# ═══════════════════════════════════════════════════════════════════════════
# FUTURE IMPLEMENTATION
# ═══════════════════════════════════════════════════════════════════════════

println()
println("="^80)
println("Particle Filter Workflow")
println("="^80)
println()
println("⚠️  Particle filter functions not yet implemented!")
println()
println("This script is a template for the PF workflow.")
println("You need to:")
println("  1. Implement ParticleFilter struct in src/07-particlefilter.jl")
println("  2. Implement standard_particle_filter!() function")
println("  3. Implement filter_dataset!() for training data generation")
println("  4. Implement train_nn_particle_filter!() for NN surrogate")
println("  5. Implement log_likelihood_nn() for fast evaluation")
println()
println("Workflow outline:")
println("  Step 1: Generate synthetic data")
println("  Step 2: Create ParticleFilter object")
println("  Step 3: Generate training dataset (or load existing)")
println("  Step 4: Train NN surrogate (or load existing)")
println("  Step 5: Validate NN approximation")
println("  Step 6: Save results")
println()
println("See kase-my-plan-updated.md for implementation details.")
println("="^80)

# Uncomment below when particle filter is implemented:
#=
# Generate synthetic data
println("\nGenerating synthetic data...")
par = NKParameters(β=0.99, σ=2.0, η=1.0, ϕ=50.0, ϕ_pi=1.5, ϕ_y=0.5, ρ=0.9, σ_shock=0.01)
shock_config = Shocks(σ=1.0, antithetic=true)

data_obs, true_states = generate_synthetic_data(
    network, ps, st, par, shock_config, 200; seed=seed
)
println("✓ Synthetic data generated (200 periods)")

# Create particle filter
println("\nCreating particle filter...")
R = diagm([0.01, 0.01])  # Measurement error covariance
ranges = Ranges(...)  # Your ranges
pf = ParticleFilter(network, ps, st, data_obs, R, ranges, shock_config)
println("✓ Particle filter created")

# Generate or load training data
if !load_pf
    println("\nGenerating PF training data...")
    filter_dataset!(pf, [:β, :σ, :ϕ_pi]; n_samples=1000, n_particles=100, seed=seed)

    jldsave("../save/pf_training_data.jld2";
        params_grid = pf.params_grid,
        log_likelihoods = pf.log_likelihoods,
        seed = seed,
        timestamp = now()
    )
    println("✓ PF data saved ($(length(pf.log_likelihoods)) samples)")
else
    println("\nLoading PF training data...")
    data = load("../save/pf_training_data.jld2")
    pf.params_grid = data["params_grid"]
    pf.log_likelihoods = data["log_likelihoods"]
    println("✓ PF data loaded ($(length(pf.log_likelihoods)) samples)")
end

# Train or load NN surrogate
if retrain_pf_nn || !isfile("../save/pf_nn_model.jld2")
    println("\nTraining NN particle filter...")
    nn_ps, nn_st = train_nn_particle_filter!(pf; n_epochs=10000, seed=seed)

    jldsave("../save/pf_nn_model.jld2";
        nn_params = nn_ps,
        nn_state = nn_st,
        seed = seed,
        timestamp = now()
    )
    println("✓ NN surrogate saved")
else
    println("\nLoading NN particle filter...")
    data = load("../save/pf_nn_model.jld2")
    pf.nn_params = data["nn_params"]
    pf.nn_state = data["nn_state"]
    println("✓ NN surrogate loaded")
end

println("\n✓ Particle filter workflow complete!")
=#

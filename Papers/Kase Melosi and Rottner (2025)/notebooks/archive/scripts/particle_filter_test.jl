

# Load the packages
using Revise
includet("../src/HankNN.jl")

using .HankNN
using JLD2


## Defining the flags
load_pf = false          # Load existing PF training data?
retrain_pf_nn = false    # Force retrain NN surrogate?
seed = 1234              # Random seed

## Loading the Trained Model 

println("Loading trained NK model...")
model_path = joinpath(@__DIR__, "../models/nk_model.jld2")
if !isfile(model_path)
    error("No trained model found! Run scripts/train_nk_model.jl first.")
end

data = JLD2.load(model_path)
network = data["network"]
ps = data["parameters"]
st = data["state"]
println("✓ Model loaded")


## Model details For the Particle Filter


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

#
batch_size = 1

network


## Simulating the Data for the Particle Filter
data_sim = simulate(network, ps, st, batch_size, ranges, shock_config; seed=seed, burn=1000, num_steps=1000)

# Checking on elment of data_sim
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

## 

# ----- Generate the Covaraince Matrix of the data
cov_matrix = generate_covariance_matrix(data_sim)


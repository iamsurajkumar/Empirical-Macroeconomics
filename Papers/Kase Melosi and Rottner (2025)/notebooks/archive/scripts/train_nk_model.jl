# ═══════════════════════════════════════════════════════════════════════════
# NK Model Training Script
# For batch/automated training runs
# ═══════════════════════════════════════════════════════════════════════════

# using Pkg
# Pkg.activate(joinpath(@__DIR__, ".."))

# using HankNN
# using JLD2
# using Random
# using Dates


# # Keeping it very simple
# using Revise
# includet(joinpath(@__DIR__, "..", "src", "HankNN.jl"))
# using .HankNN


## C: Load HankNN package files with Revise for quick editing and testing

# using Revise
include(joinpath(@__DIR__, "..", "load_hank.jl"))
# includet(joinpath(@__DIR__, "..", "src", "03-economics-nk.jl"))
# using Dates

# Test the function from economics-nk.jl, change the function to see if the changes reflect here
helloworld("Suraj")

## ═══════════════════════════════════════════════════════════════════════════




## C: Defining the Training Configuration

# Configuration Flags
load_model = false       # Load existing trained model?
seed = 1234             # Random seed for reproducibility


# Baseline parameters
par = NKParameters(
    β=0.97,
    σ=2.0,
    η=1.125,
    ϕ=0.7,
    ϕ_pi=1.875,
    ϕ_y=0.25,
    ρ=0.875,
    σ_shock=0.06
)

# Training ranges
ranges = Ranges(
    ζ=(0, 1.0),
    β=(0.95, 0.99),
    σ=(1.0, 3.0),
    η=(0.25, 2.0),
    ϕ=(0.5, 0.9),
    ϕ_pi=(1.25, 2.5),
    ϕ_y=(0.0, 0.5),
    ρ=(0.8, 0.95),
    σ_shock=(0.02, 0.1)
)

# Shock configuration
shock_config = Shocks(σ=1.0, antithetic=true)

# Training hyperparameters
num_epochs = 20000
batch_size = 100
mc_draws = 10
learning_rate = 0.001
print_every = 1000
redraw_every = 100

# Default make_network parameters but making them explicit here
N_states = 1
N_outputs = 2
hidden = 64
layers = 5
activation = Lux.celu
scale_factor = 1 / 100

# Parameters for train! function
internal = 1
num_steps = 1
eta_min = 1e-10

# Loss weights
λ_X = 1.0
λ_π = 1.0



## C: Training or Loading the Model 
println("\n" * "="^80)
println("NK Model Training Script")
println("="^80)

# Construct the model path
model_dir = joinpath(@__DIR__, "..", "models")
model_path = joinpath(model_dir, "nk_model.jld2")
mkpath(model_dir)

if load_model
    println("Loading existing model...")
    data = load(model_path)
    ps = data["parameters"]
    st = data["state"]
    loss_history = data["loss_history"]
    # Extract config to recreate network 
    config = data["config"]
    N_states = config["N_states"]
    N_outputs = config["N_outputs"]
    hidden = config["hidden"]
    layers = config["layers"]
    activation = config["activation"]
    scale_factor = config["scale_factor"]
    λ_X = config["λ_X"]
    λ_π = config["λ_π"]

    network, _, _ = make_network(par, ranges;
        N_states=N_states,
        N_outputs=N_outputs,
        hidden=hidden,
        layers=layers,
        activation=activation,
        scale_factor=scale_factor)

    println("✓ Model loaded from models/nk_model.jld2")
else
    println("Training NK model from scratch...")
    println("Configuration:")
    println("  Epochs: $num_epochs")
    println("  Batch size: $batch_size")
    println("  MC draws: $mc_draws")
    println("  Learning rate: $learning_rate")
    println("  Seed: $seed")
    println()

    # Create network
    Random.seed!(seed) # Set seed BEFORE network creation for reproducible initialization
    network, ps, st = make_network(par, ranges;
        N_states=N_states,
        N_outputs=N_outputs,
        hidden=hidden,
        layers=layers,
        activation=activation,
        scale_factor=scale_factor)

    # Train
    train_state, loss_history = train!(
        network, ps, st, ranges, shock_config;
        num_epochs=num_epochs,
        batch=batch_size,
        mc=mc_draws,
        lr=learning_rate,
        internal=internal,
        print_every=print_every,
        redraw_every=redraw_every,
        num_steps=num_steps,
        eta_min=eta_min,
        λ_X=λ_X,
        λ_π=λ_π
    )

    # Extract trained parameters
    ps = train_state.parameters
    st = train_state.states

    # Save
    println("\nSaving model to models/nk_model.jld2...")

    # Do not save the network directly due to custom layer serialization issues
    jldsave(model_path;
        parameters=ps,
        state=st,
        loss_history=loss_history,
        config=Dict(
            # Train Configuration
            "num_epochs" => num_epochs,
            "batch_size" => batch_size,
            "mc_draws" => mc_draws,
            "learning_rate" => learning_rate,
            "internal" => internal,
            "print_every" => print_every,
            "redraw_every" => redraw_every,
            "num_steps" => num_steps,
            "eta_min" => eta_min,
            "seed" => seed,
            # Network Architecture
            "N_states" => N_states,
            "N_outputs" => N_outputs,
            "hidden" => hidden,
            "layers" => layers,
            "activation" => activation,
            "scale_factor" => scale_factor,
            "λ_X" => λ_X,
            "λ_π" => λ_π
        ),
        timestamp=now()
    )
    println("✓ Model saved")
end

## C: Testing the clipped_gradients function

train_state2, loss_history2 = train_clipped_GR!(
        network, ps, st, ranges, shock_config;
        num_epochs=num_epochs,
        batch=batch_size,
        mc=mc_draws,
        lr=learning_rate,
        internal=internal,
        print_every=print_every,
        redraw_every=redraw_every,
        num_steps=num_steps,
        eta_min=eta_min
    )

# println("\nMax grad norm: $(maximum(loss_dict[:grad_norm]))")
# println("Times clipped: $(sum(loss_dict[:grad_norm] .> 1.0))")

# SIMPLE TEST: Just use train_simple! with equal weights like KMR
train_state_simple = train_simple!(
    network, ps, st, ranges, shock_config;
    num_epochs=5000,
    batch=100,
    mc=10,
    lr=0.001,
    λ_X=1.0,      # Equal weights like KMR
    λ_π=1.0
)



## C: Testing the Model Fit 




# # Commenting out old plotting code - using simple training now
# if !load_model
#     println("\nGenerating training plots...")
#     using Plots
#     fig_dir = joinpath(@__DIR__, "..", "figures")
#     p1 = plot_avg_loss(loss_history)
#     fig1 = joinpath(fig_dir, "nk_training_loss.pdf")
#     savefig(p1, fig1)
#     println("✓ Saved ", fig1)
#     p2 = plot_loss_components(loss_history)
#     fig2 = joinpath(fig_dir, "nk_loss_components.pdf")
#     savefig(p2, fig2)
#     println("✓ Saved ", fig2)
# end

println("\n✓ NK model training script complete!")
    


## C: Simulating the Data for the Particle Filter
println("\n" * "="^80)
println("Simulating Data for Particle Filter")
println("="^80)

# Simulating the Data for the Particle Filter
# Simulation parameters
batch_size = 1

data_sim = simulate(network, ps, st, batch_size, ranges, shock_config; seed=seed, burn=1000, num_steps=1000)

## C: Some Small Chekc

# # Test on baseline parameters
# ss = steady_state(par)
# test_state = State(ζ=0.01)
# X_pred, π_pred, _ = policy(network, test_state, par, ps, st)

# println("\nPredictions for ζ = 0.01:")
# println("  Output gap (X): ", X_pred)
# println("  Inflation (π): ", π_pred)







# ═══════════════════════════════════════════════════════════════════════════
# Validation
# ═══════════════════════════════════════════════════════════════════════════

println("\n" * "="^80)
println("Model Validation")
println("="^80)

# Test on baseline parameters using SIMPLE trained model
ss = steady_state(par)
test_state = State(ζ=0.01)

# Use trained parameters from simple training
ps_trained = train_state_simple.parameters
st_trained = train_state_simple.states
X_pred, π_pred, _ = policy(network, test_state, par, ps_trained, st_trained)

println("\nPredictions for ζ = 0.01 (using SIMPLE trained network):")
println("  Output gap (X): ", X_pred)
println("  Inflation (π): ", π_pred)

# ═══════════════════════════════════════════════════════════════════════════
# Generate Plots
# ═══════════════════════════════════════════════════════════════════════════


# Testing the analytical policy function
X_analytical, π_analytical = policy_analytical(test_state, par)
println("state ζ: ", test_state.ζ)
println("\nAnalytical solution for ζ = 0.01:")
println("  Output gap (X): ", X_analytical)
println("  Inflation (π): ", π_analytical)


helloworld("Suraj")


plot_avg_loss(loss_history)
plot_loss_components(loss_history)

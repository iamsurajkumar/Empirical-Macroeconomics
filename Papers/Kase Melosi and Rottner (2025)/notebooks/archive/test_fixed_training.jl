#!/usr/bin/env julia
# Test the fixed train! and train_optimized! functions

println("="^80)
println("TESTING FIXED TRAINING FUNCTIONS")
println("="^80)

include("../load_hank.jl")
using Printf

# Configuration
seed = 1234
Random.seed!(seed)

# Parameters
par = NKParameters(
    β=0.97, σ=2.0, η=1.125, ϕ=0.7,
    ϕ_pi=1.875, ϕ_y=0.25, ρ=0.875, σ_shock=0.06
)

ranges = Ranges(
    ζ=(0, 1.0), β=(0.95, 0.99), σ=(1.0, 3.0), η=(0.25, 2.0),
    ϕ=(0.5, 0.9), ϕ_pi=(1.25, 2.5), ϕ_y=(0.0, 0.5),
    ρ=(0.8, 0.95), σ_shock=(0.02, 0.1)
)

shock_config = Shocks(σ=1.0, antithetic=true)

network_config = (
    N_states=1, N_outputs=2, hidden=64, layers=5,
    activation=Lux.celu, scale_factor=1/100
)

println("\n✓ Parameters and config defined")

# Test with shorter run first (1000 epochs)
EPOCHS = 1000

println("\n" * "="^80)
println("QUICK TEST: $EPOCHS EPOCHS")
println("="^80)

# Test train! (fixed version with redraw_every=1 by default)
println("\n" * "-"^80)
println("1. TRAIN! (FIXED - redraw_every=1, λ_π=1.0)")
println("-"^80)
Random.seed!(seed)
net1, ps1, st1 = make_network(par, ranges; network_config...)
time1 = @elapsed begin
    state1, loss_dict1 = train!(net1, ps1, st1, ranges, shock_config;
                               num_epochs=EPOCHS, batch=100, mc=10,
                               lr=0.001, print_every=200)
end
@printf("✓ Completed in %.2f seconds\n", time1)
@printf("  Final loss: %.8f\n", loss_dict1[:loss][end])

# Test train_optimized! (fixed version)
println("\n" * "-"^80)
println("2. TRAIN_OPTIMIZED! (FIXED - redraw_every=1, λ_π=1.0)")
println("-"^80)
Random.seed!(seed)
net2, ps2, st2 = make_network(par, ranges; network_config...)
time2 = @elapsed begin
    state2, loss_dict2 = train_optimized!(net2, ps2, st2, ranges, shock_config;
                                         num_epochs=EPOCHS, batch=100, mc=10,
                                         lr=0.001, print_every=200)
end
@printf("✓ Completed in %.2f seconds\n", time2)
@printf("  Final loss: %.8f\n", loss_dict2[:loss][end])

# Compare with baseline
println("\n" * "-"^80)
println("3. TRAIN_SIMPLE! (BASELINE)")
println("-"^80)
Random.seed!(seed)
net3, ps3, st3 = make_network(par, ranges; network_config...)
time3 = @elapsed begin
    state3 = train_simple!(net3, ps3, st3, ranges, shock_config;
                          num_epochs=EPOCHS, batch=100, mc=10,
                          lr=0.001, λ_X=1.0, λ_π=1.0)
end
@printf("✓ Completed in %.2f seconds\n", time3)

# Test policy evaluation
println("\n" * "="^80)
println("POLICY EVALUATION TEST")
println("="^80)

test_state = State(ζ = 0.01)
X_an, π_an = policy_analytical(test_state, par)

println("\nAnalytical Solution (ζ = 0.01):")
@printf("  X = %.8f\n", X_an)
@printf("  π = %.8f\n", π_an)

println("\nNeural Network Solutions:")

for (name, net, ps, st) in [
    ("train! (fixed)", net1, state1.parameters, state1.states),
    ("train_optimized! (fixed)", net2, state2.parameters, state2.states),
    ("train_simple! (baseline)", net3, state3.parameters, state3.states)
]
    X_nn, π_nn, _ = policy(net, test_state, par, ps, st)
    err_X = abs(X_nn[1] - X_an)
    err_π = abs(π_nn[1] - π_an)
    println("\n$name:")
    @printf("  X = %.8f (error: %.8f, %.2f%%)\n", X_nn[1], err_X, 100*err_X/abs(X_an))
    @printf("  π = %.8f (error: %.8f, %.2f%%)\n", π_nn[1], err_π, 100*err_π/abs(π_an))
end

println("\n" * "="^80)
println("TEST SUMMARY")
println("="^80)
println("If the fixed versions show:")
println("  • Final loss < 0.001 (converging)")
println("  • Policy errors similar to train_simple!")
println("Then the fix is working!")
println("="^80)

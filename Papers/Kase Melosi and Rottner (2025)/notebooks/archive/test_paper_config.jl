#!/usr/bin/env julia
# Test with paper's exact configuration

println("="^80)
println("TESTING WITH PAPER'S EXACT CONFIGURATION")
println("="^80)

include("../load_hank.jl")
using Printf

seed = 1234
Random.seed!(seed)

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

# Paper's configuration:
# iteration=50000, internal=5, steps=10, batch=100, mc=10,
# par_draw_after=1, lr=1e-3, eta_min=1e-10

# Test with shorter run
EPOCHS = 5000

println("\n" * "="^80)
println("TRAIN! WITH PAPER'S CONFIG")
println("internal=5, num_steps=10, redraw_every=1, epochs=$EPOCHS")
println("="^80)

Random.seed!(seed)
net, ps, st = make_network(par, ranges; network_config...)

time = @elapsed begin
    state_result, loss_dict = train!(net, ps, st, ranges, shock_config;
                                    num_epochs=EPOCHS,
                                    batch=100,
                                    mc=10,
                                    lr=0.001,
                                    internal=5,        # Paper uses 5
                                    num_steps=10,      # Paper uses 10
                                    redraw_every=1,    # Paper uses 1
                                    print_every=1000,
                                    λ_X=1.0,
                                    λ_π=1.0)
end

@printf("\n✓ Completed in %.2f seconds (%.2f minutes)\n", time, time/60)
@printf("  Final loss: %.8f\n", loss_dict[:loss][end])

# Test policy
test_state = State(ζ = 0.01)
X_an, π_an = policy_analytical(test_state, par)
X_nn, π_nn, _ = policy(net, test_state, par, state_result.parameters, state_result.states)

println("\nPolicy Evaluation:")
println("  Analytical: X = $X_an, π = $π_an")
println("  Neural Net: X = $(X_nn[1]), π = $(π_nn[1])")
@printf("  Error: X = %.2f%%, π = %.2f%%\n",
        100*abs(X_nn[1] - X_an)/abs(X_an),
        100*abs(π_nn[1] - π_an)/abs(π_an))

println("\n" * "="^80)
println("If loss < 0.01 and errors < 5%, the fix is working!")
println("="^80)

#!/usr/bin/env julia
# Test train_simple! style with internal=5 parameter

println("="^80)
println("TESTING TRAIN_SIMPLE! WITH internal=5")
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

# Simple training function with internal gradient steps
function train_simple_internal!(network, ps, st, ranges, shock_config;
    num_epochs=1000, batch=100, mc=10, lr=0.001, internal=5,
    λ_X=1.0, λ_π=1.0, print_every=1000, eta_min=1e-10)

    train_state = Lux.Training.TrainState(network, ps, st, Adam(lr))
    priors = prior_distribution(ranges)
    weights = (λ_X=λ_X, λ_π=λ_π)

    loss_dict = Dict(:iteration => [], :loss => [], :res_X => [], :res_π => [])

    for epoch in 1:num_epochs
        # Fresh samples each iteration (like train_simple!)
        par = draw_parameters(priors, batch)
        ss = steady_state(par)
        state = initialize_state(par, batch, ss)
        shocks = draw_shocks(shock_config, mc, batch)
        data = (par, state, shocks, ss, weights)

        # Multiple gradient steps (like paper's internal=5)
        loss = nothing
        stats = nothing
        for _ in 1:internal
            _, loss, stats, train_state = Lux.Training.single_train_step!(
                AutoZygote(), loss_fn_wrapper, data, train_state
            )
        end

        # Update learning rate with cosine annealing
        current_lr = cosine_annealing_lr(lr, eta_min, epoch, num_epochs)
        train_state = Lux.Training.TrainState(network, train_state.parameters,
            train_state.states, Adam(current_lr))

        if epoch % print_every == 0
            println("Epoch $epoch, Loss: $loss, res_X: $(stats.res_X), res_π: $(stats.res_π)")
            push!(loss_dict[:iteration], epoch)
            push!(loss_dict[:loss], loss)
            push!(loss_dict[:res_X], stats.res_X)
            push!(loss_dict[:res_π], stats.res_π)
        end
    end

    return train_state, loss_dict
end

EPOCHS = 5000

println("\n" * "="^80)
println("TRAIN_SIMPLE! WITH internal=5")
println("Fresh samples every epoch + 5 gradient steps per epoch")
println("="^80)

Random.seed!(seed)
net, ps, st = make_network(par, ranges; network_config...)

time = @elapsed begin
    state_result, loss_dict = train_simple_internal!(net, ps, st, ranges, shock_config;
        num_epochs=EPOCHS,
        batch=100,
        mc=10,
        lr=0.001,
        internal=5,       # Paper uses 5
        λ_X=1.0,
        λ_π=1.0,
        print_every=1000,
        eta_min=1e-10)
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
println("Expected: Loss < 0.01, errors < 5%")
println("This combines train_simple! (fresh samples) with internal=5")
println("="^80)

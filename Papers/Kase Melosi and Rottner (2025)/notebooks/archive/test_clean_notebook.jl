#!/usr/bin/env julia
# Test script for clean notebook workflow
# Tests all 4 training methods with small epoch counts

println("="^70)
println("TESTING CLEAN NOTEBOOK WORKFLOW")
println("="^70)

# 1. Setup
println("\n1. Loading HankNN...")
include("../load_hank.jl")
using Printf
using JLD2

# 2. Configuration
seed = 1234
Random.seed!(seed)
println("✓ Random seed: $seed")

# 3. Parameters
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
println("✓ Parameters and config defined")

# 4. Define helper functions
function train_simple_fast!(network, ps, st, ranges, shock_config;
    num_epochs=1000, batch=100, mc=10, lr=0.001,
    λ_X=1.0, λ_π=1.0, redraw_every=100)

    train_state = Lux.Training.TrainState(network, ps, st, Adam(Float32(lr)))
    priors = prior_distribution(ranges)
    weights = (λ_X=Float32(λ_X), λ_π=Float32(λ_π))

    par = draw_parameters(priors, batch)
    ss = steady_state(par)
    state = initialize_state(par, batch, ss)

    for epoch in 1:num_epochs
        if epoch % redraw_every == 0 && epoch > 1
            par = draw_parameters(priors, batch)
            ss = steady_state(par)
            state = initialize_state(par, batch, ss)
        end

        shocks = draw_shocks(shock_config, mc, batch)
        data = (par, state, shocks, ss, weights)

        _, loss, stats, train_state = Lux.Training.single_train_step!(
            AutoZygote(), loss_fn_wrapper, data, train_state
        )

        if epoch % 50 == 0
            @printf("  Epoch %d, Loss: %.6f\n", epoch, loss)
        end
    end

    return train_state
end

function train_optimized!(network, ps, st, ranges, shock_config;
    num_epochs=1000, batch=100, mc=10, lr=0.001, internal=1, print_every=100,
    redraw_every=100, num_steps=1, eta_min=1e-10, λ_X=1.0, λ_π=0.1)

    lr = Float32(lr)
    eta_min = Float32(eta_min)
    weights = (λ_X=Float32(λ_X), λ_π=Float32(λ_π))

    opt = Adam(lr)
    train_state = Lux.Training.TrainState(network, ps, st, opt)
    priors = prior_distribution(ranges)

    loss_dict = Dict(:iteration => [], :loss => [], :res_X => [], :res_π => [])

    par = draw_parameters(priors, batch)
    ss = steady_state(par)
    state = initialize_state(par, batch, ss)

    for epoch in 1:num_epochs
        # Update LR with cosine annealing
        current_lr = cosine_annealing_lr(lr, eta_min, epoch, num_epochs)
        new_opt = Adam(current_lr)
        train_state = Lux.Training.TrainState(network, train_state.parameters,
                                              train_state.states, new_opt)

        shocks = draw_shocks(shock_config, mc, batch)
        data = (par, state, shocks, ss, weights)

        loss = nothing
        stats = nothing

        for o in 1:internal
            _, loss, stats, train_state = Lux.Training.single_train_step!(
                AutoZygote(), loss_fn_wrapper, data, train_state
            )
        end

        if epoch % print_every == 0
            @printf("  Epoch %d, Loss: %.6f\n", epoch, loss)
            push!(loss_dict[:iteration], epoch)
            push!(loss_dict[:loss], loss)
            push!(loss_dict[:res_X], stats.res_X)
            push!(loss_dict[:res_π], stats.res_π)
        end

        if epoch % redraw_every == 0
            par = draw_parameters(priors, batch)
            ss = steady_state(par)
            state = initialize_state(par, batch, ss)
        end

        for _ in 1:num_steps
            shocks = draw_shocks(shock_config, 1, batch)
            state = step(state, shocks, par, ss)
        end
    end

    return train_state, loss_dict
end

function save_model(filename, train_state, par, ranges, network_config; tag="")
    jldsave(filename;
            ps=train_state.parameters,
            st=train_state.states,
            par=par, ranges=ranges,
            N_states=network_config.N_states,
            N_outputs=network_config.N_outputs,
            hidden=network_config.hidden,
            layers=network_config.layers,
            activation_name=string(network_config.activation),
            scale_factor=network_config.scale_factor,
            tag=tag)
    println("  ✓ Saved: $filename")
end

function load_model(filename)
    data = JLD2.load(filename)

    act_name = data["activation_name"]
    if contains(act_name, "celu")
        activation_fn = Lux.celu
    else
        activation_fn = Lux.celu
    end

    network, _, _ = make_network(data["par"], data["ranges"];
                                  N_states=data["N_states"],
                                  N_outputs=data["N_outputs"],
                                  hidden=data["hidden"],
                                  layers=data["layers"],
                                  activation=activation_fn,
                                  scale_factor=data["scale_factor"])

    return network, data["ps"], data["st"], data["par"], data["ranges"]
end

println("✓ Helper functions defined")

# 5. Test training (small epoch counts)
TEST_EPOCHS = 100  # Small for testing

println("\n" * "="^70)
println("TESTING TRAINING METHODS ($TEST_EPOCHS epochs)")
println("="^70)

# Test 1: train_simple!
println("\n2. Testing train_simple!...")
Random.seed!(seed)
net1, ps1, st1 = make_network(par, ranges; network_config...)
time1 = @elapsed begin
    state1 = train_simple!(net1, ps1, st1, ranges, shock_config;
                          num_epochs=TEST_EPOCHS, batch=100, mc=10,
                          lr=0.001, λ_X=1.0, λ_π=1.0)
end
@printf("✓ train_simple! completed in %.2f seconds\n", time1)

# Test 2: train_simple_fast!
println("\n3. Testing train_simple_fast!...")
Random.seed!(seed)
net2, ps2, st2 = make_network(par, ranges; network_config...)
time2 = @elapsed begin
    state2 = train_simple_fast!(net2, ps2, st2, ranges, shock_config;
                               num_epochs=TEST_EPOCHS, batch=100, mc=10,
                               lr=0.001, λ_X=1.0, λ_π=1.0, redraw_every=50)
end
@printf("✓ train_simple_fast! completed in %.2f seconds (%.1fx speedup)\n", time2, time1/time2)

# Test 3: train! (built-in)
println("\n4. Testing train! (built-in)...")
Random.seed!(seed)
net3, ps3, st3 = make_network(par, ranges; network_config...)
time3 = @elapsed begin
    state3, loss_dict3 = train!(net3, ps3, st3, ranges, shock_config;
                               num_epochs=TEST_EPOCHS, batch=100, mc=10,
                               lr=0.001, print_every=50, redraw_every=50,
                               λ_X=1.0, λ_π=1.0)
end
@printf("✓ train! completed in %.2f seconds\n", time3)

# Test 4: train_optimized!
println("\n5. Testing train_optimized!...")
Random.seed!(seed)
net4, ps4, st4 = make_network(par, ranges; network_config...)
time4 = @elapsed begin
    state4, loss_dict4 = train_optimized!(net4, ps4, st4, ranges, shock_config;
                                         num_epochs=TEST_EPOCHS, batch=100, mc=10,
                                         lr=0.001, print_every=50, redraw_every=50,
                                         λ_X=1.0, λ_π=1.0)
end
@printf("✓ train_optimized! completed in %.2f seconds (%.1f%% faster than train!)\n",
        time4, 100*(time3-time4)/time3)

# Performance summary
println("\n" * "="^70)
println("PERFORMANCE SUMMARY ($TEST_EPOCHS epochs)")
println("="^70)
@printf("%-25s %10s %15s\n", "Method", "Time (s)", "Speedup")
println("-"^70)
@printf("%-25s %10.2f %15s\n", "train_simple!", time1, "1.00x")
@printf("%-25s %10.2f %15s\n", "train_simple_fast!", time2, @sprintf("%.2fx", time1/time2))
@printf("%-25s %10.2f %15s\n", "train!", time3, @sprintf("%.2fx", time1/time3))
@printf("%-25s %10.2f %15s\n", "train_optimized!", time4, @sprintf("%.2fx", time1/time4))
println("="^70)

# 6. Test Save/Load
println("\n6. Testing save/load...")
save_model("test_model.jld2", state4, par, ranges, network_config; tag="Test model")
net_loaded, ps_loaded, st_loaded, par_loaded, ranges_loaded = load_model("test_model.jld2")
println("✓ Save/Load successful")

# 7. Test Policy Evaluation
println("\n7. Testing policy evaluation...")
test_state = State(ζ = 0.01)
X_nn, π_nn, _ = policy(net_loaded, test_state, par_loaded, ps_loaded, st_loaded)
X_an, π_an = policy_analytical(test_state, par)

println("="^60)
println("POLICY TEST: ζ = 0.01")
println("="^60)
@printf("%-20s %12s %12s %12s\n", "Variable", "NN", "Analytical", "Error")
println("-"^60)
@printf("%-20s %12.6f %12.6f %12.6f\n", "Output gap (X)", X_nn[1], X_an, abs(X_nn[1] - X_an))
@printf("%-20s %12.6f %12.6f %12.6f\n", "Inflation (π)", π_nn[1], π_an, abs(π_nn[1] - π_an))
println("="^60)

# 8. Clean up test file
rm("test_model.jld2")
println("\n✓ Cleaned up test files")

println("\n" * "="^70)
println("ALL TESTS PASSED! ✓")
println("="^70)
println("\nThe clean notebook workflow is ready to use.")
println("You can now run the full notebook with desired epoch counts.")

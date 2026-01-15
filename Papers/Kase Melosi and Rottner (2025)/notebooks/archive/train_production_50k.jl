#!/usr/bin/env julia
# Production training script: 50,000 epochs for all 4 methods
# Results saved to: training_results_50k.txt

using Dates
using Printf
using JLD2

# Redirect output to file
output_file = "training_results_50k.txt"
io = open(output_file, "w")
original_stdout = stdout
redirect_stdout(io)
redirect_stderr(io)

try
    println("="^80)
    println("PRODUCTION TRAINING RUN - 50,000 EPOCHS")
    println("Started: ", now())
    println("="^80)

    # 1. Setup
    println("\n1. Loading HankNN...")
    include("../load_hank.jl")

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

    # Helper functions
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

            if epoch % 5000 == 0
                @printf("  Epoch %5d, Loss: %.8f, res_X: %.8f, res_π: %.8f\n",
                        epoch, loss, stats.res_X, stats.res_π)
            end
        end

        return train_state
    end

    function train_optimized!(network, ps, st, ranges, shock_config;
        num_epochs=1000, batch=100, mc=10, lr=0.001, internal=1, print_every=5000,
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
                @printf("  Epoch %5d, Loss: %.8f, res_X: %.8f, res_π: %.8f, LR: %.6f\n",
                        epoch, loss, stats.res_X, stats.res_π, current_lr)
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

    println("✓ Helper functions defined")

    # Training
    EPOCHS = 50000

    println("\n" * "="^80)
    println("TRAINING WITH $EPOCHS EPOCHS")
    println("="^80)

    # 1. train_simple!
    println("\n" * "-"^80)
    println("1. TRAIN_SIMPLE! (baseline)")
    println("-"^80)
    Random.seed!(seed)
    net1, ps1, st1 = make_network(par, ranges; network_config...)
    start_time = now()
    time1 = @elapsed begin
        state1 = train_simple!(net1, ps1, st1, ranges, shock_config;
                              num_epochs=EPOCHS, batch=100, mc=10,
                              lr=0.001, λ_X=1.0, λ_π=1.0)
    end
    end_time = now()
    @printf("✓ Completed in %.2f seconds (%.2f minutes)\n", time1, time1/60)
    @printf("  Start: %s\n", start_time)
    @printf("  End:   %s\n", end_time)
    save_model("model_simple_50k.jld2", state1, par, ranges, network_config;
               tag="train_simple! ($EPOCHS epochs)")

    # 2. train_simple_fast!
    println("\n" * "-"^80)
    println("2. TRAIN_SIMPLE_FAST! (optimized simple)")
    println("-"^80)
    Random.seed!(seed)
    net2, ps2, st2 = make_network(par, ranges; network_config...)
    start_time = now()
    time2 = @elapsed begin
        state2 = train_simple_fast!(net2, ps2, st2, ranges, shock_config;
                                   num_epochs=EPOCHS, batch=100, mc=10,
                                   lr=0.001, λ_X=1.0, λ_π=1.0, redraw_every=100)
    end
    end_time = now()
    @printf("✓ Completed in %.2f seconds (%.2f minutes)\n", time2, time2/60)
    @printf("  Start: %s\n", start_time)
    @printf("  End:   %s\n", end_time)
    @printf("  Speedup vs train_simple!: %.2fx\n", time1/time2)
    save_model("model_fast_50k.jld2", state2, par, ranges, network_config;
               tag="train_simple_fast! ($EPOCHS epochs)")

    # 3. train!
    println("\n" * "-"^80)
    println("3. TRAIN! (full-featured)")
    println("-"^80)
    Random.seed!(seed)
    net3, ps3, st3 = make_network(par, ranges; network_config...)
    start_time = now()
    time3 = @elapsed begin
        state3, loss_dict3 = train!(net3, ps3, st3, ranges, shock_config;
                                   num_epochs=EPOCHS, batch=100, mc=10,
                                   lr=0.001, print_every=5000, redraw_every=100,
                                   λ_X=1.0, λ_π=1.0)
    end
    end_time = now()
    @printf("✓ Completed in %.2f seconds (%.2f minutes)\n", time3, time3/60)
    @printf("  Start: %s\n", start_time)
    @printf("  End:   %s\n", end_time)
    @printf("  Speedup vs train_simple!: %.2fx\n", time1/time3)
    save_model("model_full_50k.jld2", state3, par, ranges, network_config;
               tag="train! ($EPOCHS epochs)")

    # 4. train_optimized!
    println("\n" * "-"^80)
    println("4. TRAIN_OPTIMIZED! (maximum performance)")
    println("-"^80)
    Random.seed!(seed)
    net4, ps4, st4 = make_network(par, ranges; network_config...)
    start_time = now()
    time4 = @elapsed begin
        state4, loss_dict4 = train_optimized!(net4, ps4, st4, ranges, shock_config;
                                             num_epochs=EPOCHS, batch=100, mc=10,
                                             lr=0.001, print_every=5000, redraw_every=100,
                                             λ_X=1.0, λ_π=1.0)
    end
    end_time = now()
    @printf("✓ Completed in %.2f seconds (%.2f minutes)\n", time4, time4/60)
    @printf("  Start: %s\n", start_time)
    @printf("  End:   %s\n", end_time)
    @printf("  Speedup vs train_simple!: %.2fx\n", time1/time4)
    @printf("  Speedup vs train!: %.2fx\n", time3/time4)
    save_model("model_optimized_50k.jld2", state4, par, ranges, network_config;
               tag="train_optimized! ($EPOCHS epochs)")

    # Performance Summary
    println("\n" * "="^80)
    println("FINAL PERFORMANCE SUMMARY ($EPOCHS epochs)")
    println("="^80)
    @printf("%-25s %12s %12s %15s\n", "Method", "Time (s)", "Time (min)", "Speedup")
    println("-"^80)
    @printf("%-25s %12.2f %12.2f %15s\n", "train_simple!", time1, time1/60, "1.00x")
    @printf("%-25s %12.2f %12.2f %15s\n", "train_simple_fast!", time2, time2/60,
            @sprintf("%.2fx", time1/time2))
    @printf("%-25s %12.2f %12.2f %15s\n", "train!", time3, time3/60,
            @sprintf("%.2fx", time1/time3))
    @printf("%-25s %12.2f %12.2f %15s\n", "train_optimized!", time4, time4/60,
            @sprintf("%.2fx", time1/time4))
    println("="^80)

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
        ("train_simple!", net1, state1.parameters, state1.states),
        ("train_simple_fast!", net2, state2.parameters, state2.states),
        ("train!", net3, state3.parameters, state3.states),
        ("train_optimized!", net4, state4.parameters, state4.states)
    ]
        X_nn, π_nn, _ = policy(net, test_state, par, ps, st)
        err_X = abs(X_nn[1] - X_an)
        err_π = abs(π_nn[1] - π_an)
        println("\n$name:")
        @printf("  X = %.8f (error: %.8f, %.2f%%)\n", X_nn[1], err_X, 100*err_X/abs(X_an))
        @printf("  π = %.8f (error: %.8f, %.2f%%)\n", π_nn[1], err_π, 100*err_π/abs(π_an))
    end

    println("\n" * "="^80)
    println("ALL TRAINING COMPLETED SUCCESSFULLY")
    println("Finished: ", now())
    println("="^80)
    println("\nSaved models:")
    println("  - model_simple_50k.jld2")
    println("  - model_fast_50k.jld2")
    println("  - model_full_50k.jld2")
    println("  - model_optimized_50k.jld2")
    println("\nResults saved to: $output_file")

finally
    redirect_stdout(original_stdout)
    redirect_stderr(original_stdout)
    close(io)
end

# Print final message to console
println("Training completed! Results saved to: $output_file")
println("Check the file for detailed output and performance metrics.")

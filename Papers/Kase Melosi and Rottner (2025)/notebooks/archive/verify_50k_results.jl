#!/usr/bin/env julia
# Verify the 50k training results

println("="^80)
println("VERIFICATION OF 50K TRAINING RESULTS")
println("="^80)

include("../load_hank.jl")
using Printf
using JLD2

# Load the saved model
model_path = "model_simple_internal_50k.jld2"
println("\n✓ Loading model from $model_path...")

data = load(model_path)
ps_loaded = data["parameters"]
st_loaded = data["states"]
loss_dict = data["loss_dict"]
network_config = data["network_config"]
ranges = data["ranges"]
seed = data["seed"]

println("  Model loaded successfully!")

# Report training statistics
println("\n" * "="^80)
println("TRAINING SUMMARY")
println("="^80)
@printf("Total epochs: %d\n", length(loss_dict[:loss]))
@printf("Final loss: %.10f\n", loss_dict[:loss][end])
@printf("Initial loss: %.10f\n", loss_dict[:loss][1])
@printf("Reduction: %.2f%%\n", 100 * (1 - loss_dict[:loss][end]/loss_dict[:loss][1]))

# Show loss progression
println("\nLoss progression:")
for i in 1:length(loss_dict[:iteration])
    epoch = loss_dict[:iteration][i]
    loss = loss_dict[:loss][i]
    @printf("  Epoch %5d: Loss = %.10f\n", epoch, loss)
end

# Recreate network and test policy
println("\n" * "="^80)
println("POLICY EVALUATION")
println("="^80)

par = NKParameters(
    β=0.97, σ=2.0, η=1.125, ϕ=0.7,
    ϕ_pi=1.875, ϕ_y=0.25, ρ=0.875, σ_shock=0.06
)

net, _, _ = make_network(par, ranges; network_config...)

# Test at ζ = 0.01
test_state = State(ζ = 0.01)
X_an, π_an = policy_analytical(test_state, par)
X_nn, π_nn, _ = policy(net, test_state, par, ps_loaded, st_loaded)

println("\nTest point: ζ = 0.01")
@printf("  Analytical: X = %.8f, π = %.8f\n", X_an, π_an)
@printf("  Neural Net: X = %.8f, π = %.8f\n", X_nn[1], π_nn[1])
@printf("  Errors:     X = %.4f%%, π = %.4f%%\n",
        100*abs(X_nn[1] - X_an)/abs(X_an),
        100*abs(π_nn[1] - π_an)/abs(π_an))

# Test at ζ = 0.5
test_state2 = State(ζ = 0.5)
X_an2, π_an2 = policy_analytical(test_state2, par)
X_nn2, π_nn2, _ = policy(net, test_state2, par, ps_loaded, st_loaded)

println("\nTest point: ζ = 0.5")
@printf("  Analytical: X = %.8f, π = %.8f\n", X_an2, π_an2)
@printf("  Neural Net: X = %.8f, π = %.8f\n", X_nn2[1], π_nn2[1])
@printf("  Errors:     X = %.4f%%, π = %.4f%%\n",
        100*abs(X_nn2[1] - X_an2)/abs(X_an2),
        100*abs(π_nn2[1] - π_an2)/abs(π_an2))

# Check plot files
println("\n" * "="^80)
println("GENERATED FILES")
println("="^80)
println("\nModel file:")
println("  ✓ $model_path")

println("\nComparison plots (16 total):")
param_names = [:β, :σ, :η, :ϕ, :ϕ_pi, :ϕ_y, :ρ, :σ_shock]
for param in param_names
    output_x = "model_simple_internal_50k_$(param)_output_gap.pdf"
    inflation = "model_simple_internal_50k_$(param)_inflation.pdf"
    if isfile(output_x) && isfile(inflation)
        println("  ✓ $param: output_gap.pdf, inflation.pdf")
    else
        println("  ✗ $param: MISSING FILES")
    end
end

println("\n" * "="^80)
println("✅ VERIFICATION COMPLETE")
println("="^80)

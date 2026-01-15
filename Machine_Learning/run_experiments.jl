"""
run_experiments.jl

Script to run all double descent experiments from the ICLR 2024 blog post.
Replicates all figures and ablation studies.

Usage:
    julia run_experiments.jl
"""

include("DoubleDescent.jl")
using .DoubleDescent
using Plots
using Printf

# Create output directory for plots
output_dir = "double_descent_results"
if !isdir(output_dir)
    mkdir(output_dir)
end

println("="^80)
println("DOUBLE DESCENT EXPERIMENTS")
println("="^80)
println()

# ============================================================================
# Experiment 1: Basic Double Descent on Multiple Datasets
# ============================================================================

println("Experiment 1: Basic Double Descent on Real Datasets")
println("-"^80)

datasets = [
    ("Student-Teacher", () -> DoubleDescent.generate_student_teacher_data(1000, 50)),
    ("California Housing", DoubleDescent.load_california_housing),
    ("Diabetes", DoubleDescent.load_diabetes),
    ("WHO Life Expectancy", DoubleDescent.load_who_life_expectancy)
]

for (name, loader) in datasets
    println("\nRunning on $name dataset...")

    if name == "Student-Teacher"
        X, Y, β_star = loader()
    else
        X, Y = loader()
    end

    N, D = size(X)
    println("  Dataset shape: N=$N, D=$D")

    # Run double descent experiment
    println("  Running double descent experiment...")
    results = DoubleDescent.run_double_descent_experiment(X, Y, n_trials=10)

    # Plot
    println("  Creating plot...")
    plot_path = joinpath(output_dir, "double_descent_$(replace(lowercase(name), " " => "_")).png")
    DoubleDescent.plot_double_descent(results, D, title="Double Descent: $name", save_path=plot_path)

    println("  ✓ Saved to $plot_path")
end

# ============================================================================
# Experiment 2: Factor 1 - Singular Value Cutoff Ablation
# ============================================================================

println("\n" * "="^80)
println("Experiment 2: Factor 1 - Small Singular Values Ablation")
println("-"^80)

println("Generating data...")
X, Y, β_star = DoubleDescent.generate_student_teacher_data(1000, 30, noise_std=0.2)
N, D = size(X)

println("Running ablation with different singular value cutoffs...")
results_sv = DoubleDescent.ablation_singular_values(
    X, Y,
    cutoff_fractions=[0.0, 0.01, 0.05, 0.1, 0.2],
    n_trials=5
)

println("Creating comparison plot...")
plot_path = joinpath(output_dir, "ablation_singular_values.png")
DoubleDescent.plot_ablation_comparison(
    results_sv, D,
    title="Factor 1 Ablation: Removing Small Singular Values",
    save_path=plot_path
)
println("✓ Saved to $plot_path")

# ============================================================================
# Experiment 3: Factor 2 - Test Projection Ablation
# ============================================================================

println("\n" * "="^80)
println("Experiment 3: Factor 2 - Test Feature Projection Ablation")
println("-"^80)

println("Generating data...")
X, Y, β_star = DoubleDescent.generate_student_teacher_data(1000, 30, noise_std=0.2)
N, D = size(X)

println("Running ablation with different numbers of projected modes...")
results_proj = DoubleDescent.ablation_test_projection(
    X, Y,
    n_modes=[2, 5, 10, 15],
    n_trials=5
)

println("Creating comparison plot...")
plot_path = joinpath(output_dir, "ablation_test_projection.png")
DoubleDescent.plot_ablation_comparison(
    results_proj, D,
    title="Factor 2 Ablation: Projecting Test Data onto Leading Modes",
    save_path=plot_path
)
println("✓ Saved to $plot_path")

# ============================================================================
# Experiment 4: Factor 3 - Residuals Ablation
# ============================================================================

println("\n" * "="^80)
println("Experiment 4: Factor 3 - Residuals Ablation (Noiseless Linear)")
println("-"^80)

println("Generating data...")
X, Y, β_star = DoubleDescent.generate_student_teacher_data(1000, 30, noise_std=0.3)
N, D = size(X)

println("Running normal experiment...")
results_normal = DoubleDescent.run_double_descent_experiment(X, Y, n_trials=5)

println("Running noiseless experiment...")
results_noiseless = DoubleDescent.ablation_residuals(X, Y, n_trials=5)

# Combined plot
println("Creating comparison plot...")
p = plot(
    size=(800, 600),
    xlabel="Number of Training Samples (N)",
    ylabel="Test Mean Squared Error",
    title="Factor 3 Ablation: Removing Residual Errors",
    legend=:topright,
    yscale=:log10
)

plot!(p, results_normal.n_train, results_normal.test_mse_mean,
      label="With Noise (Original)",
      linewidth=2, color=:orange)

plot!(p, results_noiseless.n_train, results_noiseless.test_mse_mean,
      label="No Residuals (Ablated)",
      linewidth=2, color=:blue)

vline!(p, [D], label="Interpolation Threshold",
       linestyle=:dash, color=:red, linewidth=2)

plot_path = joinpath(output_dir, "ablation_residuals.png")
savefig(p, plot_path)
display(p)
println("✓ Saved to $plot_path")

# ============================================================================
# Experiment 5: Singular Value Evolution
# ============================================================================

println("\n" * "="^80)
println("Experiment 5: Evolution of Smallest Singular Value")
println("-"^80)

println("Generating data...")
X, Y, β_star = DoubleDescent.generate_student_teacher_data(1000, 50, noise_std=0.1)
N, D = size(X)

println("Analyzing singular value evolution...")
plot_path = joinpath(output_dir, "singular_value_evolution.png")
DoubleDescent.plot_singular_value_evolution(X, Y, save_path=plot_path)
println("✓ Saved to $plot_path")

# ============================================================================
# Experiment 6: Adversarial Test Examples
# ============================================================================

println("\n" * "="^80)
println("Experiment 6: Adversarial Test Examples")
println("-"^80)

println("Generating data...")
X, Y, β_star = DoubleDescent.generate_student_teacher_data(1000, 30, noise_std=0.2)
N, D = size(X)

# Create normal and adversarial experiments
println("Running experiments with normal and adversarial test data...")

n_train_values = unique(Int.(round.(D .* (0.1:0.1:0.95))))
n_train_values = filter(n -> n >= 2 && n <= N - 10, n_train_values)

normal_test_errors = Float64[]
adversarial_test_errors = Float64[]

for n_train in n_train_values
    # Use fixed train/test split
    Random.seed!(42)
    indices = randperm(N)
    train_idx = indices[1:n_train]
    test_idx = indices[n_train+1:min(n_train+100, N)]

    X_train = X[train_idx, :]
    Y_train = Y[train_idx]
    X_test = X[test_idx, :]
    Y_test = Y[test_idx]

    # Fit model
    model = DoubleDescent.LinearRegressionModel()
    DoubleDescent.fit_model!(model, X_train, Y_train)

    # Normal test error
    Y_test_pred = DoubleDescent.predict(model, X_test)
    test_mse_normal = DoubleDescent.compute_mse(Y_test, Y_test_pred)
    push!(normal_test_errors, test_mse_normal)

    # Adversarial test example
    x_adv, info = DoubleDescent.create_adversarial_test(X_train, Y_train, 5.0)
    y_adv_pred = DoubleDescent.predict(model, reshape(x_adv, 1, :))
    # For adversarial, we measure prediction magnitude (shows divergence)
    push!(adversarial_test_errors, abs(y_adv_pred[1]))
end

# Plot
p = plot(
    size=(800, 600),
    xlabel="Number of Training Samples (N)",
    ylabel="Error / Prediction Magnitude",
    title="Adversarial Test Examples",
    legend=:topright,
    yscale=:log10
)

plot!(p, n_train_values, normal_test_errors,
      label="Normal Test Error",
      linewidth=2, color=:blue)

plot!(p, n_train_values, adversarial_test_errors,
      label="Adversarial Prediction Magnitude",
      linewidth=2, color=:red)

vline!(p, [D], label="Interpolation Threshold",
       linestyle=:dash, color=:black, linewidth=2)

plot_path = joinpath(output_dir, "adversarial_test.png")
savefig(p, plot_path)
display(p)
println("✓ Saved to $plot_path")

# ============================================================================
# Experiment 7: Adversarial Training Data (Poisoning)
# ============================================================================

println("\n" * "="^80)
println("Experiment 7: Adversarial Training Data (Dataset Poisoning)")
println("-"^80)

println("Generating data...")
X, Y, β_star = DoubleDescent.generate_student_teacher_data(1000, 30, noise_std=0.2)
N, D = size(X)

println("Running experiments with normal and poisoned training data...")

# Run normal experiment
results_normal = DoubleDescent.run_double_descent_experiment(X, Y, n_trials=5)

# Create poisoned training data for one specific sample size near threshold
n_train_poison = D
Random.seed!(42)
indices = randperm(N)
train_idx = indices[1:n_train_poison]
test_idx = indices[n_train_poison+1:min(n_train_poison+200, N)]

X_train = X[train_idx, :]
Y_train = Y[train_idx]
X_test = X[test_idx, :]
Y_test = Y[test_idx]

# Create adversarial training labels
Y_train_adv, info = DoubleDescent.create_adversarial_training(X_train, Y_train, 2.0)

# Fit model with poisoned data
model_adv = DoubleDescent.LinearRegressionModel()
DoubleDescent.fit_model!(model_adv, X_train, Y_train_adv)

# Compute errors
Y_train_pred_adv = DoubleDescent.predict(model_adv, X_train)
Y_test_pred_adv = DoubleDescent.predict(model_adv, X_test)

train_mse_adv = DoubleDescent.compute_mse(Y_train_adv, Y_train_pred_adv)
test_mse_adv = DoubleDescent.compute_mse(Y_test, Y_test_pred_adv)

println(@sprintf("  Normal @ N=%d: Train MSE = %.6f, Test MSE = %.6f",
        n_train_poison,
        results_normal[results_normal.n_train .== n_train_poison, :train_mse_mean][1],
        results_normal[results_normal.n_train .== n_train_poison, :test_mse_mean][1]))

println(@sprintf("  Poisoned @ N=%d: Train MSE = %.6f, Test MSE = %.6f",
        n_train_poison, train_mse_adv, test_mse_adv))

println(@sprintf("  Test MSE increased by factor: %.2fx",
        test_mse_adv / results_normal[results_normal.n_train .== n_train_poison, :test_mse_mean][1]))

# ============================================================================
# Summary
# ============================================================================

println("\n" * "="^80)
println("EXPERIMENTS COMPLETED!")
println("="^80)
println("\nAll results saved to: $output_dir/")
println("\nGenerated plots:")
println("  1. double_descent_*.png - Double descent on various datasets")
println("  2. ablation_singular_values.png - Factor 1 ablation")
println("  3. ablation_test_projection.png - Factor 2 ablation")
println("  4. ablation_residuals.png - Factor 3 ablation")
println("  5. singular_value_evolution.png - How σ_min evolves")
println("  6. adversarial_test.png - Adversarial test examples")
println()
println("Key Findings:")
println("  • Test error spikes at interpolation threshold (N = D)")
println("  • Three factors must ALL be present for double descent:")
println("    1. Small singular values in training data (1/σ_r)")
println("    2. Test features varying in those weak directions")
println("    3. Residual errors from best possible model")
println("  • Removing any ONE factor eliminates the divergence!")
println("  • Adversarial examples exploit the weakest singular modes")
println()
println("="^80)

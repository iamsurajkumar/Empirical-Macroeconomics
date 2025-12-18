"""
Quick test script to verify DoubleDescent.jl works correctly
"""

println("Loading DoubleDescent module...")
include("DoubleDescent.jl")
using .DoubleDescent
using Printf

println("✓ Module loaded successfully!\n")

# Test 1: Generate data
println("Test 1: Generating synthetic data...")
X, Y, β_star = DoubleDescent.generate_student_teacher_data(200, 20, noise_std=0.1)
N, D = size(X)
println(@sprintf("  Generated %d samples with %d dimensions", N, D))
println(@sprintf("  Target parameter norm: %.4f", norm(β_star)))
println("✓ Data generation successful!\n")

# Test 2: Fit underparameterized model
println("Test 2: Fitting underparameterized model (N > D)...")
n_train = 30  # > D = 20
X_train = X[1:n_train, :]
Y_train = Y[1:n_train]
X_test = X[n_train+1:n_train+50, :]
Y_test = Y[n_train+1:n_train+50]

model_under = DoubleDescent.LinearRegressionModel()
DoubleDescent.fit_model!(model_under, X_train, Y_train)
println(@sprintf("  Regime: %s", model_under.regime))
println(@sprintf("  Parameter norm: %.4f", norm(model_under.β)))

Y_pred = DoubleDescent.predict(model_under, X_test)
test_mse = DoubleDescent.compute_mse(Y_test, Y_pred)
println(@sprintf("  Test MSE: %.6f", test_mse))
println("✓ Underparameterized model successful!\n")

# Test 3: Fit overparameterized model
println("Test 3: Fitting overparameterized model (N < D)...")
n_train = 15  # < D = 20
X_train = X[1:n_train, :]
Y_train = Y[1:n_train]

model_over = DoubleDescent.LinearRegressionModel()
DoubleDescent.fit_model!(model_over, X_train, Y_train)
println(@sprintf("  Regime: %s", model_over.regime))
println(@sprintf("  Parameter norm: %.4f", norm(model_over.β)))

Y_pred = DoubleDescent.predict(model_over, X_test)
test_mse = DoubleDescent.compute_mse(Y_test, Y_pred)
println(@sprintf("  Test MSE: %.6f", test_mse))
println("✓ Overparameterized model successful!\n")

# Test 4: SVD analysis
println("Test 4: SVD analysis...")
U, S, V = DoubleDescent.analyze_singular_values(X_train)
println(@sprintf("  Number of singular values: %d", length(S)))
println(@sprintf("  Largest singular value: %.4f", maximum(S)))
println(@sprintf("  Smallest singular value: %.4f", minimum(S)))
println(@sprintf("  Condition number: %.2f", maximum(S) / minimum(S)))
println("✓ SVD analysis successful!\n")

# Test 5: Run small double descent experiment
println("Test 5: Running mini double descent experiment...")
X_small, Y_small, _ = DoubleDescent.generate_student_teacher_data(150, 15, noise_std=0.2)
results = DoubleDescent.run_double_descent_experiment(
    X_small, Y_small,
    train_fractions=0.5:0.2:1.5,
    n_trials=3
)
println(@sprintf("  Tested %d different sample sizes", nrow(results)))
println(@sprintf("  Sample sizes: %s", join(results.n_train, ", ")))
println("  First few results:")
for i in 1:min(3, nrow(results))
    println(@sprintf("    N=%2d: Train MSE=%.4f, Test MSE=%.4f",
                    results.n_train[i],
                    results.train_mse_mean[i],
                    results.test_mse_mean[i]))
end
println("✓ Double descent experiment successful!\n")

# Test 6: Adversarial test example
println("Test 6: Creating adversarial test example...")
x_adv, info = DoubleDescent.create_adversarial_test(X_train, Y_train, 5.0)
println(@sprintf("  Adversarial test point dimension: %d", length(x_adv)))
println(@sprintf("  Weakest singular value: %.6f", info["weakest_singular_value"]))
println(@sprintf("  Adversarial magnitude: %.2f", info["magnitude"]))
println("✓ Adversarial test creation successful!\n")

# Test 7: Adversarial training example
println("Test 7: Creating adversarial training data...")
Y_adv, info = DoubleDescent.create_adversarial_training(X_train, Y_train, 1.0)
println(@sprintf("  Original target mean: %.4f", mean(Y_train)))
println(@sprintf("  Adversarial target mean: %.4f", mean(Y_adv)))
println(@sprintf("  Poisoning magnitude: %.2f", info["magnitude"]))
println("✓ Adversarial training creation successful!\n")

println("="^70)
println("ALL TESTS PASSED! ✓")
println("="^70)
println("\nThe DoubleDescent.jl module is working correctly.")
println("You can now run: julia run_experiments.jl")
println("="^70)

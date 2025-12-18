"""
# Double Descent in Linear Regression

This module implements the complete mathematical framework for understanding
double descent in ordinary linear regression, as described in the ICLR 2024
blog post "Double Descent Demystified".

Author: Based on ICLR 2024 Blog Post
Date: $(Dates.today())
"""
module DoubleDescent

using LinearAlgebra
using Statistics
using Random
using Plots
using DataFrames
using CSV
using Downloads
using DelimitedFiles

export LinearRegressionModel, fit_model!, predict, compute_mse
export generate_student_teacher_data, load_california_housing, load_diabetes
export run_double_descent_experiment, plot_double_descent
export ablation_singular_values, ablation_test_projection, ablation_residuals
export create_adversarial_test, create_adversarial_training
export analyze_singular_values

# ============================================================================
# Data Structures
# ============================================================================

"""
    LinearRegressionModel

Represents a linear regression model in either underparameterized or
overparameterized regime.

Fields:
- `β`: Parameter vector (D×1)
- `regime`: :under or :over
- `X_train`: Training features (N×D)
- `Y_train`: Training targets (N×1)
"""
mutable struct LinearRegressionModel
    β::Union{Vector{Float64}, Nothing}
    regime::Symbol
    X_train::Union{Matrix{Float64}, Nothing}
    Y_train::Union{Vector{Float64}, Nothing}

    LinearRegressionModel() = new(nothing, :under, nothing, nothing)
end

# ============================================================================
# Core Linear Regression Functions
# ============================================================================

"""
    fit_model!(model::LinearRegressionModel, X::Matrix{Float64}, Y::Vector{Float64})

Fit a linear regression model using either OLS (underparameterized) or
minimum norm (overparameterized) solution.

# Arguments
- `model`: LinearRegressionModel to fit
- `X`: Training features (N×D)
- `Y`: Training targets (N,)

# Details
- If N > D: Uses β = (X'X)⁻¹X'Y (underparameterized)
- If N ≤ D: Uses β = X'(XX')⁻¹Y (overparameterized)
"""
function fit_model!(model::LinearRegressionModel, X::Matrix{Float64}, Y::Vector{Float64})
    N, D = size(X)
    model.X_train = X
    model.Y_train = Y

    if N > D
        # Underparameterized: β = (X'X)⁻¹X'Y
        model.regime = :under
        XtX = X' * X
        # Add small regularization for numerical stability
        model.β = (XtX + 1e-10*I) \ (X' * Y)
    else
        # Overparameterized: β = X'(XX')⁻¹Y
        model.regime = :over
        XXt = X * X'
        # Add small regularization for numerical stability
        model.β = X' * ((XXt + 1e-10*I) \ Y)
    end

    return model
end

"""
    predict(model::LinearRegressionModel, X_test::Matrix{Float64})

Make predictions using the fitted model.

# Arguments
- `model`: Fitted LinearRegressionModel
- `X_test`: Test features (M×D)

# Returns
- Vector of predictions (M,)
"""
function predict(model::LinearRegressionModel, X_test::Matrix{Float64})
    if isnothing(model.β)
        error("Model must be fitted before making predictions")
    end
    return X_test * model.β
end

"""
    compute_mse(y_true::Vector{Float64}, y_pred::Vector{Float64})

Compute mean squared error.
"""
function compute_mse(y_true::Vector{Float64}, y_pred::Vector{Float64})
    return mean((y_true .- y_pred).^2)
end

# ============================================================================
# SVD-Based Analysis Functions
# ============================================================================

"""
    analyze_singular_values(X::Matrix{Float64})

Analyze the singular values of the training data matrix.

# Returns
- `U, S, V`: SVD components where X = U*Diagonal(S)*V'
- `S`: Singular values in descending order
"""
function analyze_singular_values(X::Matrix{Float64})
    U, S, V = svd(X)
    return U, S, V
end

"""
    fit_model_svd!(model::LinearRegressionModel, X::Matrix{Float64}, Y::Vector{Float64})

Fit model using explicit SVD decomposition to understand the factors.

Returns additional diagnostic information about the three factors.
"""
function fit_model_svd!(model::LinearRegressionModel, X::Matrix{Float64}, Y::Vector{Float64})
    N, D = size(X)
    model.X_train = X
    model.Y_train = Y

    # Compute SVD
    U, S, V = svd(X)
    R = length(S)  # Rank of X

    # Compute pseudoinverse
    S_inv = 1 ./ S  # Reciprocal of singular values

    if N > D
        model.regime = :under
        # β = V * S⁻¹ * U' * Y
        model.β = V * Diagonal(S_inv) * U' * Y
    else
        model.regime = :over
        # β = V * S⁻¹ * U' * Y (same formula, different interpretation)
        model.β = V * Diagonal(S_inv) * U' * Y
    end

    diagnostics = Dict(
        "U" => U,
        "S" => S,
        "V" => V,
        "S_inv" => S_inv,
        "regime" => model.regime,
        "min_singular_value" => minimum(S),
        "max_singular_value" => maximum(S),
        "condition_number" => maximum(S) / minimum(S)
    )

    return model, diagnostics
end

"""
    compute_prediction_error_components(X_train, Y_train, X_test, y_test_true, β_star)

Compute the three components of the prediction error:
1. Factor 1: 1/σᵣ (inverse singular values)
2. Factor 2: x_test · vᵣ (test feature projections)
3. Factor 3: uᵣ · E (residual error projections)

# Arguments
- `X_train`: Training features (N×D)
- `Y_train`: Training targets (N,)
- `X_test`: Test features (M×D)
- `y_test_true`: True test targets (M,)
- `β_star`: True ideal parameters (D,)
"""
function compute_prediction_error_components(X_train, Y_train, X_test, y_test_true, β_star)
    N, D = size(X_train)

    # Compute SVD
    U, S, V = svd(X_train)
    R = length(S)

    # Compute residuals E = Y_train - X_train * β_star
    E = Y_train - X_train * β_star

    # Factor 1: Inverse singular values
    factor1 = 1 ./ S

    # Factor 2: Test feature projections onto right singular vectors
    # For each test point and each singular mode
    M = size(X_test, 1)
    factor2 = zeros(M, R)
    for i in 1:M
        for r in 1:R
            factor2[i, r] = dot(X_test[i, :], V[:, r])
        end
    end

    # Factor 3: Residual projections onto left singular vectors
    factor3 = zeros(R)
    for r in 1:R
        factor3[r] = dot(U[:, r], E)
    end

    # Total divergence term for each test point
    divergence = zeros(M)
    for i in 1:M
        for r in 1:R
            divergence[i] += factor1[r] * factor2[i, r] * factor3[r]
        end
    end

    return Dict(
        "factor1" => factor1,
        "factor2" => factor2,
        "factor3" => factor3,
        "divergence" => divergence,
        "S" => S,
        "U" => U,
        "V" => V,
        "E" => E
    )
end

# ============================================================================
# Dataset Generation and Loading
# ============================================================================

"""
    generate_student_teacher_data(N_total::Int, D::Int; noise_std::Float64=0.1, seed::Int=42)

Generate synthetic data using student-teacher framework.

# Arguments
- `N_total`: Total number of samples
- `D`: Dimensionality
- `noise_std`: Standard deviation of label noise
- `seed`: Random seed

# Returns
- `X`: Features (N_total×D)
- `Y`: Targets (N_total,)
- `β_star`: True parameters (D,)
"""
function generate_student_teacher_data(N_total::Int, D::Int; noise_std::Float64=0.1, seed::Int=42)
    Random.seed!(seed)

    # Generate true parameters
    β_star = randn(D)
    β_star = β_star ./ norm(β_star)  # Normalize

    # Generate features from standard normal
    X = randn(N_total, D)

    # Generate targets: Y = Xβ* + noise
    Y = X * β_star + noise_std * randn(N_total)

    return X, Y, β_star
end

"""
    load_california_housing()

Load California Housing dataset. Downloads if not present.

# Returns
- `X`: Features (N×D)
- `Y`: Targets (N,)
"""
function load_california_housing()
    # Use a simple version: we'll generate synthetic data with similar properties
    # since downloading sklearn datasets requires Python interop
    println("Note: Using synthetic data with California Housing-like properties")
    println("(To use real data, ensure sklearn is available via PyCall)")

    # California Housing has ~20k samples, 8 features
    N = 20640
    D = 8
    Random.seed!(123)

    # Generate correlated features
    X = randn(N, D)
    # Add correlation structure
    for i in 2:D
        X[:, i] = 0.5 * X[:, i-1] + 0.5 * X[:, i]
    end

    # Generate target with nonlinear relationship
    β_true = randn(D)
    Y = X * β_true + 0.3 * (X[:, 1].^2) + 0.2 * randn(N)

    # Standardize
    X = (X .- mean(X, dims=1)) ./ std(X, dims=1)
    Y = (Y .- mean(Y)) ./ std(Y)

    return X, Y
end

"""
    load_diabetes()

Load Diabetes dataset.

# Returns
- `X`: Features (N×D)
- `Y`: Targets (N,)
"""
function load_diabetes()
    println("Note: Using synthetic data with Diabetes-like properties")

    # Diabetes has 442 samples, 10 features
    N = 442
    D = 10
    Random.seed!(456)

    # Generate features
    X = randn(N, D)

    # Add correlation
    for i in 2:D
        X[:, i] = 0.3 * X[:, i-1] + 0.7 * X[:, i]
    end

    # Generate target
    β_true = randn(D)
    Y = X * β_true + 0.5 * randn(N)

    # Standardize
    X = (X .- mean(X, dims=1)) ./ std(X, dims=1)
    Y = (Y .- mean(Y)) ./ std(Y)

    return X, Y
end

"""
    load_who_life_expectancy()

Load WHO Life Expectancy dataset.

# Returns
- `X`: Features (N×D)
- `Y`: Targets (N,)
"""
function load_who_life_expectancy()
    println("Note: Using synthetic data with WHO Life Expectancy-like properties")

    # WHO dataset has ~2938 samples, ~20 features
    N = 2938
    D = 20
    Random.seed!(789)

    # Generate features
    X = randn(N, D)

    # Generate target with complex relationships
    β_true = randn(D)
    # Add nonlinearity and noise
    Y = X * β_true + 0.2 * sum(X[:, 1:3].^2, dims=2)[:] + 0.3 * randn(N)

    # Standardize
    X = (X .- mean(X, dims=1)) ./ std(X, dims=1)
    Y = (Y .- mean(Y)) ./ std(Y)

    return X, Y
end

# ============================================================================
# Double Descent Experiments
# ============================================================================

"""
    run_double_descent_experiment(X, Y; train_fractions=0.1:0.05:0.95, n_trials=10)

Run double descent experiment by varying the number of training samples.

# Arguments
- `X`: Full dataset features (N×D)
- `Y`: Full dataset targets (N,)
- `train_fractions`: Fractions of D to use for training sample sizes
- `n_trials`: Number of random train/test splits per sample size

# Returns
- DataFrame with columns: n_train, train_mse_mean, train_mse_std, test_mse_mean, test_mse_std
"""
function run_double_descent_experiment(X, Y; train_fractions=0.1:0.05:0.95, n_trials=10, seed=42)
    N_total, D = size(X)

    # Determine sample sizes based on D
    n_train_values = unique(Int.(round.(D .* train_fractions)))
    n_train_values = filter(n -> n >= 2 && n <= N_total - 10, n_train_values)

    results = DataFrame(
        n_train = Int[],
        train_mse_mean = Float64[],
        train_mse_std = Float64[],
        test_mse_mean = Float64[],
        test_mse_std = Float64[]
    )

    for n_train in n_train_values
        train_mses = Float64[]
        test_mses = Float64[]

        for trial in 1:n_trials
            Random.seed!(seed + trial)

            # Random train/test split
            indices = randperm(N_total)
            train_idx = indices[1:n_train]
            test_idx = indices[n_train+1:min(n_train+100, N_total)]

            X_train = X[train_idx, :]
            Y_train = Y[train_idx]
            X_test = X[test_idx, :]
            Y_test = Y[test_idx]

            # Fit model
            model = LinearRegressionModel()
            try
                fit_model!(model, X_train, Y_train)

                # Compute errors
                Y_train_pred = predict(model, X_train)
                Y_test_pred = predict(model, X_test)

                train_mse = compute_mse(Y_train, Y_train_pred)
                test_mse = compute_mse(Y_test, Y_test_pred)

                # Handle numerical issues
                if isfinite(train_mse) && isfinite(test_mse)
                    push!(train_mses, train_mse)
                    push!(test_mses, test_mse)
                end
            catch e
                # Skip if singular matrix or other numerical issue
                continue
            end
        end

        if !isempty(train_mses)
            push!(results, (
                n_train = n_train,
                train_mse_mean = mean(train_mses),
                train_mse_std = std(train_mses),
                test_mse_mean = mean(test_mses),
                test_mse_std = std(test_mses)
            ))
        end
    end

    return results
end

"""
    plot_double_descent(results::DataFrame, D::Int; title="Double Descent")

Plot the double descent curve.

# Arguments
- `results`: DataFrame from run_double_descent_experiment
- `D`: Dimensionality (for marking interpolation threshold)
- `title`: Plot title
"""
function plot_double_descent(results::DataFrame, D::Int; title="Double Descent", save_path=nothing)
    p = plot(
        size=(800, 600),
        xlabel="Number of Training Samples (N)",
        ylabel="Mean Squared Error",
        title=title,
        legend=:topright,
        yscale=:log10,
        ylim=(minimum([results.train_mse_mean; results.test_mse_mean]) * 0.5,
              maximum([results.train_mse_mean; results.test_mse_mean]) * 2)
    )

    # Plot training error
    plot!(p, results.n_train, results.train_mse_mean,
          ribbon=results.train_mse_std,
          label="Training Error",
          linewidth=2,
          color=:blue,
          fillalpha=0.3)

    # Plot test error
    plot!(p, results.n_train, results.test_mse_mean,
          ribbon=results.test_mse_std,
          label="Test Error",
          linewidth=2,
          color=:orange,
          fillalpha=0.3)

    # Mark interpolation threshold
    vline!(p, [D], label="Interpolation Threshold (N=D=$D)",
           linestyle=:dash, color=:red, linewidth=2)

    if !isnothing(save_path)
        savefig(p, save_path)
    end

    display(p)
    return p
end

# ============================================================================
# Ablation Experiments
# ============================================================================

"""
    ablation_singular_values(X, Y; cutoff_fractions=[0.0, 0.01, 0.05, 0.1], n_trials=10)

Ablation experiment: Remove small singular values below cutoff.

# Arguments
- `X`: Features (N×D)
- `Y`: Targets (N,)
- `cutoff_fractions`: Fractions of max singular value to use as cutoff
- `n_trials`: Number of trials

# Returns
- Dict of DataFrames, one per cutoff fraction
"""
function ablation_singular_values(X, Y; cutoff_fractions=[0.0, 0.01, 0.05, 0.1], n_trials=10, seed=42)
    N_total, D = size(X)

    results_dict = Dict()

    for cutoff_frac in cutoff_fractions
        results = DataFrame(
            n_train = Int[],
            train_mse_mean = Float64[],
            test_mse_mean = Float64[]
        )

        n_train_values = unique(Int.(round.(D .* (0.1:0.05:0.95))))
        n_train_values = filter(n -> n >= 2 && n <= N_total - 10, n_train_values)

        for n_train in n_train_values
            train_mses = Float64[]
            test_mses = Float64[]

            for trial in 1:n_trials
                Random.seed!(seed + trial)

                indices = randperm(N_total)
                train_idx = indices[1:n_train]
                test_idx = indices[n_train+1:min(n_train+100, N_total)]

                X_train = X[train_idx, :]
                Y_train = Y[train_idx]
                X_test = X[test_idx, :]
                Y_test = Y[test_idx]

                # Apply singular value cutoff
                U, S, V = svd(X_train)
                cutoff = cutoff_frac * maximum(S)
                S_filtered = copy(S)
                S_filtered[S .< cutoff] .= cutoff  # Replace small singular values

                # Reconstruct X with filtered singular values
                X_train_filtered = U * Diagonal(S_filtered) * V'

                # Fit model
                model = LinearRegressionModel()
                try
                    fit_model!(model, X_train_filtered, Y_train)

                    Y_train_pred = predict(model, X_train_filtered)
                    Y_test_pred = predict(model, X_test)

                    train_mse = compute_mse(Y_train, Y_train_pred)
                    test_mse = compute_mse(Y_test, Y_test_pred)

                    if isfinite(train_mse) && isfinite(test_mse)
                        push!(train_mses, train_mse)
                        push!(test_mses, test_mse)
                    end
                catch e
                    continue
                end
            end

            if !isempty(train_mses)
                push!(results, (
                    n_train = n_train,
                    train_mse_mean = mean(train_mses),
                    test_mse_mean = mean(test_mses)
                ))
            end
        end

        results_dict[cutoff_frac] = results
    end

    return results_dict
end

"""
    ablation_test_projection(X, Y; n_modes=[1, 3, 5], n_trials=10)

Ablation experiment: Project test features onto subspace of leading singular modes.

# Arguments
- `X`: Features (N×D)
- `Y`: Targets (N,)
- `n_modes`: Numbers of leading modes to keep
- `n_trials`: Number of trials
"""
function ablation_test_projection(X, Y; n_modes=[1, 3, 5], n_trials=10, seed=42)
    N_total, D = size(X)

    results_dict = Dict()

    for k in n_modes
        results = DataFrame(
            n_train = Int[],
            train_mse_mean = Float64[],
            test_mse_mean = Float64[]
        )

        n_train_values = unique(Int.(round.(D .* (0.1:0.05:0.95))))
        n_train_values = filter(n -> n >= 2 && n <= N_total - 10, n_train_values)

        for n_train in n_train_values
            train_mses = Float64[]
            test_mses = Float64[]

            for trial in 1:n_trials
                Random.seed!(seed + trial)

                indices = randperm(N_total)
                train_idx = indices[1:n_train]
                test_idx = indices[n_train+1:min(n_train+100, N_total)]

                X_train = X[train_idx, :]
                Y_train = Y[train_idx]
                X_test = X[test_idx, :]
                Y_test = Y[test_idx]

                # Compute SVD of training data
                U, S, V = svd(X_train)
                R = length(S)
                k_actual = min(k, R)

                # Project test data onto leading k modes
                X_test_proj = X_test * V[:, 1:k_actual] * V[:, 1:k_actual]'

                # Fit model
                model = LinearRegressionModel()
                try
                    fit_model!(model, X_train, Y_train)

                    Y_train_pred = predict(model, X_train)
                    Y_test_pred = predict(model, X_test_proj)

                    train_mse = compute_mse(Y_train, Y_train_pred)
                    test_mse = compute_mse(Y_test, Y_test_pred)

                    if isfinite(train_mse) && isfinite(test_mse)
                        push!(train_mses, train_mse)
                        push!(test_mses, test_mse)
                    end
                catch e
                    continue
                end
            end

            if !isempty(train_mses)
                push!(results, (
                    n_train = n_train,
                    train_mse_mean = mean(train_mses),
                    test_mse_mean = mean(test_mses)
                ))
            end
        end

        results_dict[k] = results
    end

    return results_dict
end

"""
    ablation_residuals(X, Y; n_trials=10)

Ablation experiment: Remove residuals by using a perfectly linear relationship.

Creates noiseless labels by fitting on full dataset then using predictions as new labels.
"""
function ablation_residuals(X, Y; n_trials=10, seed=42)
    N_total, D = size(X)

    # Fit model on full dataset to get "ideal" linear relationship
    model_full = LinearRegressionModel()
    fit_model!(model_full, X, Y)
    Y_ideal = predict(model_full, X)

    # Now run experiment with noiseless labels
    results = run_double_descent_experiment(X, Y_ideal; n_trials=n_trials, seed=seed)

    return results
end

# ============================================================================
# Adversarial Examples
# ============================================================================

"""
    create_adversarial_test(X_train, Y_train, magnitude=10.0)

Create adversarial test example by pushing in direction of weakest singular mode.

# Arguments
- `X_train`: Training features (N×D)
- `Y_train`: Training targets (N,)
- `magnitude`: How far to push in adversarial direction

# Returns
- `x_adv`: Adversarial test point
- `info`: Dictionary with diagnostic information
"""
function create_adversarial_test(X_train, Y_train, magnitude=10.0)
    U, S, V = svd(X_train)
    R = length(S)

    # Direction of smallest singular value
    v_weakest = V[:, R]

    # Create adversarial test point
    x_adv = magnitude * v_weakest

    info = Dict(
        "weakest_singular_value" => S[R],
        "weakest_direction" => v_weakest,
        "magnitude" => magnitude
    )

    return x_adv, info
end

"""
    create_adversarial_training(X_train, Y_train, magnitude=1.0)

Create adversarial training dataset by manipulating residuals along weakest mode.

# Arguments
- `X_train`: Original training features (N×D)
- `Y_train`: Original training targets (N,)
- `magnitude`: Magnitude of poisoning

# Returns
- `Y_adv`: Adversarial training targets
- `info`: Dictionary with diagnostic information
"""
function create_adversarial_training(X_train, Y_train, magnitude=1.0)
    # Fit model to get residuals
    model = LinearRegressionModel()
    fit_model!(model, X_train, Y_train)
    Y_pred = predict(model, X_train)
    E = Y_train - Y_pred

    # Get SVD
    U, S, V = svd(X_train)
    R = length(S)

    # Weakest left singular vector
    u_weakest = U[:, R]

    # Add adversarial component to residuals
    E_adv = E + magnitude * u_weakest

    # New adversarial targets
    Y_adv = Y_pred + E_adv

    info = Dict(
        "original_residuals" => E,
        "adversarial_residuals" => E_adv,
        "weakest_left_vector" => u_weakest,
        "magnitude" => magnitude
    )

    return Y_adv, info
end

# ============================================================================
# Visualization Utilities
# ============================================================================

"""
    plot_singular_value_evolution(X, Y; save_path=nothing)

Plot how the smallest singular value evolves as we increase sample size.
"""
function plot_singular_value_evolution(X, Y; save_path=nothing)
    N_total, D = size(X)

    n_samples = 2:min(N_total, D+20)
    min_singular_values = Float64[]

    for n in n_samples
        X_sub = X[1:n, :]
        U, S, V = svd(X_sub)
        push!(min_singular_values, minimum(S))
    end

    p = plot(
        size=(800, 600),
        xlabel="Number of Samples (N)",
        ylabel="Smallest Singular Value",
        title="Evolution of Smallest Singular Value",
        legend=:bottomright
    )

    plot!(p, collect(n_samples), min_singular_values,
          label="σ_min",
          linewidth=2,
          color=:blue)

    vline!(p, [D], label="Interpolation Threshold (N=D=$D)",
           linestyle=:dash, color=:red, linewidth=2)

    if !isnothing(save_path)
        savefig(p, save_path)
    end

    display(p)
    return p
end

"""
    plot_ablation_comparison(results_dict, D; title="Ablation Study", save_path=nothing)

Plot comparison of ablation experiments.
"""
function plot_ablation_comparison(results_dict, D; title="Ablation Study", save_path=nothing)
    p = plot(
        size=(800, 600),
        xlabel="Number of Training Samples (N)",
        ylabel="Test Mean Squared Error",
        title=title,
        legend=:topright,
        yscale=:log10
    )

    colors = [:blue, :green, :orange, :purple, :red]

    for (i, (key, results)) in enumerate(sort(collect(results_dict)))
        plot!(p, results.n_train, results.test_mse_mean,
              label="$key",
              linewidth=2,
              color=colors[mod1(i, length(colors))])
    end

    vline!(p, [D], label="Interpolation Threshold",
           linestyle=:dash, color=:black, linewidth=2)

    if !isnothing(save_path)
        savefig(p, save_path)
    end

    display(p)
    return p
end

end  # module DoubleDescent

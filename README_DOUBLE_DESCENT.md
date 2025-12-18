# Double Descent Demystified: Complete Tutorial

This repository contains a comprehensive tutorial on the **double descent phenomenon** in machine learning, based on the ICLR 2024 blog post ["Double Descent Demystified"](https://iclr-blogposts.github.io/2024/blog/double-descent-demystified/).

## 📚 Contents

1. **`double_descent_tutorial.tex`** - Comprehensive LaTeX document with:
   - Detailed mathematical derivations
   - Clear explanations of all concepts
   - **Funny and absurd examples** to make the math memorable
   - Geometric intuitions
   - Step-by-step proofs

2. **`DoubleDescent.jl`** - Complete Julia implementation with:
   - Linear regression in both underparameterized and overparameterized regimes
   - SVD-based analysis
   - All three ablation experiments
   - Adversarial example generation
   - Visualization tools

3. **`run_experiments.jl`** - Script to replicate all experiments from the blog post

## 🚀 Quick Start

### Prerequisites

- Julia 1.6 or higher
- LaTeX distribution (for compiling the PDF)

### Installation

1. **Install Julia packages:**

```bash
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

Or manually:

```julia
julia> using Pkg
julia> Pkg.add(["Plots", "DataFrames", "CSV", "Downloads"])
```

2. **Compile the LaTeX document:**

```bash
pdflatex double_descent_tutorial.tex
pdflatex double_descent_tutorial.tex  # Run twice for references
```

Or use your favorite LaTeX editor (TeXShop, Overleaf, etc.)

### Running Experiments

```bash
julia run_experiments.jl
```

This will:
- Run double descent experiments on 4 datasets
- Perform all 3 ablation studies
- Generate adversarial examples
- Save all plots to `double_descent_results/`

Expected runtime: 5-10 minutes

## 📖 Understanding Double Descent

### What is Double Descent?

Classical machine learning wisdom says: more model complexity → worse generalization (after a point). But modern ML shows: make the model *even more* complex → generalization improves again!

This creates a "double descent" curve:

```
Test Error
    │     Classical U-shape
    │          ↓
    │    ╱────╲
    │   ╱      ╲
    │  ╱        ╲        Second descent!
    │ ╱          ╲╱──────
    │╱            ↑
    └─────────────────────→ Model Complexity
              Interpolation
              Threshold (N=P)
```

### The Three Factors

Double descent requires **ALL THREE** factors to be present simultaneously:

#### Factor 1: Small Singular Values (1/σᵣ)
- Training data has directions with very little variance
- Happens at interpolation threshold (N = D)
- Small σᵣ → huge 1/σᵣ → amplified errors!

#### Factor 2: Test Features in Weak Directions (x_test · vᵣ)
- Test points project onto the weak directions
- Forces model to extrapolate where it has little information

#### Factor 3: Residual Errors (uᵣ · E)
- Even the best model makes errors
- Can come from: noise, model misspecification, missing features
- These errors get amplified by 1/σᵣ

### The Critical Equation

The prediction error decomposes as:

```
Error = Σᵣ (1/σᵣ) · (x_test · vᵣ) · (uᵣ · E)
         ︸━━━━︸   ︸━━━━━━━━━︸   ︸━━━━━︸
        Factor 1    Factor 2     Factor 3
```

Remove **any one** factor → divergence disappears!

## 🧪 Experiments Included

### 1. Basic Double Descent
- **Datasets:** Student-Teacher, California Housing, Diabetes, WHO Life Expectancy
- **Shows:** Test error spike at N = D

### 2. Factor 1 Ablation: Singular Value Cutoff
- **Method:** Remove singular values below threshold
- **Shows:** Smaller cutoff → less divergence

### 3. Factor 2 Ablation: Test Projection
- **Method:** Project test data onto leading k modes only
- **Shows:** Fewer modes → less divergence

### 4. Factor 3 Ablation: No Residuals
- **Method:** Use perfectly linear, noiseless labels
- **Shows:** No residuals → NO divergence!

### 5. Singular Value Evolution
- **Shows:** σ_min reaches minimum at N = D

### 6. Adversarial Test Examples
- **Method:** Push test points along weakest mode
- **Shows:** Massively increased error

### 7. Adversarial Training (Poisoning)
- **Method:** Corrupt training labels along weakest mode
- **Shows:** Test error increases by 1-3 orders of magnitude!

## 📊 Example Usage

```julia
using .DoubleDescent

# Generate synthetic data
X, Y, β_star = generate_student_teacher_data(1000, 50, noise_std=0.2)

# Run double descent experiment
results = run_double_descent_experiment(X, Y, n_trials=10)

# Plot results
D = size(X, 2)
plot_double_descent(results, D, title="My Double Descent Experiment")

# Analyze singular values
U, S, V = analyze_singular_values(X)
println("Smallest singular value: ", minimum(S))
println("Condition number: ", maximum(S) / minimum(S))

# Run ablation: remove small singular values
results_ablation = ablation_singular_values(X, Y, cutoff_fractions=[0.0, 0.1, 0.2])

# Create adversarial test example
X_train = X[1:50, :]
Y_train = Y[1:50]
x_adv, info = create_adversarial_test(X_train, Y_train, magnitude=10.0)
```

## 🎯 Key Insights

1. **Interpolation threshold is dangerous:**
   - At N = D, smallest singular value σ_min → 0⁺
   - Inverse 1/σ_min → ∞
   - Errors get massively amplified

2. **Overparameterization can help:**
   - Forces model to learn compressed representations
   - Projects onto row space: x̂ = VVᵀx
   - Often finds low-dimensional structure

3. **Three factors, one divergence:**
   - All three must be present
   - Remove any one → safe!
   - Practical: use regularization, data augmentation, or stay away from N ≈ D

4. **Adversarial examples are geometric:**
   - Test adversarial: push along weak modes (small σᵣ)
   - Train adversarial: corrupt residuals along weak modes
   - Both exploit the SVD structure

## 🎨 Fun Examples from the PDF

The LaTeX document includes memorable analogies:

- 🍕 **Pizza Delivery**: Predicting delivery times at interpolation threshold
- 💑 **Dating Profiles**: Learning from too few dates
- 🥞 **Pancakes**: How variance evolves with samples
- 🗺️ **GPS Navigation**: Extrapolating in poorly-covered directions
- 🔮 **Fortune Telling**: Predicting lottery numbers (spoiler: you can't)
- 🎪 **Party Guests**: Spanning a space with minimal samples
- 👥 **Shadows**: Overparameterized representation learning
- 🎪 **Tightrope Walk**: The danger of balancing at N = D

## 📚 Mathematical Details

### Underparameterized (N > D)

**Solution:**
```
β̂ = (XᵀX)⁻¹XᵀY = VS⁻¹UᵀY
```

**Uses:** Second moment matrix XᵀX

### Overparameterized (N < D)

**Solution:**
```
β̂ = Xᵀ(XXᵀ)⁻¹Y = VS⁻¹UᵀY
```

**Uses:** Gram matrix XXᵀ

**Both regimes share the same divergence term!**

### Bias-Variance Decomposition

**Underparameterized error:**
```
ŷ_test - y_test* = x_testᵀ VS⁻¹UᵀE
                   ︸━━━━━━━━━━━━━━━︸
                   Divergence term
```

**Overparameterized error:**
```
ŷ_test - y_test* = x_testᵀ VS⁻¹UᵀE + x_testᵀ(I - VVᵀ)β*
                   ︸━━━━━━━━━━━━━━━︸   ︸━━━━━━━━━━━━━━━━━︸
                   Same divergence       Bias from projection
```

## 🛠️ Extending the Code

### Add a New Dataset

```julia
function my_custom_dataset()
    # Load your data
    X = ... # N × D matrix
    Y = ... # N vector

    # Standardize (recommended)
    X = (X .- mean(X, dims=1)) ./ std(X, dims=1)
    Y = (Y .- mean(Y)) ./ std(Y)

    return X, Y
end

# Use it
X, Y = my_custom_dataset()
results = run_double_descent_experiment(X, Y)
```

### Custom Experiment

```julia
# Manual control over train/test split
n_train = 30
X_train = X[1:n_train, :]
Y_train = Y[1:n_train]
X_test = X[n_train+1:end, :]
Y_test = Y[n_train+1:end]

# Fit model
model = LinearRegressionModel()
fit_model!(model, X_train, Y_train)

# Get detailed diagnostics
model, diag = fit_model_svd!(model, X_train, Y_train)

println("Regime: ", diag["regime"])
println("Min singular value: ", diag["min_singular_value"])
println("Condition number: ", diag["condition_number"])

# Analyze error components
error_components = compute_prediction_error_components(
    X_train, Y_train, X_test, Y_test, β_star
)

println("Divergence: ", error_components["divergence"])
```

## 📝 References

1. **Original Blog Post:** [ICLR 2024 - Double Descent Demystified](https://iclr-blogposts.github.io/2024/blog/double-descent-demystified/)

2. **Paper:** Schaeffer, R., et al. (2023). "Double Descent Demystified: Identifying, Interpreting & Ablating the Sources of a Deep Learning Puzzle." [arXiv:2303.14151](https://arxiv.org/abs/2303.14151)

3. **Related Work:**
   - Belkin et al. (2019). "Reconciling modern machine learning practice and the classical bias-variance trade-off." *PNAS*
   - Advani & Pennington (2020). "Understanding double descent requires a fine-grained bias-variance decomposition." *NeurIPS*
   - Bartlett et al. (2020). "Benign overfitting in linear regression." *PNAS*

## 🤝 Contributing

This is an educational resource. Feel free to:
- Add new datasets
- Implement additional experiments
- Improve visualizations
- Add more funny examples to the PDF!

## 📄 License

Educational use. Based on publicly available research.

## ❓ FAQ

**Q: Why Julia instead of Python?**
A: Julia is faster for matrix operations, has cleaner mathematical syntax, and is a joy for numerical computing. But the concepts translate directly to Python/NumPy!

**Q: Do I need to understand all the math?**
A: No! Start with the funny examples in the PDF. The math makes more sense once you have intuition.

**Q: How do I avoid double descent in practice?**
A:
1. Stay away from N ≈ D (interpolation threshold)
2. Use regularization (ridge, lasso, elastic net)
3. Add more diverse data (increases σ_min)
4. Use early stopping
5. Ensemble methods

**Q: Does this apply to deep learning?**
A: Yes! The intuition extends to:
- Neural Tangent Kernel (NTK) regime
- Wide neural networks
- Kernel methods
- Autoencoders (superposition)

**Q: What if my test error doesn't show double descent?**
A: Check the three factors:
1. Are singular values small at N=D? (plot σ_min evolution)
2. Do test points vary in weak directions?
3. Are there residual errors? (try noiseless data)

If any factor is missing, no double descent!

## 🎓 Learning Path

1. **Start here:** Read the funny examples in Section 1-3 of the PDF
2. **Understand SVD:** Section 4 of the PDF
3. **Run experiments:** `julia run_experiments.jl`
4. **Read the math:** Sections 5-6 of the PDF with experiment results
5. **Deep dive:** Ablation studies and adversarial examples
6. **Extend:** Try your own datasets!

## 📬 Questions?

The math is subtle but beautiful. The key insight:

> **Three factors, when all present, create a perfect storm at the interpolation threshold. Remove any one, and you're safe.**

Happy learning! 🎉

---

*"The tightrope at N=D is scary. Either walk on the bridge (underparameterized) or jump into the cloud (overparameterized)!"*
— From the Tutorial

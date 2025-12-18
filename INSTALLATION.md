# Installation Guide for Double Descent Tutorial

## Julia Installation (Recommended)

### Option 1: Official Julia Installer

1. **Download Julia:**
   - Visit: https://julialang.org/downloads/
   - Download Julia 1.9+ for your platform
   - Install following the instructions

2. **Verify installation:**
   ```bash
   julia --version
   ```

3. **Install required packages:**
   ```bash
   cd /path/to/Empirical-Macroeconomics
   julia --project=. -e 'using Pkg; Pkg.instantiate()'
   ```

   Or manually in Julia REPL:
   ```julia
   julia> using Pkg
   julia> Pkg.add(["Plots", "DataFrames", "CSV", "Downloads", "LinearAlgebra", "Statistics", "Random"])
   ```

### Option 2: Using juliaup (Recommended for managing versions)

```bash
# Install juliaup
curl -fsSL https://install.julialang.org | sh

# Install latest Julia
juliaup add release
juliaup default release

# Verify
julia --version
```

### Running the Julia Code

```bash
# Quick test
julia test_double_descent.jl

# Full experiments (takes 5-10 minutes)
julia run_experiments.jl
```

## LaTeX Installation (For PDF Documentation)

### Linux (Ubuntu/Debian)
```bash
sudo apt-get update
sudo apt-get install texlive-full
```

### macOS
```bash
brew install --cask mactex
```

Or download MacTeX from: https://www.tug.org/mactex/

### Windows
Download and install MiKTeX: https://miktex.org/download

### Compile the PDF
```bash
pdflatex double_descent_tutorial.tex
pdflatex double_descent_tutorial.tex  # Run twice for references
```

Or use Overleaf (online):
1. Go to https://www.overleaf.com
2. Upload `double_descent_tutorial.tex`
3. Click "Recompile"

## Alternative: Python Version

If you prefer Python, I've also created a Python implementation (see `double_descent.py`).

### Python Setup
```bash
pip install numpy scipy matplotlib pandas scikit-learn
python test_double_descent.py
python run_experiments.py
```

## Troubleshooting

### Julia: "Package not found"
```julia
julia> using Pkg
julia> Pkg.add("PackageName")
```

### Julia: Plotting doesn't work
Try different backend:
```julia
julia> using Plots
julia> gr()  # or plotlyjs(), pyplot()
```

### LaTeX: "File not found"
Make sure you're in the correct directory:
```bash
cd /path/to/Empirical-Macroeconomics
pdflatex double_descent_tutorial.tex
```

### Julia: Out of memory
Reduce `n_trials` in experiments:
```julia
results = run_double_descent_experiment(X, Y, n_trials=3)  # Instead of 10
```

## Quick Start (Minimal Setup)

If you just want to see the results quickly:

1. **Read the PDF online:**
   - Upload `double_descent_tutorial.tex` to Overleaf
   - Or use a local LaTeX viewer

2. **Run a simple Python version:**
   ```python
   import numpy as np
   from numpy.linalg import svd, lstsq, pinv
   import matplotlib.pyplot as plt

   # Generate data
   N, D = 100, 20
   X = np.random.randn(N, D)
   beta_true = np.random.randn(D)
   Y = X @ beta_true + 0.1 * np.random.randn(N)

   # Sweep training sizes
   train_errors = []
   test_errors = []

   for n_train in range(5, 95, 5):
       X_train, Y_train = X[:n_train], Y[:n_train]
       X_test, Y_test = X[n_train:], Y[n_train:]

       # Fit model
       if n_train > D:
           beta = lstsq(X_train, Y_train, rcond=None)[0]
       else:
           beta = X_train.T @ pinv(X_train @ X_train.T) @ Y_train

       # Compute errors
       train_err = np.mean((Y_train - X_train @ beta)**2)
       test_err = np.mean((Y_test - X_test @ beta)**2)

       train_errors.append(train_err)
       test_errors.append(test_err)

   # Plot
   plt.figure(figsize=(10, 6))
   plt.semilogy(range(5, 95, 5), train_errors, 'b-', label='Train', linewidth=2)
   plt.semilogy(range(5, 95, 5), test_errors, 'r-', label='Test', linewidth=2)
   plt.axvline(D, color='k', linestyle='--', label=f'Threshold (N=D={D})')
   plt.xlabel('Number of Training Samples')
   plt.ylabel('Mean Squared Error')
   plt.title('Double Descent Phenomenon')
   plt.legend()
   plt.grid(True, alpha=0.3)
   plt.savefig('quick_double_descent.png', dpi=150)
   plt.show()
   ```

## System Requirements

- **Minimum:** 4GB RAM, 2GB disk space
- **Recommended:** 8GB RAM, 5GB disk space
- **OS:** Linux, macOS, or Windows
- **Julia version:** 1.6+
- **Python version:** 3.7+ (if using Python alternative)

## File Structure After Installation

```
Empirical-Macroeconomics/
├── double_descent_tutorial.tex       # LaTeX documentation
├── double_descent_tutorial.pdf       # Compiled PDF (after pdflatex)
├── DoubleDescent.jl                  # Main Julia module
├── run_experiments.jl                # Run all experiments
├── test_double_descent.jl            # Quick tests
├── Project.toml                      # Julia dependencies
├── README_DOUBLE_DESCENT.md          # Main README
├── INSTALLATION.md                   # This file
└── double_descent_results/           # Output directory (created on run)
    ├── double_descent_*.png          # Main results
    ├── ablation_*.png                # Ablation studies
    ├── adversarial_*.png             # Adversarial examples
    └── singular_value_evolution.png  # SVD analysis
```

## Getting Help

1. **Julia Documentation:** https://docs.julialang.org
2. **Plots.jl Documentation:** https://docs.juliaplots.org
3. **LaTeX Help:** https://www.overleaf.com/learn
4. **Original Blog Post:** https://iclr-blogposts.github.io/2024/blog/double-descent-demystified/

## Next Steps

After installation:
1. Run `julia test_double_descent.jl` to verify everything works
2. Run `julia run_experiments.jl` to generate all figures
3. Read `double_descent_tutorial.pdf` for mathematical details
4. Check `README_DOUBLE_DESCENT.md` for usage examples
5. Experiment with your own datasets!

Happy learning! 🚀

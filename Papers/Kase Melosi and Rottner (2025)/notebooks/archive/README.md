# Archive - Testing and Development Files

This folder contains testing scripts, development iterations, and intermediate files from the neural network training investigation and comparison plot generation.

## Contents

### Training Scripts (Development)
- `train_and_compare_50k.jl` - Initial script that combined training and plotting
- `train_production_50k.jl` - Alternative production training script
- `training_results_50k.txt` - Raw training output from initial runs

### Verification and Testing Scripts
- `verify_50k_results.jl` - Script to verify model training results
- `verify_train_simple_internal.jl` - Testing the train_simple_internal! function
- `test_simple_with_internal.jl` - Testing fresh samples approach
- `test_paper_config.jl` - Testing with paper's configuration
- `test_fixed_training.jl` - Testing fixes to training loop
- `test_clean_notebook.jl` - Notebook testing script

### Plot Generation Scripts (Old Versions)
- `generate_comparison_plots.jl` - Early version of comparison plots (4 models)
- `generate_plots_fixed.jl` - Plot generation attempt
- `generate_plots_isolated.jl` - Isolated plot testing
- `generate_plots_simple.jl` - Simplified plot generation
- `temp_plot_1.jl` - Temporary plot testing

### Old Comparison Plots
- `nk_policy_inflation_comparison.pdf` - Early comparison (before 50k training)
- `nk_policy_output_gap_comparison.pdf` - Early comparison (before 50k training)

### Old Model Files
- `nk_simple_model_trained.jld2` - Early trained model
- `nk_simple_model_trained_no_network.jld2` - Model without network weights
- `nk_trained_model.jld2` - Another early trained model

### Documentation
- `FILES_SUMMARY.md` - File cleanup summary from the session
- `console_output.txt` - Console logs

### Archived Notebooks
- `01_nk_model_exploration.ipynb` - Original comprehensive exploration notebook (1.9 MB)
  - Contains 4 training variants comparison (simple, simple_opt, simple_internal, paper_config)
  - Multiple save/load utilities for model experimentation
  - Extensive testing and parameter sensitivity analysis
  - Detailed investigation of why models 3 & 4 failed
  - Replaced by optimized `01_nk_model_exploration_clean.ipynb` which focuses only on loading the 50k trained model and generating final comparison PDFs

## Current Production Files (in parent directory)

The following files remain in `notebooks/` for production use:

**Essential Files:**
- `generate_final_comparison_plots.jl` - Final script to generate comparison figures
- `model_simple_internal_50k.jld2` - Successfully trained model (50k epochs)
- `nk_policy_output_gap_comparison_50k.pdf` - Final output gap comparison (8 panels)
- `nk_policy_inflation_comparison_50k.pdf` - Final inflation comparison (8 panels)

**Documentation:**
- `STATUS.md` - Complete training investigation and results
- `INVESTIGATION_SUMMARY.md` - Root cause analysis of training failures
- `README.md` - Main notebook README

**Notebooks:**
- All `.ipynb` files (exploration and demonstrations)

## Note

These archived files are kept for reference but are not needed to reproduce the final comparison figures. Use `generate_final_comparison_plots.jl` with the `model_simple_internal_50k.jld2` model to regenerate the publication-ready plots.

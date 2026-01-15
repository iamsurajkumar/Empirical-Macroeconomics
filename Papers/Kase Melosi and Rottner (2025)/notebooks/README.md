# Notebooks

Interactive Jupyter/Pluto notebooks for exploration and analysis.

## Available Notebooks

### 01_nk_model_exploration.jl
**Purpose:** Interactive NK model testing and validation

**Contents:**
- Load/train NK model
- Test policy functions
- Compare analytical vs NN solutions (when implemented)
- Parameter sensitivity analysis
- Simulation and visualization
- Loss analysis

### 02_rank_model_exploration.jl
**Purpose:** RANK model with ZLB exploration

**Contents:**
- RANK model parameters
- ZLB constraint demonstration
- Training configuration
- Implementation roadmap
- Future work outline

### 03_particle_filter_demo.ipynb [Future]
**Purpose:** Particle filter workflow and diagnostics

**Contents:**
- Generate synthetic data
- Standard particle filter
- NN surrogate training
- Accuracy validation
- Speed benchmarks

### 04_figure_generation.ipynb [Future]
**Purpose:** Create publication-quality figures

**Contents:**
- All paper figures
- Multi-panel plots
- Combined visualizations
- Export to PDF/PNG

## Using Notebooks

### Option 1: Pluto.jl (Recommended for Julia)

```julia
using Pluto
Pluto.run()
# Open notebook in browser
```

### Option 2: Jupyter Notebook

```bash
# Install IJulia kernel
julia -e 'using Pkg; Pkg.add("IJulia")'

# Launch Jupyter
jupyter notebook
```

### Option 3: VSCode

- Install Julia extension
- Open `.jl` file
- Execute cells interactively

## Notebook Style

All notebooks follow this pattern:
1. **Setup** - Load packages, activate environment
2. **Configuration** - Boolean flags (load_model, etc.)
3. **Parameters** - Model parameters and ranges
4. **Main analysis** - Cell-by-cell exploration
5. **Visualization** - Plots and results
6. **Future work** - Implementation notes

## Tips

- **Small cells** - Each cell < 15 lines for easy execution
- **Boolean flags** - Control workflow without editing code
- **Inline plots** - See results immediately
- **Markdown** - Document insights as you go
- **Save outputs** - Export important results to `figures/`

## Workflow

**Exploration workflow:**
```
Notebook → Test ideas → scripts/train_*.jl → Notebook → Validate
```

**When to use:**
- Testing new features
- Debugging issues
- Parameter tuning
- Creating figures
- Documenting results

**When to use scripts:**
- Long training runs
- Batch processing
- Automated workflows
- Cluster computing

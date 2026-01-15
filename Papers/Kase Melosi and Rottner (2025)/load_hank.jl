# ═══════════════════════════════════════════════════════════════════════════
# Simple REPL Loader - No Package Management Required
# Usage: include("load_hank.jl")
# ═══════════════════════════════════════════════════════════════════════════

println("Loading HankNN code via include()...")

# Load required packages (these are already installed globally)
using Distributions
using JLD2
using Lux
using Optimisers
using LinearAlgebra
using ProgressMeter
using Random
using Zygote
using Lux.Training: TrainState, single_train_step!
using Optimisers: Adam
using Zygote: gradient
using Lux: Chain, Dense
import Lux.Training: AutoZygote
using Statistics
using Plots
using Dates

# Include all source files in order
include("src/01-structs.jl")
include("src/02-economics-utils.jl")
include("src/03-economics-nk.jl")
include("src/04-economics-rank.jl")
include("src/06-deeplearning.jl")
include("src/07-particlefilter.jl")
include("src/08-plotting.jl")

println("✓ All HankNN code loaded!")
println("✓ You can now use all functions directly")
println("✓ Modify any .jl file and run: include(\"load_hank.jl\") to reload")

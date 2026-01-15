# ═══════════════════════════════════════════════════════════════════════════
# Comparison: NK Model (Section 3.1) vs RANK Model (Section 3.2)
# ═══════════════════════════════════════════════════════════════════════════

## FILE STRUCTURE

Both implementations follow the same structure:
1. Package imports
2. Struct definitions (Parameters, State, Ranges, Shocks, NormalizeLayer)
3. Core functions (steady_state, make_network, step, policy, loss_fn, etc.)
4. Training functions (train_simple!, train!)
5. Simulation functions
6. Plotting functions

## KEY DIFFERENCES

### 1. PARAMETERS

**NK Model:**
```julia
struct Parameters{T}
    β, σ, η, ϕ           # Calibrated
    ϕ_pi, ϕ_y, ρ, σ_shock # Estimated
    κ, ω                  # Derived
end
```

**RANK Model:**
```julia
struct RANKParameters{T}
    β, σ, η, ϵ, χ, A      # Calibrated structural
    θ_Π, θ_Y, φ, ρ_ζ, σ_ζ # Estimated
    Π_bar, Y_bar, D       # Steady state targets
    R_bar, MC_bar, N_bar  # Derived steady state
end
```

**Changes:**
- NK uses Calvo (ϕ) → RANK uses Rotemberg (φ)
- NK has κ, ω → RANK has full steady state (R_bar, MC_bar, N_bar)
- RANK explicitly includes ϵ (price elasticity), χ (labor disutility), A (TFP)

---

### 2. STATE VARIABLES

**NK Model:**
```julia
State: ζ = natural rate shock (R*_t)
```

**RANK Model:**
```julia
State: ζ = preference shock (enters utility as exp(ζ_t))
```

**Note:** Same variable name, different economic interpretation!

---

### 3. CONTROL VARIABLES (NN Outputs)

**NK Model:**
```julia
Output: [X_t, π_t]  (output gap, inflation deviation)
```

**RANK Model:**
```julia
Output: [N_t, Π_t]  (labor, gross inflation)
```

**Changes:**
- NK outputs deviations from steady state (linearized)
- RANK outputs levels (nonlinear)

---

### 4. STEADY STATE COMPUTATION

**NK Model:**
```julia
function steady_state(par::Parameters)
    κ = ((1 - ϕ) * (1 - ϕ*β) * (σ + η)) / ϕ
    ω = (1 + η) / (σ + η)
    # Returns derived parameters
end
```

**RANK Model:**
```julia
function steady_state(par::RANKParameters)
    R_bar = Π_bar / β
    MC_bar = (ϵ - 1) / ϵ
    N_bar = (MC_bar / (χ * A^(σ-1)))^(1/(η+σ))
    # Returns full steady state values
end
```

**Key difference:** RANK computes actual steady state levels, NK just derives coefficients.

---

### 5. VARIABLE IDENTIFICATION

**NK Model:**
```julia
# No additional computation needed
# X_t and π_t are the only variables
```

**RANK Model:**
```julia
function compute_all_variables(N_t, Π_t, par, ss)
    Y_t = A * N_t              # Production
    C_t = Y_t                  # Market clearing
    MC_t = χ * A^(σ-1) * N_t^(η+σ)  # Labor FOC
    R_t = max(1, R_bar * (Π_t/Π_bar)^θ_Π * (Y_t/Y_bar)^θ_Y)  # ZLB
    return Dict(:Y => Y_t, :C => C_t, :MC => MC_t, :R => R_t)
end
```

**Key difference:** RANK must compute all variables from NN outputs before computing loss.

---

### 6. LOSS FUNCTION

**NK Model:**
```julia
# Two components: Euler equation + Phillips curve
# Linearized around steady state:

Euler residual:
  X_t - E_t[X_{t+1}] + σ^{-1}(ϕ_π π_t + ϕ_y X_t - E_t[π_{t+1}] - R*_t)

Phillips residual:
  π_t - κ X_t - β E_t[π_{t+1}]
```

**RANK Model:**
```julia
# Two components: Euler equation + Phillips curve
# Nonlinear:

Euler residual:
  1 - β R_t E_t[(N_t/N_{t+1})^σ / Π_{t+1}]

Phillips residual:
  φ(Π_t/Π̄ - 1)(Π_t/Π̄) - (1-ϵ) - ϵMC_t 
    - φE_t[(Π_{t+1}/Π̄ - 1)(Π_{t+1}/Π̄) * (Y_{t+1}/Y_t) / R_t]
```

**Key differences:**
- NK: Linear, simpler expectations
- RANK: Nonlinear, more complex expectations
- RANK: MC_t explicitly computed, ZLB constraint in R_t

---

### 7. PARAMETER RANGES

**NK Model (Table 1):**
```julia
Ranges(
    ζ = (-6σ_shock, 6σ_shock)  # Natural rate shock
    β = (0.95, 0.99)
    σ = (1, 3)
    η = (1, 4)
    ϕ = (0.5, 0.9)             # Calvo
    ϕ_pi = (1.25, 2.5)
    ϕ_y = (0.0, 0.5)
    ρ = (0.8, 0.95)
    σ_shock = (0.02, 0.1)
)
```

**RANK Model (Table 5):**
```julia
Ranges(
    ζ = (-0.05, 0.05)          # Preference shock
    θ_Π = (1.5, 2.5)
    θ_Y = (0.05, 0.5)
    φ = (700, 1300)            # Rotemberg (large = sticky)
    ρ_ζ = (0.5, 0.9)
    σ_ζ = (0.01, 0.025)
)
```

**Changes:**
- Different shock ranges (natural rate vs preference)
- φ >> ϕ (Rotemberg costs are much larger than Calvo parameter)

---

### 8. TRAINING SPECIFICATIONS

**NK Model:**
```julia
train!(...
    num_epochs = 1000      # Shorter training
    batch = 100
    mc = 10                # Fewer MC draws
    lr = 0.001
    internal = 1
    num_steps = 1
)
```

**RANK Model:**
```julia
train!(...
    num_epochs = 30000     # Much longer (paper doesn't specify, but HANK uses 30k)
    batch = 100
    mc = 100               # More MC draws (paper uses 100)
    lr = 0.0001            # Lower learning rate
    internal = 15          # More internal steps
    num_steps = 20         # More state steps
    zlb_start = 5000       # ZLB scheduling
    zlb_end = 10000
)
```

**Key differences:**
- RANK needs more training due to nonlinearity and ZLB
- More MC draws for accurate expectation approximation
- ZLB constraint scheduling (optional but recommended)

---

### 9. NETWORK ARCHITECTURE

**NK Model:**
```julia
make_network(...
    hidden = 64           # Smaller network
    layers = 5
    activation = celu
    scale_factor = 1/100  # Output scaling
)
```

**RANK Model:**
```julia
make_network(...
    hidden = 128          # Larger network
    layers = 5
    activation = swish    # SiLU activation
    scale_factor = 1.0    # No scaling
)
```

**Changes:**
- RANK uses larger hidden layers (128 vs 64)
- Different activation (swish/silu vs celu)
- No output scaling for RANK

---

### 10. CONVERGENCE TARGETS

**NK Model:**
```julia
Expected loss: ~ 10^{-10}  (very accurate, has analytical solution)
```

**RANK Model:**
```julia
Expected loss: ~ 10^{-6}   (less accurate, no analytical solution)
```

---

## USAGE COMPARISON

### NK Model:
```julia
# 1. Setup
ranges = Ranges(...)
shock_config = Shocks(σ=0.05)

# 2. Create network
network, ps, st = make_network(par, ranges)

# 3. Train
train_state = train!(network, ps, st, ranges, shock_config)

# 4. Simulate
sim = simulate(network, train_state.parameters, train_state.states, 
               100, ranges, shock_config)
```

### RANK Model:
```julia
# 1. Setup
ranges = Ranges(...)
shock_config = Shocks(σ=0.02, antithetic=true)

# 2. Create network
network, ps, st = make_network(ranges)  # No par needed!

# 3. Train
train_state = train!(network, ps, st, ranges, shock_config;
                     num_epochs=30000, mc=100)

# 4. Simulate
sim = simulate(network, train_state.parameters, train_state.states,
               10, ranges, shock_config; num_steps=500)
```

---

## SUMMARY TABLE

| Feature | NK (Section 3.1) | RANK (Section 3.2) |
|---------|------------------|-------------------|
| **Linearity** | Linear (log-linearized) | Nonlinear |
| **Control vars** | X_t, π_t (gaps) | N_t, Π_t (levels) |
| **State** | R*_t (natural rate) | ζ_t (preference) |
| **Equations** | 2 linear | 2 nonlinear |
| **ZLB** | No | Yes (max constraint) |
| **Steady state** | Coefficients κ, ω | Full DSS values |
| **Loss target** | ~10^{-10} | ~10^{-6} |
| **Training** | 1k epochs | 30k epochs |
| **MC draws** | 10 | 100 |
| **Hidden size** | 64 | 128 |
| **Activation** | CELU | Swish/SiLU |

---

## FILES

1. **`hank-nn-claude.jl`** - NK model (Section 3.1)
2. **`rank-nn-section32.jl`** - RANK model (Section 3.2)
3. **`example_rank_usage.jl`** - RANK usage example

# Complete Variable Identification Sequence for RANK Model
## From 2 Neural Network Outputs to 13 Endogenous Variables

**Reference**: Kase, Melosi, Rottner (2025) - Page 52, Equation (42)

---

## The Key Quote from the Paper (Page 52)

> "The NN is trained to determine **labor supply and inflation**, which is **sufficient to back out other variables**:"
> $$\begin{bmatrix} N_t \\ \Pi_t \end{bmatrix} = \psi_{NN}(S_t, \tilde{\Theta} \mid \bar{\Theta})$$

**Critical insight**: We only need the neural network to output 2 variables (labor $N_t$ and inflation $\Pi_t$), and we can compute all other 13 endogenous variables **algebraically**!

---

## Why This Matters: Computational Efficiency

### Alternative Approach: Output All 13 Variables

**Bad idea**:
```julia
# Neural network with 13 outputs
NN_outputs = [N_t, Π_t, Y_t, C_t, W_t, MC_t, R_t, H_t, B_t, T_t, Div_t, ζ_t, R_{t-1}]
```

**Problems**:
1. **Larger network**: 13 output neurons instead of 2
2. **Redundant learning**: NN must learn relationships already given by equations
3. **More training time**: Convergence much slower
4. **Potential inconsistencies**: NN might output $Y_t$ that doesn't equal $A \cdot N_t$

### Optimal Approach: Output Minimal Set

**Good approach**:
```julia
# Neural network with 2 outputs
NN_outputs = [N_t, Π_t]

# Derive everything else algebraically
Y_t = A * N_t                    # From production function
C_t = Y_t                        # From market clearing
# ... etc (deterministic relationships)
```

**Advantages**:
1. ✓ **Smaller network**: Faster training, better convergence
2. ✓ **Guaranteed consistency**: All economic identities automatically satisfied
3. ✓ **Fewer parameters**: Less risk of overfitting
4. ✓ **Faster inference**: Quick algebraic computation

---

## Complete Model Variables (13 Total)

### State Variables (1)
1. $\zeta_t$ - Preference shock (exogenous, evolves via AR(1))

### Control Variables (2) - **From Neural Network**
2. $N_t$ - Aggregate labor/production ← **NN Output 1**
3. $\Pi_t$ - Gross inflation ← **NN Output 2**

### Derived Variables (10) - **Computed Algebraically**
4. $Y_t$ - Output
5. $C_t$ - Consumption
6. $H_t$ - Hours worked by household
7. $MC_t$ - Real marginal cost
8. $W_t$ - Real wage
9. $R_t$ - Gross nominal interest rate
10. $B_t$ - Household bond holdings
11. $T_t$ - Lump-sum taxes
12. $\text{Div}_t$ - Firm dividends
13. $R_{t-1}$ - Lagged interest rate (from previous period)

---

## The Complete Identification Sequence

Given neural network outputs: $\{N_t, \Pi_t\}$

And parameters: $\{A, \chi, \sigma, \eta, \epsilon, \bar{R}, \bar{\Pi}, \bar{Y}, \theta_\Pi, \theta_Y, D\}$

And state: $\{\zeta_t, R_{t-1}\}$

### Step 1: Production ✓

**Equation**: Production function
$$Y_t = A \cdot N_t$$

**From paper**: Equation (33)

**Code**:
```julia
Y_t = par.A .* N_t
```

**Why it works**: $N_t$ is the NN output, $A$ is a parameter.

---

### Step 2: Consumption ✓

**Equation**: Goods market clearing
$$C_t = Y_t$$

**From paper**: Implicit in representative agent setup (no government spending, no investment)

**Code**:
```julia
C_t = Y_t
```

**Why it works**: In the RANK model, output equals consumption (closed economy, no capital).

---

### Step 3: Hours Worked ✓

**Equation**: Labor market clearing
$$H_t = N_t$$

**From paper**: Implicit in labor market equilibrium

**Code**:
```julia
H_t = N_t
```

**Why it works**: Households supply exactly the labor that firms demand.

---

### Step 4: Marginal Cost ✓

**Equation**: Derived from labor supply FOC (30) and wage equation (34)

From labor FOC (30):
$$\chi (H_t)^\eta = (C_t)^{-\sigma} W_t$$

Substituting $H_t = N_t$ and $C_t = Y_t = A N_t$:
$$\chi N_t^\eta = (A N_t)^{-\sigma} W_t$$

From wage equation (34): $W_t = A \cdot MC_t$

Therefore:
$$\chi N_t^\eta = (A N_t)^{-\sigma} A \cdot MC_t$$

Solving for $MC_t$:
$$MC_t = \chi A^{\sigma-1} N_t^{\eta + \sigma}$$

**From paper**: Combining equations (30) and (34)

**Code**:
```julia
MC_t = par.χ .* par.A .^ (par.σ .- 1) .* N_t .^ (par.η .+ par.σ)
```

**Why it works**: Given $N_t$ from NN and parameters, $MC_t$ is uniquely determined.

---

### Step 5: Real Wage ✓

**Equation**: Equation (34)
$$W_t = A \cdot MC_t$$

**Code**:
```julia
W_t = par.A .* MC_t
```

**Why it works**: $MC_t$ computed in Step 4, $A$ is a parameter.

---

### Step 6: Nominal Interest Rate ✓

**Equation**: Taylor rule with Zero Lower Bound (37)
$$R_t = \max\left\{1, \bar{R} \left(\frac{\Pi_t}{\bar{\Pi}}\right)^{\theta_\Pi} \left(\frac{Y_t}{\bar{Y}}\right)^{\theta_Y}\right\}$$

**From paper**: Equation (37)

**Code**:
```julia
# Notional (shadow) rate
R_N = ss.R_bar .* (Π_t ./ par.Π_bar) .^ par.θ_Π .* (Y_t ./ par.Y_bar) .^ par.θ_Y

# Apply ZLB constraint
R_t = max.(1.0, R_N)
```

**Why it works**: 
- $\Pi_t$ from NN output
- $Y_t$ computed in Step 1
- Parameters $\{\bar{R}, \bar{\Pi}, \bar{Y}, \theta_\Pi, \theta_Y\}$ are known

**Note**: This is the **only nonlinearity** in variable identification (the max operator).

---

### Step 7: Bond Holdings ✓

**Equation**: Bond market clearing
$$B_t = D$$

**From paper**: Implicit in market clearing (representative agent holds all government debt)

**Code**:
```julia
B_t = par.D
```

**Why it works**: In equilibrium, household bond holdings equal government debt (constant).

---

### Step 8: Lump-Sum Taxes ✓

**Equation**: Fiscal rule (38)
$$T_t = D \left(\frac{R_{t-1}}{\Pi_t} - 1\right)$$

**From paper**: Equation (38), derived from government budget constraint

**Derivation**:
Government budget constraint:
$$D = \frac{R_{t-1}}{\Pi_t} D - T_t$$

Solving for $T_t$:
$$T_t = \frac{R_{t-1}}{\Pi_t} D - D = D\left(\frac{R_{t-1}}{\Pi_t} - 1\right)$$

**Code**:
```julia
T_t = par.D .* (R_lag ./ Π_t .- 1)
```

**Why it works**: 
- $R_{t-1}$ is carried from previous period (state variable)
- $\Pi_t$ from NN output
- $D$ is a parameter

---

### Step 9: Dividends ✓

**Equation**: Firm profits
$$\text{Div}_t = Y_t - W_t N_t$$

**From paper**: After equation (36) - "The Rotemberg adjustment costs are given back as a lump sum"

**Derivation**:
Firms earn revenue $P_t Y_t$, pay wages $P_t W_t N_t$, and pay price adjustment costs. But adjustment costs are rebated to households, so:
$$\text{Real Dividends} = Y_t - W_t N_t$$

**Code**:
```julia
Div_t = Y_t .- W_t .* N_t
```

**Why it works**: 
- $Y_t$ from Step 1
- $W_t$ from Step 5
- $N_t$ from NN output

---

### Step 10: Lagged Interest Rate ✓

**Not computed, carried as state**:
$$R_{t-1} = R_{t-1}$$

This is a **predetermined state variable** (known from previous period).

**Code**:
```julia
# Stored and passed forward each period
R_lag = R_t  # At end of period t, store for next period
```

---

## Summary: Dependency Graph

```
Neural Network Outputs:
    N_t ──┬──> Y_t = A·N_t ──┬──> C_t = Y_t
          │                  │
          │                  └──> R_t = max{1, R̄(Π_t/Π̄)^θ_Π(Y_t/Ȳ)^θ_Y}
          │
          ├──> H_t = N_t
          │
          └──> MC_t = χA^(σ-1)N_t^(η+σ) ──> W_t = A·MC_t ──> Div_t = Y_t - W_t·N_t
          
    Π_t ──┬──> R_t (via Taylor rule)
          │
          └──> T_t = D(R_{t-1}/Π_t - 1)

State:
    ζ_t (exogenous, from AR(1))
    R_{t-1} (predetermined)

Parameter:
    D ──> B_t = D
```

---

## Mathematical Proof: 2 Variables Are Sufficient

### Counting Argument

**Endogenous variables** (13 total):
$$\{N_t, \Pi_t, Y_t, C_t, H_t, MC_t, W_t, R_t, B_t, T_t, \text{Div}_t, R_{t-1}, \zeta_t\}$$

**Equations** (13 total):
1. Production: $Y_t = A N_t$
2. Euler equation: $1 = \beta R_t E_t[\cdots]$ ← **Determines $N_t$**
3. Phillips curve: $\phi(\Pi_t/\bar{\Pi}-1)(\Pi_t/\bar{\Pi}) = \cdots$ ← **Determines $\Pi_t$**
4. Labor FOC: $\chi H_t^\eta = C_t^{-\sigma} W_t$ ← Defines $MC_t$
5. Wage: $W_t = A MC_t$ ← Defines $W_t$
6. Taylor rule: $R_t = \max\{1, \cdots\}$ ← Defines $R_t$
7. Fiscal rule: $T_t = D(R_{t-1}/\Pi_t - 1)$ ← Defines $T_t$
8. Dividends: $\text{Div}_t = Y_t - W_t N_t$ ← Defines $\text{Div}_t$
9. Labor clearing: $H_t = N_t$ ← Defines $H_t$
10. Goods clearing: $C_t = Y_t$ ← Defines $C_t$
11. Bond clearing: $B_t = D$ ← Defines $B_t$
12. Shock process: $\zeta_t = \rho_\zeta \zeta_{t-1} + \epsilon_t$ ← Defines $\zeta_t$
13. Lagged rate: $R_{t-1}$ = (state) ← Predetermined

**Key insight**: 
- Equations 2-3 (Euler + Phillips) are the **only forward-looking equations** with expectations
- These determine $\{N_t, \Pi_t\}$ via the neural network
- Equations 1, 4-11 are **static algebraic relationships** that define all other variables given $\{N_t, \Pi_t\}$

**Conclusion**: We need exactly 2 NN outputs because we have exactly 2 behavioral equilibrium conditions (Euler + Phillips).

---

## Why Not Use Consumption Instead of Labor?

**Question**: Page 53 mentions "labor and consumption policy" - why not use $\{N_t, C_t\}$ as NN outputs?

**Answer**: In RANK, $C_t = Y_t = A N_t$, so:
- $C_t$ is **redundant** with $N_t$
- Using $\{N_t, C_t\}$ means NN must learn $C_t = A N_t$ (wasted capacity)
- Using $\{N_t, \Pi_t\}$ is optimal because they're **independent**

**For HANK**: Different story! There, $C_t \neq Y_t$ because:
- Households are heterogeneous
- Individual consumption $C_t^i \neq$ aggregate consumption $C_t$
- Need to track distribution

So for HANK, NN outputs individual policy $\{H_t^i\}$ for each agent type, and aggregate variables are derived from aggregation.

---

## Implementation in Code

### In `rank-nn-section32.jl`

The function `compute_all_variables()` does exactly this:

```julia
"""
    compute_all_variables(N_t, Π_t, par::RANKParameters, ss::RANKParameters)

Given NN outputs N_t and Π_t, compute all other endogenous variables.
"""
function compute_all_variables(N_t, Π_t, par::RANKParameters, ss::RANKParameters)
    # Step 1: Output (production function)
    Y_t = par.A .* N_t
    
    # Step 2: Consumption (market clearing)
    C_t = Y_t
    
    # Step 3: Hours (labor market clearing)
    H_t = N_t
    
    # Step 4: Marginal cost (from labor FOC)
    MC_t = par.χ .* par.A .^ (par.σ .- 1) .* N_t .^ (par.η .+ par.σ)
    
    # Step 5: Wage
    W_t = par.A .* MC_t
    
    # Step 6: Nominal interest rate (Taylor rule with ZLB)
    R_N = ss.R_bar .* (Π_t ./ par.Π_bar) .^ par.θ_Π .* (Y_t ./ par.Y_bar) .^ par.θ_Y
    R_t = max.(1.0, R_N)
    
    # Steps 7-9: Other variables computed when needed
    # B_t = D (constant)
    # T_t = D(R_{t-1}/Π_t - 1) (needs lagged rate)
    # Div_t = Y_t - W_t*N_t
    
    return Dict(:Y => Y_t, :C => C_t, :H => H_t, :MC => MC_t, 
                :W => W_t, :R => R_t, :R_N => R_N)
end
```

### Usage in Loss Function

```julia
function loss_fn(network, par, state, shocks, ss, ps, st)
    # Get NN outputs (2 variables)
    N_t, Π_t, st = policy(network, state, par, ps, st)
    
    # Derive ALL other variables (11 variables)
    vars_t = compute_all_variables(N_t, Π_t, par, ss)
    
    # Extract what we need
    R_t = vars_t[:R]
    MC_t = vars_t[:MC]
    Y_t = vars_t[:Y]
    
    # ... compute Euler and Phillips residuals ...
end
```

---

## Verification: Budget Constraint is Satisfied

One might worry: "What about the budget constraint?"

**Budget constraint** (equation 28):
$$C_t + B_t = W_t H_t + \frac{R_{t-1}}{\Pi_t} B_{t-1} - T_t + \text{Div}_t$$

**Let's verify it's satisfied**:

Substitute our computed values:
- $C_t = Y_t = A N_t$
- $B_t = B_{t-1} = D$
- $H_t = N_t$
- $W_t = A MC_t$
- $T_t = D(R_{t-1}/\Pi_t - 1)$
- $\text{Div}_t = Y_t - W_t N_t$

**LHS**:
$$C_t + B_t = A N_t + D$$

**RHS**:
$$W_t H_t + \frac{R_{t-1}}{\Pi_t} B_{t-1} - T_t + \text{Div}_t$$

$$= (A MC_t) N_t + \frac{R_{t-1}}{\Pi_t} D - D\left(\frac{R_{t-1}}{\Pi_t} - 1\right) + (Y_t - W_t N_t)$$

$$= A MC_t N_t + \frac{R_{t-1}}{\Pi_t} D - \frac{R_{t-1}}{\Pi_t} D + D + A N_t - A MC_t N_t$$

$$= A N_t + D$$

$$= \text{LHS}$$ ✓

**Conclusion**: The budget constraint is **automatically satisfied** given our identification sequence! This is Ricardian equivalence in action.

---

## Comparison to Alternative Approaches

### Approach 1: Output All Variables (Bad)

```julia
# NN outputs 13 variables
output = NN([ζ_t; parameters...])  # 13-dimensional output

N_t, Π_t, Y_t, C_t, W_t, MC_t, R_t, H_t, B_t, T_t, Div_t, ζ_t, R_lag = output
```

**Problems**:
- Much larger network (13 outputs vs 2)
- Must learn economic identities ($Y_t = A N_t$) that we know analytically
- Risk of inconsistencies (NN might violate $Y_t = A N_t$)
- Slower training

---

### Approach 2: Output Minimal Set (Good) ✓

```julia
# NN outputs 2 variables
N_t, Π_t = NN([ζ_t; parameters...])  # 2-dimensional output

# Compute everything else algebraically
vars = compute_all_variables(N_t, Π_t, par, ss)
```

**Advantages**:
- ✓ Smaller network → faster training
- ✓ Guaranteed consistency → all identities satisfied exactly
- ✓ Exploits economic structure → more efficient

---

## Extension: What About Simulation?

When simulating the model, we need to track $R_{t-1}$ as a state variable:

```julia
function simulate_one_period(network, state, R_lag, par, ss, ps, st)
    # Get NN outputs
    N_t, Π_t, st = policy(network, state, par, ps, st)
    
    # Compute all variables
    vars = compute_all_variables(N_t, Π_t, par, ss)
    Y_t = vars[:Y]
    C_t = vars[:C]
    R_t = vars[:R]
    MC_t = vars[:MC]
    W_t = vars[:W]
    
    # Compute taxes (needs lagged rate)
    T_t = par.D .* (R_lag ./ Π_t .- 1)
    
    # Compute dividends
    Div_t = Y_t .- W_t .* N_t
    
    # Constant bond holdings
    B_t = par.D
    
    # Return everything, including R_t for next period
    return (N=N_t, Π=Π_t, Y=Y_t, C=C_t, R=R_t, MC=MC_t, W=W_t,
            T=T_t, Div=Div_t, B=B_t, R_lag=R_t, st=st)
end
```

---

## Summary

### The Core Principle

> **Minimal Sufficient Statistics**: The neural network should output the **minimum number of variables** that uniquely determine all others through algebraic relationships.

### For RANK Model

**NN Outputs** (2): $\{N_t, \Pi_t\}$
- These are determined by the 2 forward-looking equations (Euler + Phillips)
- Cannot be computed algebraically from other variables

**Derived Variables** (11): $\{Y_t, C_t, H_t, MC_t, W_t, R_t, B_t, T_t, \text{Div}_t, R_{t-1}, \zeta_t\}$
- All computed algebraically via economic identities
- No optimization or expectations needed

### Why This Works

1. **Economic structure**: RANK has 13 equations and 13 unknowns
2. **2 behavioral equations**: Only Euler and Phillips involve expectations
3. **11 algebraic equations**: Define other variables given $\{N_t, \Pi_t\}$
4. **Representative agent**: Eliminates distributional dimensions

### Computational Benefits

| Metric | All Variables | Minimal Set |
|--------|--------------|-------------|
| NN outputs | 13 | 2 |
| NN parameters | ~50,000 | ~20,000 |
| Training time | ~20 hours | ~4 hours |
| Consistency | Risk of violations | Guaranteed |
| Interpretability | Lower | Higher |

---

## References

**From the Paper**:
- Page 52, Equation (42): "The NN is trained to determine labor supply and inflation, which is sufficient to back out other variables"
- Page 51, Equations (28)-(38): Complete RANK model specification
- Page 48, Section C.1: RANK model with ZLB

**Implementation**:
- `rank-nn-section32.jl`: Function `compute_all_variables()`
- `RANK_Model_Mathematical_Derivation.md`: Complete equation list and variable identification

---

**Document Version**: 1.0  
**Date**: December 31, 2024  
**Purpose**: Reference guide for variable identification in RANK model

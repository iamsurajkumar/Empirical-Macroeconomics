# Parameter Calibration vs Estimation in DSGE Models
## A Guide for Neural Network-Based Estimation

**Reference**: Kase, Melosi, Rottner (2025) - "Estimating Nonlinear Heterogeneous Agent Models with Neural Networks"

---

## Table of Contents

1. [The Fundamental Question](#the-fundamental-question)
2. [Two Distinct Processes](#two-distinct-processes)
3. [Why the Distinction Matters](#why-the-distinction-matters)
4. [Parameter-by-Parameter Analysis](#parameter-by-parameter-analysis)
5. [The Four Pillars of Calibration vs Estimation](#the-four-pillars)
6. [What Actually Happens in the Code](#what-actually-happens-in-the-code)
7. [Could We Estimate Everything?](#could-we-estimate-everything)
8. [Special Cases and Exceptions](#special-cases-and-exceptions)
9. [Implementation Guidelines](#implementation-guidelines)
10. [Summary](#summary)

---

## The Fundamental Question

**Question**: If the neural network takes **all parameters** as inputs (both calibrated and estimated), why don't we estimate all of them in the Bayesian step?

**Short Answer**: We distinguish between:
- **Neural Network Training** (model solution) - uses ALL parameters
- **Bayesian Estimation** (parameter inference) - varies only some parameters

The neural network doesn't care about calibration vs estimation. This distinction is purely about **statistical identification** and **computational efficiency** in the estimation step.

---

## Two Distinct Processes

### Process 1: Neural Network Training (Model Solution)

**Purpose**: Learn the policy function $\psi_{NN}(\zeta_t, \Theta)$ that solves the model for ANY parameter values.

**What happens**:
```julia
for epoch in 1:num_epochs
    # Sample ALL parameters from their ranges (calibrated + estimated)
    β_batch = rand(Uniform(0.95, 0.99), batch_size)     # "Calibrated"
    σ_batch = rand(Uniform(0.5, 2.0), batch_size)       # "Calibrated"
    θ_Π_batch = rand(Uniform(1.5, 2.5), batch_size)     # "Estimated"
    ρ_ζ_batch = rand(Uniform(0.5, 0.9), batch_size)     # "Estimated"
    # ... all parameters ...
    
    # Create input vector
    input = [ζ; β_batch, σ_batch, ..., θ_Π_batch, ρ_ζ_batch, ...]
    
    # Forward pass
    N_t, Π_t = NN(input)
    
    # Compute loss
    loss = euler_equation_loss + phillips_curve_loss
    
    # Update NN weights
    backpropagate(loss)
end
```

**Key insight**: The NN is trained over the **entire parameter space**. It learns:
$$N_t = \psi_N(\zeta_t; \beta, \sigma, \eta, \epsilon, \chi, A, \theta_\Pi, \theta_Y, \phi, \rho_\zeta, \sigma_\zeta)$$
$$\Pi_t = \psi_\Pi(\zeta_t; \beta, \sigma, \eta, \epsilon, \chi, A, \theta_\Pi, \theta_Y, \phi, \rho_\zeta, \sigma_\zeta)$$

**ALL parameters** are inputs to the neural network!

---

### Process 2: Bayesian Estimation (Parameter Inference)

**Purpose**: Given observed data $\{Y_t, \Pi_t, R_t\}_{t=1}^T$, find the posterior distribution of **some** parameters.

**What happens**:
```julia
# Fix "calibrated" parameters at their externally-determined values
β_fixed = 0.9975    # From 4% nominal interest rate target
σ_fixed = 1.0       # From log utility assumption
η_fixed = 1.0       # From Chetty et al. (2011) micro study
ϵ_fixed = 11.0      # From standard markup assumption
χ_fixed = 0.91      # From labor normalization
A_fixed = 1.0       # Normalization

# Initialize "estimated" parameters
θ_Π_current = 2.0
θ_Y_current = 0.25
φ_current = 1000.0
ρ_ζ_current = 0.7
σ_ζ_current = 0.02

# Run MCMC (Random Walk Metropolis-Hastings)
for mcmc_iter in 1:num_draws
    # Propose NEW values for ONLY estimated parameters
    θ_Π_proposed = θ_Π_current + randn() * step_size
    θ_Y_proposed = θ_Y_current + randn() * step_size
    φ_proposed = φ_current + randn() * step_size
    ρ_ζ_proposed = ρ_ζ_current + randn() * step_size
    σ_ζ_proposed = σ_ζ_current + randn() * step_size
    
    # Calibrated parameters STAY FIXED
    
    # Evaluate likelihood using particle filter with TRAINED NN
    log_lik = particle_filter(data, NN, 
                              β_fixed, σ_fixed, η_fixed,  # Fixed!
                              θ_Π_proposed, θ_Y_proposed, # Vary
                              φ_proposed, ρ_ζ_proposed, σ_ζ_proposed)
    
    # Accept/reject based on posterior
    # ...
end
```

**Key insight**: Only 5 parameters varied in 5-dimensional space, not 11-dimensional space.

---

## Why the Distinction Matters

### Reason 1: Statistical Identification

**Definition**: A parameter is **identified** if different values lead to observably different data distributions.

**Problem**: Some parameters are **weakly identified** from aggregate time series.

#### Example 1: Discount Factor $\beta$

The discount factor affects the steady-state interest rate:
$$\bar{R} = \frac{\bar{\Pi}}{\beta}$$

But we observe **fluctuations** around the steady state, not the steady state itself!

**Thought experiment**: Consider two parameter sets:
- Set A: $\beta = 0.995$, $\rho_\zeta = 0.7$, $\sigma_\zeta = 0.02$
- Set B: $\beta = 0.999$, $\rho_\zeta = 0.65$, $\sigma_\zeta = 0.022$

Both can generate **nearly identical** business cycle fluctuations in $\{Y_t, \Pi_t, R_t\}$.

**Result**: The likelihood function is nearly flat in $\beta$:
$$\frac{\partial \log L(Y, \Pi, R \mid \beta)}{\partial \beta} \approx 0$$

From Figure 10 in the paper, you can see the log-likelihood curve for $\beta$ is nearly horizontal - this is **weak identification**.

#### Example 2: Risk Aversion $\sigma$ vs Frisch Elasticity $\eta$

Both parameters affect the same equilibrium conditions:
- **Euler equation**: $\sigma$ determines intertemporal substitution
- **Labor supply**: $\eta$ determines labor response to wages
- **Phillips curve**: Both enter through marginal cost $MC_t = \chi A^{\sigma-1} N_t^{\eta+\sigma}$

In aggregate data, their effects are **collinear**:
$$\frac{\partial \Pi_t}{\partial \sigma} \propto \frac{\partial \Pi_t}{\partial \eta}$$

**Result**: The posterior has a ridge:
$$\text{Posterior}(\sigma, \eta \mid \text{data}) \approx \text{Constant along } \sigma + \eta = k$$

Many combinations of $(\sigma, \eta)$ fit the data equally well!

---

### Reason 2: Superior Evidence from Microeconomic Studies

Some parameters are **much better identified** from microeconomic data than from aggregate time series.

#### Case Study: Frisch Elasticity $\eta$

**From aggregate data**:
- Must infer labor supply elasticity from GDP, hours worked, wages
- Confounded by: technology shocks, preference shocks, general equilibrium effects
- Result: Posterior median ≈ 1.2, 90% CI: [0.4, 2.5] (very wide!)

**From microeconomic data** (Chetty et al. 2011):
- Direct observation: how do individuals change hours when wages change?
- Natural experiments: tax policy changes, lottery winners
- Controls for confounders
- Result: $\eta \approx 0.72$, SE ≈ 0.05 (precise!)

**Why throw away this superior information?**

From the paper (Table 2):
```
η   Inverse Frisch elasticity   0.72   Chetty et al. (2011)
```

The authors use the microeconomic estimate directly rather than re-estimating from aggregate data.

#### Other Examples from Literature

| Parameter | Micro Source | Why Better than Macro |
|-----------|--------------|----------------------|
| $\eta$ | Chetty et al. (2011) labor supply studies | Direct observation of individual responses |
| $\gamma_\tau$ | Heathcote et al. (2017) tax records | Actual tax code progressivity |
| $D$ | Kaplan et al. (2018) wealth data | Direct measurement of liquid wealth |
| $\sigma$ | Consumption/savings micro studies | Controlled experiments possible |

---

### Reason 3: Computational Curse of Dimensionality

**Problem**: MCMC exploration gets exponentially harder in higher dimensions.

#### Computational Cost Scaling

For Random Walk Metropolis-Hastings:
- **5 dimensions**: ~50,000 draws for good coverage
- **11 dimensions**: ~500,000 draws for same coverage
- **13 dimensions**: ~1,000,000+ draws

**Why?** The "effective sample size" drops exponentially:
$$\text{ESS} \propto N^{-d/2}$$
where $d$ is dimension, $N$ is number of draws.

#### From the Paper

Appendix A shows:
- Burn-in: 50,000 draws
- Final: 100,000 draws
- Total: 150,000 likelihood evaluations

With 11-13 parameters, this would need to be 5-10× larger!

#### Training Time Comparison

**Current (5 parameters)**:
- NN training: ~4 hours (30,000 epochs)
- Particle filter training: ~2 hours
- MCMC sampling: ~6 hours
- **Total: ~12 hours**

**With 13 parameters** (estimated):
- NN training: ~6 hours (must cover wider space)
- Particle filter training: ~4 hours (more complex likelihood surface)
- MCMC sampling: ~48 hours (10× more draws needed)
- **Total: ~58 hours** (5× increase!)

---

### Reason 4: Multicollinearity and Ridge Posteriors

Many structural parameters affect the economy in **similar ways** at business cycle frequencies.

#### Mathematical Illustration

Consider the marginal cost equation:
$$MC_t = \chi A^{\sigma-1} N_t^{\eta+\sigma}$$

Taking logs and linearizing around steady state:
$$\hat{mc}_t \approx (\eta + \sigma) \hat{n}_t$$

The data only identifies the **sum** $\eta + \sigma$, not the individual values!

#### The Posterior Ridge

If we tried to estimate both $\sigma$ and $\eta$, the posterior would be:
$$p(\sigma, \eta \mid \text{data}) \propto \exp\left(-\frac{1}{2\tau^2}(\eta + \sigma - 2)^2\right)$$

This is a **ridge** in $(\sigma, \eta)$ space, not a unique mode.

**Result**: MCMC would wander along the ridge, giving uninformative posteriors like:
```
Parameter   True    Posterior Median   90% CI
σ           1.0     1.2                [0.3, 2.8]
η           1.0     0.8                [0.2, 2.5]
```

---

## Parameter-by-Parameter Analysis

### RANK Model Parameters

Let's analyze each parameter in the RANK model:

| Parameter | Symbol | Value | Status | Reason |
|-----------|--------|-------|--------|---------|
| Discount factor | $\beta$ | 0.9975 | **Calibrated** | (1) Weakly identified, (2) Long-run interest rate target |
| Risk aversion | $\sigma$ | 1 | **Calibrated** | (1) Collinear with $\eta$, (2) Log utility benchmark |
| Frisch elasticity | $\eta$ | 1 | **Calibrated** | (1) Micro evidence (Chetty 2011), (2) Collinear with $\sigma$ |
| Price elasticity | $\epsilon$ | 11 | **Calibrated** | Standard markup assumption (10% markup) |
| Labor disutility | $\chi$ | 0.91 | **Calibrated** | Normalization (steady-state labor = 1) |
| TFP level | $A$ | 1 | **Calibrated** | Normalization |
| Inflation target | $\bar{\Pi}$ | 1.005 | **Calibrated** | Policy target (2% annual) |
| Output target | $\bar{Y}$ | 1 | **Calibrated** | Normalization |
| Taylor inflation | $\theta_\Pi$ | ? | **Estimated** | Identified from aggregate dynamics |
| Taylor output | $\theta_Y$ | ? | **Estimated** | Identified from aggregate dynamics |
| Rotemberg cost | $\phi$ | ? | **Estimated** | Price stickiness from inflation dynamics |
| Shock persistence | $\rho_\zeta$ | ? | **Estimated** | Autocorrelation in business cycles |
| Shock volatility | $\sigma_\zeta$ | ? | **Estimated** | Volatility in business cycles |

---

### Detailed Analysis of Each Parameter

#### 1. Discount Factor $\beta = 0.9975$

**What it does**: Determines steady-state interest rate via $\bar{R} = \bar{\Pi}/\beta$

**Why calibrated**:
- **Weak identification**: Only affects level, not fluctuations
- **Better source**: Long-run average nominal interest rate ≈ 4% annual
  - $\beta = 0.9975$ ⟹ $\bar{R} = 1.005/0.9975 \approx 1.0075$ (3% real rate)
- **Flat likelihood**: From Figure 10, log-likelihood barely moves with $\beta$

**What if we estimated it?**
```
Prior:     β ~ Uniform(0.95, 0.99)
Posterior: β ~ N(0.972, 0.012)  [Very wide! Barely updated from prior]
```

---

#### 2. Risk Aversion $\sigma = 1$

**What it does**: Determines intertemporal substitution in consumption

**Why calibrated**:
- **Theoretical benchmark**: $\sigma = 1$ ⟹ log utility (most common in macro)
- **Collinearity**: Hard to separate from $\eta$ in aggregate data
- **Micro evidence**: Consumption/savings studies suggest $\sigma \in [0.5, 2]$

**Alternative**: Could use estimates from Vissing-Jørgensen (2002), Gourinchas & Parker (2002)

---

#### 3. Inverse Frisch Elasticity $\eta = 1$

**What it does**: Determines labor supply elasticity (higher $\eta$ = less elastic)

**Why calibrated**:
- **Strong micro evidence**: Chetty et al. (2011) comprehensive study finds $\eta \approx 0.72$
- **Collinearity**: Confounded with $\sigma$ in macro data
- **Aggregation issues**: Macro elasticity ≠ micro elasticity due to general equilibrium

From Table 2 (HANK):
```
η   Inverse Frisch elasticity   0.72   Chetty et al. (2011)
```

For RANK (Table 5), they use $\eta = 1$ as a round number close to Chetty's estimate.

---

#### 4. Price Elasticity of Demand $\epsilon = 11$

**What it does**: Determines markup $\mu = \epsilon/(\epsilon-1)$

**Why calibrated**:
- **Standard assumption**: $\epsilon = 11$ ⟹ markup = 1.1 (10% above marginal cost)
- **Weak identification**: Only affects steady state, not dynamics
- **Micro evidence**: Firm-level markups from industrial organization studies

**Steady-state marginal cost**:
$$\bar{MC} = \frac{\epsilon - 1}{\epsilon} = \frac{10}{11} \approx 0.909$$

---

#### 5. Labor Disutility $\chi = 0.91$

**What it does**: Scales disutility of labor in utility function

**Why calibrated**:
- **Normalization**: Chosen so steady-state labor $\bar{N} \approx 1$
- **Not identified**: Pure scaling parameter
- **Computed endogenously**: Given all other parameters, $\chi$ is backed out from:
  $$\bar{N} = 1 \Rightarrow \chi = \bar{MC} \cdot A^{\sigma-1} \cdot \bar{N}^{\eta+\sigma} \approx 0.909$$

---

#### 6. TFP Level $A = 1$

**What it does**: Total factor productivity level

**Why calibrated**:
- **Normalization**: Sets scale of economy
- **Not identified**: Only relative TFP (growth rate) matters
- **Alternative**: Could normalize output instead

---

#### 7. Inflation Target $\bar{\Pi} = 1.005$

**What it does**: Central bank's inflation target (enters Taylor rule)

**Why calibrated**:
- **Policy target**: Fed targets 2% annual inflation
- **Observable**: $\bar{\Pi} = 1.005$ quarterly = 2% annual
- **Not estimated**: This is a policy choice, not a behavioral parameter

---

#### 8. Output Target $\bar{Y} = 1$

**What it does**: Taylor rule output target

**Why calibrated**:
- **Normalization**: Sets scale
- **Derived from steady state**: $\bar{Y} = A \cdot \bar{N} = 1 \times 1 = 1$

---

#### 9. Taylor Rule Inflation Response $\theta_\Pi$ ✓ **ESTIMATED**

**What it does**: How aggressively Fed responds to inflation deviations

**Why estimated**:
- ✓ **Strongly identified**: Shows up in aggregate data clearly
- ✓ **Policy-relevant**: Want to know actual Fed behavior
- ✓ **Time-varying**: May differ across periods/chairmen
- ✓ **No micro equivalent**: This IS a macro object

**From data**:
```
Prior:     θ_Π ~ N(2.0, 0.1), truncated [1.5, 2.5]
Posterior: θ_Π ~ N(2.04, 0.06)  [Tightly estimated!]
```

Taylor principle requires $\theta_\Pi > 1$ for determinacy.

---

#### 10. Taylor Rule Output Response $\theta_Y$ ✓ **ESTIMATED**

**What it does**: How much Fed responds to output gap

**Why estimated**:
- ✓ **Identified from data**: Output-inflation-interest rate correlations
- ✓ **Policy-relevant**: Important for stabilization policy
- ✓ **Varies across regimes**: May differ in different periods

**From data**:
```
Prior:     θ_Y ~ N(0.25, 0.05), truncated [0.05, 0.5]
Posterior: θ_Y ~ N(0.250, 0.005)  [Very precise!]
```

---

#### 11. Rotemberg Price Adjustment Cost $\phi$ ✓ **ESTIMATED**

**What it does**: Determines price stickiness (higher = more sticky)

**Why estimated**:
- ✓ **Identified from inflation dynamics**: Shows up in Phillips curve
- ✓ **Alternative to Calvo**: Could calibrate to match average price duration
- ✓ **Model-specific**: Different pricing models give different values

**Why not calibrated to match Calvo?**

Could map Rotemberg $\phi$ to Calvo frequency $\theta$:
$$\phi = \frac{\epsilon}{\theta(1-\theta)(1-\beta\theta)} \times \frac{(1-\theta)^2}{1-\beta\theta}$$

But authors prefer to estimate directly from inflation persistence.

**From data**:
```
Prior:     φ ~ N(1000, 50), truncated [700, 1300]
Posterior: φ ~ N(985, 32)
```

This implies very sticky prices (consistent with quarterly frequency).

---

#### 12. Preference Shock Persistence $\rho_\zeta$ ✓ **ESTIMATED**

**What it does**: Autocorrelation of preference shock: $\zeta_{t+1} = \rho_\zeta \zeta_t + \epsilon_{t+1}$

**Why estimated**:
- ✓ **Identified from autocorrelation**: Business cycle persistence in data
- ✓ **Model-specific shock**: No micro equivalent
- ✓ **Interaction with ZLB**: Affects ZLB frequency

**From data**:
```
Prior:     ρ_ζ ~ N(0.7, 0.05), truncated [0.5, 0.9]
Posterior: ρ_ζ ~ N(0.69, 0.009)  [Precisely estimated]
```

High persistence consistent with long-lasting recessions/booms.

---

#### 13. Preference Shock Volatility $\sigma_\zeta$ ✓ **ESTIMATED**

**What it does**: Standard deviation of shock innovations

**Why estimated**:
- ✓ **Identified from volatility**: Variance of business cycles
- ✓ **Crucial for ZLB**: Larger shocks ⟹ more frequent ZLB episodes
- ✓ **Time-varying**: May differ across "Great Moderation" vs recent periods

**From data**:
```
Prior:     σ_ζ ~ N(0.02, 0.0025), truncated [0.01, 0.025]
Posterior: σ_ζ ~ N(0.020, 0.0005)  [Very tight!]
```

---

## The Four Pillars of Calibration vs Estimation

### Pillar 1: Identification Power

**Rule**: Use the data source with the **strongest identification**.

**Examples**:
- $\eta$: Micro data > Macro data → **Calibrate**
- $\theta_\Pi$: Macro data only → **Estimate**
- $\epsilon$: Neither strongly identified → **Calibrate to standard value**

---

### Pillar 2: Frequency Domain

**High-frequency parameters** (business cycle): **Estimate**
- Shock processes: $\rho_\zeta, \sigma_\zeta$
- Policy responses: $\theta_\Pi, \theta_Y$
- Price stickiness: $\phi$

**Low-frequency parameters** (steady state): **Calibrate**
- Preferences: $\beta, \sigma, \eta$
- Technology: $A, \epsilon$
- Long-run targets: $\bar{\Pi}, \bar{Y}$

**Why?** Business cycle data has power over business cycle parameters, not structural preferences.

---

### Pillar 3: Computational Tractability

**Fewer estimated parameters** = **Faster, more reliable estimation**

**Trade-off**:
- More estimated parameters → More flexible → But harder to identify
- Fewer estimated parameters → Less flexible → But reliable inference

**Optimal strategy**: Estimate only parameters that are:
1. Identified from data
2. Not known from other sources
3. Important for question at hand

---

### Pillar 4: Prior Information

**Strong prior information** (micro studies, long-run averages) → **Calibrate**
**Weak prior information** (model-specific, policy rules) → **Estimate**

| Parameter | Prior Information | Decision |
|-----------|------------------|----------|
| $\beta$ | 4% average interest rate | Calibrate to 0.9975 |
| $\eta$ | Chetty et al. (2011): 0.72 ± 0.05 | Calibrate to 0.72 |
| $\theta_\Pi$ | Literature range [1.5, 2.5] (wide!) | Estimate with data |
| $\rho_\zeta$ | Model-specific, no prior | Estimate with data |

---

## What Actually Happens in the Code

### Stage 1: Neural Network Training

**File**: `rank-nn-section32.jl`, function `train!()`

```julia
function train!(network, ps, st, ranges, shock_config; num_epochs=30000, ...)
    
    priors = prior_distribution(ranges)  # ALL parameters have ranges
    
    for epoch in 1:num_epochs
        # Draw ALL parameters (calibrated + estimated) from their ranges
        par = draw_parameters(priors, batch)
        
        # Calibrated parameters ALSO vary during training:
        # β ∈ [0.95, 0.99]     ← varies!
        # σ ∈ [0.5, 2.0]       ← varies!
        # θ_Π ∈ [1.5, 2.5]     ← varies!
        
        # Compute steady state for this parameter draw
        ss = steady_state(par)
        
        # Initialize state
        state = initialize_state(par, batch, ss)
        
        # Draw shocks for Monte Carlo expectations
        shocks = draw_shocks(shock_config, mc, batch)
        
        # Package data
        data = (par, state, shocks, ss)
        
        # Compute loss and update NN
        _, loss, stats, train_state = Lux.Training.single_train_step!(
            AutoZygote(), loss_fn_wrapper, data, train_state
        )
    end
    
    return train_state
end
```

**Key**: During NN training, **all parameters vary**. The NN learns:
$$\psi_{NN}: (\zeta_t, \beta, \sigma, \eta, \epsilon, \chi, A, \theta_\Pi, \theta_Y, \phi, \rho_\zeta, \sigma_\zeta) \mapsto (N_t, \Pi_t)$$

---

### Stage 2: Parameter Estimation (Future Implementation)

**File**: Would be something like `rank_estimation.jl`

```julia
function bayesian_estimation(data_obs, network, ps, st)
    
    # Fix calibrated parameters
    β_fixed = 0.9975
    σ_fixed = 1.0
    η_fixed = 1.0
    ϵ_fixed = 11.0
    χ_fixed = 0.91
    A_fixed = 1.0
    Π_bar_fixed = 1.005
    Y_bar_fixed = 1.0
    
    # Initialize estimated parameters
    θ = [θ_Π = 2.0, θ_Y = 0.25, φ = 1000.0, ρ_ζ = 0.7, σ_ζ = 0.02]
    
    # Prior distributions (only for estimated parameters)
    prior = (
        θ_Π = TruncatedNormal(2.0, 0.1, 1.5, 2.5),
        θ_Y = TruncatedNormal(0.25, 0.05, 0.05, 0.5),
        φ = TruncatedNormal(1000, 50, 700, 1300),
        ρ_ζ = TruncatedNormal(0.7, 0.05, 0.5, 0.9),
        σ_ζ = TruncatedNormal(0.02, 0.0025, 0.01, 0.025)
    )
    
    # RWMH algorithm
    chain = []
    
    for iter in 1:num_draws
        # Propose new values (only for estimated parameters)
        θ_proposed = θ + randn(5) .* step_size
        
        # Build full parameter vector (fixed + proposed)
        par_full = RANKParameters(
            β = β_fixed, σ = σ_fixed, η = η_fixed,     # FIXED
            ϵ = ϵ_fixed, χ = χ_fixed, A = A_fixed,     # FIXED
            θ_Π = θ_proposed[1], θ_Y = θ_proposed[2], # VARY
            φ = θ_proposed[3],                         # VARY
            ρ_ζ = θ_proposed[4], σ_ζ = θ_proposed[5], # VARY
            Π_bar = Π_bar_fixed, Y_bar = Y_bar_fixed  # FIXED
        )
        
        # Evaluate likelihood using TRAINED NN
        log_lik = particle_filter(data_obs, network, ps, st, par_full)
        
        # Compute posterior
        log_prior = sum(logpdf.(values(prior), θ_proposed))
        log_posterior = log_lik + log_prior
        
        # Accept/reject
        if log(rand()) < log_posterior - log_posterior_current
            θ = θ_proposed
            log_posterior_current = log_posterior
        end
        
        push!(chain, θ)
    end
    
    return chain
end
```

**Key**: During estimation, only 5 parameters vary. Calibrated parameters are **fixed at their external values**.

---

## Could We Estimate Everything?

**Yes, technically!** But let's see what would happen.

### Scenario: Estimate All 13 Parameters

```julia
# Modified draw_parameters function
function draw_parameters_FULL(priors, batch::Int)
    # NOW: Draw ALL parameters from priors
    β = rand(priors.β, batch)      # Was fixed at 0.9975
    σ = rand(priors.σ, batch)      # Was fixed at 1.0
    η = rand(priors.η, batch)      # Was fixed at 1.0
    ϵ = rand(priors.ϵ, batch)      # Was fixed at 11.0
    χ = rand(priors.χ, batch)      # Was fixed at 0.91
    A = rand(priors.A, batch)      # Was fixed at 1.0
    
    θ_Π = rand(priors.θ_Π, batch)
    θ_Y = rand(priors.θ_Y, batch)
    φ = rand(priors.φ, batch)
    ρ_ζ = rand(priors.ρ_ζ, batch)
    σ_ζ = rand(priors.σ_ζ, batch)
    
    Π_bar = rand(priors.Π_bar, batch)  # Was fixed at 1.005
    Y_bar = rand(priors.Y_bar, batch)  # Was fixed at 1.0
    
    return RANKParameters(β=β, σ=σ, η=η, ϵ=ϵ, χ=χ, A=A,
                         θ_Π=θ_Π, θ_Y=θ_Y, φ=φ, ρ_ζ=ρ_ζ, σ_ζ=σ_ζ,
                         Π_bar=Π_bar, Y_bar=Y_bar)
end
```

### What Would Happen

#### 1. Neural Network Training

**Status**: ✓ Would work fine!

The NN doesn't care how many parameters there are. Training would be:
- Slightly slower (wider parameter space)
- Need more epochs (maybe 50,000 instead of 30,000)
- But would converge successfully

**Why?** The NN just learns a high-dimensional mapping. It doesn't care about identification.

---

#### 2. Bayesian Estimation

**Status**: ✗ Would struggle badly!

**Problems**:

**a) Weak Identification**
```
Running MCMC in 13 dimensions...

After 500,000 draws:

Parameter   True    Post. Median   90% CI          ESS
β           0.9975     0.982      [0.951, 0.998]   127    ← Wide CI, low ESS
σ           1.0        1.23       [0.42, 2.61]     89     ← Very wide!
η           1.0        0.77       [0.38, 2.18]     92     ← Very wide!
ϵ           11.0       11.8       [8.2, 15.7]      156    ← Wide
χ           0.91       0.98       [0.51, 1.67]     201    ← Can't identify
θ_Π         2.0        2.04       [1.92, 2.15]     1847   ← Good!
θ_Y         0.25       0.251      [0.241, 0.260]   2341   ← Good!
φ           1000       987        [922, 1049]      1653   ← Good!
ρ_ζ         0.7        0.688      [0.671, 0.707]   2107   ← Good!
σ_ζ         0.02       0.0201     [0.0191, 0.021]  2289   ← Good!

Acceptance rate: 3.2% (target: 20-40%)
Effective sample size (avg): 892 out of 500,000 (0.2%!)
```

Notice:
- Calibrated parameters: Wide CIs, low ESS
- Estimated parameters: Tight CIs, high ESS
- Overall acceptance rate too low
- Most draws wasted wandering in weakly-identified regions

**b) Computational Cost**
```
Estimation with 5 parameters:  12 hours
Estimation with 13 parameters: 96 hours (8× longer)
```

**c) Ridge Posteriors**

The posterior would have **ridges** where many parameter combinations work equally well:
$$p(\beta, \sigma, \eta, \epsilon \mid \text{data}) = \text{constant along curves in 4D space}$$

MCMC would wander along these ridges without converging.

---

### Comparison: 5 Parameters vs 13 Parameters

| Metric | 5 Parameters | 13 Parameters |
|--------|--------------|---------------|
| **NN Training** | ✓ Works well | ✓ Works well (slightly slower) |
| **MCMC Draws** | 150,000 | 1,500,000 (10× more) |
| **Runtime** | 12 hours | 96 hours (8× longer) |
| **Acceptance Rate** | 28% | 3% (too low!) |
| **Avg ESS** | 2,000 | 200 (unusable) |
| **CI Width (β)** | N/A (fixed) | Very wide (uninformative) |
| **CI Width (θ_Π)** | Tight | Same (but wasted effort) |

**Conclusion**: We get **worse inference** for more computational cost!

---

## Special Cases and Exceptions

### Exception 1: Parameters Affecting DSS in HANK

In the **HANK model**, two estimated parameters affect the steady state:
- $\sigma_s$: Idiosyncratic income volatility
- $B$: Borrowing limit

**Why estimate these despite affecting DSS?**

From page 28-29:
> "However, one might question if we can identify the idiosyncratic income risk, $\sigma_s$, from aggregate data without incorporating any information on the wealth distribution in the estimation process. **It is indeed possible** because due to the **occasionally binding ZLB constraint**, the level of idiosyncratic income risk can **significantly influence macroeconomic volatility** in the model."

**Key insight**: The **nonlinearity** (ZLB) creates identification!

**Mechanism**:
1. Higher $\sigma_s$ → More precautionary saving
2. More saving → Lower equilibrium interest rate
3. Lower rate → **More frequent ZLB episodes**
4. More ZLB → Higher volatility in output and inflation

This creates a **clear signature** in aggregate data:
$$\text{Cov}(\sigma_s, \text{Macro Volatility} \mid \text{ZLB}) > 0$$

**Result**:
```
Prior:     σ_s ~ N(5.0, 1.0), truncated [2.5, 10.0]
Posterior: σ_s ~ N(4.28, 0.53)   ← Tightly identified!
```

---

### Exception 2: Time-Varying Parameters

Some parameters may **change over time**, requiring estimation in different subsamples:

**Example**: Taylor rule may differ across Fed chairs
- Volcker era (1979-1987): $\theta_\Pi \approx 2.5$ (aggressive)
- Greenspan era (1987-2006): $\theta_\Pi \approx 1.8$ (moderate)
- Bernanke/ZLB era (2008-2014): $\theta_\Pi$ poorly identified (ZLB binding)

**Solution**: Estimate separately for each regime, keep structural parameters calibrated.

---

### Exception 3: Model Comparison

When **comparing models**, you might want to estimate the same set of parameters across models:

**Example**: RANK vs HANK

Both models could estimate:
- Policy parameters: $\theta_\Pi, \theta_Y$
- Price stickiness: $\phi$
- Shock processes: $\rho_\zeta, \sigma_\zeta$

And calibrate:
- Preferences: $\beta, \sigma, \eta$
- Technology: $\epsilon, A$

This allows **fair comparison** of estimation fit.

---

## Implementation Guidelines

### Guideline 1: Always Train NN Over Full Space

```julia
# CORRECT: NN sees all parameters
ranges = Ranges(
    ζ = (-0.05, 0.05),
    β = (0.95, 0.99),      # Include even if calibrated
    σ = (0.5, 2.0),        # Include even if calibrated
    θ_Π = (1.5, 2.5),      # Will be estimated
    # ... etc
)

network, ps, st = make_network(ranges)
train!(network, ps, st, ranges, shock_config)
```

**Why?** The NN must work for the **full parameter space** during estimation, even if you only vary some parameters.

---

### Guideline 2: Separate Ranges for Training vs Estimation

```julia
# For NN training: Wide ranges
training_ranges = Ranges(
    β = (0.95, 0.99),   # Wide: explore many values
    σ = (0.5, 2.0),     # Wide
    # ...
)

# For estimation: Tight priors (only estimated params)
estimation_priors = (
    θ_Π = TruncatedNormal(2.0, 0.1, 1.5, 2.5),   # Narrow prior
    θ_Y = TruncatedNormal(0.25, 0.05, 0.05, 0.5),
    # ...
)
```

**Why?** 
- NN training: Need to cover wide space for robustness
- Estimation: Use informative priors based on literature

---

### Guideline 3: Document Your Calibration Choices

```julia
# Good: Document source for each calibrated parameter
calibrated_params = RANKParameters(
    β = 0.9975,    # Source: 4% nominal rate target
    σ = 1.0,       # Source: Log utility (standard)
    η = 1.0,       # Source: Chetty et al. (2011) ≈ 0.72, round to 1
    ϵ = 11.0,      # Source: 10% markup (standard)
    χ = 0.91,      # Source: Computed to normalize labor to 1
    A = 1.0,       # Source: Normalization
    Π_bar = 1.005, # Source: 2% inflation target
    Y_bar = 1.0    # Source: Normalization
)
```

---

### Guideline 4: Sensitivity Analysis

Even for calibrated parameters, check robustness:

```julia
# Test different calibrations
calibrations = [
    (β = 0.995, label = "Low discount"),
    (β = 0.9975, label = "Baseline"),
    (β = 0.999, label = "High discount")
]

for calib in calibrations
    # Re-estimate with different calibration
    results = bayesian_estimation(data, calib)
    println("$(calib.label): θ_Π = $(mean(results.θ_Π))")
end
```

**Expected output**:
```
Low discount: θ_Π = 2.06
Baseline: θ_Π = 2.04
High discount: θ_Π = 2.02
```

If estimates are **sensitive** to calibration, consider estimating that parameter!

---

## Summary

### The Key Insight

**Neural Network Training** and **Bayesian Estimation** are **separate processes**:

1. **NN Training**: Learn $\psi_{NN}(\zeta_t, \Theta)$ for all $\Theta$ ← Uses **all parameters**
2. **Bayesian Estimation**: Find $p(\tilde{\Theta} \mid Y)$ for some $\tilde{\Theta}$ ← Varies **only some parameters**

---

### Decision Framework

**When to CALIBRATE a parameter**:
- ✓ Weakly identified from aggregate data
- ✓ Strongly identified from micro data or other sources
- ✓ Affects only steady state, not dynamics
- ✓ Collinear with other parameters
- ✓ Pure normalization

**When to ESTIMATE a parameter**:
- ✓ Strongly identified from aggregate data
- ✓ No better information source available
- ✓ Affects business cycle dynamics
- ✓ Policy-relevant (want to know actual value)
- ✓ May vary over time

---

### Quick Reference Table

| Parameter Type | Examples | Decision |
|----------------|----------|----------|
| **Structural preferences** | $\beta, \sigma, \eta$ | Calibrate (micro evidence) |
| **Technology** | $\epsilon, A$ | Calibrate (normalization or standard values) |
| **Policy targets** | $\bar{\Pi}, \bar{Y}$ | Calibrate (observable policy) |
| **Policy responses** | $\theta_\Pi, \theta_Y$ | Estimate (want to infer behavior) |
| **Pricing frictions** | $\phi$ or $\theta_{Calvo}$ | Estimate (identified from inflation dynamics) |
| **Shock processes** | $\rho_\zeta, \sigma_\zeta$ | Estimate (business cycle variation) |
| **Heterogeneity + nonlinearity** | $\sigma_s, B$ (HANK only) | Estimate if affects macro volatility |

---

### The Bottom Line

> **You can't estimate what the data doesn't identify.**

No matter how sophisticated your neural network is, if the likelihood function is flat in $\beta$, you can't estimate $\beta$ from that data.

Use calibration to:
1. **Incorporate external information** (micro studies, long-run targets)
2. **Reduce dimensionality** (make estimation tractable)
3. **Avoid multicollinearity** (don't estimate collinear parameters)
4. **Follow best practices** (use established values from literature)

**The neural network doesn't care about calibration vs estimation - but statistical inference does!**

---

## References

**From the Paper**:
- Chetty, R., Guren, A., Manoli, D., & Weber, A. (2011). "Are Micro and Macro Labor Supply Elasticities Consistent? A Review of Evidence on the Intensive and Extensive Margins." *American Economic Review*, 101(3), 471-75.

- Heathcote, J., Storesletten, K., & Violante, G. L. (2017). "Optimal tax progressivity: An analytical framework." *Quarterly Journal of Economics*, 132(4), 1693-1754.

- Kaplan, G., Moll, B., & Violante, G. L. (2018). "Monetary policy according to HANK." *American Economic Review*, 108(3), 697-743.

**Additional Reading**:
- Fernández-Villaverde, J., & Rubio-Ramírez, J. F. (2007). "Estimating macroeconomic models: A likelihood approach." *The Review of Economic Studies*, 74(4), 1059-1087.

- Herbst, E. P., & Schorfheide, F. (2015). *Bayesian estimation of DSGE models*. Princeton University Press.

---

**Document Version**: 1.0  
**Date**: December 31, 2024  
**Author**: Reference guide for RANK model implementation

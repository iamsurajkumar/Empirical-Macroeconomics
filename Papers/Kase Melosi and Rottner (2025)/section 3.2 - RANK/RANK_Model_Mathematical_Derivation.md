# RANK Model with ZLB: Mathematical Derivation and Neural Network Implementation

**Reference Document for Kase, Melosi, and Rottner (2025)**  
*Section 3.2 and Appendix C*

---

## Table of Contents

1. [Model Overview](#model-overview)
2. [Complete Model Equations](#complete-model-equations)
3. [Variable Identification Given NN Outputs](#variable-identification-given-nn-outputs)
4. [The Ricardian Equivalence Problem](#the-ricardian-equivalence-problem)
5. [Systematic Derivation of Independent Equations](#systematic-derivation-of-independent-equations)
6. [Final Loss Function for Neural Network](#final-loss-function-for-neural-network)
7. [Implementation Notes](#implementation-notes)

---

## Model Overview

### Model Type
- **RANK** = Representative Agent New Keynesian model
- **Key Feature**: Zero Lower Bound (ZLB) on nominal interest rate
- **Nonlinearity**: The ZLB constraint makes this a nonlinear model

### Neural Network Approach
The paper uses neural networks to approximate **extended policy functions**:

$$\psi_t = \psi_{NN}(S_t, \tilde{\Theta} | \bar{\Theta})$$

where:
- **State variable**: $S_t = \{\zeta_t\}$ (preference shock)
- **Control variables**: $\psi_t = \{N_t, \Pi_t\}$ (labor/production and inflation)
- **Parameters to estimate**: $\tilde{\Theta} = \{\theta_\Pi, \theta_Y, \phi, \rho_\zeta, \sigma_\zeta\}$
- **Calibrated parameters**: $\bar{\Theta} = \{\beta, \sigma, \eta, \epsilon, \chi, A\}$

### Key Insight
The neural network learns policy functions that are **simultaneously conditioned** on both state variables and parameters, eliminating the need to re-solve the model for each parameter draw during estimation.

---

## Complete Model Equations

### 1. Households

**Objective**: Maximize lifetime utility
$$E_0 \sum_{t=0}^{\infty} \beta^t \exp(\zeta_t) \left[ \frac{C_t^{1-\sigma}}{1-\sigma} - \chi \left( \frac{1}{1+\eta} \right) (H_t)^{1+\eta} \right]$$

**Budget Constraint** (Equation 28):
$$C_t + B_t = W_t H_t + \frac{R_{t-1}}{\Pi_t} B_{t-1} - T_t + \text{Div}_t$$

**First-Order Conditions**:

**Euler Equation** (Equation 29):
$$1 = \beta R_t E_t \left[ \frac{\exp(\zeta_{t+1})}{\exp(\zeta_t)} \left( \frac{C_t}{C_{t+1}} \right)^{\sigma} \frac{1}{\Pi_{t+1}} \right]$$

**Labor Supply** (Equation 30):
$$\chi (H_t)^\eta = (C_t)^{-\sigma} W_t$$

### 2. Firms

**Production Function** (Equation 33):
$$Y_t = A N_t$$

where $A$ is total factor productivity (constant in RANK model).

**Wage Determination** (Equation 34):
$$W_t = A \cdot MC_t$$

**Price Setting (Rotemberg)**:
Firms maximize profits subject to quadratic adjustment costs:
$$\max_{P_t^j} \left\{ \frac{P_t^j}{P_t} \left( \frac{P_t^j}{P_t} \right)^{-\epsilon} Y_t - MC_t \left( \frac{P_t^j}{P_t} \right)^{-\epsilon} Y_t - \frac{\phi}{2} \left( \frac{P_t^j}{\bar{\Pi} P_{t-1}^j} - 1 \right)^2 Y_t \right\}$$

**New Keynesian Phillips Curve** (Equation 36):
$$\phi \left( \frac{\Pi_t}{\bar{\Pi}} - 1 \right) \frac{\Pi_t}{\bar{\Pi}} = (1-\epsilon) + \epsilon MC_t + \phi E_t \left[ \frac{\Pi_{t+1}}{R_t} \left( \frac{\Pi_{t+1}}{\bar{\Pi}} - 1 \right) \frac{\Pi_{t+1}}{\bar{\Pi}} \frac{Y_{t+1}}{Y_t} \right]$$

**Real Dividends**:
$$\text{Div}_t = Y_t - W_t N_t$$

### 3. Policy Makers

**Monetary Policy - Taylor Rule with ZLB** (Equation 37):
$$R_t = \max\left\{ 1, \bar{R} \left( \frac{\Pi_t}{\bar{\Pi}} \right)^{\theta_\Pi} \left( \frac{Y_t}{\bar{Y}} \right)^{\theta_Y} \right\}$$

**Fiscal Policy** (Equation 38):
$$D = \frac{R_{t-1}}{\Pi_t} D - T_t$$

which implies:
$$T_t = D \left( \frac{R_{t-1}}{\Pi_t} - 1 \right)$$

### 4. Market Clearing

**Labor Market**:
$$N_t = H_t$$

**Bond Market**:
$$B_t = D$$

**Goods Market**:
$$Y_t = C_t$$

### 5. Shock Process

**Preference Shock** (AR(1)):
$$\zeta_t = \rho_\zeta \zeta_{t-1} + \epsilon_t^\zeta, \quad \epsilon_t^\zeta \sim N(0, \sigma_\zeta^2)$$

---

## Variable Identification Given NN Outputs

### Question
Given the neural network outputs $\{N_t, \Pi_t\}$ and the state $\zeta_t$, can we identify all other endogenous variables?

### Answer
**Yes!** All other variables can be computed algebraically. Here's the complete identification procedure:

### Step-by-Step Derivation

**Given**:
- State: $\zeta_t$
- NN outputs: $N_t$, $\Pi_t$
- Previous period: $R_{t-1}$
- Parameters: $\{\beta, \sigma, \eta, \epsilon, \chi, A, \theta_\Pi, \theta_Y, \phi, \bar{\Pi}, \bar{R}, \bar{Y}, D\}$

**Derive**:

**1. Output** (from production function, eq. 33):
$$Y_t = A \cdot N_t$$

**2. Consumption** (from goods market clearing):
$$C_t = Y_t = A \cdot N_t$$

**3. Hours worked** (from labor market clearing):
$$H_t = N_t$$

**4. Real wage** (from labor supply FOC, eq. 30):

From $\chi (H_t)^\eta = (C_t)^{-\sigma} W_t$:
$$W_t = \chi (H_t)^\eta (C_t)^\sigma = \chi (N_t)^\eta (A \cdot N_t)^\sigma$$

Simplified:
$$W_t = \chi A^\sigma (N_t)^{\eta + \sigma}$$

**5. Marginal cost** (from wage equation, eq. 34):

From $W_t = A \cdot MC_t$:
$$MC_t = \frac{W_t}{A} = \frac{\chi A^\sigma (N_t)^{\eta + \sigma}}{A} = \chi A^{\sigma - 1} (N_t)^{\eta + \sigma}$$

**6. Nominal interest rate** (from Taylor rule with ZLB, eq. 37):
$$R_t = \max\left\{ 1, \bar{R} \left( \frac{\Pi_t}{\bar{\Pi}} \right)^{\theta_\Pi} \left( \frac{A N_t}{\bar{Y}} \right)^{\theta_Y} \right\}$$

**7. Real dividends**:
$$\text{Div}_t = Y_t - W_t N_t = A N_t - \chi A^\sigma (N_t)^{\eta + \sigma + 1}$$

### Variables Requiring Calibration

**8. Government debt** (calibrated):
$$D = \text{constant (typically calibrated to match debt/GDP ratio)}$$

**9. Bond holdings** (from bond market clearing):
$$B_t = D$$

**10. Lump-sum taxes** (from fiscal rule):
$$T_t = D \left( \frac{R_{t-1}}{\Pi_t} - 1 \right)$$

### Summary Table

| **Variable** | **Formula** | **Type** |
|--------------|-------------|----------|
| $Y_t$ | $A \cdot N_t$ | Algebraic |
| $C_t$ | $Y_t$ | Algebraic |
| $H_t$ | $N_t$ | Algebraic |
| $W_t$ | $\chi A^\sigma (N_t)^{\eta + \sigma}$ | Algebraic |
| $MC_t$ | $W_t / A$ | Algebraic |
| $R_t$ | $\max\{1, \bar{R} (\Pi_t/\bar{\Pi})^{\theta_\Pi} (Y_t/\bar{Y})^{\theta_Y}\}$ | Algebraic |
| $\text{Div}_t$ | $Y_t - W_t N_t$ | Algebraic |
| $D$ | Calibrated constant | Parameter |
| $B_t$ | $D$ | Algebraic |
| $T_t$ | $D(R_{t-1}/\Pi_t - 1)$ | Algebraic |

---

## The Ricardian Equivalence Problem

### The Budget Constraint Question

**Initial claim**: We need the budget constraint as an equilibrium condition.

**Reality**: The budget constraint is **automatically satisfied** and provides no independent information!

### Mathematical Proof

Start with the household budget constraint:
$$C_t + B_t = W_t H_t + \frac{R_{t-1}}{\Pi_t} B_{t-1} - T_t + \text{Div}_t$$

Apply market clearing conditions:
- $B_t = B_{t-1} = D$
- $H_t = N_t$

Substitute:
$$C_t + D = W_t N_t + \frac{R_{t-1}}{\Pi_t} D - T_t + \text{Div}_t$$

Use the fiscal rule $T_t = D(R_{t-1}/\Pi_t - 1)$:
$$C_t + D = W_t N_t + \frac{R_{t-1}}{\Pi_t} D - D\left(\frac{R_{t-1}}{\Pi_t} - 1\right) + \text{Div}_t$$

Expand:
$$C_t + D = W_t N_t + \frac{R_{t-1}}{\Pi_t} D - D\frac{R_{t-1}}{\Pi_t} + D + \text{Div}_t$$

**The debt terms cancel completely**:
$$C_t = W_t N_t + \text{Div}_t$$

Substitute $\text{Div}_t = Y_t - W_t N_t$:
$$C_t = W_t N_t + Y_t - W_t N_t = Y_t$$

**Result**: The budget constraint reduces to **goods market clearing** ($C_t = Y_t$), which we already have!

### Implications

1. **Government debt level $D$ is indeterminate** from the equilibrium conditions
2. **Bond holdings $B_t$ and taxes $T_t$ are also indeterminate**
3. These variables don't affect real allocations (Ricardian equivalence)
4. $D$ must be **calibrated externally** (e.g., to match debt/GDP ratio)
5. The budget constraint is **not an independent equilibrium condition**

---

## Systematic Derivation of Independent Equations

### Goal
Determine the **minimal set of independent equations** that the neural network must satisfy.

### Step 1: List All Variables

**Endogenous Variables (13 total)**:
1. $C_t$ - consumption
2. $H_t$ - hours worked
3. $B_t$ - bond holdings
4. $N_t$ - labor hired by firms
5. $Y_t$ - output
6. $W_t$ - real wage
7. $MC_t$ - marginal cost
8. $\Pi_t$ - gross inflation
9. $R_t$ - gross nominal interest rate
10. $\text{Div}_t$ - real dividends
11. $T_t$ - lump-sum taxes
12. $R_{t-1}$ - lagged interest rate (predetermined)
13. $B_{t-1}$ - lagged bonds (predetermined)

**State Variables**:
- $\zeta_t$ - preference shock

**Parameters**:
- Calibrated: $\bar{\Theta} = \{\beta, \sigma, \eta, \epsilon, \chi, A, D\}$
- Estimated: $\tilde{\Theta} = \{\theta_\Pi, \theta_Y, \phi, \rho_\zeta, \sigma_\zeta\}$
- Steady-state targets: $\{\bar{\Pi}, \bar{R}, \bar{Y}\}$

### Step 2: Classify Equations

| **Equation** | **Type** | **Variables Defined** |
|--------------|----------|----------------------|
| Production: $Y_t = A N_t$ | Definition | $Y_t$ |
| Wage: $W_t = A \cdot MC_t$ | Definition | $W_t$ |
| Dividends: $\text{Div}_t = Y_t - W_t N_t$ | Definition | $\text{Div}_t$ |
| Fiscal rule: $T_t = D(R_{t-1}/\Pi_t - 1)$ | Definition | $T_t$ |
| Labor clearing: $N_t = H_t$ | Definition | Eliminates $H_t$ |
| Bond clearing: $B_t = D$ | Definition | Eliminates $B_t$ |
| Goods clearing: $C_t = Y_t$ | Constraint | $C_t$ |
| **Labor FOC**: $\chi (H_t)^\eta = (C_t)^{-\sigma} W_t$ | **Behavioral** | Links $N_t, MC_t$ |
| **Euler equation**: eq. (29) | **Behavioral** | Intertemporal |
| **Phillips curve**: eq. (36) | **Behavioral** | Price setting |
| **Taylor rule**: eq. (37) | **Policy** | $R_t$ |
| Budget constraint | **Redundant** | Satisfied automatically |

### Step 3: Reduce the System

After substituting all **definitions**, we're left with:

**Free Variables** (2):
- $N_t$ (chosen by NN)
- $\Pi_t$ (chosen by NN)

**Computed Variables**:
- $MC_t$ - from labor FOC
- $R_t$ - from Taylor rule
- All others - from definitions

### Step 4: The Labor FOC is Actually a Definition!

Substituting $C_t = A N_t$ and $W_t = A \cdot MC_t$ into the labor FOC:

$$\chi (N_t)^\eta = (A N_t)^{-\sigma} \cdot (A \cdot MC_t)$$

Solve for $MC_t$:
$$MC_t = \frac{\chi (N_t)^\eta \cdot (A N_t)^\sigma}{A} = \chi A^{\sigma - 1} (N_t)^{\eta + \sigma}$$

**This means $MC_t$ is uniquely determined by $N_t$!**

The labor FOC is not an independent equilibrium condition - it's a **definition** of how marginal cost relates to labor.

### Step 5: The Taylor Rule is a Policy Function

Given $\Pi_t$ and $N_t$ (which determines $Y_t$), the Taylor rule **computes** $R_t$:

$$R_t = \max\left\{ 1, \bar{R} \left( \frac{\Pi_t}{\bar{\Pi}} \right)^{\theta_\Pi} \left( \frac{A N_t}{\bar{Y}} \right)^{\theta_Y} \right\}$$

This is not an equilibrium condition to be satisfied - it's a **deterministic mapping**.

### Step 6: Identify True Equilibrium Conditions

We're left with **exactly 2 equations with expectations** that constrain the dynamics:

**1. Euler Equation** (intertemporal optimization):
$$1 = \beta R_t E_t \left[ \frac{\exp(\zeta_{t+1})}{\exp(\zeta_t)} \left( \frac{C_t}{C_{t+1}} \right)^{\sigma} \frac{1}{\Pi_{t+1}} \right]$$

**2. Phillips Curve** (optimal price setting):
$$\phi \left( \frac{\Pi_t}{\bar{\Pi}} - 1 \right) \frac{\Pi_t}{\bar{\Pi}} = (1-\epsilon) + \epsilon MC_t + \phi E_t \left[ \frac{\Pi_{t+1}}{R_t} \left( \frac{\Pi_{t+1}}{\bar{\Pi}} - 1 \right) \frac{\Pi_{t+1}}{\bar{\Pi}} \frac{Y_{t+1}}{Y_t} \right]$$

These are the **only two independent equilibrium conditions** with forward-looking expectations.

### Step 7: Why Only Two Equations?

**Counting argument**:
- NN outputs: 2 variables ($N_t$, $\Pi_t$)
- Independent equilibrium conditions needed: 2
- Additional equations: All are either definitions or automatically satisfied

**Economic intuition**:
- **Euler equation**: Households' optimal saving decision
- **Phillips curve**: Firms' optimal pricing decision
- Everything else follows from these two behavioral decisions plus market clearing

This is analogous to the linearized NK model (Section 3.1), which also has **2 equations**: Euler + Phillips.

---

## Final Loss Function for Neural Network

### Minimal Loss Function

The neural network must minimize:

$$\phi_L = \alpha_1 \mathcal{L}_{\text{Euler}} + \alpha_2 \mathcal{L}_{\text{Phillips}}$$

### Equation 1: Euler Equation Residual

Substituting $C_t = A N_t$:

$$\mathcal{L}_{\text{Euler}} = \left| 1 - \beta R_t E_t \left[ \frac{\exp(\zeta_{t+1})}{\exp(\zeta_t)} \left( \frac{N_t}{N_{t+1}} \right)^{\sigma} \frac{1}{\Pi_{t+1}} \right] \right|^2$$

### Equation 2: Phillips Curve Residual

Substituting $Y_t = A N_t$ and $MC_t = \chi A^{\sigma-1} (N_t)^{\eta + \sigma}$:

$$\begin{aligned}
\mathcal{L}_{\text{Phillips}} = \Bigg| &\phi \left( \frac{\Pi_t}{\bar{\Pi}} - 1 \right) \frac{\Pi_t}{\bar{\Pi}} - (1-\epsilon) - \epsilon \cdot \chi A^{\sigma-1} (N_t)^{\eta + \sigma} \\
&- \phi E_t \left[ \frac{\Pi_{t+1}}{R_t} \left( \frac{\Pi_{t+1}}{\bar{\Pi}} - 1 \right) \frac{\Pi_{t+1}}{\bar{\Pi}} \frac{N_{t+1}}{N_t} \right] \Bigg|^2
\end{aligned}$$

### Computing Auxiliary Variables

Before computing the loss, we need:

**Marginal Cost**:
$$MC_t = \chi A^{\sigma - 1} (N_t)^{\eta + \sigma}$$

**Interest Rate**:
$$R_t = \max\left\{ 1, \bar{R} \left( \frac{\Pi_t}{\bar{\Pi}} \right)^{\theta_\Pi} \left( \frac{A N_t}{\bar{Y}} \right)^{\theta_Y} \right\}$$

### Evaluating Expectations via Monte Carlo

Both loss components require expectations $E_t[\cdot]$. These are computed using Monte Carlo simulation:

**Algorithm**:
1. Draw $M$ future shocks: $\epsilon_{t+1}^{(m)} \sim N(0, \sigma_\zeta^2)$ for $m = 1, \ldots, M$
2. Compute future states: $\zeta_{t+1}^{(m)} = \rho_\zeta \zeta_t + \epsilon_{t+1}^{(m)}$
3. Evaluate NN at future states: $(N_{t+1}^{(m)}, \Pi_{t+1}^{(m)}) = \psi_{NN}(\zeta_{t+1}^{(m)}, \tilde{\Theta} | \bar{\Theta})$
4. Compute expectation: $E_t[\cdot] \approx \frac{1}{M} \sum_{m=1}^M [\cdot]^{(m)}$

The paper uses $M = 100$ Monte Carlo draws with antithetic variates.

### Batched Loss Function

For efficient GPU training, the loss is computed over a batch of size $B$:

$$\bar{\phi}_L = \frac{1}{B} \sum_{b=1}^{B} \left[ \alpha_1 \mathcal{L}_{\text{Euler}}^{(b)} + \alpha_2 \mathcal{L}_{\text{Phillips}}^{(b)} \right]$$

where each batch element has:
- Different state: $\zeta_t^{(b)}$
- Different parameters: $\tilde{\Theta}^{(b)}$ (sampled from prior)

### Loss Weights

The weights $\alpha_1$ and $\alpha_2$ can be adjusted to balance the magnitude of the two residuals. Common approaches:
- Equal weights: $\alpha_1 = \alpha_2 = 1$
- Inverse magnitude scaling: normalize by typical residual size
- Adaptive weighting during training

---

## Implementation Notes

### Neural Network Architecture

Based on the paper (page 53, Table 5):
- **Input dimension**: 6 (1 state $\zeta_t$ + 5 parameters)
- **Output dimension**: 2 ($N_t$, $\Pi_t$)
- **Hidden layers**: Not explicitly stated for RANK, but NK model uses 5 layers × 256 neurons
- **Activation functions**: CELU, Leaky ReLU, or SiLU
- **Recommended for RANK**: 5 layers × 128 neurons (based on HANK specifications)

### Training Details

**Optimizer**:
- AdamW with cosine annealing learning rate schedule
- Initial learning rate: $10^{-4}$
- Final learning rate: $10^{-6}$

**Batch size**: 100 (typical for this class of problems)

**Iterations**: 
- NK model: 500,000 iterations
- RANK model: Not explicitly stated, likely similar or less since it's simpler than HANK

**Convergence criterion**: Mean squared error < $10^{-6}$

### Handling the ZLB Constraint

The ZLB creates a **kink** in the interest rate function. The paper uses **gradual introduction**:

1. Start training without the ZLB (smooth problem)
2. Slowly introduce the constraint using:
   $$R_t = \max[R_t^N, 1] + a_{\text{ZLB}} \min[R_t^N - 1, 0]$$
3. Gradually increase $a_{\text{ZLB}}$ from 0 to 1 over iterations 5,000–10,000

This curriculum learning approach helps the NN converge despite the nonlinearity.

### Parameter Space Truncation

The NN is trained over a bounded parameter space (see Table 5):
- $\theta_\Pi \in [1.5, 2.5]$
- $\theta_Y \in [0.05, 0.5]$
- $\phi \in [700, 1300]$ (large values = sticky prices)
- $\rho_\zeta \in [0.5, 0.9]$
- $\sigma_\zeta \in [0.01, 0.025]$

### Validation

After training, validate the NN solution by:
1. **Comparing to global methods**: The paper compares to time iteration with piecewise linear policy functions
2. **Checking impulse responses**: Verify they match economic intuition
3. **Verifying parameter recovery**: Estimate the model with simulated data and check if true parameters are recovered

---

## Deterministic Steady State (DSS) for RANK Model

### Why Compute the Steady State?

The Taylor rule (equation 37) is specified relative to steady-state values:

$$R_t = \max\left\{ 1, \bar{R} \left( \frac{\Pi_t}{\bar{\Pi}} \right)^{\theta_\Pi} \left( \frac{Y_t}{\bar{Y}} \right)^{\theta_Y} \right\}$$

We need to know $\bar{R}$ and $\bar{Y}$ to evaluate the policy rule. Unlike HANK models where parameters affect the steady state (requiring a neural network), RANK has an **analytical steady state solution**.

### Definition of Steady State

In the **deterministic steady state (DSS)**:

1. All shocks are at their mean: $\zeta = 0$ (equivalently, $\exp(\zeta) = 1$)
2. All variables are constant over time: $X_t = X_{t+1} = \bar{X}$
3. All expectations are realized: $E_t[X_{t+1}] = \bar{X}$
4. The ZLB constraint is not binding: $\bar{R} > 1$

### Step-by-Step Analytical Derivation

**Step 1: Inflation at Target**

In steady state, inflation equals the central bank's target:
$$\bar{\Pi} = \Pi \quad \text{(exogenous parameter)}$$

**Step 2: Nominal Interest Rate from Euler Equation**

From the Euler equation (29) at steady state with $\exp(\zeta) = 1$:
$$1 = \beta \bar{R} \frac{1}{\bar{\Pi}}$$

Solving for $\bar{R}$:
$$\bar{R} = \frac{\bar{\Pi}}{\beta}$$

**Numerical Example**: With $\beta = 0.9975$ (quarterly) and $\bar{\Pi} = 1.005$ (2% annual):
$$\bar{R} = \frac{1.005}{0.9975} \approx 1.00753$$

This corresponds to approximately 3% annual nominal interest rate.

**Step 3: Verify Taylor Rule Consistency**

At steady state, the Taylor rule (37) should yield $\bar{R}$:
$$\bar{R} = \bar{R} \left( \frac{\bar{\Pi}}{\bar{\Pi}} \right)^{\theta_\Pi} \left( \frac{\bar{Y}}{\bar{Y}} \right)^{\theta_Y} = \bar{R} \cdot 1^{\theta_\Pi} \cdot 1^{\theta_Y} = \bar{R}$$ 

✓ Consistent by construction.

**Step 4: Marginal Cost from Phillips Curve**

From the Phillips curve (36) at steady state (with $\Pi_t = \Pi_{t+1} = \bar{\Pi}$):
$$\phi \left( \frac{\bar{\Pi}}{\bar{\Pi}} - 1 \right) \frac{\bar{\Pi}}{\bar{\Pi}} = (1-\epsilon) + \epsilon \bar{MC} + \beta \phi \left( \frac{\bar{\Pi}}{\bar{\Pi}} - 1 \right) \frac{\bar{\Pi}}{\bar{\Pi}}$$

Simplifying:
$$0 = (1-\epsilon) + \epsilon \bar{MC}$$

Therefore:
$$\bar{MC} = \frac{\epsilon - 1}{\epsilon}$$

This is the standard markup relationship: marginal cost equals the inverse of the gross markup $\mu = \epsilon/(\epsilon-1)$.

**Numerical Example**: With $\epsilon = 11$:
$$\bar{MC} = \frac{10}{11} \approx 0.909$$

**Step 5: Labor/Production from Labor FOC**

From the labor supply FOC (30) and the definitions, we derived:
$$MC_t = \chi A^{\sigma - 1} (N_t)^{\eta + \sigma}$$

At steady state:
$$\bar{MC} = \chi A^{\sigma - 1} (\bar{N})^{\eta + \sigma}$$

Solving for $\bar{N}$:
$$\bar{N} = \left( \frac{\bar{MC}}{\chi A^{\sigma - 1}} \right)^{\frac{1}{\eta + \sigma}}$$

**Step 6: Output**

From the production function:
$$\bar{Y} = A \cdot \bar{N}$$

**Step 7: Consumption**

From goods market clearing:
$$\bar{C} = \bar{Y}$$

**Step 8: Real Wage**

From the wage equation (34):
$$\bar{W} = A \cdot \bar{MC}$$

Or equivalently from the labor FOC (30):
$$\bar{W} = \chi (\bar{N})^\eta (\bar{C})^\sigma$$

**Step 9: Hours Worked**

From labor market clearing:
$$\bar{H} = \bar{N}$$

**Step 10: Government Debt (Calibrated)**

The debt level $D$ is **not determined** by the steady state equations (Ricardian equivalence). It must be **calibrated externally**, typically to match a target debt-to-GDP ratio:
$$D = \text{calibrated constant}$$

From Table 5, the paper doesn't specify $D$ for RANK, but for HANK they set it to achieve 25% of GDP.

**Step 11: Bond Holdings**

From bond market clearing:
$$\bar{B} = D$$

**Step 12: Taxes**

From the fiscal rule (38):
$$T_t = D \left( \frac{R_{t-1}}{\Pi_t} - 1 \right)$$

At steady state:
$$\bar{T} = D \left( \frac{\bar{R}}{\bar{\Pi}} - 1 \right) = D(\beta^{-1} - 1) = D \frac{1-\beta}{\beta}$$

**Step 13: Dividends**

$$\bar{\text{Div}} = \bar{Y} - \bar{W} \bar{N}$$

---

### Complete Steady State Algorithm

**Inputs**: 
- Structural parameters: $\{\beta, \sigma, \eta, \epsilon, \chi, A\}$
- Policy parameters: $\{\bar{\Pi}\}$ (inflation target)
- Calibrated: $D$ (government debt)

**Outputs**: 
- Steady state values: $\{\bar{R}, \bar{Y}, \bar{N}, \bar{C}, \bar{W}, \bar{MC}, \bar{H}, \bar{T}, \bar{B}, \bar{\text{Div}}\}$

**Algorithm**:

```
1. Compute nominal interest rate:
   R̄ = Π̄ / β

2. Compute marginal cost:
   MC̄ = (ε - 1) / ε

3. Compute labor:
   N̄ = [MC̄ / (χ A^(σ-1))]^(1/(η+σ))

4. Compute output:
   Ȳ = A × N̄

5. Compute consumption:
   C̄ = Ȳ

6. Compute wage:
   W̄ = A × MC̄

7. Compute hours:
   H̄ = N̄

8. Compute taxes:
   T̄ = D(1-β)/β

9. Compute bonds:
   B̄ = D

10. Compute dividends:
    Div̄ = Ȳ - W̄ N̄
```

---

### Calibration from Table 5

From the paper's RANK model calibration:

| **Parameter** | **Symbol** | **Value** | **Target/Implication** |
|---------------|------------|-----------|------------------------|
| Discount factor | $\beta$ | 0.9975 | ~4% annual nominal rate |
| Risk aversion | $\sigma$ | 1 | Log utility |
| Inverse Frisch | $\eta$ | 1 | Unit elasticity |
| Price elasticity | $\epsilon$ | 11 | Markup = 1.1, $\bar{MC} \approx 0.909$ |
| Disutility labor | $\chi$ | 0.91 | Normalizes labor |
| Rotemberg cost | $\phi$ | 1000 | Very sticky prices |
| Inflation target | $4\log(\bar{\Pi})$ | 2% | $\bar{\Pi} = 1.005$ (quarterly) |
| Output target | $\bar{Y}$ | 1 | Normalization |

**Deriving $\chi$ from Normalization**:

If we normalize $\bar{Y} = 1$ and set $A = 1$, then $\bar{N} = 1$.

From the marginal cost equation:
$$\bar{MC} = \chi A^{\sigma-1} \bar{N}^{\eta+\sigma}$$

With $\sigma = \eta = 1$, $A = 1$, $\bar{N} = 1$:
$$\frac{10}{11} = \chi \cdot 1^0 \cdot 1^2 = \chi$$

Therefore: $\chi \approx 0.909$

The paper uses $\chi = 0.91$, which is consistent with this calculation (likely with slight numerical differences in normalization).

---

### Implementation in Julia

```julia
struct RANKSteadyState{T}
    R_bar::T
    Y_bar::T
    N_bar::T
    C_bar::T
    W_bar::T
    MC_bar::T
    H_bar::T
    Π_bar::T
    T_bar::T
    B_bar::T
    Div_bar::T
end

function compute_rank_steady_state(params::RANKParameters{T}) where T
    # Unpack parameters
    @unpack β, σ, η, ϵ, χ, A, Π_bar, D = params
    
    # Step 1: Nominal interest rate (from Euler equation)
    R_bar = Π_bar / β
    
    # Step 2: Marginal cost (from zero-inflation Phillips curve)
    MC_bar = (ϵ - 1) / ϵ
    
    # Step 3: Labor (from labor FOC and MC definition)
    N_bar = (MC_bar / (χ * A^(σ - 1)))^(1 / (η + σ))
    
    # Step 4: Output
    Y_bar = A * N_bar
    
    # Step 5: Consumption (market clearing)
    C_bar = Y_bar
    
    # Step 6: Wage
    W_bar = A * MC_bar
    
    # Step 7: Hours
    H_bar = N_bar
    
    # Step 8: Taxes
    T_bar = D * (1 - β) / β
    
    # Step 9: Bonds
    B_bar = D
    
    # Step 10: Dividends
    Div_bar = Y_bar - W_bar * N_bar
    
    # Return steady state
    return RANKSteadyState(
        R_bar, Y_bar, N_bar, C_bar, W_bar, MC_bar, 
        H_bar, Π_bar, T_bar, B_bar, Div_bar
    )
end
```

---

### Why RANK Doesn't Need a DSS Neural Network

**RANK Model**:
- ✅ **Analytical steady state**: Closed-form solution exists
- ✅ **No parameter dependence**: Estimated parameters $\{\theta_\Pi, \theta_Y, \phi, \rho_\zeta, \sigma_\zeta\}$ don't affect DSS
- ✅ **Simple computation**: Just evaluate formulas
- ✅ **Fast**: No iteration or neural network needed

**HANK Model** (by contrast):
- ❌ **No analytical steady state**: Must iterate to find equilibrium
- ❌ **Parameter dependence**: $\sigma_s$ (idiosyncratic risk) and $B$ (borrowing limit) **affect DSS**
- ❌ **Wealth distribution**: Must solve for distribution of bonds across agents
- ❌ **Computationally intensive**: Requires separate neural network ($\psi_{NN}^{SS}$) to map parameters to DSS

From the paper (Appendix E.1):
> "We need the mapping from the parameters to the deterministic steady state values of the interest rate $R$ and detrended output $\tilde{Y}$, as these objects enter the Taylor rule. To obtain this mapping, we solve for the deterministic steady state (DSS) of our HANK model..."

For RANK, this mapping is trivial: $\bar{R} = \bar{\Pi}/\beta$ and $\bar{Y} = A\bar{N}$ where $\bar{N}$ is computed from $\bar{MC}$.

### Verification

To verify the steady state is correct, check:

1. **Euler equation**: $1 = \beta \bar{R} / \bar{\Pi}$ ✓
2. **Phillips curve**: $0 = (1-\epsilon) + \epsilon \bar{MC}$ ✓
3. **Labor FOC**: $\chi \bar{N}^\eta = \bar{C}^{-\sigma} \bar{W}$ ✓
4. **Production**: $\bar{Y} = A \bar{N}$ ✓
5. **Market clearing**: $\bar{C} = \bar{Y}$ ✓
6. **Taylor rule**: $\bar{R} = \bar{R}$ ✓
7. **Fiscal rule**: $\bar{T} = D(\bar{R}/\bar{\Pi} - 1)$ ✓

---

## Comparison to Other Models

### NK Model (Section 3.1)
- **Linearized** around steady state
- **2 equations**: Euler + Phillips (same as RANK!)
- **No ZLB**: Linear system has analytical solution
- **Simpler**: Used to validate the NN method
- **Steady state**: Linearization point, analytical

### RANK Model (Section 3.2)
- **Nonlinear** due to ZLB constraint
- **2 equations**: Euler + Phillips
- **Representative agent**: No heterogeneity
- **Ricardian equivalence**: Debt doesn't matter
- **Steady state**: Analytical, parameter-independent

### HANK Model (Section 3.3)
- **Highly nonlinear**: ZLB + borrowing constraints
- **Heterogeneous agents**: 100+ agents tracked
- **6+ equations**: Individual Euler equations, aggregate Phillips curve, bond market clearing (×2), resource constraint (×2)
- **No Ricardian equivalence**: Fiscal policy matters!
- **Computationally intensive**: Requires separate NNs for individual and aggregate policy functions
- **Steady state**: Requires neural network, parameter-dependent

---

## Key Takeaways

1. **Only 2 independent equilibrium conditions** for RANK: Euler equation and Phillips curve

2. **All other variables are either**:
   - Computed from definitions (production, wages, etc.)
   - Determined by policy rules (Taylor rule)
   - Automatically satisfied (budget constraint via Ricardian equivalence)

3. **The neural network** learns the mapping:
   $$(\zeta_t, \tilde{\Theta}) \mapsto (N_t, \Pi_t)$$
   such that the Euler equation and Phillips curve residuals are minimized

4. **Government debt is indeterminate** in RANK and must be calibrated externally

5. **The ZLB nonlinearity** requires careful training with gradual constraint introduction

6. **This structure parallels the NK model**, confirming that RANK is essentially NK with a ZLB constraint

---

## References

Kase, H., Melosi, L., & Rottner, M. (2025). *Estimating nonlinear heterogeneous agents model with neural networks*. Bank for International Settlements Working Paper.

- **Section 3.2**: RANK model with ZLB
- **Appendix C**: Complete RANK model specification
- **Table 5**: RANK parameter values and estimation results
- **Figure 11**: RANK training convergence

---

**Document Created**: December 31, 2025  
**Last Updated**: December 31, 2025

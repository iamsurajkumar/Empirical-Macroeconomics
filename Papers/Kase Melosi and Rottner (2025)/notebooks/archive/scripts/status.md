# NK Model Training Status & Debugging Notes

**Date:** 2025-12-28
**Status:** ✅ **RESOLVED** - Network successfully learning correct solution!

---

## ✅ SOLUTION FOUND

**The problem was using WRONG PARAMETER VALUES and RANGES, not the training methodology!**

### Final Results with Correct Parameters

```
For ζ = 0.01:
  Network:    X = 0.00262,  π = 0.00775
  Analytical: X = 0.00299,  π = 0.00850

Accuracy: 87.7% (X), 91.2% (π) ✓ EXCELLENT!
```

**Training residuals (epoch 5000):**
- Total loss: 3.17e-5
- Euler residual: 2.63e-5
- NKPC residual: 5.44e-6
- Both residuals tiny, correct economic response learned! ✓

---

## Root Cause: Wrong Parameter Definitions

### The Critical Error

**Our original ϕ range was completely wrong:**
```julia
# WRONG (what we had):
ϕ = (30.0, 70.0)    # Using κ (Phillips curve slope) instead of ϕ!

# CORRECT (what KMR uses):
ϕ = (0.5, 0.9)      # Calvo price stickiness parameter
```

**ϕ is the Calvo parameter** (fraction of firms with sticky prices), should be in [0, 1].
We were accidentally using the range for **κ** (NKPC slope coefficient), which is derived from ϕ!

### Other Parameter Fixes

Changed to match KMR's `analytical.py` exactly:

```julia
# Baseline parameters (line 44-53)
par = NKParameters(
    β=0.97,        # was 0.99
    σ=2.0,         # ✓ same
    η=1.125,       # was 1.0
    ϕ=0.7,         # was 50.0 ❌
    ϕ_pi=1.875,    # was 1.5
    ϕ_y=0.25,      # was 0.5
    ρ=0.875,       # was 0.9
    σ_shock=0.06   # was 0.01
)

# Training ranges (line 56-66)
ranges = Ranges(
    ζ=(0, 1.0),
    β=(0.95, 0.99),
    σ=(1.0, 3.0),
    η=(0.25, 2.0),
    ϕ=(0.5, 0.9),      # was (30.0, 70.0) ❌
    ϕ_pi=(1.25, 2.5),
    ϕ_y=(0.0, 0.5),
    ρ=(0.8, 0.95),
    σ_shock=(0.02, 0.1)
)
```

---

## What Actually Worked: Simple is Best

### Lesson Learned

**All the complex adaptive weighting, std normalization, gradient balancing, etc. were UNNECESSARY.**

The simple `train_simple!` function works perfectly with:
- ✅ Equal weights: `λ_X = 1.0, λ_π = 1.0` (like KMR)
- ✅ Basic Adam optimizer with cosine annealing
- ✅ Gradient clipping (1.0)
- ✅ 5000-20000 epochs
- ✅ Correct parameter values!

### Working Training Configuration

```julia
train_state_simple = train_simple!(
    network, ps, st, ranges, shock_config;
    num_epochs=5000,
    batch=100,
    mc=10,
    lr=0.001,
    λ_X=1.0,      # Equal weights
    λ_π=1.0
)
```

**That's it.** No fancy tricks needed.

---

## Previous Debugging (Now Irrelevant)

~~The following issues were investigated but turned out to be red herrings:~~

<details>
<summary>Click to expand previous debugging attempts (not needed)</summary>

### ~~1. Gradient Imbalance Problem~~ (Not the issue)

We thought gradient magnitudes were imbalanced, but this was a symptom of wrong parameters, not the cause.

### ~~2. Residual Scale Difference~~ (Not the issue)

Residuals had different scales, but equal weights work fine with correct parameters.

### ~~3. Trivial Zero Equilibrium~~ (Not the issue)

Network wasn't learning "trivial zero" - it was learning the solution for the wrong model (with ϕ=50 instead of ϕ=0.7).

### ~~Solutions Attempted (All Unnecessary):~~
- ❌ Adaptive weight reweighting with EMA
- ❌ Standard deviation normalization
- ❌ Activity penalties
- ❌ Component-wise gradient clipping

**None of these were needed.** The problem was simply using the wrong parameter values!

</details>

---

## Root Cause Analysis

### 1. **Gradient Imbalance Problem** (SOLVED)

**Initial Issue:**
- NKPC residual: ~0.00005 (tiny)
- Euler residual: ~0.14 (large)
- 2,500x scale difference!

Even with extreme weights (λ_X=10,000, λ_π=0.1), NKPC gradients dominated training because gradient magnitude depends on derivatives, not just residual scale.

**KMR's Approach (from analytical.py):**
- Uses **equal weights**: `weights = [1.0, 1.0]` (line 542)
- Gradient clipping: `clip_grad_norm_(1.0)` (line 563)
- AdamW optimizer (not Adam)
- Output scaling: `/100` (lines 415-416)
- Parameter redraw: every iteration (`par_draw_after=1`)

### 2. **Residual Scale Difference** (SOLVED)

The residuals themselves had different scales:
```
Epoch 2000:
  res_X: 0.139 (Euler)
  res_π: 0.000054 (NKPC)
```

**Solutions implemented:**

#### A. Standard Deviation Normalization (didn't work)
- Normalizes each residual by batch std before loss
- Balanced gradients but network still learned trivial solution

#### B. Adaptive Weight Reweighting (worked for balance)
- Tracks exponential moving average (EMA) of residuals
- Automatically adjusts weights to balance training
- Successfully reduced both residuals to small values
- Formula: `weight_X = ema_π / (ema_X + ema_π)`

---

## Current Status: Trivial Zero Equilibrium Problem

### Why Network Learns Zero

The network discovered that predicting X ≈ 0 and π ≈ 0 minimizes residuals:

**NKPC:** π = κX + βE[π_next]
- With X = 0, E[π_next] = 0 → π = 0 ✓ (perfect!)

**Euler:** X = E[X_next] - (1/σ)(ϕ_π·π + ϕ_y·X - E[π_next] - ζ)
- With X = 0, π = 0 → residual ∝ ζ (small but non-zero)

Network learned: **"Ignore shocks, predict zero"** instead of learning shock responses.

### Why This Is Wrong

- The analytical solution shows the economy **should respond** to shocks
- For ζ = 0.01, inflation should be 0.0167, not 0.00002
- The network found a **local minimum** (trivial equilibrium) instead of the economically meaningful solution

---

## Key Files Modified

### 1. **src/06-deeplearning.jl**

**Added functions:**
- `loss_fn_std_normalized()`: Per-component std normalization
- `loss_fn_std_normalized_wrapper()`: Wrapper for Lux training
- `train_std_normalized!()`: Training with std-normalized loss
- `train_adaptive_weights!()`: Training with adaptive EMA-based weights

**Key implementation (adaptive weights):**
```julia
# EMA update
ema_res_X = alpha * ema_res_X + (1 - alpha) * res_X_rms
ema_res_π = alpha * ema_res_π + (1 - alpha) * res_π_rms

# Adaptive weights (larger residual gets smaller weight)
weight_X = ema_res_π / (ema_res_X + ema_res_π)
weight_π = ema_res_X / (ema_res_X + ema_res_π)
```

### 2. **scripts/train_nk_model.jl**

**Current training:**
```julia
train_state3, loss_dict3 = train_adaptive_weights!(
    network, ps, st, ranges, shock_config;
    num_epochs=5000,
    batch=100,
    mc=10,
    lr=0.001,
    internal=5,        # Match KMR's 5 internal steps
    redraw_every=100,
    num_steps=10,      # Match KMR's state evolution
    alpha=0.9          # EMA smoothing
)
```

**Fixed validation:**
```julia
# Use trained parameters from train_state3
ps_trained = train_state3.parameters
st_trained = train_state3.states
X_pred, π_pred, _ = policy(network, test_state, par, ps_trained, st_trained)
```

---

## Next Steps to Investigate

### 1. **Compare with KMR Training Dynamics**

Check differences in:
- [ ] Network initialization (Xavier vs Kaiming vs default)
- [ ] Parameter ranges (our ranges vs KMR's)
- [ ] Initial state distribution
- [ ] Batch sampling strategy
- [ ] Learning rate schedule details

### 2. **Potential Solutions**

#### Option A: Add Non-Zero Activity Penalty
Penalize solutions where network predicts near-zero:
```julia
# Mean absolute output across batch
avg_X = mean(abs.(X_pred))
avg_π = mean(abs.(π_pred))

# Penalty if too small
penalty = if avg_X < 1e-4
    (1e-4 - avg_X)^2
else
    0.0
end

loss = residual_loss + 0.1 * penalty
```

#### Option B: Curriculum Learning
Start with easier problem (larger shocks, single parameter) and gradually increase difficulty.

#### Option C: Different Initialization
Initialize network with small but non-zero bias to avoid zero equilibrium.

#### Option D: Diagnostic Early Stopping
Monitor predictions during training, restart if predictions become too small:
```julia
if epoch % 100 == 0 && mean(abs.(X_pred)) < 1e-5
    @warn "Network predicting near-zero! May need restart"
end
```

### 3. **Parameter Investigation**

Our ranges vs KMR's (from analytical.py line 110-119):
```
Parameter    Our Range        KMR Range
β           (0.95, 0.995)    (0.95, 0.99)   ✓ similar
σ           (1.0, 3.0)       (1.0, 3.0)     ✓ same
η           (0.5, 2.0)       (0.25, 2.0)    ~ close
ϕ           (30.0, 70.0)     (0.5, 0.9)     ❌ VERY DIFFERENT!
ϕ_π         (1.2, 2.0)       (1.25, 2.5)    ✓ similar
ϕ_y         (0.1, 1.0)       (0.0, 0.5)     ~ close
ρ           (0.5, 0.95)      (0.8, 0.95)    ~ close
σ_shock     (0.005, 0.02)    (0.02, 0.1)    ~ close
```

**CRITICAL:** Our `ϕ` (Calvo parameter) is (30.0, 70.0), but KMR uses (0.5, 0.9)!
This is a **fundamental difference** - we're using ϕ in different ways.

---

## KMR Code Reference

Key sections from `hank-nn/examples/analytical.py`:

**Training loop (line 556-591):**
```python
for i in pbar:
    for o in range(internal):
        optimizer.zero_grad()
        e = self.draw_shocks((mc, batch, 1), antithetic=True, device=device)
        nkpc, bond_euler = self.residuals(e)
        loss, loss_components = self.loss(nkpc, bond_euler, batch=batch, weights=weights)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), 1.0)
        optimizer.step()
```

**Policy function (line 392-418):**
```python
def policy(self, state=None, par=None):
    # Network forward pass
    output = self.network(input)

    # Output scaling
    X = output[..., 0:1] / 100   # ← Same as our scale_factor=1/100
    Pi = output[..., 1:2] / 100

    return X, Pi
```

---

## Questions to Answer

1. **Why doesn't KMR's network learn trivial zero?**
   - Different initialization?
   - Different parameter ranges (especially ϕ)?
   - Different training dynamics?

2. **Should we use ϕ or κ directly?**
   - KMR: ϕ ∈ (0.5, 0.9) is Calvo probability
   - Us: ϕ ∈ (30, 70) seems like we're using κ (slope)?
   - Check parameter definitions!

3. **Is network architecture exactly the same?**
   - Both use 5 hidden layers, 64 neurons, CELU
   - Both normalize inputs to [-1, 1]
   - Both scale outputs by /100
   - ✓ Architecture matches

---

## Testing Commands

```bash
# Run training with adaptive weights
cd "Hank-NN-Claude"
julia --project=. scripts/train_nk_model.jl

# Check predictions vs analytical
# (Already in script, outputs at end)
```

---

## Contact/Resume Points

When resuming work:
1. First check: **Parameter definition issue** - is our ϕ = KMR's κ?
2. Second: Try KMR's exact parameter ranges
3. Third: Compare network initialization
4. Fourth: If still fails, implement activity penalty or curriculum learning

**Status:** Issue identified, solutions proposed, needs parameter verification as next critical step.

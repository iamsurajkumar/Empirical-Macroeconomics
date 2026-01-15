# ═══════════════════════════════════════════════════════════════════════════
# RANK MODEL WITH ZLB - SPECIFIC FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════

# ───────────────────────────────────────────────────────────────────────────
# ZLB Constraint Scheduling
# ───────────────────────────────────────────────────────────────────────────

"""
    apply_zlb_schedule(R_natural, iter::Int, config::TrainingConfig)

Gradually introduce ZLB constraint during training (Kase et al. Equation 44).

Formula: R = max(R_natural, 1) + a^ZLB * min(R_natural - 1, 0)

where a^ZLB ∈ [0,1] controls the transition:
- Before zlb_start_iter: a^ZLB = 1 → no ZLB
- After zlb_end_iter: a^ZLB = 0 → full ZLB
- In between: linear decrease

Args:
    R_natural: Unconstrained interest rate from Taylor rule
    iter: Current training iteration
    config: TrainingConfig with zlb_start_iter and zlb_end_iter

Returns:
    R_constrained: Interest rate with (possibly partial) ZLB constraint
"""
function apply_zlb_schedule(R_natural, iter::Int, config::TrainingConfig)
    if iter < config.zlb_start_iter
        # No ZLB yet
        a_zlb = 1.0
    elseif iter > config.zlb_end_iter
        # Full ZLB
        a_zlb = 0.0
    else
        # Gradual transition
        progress = (iter - config.zlb_start_iter) / (config.zlb_end_iter - config.zlb_start_iter)
        a_zlb = 1.0 - progress
    end

    return max.(R_natural, 1.0) .+ a_zlb .* min.(R_natural .- 1.0, 0.0)
end

# ───────────────────────────────────────────────────────────────────────────
# Policy Functions
# ───────────────────────────────────────────────────────────────────────────

"""
    policy(network, state::State, par::RANKParameters, ps, st)

Evaluate NN policy for RANK model.
Returns (N_t, Π_t) given current state.

[FUTURE IMPLEMENTATION - Similar to NK but outputs labor and inflation]
"""
function policy(network, state::State, par::RANKParameters, ps, st)
    error("RANK policy not yet implemented - future work")
end

# ───────────────────────────────────────────────────────────────────────────
# State Transition
# ───────────────────────────────────────────────────────────────────────────

"""
    step(network, state::State, shock, par::RANKParameters, ps, st,
         iter::Int, config::TrainingConfig)

One-period state transition with ZLB constraint.
Key difference from NK: applies ZLB to interest rate using apply_zlb_schedule().

Process:
1. Get policy from network: (N_t, Π_t)
2. Compute Taylor rule: R_natural
3. Apply ZLB schedule: R_t = apply_zlb_schedule(R_natural, iter, config)
4. Update state with shock

[FUTURE IMPLEMENTATION]
"""
function step(network, state::State, shock, par::RANKParameters, ps, st,
             iter::Int, config::TrainingConfig)
    error("RANK step not yet implemented - future work")
end

# ───────────────────────────────────────────────────────────────────────────
# Residuals
# ───────────────────────────────────────────────────────────────────────────

"""
    residuals(network, state::State, par::RANKParameters, ps, st, shocks)

Compute residuals for RANK model with ZLB:
1. Euler equation residual
2. NKPC residual
3. ZLB constraint automatically handled in step() function

Note: ZLB affects state transitions, not residuals directly.
The residuals check if Euler + NKPC hold given the policy functions.

[FUTURE IMPLEMENTATION]
"""
function residuals(network, state::State, par::RANKParameters, ps, st, shocks)
    error("RANK residuals not yet implemented - future work")
end

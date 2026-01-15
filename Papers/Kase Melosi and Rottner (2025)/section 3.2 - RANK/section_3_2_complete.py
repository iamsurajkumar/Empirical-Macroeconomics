"""
Section 3.2: RANK Model with ZLB - Complete Implementation
==========================================================

This file contains all code for implementing Section 3.2 from
Kase, Melosi, and Rottner (2025): "Estimating nonlinear heterogeneous
agents models with neural networks"

Author: [Your Name]
Date: December 2025

Project Structure:
-----------------
section_3_2_RANK/
├── models/
│   └── rank_model.py          (Tasks 2-5)
├── networks/
│   ├── policy_nn.py           (Tasks 6, 8)
│   └── pf_nn.py               (Task 14)
├── estimation/
│   ├── particle_filter.py     (Tasks 11-12)
│   ├── train_policy.py        (Tasks 9-10)
│   ├── train_pf.py            (Tasks 13-14)
│   └── rwmh.py                (Tasks 16-18)
├── utils/
│   ├── priors.py              (Task 15)
│   ├── quasi_random.py        (Task 7)
│   └── plotting.py            (Tasks 18, 20-21)
└── main.py                    (Tasks 17, 22)

Usage:
------
1. Train policy network:
   python main.py --task train_policy

2. Generate PF training data:
   python main.py --task generate_pf_data

3. Train NN particle filter:
   python main.py --task train_pf

4. Run estimation:
   python main.py --task estimate

5. Full pipeline:
   python main.py --task all
"""

# ============================================================================
# IMPORTS
# ============================================================================

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import truncnorm
from scipy.stats.qmc import Sobol

from dataclasses import dataclass
from typing import Dict, Tuple, Optional
from tqdm import tqdm
import argparse
import os
from pathlib import Path


# ============================================================================
# TASK 2: DEFINE RANK MODEL PARAMETERS
# ============================================================================

@dataclass
class RANKParameters:
    """
    RANK model parameters (Appendix C)
    
    Calibrated parameters (fixed):
    - β: Discount factor = 0.9975 (4% annual interest rate)
    - σ: Risk aversion = 1.0
    - η: Inverse Frisch elasticity = 0.72
    - ε: Elasticity of substitution = 10.0
    - χ: Disutility of labor (calibrated for H=1)
    
    Estimated parameters:
    - θ_π: Taylor rule inflation response
    - θ_y: Taylor rule output response  
    - φ: Rotemberg price adjustment cost
    - ρ_ζ: Preference shock persistence
    - σ_ζ: Preference shock std dev
    """
    
    # Calibrated (fixed)
    beta: float = 0.9975
    sigma: float = 1.0
    eta: float = 0.72
    epsilon: float = 10.0
    chi: float = 0.74  # Calibrated to hit H=1 in steady state
    Pi_target: float = 1.0  # Zero inflation target
    
    # Estimated (variable)
    theta_pi: float = 2.0
    theta_y: float = 0.5
    phi: float = 100.0
    rho_zeta: float = 0.7
    sigma_zeta: float = 0.02
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary for vectorization"""
        return {
            'theta_pi': self.theta_pi,
            'theta_y': self.theta_y,
            'phi': self.phi,
            'rho_zeta': self.rho_zeta,
            'sigma_zeta': self.sigma_zeta
        }
    
    @staticmethod
    def get_bounds():
        """Parameter bounds for quasi-random sampling"""
        return {
            'theta_pi': (1.5, 3.0),
            'theta_y': (0.1, 1.0),
            'phi': (50.0, 200.0),
            'rho_zeta': (0.5, 0.95),
            'sigma_zeta': (0.01, 0.05)
        }


# ============================================================================
# TASK 3: IMPLEMENT MODEL EQUILIBRIUM CONDITIONS
# ============================================================================

class RANKModel:
    """
    RANK model with ZLB constraint
    
    Equations (Appendix C):
    1. Euler equation: C_t^(-σ) = β E_t[C_{t+1}^(-σ) R_t / Π_{t+1}]
    2. NKPC: φ(Π_t/Π - 1)(Π_t/Π) = (1-ε) + ε*MC_t + ...
    3. Taylor rule: R_t = max[1, R*(Π_t/Π)^θ_π * (Y_t/Y)^θ_y]
    4. Market clearing: Y_t = C_t
    5. Production: Y_t = N_t
    6. Labor supply: χ*H_t^η = W_t * C_t^(-σ)
    7. Marginal cost: MC_t = W_t
    8. Preference shock: ζ_t = ρ_ζ * ζ_{t-1} + ε_t^ζ
    """
    
    def __init__(self, params: RANKParameters):
        self.params = params
        self.steady_state = self._compute_steady_state()
    
    def _compute_steady_state(self):
        """
        Compute deterministic steady state
        
        In steady state (no shocks, no ZLB):
        - R = 1/β
        - Π = Π_target = 1.0
        - MC = (ε-1)/ε
        - W = MC
        - C^(-σ) = β R
        - H^η = W * C^(-σ) / χ
        - Y = C = N
        """
        p = self.params
        
        R_ss = 1.0 / p.beta
        Pi_ss = p.Pi_target
        MC_ss = (p.epsilon - 1) / p.epsilon
        W_ss = MC_ss
        
        # From Euler equation
        C_ss = (p.beta * R_ss) ** (-1/p.sigma)
        
        # From labor supply
        H_ss = ((W_ss * C_ss**(-p.sigma)) / p.chi) ** (1/p.eta)
        
        Y_ss = C_ss
        N_ss = H_ss
        
        return {
            'R': R_ss,
            'Pi': Pi_ss,
            'C': C_ss,
            'Y': Y_ss,
            'N': N_ss,
            'H': H_ss,
            'W': W_ss,
            'MC': MC_ss
        }
    
    def equilibrium_residuals(self, states, controls, next_states, next_controls, 
                             params_tensor=None):
        """
        Compute residuals of equilibrium conditions
        
        Args:
            states: (batch, 1) - current ζ_t
            controls: (batch, 2) - [N_t, Π_t]
            next_states: (batch, 1) - next ζ_{t+1}
            next_controls: (batch, 2) - [N_{t+1}, Π_{t+1}]
            params_tensor: (batch, 5) - [θ_π, θ_y, φ, ρ_ζ, σ_ζ] (optional)
        
        Returns:
            residuals: (batch, 2) - [euler_residual, nkpc_residual]
        """
        if params_tensor is None:
            # Use fixed parameters
            p = self.params
            theta_pi = torch.tensor(p.theta_pi, dtype=torch.float32)
            theta_y = torch.tensor(p.theta_y, dtype=torch.float32)
            phi = torch.tensor(p.phi, dtype=torch.float32)
        else:
            # Extended policy functions (parameters as inputs)
            theta_pi = params_tensor[:, 0]
            theta_y = params_tensor[:, 1]
            phi = params_tensor[:, 2]
        
        # Unpack states and controls
        zeta = states[:, 0]
        N_t = controls[:, 0]
        Pi_t = controls[:, 1]
        
        zeta_next = next_states[:, 0]
        N_next = next_controls[:, 0]
        Pi_next = next_controls[:, 1]
        
        # Current period variables
        Y_t = N_t  # Production function
        C_t = Y_t  # Market clearing
        W_t = (self.params.chi * N_t**self.params.eta) / (C_t**(-self.params.sigma))
        MC_t = W_t
        
        # Next period variables
        Y_next = N_next
        C_next = Y_next
        
        # Taylor rule (notional rate, before ZLB)
        R_ss = self.steady_state['R']
        Y_ss = self.steady_state['Y']
        Pi_target = self.params.Pi_target
        
        R_notional = R_ss * (Pi_t / Pi_target)**theta_pi * (Y_t / Y_ss)**theta_y
        R_t = torch.maximum(R_notional, torch.ones_like(R_notional))  # ZLB
        
        # Residual 1: Euler equation
        euler_lhs = C_t**(-self.params.sigma) * torch.exp(zeta)
        euler_rhs = self.params.beta * C_next**(-self.params.sigma) * \
                   torch.exp(zeta_next) * R_t / Pi_next
        euler_residual = euler_lhs - euler_rhs
        
        # Residual 2: NKPC
        nkpc_lhs = phi * (Pi_t/Pi_target - 1) * (Pi_t/Pi_target)
        nkpc_rhs = (1 - self.params.epsilon) + self.params.epsilon * MC_t + \
                   phi * (R_t/Pi_next) * (Pi_next/Pi_target - 1) * \
                   (Pi_next/Pi_target) * (Y_next/Y_t)
        nkpc_residual = nkpc_lhs - nkpc_rhs
        
        return torch.stack([euler_residual, nkpc_residual], dim=1)


# ============================================================================
# TASK 4: IMPLEMENT STATE TRANSITION (SHOCK PROCESS)
# ============================================================================

    def simulate_shocks(self, T, params=None, seed=None):
        """
        Simulate preference shock path
        
        ζ_t = ρ_ζ * ζ_{t-1} + ε_t^ζ, where ε_t^ζ ~ N(0, σ_ζ^2)
        """
        if params is None:
            params = self.params
        
        if seed is not None:
            torch.manual_seed(seed)
        
        zeta = torch.zeros(T)
        innovations = torch.randn(T) * params.sigma_zeta
        
        for t in range(1, T):
            zeta[t] = params.rho_zeta * zeta[t-1] + innovations[t]
        
        return zeta


# ============================================================================
# TASK 5: CREATE DATA SIMULATION FUNCTION
# ============================================================================

    def simulate_data(self, T, params, policy_nn, seed=None):
        """
        Simulate model using trained policy functions
        
        Returns:
            simulated_data: Dictionary with observables
        """
        # Simulate shocks
        zeta = self.simulate_shocks(T, params, seed)
        
        # Storage
        N = torch.zeros(T)
        Pi = torch.zeros(T)
        Y = torch.zeros(T)
        R = torch.zeros(T)
        
        # Initial conditions (steady state)
        N[0] = torch.tensor(self.steady_state['N'], dtype=torch.float32)
        Pi[0] = torch.tensor(self.steady_state['Pi'], dtype=torch.float32)
        
        # Simulate forward using policy functions
        policy_nn.eval()
        with torch.no_grad():
            for t in range(T):
                # Policy NN input: [ζ_t, θ_π, θ_y, φ, ρ_ζ, σ_ζ]
                state_params = torch.tensor([[
                    zeta[t].item(),
                    params.theta_pi,
                    params.theta_y,
                    params.phi,
                    params.rho_zeta,
                    params.sigma_zeta
                ]], dtype=torch.float32)
                
                # Get policy function values
                controls = policy_nn(state_params)
                N[t] = controls[0, 0]
                Pi[t] = controls[0, 1]
                
                # Compute observables
                Y[t] = N[t]
                
                # Taylor rule
                R_ss = self.steady_state['R']
                Y_ss = self.steady_state['Y']
                R_notional = R_ss * (Pi[t] / self.params.Pi_target)**params.theta_pi * \
                            (Y[t] / Y_ss)**params.theta_y
                R[t] = torch.maximum(R_notional, torch.ones_like(R_notional))
        
        return {
            'Y': Y,
            'Pi': Pi,
            'R': R,
            'zeta': zeta
        }


# ============================================================================
# TASK 6: DEFINE POLICY FUNCTION NEURAL NETWORK ARCHITECTURE
# ============================================================================

class PolicyNetwork(nn.Module):
    """
    Neural network for RANK policy functions
    
    Architecture (from paper Section 3.2):
    - Input: (batch, 6) = [ζ_t, θ_π, θ_y, φ, ρ_ζ, σ_ζ]
    - 5 hidden layers, 128 neurons each
    - Activation: SiLU
    - Output: (batch, 2) = [N_t, Π_t]
    """
    
    def __init__(self, n_inputs=6, n_outputs=2, hidden_size=128, n_layers=5):
        super().__init__()
        
        layers = []
        
        # Input layer
        layers.append(nn.Linear(n_inputs, hidden_size))
        layers.append(nn.SiLU())
        
        # Hidden layers
        for _ in range(n_layers - 1):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.SiLU())
        
        # Output layer
        layers.append(nn.Linear(hidden_size, n_outputs))
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
    
    def forward(self, x):
        """
        Args:
            x: (batch, 6) = [ζ_t, θ_π, θ_y, φ, ρ_ζ, σ_ζ]
        Returns:
            (batch, 2) = [N_t, Π_t]
        """
        return self.network(x)


# ============================================================================
# TASK 7: IMPLEMENT QUASI-RANDOM PARAMETER SAMPLING
# ============================================================================

def quasi_random_sample(bounds, n_samples, seed=None):
    """
    Generate quasi-random samples using Sobol sequence
    
    Args:
        bounds: Dict of parameter bounds
        n_samples: Number of samples
        seed: Random seed
        
    Returns:
        samples: (n_samples, n_params) tensor
        param_names: List of parameter names
    """
    n_params = len(bounds)
    
    # Generate Sobol sequence in [0, 1]^n_params
    sampler = Sobol(d=n_params, scramble=True, seed=seed)
    sobol_samples = sampler.random(n_samples)
    
    # Scale to parameter bounds
    samples = np.zeros_like(sobol_samples)
    param_names = list(bounds.keys())
    
    for i, param_name in enumerate(param_names):
        lower, upper = bounds[param_name]
        samples[:, i] = sobol_samples[:, i] * (upper - lower) + lower
    
    return torch.tensor(samples, dtype=torch.float32), param_names


# ============================================================================
# TASK 8: IMPLEMENT ZLB CONSTRAINT SCHEDULING
# ============================================================================

class ZLBScheduler:
    """
    Gradually introduce ZLB constraint during training
    
    R_t = max[R_t^N, 1] + α^ZLB * min[R_t^N - 1, 0]
    α^ZLB goes from 0 → 1 over iterations
    """
    
    def __init__(self, start_iter=5000, end_iter=10000):
        self.start_iter = start_iter
        self.end_iter = end_iter
    
    def get_alpha(self, iteration):
        """Get current ZLB constraint strength"""
        if iteration < self.start_iter:
            return 0.0
        elif iteration > self.end_iter:
            return 1.0
        else:
            # Linear interpolation
            progress = (iteration - self.start_iter) / (self.end_iter - self.start_iter)
            return progress
    
    def apply_zlb(self, R_notional, iteration):
        """Apply ZLB constraint with current strength"""
        alpha = self.get_alpha(iteration)
        
        R_zlb = torch.maximum(R_notional, torch.ones_like(R_notional))
        soft_penalty = alpha * torch.minimum(R_notional - 1, torch.zeros_like(R_notional))
        
        return R_zlb + soft_penalty


# ============================================================================
# TASK 9: IMPLEMENT POLICY FUNCTION TRAINING LOOP
# ============================================================================

def train_policy_network(
    model,
    policy_nn,
    n_iterations=100000,
    batch_size=100,
    lr_initial=1e-4,
    lr_final=1e-7,
    device='cuda' if torch.cuda.is_available() else 'cpu',
    log_dir='runs/rank_policy',
    save_path='results/policy_nn.pt'
):
    """
    Train policy function neural network
    
    Loss: MSE of equilibrium residuals
    """
    policy_nn = policy_nn.to(device)
    
    # Optimizer: AdamW
    optimizer = optim.AdamW(policy_nn.parameters(), lr=lr_initial)
    
    # Learning rate scheduler: Cosine annealing
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=n_iterations, eta_min=lr_final
    )
    
    # ZLB scheduler
    zlb_scheduler = ZLBScheduler(start_iter=5000, end_iter=10000)
    
    # TensorBoard
    writer = SummaryWriter(log_dir)
    
    # Get parameter bounds
    bounds = RANKParameters.get_bounds()
    
    # Training history
    losses = []
    
    # Training loop
    pbar = tqdm(range(n_iterations), desc='Training Policy NN')
    for iteration in pbar:
        # Sample shocks from stationary distribution
        # Var(ζ) = σ_ζ^2 / (1 - ρ_ζ^2)
        stationary_std = 0.02 / np.sqrt(1 - 0.7**2)
        zeta = torch.randn(batch_size, 1) * stationary_std
        
        # Sample parameters (quasi-random for better coverage)
        if iteration % 1000 == 0:
            param_samples, _ = quasi_random_sample(bounds, batch_size, seed=iteration//1000)
        else:
            # Use uniform random after initial spread
            param_samples = torch.rand(batch_size, 5)
            for i, (param_name, (lower, upper)) in enumerate(bounds.items()):
                param_samples[:, i] = param_samples[:, i] * (upper - lower) + lower
        
        param_samples = param_samples.to(device)
        
        # Combine into NN input: [ζ, θ_π, θ_y, φ, ρ_ζ, σ_ζ]
        inputs = torch.cat([zeta.to(device), param_samples], dim=1)
        
        # Forward pass: get policy functions
        controls = policy_nn(inputs)  # [N_t, Π_t]
        
        # Sample next period shocks
        rho_zeta = param_samples[:, 3:4]
        sigma_zeta = param_samples[:, 4:5]
        innovations = torch.randn(batch_size, 1, device=device) * sigma_zeta
        zeta_next = rho_zeta * zeta.to(device) + innovations
        
        inputs_next = torch.cat([zeta_next, param_samples], dim=1)
        controls_next = policy_nn(inputs_next)
        
        # Compute equilibrium residuals
        residuals = model.equilibrium_residuals(
            states=zeta.to(device),
            controls=controls,
            next_states=zeta_next,
            next_controls=controls_next,
            params_tensor=param_samples
        )
        
        # Loss: MSE of residuals
        loss = (residuals ** 2).mean()
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        # Logging
        losses.append(loss.item())
        pbar.set_postfix({'loss': f'{loss.item():.6f}'})
        
        if iteration % 100 == 0:
            writer.add_scalar('Loss/train', loss.item(), iteration)
            writer.add_scalar('LR', scheduler.get_last_lr()[0], iteration)
            writer.add_scalar('ZLB_alpha', zlb_scheduler.get_alpha(iteration), iteration)
    
    writer.close()
    
    # Save model
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(policy_nn.state_dict(), save_path)
    print(f"\nPolicy network saved to {save_path}")
    
    return policy_nn, {'losses': losses}


# ============================================================================
# TASK 10: VALIDATE POLICY FUNCTION ACCURACY
# ============================================================================

def validate_policy_network(model, policy_nn, n_test=1000, device='cpu'):
    """
    Validate trained policy network
    """
    policy_nn.eval()
    policy_nn = policy_nn.to(device)
    
    # Generate test data
    bounds = RANKParameters.get_bounds()
    zeta_test = torch.linspace(-0.05, 0.05, n_test).unsqueeze(1).to(device)
    
    # Use median parameter values
    params_median = torch.tensor([[
        (bounds['theta_pi'][0] + bounds['theta_pi'][1]) / 2,
        (bounds['theta_y'][0] + bounds['theta_y'][1]) / 2,
        (bounds['phi'][0] + bounds['phi'][1]) / 2,
        (bounds['rho_zeta'][0] + bounds['rho_zeta'][1]) / 2,
        (bounds['sigma_zeta'][0] + bounds['sigma_zeta'][1]) / 2,
    ]], dtype=torch.float32).repeat(n_test, 1).to(device)
    
    inputs = torch.cat([zeta_test, params_median], dim=1)
    
    with torch.no_grad():
        controls = policy_nn(inputs)
        
        # Compute residuals
        zeta_next = params_median[:, 3:4] * zeta_test
        inputs_next = torch.cat([zeta_next, params_median], dim=1)
        controls_next = policy_nn(inputs_next)
        
        residuals = model.equilibrium_residuals(
            states=zeta_test,
            controls=controls,
            next_states=zeta_next,
            next_controls=controls_next,
            params_tensor=params_median
        )
        
        # Metrics
        max_residual = residuals.abs().max().item()
        mean_residual = residuals.abs().mean().item()
    
    print("\n" + "="*60)
    print("POLICY NETWORK VALIDATION")
    print("="*60)
    print(f"Maximum residual: {max_residual:.6e}")
    print(f"Mean residual: {mean_residual:.6e}")
    print(f"N range: [{controls[:, 0].min().item():.4f}, {controls[:, 0].max().item():.4f}]")
    print(f"Π range: [{controls[:, 1].min().item():.4f}, {controls[:, 1].max().item():.4f}]")
    print("="*60)
    
    return {
        'max_residual': max_residual,
        'mean_residual': mean_residual,
        'N_range': (controls[:, 0].min().item(), controls[:, 0].max().item()),
        'Pi_range': (controls[:, 1].min().item(), controls[:, 1].max().item())
    }


# ============================================================================
# TASK 11: IMPLEMENT STANDARD BOOTSTRAP PARTICLE FILTER
# ============================================================================

class ParticleFilter:
    """
    Bootstrap particle filter for nonlinear state-space models
    
    Reference: Herbst & Schorfheide (2015)
    """
    
    def __init__(self, model, policy_nn, n_particles=100, device='cpu'):
        self.model = model
        self.policy_nn = policy_nn.to(device)
        self.n_particles = n_particles
        self.device = device
    
    def initialize_particles(self, params):
        """Initialize particles from stationary distribution"""
        stationary_var = params.sigma_zeta**2 / (1 - params.rho_zeta**2)
        particles = torch.randn(self.n_particles, device=self.device) * np.sqrt(stationary_var)
        weights = torch.ones(self.n_particles, device=self.device) / self.n_particles
        
        return particles, weights
    
    def propagate_particles(self, particles, params):
        """Propagate particles one step forward"""
        innovations = torch.randn(self.n_particles, device=self.device) * params.sigma_zeta
        particles_new = params.rho_zeta * particles + innovations
        
        return particles_new
    
    def observation_density(self, particles, observation, params, 
                           measurement_error_std=0.01):
        """
        Compute p(y_t | ζ_t, θ)
        
        Args:
            particles: (n_particles,) current states
            observation: (3,) = [Y_t, Π_t, R_t]
            params: RANKParameters
            measurement_error_std: Std dev of measurement error
        """
        n = self.n_particles
        
        # Prepare inputs for policy network
        zeta = particles.unsqueeze(1)  # (n_particles, 1)
        params_tensor = torch.tensor([[
            params.theta_pi,
            params.theta_y,
            params.phi,
            params.rho_zeta,
            params.sigma_zeta
        ]], dtype=torch.float32, device=self.device).repeat(n, 1)
        
        inputs = torch.cat([zeta, params_tensor], dim=1)
        
        # Get model predictions
        with torch.no_grad():
            controls = self.policy_nn(inputs)  # (n_particles, 2)
            
            N_pred = controls[:, 0]
            Pi_pred = controls[:, 1]
            
            # Compute R from Taylor rule
            R_ss = self.model.steady_state['R']
            Y_ss = self.model.steady_state['Y']
            Y_pred = N_pred
            
            R_notional = R_ss * (Pi_pred / params.Pi_target)**params.theta_pi * \
                        (Y_pred / Y_ss)**params.theta_y
            R_pred = torch.maximum(R_notional, torch.ones_like(R_notional))
            
            # Stack predictions
            predictions = torch.stack([Y_pred, Pi_pred, R_pred], dim=1)
        
        # Compute likelihood under measurement error
        observation = observation.to(self.device)
        errors = observation.unsqueeze(0) - predictions  # (n_particles, 3)
        
        # Multivariate normal likelihood
        log_densities = -0.5 * ((errors / measurement_error_std)**2).sum(dim=1) - \
                       1.5 * np.log(2 * np.pi * measurement_error_std**2)
        
        densities = torch.exp(log_densities)
        
        return densities
    
    def resample(self, particles, weights):
        """Systematic resampling"""
        n = self.n_particles
        
        # Cumulative sum of weights
        cumsum = torch.cumsum(weights, dim=0)
        
        # Systematic resampling
        u = torch.rand(1, device=self.device) / n
        indices = torch.searchsorted(cumsum, u + torch.arange(n, dtype=torch.float32, 
                                                               device=self.device) / n)
        
        particles_resampled = particles[indices]
        weights_resampled = torch.ones(n, device=self.device) / n
        
        return particles_resampled, weights_resampled
    
    def filter(self, data, params):
        """
        Run particle filter on full dataset
        
        Returns:
            log_likelihood: Log-likelihood value
            filtered_states: (T, n_particles) filtered particles
        """
        T = len(data['Y'])
        
        # Initialize
        particles, weights = self.initialize_particles(params)
        
        log_likelihood = 0.0
        filtered_states = []
        
        for t in range(T):
            # Current observation
            obs = torch.tensor([data['Y'][t], data['Pi'][t], data['R'][t]], 
                             dtype=torch.float32)
            
            # Propagate
            particles = self.propagate_particles(particles, params)
            
            # Weight update
            likelihoods = self.observation_density(particles, obs, params)
            weights = weights * likelihoods
            
            # Normalize weights
            weight_sum = weights.sum()
            if weight_sum > 0:
                weights = weights / weight_sum
                log_likelihood += torch.log(weight_sum)
            else:
                # Numerical issue - reinitialize
                particles, weights = self.initialize_particles(params)
                weight_sum = weights.sum()
                weights = weights / weight_sum
            
            # Resample if effective sample size is low
            ess = 1.0 / (weights**2).sum()
            if ess < self.n_particles / 2:
                particles, weights = self.resample(particles, weights)
            
            filtered_states.append(particles.clone())
        
        return log_likelihood.item(), torch.stack(filtered_states)


# ============================================================================
# TASK 12: TEST PARTICLE FILTER ON SIMULATED DATA
# ============================================================================

def test_particle_filter():
    """Test particle filter on simulated data"""
    print("\n" + "="*60)
    print("TESTING PARTICLE FILTER")
    print("="*60)
    
    # True parameters
    true_params = RANKParameters(
        theta_pi=2.0,
        theta_y=0.5,
        phi=100.0,
        rho_zeta=0.7,
        sigma_zeta=0.02
    )
    
    # Initialize model
    model = RANKModel(true_params)
    
    # Load trained policy network
    policy_nn = PolicyNetwork()
    if os.path.exists('results/policy_nn.pt'):
        policy_nn.load_state_dict(torch.load('results/policy_nn.pt'))
        print("Loaded trained policy network")
    else:
        print("WARNING: Using untrained policy network!")
    
    # Simulate data
    T = 100
    data = model.simulate_data(T, true_params, policy_nn, seed=42)
    
    # Run particle filter
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    pf = ParticleFilter(model, policy_nn, n_particles=100, device=device)
    
    print("Running particle filter...")
    log_lik, filtered_states = pf.filter(data, true_params)
    
    print(f"Log-likelihood: {log_lik:.2f}")
    print(f"Likelihood per observation: {log_lik/T:.2f}")
    print("="*60)
    
    return log_lik


# ============================================================================
# TASK 13: GENERATE TRAINING DATA FOR NN PARTICLE FILTER
# ============================================================================

def generate_pf_training_data(
    model,
    policy_nn,
    n_samples=10000,
    n_particles=100,
    T=100,
    seed=None,
    save_path='data/pf_training_data.pt',
    device='cpu'
):
    """
    Generate training data for NN particle filter
    
    For each parameter draw:
    1. Simulate data
    2. Run standard particle filter
    3. Record (params, log_likelihood)
    """
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
    
    # Quasi-random parameter samples
    bounds = RANKParameters.get_bounds()
    param_samples, param_names = quasi_random_sample(bounds, n_samples, seed=seed)
    
    log_likelihoods = []
    
    # Initialize particle filter
    pf = ParticleFilter(model, policy_nn, n_particles=n_particles, device=device)
    
    print("\n" + "="*60)
    print("GENERATING PF TRAINING DATA")
    print("="*60)
    print(f"Number of samples: {n_samples}")
    print(f"Particles per sample: {n_particles}")
    print(f"Time periods: {T}")
    
    for i in tqdm(range(n_samples), desc='Generating data'):
        # Create parameters
        params = RANKParameters(
            theta_pi=param_samples[i, 0].item(),
            theta_y=param_samples[i, 1].item(),
            phi=param_samples[i, 2].item(),
            rho_zeta=param_samples[i, 3].item(),
            sigma_zeta=param_samples[i, 4].item()
        )
        
        # Simulate data
        data = model.simulate_data(T, params, policy_nn, 
                                   seed=seed+i if seed else None)
        
        # Run particle filter
        try:
            log_lik, _ = pf.filter(data, params)
            log_likelihoods.append(log_lik)
        except Exception as e:
            print(f"\nWarning: PF failed for sample {i}: {e}")
            log_likelihoods.append(-1e10)  # Very low likelihood
    
    log_likelihoods = torch.tensor(log_likelihoods, dtype=torch.float32)
    
    # Save
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save({
        'param_samples': param_samples,
        'log_likelihoods': log_likelihoods,
        'param_names': param_names
    }, save_path)
    
    print(f"\nTraining data saved to {save_path}")
    print(f"Log-likelihood range: [{log_likelihoods.min():.2f}, {log_likelihoods.max():.2f}]")
    print("="*60)
    
    return param_samples, log_likelihoods


# ============================================================================
# TASK 14: TRAIN NN PARTICLE FILTER
# ============================================================================

class ParticleFilterNN(nn.Module):
    """
    Neural network to approximate likelihood function
    
    Architecture:
    - Input: (batch, 5) = [θ_π, θ_y, φ, ρ_ζ, σ_ζ]
    - 4 hidden layers, 128 neurons each
    - Activation: CELU
    - Output: (batch, 1) = log-likelihood
    """
    
    def __init__(self, n_inputs=5, hidden_size=128, n_layers=4):
        super().__init__()
        
        layers = []
        layers.append(nn.Linear(n_inputs, hidden_size))
        layers.append(nn.CELU())
        
        for _ in range(n_layers - 1):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.CELU())
        
        layers.append(nn.Linear(hidden_size, 1))
        
        self.network = nn.Sequential(*layers)
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
    
    def forward(self, params):
        return self.network(params)


def train_pf_network(
    param_samples,
    log_likelihoods,
    pf_nn=None,
    n_epochs=20000,
    batch_size=100,
    train_split=0.8,
    lr_initial=2e-4,
    lr_final=1e-7,
    device='cuda' if torch.cuda.is_available() else 'cpu',
    log_dir='runs/rank_pf',
    save_path='results/pf_nn.pt'
):
    """Train NN particle filter"""
    
    # Train/validation split
    n_train = int(len(param_samples) * train_split)
    
    train_params = param_samples[:n_train]
    train_liks = log_likelihoods[:n_train].unsqueeze(1)
    
    val_params = param_samples[n_train:]
    val_liks = log_likelihoods[n_train:].unsqueeze(1)
    
    # Create data loaders
    train_dataset = TensorDataset(train_params, train_liks)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # Initialize network
    if pf_nn is None:
        pf_nn = ParticleFilterNN()
    pf_nn = pf_nn.to(device)
    
    # Optimizer
    optimizer = optim.AdamW(pf_nn.parameters(), lr=lr_initial)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=n_epochs, eta_min=lr_final
    )
    
    # TensorBoard
    writer = SummaryWriter(log_dir)
    
    # Training history
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    
    print("\n" + "="*60)
    print("TRAINING NN PARTICLE FILTER")
    print("="*60)
    
    # Training loop
    pbar = tqdm(range(n_epochs), desc='Training PF NN')
    for epoch in pbar:
        # Training
        pf_nn.train()
        epoch_loss = 0.0
        
        for batch_params, batch_liks in train_loader:
            batch_params = batch_params.to(device)
            batch_liks = batch_liks.to(device)
            
            # Forward
            pred_liks = pf_nn(batch_params)
            loss = nn.MSELoss()(pred_liks, batch_liks)
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        scheduler.step()
        
        # Validation
        pf_nn.eval()
        with torch.no_grad():
            val_params_device = val_params.to(device)
            val_liks_device = val_liks.to(device)
            val_pred = pf_nn(val_params_device)
            val_loss = nn.MSELoss()(val_pred, val_liks_device).item()
        
        # Logging
        avg_train_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        val_losses.append(val_loss)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(pf_nn.state_dict(), save_path)
        
        pbar.set_postfix({
            'train': f'{avg_train_loss:.4f}',
            'val': f'{val_loss:.4f}',
            'best': f'{best_val_loss:.4f}'
        })
        
        if epoch % 100 == 0:
            writer.add_scalar('Loss/train', avg_train_loss, epoch)
            writer.add_scalar('Loss/val', val_loss, epoch)
            writer.add_scalar('LR', scheduler.get_last_lr()[0], epoch)
    
    writer.close()
    
    print(f"\nBest validation loss: {best_val_loss:.6f}")
    print(f"Model saved to {save_path}")
    print("="*60)
    
    return pf_nn, {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'best_val_loss': best_val_loss
    }


# ============================================================================
# TASK 15: IMPLEMENT PRIOR DISTRIBUTIONS
# ============================================================================

class TruncatedNormalPrior:
    """Truncated normal prior distribution"""
    
    def __init__(self, mean, std, lower, upper):
        self.mean = mean
        self.std = std
        self.lower = lower
        self.upper = upper
        
        # Scipy truncnorm uses standardized bounds
        self.a = (lower - mean) / std
        self.b = (upper - mean) / std
        self.dist = truncnorm(self.a, self.b, loc=mean, scale=std)
    
    def pdf(self, x):
        return self.dist.pdf(x)
    
    def logpdf(self, x):
        return self.dist.logpdf(x)
    
    def sample(self, n=1):
        return self.dist.rvs(size=n)


class RANKPriors:
    """Prior distributions for RANK model parameters"""
    
    def __init__(self):
        # These values match the paper's setup
        self.priors = {
            'theta_pi': TruncatedNormalPrior(mean=2.0, std=0.2, lower=1.5, upper=3.0),
            'theta_y': TruncatedNormalPrior(mean=0.5, std=0.1, lower=0.1, upper=1.0),
            'phi': TruncatedNormalPrior(mean=100.0, std=20.0, lower=50.0, upper=200.0),
            'rho_zeta': TruncatedNormalPrior(mean=0.7, std=0.1, lower=0.5, upper=0.95),
            'sigma_zeta': TruncatedNormalPrior(mean=0.02, std=0.005, lower=0.01, upper=0.05)
        }
        self.param_names = list(self.priors.keys())
    
    def log_prior(self, params_dict):
        """Evaluate log prior density"""
        log_p = 0.0
        for name, value in params_dict.items():
            log_p += self.priors[name].logpdf(value)
        return log_p
    
    def sample_prior(self, n=1):
        """Sample from prior"""
        samples = np.zeros((n, len(self.param_names)))
        for i, name in enumerate(self.param_names):
            samples[:, i] = self.priors[name].sample(n)
        return samples


# ============================================================================
# TASK 16-18: IMPLEMENT RWMH ALGORITHM
# ============================================================================

class RWMH:
    """
    Random Walk Metropolis-Hastings algorithm
    Following Appendix A of the paper
    """
    
    def __init__(self, pf_nn, priors, device='cpu'):
        self.pf_nn = pf_nn.to(device)
        self.priors = priors
        self.device = device
        self.pf_nn.eval()
    
    def log_posterior(self, params_array):
        """Evaluate log posterior: log p(θ|Y) = log p(Y|θ) + log p(θ)"""
        # Convert to dict
        params_dict = {name: params_array[i] 
                      for i, name in enumerate(self.priors.param_names)}
        
        # Log prior
        log_prior = self.priors.log_prior(params_dict)
        
        # Log likelihood (from NN)
        params_tensor = torch.tensor(params_array, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            log_lik = self.pf_nn(params_tensor).item()
        
        return log_lik + log_prior
    
    def initialize(self, n_init=1000):
        """Step 1: Initialize proposal density"""
        print("\n" + "="*60)
        print("RWMH INITIALIZATION")
        print("="*60)
        
        # Draw from prior
        candidates = self.priors.sample_prior(n_init)
        log_posts = np.array([self.log_posterior(theta) 
                             for theta in tqdm(candidates, desc='Evaluating prior draws')])
        
        # Find mode
        mode_idx = np.argmax(log_posts)
        theta_mode = candidates[mode_idx]
        
        # Top 10% for covariance
        threshold = np.percentile(log_posts, 90)
        good_draws = candidates[log_posts > threshold]
        
        # Covariance matrix
        proposal_cov = np.cov(good_draws.T)
        
        print(f"Mode found: {theta_mode}")
        print(f"Log posterior at mode: {log_posts[mode_idx]:.2f}")
        print("="*60)
        
        return theta_mode, proposal_cov
    
    def run_burn_in(self, theta_init, proposal_cov, n_burn=10000, 
                   c_scale=1.0, target_accept=(0.2, 0.4)):
        """Step 2: Burn-in phase"""
        print("\n" + "="*60)
        print("RWMH BURN-IN")
        print("="*60)
        
        theta_current = theta_init.copy()
        log_post_current = self.log_posterior(theta_current)
        
        burn_in_draws = []
        accepts = 0
        
        for _ in tqdm(range(n_burn), desc='Burn-in'):
            # Propose
            theta_prop = np.random.multivariate_normal(theta_current, c_scale * proposal_cov)
            
            # Check bounds
            in_bounds = all(
                self.priors.priors[name].lower <= theta_prop[i] <= self.priors.priors[name].upper
                for i, name in enumerate(self.priors.param_names)
            )
            
            if not in_bounds:
                burn_in_draws.append(theta_current)
                continue
            
            # Evaluate log posterior
            log_post_prop = self.log_posterior(theta_prop)
            
            # Acceptance probability
            log_alpha = log_post_prop - log_post_current
            
            if np.log(np.random.rand()) < log_alpha:
                theta_current = theta_prop
                log_post_current = log_post_prop
                accepts += 1
            
            burn_in_draws.append(theta_current.copy())
        
        acceptance_rate = accepts / n_burn
        print(f"Burn-in acceptance rate: {acceptance_rate:.2%}")
        
        # Update covariance
        burn_in_draws = np.array(burn_in_draws)
        recent_draws = burn_in_draws[int(0.25 * n_burn):]
        proposal_cov_updated = np.cov(recent_draws.T)
        
        # Adjust c_scale
        if acceptance_rate < target_accept[0]:
            c_scale *= 0.8
            print(f"Reducing c_scale to {c_scale:.3f}")
        elif acceptance_rate > target_accept[1]:
            c_scale *= 1.2
            print(f"Increasing c_scale to {c_scale:.3f}")
        
        print("="*60)
        
        return theta_current, proposal_cov_updated, c_scale
    
    def run_sampling(self, theta_init, proposal_cov, c_scale, n_draws=50000):
        """Step 3: Main sampling"""
        print("\n" + "="*60)
        print("RWMH MAIN SAMPLING")
        print("="*60)
        
        theta_current = theta_init.copy()
        log_post_current = self.log_posterior(theta_current)
        
        draws = np.zeros((n_draws, len(theta_current)))
        log_posts = np.zeros(n_draws)
        accepts = 0
        
        for i in tqdm(range(n_draws), desc='Sampling'):
            # Propose
            theta_prop = np.random.multivariate_normal(theta_current, c_scale * proposal_cov)
            
            # Check bounds
            in_bounds = all(
                self.priors.priors[name].lower <= theta_prop[j] <= self.priors.priors[name].upper
                for j, name in enumerate(self.priors.param_names)
            )
            
            if not in_bounds:
                draws[i] = theta_current
                log_posts[i] = log_post_current
                continue
            
            # Evaluate
            log_post_prop = self.log_posterior(theta_prop)
            
            # Accept/reject
            log_alpha = log_post_prop - log_post_current
            
            if np.log(np.random.rand()) < log_alpha:
                theta_current = theta_prop
                log_post_current = log_post_prop
                accepts += 1
            
            draws[i] = theta_current
            log_posts[i] = log_post_current
        
        acceptance_rate = accepts / n_draws
        print(f"Final acceptance rate: {acceptance_rate:.2%}")
        print("="*60)
        
        return draws, log_posts, acceptance_rate
    
    def estimate(self, n_init=1000, n_burn=10000, n_draws=50000,
                save_path='results/posterior_results.npz'):
        """Full RWMH estimation procedure"""
        
        # Step 1: Initialize
        theta_mode, proposal_cov = self.initialize(n_init)
        
        # Step 2: Burn-in
        theta_start, proposal_cov, c_scale = self.run_burn_in(
            theta_mode, proposal_cov, n_burn
        )
        
        # Step 3: Sample
        draws, log_posts, acceptance_rate = self.run_sampling(
            theta_start, proposal_cov, c_scale, n_draws
        )
        
        # Save results
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        np.savez(save_path,
                draws=draws,
                log_posts=log_posts,
                acceptance_rate=acceptance_rate,
                param_names=self.priors.param_names)
        
        print(f"\nResults saved to {save_path}")
        
        return {
            'draws': draws,
            'log_posts': log_posts,
            'acceptance_rate': acceptance_rate,
            'param_names': self.priors.param_names
        }


# ============================================================================
# TASK 18, 20, 21: PLOTTING AND ANALYSIS FUNCTIONS
# ============================================================================

def plot_posterior_diagnostics(results, true_params=None, save_path=None):
    """Create diagnostic plots for RWMH output"""
    draws = results['draws']
    param_names = results['param_names']
    n_params = len(param_names)
    
    fig, axes = plt.subplots(n_params, 2, figsize=(12, 3*n_params))
    
    for i, name in enumerate(param_names):
        # Trace plot
        axes[i, 0].plot(draws[:, i], alpha=0.5, linewidth=0.5)
        axes[i, 0].set_title(f'Trace: {name}')
        axes[i, 0].set_xlabel('Iteration')
        
        if true_params is not None:
            true_val = getattr(true_params, name)
            axes[i, 0].axhline(true_val, color='red', linestyle='--', 
                             linewidth=2, label='True')
            axes[i, 0].legend()
        
        # Posterior histogram
        axes[i, 1].hist(draws[:, i], bins=50, density=True, alpha=0.6, 
                       edgecolor='black', color='steelblue')
        axes[i, 1].set_title(f'Posterior: {name}')
        axes[i, 1].set_xlabel(name)
        axes[i, 1].set_ylabel('Density')
        
        if true_params is not None:
            true_val = getattr(true_params, name)
            axes[i, 1].axvline(true_val, color='red', linestyle='--', 
                             linewidth=2, label='True')
            axes[i, 1].legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    return fig


def compute_posterior_moments(results):
    """Compute posterior statistics"""
    draws = results['draws']
    param_names = results['param_names']
    
    stats = []
    for i, name in enumerate(param_names):
        stats.append({
            'parameter': name,
            'mean': draws[:, i].mean(),
            'median': np.median(draws[:, i]),
            'std': draws[:, i].std(),
            'q05': np.percentile(draws[:, i], 5),
            'q95': np.percentile(draws[:, i], 95)
        })
    
    df = pd.DataFrame(stats)
    
    print("\n" + "="*60)
    print("POSTERIOR MOMENTS")
    print("="*60)
    print(df.to_string(index=False))
    print("="*60)
    
    return df


def plot_likelihood_surface(pf_nn, param_name, bounds, fixed_params, 
                           n_points=100, save_path=None, device='cpu'):
    """Plot likelihood as function of one parameter (Figure 5 replication)"""
    
    # Create grid
    param_grid = np.linspace(bounds[0], bounds[1], n_points)
    
    # Parameter index
    param_names = ['theta_pi', 'theta_y', 'phi', 'rho_zeta', 'sigma_zeta']
    param_idx = param_names.index(param_name)
    
    # Evaluate NN likelihood
    log_liks = []
    pf_nn = pf_nn.to(device)
    pf_nn.eval()
    
    for val in param_grid:
        params = fixed_params.copy()
        params[param_name] = val
        
        # Convert to tensor
        param_array = torch.tensor([[
            params['theta_pi'],
            params['theta_y'],
            params['phi'],
            params['rho_zeta'],
            params['sigma_zeta']
        ]], dtype=torch.float32, device=device)
        
        with torch.no_grad():
            log_lik = pf_nn(param_array).item()
        
        log_liks.append(log_lik)
    
    # Plot
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(param_grid, log_liks, 'b-', linewidth=2, label='Neural Network')
    ax.set_xlabel(param_name, fontsize=12)
    ax.set_ylabel('Log Likelihood', fontsize=12)
    ax.set_title(f'Log Likelihood conditioned on {param_name}', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    return fig


# ============================================================================
# TASK 17, 22: MAIN EXECUTION AND PIPELINE
# ============================================================================

def run_full_pipeline():
    """
    Execute complete Section 3.2 pipeline
    
    1. Train policy network
    2. Generate PF training data
    3. Train NN particle filter
    4. Generate simulated data
    5. Run RWMH estimation
    6. Analyze results
    """
    
    print("\n" + "="*70)
    print(" "*20 + "SECTION 3.2: RANK MODEL WITH ZLB")
    print("="*70)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Create directories
    os.makedirs('results', exist_ok=True)
    os.makedirs('data', exist_ok=True)
    os.makedirs('figures', exist_ok=True)
    
    # True parameters (for data generation)
    true_params = RANKParameters(
        theta_pi=2.0,
        theta_y=0.5,
        phi=100.0,
        rho_zeta=0.7,
        sigma_zeta=0.02
    )
    
    # Initialize model
    model = RANKModel(true_params)
    
    # ========================================================================
    # STEP 1: Train Policy Network
    # ========================================================================
    print("\n" + "="*70)
    print("STEP 1: TRAINING POLICY NETWORK")
    print("="*70)
    
    policy_nn = PolicyNetwork()
    
    if not os.path.exists('results/policy_nn.pt'):
        policy_nn, train_hist = train_policy_network(
            model=model,
            policy_nn=policy_nn,
            n_iterations=100000,
            device=device
        )
        
        # Validate
        validate_policy_network(model, policy_nn, device=device)
    else:
        print("Loading existing policy network...")
        policy_nn.load_state_dict(torch.load('results/policy_nn.pt'))
        validate_policy_network(model, policy_nn, device=device)
    
    # ========================================================================
    # STEP 2: Generate PF Training Data
    # ========================================================================
    print("\n" + "="*70)
    print("STEP 2: GENERATING PARTICLE FILTER TRAINING DATA")
    print("="*70)
    
    if not os.path.exists('data/pf_training_data.pt'):
        param_samples, log_likelihoods = generate_pf_training_data(
            model=model,
            policy_nn=policy_nn,
            n_samples=10000,
            n_particles=100,
            T=100,
            seed=42,
            device=device
        )
    else:
        print("Loading existing PF training data...")
        data_dict = torch.load('data/pf_training_data.pt')
        param_samples = data_dict['param_samples']
        log_likelihoods = data_dict['log_likelihoods']
        print(f"Loaded {len(param_samples)} samples")
    
    # ========================================================================
    # STEP 3: Train NN Particle Filter
    # ========================================================================
    print("\n" + "="*70)
    print("STEP 3: TRAINING NN PARTICLE FILTER")
    print("="*70)
    
    if not os.path.exists('results/pf_nn.pt'):
        pf_nn, train_hist = train_pf_network(
            param_samples=param_samples,
            log_likelihoods=log_likelihoods,
            n_epochs=20000,
            device=device
        )
    else:
        print("Loading existing NN particle filter...")
        pf_nn = ParticleFilterNN()
        pf_nn.load_state_dict(torch.load('results/pf_nn.pt'))
        print("Loaded successfully")
    
    # Plot likelihood surface
    bounds = RANKParameters.get_bounds()
    fixed_params = {
        'theta_pi': 2.0,
        'theta_y': 0.5,
        'phi': 100.0,
        'rho_zeta': 0.7,
        'sigma_zeta': 0.02
    }
    
    plot_likelihood_surface(
        pf_nn, 'theta_pi', bounds['theta_pi'], fixed_params,
        save_path='figures/likelihood_theta_pi.png',
        device=device
    )
    
    # ========================================================================
    # STEP 4: Generate Simulated Data for Estimation
    # ========================================================================
    print("\n" + "="*70)
    print("STEP 4: GENERATING SIMULATED DATA")
    print("="*70)
    
    T = 500  # 500 periods
    simulated_data = model.simulate_data(T, true_params, policy_nn, seed=12345)
    torch.save(simulated_data, 'data/simulated_data.pt')
    print(f"Generated {T} periods of data")
    
    # ========================================================================
    # STEP 5: Run RWMH Estimation
    # ========================================================================
    print("\n" + "="*70)
    print("STEP 5: BAYESIAN ESTIMATION (RWMH)")
    print("="*70)
    
    priors = RANKPriors()
    rwmh = RWMH(pf_nn, priors, device=device)
    
    if not os.path.exists('results/posterior_results.npz'):
        results = rwmh.estimate(
            n_init=1000,
            n_burn=10000,
            n_draws=50000
        )
    else:
        print("Loading existing posterior results...")
        data = np.load('results/posterior_results.npz')
        results = {
            'draws': data['draws'],
            'log_posts': data['log_posts'],
            'acceptance_rate': float(data['acceptance_rate']),
            'param_names': list(data['param_names'])
        }
        print(f"Loaded {len(results['draws'])} posterior draws")
    
    # ========================================================================
    # STEP 6: Analyze Results
    # ========================================================================
    print("\n" + "="*70)
    print("STEP 6: POSTERIOR ANALYSIS")
    print("="*70)
    
    # Compute moments
    moments_df = compute_posterior_moments(results)
    moments_df.to_csv('results/posterior_moments.csv', index=False)
    
    # Plot diagnostics
    plot_posterior_diagnostics(
        results, 
        true_params=true_params,
        save_path='figures/posterior_diagnostics.png'
    )
    
    # Compare to true values
    print("\n" + "="*70)
    print("COMPARISON TO TRUE VALUES")
    print("="*70)
    
    for i, name in enumerate(results['param_names']):
        true_val = getattr(true_params, name)
        median = np.median(results['draws'][:, i])
        q05 = np.percentile(results['draws'][:, i], 5)
        q95 = np.percentile(results['draws'][:, i], 95)
        
        in_interval = q05 <= true_val <= q95
        status = "✓" if in_interval else "✗"
        
        print(f"{name:12s}: True={true_val:.4f}, "
              f"Median={median:.4f}, "
              f"90% CI=[{q05:.4f}, {q95:.4f}] {status}")
    
    print("="*70)
    print("\n✓ SECTION 3.2 COMPLETE!")
    print("\nResults saved in:")
    print("  - results/policy_nn.pt")
    print("  - results/pf_nn.pt")
    print("  - results/posterior_results.npz")
    print("  - results/posterior_moments.csv")
    print("  - figures/posterior_diagnostics.png")
    print("  - figures/likelihood_theta_pi.png")
    print("="*70)


# ============================================================================
# COMMAND LINE INTERFACE
# ============================================================================

def main():
    """Main entry point with argument parsing"""
    parser = argparse.ArgumentParser(description='Section 3.2: RANK Model with ZLB')
    parser.add_argument('--task', type=str, default='all',
                       choices=['train_policy', 'generate_pf_data', 
                               'train_pf', 'estimate', 'all'],
                       help='Which task to run')
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cuda', 'cpu'],
                       help='Device to use')
    
    args = parser.parse_args()
    
    # Set device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    print(f"Using device: {device}")
    
    # Run requested task
    if args.task == 'all':
        run_full_pipeline()
    elif args.task == 'train_policy':
        model = RANKModel(RANKParameters())
        policy_nn = PolicyNetwork()
        train_policy_network(model, policy_nn, device=device)
    elif args.task == 'generate_pf_data':
        model = RANKModel(RANKParameters())
        policy_nn = PolicyNetwork()
        policy_nn.load_state_dict(torch.load('results/policy_nn.pt'))
        generate_pf_training_data(model, policy_nn, device=device)
    elif args.task == 'train_pf':
        data = torch.load('data/pf_training_data.pt')
        train_pf_network(data['param_samples'], data['log_likelihoods'], device=device)
    elif args.task == 'estimate':
        pf_nn = ParticleFilterNN()
        pf_nn.load_state_dict(torch.load('results/pf_nn.pt'))
        priors = RANKPriors()
        rwmh = RWMH(pf_nn, priors, device=device)
        rwmh.estimate()


if __name__ == '__main__':
    main()

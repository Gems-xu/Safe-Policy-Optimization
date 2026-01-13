# Copyright 2025 Gems Team. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""
PHS-MAPPO Actor: True Port-Hamiltonian System Embedded in MAPPO Actor.

This module implements the SafePinnPPO architecture from PHS-MAPPO.md, where
the action is directly computed from Port-Hamiltonian dynamics rather than
just using physics as additional features.

Core Innovation:
    Action is derived from PHS dynamics: u = F⁻¹(dx - (J_sys - R_sys) ∇H_sys)
    
    Where:
    - dx = (J - R) ∇H_total (learned PHS dynamics)
    - J: Learned skew-symmetric interconnection matrix (energy-conserving)
    - R: Learned positive semi-definite dissipation matrix (damping)
    - H_total = H_task + H_barrier + H_kin (combined potentials)
    - H_task = H_goal + H_task_learned (explicit goal attraction + learned component)

Key Differences from v2:
    1. Action is COMPUTED from physics, not just informed by physics features
    2. Multi-agent coupling via Laplacian matrix
    3. Learned stiffness k_ij for agent-agent barriers (SoftBarrierHead)
    4. Barrier warmup strategy for stable training
    5. Automatic differentiation for gradient computation

For SafetyMultiGoal environments:
    - Observation: ~152-dim (accelerometer, velocimeter, gyro, magnetometer, lidars)
    - Action: 2-dim (forward force, turning velocity)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def check(input):
    """Convert numpy array to torch tensor if needed."""
    output = torch.from_numpy(input) if type(input) == np.ndarray else input
    return output


# =============================================================================
# Attention-based Modules for Learning System Matrices
# =============================================================================

class AttentionLEMURS(nn.Module):
    """
    LEMURS-style Attention module for learning PHS system matrices.
    
    Used to learn:
    - J_mean: Interconnection matrix (skew-symmetric)
    - R_mean: Dissipation matrix (symmetric positive definite)
    - H_task: Task potential function
    """
    
    def __init__(self, input_dim, hidden_dim, output_dim, n_heads=4):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.n_heads = n_heads
        
        # Multi-head attention layers
        self.query = nn.Linear(input_dim, hidden_dim)
        self.key = nn.Linear(input_dim, hidden_dim)
        self.value = nn.Linear(input_dim, hidden_dim)
        
        # Output projection
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, output_dim)
        )
        
        # Layer normalization for stability
        self.norm = nn.LayerNorm(hidden_dim)
        
    def forward(self, x, adj=None):
        """
        Forward pass with optional adjacency masking.
        
        Args:
            x: [batch, n_agents, input_dim] or [batch, input_dim]
            adj: [batch, n_agents, n_agents] adjacency/Laplacian matrix (optional)
            
        Returns:
            output: [batch, n_agents, output_dim] or [batch, output_dim]
        """
        # Handle both batched and non-batched inputs
        if x.dim() == 2:
            x = x.unsqueeze(1)  # [batch, 1, input_dim]
            squeeze_output = True
        else:
            squeeze_output = False
            
        batch_size, n_agents, _ = x.shape
        
        # Compute Q, K, V
        Q = self.query(x)  # [batch, n_agents, hidden]
        K = self.key(x)    # [batch, n_agents, hidden]
        V = self.value(x)  # [batch, n_agents, hidden]
        
        # Scaled dot-product attention
        scale = np.sqrt(self.hidden_dim)
        attn = torch.bmm(Q, K.transpose(-1, -2)) / scale  # [batch, n_agents, n_agents]
        
        # Apply adjacency mask if provided
        if adj is not None:
            # Mask attention scores where adjacency is 0
            attn = attn * adj + (1 - adj) * (-1e9)
        
        attn = F.softmax(attn, dim=-1)
        
        # Apply attention to values
        context = torch.bmm(attn, V)  # [batch, n_agents, hidden]
        context = self.norm(context)
        
        # Project to output dimension
        output = self.proj(context)  # [batch, n_agents, output_dim]
        
        if squeeze_output:
            output = output.squeeze(1)
            
        return output


class SoftBarrierHead(nn.Module):
    """
    Learnable barrier stiffness network for agent-agent collision avoidance.
    
    Computes pairwise stiffness coefficients k_ij based on agent states,
    which determines the strength of the repulsive barrier potential.
    
    Key Features:
    - Pairwise expansion: computes k_ij for all (i,j) agent pairs
    - Learnable smoothness parameter
    - Adjacency masking: only nearby agents interact
    """
    
    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()
        
        # Shared state encoder
        self.mlp_shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ELU()
        )
        
        # Pairwise stiffness predictor (concatenated features)
        self.mlp_k = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ELU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # Learnable smoothness parameter
        self.log_smoothness = nn.Parameter(torch.tensor(0.0))
        
    def forward(self, x, adj):
        """
        Compute pairwise barrier stiffness.
        
        Args:
            x: [batch, n_agents, input_dim] agent states
            adj: [batch, n_agents, n_agents] adjacency matrix (Laplacian)
            
        Returns:
            k_ij: [batch, n_agents, n_agents] pairwise stiffness coefficients
        """
        batch_size, n_agents, _ = x.shape
        
        # Encode agent states
        z = self.mlp_shared(x)  # [batch, n_agents, hidden/2]
        
        # Pairwise expansion: create all (i,j) combinations
        z_i = z.unsqueeze(2).expand(-1, -1, n_agents, -1)  # [batch, n, n, hidden/2]
        z_j = z.unsqueeze(1).expand(-1, n_agents, -1, -1)  # [batch, n, n, hidden/2]
        z_combined = torch.cat([z_i, z_j], dim=-1)  # [batch, n, n, hidden]
        
        # Predict raw stiffness
        k_ij_raw = self.mlp_k(z_combined).squeeze(-1)  # [batch, n, n]
        k_ij_raw = torch.clamp(k_ij_raw, min=-10, max=10)
        
        # Apply learnable smoothness
        smoothness = F.softplus(self.log_smoothness) + 0.1
        k_ij = F.softplus(k_ij_raw) * smoothness
        k_ij = torch.clamp(k_ij, min=0, max=10)
        
        # Mask with adjacency (only nearby agents interact)
        k_ij = k_ij * adj
        
        return k_ij


# =============================================================================
# PHS-MAPPO Actor: True Port-Hamiltonian Embedded Actor
# =============================================================================

class PHSMAPPOActor(nn.Module):
    """
    Port-Hamiltonian System embedded MAPPO Actor.
    
    This implements the true PHS framework from PHS-MAPPO.md where actions
    are computed directly from physical dynamics rather than just using
    physics as features.
    
    Key Architecture:
        1. Potential Functions:
           - H_goal: Explicit quadratic goal attraction (from goal lidar)
           - H_task_learned: Learned task-specific potential (neural network)
           - H_task = H_goal + H_task_learned (combined task potential)
           - H_barrier: Log barrier for collision avoidance
           - H_kin: Kinetic energy penalty
           
        2. System Matrices:
           - J: Skew-symmetric interconnection (learned via attention)
           - R: Symmetric PSD dissipation (learned via attention)
           
        3. Action Computation:
           dx = (J - R) ∇H_total
           u_mean = F⁻¹(dx - (J_sys - R_sys) ∇H_sys)
    
    For Point/Car agents in SafetyMultiGoal:
        - Observation: ~152-dim
        - State: (q_pos, q_vel) = 4-dim
        - Action: 2-dim (forward force, turning)
    """
    
    def __init__(self, config, obs_space, act_space, device=torch.device("cpu"), n_agents=1):
        super().__init__()
        
        self.config = config
        self.device = device
        self.obs_dim = obs_space.shape[0]
        self.act_dim = act_space.shape[0]
        self.n_agents = n_agents
        
        # ===================
        # PHS Configuration
        # ===================
        self.hidden_size = config.get("hidden_size", 256)
        self.physics_hidden = config.get("physics_hidden", 128)
        self.state_dim = 4  # (x, y, vx, vy) or (vx, vy, ax, ay)
        
        # Physical parameters
        self.f_max = config.get("phs_f_max", 0.8)  # Max control force
        self.drag = config.get("phs_drag", 0.25)  # Base damping
        self.dt = config.get("phs_dt", 0.05)  # Time step
        
        # Barrier potential parameters
        self.r_collision = config.get("r_collision", 0.17)  # Collision radius
        self.r_communication = config.get("r_communication", 0.45)  # Communication radius
        self.barrier_epsilon = config.get("barrier_epsilon", 0.06)  # Numerical stability
        
        # Potential weights
        self.task_weight = config.get("task_weight", 1.3)
        self.barrier_weight = config.get("barrier_weight", 0.12)
        self.barrier_weight_max = config.get("barrier_weight_max", 0.20)
        self.obstacle_barrier_weight = config.get("obstacle_barrier_weight", 0.45)
        
        # Barrier warmup parameters
        self.barrier_warmup_steps = config.get("barrier_warmup_steps", 200)
        self.barrier_decay_start = config.get("barrier_decay_start", 400)
        self.barrier_decay_rate = config.get("barrier_decay_rate", 0.50)
        self._training_step = 0
        
        # Multi-agent scaling
        self.auto_scale_by_agents = config.get("auto_scale_by_agents", True)
        
        # Observation indices (Point agent)
        # obs[0:3] = accelerometer (ax, ay, az)
        # obs[3:6] = velocimeter (vx, vy, vz)
        # obs[6:9] = gyro
        # obs[9:12] = magnetometer
        # obs[12:28] = goal_red lidar
        # obs[28:44] = goal_blue lidar
        # obs[44:60] = hazard lidar
        # obs[60:76] = vases lidar
        # obs[76:92] = other_agent lidar
        self.vel_indices = [3, 4]
        self.acc_indices = [0, 1]
        self.goal_lidar_start = 12
        self.goal_lidar_end = 44
        self.hazard_lidar_start = 44
        self.hazard_lidar_end = 60
        self.agent_lidar_start = 76
        self.agent_lidar_end = 92
        
        # ===================
        # Network Modules
        # ===================
        
        # Feature extractor
        self.feature_norm = nn.LayerNorm(self.obs_dim)
        self.state_encoder = nn.Sequential(
            nn.Linear(self.obs_dim, self.hidden_size),
            nn.ELU(),
            nn.LayerNorm(self.hidden_size),
            nn.Linear(self.hidden_size, self.physics_hidden),
            nn.ELU(),
        )
        
        # Learned system matrices
        self.J_net = AttentionLEMURS(self.physics_hidden, self.physics_hidden, self.state_dim * self.state_dim)
        self.R_net = AttentionLEMURS(self.physics_hidden, self.physics_hidden, self.state_dim * self.state_dim)
        
        # Task potential network
        self.H_task_net = AttentionLEMURS(self.physics_hidden, self.physics_hidden, 1)
        
        # Barrier stiffness for agent-agent interaction
        self.H_barrier_head = SoftBarrierHead(self.physics_hidden, hidden_dim=64)
        
        # Obstacle barrier stiffness network
        self.obstacle_k_net = nn.Sequential(
            nn.Linear(self.obs_dim, self.physics_hidden // 2),
            nn.ELU(),
            nn.Linear(self.physics_hidden // 2, 1),
            nn.Softplus()
        )
        
        # Action std network
        self.std_net = nn.Sequential(
            nn.Linear(self.physics_hidden + self.act_dim, self.physics_hidden // 2),
            nn.ELU(),
            nn.Linear(self.physics_hidden // 2, self.act_dim)
        )
        
        # Precompute base system matrices (Port-Hamiltonian structure)
        self._init_base_phs_matrices()
        
        # Initialize weights
        self._init_weights()
        
        self.to(device)
        
    def _init_base_phs_matrices(self):
        """Initialize base PHS system matrices with standard Hamiltonian structure."""
        dim = self.state_dim // 2  # Position or velocity dimension (2 for 2D)
        
        # Standard Hamiltonian J matrix: [[0, I], [-I, 0]]
        J_sys = torch.zeros(self.state_dim, self.state_dim)
        J_sys[:dim, dim:] = torch.eye(dim)
        J_sys[dim:, :dim] = -torch.eye(dim)
        self.register_buffer('J_sys', J_sys)
        
        # Standard dissipation R matrix: [[0, 0], [0, drag*I]]
        R_sys = torch.zeros(self.state_dim, self.state_dim)
        R_sys[dim:, dim:] = self.drag * torch.eye(dim)
        self.register_buffer('R_sys', R_sys)
        
        # Control input matrix F = [[0], [I]]
        F_sys = torch.zeros(self.state_dim, self.act_dim)
        F_sys[dim:, :] = torch.eye(self.act_dim)
        self.register_buffer('F_sys', F_sys)
        
        # Pseudo-inverse of F for control computation
        F_pinv = torch.pinverse(F_sys)
        self.register_buffer('F_pinv', F_pinv)
        
    def _init_weights(self):
        """Initialize network weights using orthogonal initialization."""
        def init_layer(m, gain=np.sqrt(2)):
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=gain)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        
        for m in self.state_encoder:
            init_layer(m)
            
        for m in self.obstacle_k_net:
            if isinstance(m, nn.Linear):
                init_layer(m, gain=0.1)
        
        for m in self.std_net:
            init_layer(m, gain=0.1)
    
    def _get_current_barrier_weight(self):
        """
        Compute current barrier weight based on training step.
        
        Implements warmup → plateau → decay schedule:
        - Phase 1 [0, warmup]: 0 → barrier_weight_max
        - Phase 2 [warmup, decay_start]: barrier_weight_max (plateau)
        - Phase 3 [decay_start, ∞]: decay to barrier_weight * decay_rate
        """
        step = self._training_step
        
        if step < self.barrier_warmup_steps:
            # Warmup phase: linear increase
            progress = step / self.barrier_warmup_steps
            return progress * self.barrier_weight_max
        elif step < self.barrier_decay_start:
            # Plateau phase
            return self.barrier_weight_max
        else:
            # Decay phase: exponential decay
            decay_steps = step - self.barrier_decay_start
            decay_factor = np.exp(-decay_steps / 500.0)  # Smooth decay over ~500 steps
            target = self.barrier_weight * self.barrier_decay_rate
            return target + (self.barrier_weight_max - target) * decay_factor
    
    def _extract_physics_state(self, obs):
        """
        Extract physics state from observation.
        
        Returns state = [vx, vy, ax, ay] for compatibility with PHS dynamics.
        """
        vel = obs[..., self.vel_indices]  # [batch, 2] or [batch, n_agents, 2]
        acc = obs[..., self.acc_indices]  # [batch, 2] or [batch, n_agents, 2]
        state = torch.cat([vel, acc], dim=-1)  # [batch, 4] or [batch, n_agents, 4]
        return state
    
    def _compute_laplacian_matrix(self, obs):
        """
        Compute soft Laplacian/adjacency matrix based on agent positions.
        
        Uses observation to infer relative positions and compute adjacency.
        Agents within r_communication have strong coupling.
        """
        batch_size = obs.shape[0]
        
        # For single-agent case, return identity (derived from obs to support gradient)
        if obs.dim() == 2 or obs.shape[1] == 1:
            zero_tensor = obs[..., :1].sum(dim=-1, keepdim=True) * 0.0
            return zero_tensor.unsqueeze(-1) + 1.0  # Shape: [batch, 1, 1]
        
        n_agents = obs.shape[1]
        
        # Extract agent lidar for proximity estimation
        # Higher lidar reading = closer agent
        agent_lidar_end = min(self.agent_lidar_end, obs.shape[-1])
        if self.agent_lidar_start < agent_lidar_end:
            agent_lidar = obs[..., self.agent_lidar_start:agent_lidar_end]  # [batch, n_agents, 16]
            # Estimate proximity to each direction
            proximity = agent_lidar.max(dim=-1)[0]  # [batch, n_agents]
        else:
            # Create zeros derived from obs to maintain gradient chain
            proximity = obs[..., :1].expand_as(obs[..., :n_agents]) * 0.0
        
        # Create pairwise proximity matrix (symmetric approximation)
        # L_ij = sigmoid(proximity_i + proximity_j)
        prox_i = proximity.unsqueeze(2)  # [batch, n_agents, 1]
        prox_j = proximity.unsqueeze(1)  # [batch, 1, n_agents]
        
        # Soft adjacency: high when both agents see each other
        adjacency = torch.sigmoid(2.0 * (prox_i + prox_j - 0.5))  # [batch, n_agents, n_agents]
        
        # Zero out diagonal (self-interaction)
        eye = torch.eye(n_agents, device=self.device).unsqueeze(0)
        adjacency = adjacency * (1 - eye)
        
        return adjacency
    
    def _compute_goal_potential(self, obs):
        """
        Compute explicit goal attraction potential H_goal from goal lidar.
        
        Uses goal lidar to estimate goal direction and distance.
        H_goal = 0.5 * k * distance_to_goal^2 (quadratic spring)
        
        This provides explicit guidance toward the goal.
        """
        # Extract goal lidar
        goal_end = min(self.goal_lidar_end, obs.shape[-1])
        goal_lidar = obs[..., self.goal_lidar_start:goal_end]  # [batch, ..., 32]
        
        # Max goal proximity (higher = closer to goal)
        goal_proximity = goal_lidar.max(dim=-1, keepdim=True)[0]  # [batch, ..., 1]
        
        # Convert proximity to effective distance (inverse relationship)
        # When proximity=1, distance≈0; when proximity=0, distance≈1
        goal_distance = 1.0 - goal_proximity
        
        # Quadratic potential: lower when close to goal
        k_goal = 10.0
        H_goal = 0.5 * k_goal * goal_distance.pow(2)
        
        return H_goal
    
    def _compute_task_potential_learned(self, state_features):
        """
        Compute learned task potential H_task_learned.
        
        Uses neural network to learn task-specific potential field
        beyond the explicit goal attraction.
        """
        H_task_learned = self.H_task_net(state_features)
        return H_task_learned
    
    def _compute_obstacle_barrier(self, obs):
        """
        Compute obstacle barrier potential using log barrier.
        
        H_barrier = -k * log((d - r_collision) / r_collision)
        
        Where d is estimated from hazard lidar readings.
        """
        # Extract hazard lidar
        hazard_end = min(self.hazard_lidar_end, obs.shape[-1])
        hazard_lidar = obs[..., self.hazard_lidar_start:hazard_end]  # [batch, ..., 16]
        
        # Max proximity (higher = closer = more dangerous)
        max_proximity = hazard_lidar.max(dim=-1, keepdim=True)[0]
        
        # Convert to effective distance (0 = collision, 1 = safe)
        # proximity = 1 means obstacle at boundary, proximity = 0 means far away
        # Use torch.tensor to ensure gradient support
        one = torch.tensor(1.0, device=obs.device, dtype=obs.dtype)
        eps = torch.tensor(self.barrier_epsilon, device=obs.device, dtype=obs.dtype)
        effective_dist = one - max_proximity + eps
        
        # Get adaptive stiffness
        k = self.obstacle_k_net(obs)  # [batch, ..., 1]
        k = torch.clamp(k * 2.0 + 0.1, min=0.1, max=5.0)
        
        # Log barrier potential
        H_barrier = -k * torch.log(effective_dist.clamp(min=self.barrier_epsilon))
        H_barrier = torch.clamp(H_barrier, max=10.0)
        
        return H_barrier
    
    def _compute_agent_barrier(self, obs, state_features, laplacian):
        """
        Compute agent-agent collision barrier potential.
        
        Uses SoftBarrierHead to learn pairwise stiffness k_ij,
        then computes log barrier between all agent pairs.
        """
        batch_size = obs.shape[0]
        
        # For single-agent, return zero with gradient support
        if obs.dim() == 2 or obs.shape[1] == 1:
            # Return a tensor derived from obs to maintain gradient chain
            return torch.zeros_like(obs[..., :1]).sum(dim=-1, keepdim=True) * 0.0
        
        n_agents = obs.shape[1]
        
        # Get pairwise stiffness from SoftBarrierHead
        k_ij = self.H_barrier_head(state_features, laplacian)  # [batch, n_agents, n_agents]
        
        # Extract agent lidar for distance estimation
        agent_lidar_end = min(self.agent_lidar_end, obs.shape[-1])
        if self.agent_lidar_start < agent_lidar_end:
            agent_lidar = obs[..., self.agent_lidar_start:agent_lidar_end]  # [batch, n_agents, 16]
            # Use max proximity as distance proxy
            proximity = agent_lidar.max(dim=-1)[0]  # [batch, n_agents]
        else:
            # Return zero with gradient support
            return torch.zeros_like(obs[..., :1]).sum(dim=-1, keepdim=True) * 0.0
        
        # Create pairwise distance matrix (symmetric)
        prox_i = proximity.unsqueeze(2)  # [batch, n_agents, 1]
        prox_j = proximity.unsqueeze(1)  # [batch, 1, n_agents]
        pairwise_proximity = (prox_i + prox_j) / 2  # [batch, n_agents, n_agents]
        
        # Convert to effective distance - use tensor operations to maintain gradient
        one = torch.tensor(1.0, device=obs.device, dtype=obs.dtype)
        eps = torch.tensor(self.barrier_epsilon, device=obs.device, dtype=obs.dtype)
        effective_dist = one - pairwise_proximity + eps
        
        # Log barrier for each pair
        H_barrier_pairs = -k_ij * torch.log(effective_dist.clamp(min=self.barrier_epsilon))
        H_barrier_pairs = torch.clamp(H_barrier_pairs, max=10.0)
        
        # Sum over pairs (upper triangular to avoid double counting)
        triu_mask = torch.triu(torch.ones(n_agents, n_agents, device=self.device), diagonal=1)
        H_barrier_total = (H_barrier_pairs * triu_mask.unsqueeze(0)).sum(dim=(-1, -2), keepdim=True)
        
        # Scale by number of pairs
        n_pairs = n_agents * (n_agents - 1) / 2
        if self.auto_scale_by_agents and n_pairs > 0:
            # Scale barrier to maintain consistent strength
            scale = 6.0 / n_pairs  # Reference: 4 agents = 6 pairs
            H_barrier_total = H_barrier_total * scale
        
        return H_barrier_total
    
    def _compute_kinetic_energy(self, obs):
        """Compute kinetic energy from velocity."""
        vel = obs[..., self.vel_indices]  # [batch, ..., 2]
        H_kin = 0.5 * (vel.pow(2).sum(dim=-1, keepdim=True))
        return H_kin
    
    def _construct_J_matrix(self, J_flat, batch_size):
        """Construct skew-symmetric J matrix from network output."""
        J = J_flat.view(batch_size, self.state_dim, self.state_dim)
        # Make skew-symmetric: J = (A - A^T) / 2
        J = (J - J.transpose(-1, -2)) / 2
        return J
    
    def _construct_R_matrix(self, R_flat, batch_size):
        """Construct positive semi-definite R matrix from network output."""
        R = R_flat.view(batch_size, self.state_dim, self.state_dim)
        # Make symmetric positive semi-definite: R = A @ A^T + eps*I
        R = torch.bmm(R, R.transpose(-1, -2))
        R = R + 0.01 * torch.eye(self.state_dim, device=self.device).unsqueeze(0)
        return R
    
    def _compute_hamiltonian_gradient(self, obs, state_features, laplacian):
        """
        Compute total Hamiltonian and its gradient via automatic differentiation.
        
        H_total = task_weight * (H_task + H_kin) + barrier_weight * (H_barrier_obs + H_barrier_agent)
        """
        batch_size = obs.shape[0]
        is_multi_agent = obs.dim() == 3
        
        # Ensure we can compute gradients even in eval mode
        with torch.enable_grad():
            # Create variable for gradient computation with explicit requires_grad
            obs_var = obs.clone().detach().requires_grad_(True)
            
            # Recompute state_features from obs_var to maintain gradient chain
            obs_var_norm = self.feature_norm(obs_var)
            state_features_var = self.state_encoder(obs_var_norm)
            
            # Compute all potential components
            H_kin = self._compute_kinetic_energy(obs_var)
            H_goal = self._compute_goal_potential(obs_var)  # Explicit goal attraction
            H_task_learned = self._compute_task_potential_learned(state_features_var)  # Learned task component
            H_task = H_goal + H_task_learned  # Combined task potential
            
            # Barrier potentials
            H_barrier_obs = self._compute_obstacle_barrier(obs_var)
            
            if is_multi_agent:
                # Recompute laplacian from obs_var to maintain gradient chain
                laplacian_var = self._compute_laplacian_matrix(obs_var)
                H_barrier_agent = self._compute_agent_barrier(obs_var, state_features_var, laplacian_var)
            else:
                # Ensure zeros are derived from obs_var for gradient chain
                H_barrier_agent = H_barrier_obs * 0.0
            
            # Get current barrier weight with warmup schedule
            current_barrier_weight = self._get_current_barrier_weight()
            
            # Convert to tensors to maintain gradient chain
            task_weight_tensor = torch.tensor(self.task_weight, device=obs_var.device, dtype=obs_var.dtype)
            barrier_weight_tensor = torch.tensor(current_barrier_weight, device=obs_var.device, dtype=obs_var.dtype)
            obstacle_weight_tensor = torch.tensor(self.obstacle_barrier_weight, device=obs_var.device, dtype=obs_var.dtype)
            
            # Total Hamiltonian (H_task = H_goal + H_task_learned)
            H_total = (
                task_weight_tensor * (H_task + H_kin) +
                barrier_weight_tensor * H_barrier_agent +
                obstacle_weight_tensor * H_barrier_obs
            )
            
            # Compute gradient via autograd
            grad_H = torch.autograd.grad(
                H_total.sum(),
                obs_var,
                create_graph=True,
                retain_graph=True
            )[0]
        
        # Extract gradient w.r.t. velocity (indices 3,4) as 2D action gradient
        grad_H_vel = grad_H[..., self.vel_indices]  # [batch, ..., 2]
        
        return H_total.detach(), grad_H_vel.detach(), {
            'H_goal': H_goal.detach(),
            'H_task_learned': H_task_learned.detach(),
            'H_task': H_task.detach(),
            'H_kin': H_kin.detach(),
            'H_barrier_obs': H_barrier_obs.detach(),
            'H_barrier_agent': H_barrier_agent.detach() if is_multi_agent else None,
            'barrier_weight': current_barrier_weight
        }
    
    def _compute_phs_action(self, obs, state_features, laplacian):
        """
        Compute action from Port-Hamiltonian dynamics.
        
        dx = (J - R) ∇H_total
        u = F⁻¹(dx - (J_sys - R_sys) ∇H_sys)
        
        Returns action mean derived from PHS dynamics.
        """
        batch_size = obs.shape[0]
        is_multi_agent = obs.dim() == 3
        
        if is_multi_agent:
            # Flatten for processing
            n_agents = obs.shape[1]
            obs_flat = obs.view(batch_size * n_agents, -1)
            state_features_flat = state_features.view(batch_size * n_agents, -1)
            laplacian_flat = None  # Will handle separately
        else:
            obs_flat = obs
            state_features_flat = state_features
        
        # Get learned system matrices
        J_flat = self.J_net(state_features_flat)  # [batch, state_dim^2]
        R_flat = self.R_net(state_features_flat)  # [batch, state_dim^2]
        
        J_learned = self._construct_J_matrix(J_flat, obs_flat.shape[0])
        R_learned = self._construct_R_matrix(R_flat, obs_flat.shape[0])
        
        # Compute Hamiltonian gradient
        H_total, grad_H_vel, H_info = self._compute_hamiltonian_gradient(
            obs, state_features, laplacian
        )
        
        # Flatten gradient if multi-agent
        if is_multi_agent:
            grad_H_vel_flat = grad_H_vel.view(batch_size * n_agents, -1)
        else:
            grad_H_vel_flat = grad_H_vel
        
        # Extend gradient to full state dimension
        grad_H_full = torch.zeros(obs_flat.shape[0], self.state_dim, device=self.device)
        grad_H_full[:, :2] = grad_H_vel_flat  # Velocity gradient
        
        # Compute PHS dynamics: dx = (J - R) ∇H
        J_minus_R = J_learned - R_learned
        dx = torch.bmm(J_minus_R, grad_H_full.unsqueeze(-1)).squeeze(-1)  # [batch, state_dim]
        
        # Compute base system dynamics
        J_sys_expanded = self.J_sys.unsqueeze(0).expand(obs_flat.shape[0], -1, -1)
        R_sys_expanded = self.R_sys.unsqueeze(0).expand(obs_flat.shape[0], -1, -1)
        J_R_sys = J_sys_expanded - R_sys_expanded
        
        dHdx_sys = grad_H_full  # Use same gradient for simplicity
        dx_sys = torch.bmm(J_R_sys, dHdx_sys.unsqueeze(-1)).squeeze(-1)
        
        # Compute control: u = F⁻¹(dx - dx_sys)
        # Use velocity derivative part (indices 2:4) as control signal
        delta_dx = dx[:, 2:self.state_dim] - dx_sys[:, 2:self.state_dim]  # [batch, 2]
        
        # Compute control via matrix multiply
        # F_pinv shape is [act_dim, state_dim], so use column slicing [:, 2:4]
        F_pinv_vel = self.F_pinv[:, 2:self.state_dim].T  # [2, act_dim]
        u_mean = torch.matmul(delta_dx, F_pinv_vel)  # [batch, 2] @ [2, act_dim] -> [batch, act_dim]
        
        # Clip to max force
        u_mean = torch.clamp(u_mean, -self.f_max, self.f_max)
        
        # Reshape if multi-agent
        if is_multi_agent:
            u_mean = u_mean.view(batch_size, n_agents, -1)
        
        return u_mean, H_info, state_features_flat if not is_multi_agent else state_features
    
    def forward(self, obs, rnn_states=None, masks=None, available_actions=None, deterministic=False):
        """
        Forward pass: compute action from PHS dynamics.
        
        Args:
            obs: [batch, obs_dim] or [batch, n_agents, obs_dim] observation
            rnn_states: RNN hidden states (unused, for compatibility)
            masks: Episode masks (unused)
            available_actions: Available actions mask (unused)
            deterministic: Whether to use mean action
            
        Returns:
            actions: Sampled or mean actions
            action_log_probs: Log probabilities of actions
            rnn_states: Unchanged RNN states
        """
        obs = check(obs).to(self.device)
        is_multi_agent = obs.dim() == 3
        batch_size = obs.shape[0]
        
        # Normalize and encode state
        obs_norm = self.feature_norm(obs)
        state_features = self.state_encoder(obs_norm)
        
        # Compute Laplacian for multi-agent
        if is_multi_agent:
            laplacian = self._compute_laplacian_matrix(obs)
        else:
            laplacian = None
        
        # Compute action from PHS dynamics
        u_mean, H_info, state_feat = self._compute_phs_action(obs, state_features, laplacian)
        
        # Compute action std
        if is_multi_agent:
            n_agents = obs.shape[1]
            state_feat_for_std = state_features.view(batch_size * n_agents, -1)
            u_mean_flat = u_mean.view(batch_size * n_agents, -1)
        else:
            state_feat_for_std = state_features
            u_mean_flat = u_mean
            
        std_input = torch.cat([state_feat_for_std, u_mean_flat], dim=-1)
        u_log_std = self.std_net(std_input)
        u_log_std = torch.clamp(u_log_std, -2.0, 0.5)  # Reasonable std range
        u_std = torch.exp(u_log_std)
        
        if is_multi_agent:
            u_std = u_std.view(batch_size, n_agents, -1)
        
        # Create distribution and sample
        dist = torch.distributions.Normal(u_mean, u_std)
        
        if deterministic:
            action = u_mean
        else:
            action = dist.rsample()
        
        # Clip actions
        action = torch.clamp(action, -1.0, 1.0)
        
        # Compute log probs
        action_log_probs = dist.log_prob(action)
        
        # Return RNN states for compatibility
        if rnn_states is None:
            rnn_states = torch.zeros(batch_size, 1, 1, device=self.device)
        
        return action, action_log_probs, rnn_states
    
    def evaluate_actions(self, obs, rnn_states, action, masks, available_actions=None, active_masks=None):
        """
        Evaluate given actions for PPO update.
        
        Args:
            obs: [batch, obs_dim] observations
            rnn_states: RNN states (unused)
            action: [batch, act_dim] actions to evaluate
            masks: Episode masks
            available_actions: Available actions (unused)
            active_masks: Active agent masks
            
        Returns:
            action_log_probs: Log probability of actions
            dist_entropy: Entropy of action distribution
        """
        obs = check(obs).to(self.device)
        action = check(action).to(self.device)
        is_multi_agent = obs.dim() == 3
        batch_size = obs.shape[0]
        
        # Normalize and encode state
        obs_norm = self.feature_norm(obs)
        state_features = self.state_encoder(obs_norm)
        
        # Compute Laplacian for multi-agent
        if is_multi_agent:
            laplacian = self._compute_laplacian_matrix(obs)
        else:
            laplacian = None
        
        # Compute action from PHS dynamics
        u_mean, H_info, state_feat = self._compute_phs_action(obs, state_features, laplacian)
        
        # Compute action std (must match forward())
        if is_multi_agent:
            n_agents = obs.shape[1]
            state_feat_for_std = state_features.view(batch_size * n_agents, -1)
            u_mean_flat = u_mean.view(batch_size * n_agents, -1)
        else:
            state_feat_for_std = state_features
            u_mean_flat = u_mean
            
        std_input = torch.cat([state_feat_for_std, u_mean_flat], dim=-1)
        u_log_std = self.std_net(std_input)
        u_log_std = torch.clamp(u_log_std, -2.0, 0.5)
        u_std = torch.exp(u_log_std)
        
        if is_multi_agent:
            u_std = u_std.view(batch_size, n_agents, -1)
        
        # Create distribution
        dist = torch.distributions.Normal(u_mean, u_std)
        
        # Compute log prob of given action
        action_log_probs = dist.log_prob(action)
        
        # Compute entropy
        dist_entropy = dist.entropy().mean()
        
        return action_log_probs, dist_entropy
    
    def get_physics_info(self, obs):
        """Get physics information for logging/visualization."""
        obs = check(obs).to(self.device)
        is_multi_agent = obs.dim() == 3
        
        obs_norm = self.feature_norm(obs)
        state_features = self.state_encoder(obs_norm)
        
        if is_multi_agent:
            laplacian = self._compute_laplacian_matrix(obs)
        else:
            laplacian = None
        
        H_total, grad_H_vel, H_info = self._compute_hamiltonian_gradient(
            obs, state_features, laplacian
        )
        
        # Extract lidar info
        hazard_end = min(self.hazard_lidar_end, obs.shape[-1])
        hazard_lidar = obs[..., self.hazard_lidar_start:hazard_end]
        proximity = hazard_lidar.max(dim=-1, keepdim=True)[0]
        
        return {
            'H_total': H_total.detach(),
            'H_goal': H_info['H_goal'],
            'H_task_learned': H_info['H_task_learned'],
            'H_task': H_info['H_task'],
            'H_barrier_obs': H_info['H_barrier_obs'],
            'H_barrier_agent': H_info['H_barrier_agent'],
            'grad_H': grad_H_vel.detach(),
            'proximity': proximity.detach(),
            'barrier_weight': H_info['barrier_weight']
        }
    
    def set_training_step(self, step):
        """Set current training step for barrier warmup schedule."""
        self._training_step = step
    
    def _compute_task_potential(self, obs):
        """Wrapper for compatibility with trainer's auxiliary loss."""
        H_goal = self._compute_goal_potential(obs)
        obs_norm = self.feature_norm(obs)
        state_features = self.state_encoder(obs_norm)
        H_task_learned = self._compute_task_potential_learned(state_features)
        return H_goal + H_task_learned, None
    
    def _compute_barrier_potential(self, obs):
        """Wrapper for compatibility with trainer's auxiliary loss."""
        H_barrier = self._compute_obstacle_barrier(obs)
        
        # Compute approximate gradient direction
        hazard_end = min(self.hazard_lidar_end, obs.shape[-1])
        hazard_lidar = obs[..., self.hazard_lidar_start:hazard_end]
        
        num_bins = hazard_lidar.shape[-1]
        angles = torch.linspace(0, 2 * np.pi, num_bins + 1, device=self.device)[:-1]
        
        weights = torch.exp(3.0 * hazard_lidar) - 1.0
        weights_sum = weights.sum(dim=-1, keepdim=True) + 1e-6
        
        obs_dir_x = (weights * torch.cos(angles)).sum(dim=-1, keepdim=True) / weights_sum
        obs_dir_y = (weights * torch.sin(angles)).sum(dim=-1, keepdim=True) / weights_sum
        
        k = self.obstacle_k_net(obs)
        grad_mag = k * 5.0 * hazard_lidar.max(dim=-1, keepdim=True)[0]
        
        grad_H = torch.cat([-grad_mag * obs_dir_x, -grad_mag * obs_dir_y], dim=-1)
        
        return H_barrier, grad_H
    
    def _extract_lidar_info(self, obs):
        """Extract hazard lidar for compatibility."""
        hazard_end = min(self.hazard_lidar_end, obs.shape[-1])
        hazard_lidar = obs[..., self.hazard_lidar_start:hazard_end]
        
        agent_end = min(self.agent_lidar_end, obs.shape[-1])
        if self.agent_lidar_start < agent_end:
            agent_lidar = obs[..., self.agent_lidar_start:agent_end]
            combined = torch.maximum(hazard_lidar, agent_lidar)
        else:
            combined = hazard_lidar
        
        proximity = combined.max(dim=-1, keepdim=True)[0]
        return combined, proximity
    
    def _extract_goal_lidar_info(self, obs):
        """Extract goal lidar for compatibility."""
        goal_end = min(self.goal_lidar_end, obs.shape[-1])
        goal_lidar = obs[..., self.goal_lidar_start:goal_end]
        goal_proximity = goal_lidar.max(dim=-1, keepdim=True)[0]
        return goal_lidar, goal_proximity
    
    @property
    def barrier_k_net(self):
        """Compatibility alias for obstacle_k_net."""
        return self.obstacle_k_net

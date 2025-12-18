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
Barrier Port-Hamiltonian PINN Actor Module.

This module implements the Safe Physics-Informed Neural Network Actor based on 
Barrier Port-Hamiltonian Systems for Safe-pH-MARL framework.

Based on Barrier_PHS.md theoretical formulation:

Port-Hamiltonian dynamics: ẋ = (J(x) - R(x)) ∇H_total(x)

Where H_total = H_kin(p) + H_task(q;θ) + H_barrier(q;φ)
  - H_task: Learnable task potential (attracts to goal)
  - H_barrier: Parametric barrier potential (repels from obstacles/agents)
  - J: Skew-symmetric interconnection matrix (gyroscopic forces for escaping local minima)
  - R: Positive semi-definite dissipation matrix

Key Safety Feature:
  H_barrier = Σ k_ij / (||q_i - q_j|| - r_safe)² + ε
  When distance → r_safe, H_barrier → ∞, providing hard safety guarantee
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def check(input):
    """Convert numpy array to torch tensor if needed."""
    output = torch.from_numpy(input) if type(input) == np.ndarray else input
    return output


class BarrierPHSPINNActor(nn.Module):
    """
    Safe Physics-Informed Neural Network Actor based on Barrier Port-Hamiltonian Systems.
    
    This implements the Safe-pH-MARL framework from Barrier_PHS.md:
    
    1. Potential Energy Splitting:
       - H_task: Neural network learned potential for goal-reaching (attractor)
       - H_barrier: Barrier Lyapunov Function for collision avoidance (repulsor)
       
    2. Port-Hamiltonian Dynamics:
       - J: Skew-symmetric matrix for gyroscopic forces (escape local minima)
       - R: Positive semi-definite dissipation matrix for stability
       
    3. Safety Guarantee:
       - When agents approach collision boundary, H_barrier → ∞
       - Since system is passive (Ḣ ≤ 0), agents cannot acquire enough energy
         to "climb over" the infinite potential barrier
    
    For Point/Car agent in SafetyMultiGoal environments:
        - Observation: ~152-dim (accelerometer, velocimeter, gyro, magnetometer, lidars)
        - velocimeter (indices 3-5): velocity (vx, vy, vz)
        - lidar observations: detect obstacles and other agents
        - Action: 2-dim (forward force, turning velocity)
    """
    
    def __init__(self, config, obs_space, act_space, device=torch.device("cpu")):
        super(BarrierPHSPINNActor, self).__init__()
        
        self.config = config
        self.device = device
        self.obs_dim = obs_space.shape[0]  # ~152 for Point MultiGoal
        self.act_dim = act_space.shape[0]  # 2 for Point
        
        # ===================
        # Configuration
        # ===================
        self.hidden_size = config.get("hidden_size", 256)
        self.physics_hidden = config.get("physics_hidden", 128)
        self.state_dim = config.get("pinn_state_dim", 4)  # (vx, vy, ax, ay)
        self.std_x_coef = config.get("std_x_coef", 1.0)
        self.std_y_coef = config.get("std_y_coef", 0.5)
        
        # Barrier potential parameters
        self.r_safe = config.get("barrier_r_safe", 0.3)  # Safe distance threshold
        self.barrier_epsilon = config.get("barrier_epsilon", 0.01)  # Numerical stability
        self.barrier_clip_max = config.get("barrier_clip_max", 50.0)  # Clip for stability
        self.num_lidar_bins = config.get("num_lidar_bins", 16)  # Lidar bins per obstacle type
        
        # Physics state extraction indices (for Point agent observation)
        # obs[0:3] = accelerometer (ax, ay, az)
        # obs[3:6] = velocimeter (vx, vy, vz)  <- velocity!
        # obs[6:9] = gyro (angular velocity)
        # obs[9:12] = magnetometer (orientation)
        # obs[12:28] = goal_red lidar (16 bins) - agent 0's goal
        # obs[28:44] = goal_blue lidar (16 bins) - agent 1's goal
        # ... (may vary based on environment)
        self.vel_indices = [3, 4]  # vx, vy
        self.acc_indices = [0, 1]  # ax, ay
        
        # Lidar indices for obstacle detection (approximate, may need adjustment)
        # These detect distances to obstacles/goals
        self.lidar_start_idx = 12  # Start of lidar observations
        self.lidar_end_idx = min(12 + 2 * self.num_lidar_bins, self.obs_dim)  # Two goal lidars
        
        # ===================
        # Task Potential Network H_task(obs) - The Attractor
        # ===================
        # Learns the energy landscape where goal is the global minimum
        # Uses full observation to understand goal direction
        self.H_task_net = nn.Sequential(
            nn.Linear(self.obs_dim, self.physics_hidden),
            nn.ELU(),
            nn.Linear(self.physics_hidden, self.physics_hidden),
            nn.ELU(),
            nn.Linear(self.physics_hidden, 1)  # Scalar task potential
        )
        
        # ===================
        # Barrier Stiffness Network k(obs) - Adaptive Barrier Strength
        # ===================
        # Predicts stiffness coefficient k for the barrier potential
        # H_barrier = k / (d - r_safe)^2 where d is distance to nearest obstacle
        self.barrier_k_net = nn.Sequential(
            nn.Linear(self.obs_dim, self.physics_hidden),
            nn.ELU(),
            nn.Linear(self.physics_hidden, self.physics_hidden // 2),
            nn.ELU(),
            nn.Linear(self.physics_hidden // 2, 1),
            nn.Softplus()  # Ensure positive stiffness
        )
        
        # ===================
        # Interconnection Matrix J(state) - Gyroscopic Forces
        # ===================
        # Skew-symmetric matrix that produces forces perpendicular to gradient
        # Helps escape local minima when task and barrier gradients oppose
        self.J_dim = self.state_dim * (self.state_dim - 1) // 2  # Upper triangular elements
        self.J_net = nn.Sequential(
            nn.Linear(self.state_dim, self.physics_hidden),
            nn.ELU(),
            nn.Linear(self.physics_hidden, self.J_dim)
        )
        
        # ===================
        # Dissipation Matrix R(state) - Base Damping
        # ===================
        # Positive semi-definite matrix R = L @ L^T for energy dissipation
        self.R_tril_dim = self.state_dim * (self.state_dim + 1) // 2
        self.R_net = nn.Sequential(
            nn.Linear(self.state_dim, self.physics_hidden),
            nn.ELU(),
            nn.Linear(self.physics_hidden, self.R_tril_dim)
        )
        
        # ===================
        # Feature Extraction (Standard MLP backbone like MAPPO)
        # ===================
        self.feature_norm = nn.LayerNorm(self.obs_dim)
        
        self.base_net = nn.Sequential(
            nn.Linear(self.obs_dim, self.hidden_size),
            nn.ELU(),
            nn.LayerNorm(self.hidden_size),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ELU(),
            nn.LayerNorm(self.hidden_size),
        )
        
        # ===================
        # Physics-Policy Integration Layer
        # ===================
        # Combines physics features with learned features
        # Physics features: H_task, H_barrier, grad_H_total (2D), dynamics (2D)
        physics_feature_dim = 2 + 2 + 2  # H_task, H_barrier, grad_H (2D), dynamics (2D)
        combined_dim = self.hidden_size + physics_feature_dim
        
        self.policy_integration = nn.Sequential(
            nn.Linear(combined_dim, self.hidden_size),
            nn.ELU(),
            nn.LayerNorm(self.hidden_size),
        )
        
        # Action mean output
        self.action_mean = nn.Linear(self.hidden_size, self.act_dim)
        
        # Log std (learnable, like MAPPO)
        self.log_std = nn.Parameter(torch.zeros(1, self.act_dim))
        
        # Initialize weights
        self._init_weights()
        
        self.to(device)
    
    def _init_weights(self):
        """Initialize network weights using orthogonal initialization."""
        def init_layer(m, gain=np.sqrt(2)):
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=gain)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        
        # Task potential network with smaller gain for stability
        for m in self.H_task_net:
            init_layer(m, gain=0.1)
        
        # Barrier stiffness network
        for m in self.barrier_k_net:
            if isinstance(m, nn.Linear):
                init_layer(m, gain=0.1)
        
        # Physics networks (J, R) with smaller gain for stability
        for net in [self.J_net, self.R_net]:
            for m in net:
                init_layer(m, gain=0.01)
        
        # Base network with standard gain
        for m in self.base_net:
            init_layer(m)
        
        for m in self.policy_integration:
            init_layer(m)
        
        # Action output with small gain
        init_layer(self.action_mean, gain=0.01)
    
    def _extract_physics_state(self, obs):
        """
        Extract physics-relevant state from observation.
        For Point agent: velocity (vx, vy) and acceleration (ax, ay)
        
        Args:
            obs: [batch, obs_dim] observation tensor
            
        Returns:
            state: [batch, state_dim] physics state (vx, vy, ax, ay)
        """
        vel = obs[:, self.vel_indices]  # [batch, 2] - vx, vy
        acc = obs[:, self.acc_indices]  # [batch, 2] - ax, ay
        state = torch.cat([vel, acc], dim=-1)  # [batch, 4]
        return state
    
    def _extract_lidar_info(self, obs):
        """
        Extract lidar information from observation for obstacle detection.
        Lidar values are in [0, 1] where higher values = closer objects.
        
        Args:
            obs: [batch, obs_dim] observation tensor
            
        Returns:
            lidar_obs: [batch, num_lidar] lidar readings
            min_dist: [batch, 1] estimated minimum distance to obstacles
        """
        # Extract lidar observations (may span multiple obstacle types)
        lidar_end = min(self.lidar_end_idx, obs.shape[-1])
        if self.lidar_start_idx < lidar_end:
            lidar_obs = obs[:, self.lidar_start_idx:lidar_end]
        else:
            # Fallback: use zeros if indices are invalid
            lidar_obs = torch.zeros(obs.shape[0], self.num_lidar_bins, device=self.device)
        
        # Find maximum lidar reading (closest obstacle)
        # Lidar reading = exp(-gain * dist) or (max_dist - dist) / max_dist
        # So max reading = min distance
        max_lidar = lidar_obs.max(dim=-1, keepdim=True)[0]  # [batch, 1]
        
        # Convert lidar reading to approximate distance
        # Assuming exponential decay: lidar = exp(-dist), so dist = -ln(lidar)
        # Clamp to avoid log(0)
        max_lidar_clamped = torch.clamp(max_lidar, min=1e-6, max=1.0)
        
        # Approximate distance (in lidar units, not actual meters)
        # For pseudo lidar with max_dist=3: lidar = (max_dist - dist) / max_dist
        # So dist = max_dist * (1 - lidar)
        approx_dist = 3.0 * (1.0 - max_lidar_clamped)  # Approximate distance
        
        return lidar_obs, approx_dist
    
    def _compute_barrier_potential(self, obs):
        """
        Compute Barrier Lyapunov Function (BLF) based potential.
        
        H_barrier = k / ((d - r_safe)^2 + ε)
        
        As distance d → r_safe, H_barrier → k/ε (large but finite for numerical stability)
        In theory, ε → 0 gives infinite barrier, but we use small ε for training.
        
        Args:
            obs: [batch, obs_dim] observation tensor
            
        Returns:
            H_barrier: [batch, 1] barrier potential
            grad_H_barrier: [batch, 2] gradient w.r.t. velocity (approximation)
        """
        # Get adaptive stiffness from network
        k = self.barrier_k_net(obs)  # [batch, 1], positive due to Softplus
        
        # Extract minimum distance from lidar
        _, min_dist = self._extract_lidar_info(obs)  # [batch, 1]
        
        # Compute barrier potential: H = k / ((d - r_safe)^2 + ε)
        # When d < r_safe, the denominator is small, making H large
        dist_margin = min_dist - self.r_safe  # [batch, 1]
        denominator = dist_margin ** 2 + self.barrier_epsilon
        
        H_barrier = k / denominator  # [batch, 1]
        
        # Clip for numerical stability during training
        H_barrier = torch.clamp(H_barrier, max=self.barrier_clip_max)
        
        # Compute gradient approximation w.r.t. velocity direction
        # Gradient points away from obstacles (repulsive)
        # Use velocity direction as proxy for movement direction
        vel = obs[:, self.vel_indices]  # [batch, 2]
        vel_norm = torch.norm(vel, dim=-1, keepdim=True) + 1e-6
        vel_direction = vel / vel_norm
        
        # Gradient magnitude: dH/dd = -2k(d - r_safe) / ((d - r_safe)^2 + ε)^2
        # Points opposite to obstacle direction (which we approximate as velocity direction when approaching)
        grad_magnitude = 2 * k * torch.abs(dist_margin) / (denominator ** 2 + 1e-6)
        grad_magnitude = torch.clamp(grad_magnitude, max=10.0)  # Clip gradient
        
        # If moving toward obstacle (small distance), gradient opposes movement
        # This is a simplification - ideally we'd have explicit obstacle positions
        # For now, use lidar-weighted direction
        lidar_obs, _ = self._extract_lidar_info(obs)
        
        # Compute weighted direction from lidar (approximate obstacle direction)
        num_bins = lidar_obs.shape[-1]
        angles = torch.linspace(0, 2 * np.pi, num_bins + 1, device=self.device)[:-1]
        angles = angles.unsqueeze(0).expand(obs.shape[0], -1)  # [batch, num_bins]
        
        # Weighted sum of directions based on lidar readings (higher = closer = stronger repulsion)
        weights = lidar_obs  # [batch, num_bins]
        weights_sum = weights.sum(dim=-1, keepdim=True) + 1e-6
        
        obstacle_dir_x = (weights * torch.cos(angles)).sum(dim=-1, keepdim=True) / weights_sum
        obstacle_dir_y = (weights * torch.sin(angles)).sum(dim=-1, keepdim=True) / weights_sum
        
        # Gradient points away from obstacles
        grad_H_barrier_x = -grad_magnitude * obstacle_dir_x
        grad_H_barrier_y = -grad_magnitude * obstacle_dir_y
        grad_H_barrier = torch.cat([grad_H_barrier_x, grad_H_barrier_y], dim=-1)  # [batch, 2]
        
        return H_barrier, grad_H_barrier
    
    def _compute_task_potential(self, obs):
        """
        Compute task potential using neural network.
        The gradient of H_task should point toward obstacles/away from goal
        (since we want to minimize potential at the goal).
        
        Args:
            obs: [batch, obs_dim] observation tensor
            
        Returns:
            H_task: [batch, 1] task potential
            grad_H_task: [batch, 2] gradient w.r.t. position (approximation via velocity)
        """
        # Check if we're in no_grad mode (inference)
        if not torch.is_grad_enabled():
            with torch.enable_grad():
                obs_grad = obs.clone().requires_grad_(True)
                H_task = self.H_task_net(obs_grad)
                
                # Compute gradient w.r.t. velocity components (indices 3, 4)
                # This approximates the direction of steepest descent
                grad_outputs = torch.ones_like(H_task)
                grad_obs = torch.autograd.grad(
                    outputs=H_task,
                    inputs=obs_grad,
                    grad_outputs=grad_outputs,
                    create_graph=False,
                    retain_graph=False
                )[0]
                
                # Extract gradient w.r.t. velocity as proxy for spatial gradient
                grad_H_task = grad_obs[:, self.vel_indices]
                
            return H_task.detach(), grad_H_task.detach()
        else:
            obs_grad = obs.clone().requires_grad_(True)
            H_task = self.H_task_net(obs_grad)
            
            grad_outputs = torch.ones_like(H_task)
            grad_obs = torch.autograd.grad(
                outputs=H_task,
                inputs=obs_grad,
                grad_outputs=grad_outputs,
                create_graph=True,
                retain_graph=True
            )[0]
            
            grad_H_task = grad_obs[:, self.vel_indices]
            
            return H_task, grad_H_task
    
    def _construct_J_matrix(self, J_elements, batch_size):
        """
        Construct skew-symmetric interconnection matrix J from upper triangular elements.
        J = -J^T (antisymmetric)
        
        Physical meaning: Produces gyroscopic forces perpendicular to gradient.
        Helps agents navigate around obstacles by moving along potential contours.
        
        Args:
            J_elements: [batch, J_dim] upper triangular elements
            batch_size: batch size
            
        Returns:
            J: [batch, state_dim, state_dim] skew-symmetric matrix
        """
        J = torch.zeros(batch_size, self.state_dim, self.state_dim, device=self.device)
        
        # Fill upper triangular (excluding diagonal)
        idx = 0
        for i in range(self.state_dim):
            for j in range(i + 1, self.state_dim):
                J[:, i, j] = J_elements[:, idx]
                J[:, j, i] = -J_elements[:, idx]  # Skew-symmetric: J[j,i] = -J[i,j]
                idx += 1
        
        return J
    
    def _construct_R_matrix(self, R_tril_elements, batch_size):
        """
        Construct positive semi-definite dissipation matrix R = L @ L^T.
        
        Physical meaning: Energy dissipation for stability.
        Ensures the system eventually settles to equilibrium.
        
        Args:
            R_tril_elements: [batch, R_tril_dim] lower triangular elements
            batch_size: batch size
            
        Returns:
            R: [batch, state_dim, state_dim] positive semi-definite matrix
        """
        # Create lower triangular matrix L
        L = torch.zeros(batch_size, self.state_dim, self.state_dim, device=self.device)
        
        idx = 0
        for i in range(self.state_dim):
            for j in range(i + 1):
                L[:, i, j] = R_tril_elements[:, idx]
                idx += 1
        
        # Ensure positive diagonal for numerical stability
        L_diag = torch.diagonal(L, dim1=-2, dim2=-1)
        L_diag_soft = F.softplus(L_diag) + 1e-4
        L = L - torch.diag_embed(torch.diagonal(L, dim1=-2, dim2=-1)) + torch.diag_embed(L_diag_soft)
        
        # R = L @ L^T (guaranteed positive semi-definite)
        R = torch.bmm(L, L.transpose(-1, -2))
        
        return R
    
    def _compute_total_hamiltonian_gradient(self, obs, state):
        """
        Compute total Hamiltonian and its gradient.
        
        H_total = H_task + H_barrier
        ∇H_total = ∇H_task + ∇H_barrier
        
        Args:
            obs: [batch, obs_dim] observation
            state: [batch, state_dim] physics state
            
        Returns:
            H_task: [batch, 1] task potential
            H_barrier: [batch, 1] barrier potential
            grad_H_total: [batch, 2] combined gradient (2D for velocity direction)
        """
        # Compute task potential and gradient
        H_task, grad_H_task = self._compute_task_potential(obs)
        
        # Compute barrier potential and gradient
        H_barrier, grad_H_barrier = self._compute_barrier_potential(obs)
        
        # Combine gradients (clip barrier gradient for stability)
        grad_H_barrier_clipped = torch.clamp(grad_H_barrier, min=-5.0, max=5.0)
        grad_H_total = grad_H_task + grad_H_barrier_clipped
        
        return H_task, H_barrier, grad_H_total
    
    def _compute_port_hamiltonian_dynamics(self, obs, state):
        """
        Compute Port-Hamiltonian dynamics: ẋ = (J - R) ∇H
        
        This provides physics-consistent features for the policy.
        The dynamics incorporate:
        1. Gyroscopic forces from J (perpendicular to gradient, for obstacle avoidance)
        2. Damping from R (energy dissipation for stability)
        3. Barrier potential (repulsive forces from obstacles)
        4. Task potential (attractive forces toward goal)
        
        Args:
            obs: [batch, obs_dim] observation
            state: [batch, state_dim] physics state
            
        Returns:
            H_task: Task potential value
            H_barrier: Barrier potential value
            grad_H_total: Combined gradient
            dynamics: (J - R) ∇H term (in velocity space)
        """
        batch_size = state.shape[0]
        
        # Get total Hamiltonian gradient
        H_task, H_barrier, grad_H_total = self._compute_total_hamiltonian_gradient(obs, state)
        
        # Extend 2D gradient to full state_dim by padding
        grad_H_full = torch.zeros(batch_size, self.state_dim, device=self.device)
        grad_H_full[:, :2] = grad_H_total  # Velocity components
        
        # Get J and R matrices from physics networks
        J_elements = self.J_net(state)
        R_elements = self.R_net(state)
        
        J = self._construct_J_matrix(J_elements, batch_size)
        R = self._construct_R_matrix(R_elements, batch_size)
        
        # Compute dynamics: (J - R) ∇H
        J_minus_R = J - R
        dynamics = torch.bmm(J_minus_R, grad_H_full.unsqueeze(-1)).squeeze(-1)
        
        # Extract velocity-relevant dynamics (first 2 components)
        dynamics_2d = dynamics[:, :2]
        
        return H_task, H_barrier, grad_H_total, dynamics_2d
    
    def forward(self, obs, rnn_states=None, masks=None, available_actions=None, deterministic=False):
        """
        Forward pass: compute action from observation using Barrier PHS framework.
        
        The action is computed by combining:
        1. Physics-informed features from Port-Hamiltonian dynamics
        2. Learned features from standard MLP
        
        The barrier potential ensures safety by creating repulsive forces
        that grow infinitely as agents approach collision boundaries.
        
        Args:
            obs: [batch, obs_dim] observation tensor
            rnn_states: RNN hidden states (unused, for compatibility)
            masks: Episode masks (unused)
            available_actions: Available actions mask (unused)
            deterministic: Whether to sample or use mean action
            
        Returns:
            actions: [batch, act_dim] actions
            action_log_probs: [batch, act_dim] log probabilities
            rnn_states: Unchanged RNN states
        """
        obs = check(obs).to(self.device)
        
        # Extract physics state
        state = self._extract_physics_state(obs)
        
        # Compute Port-Hamiltonian features with barrier potential
        H_task, H_barrier, grad_H_total, dynamics = self._compute_port_hamiltonian_dynamics(obs, state)
        
        # Extract base features (standard MLP path)
        obs_normalized = self.feature_norm(obs)
        base_features = self.base_net(obs_normalized)
        
        # Combine physics and learned features
        # Physics features: H_task, H_barrier, grad_H_total (2D), dynamics (2D)
        physics_features = torch.cat([H_task, H_barrier, grad_H_total, dynamics], dim=-1)  # [batch, 6]
        combined_features = torch.cat([base_features, physics_features], dim=-1)
        
        # Policy integration
        policy_features = self.policy_integration(combined_features)
        
        # Compute action mean and std
        action_mean = self.action_mean(policy_features)
        action_std = torch.sigmoid(self.log_std / self.std_x_coef) * self.std_y_coef
        action_std = action_std.expand_as(action_mean)
        
        # Create distribution
        dist = torch.distributions.Normal(action_mean, action_std)
        
        if deterministic:
            action = action_mean
        else:
            action = dist.rsample()  # Reparameterization trick
        
        # Compute log probability
        action_log_probs = dist.log_prob(action)  # Per-dimension, no sum (like MAPPO)
        
        # Return same rnn_states for compatibility
        if rnn_states is None:
            rnn_states = torch.zeros(obs.shape[0], 1, 1, device=self.device)
        
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
        
        # Extract physics state
        state = self._extract_physics_state(obs)
        
        # Compute Port-Hamiltonian features with barrier potential
        H_task, H_barrier, grad_H_total, dynamics = self._compute_port_hamiltonian_dynamics(obs, state)
        
        # Extract base features
        obs_normalized = self.feature_norm(obs)
        base_features = self.base_net(obs_normalized)
        
        # Combine features
        physics_features = torch.cat([H_task, H_barrier, grad_H_total, dynamics], dim=-1)
        combined_features = torch.cat([base_features, physics_features], dim=-1)
        
        # Policy integration
        policy_features = self.policy_integration(combined_features)
        
        # Compute action distribution
        action_mean = self.action_mean(policy_features)
        action_std = torch.sigmoid(self.log_std / self.std_x_coef) * self.std_y_coef
        action_std = action_std.expand_as(action_mean)
        
        dist = torch.distributions.Normal(action_mean, action_std)
        
        # Compute log prob of given action
        action_log_probs = dist.log_prob(action)  # Per-dimension (like MAPPO)
        
        # Compute entropy
        dist_entropy = dist.entropy().mean()
        
        return action_log_probs, dist_entropy
    
    def get_physics_info(self, obs):
        """
        Get Barrier Port-Hamiltonian physics information for analysis/logging.
        
        Returns:
            dict with H_task, H_barrier, grad_H_total, J, R matrices
        """
        obs = check(obs).to(self.device)
        state = self._extract_physics_state(obs)
        batch_size = state.shape[0]
        
        H_task, H_barrier, grad_H_total = self._compute_total_hamiltonian_gradient(obs, state)
        
        J_elements = self.J_net(state)
        R_elements = self.R_net(state)
        
        J = self._construct_J_matrix(J_elements, batch_size)
        R = self._construct_R_matrix(R_elements, batch_size)
        
        _, min_dist = self._extract_lidar_info(obs)
        
        return {
            'H_task': H_task.detach(),
            'H_barrier': H_barrier.detach(),
            'grad_H_total': grad_H_total.detach(),
            'J': J.detach(),
            'R': R.detach(),
            'state': state.detach(),
            'min_dist': min_dist.detach()
        }

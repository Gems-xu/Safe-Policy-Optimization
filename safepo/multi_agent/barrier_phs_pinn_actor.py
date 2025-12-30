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
        
        # Barrier potential parameters (v4.0 - balanced for learning)
        # Key insight: directly use lidar values, not converted distances!
        self.r_safe = config.get("barrier_r_safe", 0.3)  # Safe threshold in lidar-space (0.3 = lidar > 0.7)
        self.barrier_epsilon = config.get("barrier_epsilon", 0.01)  # Small for sharper barrier
        self.barrier_clip_max = config.get("barrier_clip_max", 10.0)  # Reduced clip for stability
        self.num_lidar_bins = config.get("num_lidar_bins", 16)  # Lidar bins per obstacle type
        
        # Enhanced safety parameters (v6.0 - learning-based, no forced correction)
        # Let the agent learn safety through reward shaping, not action modification
        self.barrier_k_scale = config.get("barrier_k_scale", 2.0)  # Moderate stiffness
        self.barrier_gradient_scale = config.get("barrier_gradient_scale", 2.0)  # Moderate gradient
        self.barrier_decay_rate = config.get("barrier_decay_rate", 2.0)  # Balanced decay
        self.min_barrier_k = config.get("min_barrier_k", 0.3)  # Reasonable minimum
        
        # Cost-aware safety: used for reward shaping, not action correction
        self.cost_aware_weight = config.get("cost_aware_weight", 0.3)
        self.danger_zone_threshold = config.get("danger_zone_threshold", 0.5)  # Balanced threshold
        
        # v6.0: Safety layer DISABLED - let agent learn through rewards
        self.use_safety_layer = config.get("use_safety_layer", False)  # Disabled!
        self.safety_gamma = config.get("safety_gamma", 0.1)
        self.action_clip_margin = config.get("action_clip_margin", 0.05)
        
        # v6.0: NO direct potential-to-action modification
        # Potential information is provided as features for the network to learn from
        self.potential_action_weight = config.get("potential_action_weight", 0.0)  # Disabled!
        self.barrier_action_weight = config.get("barrier_action_weight", 0.0)  # Disabled!
        
        # Physics state extraction indices (for Point agent observation)
        # MultiGoal observation structure (verified):
        # obs[0:3]   = accelerometer (ax, ay, az)
        # obs[3:6]   = velocimeter (vx, vy, vz)  <- velocity!
        # obs[6:9]   = gyro (angular velocity)
        # obs[9:12]  = magnetometer (orientation)
        # obs[12:28] = goal_red lidar (16 bins) - agent 0's goal
        # obs[28:44] = goal_blue lidar (16 bins) - agent 1's goal  
        # obs[44:60] = HAZARD lidar (16 bins) - obstacles/hazards <- CRITICAL for barrier!
        # obs[60:76] = vases lidar (16 bins) - optional
        # obs[76:92] = other_agent lidar (16 bins) - other agents <- for collision avoidance
        self.vel_indices = [3, 4]  # vx, vy
        self.acc_indices = [0, 1]  # ax, ay
        
        # === CRITICAL: Hazard Lidar indices for obstacle detection ===
        # MUST use hazard lidar (44:60), NOT goal lidar (12:44)!
        self.hazard_lidar_start_idx = 44  # Start of HAZARD lidar
        self.hazard_lidar_end_idx = 60    # End of HAZARD lidar (16 bins)
        
        # Goal lidar indices (for task potential)
        self.goal_lidar_start_idx = 12    # goal_red lidar start
        self.goal_lidar_end_idx = 44      # goal_blue lidar end
        
        # Other agent lidar indices (for inter-agent collision avoidance)
        self.agent_lidar_start_idx = 76   # other_agent lidar start (if available)
        self.agent_lidar_end_idx = 92     # other_agent lidar end
        
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
        Extract HAZARD lidar information from observation for obstacle detection.
        
        CRITICAL: Must use hazard lidar (obs[44:60]), NOT goal lidar (obs[12:44])!
        Lidar values are in [0, 1] where higher values = closer objects.
        
        v4.0 FIX: Return proximity (max_lidar) directly instead of fake distance!
        The barrier potential should be computed based on lidar proximity,
        not on an unreliable distance conversion.
        
        Args:
            obs: [batch, obs_dim] observation tensor
            
        Returns:
            hazard_lidar: [batch, 16] hazard lidar readings
            proximity: [batch, 1] maximum proximity (0=far, 1=collision)
        """
        batch_size = obs.shape[0]
        obs_dim = obs.shape[-1]
        
        # === Extract HAZARD lidar (obs[44:60]) - obstacles/hazards ===
        hazard_end = min(self.hazard_lidar_end_idx, obs_dim)
        if self.hazard_lidar_start_idx < hazard_end:
            hazard_lidar = obs[:, self.hazard_lidar_start_idx:hazard_end]
        else:
            # Fallback: use zeros if indices are invalid
            hazard_lidar = torch.zeros(batch_size, self.num_lidar_bins, device=self.device)
        
        # === Extract OTHER AGENT lidar (obs[76:92]) - inter-agent collision ===
        agent_end = min(self.agent_lidar_end_idx, obs_dim)
        if self.agent_lidar_start_idx < agent_end:
            agent_lidar = obs[:, self.agent_lidar_start_idx:agent_end]
            # Combine hazard and agent lidar for full obstacle awareness
            # Use max to get the most dangerous reading from either source
            combined_lidar = torch.maximum(hazard_lidar, agent_lidar)
        else:
            # No agent lidar available, just use hazard lidar
            combined_lidar = hazard_lidar
        
        # Find maximum lidar reading (closest obstacle/agent)
        # Higher reading = closer object = MORE DANGEROUS
        max_lidar = combined_lidar.max(dim=-1, keepdim=True)[0]  # [batch, 1]
        
        # v4.0: Return proximity DIRECTLY, don't convert to fake distance!
        # proximity ∈ [0, 1]: 0 = safe (far), 1 = danger (collision)
        proximity = torch.clamp(max_lidar, min=0.0, max=1.0)
        
        return combined_lidar, proximity
    
    def _extract_goal_lidar_info(self, obs):
        """
        Extract GOAL lidar information from observation for task potential.
        
        Goal lidar is at obs[12:44] (goal_red + goal_blue).
        
        Args:
            obs: [batch, obs_dim] observation tensor
            
        Returns:
            goal_lidar: [batch, 32] goal lidar readings (red + blue)
            goal_proximity: [batch, 1] maximum goal proximity (closer = higher)
        """
        batch_size = obs.shape[0]
        obs_dim = obs.shape[-1]
        
        # Extract goal lidar (obs[12:44])
        goal_end = min(self.goal_lidar_end_idx, obs_dim)
        if self.goal_lidar_start_idx < goal_end:
            goal_lidar = obs[:, self.goal_lidar_start_idx:goal_end]
        else:
            goal_lidar = torch.zeros(batch_size, 2 * self.num_lidar_bins, device=self.device)
        
        # Maximum goal proximity
        goal_proximity = goal_lidar.max(dim=-1, keepdim=True)[0]
        
        return goal_lidar, goal_proximity
    
    def _compute_barrier_potential(self, obs):
        """
        Compute Barrier Lyapunov Function (BLF) based potential.
        
        v7.0: Simplified and smoother formula that doesn't saturate quickly.
        H_barrier = k * proximity^2 / (1 - proximity^2 + ε)
        
        This provides a smoother gradient and doesn't explode to clip_max immediately.
        
        Where proximity ∈ [0, 1]: 0 = safe (far), 1 = danger (collision)
        
        Args:
            obs: [batch, obs_dim] observation tensor
            
        Returns:
            H_barrier: [batch, 1] barrier potential
            grad_H_barrier: [batch, 2] gradient w.r.t. velocity (approximation)
        """
        # Get adaptive stiffness from network and scale it
        k_base = self.barrier_k_net(obs)  # [batch, 1], positive due to Softplus
        k = torch.clamp(k_base * self.barrier_k_scale + self.min_barrier_k, min=self.min_barrier_k)
        
        # Extract lidar info and proximity
        lidar_obs, proximity = self._extract_lidar_info(obs)  # proximity ∈ [0, 1]
        
        # v7.0: Simplified barrier formula with smoother behavior
        # Use squared proximity for smoother gradient near safety boundary
        proximity_sq = proximity ** 2
        safety_margin = torch.clamp(1.0 - proximity_sq, min=0.05)  # Larger min to prevent saturation
        
        # H_barrier = k * proximity^2 / safety_margin
        # This gives smoother gradient and doesn't explode as quickly
        H_barrier = k * proximity_sq / (safety_margin + self.barrier_epsilon)
        
        # Reduced clip for more gradual behavior
        H_barrier = torch.clamp(H_barrier, max=self.barrier_clip_max)
        
        # Compute gradient approximation
        vel = obs[:, self.vel_indices]  # [batch, 2]
        vel_norm = torch.norm(vel, dim=-1, keepdim=True) + 1e-6
        
        # v7.0: Smoother gradient magnitude
        # dH/d(proximity) ≈ 2k * proximity / safety_margin
        grad_magnitude = 2.0 * k * proximity / (safety_margin + self.barrier_epsilon)
        grad_magnitude = grad_magnitude * self.barrier_gradient_scale
        grad_magnitude = torch.clamp(grad_magnitude, max=10.0)  # Lower clip
        
        # Use lidar-weighted direction for obstacle avoidance
        num_bins = lidar_obs.shape[-1]
        angles = torch.linspace(0, 2 * np.pi, num_bins + 1, device=self.device)[:-1]
        angles = angles.unsqueeze(0).expand(obs.shape[0], -1)
        
        # Weighted sum of directions based on lidar readings
        weights = lidar_obs ** 2
        weights_sum = weights.sum(dim=-1, keepdim=True) + 1e-6
        
        obstacle_dir_x = (weights * torch.cos(angles)).sum(dim=-1, keepdim=True) / weights_sum
        obstacle_dir_y = (weights * torch.sin(angles)).sum(dim=-1, keepdim=True) / weights_sum
        
        # Gradient points away from obstacles (repulsive)
        grad_H_barrier_x = -grad_magnitude * obstacle_dir_x
        grad_H_barrier_y = -grad_magnitude * obstacle_dir_y
        grad_H_barrier = torch.cat([grad_H_barrier_x, grad_H_barrier_y], dim=-1)
        
        return H_barrier, grad_H_barrier
        
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
        
        v6.0: Balanced weighting - let the network learn the right balance.
        The gradient is used as a feature for the policy network, not for direct action control.
        
        H_total = H_task + H_barrier
        ∇H_total = ∇H_task + λ * ∇H_barrier
        
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
        
        # v6.0: Balanced adaptive weighting (not too aggressive)
        # Base weight is 1.0, increases to 3.0 when near obstacles
        barrier_base_weight = 1.0
        barrier_weight = barrier_base_weight + torch.clamp(H_barrier / 10.0, max=2.0)  # [1.0, 3.0]
        
        # Clip gradients for numerical stability
        grad_H_barrier_clipped = torch.clamp(grad_H_barrier, min=-10.0, max=10.0)
        grad_H_task_clipped = torch.clamp(grad_H_task, min=-5.0, max=5.0)
        
        # Combine gradients with balanced weighting
        grad_H_total = grad_H_task_clipped + barrier_weight * grad_H_barrier_clipped
        
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
        
        # === v6.0: NO forced action correction ===
        # The policy network learns from physics features (H_task, H_barrier, gradients, dynamics)
        # The network can learn to use these features to produce safe actions
        # Reward shaping in the trainer will guide the learning process
        
        # Create distribution with learned action mean (no correction)
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
        
        v5.0: Must use same potential-corrected distribution as forward() for consistency.
        
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
        
        # Compute action distribution (must match forward())
        action_mean = self.action_mean(policy_features)
        action_std = torch.sigmoid(self.log_std / self.std_x_coef) * self.std_y_coef
        action_std = action_std.expand_as(action_mean)
        
        # === v6.0: No action correction - matches forward() ===
        # Distribution uses learned action mean directly
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
        
        # v4.0: Get proximity instead of min_dist
        _, proximity = self._extract_lidar_info(obs)
        
        return {
            'H_task': H_task.detach(),
            'H_barrier': H_barrier.detach(),
            'grad_H_total': grad_H_total.detach(),
            'J': J.detach(),
            'R': R.detach(),
            'state': state.detach(),
            'proximity': proximity.detach(),  # v4.0: renamed from min_dist
            'min_dist': (1.0 - proximity).detach()  # Backward compatibility
        }

    def _apply_safety_correction(self, action, obs, H_barrier, grad_H_barrier):
        """
        Apply safety correction to action based on barrier potential gradient.
        
        v5.0: Enhanced safety layer that actively steers agents away from obstacles.
        Uses Control Barrier Function (CBF)-inspired approach with strong corrections.
        
        Args:
            action: [batch, act_dim] raw action from policy
            obs: [batch, obs_dim] observation
            H_barrier: [batch, 1] barrier potential
            grad_H_barrier: [batch, 2] barrier gradient
            
        Returns:
            corrected_action: [batch, act_dim] safety-corrected action
        """
        # v5.0: Safety layer is now ENABLED by default
        if not self.use_safety_layer:
            return action
        
        # Extract current velocity
        vel = obs[:, self.vel_indices]  # [batch, 2]
        vel_norm = torch.norm(vel, dim=-1, keepdim=True) + 1e-6
        vel_dir = vel / vel_norm
        
        # Extract lidar info for danger detection
        lidar_obs, proximity = self._extract_lidar_info(obs)
        
        # v5.0: More aggressive danger detection with lower threshold
        # danger_level: 0 when safe, 1 when very close to obstacle
        danger_level = torch.clamp(
            (proximity - self.danger_zone_threshold) / (1.0 - self.danger_zone_threshold + 1e-6), 
            min=0.0, max=1.0
        )
        
        # Also consider H_barrier as danger indicator
        H_barrier_normalized = H_barrier / (self.barrier_clip_max + 1e-6)
        combined_danger = torch.maximum(danger_level, H_barrier_normalized)
        
        # Normalize barrier gradient
        grad_norm = torch.norm(grad_H_barrier, dim=-1, keepdim=True) + 1e-6
        grad_dir = grad_H_barrier / grad_norm  # Direction of increasing potential (toward obstacles)
        
        # v5.0: Much stronger correction strength based on danger
        # Uses exponential scaling for more aggressive response near obstacles
        correction_strength = combined_danger * self.safety_gamma * (1.0 + 2.0 * combined_danger)
        correction_strength = torch.clamp(correction_strength, max=0.8)  # Max 80% correction
        
        # Compute dot product between velocity direction and gradient
        # Negative means we're moving toward obstacle (dangerous)
        vel_grad_dot = (vel_dir * grad_dir).sum(dim=-1, keepdim=True)
        moving_toward_danger = torch.clamp(-vel_grad_dot, min=0.0)  # How much moving toward danger
        
        # Correction mask: full correction when in danger and moving toward it
        correction_mask = combined_danger * (0.3 + 0.7 * moving_toward_danger)  # Always some correction in danger
        
        action_corrected = action.clone()
        
        # === Forward action correction ===
        # Reduce forward velocity when dangerous, proportional to danger level
        # v5.0: Can fully stop or even reverse if very dangerous
        forward_reduction = correction_strength * correction_mask
        action_corrected[:, 0:1] = action[:, 0:1] * (1.0 - forward_reduction * 0.8)
        
        # If danger is extreme (>0.8), actively slow down or reverse
        extreme_danger = (combined_danger > 0.8).float()
        action_corrected[:, 0:1] = action_corrected[:, 0:1] - extreme_danger * 0.3 * moving_toward_danger
        
        # === Turning action correction ===
        # Turn away from obstacle using cross product to determine direction
        cross = vel_dir[:, 0:1] * grad_dir[:, 1:2] - vel_dir[:, 1:2] * grad_dir[:, 0:1]
        
        # v5.0: Stronger turn correction proportional to danger and moving-toward-danger
        turn_correction = correction_strength * (0.5 + 0.5 * moving_toward_danger) * torch.sign(cross) * 0.8
        action_corrected[:, 1:2] = action[:, 1:2] + turn_correction
        
        # Clip corrected action to valid range
        action_corrected = torch.clamp(action_corrected, -1.0 + self.action_clip_margin, 
                                        1.0 - self.action_clip_margin)
        
        return action_corrected
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
PHS-MAPPO Actor v8.0: Learnable Policy with PHS Guidance.

CRITICAL CHANGES FROM v7.x:
    Previous versions had a fatal flaw: PHS gradients were DETACHED from the
    computation graph, meaning RL could only learn 4 scalar gain parameters.
    This severely limited policy expressiveness and prevented effective learning.

v8.0 Architecture - "PHS-Guided Learnable Policy":
    1. FULL POLICY NETWORK: A complete MLP that maps observation → action
       This provides full expressiveness for the RL algorithm to learn.
    
    2. PHS BIAS PRIOR: The PHS-computed goal/barrier gradients serve as
       an initial bias/prior, giving the policy a good starting point.
    
    3. LEARNABLE MIXING: A learned weight controls how much to rely on
       PHS guidance vs. the learned policy. Starts with more PHS guidance,
       gradually shifts to learned policy as training progresses.
    
Key Formula:
    action = α * policy_net(obs) + (1-α) * phs_bias + residual
    
    Where:
    - policy_net: Full MLP with gradient flow (LEARNABLE)
    - phs_bias: PHS-computed direction (provides good initialization)
    - α: Learnable mixing coefficient (starts ~0.3, can increase)
    - residual: Small exploration term

This ensures:
    - RL has full gradient flow through policy_net
    - PHS provides good inductive bias for safety and goal-seeking
    - The agent CAN learn, not just tune 4 parameters

For SafetyMultiGoal environments:
    - Observation: Point ~152-dim, Car ~176-dim
    - Action: 2-dim (Point: [forward, turn], Car: [left_wheel, right_wheel])
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
        - Observation: Point ~76-dim, Car ~100-dim (with ball joint sensors)
        - State: (q_pos, q_vel) = 4-dim
        - Action: 
            - Point: 2-dim (forward force, turning velocity)
            - Car: 2-dim (left wheel torque, right wheel torque) - differential drive!
    
    CRITICAL: Car uses differential drive (left/right wheel torques), not [forward, turn]!
    We must convert PHS output (forward, turn) → (left_wheel, right_wheel).
    """
    
    def __init__(self, config, obs_space, act_space, device=torch.device("cuda"), n_agents=1, agent_id=0):
        super().__init__()
        
        self.config = config
        self.device = device
        self.obs_dim = obs_space.shape[0]
        self.act_dim = act_space.shape[0]
        self.n_agents = n_agents
        self.agent_id = agent_id  # Which agent this actor belongs to
        
        # ===================
        # Agent Type Detection (Car vs Point)
        # ===================
        # Car has ball joint sensors (ballquat_rear: 9 dims, ballangvel_rear: 3 dims)
        # Total basic sensors: Car = 24 dims, Point = 12 dims
        # 
        # Observed dimensions for MultiGoal Level1:
        # - Point: 152 dims
        # - Car: 176 dims
        # Difference: 24 dims (exactly the ball joint sensors)
        #
        # Detection strategy:
        # 1. Check config for explicit agent_type
        # 2. Check if env_name contains 'Car' or 'Point'
        # 3. Use dimension difference (Car is always 24 dims larger for same env)
        self.agent_type = config.get("agent_type", "auto")
        if self.agent_type == "auto":
            # Try to get from environment name
            env_name = config.get("env_name", "")
            if "Car" in env_name:
                self.agent_type = "car"
            elif "Point" in env_name:
                self.agent_type = "point"
            elif "Racecar" in env_name:
                self.agent_type = "car"  # Racecar also uses differential drive
            elif "Doggo" in env_name or "Ant" in env_name:
                self.agent_type = "point"  # Use point-style control for these
            else:
                # Fallback to dimension-based detection
                # Car has 176 dims, Point has 152 dims for MultiGoal Level1
                if self.obs_dim >= 170:
                    self.agent_type = "car"
                else:
                    self.agent_type = "point"
        
        # Base sensor offset (before lidar starts)
        if self.agent_type == "car":
            self.base_sensor_dim = 24  # accelerometer(3) + velocimeter(3) + gyro(3) + magnetometer(3) + ballangvel(3) + ballquat(9)
        else:  # point
            self.base_sensor_dim = 12  # accelerometer(3) + velocimeter(3) + gyro(3) + magnetometer(3)
        
        # ===================
        # PHS Configuration
        # ===================
        self.hidden_size = config.get("hidden_size", 256)
        self.physics_hidden = config.get("physics_hidden", 128)
        self.state_dim = 4  # (x, y, vx, vy) or (vx, vy, ax, ay)
        
        # Physical parameters (v7.0)
        self.f_max = config.get("phs_f_max", 1.0)  # Max control force
        self.drag = config.get("phs_drag", 0.1)  # Base damping
        self.dt = config.get("phs_dt", 0.05)  # Time step
        
        # Barrier potential parameters
        self.r_collision = config.get("r_collision", 0.17)  # Collision radius
        self.r_communication = config.get("r_communication", 0.45)  # Communication radius
        self.barrier_epsilon = config.get("barrier_epsilon", 0.06)  # Numerical stability
        self.fixed_barrier_potential = config.get("fixed_barrier_potential", True)
        self.obstacle_barrier_k = config.get("obstacle_barrier_k", 1.0)
        self.obstacle_barrier_alpha = config.get("obstacle_barrier_alpha", 4.0)
        self.obstacle_barrier_threshold = config.get("obstacle_barrier_threshold", 0.75)
        self.obstacle_barrier_scale = config.get("obstacle_barrier_scale", 10.0)
        self.agent_barrier_k = config.get("agent_barrier_k", 1.2)
        
        # Potential weights (v7.4 - Immediate barrier activation)
        self.task_weight = config.get("task_weight", 1.0)
        self.barrier_weight = config.get("barrier_weight", 1.0)  # Increased from 0.5
        self.barrier_weight_max = config.get("barrier_weight_max", 1.5)  # Increased
        self.obstacle_barrier_weight = config.get("obstacle_barrier_weight", 1.0)  # Increased from 0.5
        
        # Barrier warmup parameters (v8.2 - gradual activation)
        self.barrier_warmup_steps = config.get("barrier_warmup_steps", 0)
        self.barrier_decay_start = config.get("barrier_decay_start", 5000)
        self.barrier_decay_rate = config.get("barrier_decay_rate", 0.95)
        self._training_step = 0

        # Decouple Barrier PHS from Lagrange learning
        self.decouple_barrier_lagrange = config.get("decouple_barrier_lagrange", True)
        self.phs_prior_weight = config.get("phs_prior_weight", 0.0)
        self.phs_goal_guidance_weight = config.get("phs_goal_guidance_weight", 0.35)
        self.phs_barrier_guidance_weight = config.get("phs_barrier_guidance_weight", 0.15)
        
        # Multi-agent scaling
        self.auto_scale_by_agents = config.get("auto_scale_by_agents", True)
        
        # ===================
        # Observation Indices (Dynamic based on agent type)
        # ===================
        # Common sensors (same position for both):
        # obs[0:3] = accelerometer (ax, ay, az)
        # obs[3:6] = velocimeter (vx, vy, vz)
        # obs[6:9] = gyro
        # obs[9:12] = magnetometer
        # 
        # Car only (after magnetometer):
        # obs[12:15] = ballangvel_rear
        # obs[15:24] = ballquat_rear (flattened 3x3 rotation matrix)
        #
        # Lidar starts at base_sensor_dim:
        # Point: obs[12:...] = lidar data
        # Car: obs[24:...] = lidar data
        
        self.vel_indices = [3, 4]  # vx, vy
        self.acc_indices = [0, 1]  # ax, ay
        self.magnetometer_indices = [9, 10]  # cos(θ), sin(θ) for orientation
        
        # Lidar configuration (each lidar has 16 bins)
        lidar_bins = 16
        lidar_start = self.base_sensor_dim
        
        # ACTUAL MultiGoal lidar order (verified from _obstacles):
        # 1. goal_red (16 bins)
        # 2. goal_blue (16 bins)
        # 3. hazards (16 bins)
        # 4. vases (16 bins)
        # 5. agents (16 bins, if present)
        
        # Lidar configuration (each lidar has 16 bins)
        lidar_bins = 16
        
        # Goal lidars come FIRST
        goal_red_start = lidar_start
        goal_red_end = goal_red_start + lidar_bins
        goal_blue_start = goal_red_end
        goal_blue_end = goal_blue_start + lidar_bins
        
        # CRITICAL: Each agent should focus on its own goal!
        # Agent 0 -> goal_red, Agent 1 -> goal_blue
        if self.agent_id == 0:
            self.goal_lidar_start = goal_red_start
            self.goal_lidar_end = goal_red_end  # goal_red only
        else:
            self.goal_lidar_start = goal_blue_start
            self.goal_lidar_end = goal_blue_end  # goal_blue only
        
        # Hazards lidar follows goals
        self.hazard_lidar_start = goal_blue_end
        self.hazard_lidar_end = self.hazard_lidar_start + lidar_bins
        
        # Vases lidar follows hazards
        self.vases_lidar_start = self.hazard_lidar_end
        self.vases_lidar_end = self.vases_lidar_start + lidar_bins
        
        # Agent lidar (other agents) may follow vases
        self.agent_lidar_start = self.vases_lidar_end
        self.agent_lidar_end = self.agent_lidar_start + lidar_bins
        
        # Log detected configuration
        print(f"[PHS-MAPPO v8.0] Agent {agent_id}: type={self.agent_type}, obs_dim={self.obs_dim}, base_sensors={self.base_sensor_dim}")
        print(f"[PHS-MAPPO v8.0] Lidar indices: goal=[{self.goal_lidar_start}:{self.goal_lidar_end}], "
              f"hazard=[{self.hazard_lidar_start}:{self.hazard_lidar_end}], agent=[{self.agent_lidar_start}:{self.agent_lidar_end}]")
        
        # ===================
        # Network Modules (v8.1 - Pure Learnable Policy with PHS Features)
        # ===================
        # 
        # v8.1 KEY INSIGHT: Use PHS gradients as INPUT features, not output bias.
        # This gives the policy physics information while maintaining clean RL optimization.
        
        # ========== 1. Feature Extraction (Shared) ==========
        self.feature_norm = nn.LayerNorm(self.obs_dim)
        
        # Feature encoder - maps observation to hidden representation
        self.state_encoder = nn.Sequential(
            nn.Linear(self.obs_dim, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.ELU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.ELU(),
            nn.Linear(self.hidden_size, self.physics_hidden),
            nn.ELU(),
        )
        
        # ========== 2. MAIN POLICY NETWORK (v8.1 - Direct Output) ==========
        # Standard MLP policy with PHS directional features
        # Input: physics_hidden features + goal/barrier gradients (4 dims)
        # Output: action directly
        policy_hidden = self.hidden_size * 2  # 512 if hidden_size=256
        
        self.policy_net = nn.Sequential(
            nn.Linear(self.physics_hidden + 4, policy_hidden),
            nn.LayerNorm(policy_hidden),
            nn.ELU(),
            nn.Linear(policy_hidden, policy_hidden),
            nn.LayerNorm(policy_hidden),
            nn.ELU(),
            nn.Linear(policy_hidden, policy_hidden // 2),
            nn.ELU(),
            nn.Linear(policy_hidden // 2, self.act_dim),
        )
        
        # ========== 3. PHS Guidance Network (Simplified) ==========
        # Computes PHS-based action bias (goal-seeking + obstacle-avoidance)
        # This provides INDUCTIVE BIAS, not the main policy!
        self.phs_gain_net = nn.Sequential(
            nn.Linear(self.physics_hidden + 4, self.hidden_size // 2),  # +4 for goal/barrier gradients
            nn.ELU(),
            nn.Linear(self.hidden_size // 2, self.act_dim),
        )
        
        # ========== 4. Mixing Coefficient (Policy vs PHS) ==========
        # Learnable parameter controlling balance between policy_net and PHS bias
        # Initialized so policy has more weight from the start
        self.policy_weight = nn.Parameter(torch.tensor(1.0))  # sigmoid(1.0) ≈ 0.73 for policy
        
        # ========== 5. Residual for Exploration ==========
        self.residual_mlp = nn.Sequential(
            nn.Linear(self.physics_hidden, self.hidden_size // 2),
            nn.ELU(),
            nn.Linear(self.hidden_size // 2, self.act_dim),
        )
        self.residual_weight = nn.Parameter(torch.tensor(0.0))  # sigmoid(0) = 0.5
        
        # ========== 6. Standard Deviation Network ==========
        self.std_net = nn.Sequential(
            nn.Linear(self.physics_hidden + self.act_dim, self.hidden_size // 2),
            nn.ELU(),
            nn.Linear(self.hidden_size // 2, self.act_dim)
        )
        
        # ========== Legacy Networks (Kept for compatibility) ==========
        self.J_net = AttentionLEMURS(self.physics_hidden, self.physics_hidden, self.state_dim * self.state_dim)
        self.R_net = AttentionLEMURS(self.physics_hidden, self.physics_hidden, self.state_dim * self.state_dim)
        self.H_task_net = AttentionLEMURS(self.physics_hidden, self.physics_hidden, 1)
        self.H_barrier_head = SoftBarrierHead(self.physics_hidden, hidden_dim=64)
        self.obstacle_k_net = nn.Sequential(
            nn.Linear(self.obs_dim, self.physics_hidden // 2),
            nn.ELU(),
            nn.Linear(self.physics_hidden // 2, 1),
            nn.Softplus()
        )
        self.barrier_shape_net = nn.Sequential(
            nn.Linear(self.obs_dim, self.physics_hidden // 4),
            nn.ELU(),
            nn.Linear(self.physics_hidden // 4, 2),
            nn.Tanh()
        )
        
        # Precompute base system matrices (Port-Hamiltonian structure)
        self._init_base_phs_matrices()
        
        # Initialize weights
        self._init_weights()
        
        self.to(device)

        # Freeze barrier networks when decoupled or fixed barrier potential
        if self.decouple_barrier_lagrange or self.fixed_barrier_potential:
            for net in (self.obstacle_k_net, self.barrier_shape_net, self.H_barrier_head):
                for param in net.parameters():
                    param.requires_grad_(False)
        
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
        
        # Initialize main policy network (v8.0)
        for m in self.policy_net:
            if isinstance(m, nn.Linear):
                init_layer(m, gain=1.0)  # Standard gain for main policy
        
        # Initialize PHS guidance network
        for m in self.phs_gain_net:
            if isinstance(m, nn.Linear):
                init_layer(m, gain=0.5)
        
        # Initialize residual MLP with smaller gain
        for m in self.residual_mlp:
            if isinstance(m, nn.Linear):
                init_layer(m, gain=0.3)
            
        for m in self.obstacle_k_net:
            if isinstance(m, nn.Linear):
                init_layer(m, gain=0.1)
        
        # Initialize barrier shape network with small weights for stable start
        for m in self.barrier_shape_net:
            if isinstance(m, nn.Linear):
                init_layer(m, gain=0.01)
        
        for m in self.std_net:
            init_layer(m, gain=0.1)
    
    def _get_current_barrier_weight(self):
        """
        Compute current barrier weight based on training step.
        
        v8.2: Warmup → Plateau → Decay

        - Phase 0 [0, warmup]: ramp from barrier_weight to barrier_weight_max
        - Phase 1 [warmup, decay_start]: plateau at barrier_weight_max
        - Phase 2 [decay_start, ∞]: gradual decay to barrier_weight * decay_rate
        """
        step = self._training_step

        if self.barrier_warmup_steps > 0 and step < self.barrier_warmup_steps:
            # Linear warmup from barrier_weight to barrier_weight_max
            warmup_ratio = step / float(self.barrier_warmup_steps)
            return self.barrier_weight + (self.barrier_weight_max - self.barrier_weight) * warmup_ratio

        if step < self.barrier_decay_start:
            return self.barrier_weight_max

        # Decay phase: slow exponential decay
        decay_steps = step - self.barrier_decay_start
        decay_factor = np.exp(-decay_steps / 2000.0)
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
        # Extract goal lidar (agent-specific: red for agent 0, blue for agent 1)
        goal_end = min(self.goal_lidar_end, obs.shape[-1])
        goal_lidar = obs[..., self.goal_lidar_start:goal_end]  # [batch, ..., 16]
        
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
        Compute obstacle barrier potential using fixed exponential barrier.
        
        H_barrier = k * (exp(α * proximity) - 1) only when proximity > threshold
        
        Fixed Structure (Barrier PHS Actor):
        - k, alpha, threshold, scale are constants
        - No learnable shape or stiffness
        """
        # Extract hazard lidar
        hazard_end = min(self.hazard_lidar_end, obs.shape[-1])
        hazard_lidar = obs[..., self.hazard_lidar_start:hazard_end]  # [batch, ..., 16]
        
        # Max proximity (higher = closer = more dangerous)
        max_proximity = hazard_lidar.max(dim=-1, keepdim=True)[0]
        
        # Fixed parameters (no learning)
        k_base = torch.tensor(self.obstacle_barrier_k, device=obs.device, dtype=obs.dtype)
        k_base = k_base.view(*([1] * (max_proximity.dim() - 1)), 1)

        alpha = torch.tensor(self.obstacle_barrier_alpha, device=obs.device, dtype=obs.dtype)
        alpha = alpha.view(*([1] * (max_proximity.dim() - 1)), 1)

        activation_threshold = torch.tensor(self.obstacle_barrier_threshold, device=obs.device, dtype=obs.dtype)
        activation_threshold = activation_threshold.view(*([1] * (max_proximity.dim() - 1)), 1)
        activation_threshold = torch.clamp(activation_threshold, min=0.65, max=0.85)

        scale = torch.tensor(self.obstacle_barrier_scale, device=obs.device, dtype=obs.dtype)
        scale = scale.view(*([1] * (max_proximity.dim() - 1)), 1)
        
        # Normalized proximity range after threshold
        effective_range = 1.0 - activation_threshold  # ~0.20-0.35
        
        # Shift proximity by threshold and normalize to [0, 1]
        shifted_proximity = torch.clamp(
            (max_proximity - activation_threshold) / effective_range, 
            min=0.0, max=1.0
        )
        
        # Exponential barrier: H = scale * (exp(α * shifted) - 1) / (exp(α) - 1)
        exp_term = torch.exp(torch.clamp(alpha * shifted_proximity, max=8.0))
        exp_alpha = torch.exp(torch.clamp(alpha, max=8.0))

        H_barrier = k_base * scale * (exp_term - 1.0) / (exp_alpha - 1.0 + 1e-6)
        
        # Explicitly zero out where proximity is below activation threshold
        H_barrier = torch.where(max_proximity < activation_threshold, torch.zeros_like(H_barrier), H_barrier)
        
        # Clamp to reasonable range and check for NaN
        H_barrier = torch.clamp(H_barrier, min=0.0, max=float(self.obstacle_barrier_scale))
        H_barrier = torch.where(torch.isnan(H_barrier), torch.zeros_like(H_barrier), H_barrier)
        
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
        
        # Fixed pairwise stiffness (no learning)
        k_val = torch.tensor(self.agent_barrier_k, device=obs.device, dtype=obs.dtype)
        k_ij = k_val * laplacian
        
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
    
    def _compute_hamiltonian_gradient(self, obs, state_features, laplacian, detach=True):
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
        
        if detach:
            H_total_out = H_total.detach()
            grad_H_vel_out = grad_H_vel.detach()
            H_goal_out = H_goal.detach()
            H_task_learned_out = H_task_learned.detach()
            H_task_out = H_task.detach()
            H_kin_out = H_kin.detach()
            H_barrier_obs_out = H_barrier_obs.detach()
            H_barrier_agent_out = H_barrier_agent.detach() if is_multi_agent else None
        else:
            H_total_out = H_total
            grad_H_vel_out = grad_H_vel
            H_goal_out = H_goal
            H_task_learned_out = H_task_learned
            H_task_out = H_task
            H_kin_out = H_kin
            H_barrier_obs_out = H_barrier_obs
            H_barrier_agent_out = H_barrier_agent if is_multi_agent else None

        return H_total_out, grad_H_vel_out, {
            'H_goal': H_goal_out,
            'H_task_learned': H_task_learned_out,
            'H_task': H_task_out,
            'H_kin': H_kin_out,
            'H_barrier_obs': H_barrier_obs_out,
            'H_barrier_agent': H_barrier_agent_out,
            'barrier_weight': current_barrier_weight
        }
    
    def _compute_directional_gradient(self, obs):
        """
        Compute directional gradients for goal-seeking and obstacle avoidance.
        
        CRITICAL FIX (v7.1): Output in BODY-FRAME coordinates!
        
        Point agent action space:
            action[0] = forward force (along agent's heading)
            action[1] = turning velocity
        
        So we must convert world-frame directions to body-frame commands:
            forward = dot(world_dir, heading_vector)
            turning = cross(heading_vector, world_dir) = sin(angle_diff)
        
        Returns:
            goal_gradient: [batch, 2] (forward, turn) toward goal
            barrier_gradient: [batch, 2] (forward, turn) away from obstacles
        """
        device = obs.device
        
        # NOTE: Observations are raw in MultiGoalEnv wrapper.
        # Lidar values are in [0, 1], higher means closer objects.
        
        # ========== Get Agent Orientation from Magnetometer ==========
        # magnetometer gives (cos θ, sin θ) where θ is heading angle
        cos_theta = obs[..., self.magnetometer_indices[0]:self.magnetometer_indices[0]+1]  # [batch, 1]
        sin_theta = obs[..., self.magnetometer_indices[1]:self.magnetometer_indices[1]+1]  # [batch, 1]
        
        # ========== Goal Direction (Body Frame) ==========
        # Each agent uses its own goal lidar (red for agent 0, blue for agent 1)
        goal_end = min(self.goal_lidar_end, obs.shape[-1])
        goal_lidar = obs[..., self.goal_lidar_start:goal_end]  # [batch, ..., 16]
        
        num_goal_bins = goal_lidar.shape[-1]
        # Lidar bins are in BODY frame, spanning 360 degrees
        # bin 0 is typically forward, angles increase counter-clockwise
        goal_angles_body = torch.linspace(0, 2 * np.pi, num_goal_bins + 1, device=device)[:-1]
        
        # Use centered logits for stable weighting on raw lidar
        goal_logits = (goal_lidar - 0.5) * 6.0
        goal_weights = F.softmax(goal_logits, dim=-1)
        
        # Weighted average direction in BODY frame
        goal_dir_forward = (goal_weights * torch.cos(goal_angles_body)).sum(dim=-1, keepdim=True)
        goal_dir_lateral = (goal_weights * torch.sin(goal_angles_body)).sum(dim=-1, keepdim=True)
        
        # Goal proximity for scaling (use max value, clamp to reasonable range)
        goal_proximity_scaled = torch.clamp(goal_lidar.max(dim=-1, keepdim=True)[0], min=0.0, max=1.0)
        
        # Body-frame goal gradient:
        # forward = how much goal is in front (cos of angle to goal)
        # turn = how much we need to turn toward goal (sin of angle to goal)
        goal_magnitude = torch.sqrt(goal_dir_forward**2 + goal_dir_lateral**2 + 1e-6)
        goal_gradient = torch.cat([
            goal_dir_forward / goal_magnitude,   # Forward component (normalized)
            goal_dir_lateral / goal_magnitude    # Turn component (normalized)
        ], dim=-1) * (0.3 + 0.7 * goal_proximity_scaled)  # Scale by proximity
        
        # ========== Barrier Direction (Body Frame) ==========
        hazard_end = min(self.hazard_lidar_end, obs.shape[-1])
        hazard_lidar = obs[..., self.hazard_lidar_start:hazard_end]  # [batch, ..., 16]
        
        num_hazard_bins = hazard_lidar.shape[-1]
        hazard_angles_body = torch.linspace(0, 2 * np.pi, num_hazard_bins + 1, device=device)[:-1]
        
        # Use centered logits for stable weighting on raw lidar
        hazard_logits = (hazard_lidar - 0.5) * 6.0
        hazard_weights = F.softmax(hazard_logits, dim=-1)
        
        # Direction TOWARD hazards in body frame
        hazard_dir_forward = (hazard_weights * torch.cos(hazard_angles_body)).sum(dim=-1, keepdim=True)
        hazard_dir_lateral = (hazard_weights * torch.sin(hazard_angles_body)).sum(dim=-1, keepdim=True)
        
        # Barrier gradient points AWAY from hazards (negate direction)
        # If hazard is in front: go backward (negative forward)
        # If hazard is on left: turn right (negative turn)
        max_hazard = torch.clamp(hazard_lidar.max(dim=-1, keepdim=True)[0], min=0.0, max=1.0)

        # Repulsion strength increases quadratically near hazards
        repulsion_strength = max_hazard.pow(2)
        
        hazard_magnitude = torch.sqrt(hazard_dir_forward**2 + hazard_dir_lateral**2 + 1e-6)
        barrier_gradient = torch.cat([
            -hazard_dir_forward / hazard_magnitude,  # Backward if hazard in front
            -hazard_dir_lateral / hazard_magnitude   # Turn away from hazard
        ], dim=-1) * repulsion_strength
        
        return goal_gradient, barrier_gradient, goal_proximity_scaled, repulsion_strength
    
    def _compute_phs_action(self, obs, state_features, laplacian):
        """
        Compute action using Barrier PHS Actor (true port-Hamiltonian action mapping).
        
        Action generation:
            dx_target = π_θ([features]) + guidance
            dx = (J - R) ∇H_total + F * a
            a = F^+ (dx_target - (J - R) ∇H_total)
        
        This injects Barrier PHS structure directly into action generation.
        """
        batch_size = obs.shape[0]
        is_multi_agent = obs.dim() == 3
        
        if is_multi_agent:
            n_agents = obs.shape[1]
            obs_flat = obs.view(batch_size * n_agents, -1)
            state_features_flat = state_features.view(batch_size * n_agents, -1)
        else:
            obs_flat = obs
            state_features_flat = state_features
            n_agents = 1
        
        # ========== 1. Compute Directional Gradients (as features + guidance) ==========
        goal_grad, barrier_grad, goal_prox, hazard_prox = self._compute_directional_gradient(obs_flat)

        # ========== 2. Policy Output (desired state change) ==========
        policy_input = torch.cat([state_features_flat, goal_grad, barrier_grad], dim=-1)
        policy_output = self.policy_net(policy_input)  # [batch, act_dim]

        residual = self.residual_mlp(state_features_flat)
        residual_w = torch.sigmoid(self.residual_weight) * 0.3

        # Guidance encourages goal progress and basic avoidance
        dx_guidance = (
            self.phs_goal_guidance_weight * goal_grad +
            self.phs_barrier_guidance_weight * barrier_grad
        )

        dx_target_body = policy_output + residual_w * residual + dx_guidance

        # ========== 3. PHS Drift (Barrier + Task) ==========
        if is_multi_agent:
            H_total, grad_H_vel, H_info = self._compute_hamiltonian_gradient(
                obs, state_features, laplacian, detach=False
            )
            grad_H_vel = grad_H_vel.view(batch_size * n_agents, -1)
        else:
            H_total, grad_H_vel, H_info = self._compute_hamiltonian_gradient(
                obs, state_features, laplacian, detach=False
            )

        # Assemble gradient in state space (pos part = 0, vel part = grad_H_vel)
        grad_H_state = torch.zeros(dx_target_body.shape[0], self.state_dim, device=obs.device, dtype=obs.dtype)
        grad_H_state[:, -self.act_dim:] = grad_H_vel

        J_R = self.J_sys - self.R_sys
        phs_drift = torch.matmul(J_R, grad_H_state.unsqueeze(-1)).squeeze(-1)

        # Desired state change in state space
        dx_target_state = torch.zeros_like(grad_H_state)
        dx_target_state[:, :self.act_dim] = dx_target_body

        # Compute port action a via pseudo-inverse of F
        u_body = torch.matmul(self.F_pinv, (dx_target_state - phs_drift).unsqueeze(-1)).squeeze(-1)
        
        # ========== 4. Convert to Agent-Specific Action Space ==========
        if self.agent_type == "car":
            # Differential drive: [forward, turn] -> [left_wheel, right_wheel]
            forward = u_body[:, 0:1]
            turn = u_body[:, 1:2]
            
            # Simple conversion without complex clamping
            turn_mix = 0.6
            left_wheel = forward + turn_mix * turn
            right_wheel = forward - turn_mix * turn
            
            u_mean = torch.cat([left_wheel, right_wheel], dim=-1)
        else:
            u_mean = u_body
        
        # ========== 5. Scale ==========
        u_mean = torch.tanh(u_mean) * self.f_max
        
        # Reshape if multi-agent
        if is_multi_agent:
            u_mean = u_mean.view(batch_size, n_agents, -1)
        
        # ========== 6. Logging Info ==========
        barrier_weight = self._get_current_barrier_weight()

        H_info.update({
            'barrier_weight': barrier_weight,
            'goal_prox': goal_prox.mean().item(),
            'hazard_prox': hazard_prox.mean().item(),
            'agent_type': self.agent_type,
            'policy_output_mean': policy_output.mean().item(),
            'goal_grad_forward': goal_grad[:, 0].mean().item(),
            'goal_grad_turn': goal_grad[:, 1].mean().item(),
            'dx_target_mean': dx_target_body.mean().item(),
        })
        
        return u_mean, H_info, state_features
    
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
        
        # Compute directional gradients (v7.1)
        if is_multi_agent:
            obs_flat = obs.view(-1, obs.shape[-1])
        else:
            obs_flat = obs
        goal_grad, barrier_grad, goal_prox, hazard_prox = self._compute_directional_gradient(obs_flat)
        
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
            'barrier_weight': H_info['barrier_weight'],
            # v7.1 Debug info
            'goal_grad_forward': goal_grad[:, 0].detach(),
            'goal_grad_turn': goal_grad[:, 1].detach(),
            'barrier_grad_forward': barrier_grad[:, 0].detach(),
            'barrier_grad_turn': barrier_grad[:, 1].detach(),
            'goal_prox': goal_prox.detach(),
            'hazard_prox': hazard_prox.detach(),
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
        
        if self.fixed_barrier_potential:
            k = torch.tensor(self.obstacle_barrier_k, device=obs.device, dtype=obs.dtype)
            k = k.view(*([1] * (hazard_lidar.dim() - 1)), 1)
        else:
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

    def get_obstacle_k(self, obs):
        """Return obstacle stiffness (fixed or learnable)."""
        if self.fixed_barrier_potential:
            k = torch.tensor(self.obstacle_barrier_k, device=obs.device, dtype=obs.dtype)
            return k.view(*([1] * (obs.dim() - 1)), 1)
        return self.obstacle_k_net(obs)

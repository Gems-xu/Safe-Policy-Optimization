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
    if isinstance(output, torch.Tensor) and output.dtype == torch.float64:
        output = output.float()
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
        # Task Type Detection (MultiGoal vs Velocity/MuJoCo)
        # ===================
        # Velocity tasks (MAMuJoCo) have different observation structure and action dims
        # MultiGoal tasks have lidar observations and 2D action space (forward, turn)
        env_name = config.get("env_name", "")
        self.is_velocity_task = (
            "Velocity" in env_name or 
            "Ant" in env_name and "MultiGoal" not in env_name or
            "HalfCheetah" in env_name or 
            "Hopper" in env_name or
            "Walker" in env_name or
            "Swimmer" in env_name or
            "Humanoid" in env_name and "MultiGoal" not in env_name
        )
        self.is_halfcheetah_velocity = self.is_velocity_task and ("HalfCheetah" in env_name)
        self.is_ant_velocity = self.is_velocity_task and ("Ant" in env_name)
        
        # For Velocity tasks, we use Barrier PHS with R matrix for velocity safety control
        if self.is_velocity_task:
            print(f"[Barrier-PHS v8.0] Agent {agent_id}: VELOCITY TASK detected (env={env_name})")
            print(f"[Barrier-PHS v8.0] Using R-matrix based PHS for velocity safety control")
            print(f"[Barrier-PHS v8.0] H = 0.5*v^T*M*v (kinetic), R = R0 + R_learned (adaptive damping)")
        
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
        self.drag = config.get("phs_drag", 0.1)  # Base damping (R0 diagonal)
        self.dt = config.get("phs_dt", 0.05)  # Time step
        
        # Velocity Task: R matrix configuration
        # R = R0 + R_learned where R0 is base damping, R_learned is adaptive
        self.velocity_r_base = config.get("velocity_r_base", 0.1)  # R0 base value
        self.velocity_r_max = config.get("velocity_r_max", 1.0)  # Max R_learned scale
        self.velocity_safety_threshold = config.get("velocity_safety_threshold", 0.8)  # Speed threshold for safety
        self.velocity_posture_threshold = config.get("velocity_posture_threshold", 0.35)
        self.velocity_posture_r_scale = config.get("velocity_posture_r_scale", 1.2)
        self.velocity_posture_gate = config.get("velocity_posture_gate", 0.5)
        self.velocity_posture_correction_weight = config.get("velocity_posture_correction_weight", 0.25)
        self.velocity_posture_correction_max = config.get("velocity_posture_correction_max", 0.5)
        self.velocity_policy_gate_floor = config.get("velocity_policy_gate_floor", 0.2)
        self.velocity_stability_threshold = config.get("velocity_stability_threshold", 0.6)
        self.velocity_stability_r_scale = config.get("velocity_stability_r_scale", 2.0)
        self.velocity_coordination_weight = config.get("velocity_coordination_weight", 0.3)
        self.velocity_phs_blend_base = config.get("velocity_phs_blend_base", 0.25)
        self.velocity_phs_blend_risk_scale = config.get("velocity_phs_blend_risk_scale", 0.65)
        self.velocity_phs_comp_max = config.get("velocity_phs_comp_max", 0.5)
        self.velocity_pitch_threshold = config.get("velocity_pitch_threshold", 0.45)
        self.velocity_pitch_r_scale = config.get("velocity_pitch_r_scale", 1.8)
        self.velocity_pitch_gate = config.get("velocity_pitch_gate", 0.6)
        self.velocity_speed_r_scale = config.get("velocity_speed_r_scale", 1.4)
        self.velocity_speed_gate = config.get("velocity_speed_gate", 0.5)
        self.velocity_energy_r_scale = config.get("velocity_energy_r_scale", 2.0)
        self.velocity_directional_r_scale = config.get("velocity_directional_r_scale", 1.2)
        self.velocity_r_total_max = config.get("velocity_r_total_max", 8.0)
        self.velocity_preemptive_ratio = config.get("velocity_preemptive_ratio", 0.86)
        self.velocity_preemptive_r_scale = config.get("velocity_preemptive_r_scale", 1.5)
        self.velocity_thigh_r_relief = config.get("velocity_thigh_r_relief", 0.20)
        self.velocity_distal_r_boost = config.get("velocity_distal_r_boost", 0.30)
        self.velocity_thigh_action_gain = config.get("velocity_thigh_action_gain", 1.18)
        self.velocity_distal_action_gain = config.get("velocity_distal_action_gain", 0.92)
        self.velocity_front_action_boost = config.get("velocity_front_action_boost", 1.16)
        self.velocity_height_threshold = config.get("velocity_height_threshold", 0.55)
        self.velocity_height_r_scale = config.get("velocity_height_r_scale", 2.2)
        self.velocity_pitch_rate_threshold = config.get("velocity_pitch_rate_threshold", 1.2)
        self.velocity_pitch_rate_r_scale = config.get("velocity_pitch_rate_r_scale", 1.8)
        self.velocity_back_thigh_target = config.get("velocity_back_thigh_target", -0.20)
        self.velocity_front_thigh_target = config.get("velocity_front_thigh_target", 0.60)
        self.velocity_thigh_target_gain = config.get("velocity_thigh_target_gain", 0.35)
        self.velocity_thigh_target_max = config.get("velocity_thigh_target_max", 0.45)
        self.velocity_thigh_recovery_gain = config.get("velocity_thigh_recovery_gain", 1.2)
        self.velocity_thigh_recovery_threshold = config.get("velocity_thigh_recovery_threshold", 0.05)
        self.velocity_front_lift_bias = config.get("velocity_front_lift_bias", 0.18)
        self.velocity_back_push_bias = config.get("velocity_back_push_bias", 0.12)
        self.velocity_front_distal_r_relief = config.get("velocity_front_distal_r_relief", 0.45)
        self.velocity_front_shin_abs_target = config.get("velocity_front_shin_abs_target", 0.35)
        self.velocity_front_foot_abs_target = config.get("velocity_front_foot_abs_target", 0.25)
        self.velocity_front_distal_extension_gain = config.get("velocity_front_distal_extension_gain", 1.8)
        self.velocity_front_distal_extension_max = config.get("velocity_front_distal_extension_max", 0.50)
        self.velocity_control_warmup_steps = int(config.get("velocity_control_warmup_steps", 0))
        self.velocity_front_distal_warmup_steps = int(
            config.get("velocity_front_distal_warmup_steps", self.velocity_control_warmup_steps)
        )
        
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
        
        # ===================
        # Lidar Configuration (MultiGoal Only)
        # ===================
        # For Velocity tasks, there's no lidar data - set empty ranges
        if self.is_velocity_task:
            # Set all lidar indices to empty ranges (start >= end)
            self.goal_lidar_start = 0
            self.goal_lidar_end = 0
            self.hazard_lidar_start = 0
            self.hazard_lidar_end = 0
            self.vases_lidar_start = 0
            self.vases_lidar_end = 0
            self.agent_lidar_start = 0
            self.agent_lidar_end = 0
            print(f"[PHS-MAPPO v8.0] Agent {agent_id}: Velocity task - no lidar data")
        else:
            # MultiGoal task: Configure lidar indices
            # Lidar configuration (each lidar has 16 bins)
            lidar_bins = 16
            lidar_start = self.base_sensor_dim
            
            # ACTUAL MultiGoal lidar order (verified from _obstacles):
            # 1. goal_red (16 bins)
            # 2. goal_blue (16 bins)
            # 3. hazards (16 bins)
            # 4. vases (16 bins)
            # 5. agents (16 bins, if present)
            
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
        
        # ========== 0. Velocity Task: Learnable R Matrix Network ==========
        # For Velocity tasks, the core of Barrier PHS is the R matrix:
        # R = R0 + R_learned(state)
        # - R0: Fixed base damping (energy dissipation)
        # - R_learned: Adaptive damping for velocity safety control
        # H = 0.5 * v^T * M * v (pure kinetic energy)
        # grad_H = v (velocity itself)
        if self.is_velocity_task:
            # Infer qpos/qvel split for MuJoCo velocity tasks (before networks)
            self.velocity_state_raw_dim = max(self.obs_dim - self.n_agents, 1)
            self.velocity_n_qpos = self.velocity_state_raw_dim // 2
            self.velocity_n_qvel = max(self.velocity_state_raw_dim - self.velocity_n_qpos, 1)
            
            # NEW: Per-agent coordination for multi-agent velocity tasks
            # When n_agents > 1, agents need to coordinate their actions
            self.use_agent_coordination = (self.n_agents > 1)
            if self.use_agent_coordination:
                # Agent coordination via attention mechanism
                coord_hidden = self.hidden_size // 2
                self.coord_query = nn.Linear(self.obs_dim, coord_hidden)
                self.coord_key = nn.Linear(self.obs_dim, coord_hidden)
                self.coord_value = nn.Linear(self.obs_dim, coord_hidden)
                self.coord_out = nn.Linear(coord_hidden, self.act_dim)
                print(f"[PHS-MAPPO v8.5] Agent {agent_id}: Enabled agent coordination for multi-agent velocity task")
            
            # NEW: Joint-aware R-matrix (different damping for different action dimensions)
            # Some joints need more freedom (limb swing), others need stability (torso)
            self.R_joint_net = nn.Sequential(
                nn.Linear(self.obs_dim + self.act_dim, self.hidden_size),
                nn.LayerNorm(self.hidden_size),
                nn.ELU(),
                nn.Linear(self.hidden_size, self.hidden_size // 2),
                nn.ELU(),
                nn.Linear(self.hidden_size // 2, self.act_dim),
                nn.Softplus()  # Ensure positive (dissipation must be positive)
            )
            
            # Velocity safety network: predicts safety-related damping boost
            # When agent is moving too fast or in risky state, increase R
            self.velocity_safety_net = nn.Sequential(
                nn.Linear(self.obs_dim, self.hidden_size // 2),
                nn.ELU(),
                nn.Linear(self.hidden_size // 2, self.act_dim),
                nn.Sigmoid()  # Output in [0, 1] for safety factor
            )

            # Project full qvel to act_dim for better control alignment
            self.velocity_proj_net = nn.Sequential(
                nn.Linear(self.velocity_n_qvel, self.hidden_size // 2),
                nn.ELU(),
                nn.Linear(self.hidden_size // 2, self.act_dim)
            )

            # NEW: Per-agent posture correction network
            # Different agents (front legs vs back legs) need different correction strategies
            self.velocity_posture_net = nn.Sequential(
                nn.Linear(self.velocity_n_qpos + 1, self.hidden_size // 2),  # +1 for agent_id
                nn.LayerNorm(self.hidden_size // 2),
                nn.ELU(),
                nn.Linear(self.hidden_size // 2, self.act_dim),
                nn.Tanh()
            )
            
            # NEW: Stability risk predictor - predicts fall risk from state
            self.stability_risk_net = nn.Sequential(
                nn.Linear(self.velocity_n_qpos + self.velocity_n_qvel, self.hidden_size // 4),
                nn.ELU(),
                nn.Linear(self.hidden_size // 4, 1),
                nn.Sigmoid()  # Output fall risk in [0, 1]
            )
        
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
        # For MultiGoal: Input = physics_hidden + goal/barrier gradients (4 dims)
        # For Velocity: Input = physics_hidden only (no lidar gradients)
        # Output: action directly
        policy_hidden = self.hidden_size * 2  # 512 if hidden_size=256
        
        # Gradient feature dimension: 4 for MultiGoal (2D goal + 2D barrier), 0 for Velocity
        self.gradient_feature_dim = 0 if self.is_velocity_task else 4
        policy_input_dim = self.physics_hidden + self.gradient_feature_dim
        
        self.policy_net = nn.Sequential(
            nn.Linear(policy_input_dim, policy_hidden),
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
        # Only used for MultiGoal tasks
        self.phs_gain_net = nn.Sequential(
            nn.Linear(policy_input_dim, self.hidden_size // 2),
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
        # For Velocity tasks, adapt state_dim to match act_dim
        if self.is_velocity_task:
            # For Velocity tasks, use act_dim-based state space
            # state_dim = 2 * act_dim (positions + velocities for each actuator)
            effective_state_dim = 2 * self.act_dim
            dim = self.act_dim  # velocity dimension = act_dim
        else:
            # For MultiGoal tasks, use fixed 2D state space (x, y, vx, vy)
            effective_state_dim = self.state_dim  # 4
            dim = self.state_dim // 2  # 2
        
        # Store for use in other methods
        self.effective_state_dim = effective_state_dim
        self.effective_dim = dim
        
        # Standard Hamiltonian J matrix: [[0, I], [-I, 0]]
        J_sys = torch.zeros(effective_state_dim, effective_state_dim)
        J_sys[:dim, dim:] = torch.eye(dim)
        J_sys[dim:, :dim] = -torch.eye(dim)
        self.register_buffer('J_sys', J_sys)
        
        # Standard dissipation R matrix: [[0, 0], [0, drag*I]]
        R_sys = torch.zeros(effective_state_dim, effective_state_dim)
        R_sys[dim:, dim:] = self.drag * torch.eye(dim)
        self.register_buffer('R_sys', R_sys)
        
        # Control input matrix F = [[0], [I]]
        # F maps control inputs to velocity changes
        F_sys = torch.zeros(effective_state_dim, self.act_dim)
        # For both task types, control affects the velocity part (second half of state)
        min_dim = min(dim, self.act_dim)
        F_sys[dim:dim+min_dim, :min_dim] = torch.eye(min_dim)
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

        if self.is_velocity_task:
            for m in self.velocity_proj_net:
                if isinstance(m, nn.Linear):
                    init_layer(m, gain=0.5)
            for m in self.velocity_posture_net:
                if isinstance(m, nn.Linear):
                    init_layer(m, gain=0.3)
    
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

    def _get_velocity_warmup_ratio(self):
        """Warmup ratio in [0, 1] for velocity control terms."""
        if self.velocity_control_warmup_steps <= 0:
            return 1.0
        return float(np.clip(self._training_step / float(self.velocity_control_warmup_steps), 0.0, 1.0))

    def _get_front_distal_warmup_ratio(self):
        """Warmup ratio in [0, 1] for 6x1 front-distal extension assist."""
        if self.velocity_front_distal_warmup_steps <= 0:
            return 1.0
        return float(np.clip(self._training_step / float(self.velocity_front_distal_warmup_steps), 0.0, 1.0))
    
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
        if goal_lidar.shape[-1] == 0:
            return torch.zeros((*obs.shape[:-1], 1), device=obs.device, dtype=obs.dtype)
        
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
        if hazard_lidar.shape[-1] == 0:
            return torch.zeros((*obs.shape[:-1], 1), device=obs.device, dtype=obs.dtype)
        
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
        if goal_lidar.shape[-1] == 0:
            goal_gradient = torch.zeros(obs.shape[0], 2, device=device)
            goal_proximity_scaled = torch.zeros(obs.shape[0], 1, device=device)
        else:
        
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
        if hazard_lidar.shape[-1] == 0:
            barrier_gradient = torch.zeros(obs.shape[0], 2, device=device)
            repulsion_strength = torch.zeros(obs.shape[0], 1, device=device)
        else:
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

    def _compute_velocity_posture_dev(self, qpos):
        """Compute posture deviation used by velocity-task stabilization."""
        if qpos.shape[-1] <= 1:
            return torch.zeros(qpos.shape[0], 1, device=qpos.device)
        # HalfCheetah stable gait naturally uses large leg articulation. Penalizing all
        # joint angles suppresses stride, so use torso pitch as the posture proxy.
        if self.is_halfcheetah_velocity:
            idx = self._get_halfcheetah_qpos_indices(qpos.shape[-1])
            if idx["pitch"] is not None:
                return torch.abs(qpos[:, idx["pitch"]:idx["pitch"] + 1])
        # Ant: use torso tilt (quaternion qx, qy) instead of all joint angles.
        # Ant qpos (rootx/y removed): [z, qw, qx, qy, qz, hip1, ankle1, ...]
        # Tilt = sqrt(qx^2 + qy^2); near 0 for upright, avoids penalizing natural gaits.
        if self.is_ant_velocity and qpos.shape[-1] >= 5:
            tilt = torch.sqrt(qpos[:, 2:3] ** 2 + qpos[:, 3:4] ** 2 + 1e-6)
            return tilt
        return torch.mean(torch.abs(qpos[:, 1:]), dim=-1, keepdim=True)

    def _get_halfcheetah_qpos_indices(self, qpos_dim):
        """Resolve HalfCheetah qpos indices for both full and rootx-dropped state vectors."""
        # Full MuJoCo qpos: [rootx, rootz, rooty, bthigh, bshin, bfoot, fthigh, fshin, ffoot] (len=9)
        # ShareEnv.state() often drops rootx, resulting in len=8.
        if qpos_dim >= 9:
            return {"z": 1, "pitch": 2, "bthigh": 3, "bshin": 4, "bfoot": 5, "fthigh": 6, "fshin": 7, "ffoot": 8}
        if qpos_dim >= 8:
            return {"z": 0, "pitch": 1, "bthigh": 2, "bshin": 3, "bfoot": 4, "fthigh": 5, "fshin": 6, "ffoot": 7}
        return {"z": None, "pitch": None, "bthigh": None, "bshin": None, "bfoot": None, "fthigh": None, "fshin": None, "ffoot": None}

    def _get_halfcheetah_qvel_indices(self, qvel_dim):
        """Resolve HalfCheetah qvel indices for both full and rootx-dropped state vectors."""
        # Typical qvel: [rootx, rootz, rooty, bthigh, bshin, bfoot, fthigh, fshin, ffoot]
        if qvel_dim >= 9:
            return {"bthigh": 3, "bshin": 4, "bfoot": 5, "fthigh": 6, "fshin": 7, "ffoot": 8}
        if qvel_dim >= 8:
            return {"bthigh": 2, "bshin": 3, "bfoot": 4, "fthigh": 5, "fshin": 6, "ffoot": 7}
        return {"bthigh": None, "bshin": None, "bfoot": None, "fthigh": None, "fshin": None, "ffoot": None}

    def _get_halfcheetah_local_control_vel(self, qvel):
        """For 6x1 HalfCheetah, return the local joint velocity controlled by this agent."""
        if not (self.is_halfcheetah_velocity and self.n_agents >= 6 and self.act_dim == 1):
            return None
        order = ["bthigh", "bshin", "bfoot", "fthigh", "fshin", "ffoot"]
        if self.agent_id < 0 or self.agent_id >= len(order):
            return None
        idx_map = self._get_halfcheetah_qvel_indices(qvel.shape[-1])
        idx = idx_map.get(order[self.agent_id], None)
        if idx is None or idx >= qvel.shape[-1]:
            return None
        return qvel[:, idx:idx + 1]
    
    def _compute_phs_action(self, obs, state_features, laplacian):
        """
        Compute action using Barrier PHS Actor (true port-Hamiltonian action mapping).
        
        For MultiGoal tasks:
            dx_target = π_θ([features, goal_grad, barrier_grad]) + guidance
            dx = (J - R) ∇H_total + F * a
            a = F^+ (dx_target - (J - R) ∇H_total)
        
        For Velocity tasks:
            Uses simplified MLP policy without 2D lidar gradients.
            a = π_θ([features])
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
        
        # ========== VELOCITY TASK: Barrier PHS with R Matrix Control ==========
        if self.is_velocity_task:
            velocity_warmup = self._get_velocity_warmup_ratio()
            front_distal_warmup = self._get_front_distal_warmup_ratio()
            # ========== Core Barrier PHS for Velocity Tasks ==========
            # H = 0.5 * v^T * M * v (pure kinetic energy)
            # ∇H = M * v ≈ v (assuming unit mass)
            # dx/dt = (J - R) * ∇H + F * u
            # R = R0 + R_learned(obs) where:
            #   - R0: Fixed base damping (energy dissipation)
            #   - R_learned: Adaptive damping for velocity safety control
            
            # ========== 1. Extract velocity from observation ==========
            # MAMuJoCo observation structure: [qpos, qvel, agent_id_onehot]
            # Velocity typically in the middle portion
            vel_start = self.velocity_n_qpos
            vel_end = vel_start + self.velocity_n_qvel

            # Extract full qvel
            obs_vel_full = obs_flat[:, vel_start:vel_end]
            if obs_vel_full.shape[-1] < self.velocity_n_qvel:
                pad_size = self.velocity_n_qvel - obs_vel_full.shape[-1]
                obs_vel_full = torch.cat(
                    [obs_vel_full, torch.zeros(obs_vel_full.shape[0], pad_size, device=obs_vel_full.device)], dim=-1
                )
            elif obs_vel_full.shape[-1] > self.velocity_n_qvel:
                obs_vel_full = obs_vel_full[:, :self.velocity_n_qvel]

            # ========== Enhanced Posture and Stability Analysis ==========
            qpos = obs_flat[:, :self.velocity_n_qpos]
            qvel = obs_flat[:, self.velocity_n_qpos:self.velocity_n_qpos + self.velocity_n_qvel]
            hc_idx = self._get_halfcheetah_qpos_indices(qpos.shape[-1]) if self.is_halfcheetah_velocity else None
            obs_vel_proj = self.velocity_proj_net(obs_vel_full)
            obs_vel_local = self._get_halfcheetah_local_control_vel(qvel)
            # 6x1 uses local joint velocity for PHS control alignment; others use projected velocity.
            obs_vel = obs_vel_local if obs_vel_local is not None else obs_vel_proj
            
            # Stability risk estimation (comprehensive fall risk prediction)
            qpos_qvel_combined = torch.cat([qpos, qvel], dim=-1)
            stability_risk = self.stability_risk_net(qpos_qvel_combined)  # [batch, 1]
            stability_excess = torch.clamp(
                stability_risk - self.velocity_stability_threshold,
                min=0.0,
            )
            
            # Posture deviation (HalfCheetah uses torso pitch proxy; others keep generic rule)
            posture_dev = self._compute_velocity_posture_dev(qpos)
            posture_risk = torch.clamp(posture_dev - self.velocity_posture_threshold, min=0.0)

            # Pitch risk: HalfCheetah-like forward fall is captured by root pitch angle
            if self.is_halfcheetah_velocity and hc_idx["pitch"] is not None:
                root_pitch = qpos[:, hc_idx["pitch"]:hc_idx["pitch"] + 1]
                pitch_risk = torch.clamp(torch.abs(root_pitch) - self.velocity_pitch_threshold, min=0.0)
            elif qpos.shape[-1] > 2:
                root_pitch = qpos[:, 2:3]
                pitch_risk = torch.clamp(torch.abs(root_pitch) - self.velocity_pitch_threshold, min=0.0)
            else:
                root_pitch = torch.zeros(qpos.shape[0], 1, device=qpos.device)
                pitch_risk = torch.zeros(qpos.shape[0], 1, device=qpos.device)

            # Height risk and pitch-rate risk for anti-forward-fall stabilization
            if self.is_halfcheetah_velocity and hc_idx["z"] is not None:
                torso_height = qpos[:, hc_idx["z"]:hc_idx["z"] + 1]
                height_risk = torch.clamp(self.velocity_height_threshold - torso_height, min=0.0)
            elif self.is_ant_velocity and qpos.shape[-1] >= 1:
                # Ant: qpos[0] = z position (standing height ~0.75)
                torso_height = qpos[:, 0:1]
                height_risk = torch.clamp(self.velocity_height_threshold - torso_height, min=0.0)
            elif qpos.shape[-1] > 1:
                torso_height = qpos[:, 1:2]
                height_risk = torch.clamp(self.velocity_height_threshold - torso_height, min=0.0)
            else:
                torso_height = torch.zeros(qpos.shape[0], 1, device=qpos.device)
                height_risk = torch.zeros(qpos.shape[0], 1, device=qpos.device)
            if self.is_ant_velocity and qvel.shape[-1] >= 6:
                # Ant: use angular velocity magnitude (qvel[3:6] = wx,wy,wz)
                pitch_rate = torch.norm(qvel[:, 3:6], dim=-1, keepdim=True)
                pitch_rate_risk = torch.clamp(pitch_rate - self.velocity_pitch_rate_threshold, min=0.0)
            elif qvel.shape[-1] > 2:
                pitch_rate = qvel[:, 2:3]
                pitch_rate_risk = torch.clamp(torch.abs(pitch_rate) - self.velocity_pitch_rate_threshold, min=0.0)
            else:
                pitch_rate = torch.zeros(qpos.shape[0], 1, device=qpos.device)
                pitch_rate_risk = torch.zeros(qpos.shape[0], 1, device=qpos.device)
            # Divergent body rotation risk: angle and angular rate with same sign means falling trend.
            fall_pitch_risk = torch.clamp(root_pitch * pitch_rate, min=0.0) if qpos.shape[-1] > 2 and qvel.shape[-1] > 2 else torch.zeros_like(pitch_risk)

            # Speed risk: align with env cost (sqrt(vx^2 + vy^2) threshold)
            if qvel.shape[-1] >= 1:
                forward_speed = torch.abs(qvel[:, 0:1])
            else:
                forward_speed = torch.zeros(qvel.shape[0], 1, device=qvel.device)
            if qvel.shape[-1] >= 2:
                planar_speed = torch.norm(qvel[:, :2], dim=-1, keepdim=True)
            else:
                planar_speed = forward_speed
            speed_ratio = planar_speed / (self.velocity_safety_threshold + 1e-6)
            speed_risk = torch.clamp(
                speed_ratio - 1.0,
                min=0.0,
            )
            preemptive_speed_risk = torch.clamp(speed_ratio - self.velocity_preemptive_ratio, min=0.0)
            
            # Per-agent posture correction with agent_id as input
            if is_multi_agent:
                # Create agent_id feature: normalize to [-1, 1]
                agent_id_feature = torch.full((qpos.shape[0], 1), 
                                             (self.agent_id - self.n_agents/2) / (self.n_agents/2),
                                             device=qpos.device)
                qpos_with_id = torch.cat([qpos, agent_id_feature], dim=-1)
            else:
                agent_id_feature = torch.zeros(qpos.shape[0], 1, device=qpos.device)
                qpos_with_id = torch.cat([qpos, agent_id_feature], dim=-1)
            
            posture_correction = -self.velocity_posture_net(qpos_with_id)
            
            # ========== 2. Compute Kinetic Energy Hamiltonian ==========
            # H = 0.5 * ||v||^2 (unit mass assumption)
            H_kin = 0.5 * (obs_vel_full ** 2).sum(dim=-1, keepdim=True)  # [batch, 1]
            
            # ∇H = v (gradient of kinetic energy w.r.t. velocity)
            grad_H_vel = obs_vel  # [batch, act_dim]

            # Energy risk based on kinetic energy overshoot
            H_safe = 0.5 * (self.velocity_safety_threshold ** 2)
            energy_risk = torch.clamp((H_kin - H_safe) / (H_safe + 1e-6), min=0.0)
            
            # ========== 3. Compute Joint-Aware Adaptive R Matrix ==========
            # Joint-aware R-matrix: different joints need different damping
            # Input: state + current velocity (velocity provides context for damping)
            R_input = torch.cat([obs_flat, obs_vel], dim=-1)  # [batch, obs_dim + act_dim]
            R_diag_learned = self.R_joint_net(R_input)  # [batch, act_dim], positive via Softplus
            
            # Safety factor: boost damping when moving fast or in risky state
            safety_factor = self.velocity_safety_net(obs_flat)  # [batch, act_dim], in [0, 1]
            
            # Compute velocity magnitude for adaptive damping
            vel_magnitude = torch.norm(obs_vel, dim=-1, keepdim=True)  # [batch, 1]
            vel_normalized = vel_magnitude / (self.velocity_safety_threshold + 1e-6)
            vel_risk = torch.clamp(vel_normalized - 1.0, min=0.0)  # Risk when exceeding threshold
            vel_risk = torch.maximum(vel_risk, speed_risk)

            # Directional damping: increase damping along current motion direction
            dir_weight = grad_H_vel.pow(2)
            dir_weight = dir_weight / (dir_weight.sum(dim=-1, keepdim=True) + 1e-6)
            
            # Total R diagonal: R0 + R_learned * (1 + safety + posture + stability)
            # Higher stability_risk -> more damping to prevent fall
            R_diag_total = (
                self.velocity_r_base +
                R_diag_learned * (1.0 + 
                                 safety_factor * vel_risk * self.velocity_r_max + 
                                 posture_risk * self.velocity_posture_r_scale +
                                 stability_excess * self.velocity_stability_r_scale +
                                 pitch_risk * self.velocity_pitch_r_scale +
                                 height_risk * self.velocity_height_r_scale +
                                 pitch_rate_risk * self.velocity_pitch_rate_r_scale +
                                 fall_pitch_risk * self.velocity_pitch_rate_r_scale +
                                 speed_risk * self.velocity_speed_r_scale +
                                 energy_risk * self.velocity_energy_r_scale +
                                 preemptive_speed_risk * self.velocity_preemptive_r_scale) +
                self.velocity_directional_r_scale * dir_weight * (speed_risk + energy_risk + preemptive_speed_risk + pitch_rate_risk + fall_pitch_risk)
            )  # [batch, act_dim]

            # HalfCheetah joint-aware damping prior:
            # lower damping on thigh (to increase stride amplitude), higher on distal joints.
            if self.is_halfcheetah_velocity and self.act_dim >= 3:
                joint_prior = torch.ones_like(R_diag_total)
                thigh_scale = max(1.0 - float(self.velocity_thigh_r_relief), 0.55)
                joint_prior[:, 0:1] = thigh_scale
                joint_prior[:, 1:] = 1.0 + self.velocity_distal_r_boost
                R_diag_total = R_diag_total * joint_prior
            elif self.is_halfcheetah_velocity and self.n_agents >= 6 and self.act_dim == 1:
                if self.agent_id in (0, 3):
                    local_r_prior = max(1.0 - float(self.velocity_thigh_r_relief), 0.55)
                elif self.agent_id in (4, 5):
                    local_r_prior = max(1.0 - float(self.velocity_front_distal_r_relief), 0.45)
                else:
                    local_r_prior = 1.0 + 0.5 * float(self.velocity_distal_r_boost)
                R_diag_total = R_diag_total * local_r_prior
            R_diag_total = torch.clamp(R_diag_total, min=self.velocity_r_base, max=self.velocity_r_total_max)
            
            # ========== 4. Policy Network with Agent Coordination ==========
            policy_output = self.policy_net(state_features_flat)  # [batch, act_dim]
            
            residual = self.residual_mlp(state_features_flat)
            residual_w = torch.sigmoid(self.residual_weight) * 0.3
            
            # NEW: Agent coordination for multi-agent velocity tasks
            coordination_signal = torch.zeros_like(policy_output)
            if self.use_agent_coordination and is_multi_agent:
                # Reshape to [batch, n_agents, obs_dim] for coordination
                obs_ma_view = obs_flat.view(batch_size, n_agents, -1)
                
                # Compute attention: each agent attends to other agents
                Q = self.coord_query(obs_ma_view)  # [batch, n_agents, coord_hidden]
                K = self.coord_key(obs_ma_view)    # [batch, n_agents, coord_hidden]
                V = self.coord_value(obs_ma_view)  # [batch, n_agents, coord_hidden]
                
                # Scaled dot-product attention
                scores = torch.matmul(Q, K.transpose(-2, -1)) / (Q.shape[-1] ** 0.5)
                # Mask self-attention (agent shouldn't attend to itself)
                mask = torch.eye(n_agents, device=obs.device).unsqueeze(0).bool()
                scores = scores.masked_fill(mask, float('-inf'))
                attn_weights = F.softmax(scores, dim=-1)  # [batch, n_agents, n_agents]
                
                # Aggregate information from other agents
                coord_context = torch.matmul(attn_weights, V)  # [batch, n_agents, coord_hidden]
                coord_context_flat = coord_context.view(batch_size * n_agents, -1)
                coordination_signal = self.coord_out(coord_context_flat)  # [batch*n_agents, act_dim]
            
            # Gating based on posture risk and stability risk
            # When unstable, rely more on posture correction and less on policy
            combined_risk = posture_risk + stability_excess + pitch_risk + 0.5 * fall_pitch_risk
            posture_gate = 1.0 - torch.clamp(self.velocity_posture_gate * combined_risk, min=0.0, max=0.8)
            posture_gate = posture_gate * (1.0 - torch.clamp(self.velocity_pitch_gate * pitch_risk, min=0.0, max=0.6))
            if self.is_halfcheetah_velocity:
                posture_gate = torch.clamp(posture_gate, min=self.velocity_policy_gate_floor, max=1.0)
            posture_scale = torch.clamp(
                self.velocity_posture_correction_weight * combined_risk,
                min=0.0,
                max=self.velocity_posture_correction_max
            )
            posture_scale = posture_scale * velocity_warmup
            
            # Combine policy, coordination, and posture correction
            # coordination_signal helps balance forces across agents
            coordination_weight = self.velocity_coordination_weight if self.use_agent_coordination else 0.0
            dx_target = ((policy_output + residual_w * residual) * posture_gate + 
                        coordination_weight * coordination_signal +
                        posture_scale * posture_correction)

            # HalfCheetah thigh extension target correction (small, local, PHS-consistent)
            thigh_target_correction = torch.zeros_like(dx_target)
            recovery_gate = torch.zeros_like(posture_risk)
            if self.is_halfcheetah_velocity and qpos.shape[-1] > 6:
                recovery_risk = pitch_risk + height_risk + 0.5 * pitch_rate_risk + fall_pitch_risk
                recovery_excess = torch.clamp(recovery_risk - self.velocity_thigh_recovery_threshold, min=0.0)
                recovery_gate = torch.clamp(
                    recovery_excess / (self.velocity_thigh_recovery_threshold + 1e-6),
                    min=0.0,
                    max=1.0,
                )
                recovery_gate = recovery_gate * velocity_warmup
                recovery_scale = 1.0 + self.velocity_thigh_recovery_gain * torch.clamp(recovery_excess, max=1.0)
                thigh_angle = None
                thigh_target = None
                if self.n_agents == 2 and self.act_dim >= 3:
                    if self.agent_id == 0:
                        if hc_idx["bthigh"] is not None:
                            thigh_angle = qpos[:, hc_idx["bthigh"]:hc_idx["bthigh"] + 1]
                        thigh_target = self.velocity_back_thigh_target
                    elif self.agent_id == 1:
                        if hc_idx["fthigh"] is not None:
                            thigh_angle = qpos[:, hc_idx["fthigh"]:hc_idx["fthigh"] + 1]
                        thigh_target = self.velocity_front_thigh_target
                    if thigh_angle is not None:
                        thigh_err = thigh_target - thigh_angle
                        thigh_cmd = torch.clamp(
                            self.velocity_thigh_target_gain * thigh_err,
                            min=-self.velocity_thigh_target_max,
                            max=self.velocity_thigh_target_max,
                        )
                        thigh_cmd = torch.clamp(thigh_cmd * recovery_scale, min=-self.velocity_thigh_target_max, max=self.velocity_thigh_target_max)
                        if self.agent_id == 1:
                            thigh_cmd = torch.clamp(thigh_cmd + self.velocity_front_lift_bias * recovery_excess, min=-self.velocity_thigh_target_max, max=self.velocity_thigh_target_max)
                        elif self.agent_id == 0:
                            thigh_cmd = torch.clamp(thigh_cmd - self.velocity_back_push_bias * recovery_excess, min=-self.velocity_thigh_target_max, max=self.velocity_thigh_target_max)
                        thigh_target_correction[:, 0:1] = thigh_cmd * recovery_gate
                elif self.n_agents >= 6 and self.act_dim == 1:
                    if self.agent_id == 0:
                        if hc_idx["bthigh"] is not None:
                            thigh_angle = qpos[:, hc_idx["bthigh"]:hc_idx["bthigh"] + 1]
                        thigh_target = self.velocity_back_thigh_target
                    elif self.agent_id == 3:
                        if hc_idx["fthigh"] is not None:
                            thigh_angle = qpos[:, hc_idx["fthigh"]:hc_idx["fthigh"] + 1]
                        thigh_target = self.velocity_front_thigh_target
                    if thigh_angle is not None:
                        thigh_err = thigh_target - thigh_angle
                        thigh_cmd = torch.clamp(
                            self.velocity_thigh_target_gain * thigh_err,
                            min=-self.velocity_thigh_target_max,
                            max=self.velocity_thigh_target_max,
                        )
                        thigh_target_correction[:, 0:1] = torch.clamp(
                            thigh_cmd * recovery_scale,
                            min=-self.velocity_thigh_target_max,
                            max=self.velocity_thigh_target_max,
                        ) * recovery_gate
            dx_target = dx_target + thigh_target_correction

            # 6x1 front distal extension assist:
            # encourage fshin/ffoot away from near-zero tucked posture while preserving sign.
            front_distal_extension = torch.zeros_like(dx_target)
            if (
                self.is_halfcheetah_velocity
                and self.n_agents >= 6
                and self.act_dim == 1
                and self.agent_id in (4, 5)
            ):
                joint_key = "fshin" if self.agent_id == 4 else "ffoot"
                joint_idx = self._get_halfcheetah_qpos_indices(qpos.shape[-1]).get(joint_key, None)
                if joint_idx is not None and joint_idx < qpos.shape[-1]:
                    joint_angle = qpos[:, joint_idx:joint_idx + 1]
                    target_abs = self.velocity_front_shin_abs_target if self.agent_id == 4 else self.velocity_front_foot_abs_target
                    mag_err = torch.clamp(target_abs - torch.abs(joint_angle), min=0.0)
                    sign = torch.where(joint_angle >= 0, torch.ones_like(joint_angle), -torch.ones_like(joint_angle))
                    ext_cmd = self.velocity_front_distal_extension_gain * mag_err * sign
                    ext_cmd = torch.clamp(
                        ext_cmd,
                        min=-self.velocity_front_distal_extension_max,
                        max=self.velocity_front_distal_extension_max,
                    )
                    risk_mix = front_distal_warmup * (0.35 + 0.65 * recovery_gate)
                    front_distal_extension[:, 0:1] = ext_cmd * risk_mix
            dx_target = dx_target + front_distal_extension

            # HalfCheetah actuation rebalance for gait quality:
            # amplify thigh channel and slightly suppress distal channels.
            if self.is_halfcheetah_velocity and self.act_dim >= 3:
                action_gain = torch.ones_like(dx_target)
                action_gain[:, 0:1] = self.velocity_thigh_action_gain
                action_gain[:, 1:] = self.velocity_distal_action_gain
                if self.n_agents == 2 and self.agent_id == 1:
                    # Front half agent gets extra drive to avoid "rear-dominant" gait and forward collapse.
                    action_gain = action_gain * self.velocity_front_action_boost
                dx_target = dx_target * action_gain
            elif self.is_halfcheetah_velocity and self.n_agents >= 6 and self.act_dim == 1:
                # For 6x1, explicitly boost thigh-controlled agents.
                if self.agent_id in (0, 3):
                    local_gain = self.velocity_thigh_action_gain
                    if self.agent_id == 3:
                        local_gain = local_gain * self.velocity_front_action_boost
                elif self.agent_id in (4, 5):
                    # Front distal joints need larger excursion to avoid "tucked front leg" gait.
                    local_gain = max(self.velocity_distal_action_gain, 1.05)
                    local_gain = local_gain * max(self.velocity_front_action_boost * 1.05, 1.10)
                else:
                    local_gain = self.velocity_distal_action_gain
                local_gain = 1.0 + (local_gain - 1.0) * velocity_warmup
                dx_target = dx_target * local_gain
            
            # ========== 5. PHS Dynamics: (J - R) * ∇H ==========
            # For velocity-only state, we simplify to just the velocity dynamics
            # J = 0 (no position-velocity coupling in pure velocity control)
            # So drift = -R * ∇H = -R * v (pure dissipation)
            
            phs_drift = -R_diag_total * grad_H_vel  # [batch, act_dim]
            
            # ========== 6. Compute Control Action ==========
            # Use risk-adaptive blending to avoid over-dominant PHS compensation in nominal gait.
            control_risk = torch.clamp(
                pitch_risk + height_risk + 0.5 * pitch_rate_risk + fall_pitch_risk + stability_excess,
                min=0.0,
                max=1.5,
            )
            phs_blend = torch.clamp(
                self.velocity_phs_blend_base + self.velocity_phs_blend_risk_scale * control_risk,
                min=0.0,
                max=1.0,
            )
            phs_blend = phs_blend * velocity_warmup
            phs_comp = torch.clamp(
                -phs_blend * phs_drift,
                min=-self.velocity_phs_comp_max,
                max=self.velocity_phs_comp_max,
            )
            u_mean = dx_target + phs_comp
            
            # Scale action
            u_mean = torch.tanh(u_mean) * self.f_max
            
            # Reshape if multi-agent
            if is_multi_agent:
                u_mean = u_mean.view(batch_size, n_agents, -1)
            
            # ========== 7. Enhanced Logging Info ==========
            H_info = {
                'H_goal': torch.tensor(0.0),
                'H_task_learned': torch.tensor(0.0),
                'H_task': torch.tensor(0.0),
                'H_kin': H_kin.mean().detach(),
                'H_barrier_obs': torch.tensor(0.0),
                'H_barrier_agent': torch.tensor(0.0),
                'barrier_weight': self._get_current_barrier_weight(),
                'goal_prox': torch.tensor(0.0),
                'hazard_prox': torch.tensor(0.0),
                'agent_type': self.agent_type,
                'policy_output_mean': policy_output.mean().item(),
                'goal_grad_forward': 0.0,
                'goal_grad_turn': 0.0,
                'dx_target_mean': dx_target.mean().item(),
                # Velocity task specific metrics
                'R_learned_mean': R_diag_learned.mean().item(),
                'R_total_mean': R_diag_total.mean().item(),
                'safety_factor_mean': safety_factor.mean().item(),
                'vel_magnitude_mean': vel_magnitude.mean().item(),
                'phs_drift_mean': phs_drift.mean().item(),
                'phs_blend_mean': phs_blend.mean().item(),
                'velocity_warmup': velocity_warmup,
                'front_distal_warmup': front_distal_warmup,
                'posture_risk_mean': posture_risk.mean().item(),
                'posture_correction_mean': posture_correction.mean().item(),
                # NEW: Enhanced stability metrics
                'stability_risk_mean': stability_risk.mean().item(),
                'stability_excess_mean': stability_excess.mean().item(),
                'coordination_signal_mean': coordination_signal.mean().item() if self.use_agent_coordination else 0.0,
                'combined_risk_mean': combined_risk.mean().item(),
                'pitch_risk_mean': pitch_risk.mean().item(),
                'fall_pitch_risk_mean': fall_pitch_risk.mean().item(),
                'speed_risk_mean': speed_risk.mean().item(),
                'forward_speed_mean': forward_speed.mean().item(),
                'planar_speed_mean': planar_speed.mean().item(),
                'energy_risk_mean': energy_risk.mean().item(),
                'preemptive_speed_risk_mean': preemptive_speed_risk.mean().item(),
                'height_risk_mean': height_risk.mean().item(),
                'pitch_rate_risk_mean': pitch_rate_risk.mean().item(),
                'torso_height_mean': torso_height.mean().item(),
                'thigh_target_corr_mean': thigh_target_correction.mean().item(),
                'front_distal_ext_mean': front_distal_extension.mean().item(),
                'recovery_risk_mean': (pitch_risk + height_risk + 0.5 * pitch_rate_risk).mean().item(),
                'recovery_gate_mean': recovery_gate.mean().item(),
            }
            
            return u_mean, H_info, state_features
        
        # ========== MULTIGOAL TASK: Full PHS with 2D Gradients ==========
        # 1. Compute Directional Gradients (as features + guidance)
        goal_grad, barrier_grad, goal_prox, hazard_prox = self._compute_directional_gradient(obs_flat)

        # 2. Policy Output (desired state change)
        policy_input = torch.cat([state_features_flat, goal_grad, barrier_grad], dim=-1)
        policy_output = self.policy_net(policy_input)  # [batch, act_dim]

        residual = self.residual_mlp(state_features_flat)
        residual_w = torch.sigmoid(self.residual_weight) * 0.3

        # Guidance encourages goal progress and basic avoidance (2D)
        dx_guidance = (
            self.phs_goal_guidance_weight * goal_grad +
            self.phs_barrier_guidance_weight * barrier_grad
        )

        # For MultiGoal, act_dim should be 2, so this works
        dx_target_body = policy_output + residual_w * residual + dx_guidance

        # 3. PHS Drift (Barrier + Task)
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
        # Use effective_state_dim for proper dimensions
        grad_H_state = torch.zeros(dx_target_body.shape[0], self.effective_state_dim, device=obs.device, dtype=obs.dtype)
        effective_dim = self.effective_dim
        
        # Handle grad_H_vel dimension mismatch
        if grad_H_vel.shape[-1] > effective_dim:
            grad_vel = grad_H_vel[:, :effective_dim]
        elif grad_H_vel.shape[-1] < effective_dim:
            pad = torch.zeros(grad_H_vel.shape[0], effective_dim - grad_H_vel.shape[-1], device=obs.device, dtype=obs.dtype)
            grad_vel = torch.cat([grad_H_vel, pad], dim=-1)
        else:
            grad_vel = grad_H_vel
        grad_H_state[:, effective_dim:] = grad_vel

        J_R = self.J_sys - self.R_sys
        phs_drift = torch.matmul(J_R, grad_H_state.unsqueeze(-1)).squeeze(-1)

        # Desired state change in state space
        dx_target_state = torch.zeros_like(grad_H_state)
        # For MultiGoal, act_dim=2 and effective_dim=2, so this should work
        min_dim = min(self.act_dim, effective_dim)
        dx_target_state[:, :min_dim] = dx_target_body[:, :min_dim]

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
        
        # For Velocity tasks, compute R-matrix based PHS physics info
        if self.is_velocity_task:
            obs_norm = self.feature_norm(obs)
            state_features = self.state_encoder(obs_norm)
            
            if is_multi_agent:
                obs_flat = obs.view(-1, obs.shape[-1])
                state_features_flat = state_features.view(-1, state_features.shape[-1])
            else:
                obs_flat = obs
                state_features_flat = state_features
            
            # Extract full velocity from observation
            vel_start = self.velocity_n_qpos
            vel_end = vel_start + self.velocity_n_qvel
            obs_vel_full = obs_flat[:, vel_start:vel_end]
            if obs_vel_full.shape[-1] < self.velocity_n_qvel:
                pad_size = self.velocity_n_qvel - obs_vel_full.shape[-1]
                obs_vel_full = torch.cat(
                    [obs_vel_full, torch.zeros(obs_vel_full.shape[0], pad_size, device=obs_vel_full.device)], dim=-1
                )
            elif obs_vel_full.shape[-1] > self.velocity_n_qvel:
                obs_vel_full = obs_vel_full[:, :self.velocity_n_qvel]

            # Project to control-relevant velocity
            obs_vel_proj = self.velocity_proj_net(obs_vel_full)
            qvel = obs_flat[:, self.velocity_n_qpos:self.velocity_n_qpos + self.velocity_n_qvel]
            obs_vel_local = self._get_halfcheetah_local_control_vel(qvel)
            obs_vel = obs_vel_local if obs_vel_local is not None else obs_vel_proj

            # Posture risk from qpos (same rule as actor path)
            qpos = obs_flat[:, :self.velocity_n_qpos]
            posture_dev = self._compute_velocity_posture_dev(qpos)
            posture_risk = torch.clamp(posture_dev - self.velocity_posture_threshold, min=0.0)
            
            # Compute kinetic energy
            H_kin = 0.5 * (obs_vel_full ** 2).sum(dim=-1, keepdim=True)
            
            # Compute R matrix components
            R_input = torch.cat([obs_flat, obs_vel], dim=-1)
            R_diag_learned = self.R_joint_net(R_input)
            safety_factor = self.velocity_safety_net(obs_flat)
            vel_magnitude = torch.norm(obs_vel, dim=-1, keepdim=True)
            vel_normalized = vel_magnitude / (self.velocity_safety_threshold + 1e-6)
            vel_risk = torch.clamp(vel_normalized - 1.0, min=0.0)
            R_diag_total = (
                self.velocity_r_base +
                R_diag_learned * (1.0 + safety_factor * vel_risk * self.velocity_r_max + posture_risk * self.velocity_posture_r_scale)
            )
            
            return {
                'H_total': H_kin.mean().detach(),
                'H_goal': torch.tensor(0.0),
                'H_task_learned': torch.tensor(0.0),
                'H_task': torch.tensor(0.0),
                'H_kin': H_kin.mean().detach(),
                'H_barrier_obs': torch.tensor(0.0),
                'H_barrier_agent': torch.tensor(0.0),
                'grad_H': obs_vel.detach(),  # ∇H = v for kinetic energy
                'proximity': torch.zeros(obs_flat.shape[0], 1, device=obs.device),
                'barrier_weight': self._get_current_barrier_weight(),
                'goal_grad_forward': torch.tensor(0.0),
                'goal_grad_turn': torch.tensor(0.0),
                'barrier_grad_forward': torch.tensor(0.0),
                'barrier_grad_turn': torch.tensor(0.0),
                'goal_prox': torch.tensor(0.0),
                'hazard_prox': torch.tensor(0.0),
                # R-matrix specific metrics
                'R_learned_mean': R_diag_learned.mean().detach(),
                'R_total_mean': R_diag_total.mean().detach(),
                'safety_factor_mean': safety_factor.mean().detach(),
                'vel_magnitude_mean': vel_magnitude.mean().detach(),
                'vel_risk_mean': vel_risk.mean().detach(),
                'posture_risk_mean': posture_risk.mean().detach(),
                'posture_correction_mean': torch.tensor(0.0, device=obs.device),
            }
        
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
        if hazard_lidar.shape[-1] == 0:
            grad_H = torch.zeros((*obs.shape[:-1], 2), device=obs.device, dtype=obs.dtype)
            proximity = torch.zeros((*obs.shape[:-1], 1), device=obs.device, dtype=obs.dtype)
        else:
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
        if hazard_lidar.shape[-1] == 0:
            grad_H = torch.zeros((*obs.shape[:-1], 2), device=obs.device, dtype=obs.dtype)
            return H_barrier, grad_H
        
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

        if hazard_lidar.shape[-1] == 0:
            zeros_shape = (*obs.shape[:-1], 1)
            zeros = torch.zeros(zeros_shape, device=obs.device, dtype=obs.dtype)
            return hazard_lidar, zeros
        
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
        if goal_lidar.shape[-1] == 0:
            zeros_shape = (*obs.shape[:-1], 1)
            zeros = torch.zeros(zeros_shape, device=obs.device, dtype=obs.dtype)
            return goal_lidar, zeros
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

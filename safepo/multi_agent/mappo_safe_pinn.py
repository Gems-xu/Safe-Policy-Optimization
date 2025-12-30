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
MAPPO-Safe-PINN: MAPPO with Barrier Port-Hamiltonian Neural Network Actor.

This algorithm implements Safe-pH-MARL (Safe Physics-Informed Multi-Agent RL)
based on the theoretical framework in Barrier_PHS.md.

Core Design:
    1. Potential Energy Splitting:
       - H_task: Neural network learned task potential (attracts toward goal)
       - H_barrier: Barrier Lyapunov Function potential (repels from obstacles)
       
    2. Port-Hamiltonian Dynamics:
       ẋ = (J(x) - R(x)) ∇H_total(x)
       
       Where:
       - H_total = H_task + H_barrier
       - J(x): Skew-symmetric interconnection (gyroscopic forces for escaping local minima)
       - R(x): Positive semi-definite dissipation (energy damping for stability)
       
    3. Safety by Construction:
       H_barrier = k / ((d - r_safe)² + ε)
       When d → r_safe, H_barrier → ∞
       Since system is passive (Ḣ ≤ 0), agents cannot acquire enough energy
       to cross the infinite potential barrier → intrinsic hard safety

For Point/Car agents in SafetyMultiGoal environments:
    - Observation: ~152-dim (accelerometer, velocimeter, gyro, magnetometer, lidars)
    - velocimeter (indices 3-5): velocity (vx, vy, vz)
    - lidar: detect obstacles and goals
    - Action: 2-dim (forward force, turning velocity)
"""

import copy
import numpy as np
try: 
    import isaacgym
except:
    pass
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys
import time
from tqdm import tqdm

# Initialize CUDA context at module load time to prevent cuBLAS warnings
# This must happen before any other CUDA operations
# We perform a complete forward + backward pass to fully initialize the autograd engine
if torch.cuda.is_available():
    try:
        # Set device 0 (when CUDA_VISIBLE_DEVICES is set, this is the first visible device)
        torch.cuda.set_device(0)
        
        # Create a simple computation graph and run backward to initialize cuBLAS
        # This is the most thorough way to prevent the cuBLAS warning
        _x = torch.randn(4, 4, device='cuda', requires_grad=True)
        _y = torch.randn(4, 4, device='cuda', requires_grad=True)
        
        # Perform operations that use cuBLAS
        _z = torch.mm(_x, _y)  # Matrix multiplication (cuBLAS)
        _loss = _z.sum()
        
        # Run backward pass to initialize the entire autograd engine + cuBLAS
        _loss.backward()
        
        # Synchronize to ensure all initialization is complete
        torch.cuda.synchronize()
        
        # Clean up
        del _x, _y, _z, _loss
        
    except Exception:
        # Silently fail if CUDA initialization fails
        pass

from safepo.common.env import make_ma_mujoco_env, make_ma_isaac_env, make_ma_multi_goal_env
from safepo.common.popart import PopArt
from safepo.common.model import MultiAgentCritic as Critic
from safepo.common.buffer import SeparatedReplayBuffer
from safepo.common.logger import EpochLogger
from safepo.common.video_recorder import MultiAgentVideoRecorder, setup_headless_rendering
from safepo.utils.config import multi_agent_args, parse_sim_params, set_np_formatting, set_seed, multi_agent_velocity_map, isaac_gym_map, multi_agent_goal_tasks
from safepo.utils.visualize_barrier_potential import BarrierPotentialVisualizer
from safepo.utils.barrier_potential_video_visualizer import BarrierPotentialVideoVisualizer, EvalPotentialFieldRecorder

# Import the modularized Barrier Port-Hamiltonian PINN Actor
from safepo.multi_agent.barrier_phs_pinn_actor import BarrierPHSPINNActor


def check(input):
    output = torch.from_numpy(input) if type(input) == np.ndarray else input
    return output

def huber_loss(e, d):
    a = (abs(e) <= d).float()
    b = (e > d).float()
    return a*e**2/2 + b*d*(abs(e)-d/2)


# =============================================================================
# MAPPO-Safe-PINN Policy with Barrier Port-Hamiltonian Actor
# =============================================================================
# 
# The BarrierPHSPINNActor module has been extracted to barrier_phs_pinn_actor.py
# for better modularity and code organization.
# 
# Based on Barrier_PHS.md theoretical formulation:
# 
# Port-Hamiltonian dynamics: ẋ = (J(x) - R(x)) ∇H_total(x)
# 
# Where H_total = H_kin(p) + H_task(q;θ) + H_barrier(q;φ)
#   - H_task: Learnable task potential (attracts to goal)
#   - H_barrier: Parametric barrier potential (repels from obstacles/agents)
#   - J: Skew-symmetric interconnection matrix (gyroscopic forces for escaping local minima)
#   - R: Positive semi-definite dissipation matrix
#
# Key Safety Feature:
#   H_barrier = Σ k_ij / (||q_i - q_j|| - r_safe)² + ε
#   When distance → r_safe, H_barrier → ∞, providing hard safety guarantee
# =============================================================================


# BarrierPHSPINNActor class now imported from barrier_phs_pinn_actor module


# =============================================================================
# MAPPO-Safe-PINN Policy with Barrier Port-Hamiltonian Actor
# =============================================================================

class MAPPOSafePINNPolicy:
    """MAPPO policy with Barrier Port-Hamiltonian PINN Actor for Point/Car agents.
    
    This policy uses the Safe-pH-MARL framework that provides:
    1. Safety by Construction: Barrier potential prevents collisions through energy conservation
    2. Goal-directed behavior: Task potential guides agents toward goals
    3. Deadlock avoidance: Gyroscopic forces from J matrix help escape local minima
    """

    def __init__(self, config, obs_space, cent_obs_space, act_space):
        self.config = config
        self.obs_space = obs_space
        self.act_space = act_space
        self.share_obs_space = cent_obs_space

        # Use Barrier Port-Hamiltonian PINN Actor
        self.actor = BarrierPHSPINNActor(config, self.obs_space, self.act_space, self.config["device"])
        
        # Use standard Critic (like MAPPO)
        self.critic = Critic(config, self.share_obs_space, self.config["device"])

        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), 
            lr=self.config["actor_lr"], 
            eps=self.config["opti_eps"], 
            weight_decay=self.config["weight_decay"]
        )
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), 
            lr=self.config["critic_lr"], 
            eps=self.config["opti_eps"], 
            weight_decay=self.config["weight_decay"]
        )

    def get_actions(self, cent_obs, obs, rnn_states_actor, rnn_states_critic, masks, available_actions=None,
                    deterministic=False):
        actions, action_log_probs, rnn_states_actor = self.actor(obs,
                                                                 rnn_states_actor,
                                                                 masks,
                                                                 available_actions,
                                                                 deterministic)

        values, rnn_states_critic = self.critic(cent_obs, rnn_states_critic, masks)
        return values, actions, action_log_probs, rnn_states_actor, rnn_states_critic

    def get_values(self, cent_obs, rnn_states_critic, masks):
        values, _ = self.critic(cent_obs, rnn_states_critic, masks)
        return values

    def evaluate_actions(self, cent_obs, obs, rnn_states_actor, rnn_states_critic, action, masks,
                         available_actions=None, active_masks=None):
        action_log_probs, dist_entropy = self.actor.evaluate_actions(obs,
                                                                     rnn_states_actor,
                                                                     action,
                                                                     masks,
                                                                     available_actions,
                                                                     active_masks)

        values, _ = self.critic(cent_obs, rnn_states_critic, masks)
        return values, action_log_probs, dist_entropy

    def act(self, obs, rnn_states_actor, masks, available_actions=None, deterministic=False):
        actions, _, rnn_states_actor = self.actor(obs, rnn_states_actor, masks, available_actions, deterministic)
        return actions, rnn_states_actor


# =============================================================================
# MAPPO-Safe-PINN Trainer (same as MAPPO, but with physics logging)
# =============================================================================

class MAPPOSafePINNTrainer():

    def __init__(self, config, policy):
        
        self.config = config
        self.tpdv = dict(dtype=torch.float32, device=self.config["device"])
        self.policy = policy

        self.value_normalizer = PopArt(1, device=self.config["device"])
        
        # v7.0: Minimal auxiliary loss - let PPO do the main learning
        # Physics features are provided to the network, but we don't force behavior
        self.aux_task_potential_weight = config.get("aux_task_potential_weight", 0.01)
        self.aux_barrier_potential_weight = config.get("aux_barrier_potential_weight", 0.02)
        self.aux_safety_weight = config.get("aux_safety_weight", 0.01)
        
        # v7.0: DISABLE reward shaping - it was causing train/eval mismatch
        # The huge gap (380 vs 8) was because shaping dominated training rewards
        self.use_reward_shaping = config.get("use_reward_shaping", False)  # DISABLED!
        self.reward_shaping_weight = config.get("reward_shaping_weight", 0.0)
        self.barrier_reward_weight = config.get("barrier_reward_weight", 0.0)
        self.task_reward_weight = config.get("task_reward_weight", 0.0)
        
        # === Lagrangian Constraint for Cost ===
        self.use_lagrangian = config.get("use_lagrangian", True)
        self.cost_limit = config.get("cost_limit", 25.0)  # Target cost limit
        self.lagrangian_multiplier = nn.Parameter(
            torch.tensor(config.get("lagrangian_init", 0.001), 
                        dtype=torch.float32, device=config["device"]),
            requires_grad=True
        )
        self.lagrangian_lr = config.get("lagrangian_lr", 0.035)
        self.lagrangian_max = config.get("lagrangian_max", 10.0)
        self.lagrangian_optimizer = torch.optim.Adam([self.lagrangian_multiplier], lr=self.lagrangian_lr)
        
        # Cost buffer for Lagrangian update
        self.recent_costs = []
        self.cost_window = 10  # Average over last N episodes

    def cal_value_loss(self, values, value_preds_batch, return_batch, active_masks_batch):
        value_pred_clipped = value_preds_batch + (values - value_preds_batch).clamp(-self.config["clip_param"],
                                                                                    self.config["clip_param"])
        error_clipped = self.value_normalizer(return_batch) - value_pred_clipped
        error_original = self.value_normalizer(return_batch) - values

        value_loss_clipped = huber_loss(error_clipped, self.config["huber_delta"])
        value_loss_original = huber_loss(error_original, self.config["huber_delta"])

        value_loss = torch.max(value_loss_original, value_loss_clipped)

        return value_loss.mean()

    def compute_auxiliary_physics_loss(self, obs_batch):
        """
        Compute auxiliary losses to help train H_task and barrier_k networks.
        
        Enhanced with stronger safety guidance:
        1. Task potential should be lower near goals (based on goal lidar readings)
        2. Barrier potential should be higher near obstacles (based on hazard lidar)
        3. Safety loss: penalize being close to hazards to reduce cost
        
        Args:
            obs_batch: [batch, obs_dim] observation tensor
            
        Returns:
            aux_loss: Combined auxiliary loss
            aux_info: Dict with individual loss components
        """
        batch_size = obs_batch.shape[0]
        device = obs_batch.device
        
        # Get physics info from actor
        actor = self.policy.actor
        
        # Compute potentials
        H_task, _ = actor._compute_task_potential(obs_batch)
        H_barrier, _ = actor._compute_barrier_potential(obs_batch)
        
        # Extract lidar readings using actor's methods for consistency
        # Goal lidar: obs[12:44] - higher means closer to goal
        goal_lidar, goal_proximity = actor._extract_goal_lidar_info(obs_batch)
        
        # Hazard + Agent lidar: obs[44:60] + obs[76:92] (combined in _extract_lidar_info)
        # Higher means closer to hazard/other agent
        # v4.0: Returns proximity ∈ [0, 1], not approx_dist
        hazard_lidar, hazard_proximity = actor._extract_lidar_info(obs_batch)
        
        # === Auxiliary Loss 1: Task Potential ===
        # H_task should be LOWER when close to goal (goal_proximity high)
        # Enhanced: use negative exponential for stronger gradient near goal
        target_H_task = 1.0 - goal_proximity.clamp(0, 1)
        task_potential_loss = F.mse_loss(torch.sigmoid(H_task), target_H_task)
        
        # === Auxiliary Loss 2: Barrier Awareness ===
        # barrier_k should be HIGHER when close to hazards
        k = actor.barrier_k_net(obs_batch)  # [batch, 1]
        # Enhanced: stronger response to hazards
        target_k_scale = 0.3 + hazard_proximity.clamp(0, 1) * 3.0  # Range [0.3, 3.3]
        barrier_k_loss = F.mse_loss(k, target_k_scale)
        
        # === Auxiliary Loss 3: Safety Loss (NEW) ===
        # Penalize being in dangerous positions (high hazard proximity)
        # This encourages the network to learn safer policies
        # Key insight: H_barrier should be responsive to hazard proximity
        danger_mask = (hazard_proximity > 0.5).float()  # In danger zone
        safety_loss = (danger_mask * (1.0 - H_barrier / (H_barrier.detach().max() + 1e-6))).mean()
        
        # === Auxiliary Loss 4: Gradient Alignment ===
        # Encourage barrier gradient to point away from obstacles
        # This helps the policy learn correct avoidance behavior
        _, grad_H_barrier = actor._compute_barrier_potential(obs_batch)
        grad_magnitude = torch.norm(grad_H_barrier, dim=-1, keepdim=True)
        # Barrier gradient should be large when near hazards
        target_grad_mag = hazard_proximity * 5.0
        grad_alignment_loss = F.mse_loss(grad_magnitude.clamp(max=5.0), target_grad_mag.clamp(max=5.0))
        
        # v6.0: Simplified auxiliary loss - focus on potential learning, not action penalization
        # The action danger penalty is removed; reward shaping handles safety guidance
        aux_loss = (self.aux_task_potential_weight * task_potential_loss + 
                    self.aux_barrier_potential_weight * (barrier_k_loss + grad_alignment_loss) +
                    self.aux_safety_weight * safety_loss)
        
        aux_info = {
            'aux_task_loss': task_potential_loss.item(),
            'aux_barrier_k_loss': barrier_k_loss.item(),
            'aux_safety_loss': safety_loss.item(),
            'aux_grad_align_loss': grad_alignment_loss.item(),
            'H_task_mean': H_task.mean().item(),
            'H_task_std': H_task.std().item(),
            'H_barrier_mean': H_barrier.mean().item(),
            'k_mean': k.mean().item(),
            'k_std': k.std().item(),
            'hazard_proximity_mean': hazard_proximity.mean().item(),
        }
        
        return aux_loss, aux_info

    def compute_potential_reward_shaping(self, obs_batch, next_obs_batch):
        """
        Compute potential-based reward shaping for guiding safe policy learning.
        
        v6.0: This is the key mechanism for learning safety through rewards.
        
        Using potential-based reward shaping (PBRS) which preserves optimal policy:
        F(s, s') = γ * Φ(s') - Φ(s)
        
        Where Φ is the potential function based on H_task and H_barrier.
        
        The reward shaping encourages:
        1. Moving toward goal (decreasing H_task)
        2. Staying away from obstacles (decreasing H_barrier or maintaining low H_barrier)
        
        Args:
            obs_batch: [batch, obs_dim] current observations
            next_obs_batch: [batch, obs_dim] next observations
            
        Returns:
            shaping_reward: [batch, 1] additional reward for safe behavior
        """
        actor = self.policy.actor
        gamma = self.config.get("gamma", 0.99)
        
        with torch.no_grad():
            # Compute potentials at current and next states
            H_task_curr, _ = actor._compute_task_potential(obs_batch)
            H_task_next, _ = actor._compute_task_potential(next_obs_batch)
            
            H_barrier_curr, _ = actor._compute_barrier_potential(obs_batch)
            H_barrier_next, _ = actor._compute_barrier_potential(next_obs_batch)
            
            # Task potential shaping: reward for decreasing H_task (moving toward goal)
            # Φ_task = -H_task (so decreasing H_task increases potential)
            task_shaping = gamma * (-H_task_next) - (-H_task_curr)  # = H_task_curr - γ * H_task_next
            
            # Barrier potential shaping: reward for staying safe (low H_barrier)
            # Φ_barrier = -H_barrier (so low H_barrier is good)
            # We want to encourage staying in low H_barrier regions
            barrier_shaping = gamma * (-H_barrier_next) - (-H_barrier_curr)  # = H_barrier_curr - γ * H_barrier_next
            
            # Additional: penalize entering danger zone
            _, proximity_curr = actor._extract_lidar_info(obs_batch)
            _, proximity_next = actor._extract_lidar_info(next_obs_batch)
            
            # Penalty for getting closer to obstacles
            proximity_penalty = torch.clamp(proximity_next - proximity_curr, min=0.0) * 2.0
            
            # Combine shaping rewards
            shaping_reward = (
                self.task_reward_weight * task_shaping +
                self.barrier_reward_weight * barrier_shaping -
                self.reward_shaping_weight * proximity_penalty
            )
            
            # Clip to prevent too large shaping rewards
            shaping_reward = torch.clamp(shaping_reward, min=-1.0, max=1.0)
        
        return shaping_reward

    def ppo_update(self, sample):
        share_obs_batch, obs_batch, rnn_states_batch, rnn_states_critic_batch, actions_batch, \
        value_preds_batch, return_batch, masks_batch, active_masks_batch, old_action_log_probs_batch, \
        adv_targ, available_actions_batch, _ = sample
        old_action_log_probs_batch, adv_targ, value_preds_batch, return_batch, active_masks_batch = [
            check(x).to(**self.tpdv) for x in [
                old_action_log_probs_batch, adv_targ, value_preds_batch, return_batch, active_masks_batch
            ]
        ]

        values, action_log_probs, dist_entropy = self.policy.evaluate_actions(share_obs_batch,
                                                                              obs_batch,
                                                                              rnn_states_batch,
                                                                              rnn_states_critic_batch,
                                                                              actions_batch,
                                                                              masks_batch,
                                                                              available_actions_batch,
                                                                              active_masks_batch)
        imp_weights = torch.exp(action_log_probs - old_action_log_probs_batch)

        surr1 = imp_weights * adv_targ
        surr2 = torch.clamp(imp_weights, 1.0 - self.config["clip_param"], 1.0 + self.config["clip_param"]) * adv_targ

        if self.config["use_policy_active_masks"]:
            policy_action_loss = (-torch.sum(torch.min(surr1, surr2),
                                             dim=-1,
                                             keepdim=True) * active_masks_batch).sum() / active_masks_batch.sum()
        else:
            policy_action_loss = -torch.sum(torch.min(surr1, surr2), dim=-1, keepdim=True).mean()

        policy_loss = policy_action_loss
        
        # Compute auxiliary physics loss for H_task and barrier_k networks
        aux_loss, aux_info = self.compute_auxiliary_physics_loss(check(obs_batch).to(**self.tpdv))
        
        # === Lagrangian Penalty for Cost Constraint (NEW) ===
        lagrangian_penalty = torch.tensor(0.0, device=self.config["device"])
        if self.use_lagrangian and len(self.recent_costs) > 0:
            # Use recent average cost to compute penalty
            avg_cost = sum(self.recent_costs) / len(self.recent_costs)
            cost_violation = max(0, avg_cost - self.cost_limit)
            
            # Lagrangian penalty: λ * (cost - limit)
            # This adds cost awareness to the policy loss
            lagrangian_penalty = self.lagrangian_multiplier.clamp(min=0.0) * cost_violation
            aux_info['lagrangian_multiplier'] = self.lagrangian_multiplier.item()
            aux_info['cost_violation'] = cost_violation
            aux_info['lagrangian_penalty'] = lagrangian_penalty.item()
        
        # Total actor loss = policy loss + auxiliary physics loss + lagrangian penalty
        total_actor_loss = policy_loss - dist_entropy * self.config["entropy_coef"] + aux_loss + lagrangian_penalty

        self.policy.actor_optimizer.zero_grad()
        total_actor_loss.backward()
        actor_grad_norm = nn.utils.clip_grad_norm_(self.policy.actor.parameters(), self.config["max_grad_norm"])
        self.policy.actor_optimizer.step()

        value_loss = self.cal_value_loss(values, value_preds_batch, return_batch, active_masks_batch)
        self.policy.critic_optimizer.zero_grad()
        (value_loss * self.config["value_loss_coef"]).backward()
        critic_grad_norm = nn.utils.clip_grad_norm_(self.policy.critic.parameters(), self.config["max_grad_norm"])

        self.policy.critic_optimizer.step()

        return value_loss, critic_grad_norm, policy_loss, dist_entropy, actor_grad_norm, imp_weights, aux_info

    def update_lagrangian(self, episode_cost):
        """
        Update Lagrangian multiplier based on episode cost.
        
        Uses dual gradient ascent:
        λ = λ + lr * (cost - cost_limit)
        
        Args:
            episode_cost: Average cost of the episode
        """
        if not self.use_lagrangian:
            return
        
        # Add to recent costs buffer
        self.recent_costs.append(episode_cost)
        if len(self.recent_costs) > self.cost_window:
            self.recent_costs.pop(0)
        
        # Compute cost violation
        avg_cost = sum(self.recent_costs) / len(self.recent_costs)
        cost_violation = avg_cost - self.cost_limit
        
        # Update Lagrangian multiplier using gradient ascent
        # λ += lr * (cost - limit)
        with torch.no_grad():
            self.lagrangian_multiplier.add_(self.lagrangian_lr * cost_violation)
            # Clamp to [0, max]
            self.lagrangian_multiplier.clamp_(min=0.0, max=self.lagrangian_max)

    def train(self, buffer, logger):
        advantages = buffer.returns[:-1] - self.value_normalizer.denormalize(buffer.value_preds[:-1])
        advantages_copy = advantages.clone()
        mean_advantages = torch.mean(advantages_copy)
        std_advantages = torch.std(advantages_copy)
        advantages = (advantages - mean_advantages) / (std_advantages + 1e-5)

        for _ in range(self.config["learning_iters"]):
            data_generator = buffer.feed_forward_generator(advantages, self.config["num_mini_batch"])

            for sample in data_generator:
                value_loss, critic_grad_norm, policy_loss, dist_entropy, actor_grad_norm, imp_weights, aux_info \
                    = self.ppo_update(sample)
        
        # Add Lagrangian info to logger
        lagrangian_info = {
            "Loss/Loss_reward_critic": value_loss.item(),
            "Loss/Loss_actor": policy_loss.item(),
            "Loss/Aux_task_potential": aux_info['aux_task_loss'],
            "Loss/Aux_barrier_k": aux_info['aux_barrier_k_loss'],
            "Loss/Aux_safety": aux_info['aux_safety_loss'],
            "Loss/Aux_grad_align": aux_info['aux_grad_align_loss'],
            "Safe/H_task_mean": aux_info['H_task_mean'],
            "Safe/H_task_std": aux_info['H_task_std'],
            "Safe/H_barrier_mean": aux_info['H_barrier_mean'],
            "Safe/k_mean": aux_info['k_mean'],
            "Safe/k_std": aux_info['k_std'],
            "Safe/hazard_proximity": aux_info['hazard_proximity_mean'],
            "Misc/Reward_critic_norm": critic_grad_norm.item(),
            "Misc/Entropy": dist_entropy.item(),
            "Misc/Ratio": imp_weights.detach().mean().item(),
        }
        
        # Add Lagrangian metrics if available
        if 'lagrangian_multiplier' in aux_info:
            lagrangian_info["Safe/lagrangian_multiplier"] = aux_info['lagrangian_multiplier']
            lagrangian_info["Safe/cost_violation"] = aux_info['cost_violation']
            lagrangian_info["Loss/lagrangian_penalty"] = aux_info['lagrangian_penalty']
        else:
            # Ensure these keys always exist to prevent logging errors
            if self.use_lagrangian:
                lagrangian_info["Safe/lagrangian_multiplier"] = 0.0  # Default value
                lagrangian_info["Safe/cost_violation"] = 0.0        # Default value
                lagrangian_info["Loss/lagrangian_penalty"] = 0.0    # Default value
        logger.store(**lagrangian_info)

    def prep_training(self):
        self.policy.actor.train()
        self.policy.critic.train()

    def prep_rollout(self):
        self.policy.actor.eval()
        self.policy.critic.eval()


# =============================================================================
# Runner (uses Barrier-PHS Safe PINN Policy/Trainer for Safe-pH-MARL)
# =============================================================================

class Runner:

    def __init__(self,
                 vec_env,
                 vec_eval_env,
                 config,
                 model_dir=""
                 ):
        self.envs = vec_env
        self.eval_envs = vec_eval_env
        self.config = config
        self.model_dir = model_dir

        self.num_agents = self.envs.num_agents

        # Counter for fixed-interval video rendering
        self.eval_count = 0
        self.video_record_freq = config.get("video_record_freq", 1)  # Record every N evals

        torch.autograd.set_detect_anomaly(True)
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
        
        # Initialize CUDA context early to avoid cuBLAS warning
        # This ensures the CUDA context is properly established before any operations
        # if "cuda" in str(self.config["device"]):
        #     device = self.config["device"]
        #     if isinstance(device, str) and ":" in device:
        #         device_id = int(device.split(":")[1])
        #     else:
        #         device_id = 0
        #     torch.cuda.set_device(device_id)
        #     # Create a dummy tensor to fully initialize the CUDA context
        #     _ = torch.zeros(1, device=self.config["device"])

        # Setup headless rendering for video recording
        setup_headless_rendering()
        
        # Initialize logger with wandb
        self.logger = EpochLogger(
            log_dir = config["log_dir"],
            seed = str(config["seed"]),
            use_wandb=config.get("use_wandb", True),
            wandb_project=config.get("wandb_project", "safepo"),
            wandb_config=config,
            verbose=False,
        )
        self.save_dir = str(config["log_dir"]+'/models_seed{}'.format(self.config["seed"]))
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        self.logger.save_config(config)
        
        # Initialize video recorder for evaluation
        self.video_recorder = MultiAgentVideoRecorder(
            fps=30,
            enabled=config.get("record_video", True),
            record_freq=config.get("video_record_freq", 10),
            max_episode_length=config.get("episode_length", 1000)
        )
        
        # Create Barrier-PHS Safe PINN policies for each agent
        self.policy = []
        for agent_id in range(self.num_agents):
            share_observation_space = self.envs.share_observation_space[agent_id]
            po = MAPPOSafePINNPolicy(config,
                        self.envs.observation_space[agent_id],
                        share_observation_space,
                        self.envs.action_space[agent_id]
                        )
            self.policy.append(po)

        if self.model_dir != "":
            self.restore()

        self.trainer = []
        self.buffer = []
        for agent_id in range(self.num_agents):
            tr = MAPPOSafePINNTrainer(config, self.policy[agent_id])
            share_observation_space = self.envs.share_observation_space[agent_id]

            bu = SeparatedReplayBuffer(config,
                                       self.envs.observation_space[agent_id],
                                       share_observation_space,
                                       self.envs.action_space[agent_id])
            self.buffer.append(bu)
            self.trainer.append(tr)

    def run(self):
        self.warmup()

        start = time.time()
        episodes = int(self.config["num_env_steps"]) // self.config["episode_length"] // self.config["n_rollout_threads"]

        train_episode_rewards = torch.zeros(1, self.config["n_rollout_threads"], device=self.config["device"])
        train_episode_costs = torch.zeros(1, self.config["n_rollout_threads"], device=self.config["device"])
        eval_rewards=0.0
        eval_costs=0.0
        
        # v7.0: Reward shaping is DISABLED by default
        # It caused train/eval mismatch (380 vs 8 reward gap)
        use_reward_shaping = self.config.get("use_reward_shaping", False)
        
        pbar = tqdm(range(episodes), desc="Safe-pH-MARL Training", ncols=100)
        for episode in pbar:

            done_episodes_rewards = []
            done_episodes_costs = []

            for step in range(self.config["episode_length"]):
                # Sample actions (no need to save prev_obs if shaping disabled)
                values, actions, action_log_probs, rnn_states, rnn_states_critic = self.collect(step)
                obs, share_obs, rewards, costs, dones, infos, _ = self.envs.step(actions)

                # v7.0: Reward shaping disabled - use original rewards only
                # This ensures train and eval rewards are comparable

                dones_env = torch.all(dones, dim=1)

                reward_env = torch.mean(rewards, dim=1).flatten()
                cost_env = torch.mean(costs, dim=1).flatten()

                train_episode_rewards += reward_env
                train_episode_costs += cost_env

                for t in range(self.config["n_rollout_threads"]):
                    if dones_env[t]:
                        done_episodes_rewards.append(train_episode_rewards[:, t].clone())
                        train_episode_rewards[:, t] = 0
                        done_episodes_costs.append(train_episode_costs[:, t].clone())
                        train_episode_costs[:, t] = 0

                data = obs, share_obs, rewards, dones, infos, \
                       values, actions, action_log_probs, \
                       rnn_states, rnn_states_critic

                self.insert(data)
            self.compute()
            self.train()

            total_num_steps = (episode + 1) * self.config["episode_length"] * self.config["n_rollout_threads"]

            if (episode % self.config["save_interval"] == 0 or episode == episodes - 1):
                self.save()
                
            end = time.time()
            
            if (episode % self.config["eval_interval"] == 0 or episode == episodes - 1) and self.config["use_eval"]:
                eval_rewards, eval_costs = self.eval(eval_episodes=1, total_steps=total_num_steps)

            if len(done_episodes_rewards) != 0:
                aver_episode_rewards = torch.stack(done_episodes_rewards).mean()
                aver_episode_costs = torch.stack(done_episodes_costs).mean()
                self.return_aver_cost(aver_episode_costs)
                
                # === Update Lagrangian multiplier based on episode cost ===
                for agent_id in range(self.num_agents):
                    self.trainer[agent_id].update_lagrangian(aver_episode_costs.item())
                
                # Collect barrier physics information for Safe module
                barrier_info = self.collect_barrier_physics_info(obs)
                
                log_dict = {
                    "Metrics/EpRet": aver_episode_rewards.item(),
                    "Metrics/EpCost": aver_episode_costs.item(),
                    "Eval/EpRet": eval_rewards,
                    "Eval/EpCost": eval_costs,
                }
                
                # Add Lagrangian info to log
                if self.trainer[0].use_lagrangian:
                    log_dict["Safe/lagrangian_multiplier"] = self.trainer[0].lagrangian_multiplier.item()
                    if len(self.trainer[0].recent_costs) > 0:
                        avg_cost = sum(self.trainer[0].recent_costs) / len(self.trainer[0].recent_costs)
                        log_dict["Safe/avg_recent_cost"] = avg_cost
                        log_dict["Safe/cost_limit"] = self.trainer[0].cost_limit
                
                # Add barrier physics to log dict
                log_dict.update(barrier_info)
                
                self.logger.store(**log_dict)
                
                self.logger.log_tabular("Metrics/EpRet", min_and_max=True, std=True)
                self.logger.log_tabular("Metrics/EpCost", min_and_max=True, std=True)
                self.logger.log_tabular("Eval/EpRet")
                self.logger.log_tabular("Eval/EpCost")
                self.logger.log_tabular("Train/Epoch", episode)
                self.logger.log_tabular("Train/TotalSteps", total_num_steps)
                self.logger.log_tabular("Loss/Loss_reward_critic")
                self.logger.log_tabular("Loss/Loss_actor")
                self.logger.log_tabular("Misc/Reward_critic_norm")
                self.logger.log_tabular("Misc/Entropy")
                self.logger.log_tabular("Misc/Ratio")
                
                # Log auxiliary physics losses (enhanced safety learning)
                self.logger.log_tabular("Loss/Aux_task_potential")
                self.logger.log_tabular("Loss/Aux_barrier_k")
                self.logger.log_tabular("Loss/Aux_safety")
                self.logger.log_tabular("Loss/Aux_grad_align")
                self.logger.log_tabular("Safe/H_task_mean")
                self.logger.log_tabular("Safe/H_task_std")
                self.logger.log_tabular("Safe/H_barrier_mean")
                self.logger.log_tabular("Safe/k_mean")
                self.logger.log_tabular("Safe/k_std")
                self.logger.log_tabular("Safe/hazard_proximity")
                
                # Log Lagrangian constraint info
                if self.trainer[0].use_lagrangian:
                    self.logger.log_tabular("Safe/lagrangian_multiplier")
                    self.logger.log_tabular("Safe/avg_recent_cost")
                    self.logger.log_tabular("Safe/cost_limit")
                    self.logger.log_tabular("Loss/lagrangian_penalty")
                    self.logger.log_tabular("Safe/cost_violation")
                
                # Log barrier physics parameters (Safe module)
                for physics_key in barrier_info.keys():
                    self.logger.log_tabular(physics_key)
                
                self.logger.log_tabular("Time/Total", end - start)
                self.logger.log_tabular("Time/FPS", int(total_num_steps / (end - start)))
                self.logger.dump_tabular(step=total_num_steps)
                
                # Update tqdm progress bar with key metrics
                pbar.set_postfix({
                    'EpRet': f"{aver_episode_rewards.item():.2f}",
                    'EpCost': f"{aver_episode_costs.item():.2f}",
                })
        pbar.close()

    def return_aver_cost(self, aver_episode_costs):
        for agent_id in range(self.num_agents):
            self.buffer[agent_id].return_aver_insert(aver_episode_costs)

    def warmup(self):
        # reset env
        obs, share_obs, _ = self.envs.reset()

        for agent_id in range(self.num_agents):
            self.buffer[agent_id].share_obs[0].copy_(share_obs[:, agent_id])
            if 'Frank'in self.config['env_name']:
                self.buffer[agent_id].obs[0].copy_(obs[agent_id])
            else:
                self.buffer[agent_id].obs[0].copy_(obs[:, agent_id])

    @torch.no_grad()
    def collect(self, step):
        value_collector = []
        action_collector = []
        action_log_prob_collector = []
        rnn_state_collector = []
        rnn_state_critic_collector = []
        for agent_id in range(self.num_agents):
            self.trainer[agent_id].prep_rollout()
            value, action, action_log_prob, rnn_state, rnn_state_critic \
                = self.trainer[agent_id].policy.get_actions(self.buffer[agent_id].share_obs[step],
                                                            self.buffer[agent_id].obs[step],
                                                            self.buffer[agent_id].rnn_states[step],
                                                            self.buffer[agent_id].rnn_states_critic[step],
                                                            self.buffer[agent_id].masks[step])
            value_collector.append(value.detach())
            action_collector.append(action.detach())

            action_log_prob_collector.append(action_log_prob.detach())
            rnn_state_collector.append(rnn_state.detach())
            rnn_state_critic_collector.append(rnn_state_critic.detach())
        if self.config["env_name"] == "Safety9|8HumanoidVelocity-v0":
            zeros = torch.zeros(action_collector[-1].shape[0], 1)
            action_collector[-1]=torch.cat((action_collector[-1], zeros), dim=1)
        values = torch.transpose(torch.stack(value_collector), 1, 0)
        rnn_states = torch.transpose(torch.stack(rnn_state_collector), 1, 0)
        rnn_states_critic = torch.transpose(torch.stack(rnn_state_critic_collector), 1, 0)

        return values, action_collector, action_log_prob_collector, rnn_states, rnn_states_critic

    @torch.no_grad()
    def collect_barrier_physics_info(self, obs):
        """
        Collect Barrier Port-Hamiltonian physics information for logging to wandb Safe module.
        
        Args:
            obs: observation tensor 
            
        Returns:
            dict with physics metrics averaged over agents
        """
        physics_info = {}
        
        for agent_id in range(self.num_agents):
            try:
                # Get physics info from actor
                if 'Frank' in self.config['env_name']:
                    obs_to_analyze = obs[agent_id]
                else:
                    obs_to_analyze = obs[:, agent_id]
                
                # Ensure obs is on the correct device
                obs_to_analyze = obs_to_analyze.to(self.config["device"])
                
                info = self.policy[agent_id].actor.get_physics_info(obs_to_analyze)
                
                # Average over batch dimension and convert to numpy
                for key, value in info.items():
                    if isinstance(value, torch.Tensor):
                        avg_val = value.mean().item()
                        physics_key = f"Safe/Agent{agent_id}_{key}"
                        if physics_key not in physics_info:
                            physics_info[physics_key] = []
                        physics_info[physics_key].append(avg_val)
            except Exception as e:
                # Silently skip if get_physics_info fails
                pass
        
        # Average across agents
        averaged_physics = {}
        for key, values in physics_info.items():
            if len(values) > 0:
                averaged_physics[key] = np.mean(values)
        
        return averaged_physics

    def _extract_agent_positions(self):
        """
        Extract current agent positions from the environment.
        
        For SafetyMultiGoal environments, agents are Point/Car robots with positions
        accessible via agent.pos_0 and agent.pos_1.
        
        Returns:
            agent_positions: List of (x, y) tuples for all agents
        """
        agent_positions = []
        
        try:
            if hasattr(self.eval_envs, 'envs') and len(self.eval_envs.envs) > 0:
                env = self.eval_envs.envs[0]
                
                # For MultiGoalEnv wrapper
                if hasattr(env, 'env') and hasattr(env.env, 'task'):
                    task = env.env.task
                    
                    # Multi-agent: access pos_0 and pos_1
                    if hasattr(task, 'agent'):
                        agent = task.agent
                        
                        if hasattr(agent, 'pos_0'):
                            pos_0 = agent.pos_0
                            agent_positions.append((float(pos_0[0]), float(pos_0[1])))
                        
                        if hasattr(agent, 'pos_1'):
                            pos_1 = agent.pos_1
                            agent_positions.append((float(pos_1[0]), float(pos_1[1])))
                
                # Alternative: Direct task access
                elif hasattr(env, 'task'):
                    task = env.task
                    if hasattr(task, 'agent'):
                        agent = task.agent
                        if hasattr(agent, 'pos_0'):
                            pos_0 = agent.pos_0
                            agent_positions.append((float(pos_0[0]), float(pos_0[1])))
                        if hasattr(agent, 'pos_1'):
                            pos_1 = agent.pos_1
                            agent_positions.append((float(pos_1[0]), float(pos_1[1])))
        
        except Exception as e:
            # Fallback: return empty (will use default)
            pass
        
        return agent_positions

    def _extract_env_obstacles(self):
        """
        Extract obstacle and goal positions from the environment for visualization.
        
        Returns:
            obstacle_positions: List of (x, y) tuples for hazards
            goal_positions: List of (x, y) tuples for goals
            hazard_radius: Radius of hazard obstacles
        """
        obstacle_positions = []
        goal_positions = []
        hazard_radius = 0.25  # Default hazard radius in safety_gymnasium MultiGoal
        
        try:
            # Try to get environment information from eval_envs
            # The underlying environment might have task info
            if hasattr(self.eval_envs, 'envs') and len(self.eval_envs.envs) > 0:
                env = self.eval_envs.envs[0]
                
                # For MultiGoalEnv wrapper: env.env is the actual PettingZoo environment
                task = None
                if hasattr(env, 'env') and hasattr(env.env, 'task'):
                    task = env.env.task
                elif hasattr(env, 'task'):
                    task = env.task
                
                if task is not None:
                    # Get hazard positions
                    if hasattr(task, 'hazards') and hasattr(task.hazards, 'pos'):
                        hazards_pos = task.hazards.pos
                        if hazards_pos is not None:
                            for pos in hazards_pos:
                                obstacle_positions.append((float(pos[0]), float(pos[1])))
                        if hasattr(task.hazards, 'size'):
                            hazard_radius = float(task.hazards.size)
                    
                    # Get goal positions
                    if hasattr(task, 'goal_red') and hasattr(task.goal_red, 'pos'):
                        goal_red_pos = task.goal_red.pos
                        if goal_red_pos is not None:
                            goal_positions.append((float(goal_red_pos[0]), float(goal_red_pos[1])))
                    
                    if hasattr(task, 'goal_blue') and hasattr(task.goal_blue, 'pos'):
                        goal_blue_pos = task.goal_blue.pos
                        if goal_blue_pos is not None:
                            goal_positions.append((float(goal_blue_pos[0]), float(goal_blue_pos[1])))
        
        except Exception as e:
            print(f"[Barrier Viz] Could not extract env obstacles: {e}")
        
        # Use defaults if extraction failed
        if len(obstacle_positions) == 0:
            # Default: 8 hazards evenly distributed (typical for MultiGoal1)
            obstacle_positions = [
                (0.8, 0.0), (-0.8, 0.0), (0.0, 0.8), (0.0, -0.8),
                (0.6, 0.6), (-0.6, 0.6), (0.6, -0.6), (-0.6, -0.6)
            ]
        
        if len(goal_positions) == 0:
            goal_positions = [(1.2, 1.2), (-1.2, -1.2)]
        
        return obstacle_positions, goal_positions, hazard_radius

    def insert(self, data):
        obs, share_obs, rewards, dones, infos, \
        values, actions, action_log_probs, rnn_states, rnn_states_critic = data

        dones_env = torch.all(dones, axis=1)

        rnn_states[dones_env == True] = torch.zeros(
            (dones_env == True).sum(), self.num_agents, self.config["recurrent_N"], self.config["hidden_size"], device=self.config["device"])
        rnn_states_critic[dones_env == True] = torch.zeros(
            (dones_env == True).sum(), self.num_agents, *self.buffer[0].rnn_states_critic.shape[2:], device=self.config["device"])

        masks = torch.ones(self.config["n_rollout_threads"], self.num_agents, 1, device=self.config["device"])
        masks[dones_env == True] = torch.zeros((dones_env == True).sum(), self.num_agents, 1, device=self.config["device"])

        active_masks = torch.ones(self.config["n_rollout_threads"], self.num_agents, 1, device=self.config["device"])
        active_masks[dones == True] = torch.zeros((dones == True).sum(), 1, device=self.config["device"])
        active_masks[dones_env == True] = torch.ones((dones_env == True).sum(), self.num_agents, 1, device=self.config["device"])

        if self.config["env_name"] == "Safety9|8HumanoidVelocity-v0":
            actions[1]=actions[1][:, :8]
        for agent_id in range(self.num_agents):
            if 'Frank'in self.config['env_name']:
                obs_to_insert = obs[agent_id]
            else:
                obs_to_insert = obs[:, agent_id]
            self.buffer[agent_id].insert(share_obs[:, agent_id], obs_to_insert, rnn_states[:, agent_id],
                                         rnn_states_critic[:, agent_id], actions[agent_id],
                                         action_log_probs[agent_id],
                                         values[:, agent_id], rewards[:, agent_id].unsqueeze(-1), masks[:, agent_id], None,
                                         active_masks[:, agent_id], None)

    def train(self):
        action_dim = 1
        factor = torch.ones(self.config["episode_length"], self.config["n_rollout_threads"], action_dim, device=self.config["device"])

        for agent_id in torch.randperm(self.num_agents):
            action_dim=self.buffer[agent_id].actions.shape[-1]

            self.trainer[agent_id].prep_training()
            self.buffer[agent_id].update_factor(factor)
            available_actions = None if self.buffer[agent_id].available_actions is None \
                else self.buffer[agent_id].available_actions[:-1].reshape(-1, *self.buffer[agent_id].available_actions.shape[2:])

            old_actions_logprob, _ =self.trainer[agent_id].policy.actor.evaluate_actions(self.buffer[agent_id].obs[:-1].reshape(-1, *self.buffer[agent_id].obs.shape[2:]),
                                                        self.buffer[agent_id].rnn_states[0:1].reshape(-1, *self.buffer[agent_id].rnn_states.shape[2:]),
                                                        self.buffer[agent_id].actions.reshape(-1, *self.buffer[agent_id].actions.shape[2:]),
                                                        self.buffer[agent_id].masks[:-1].reshape(-1, *self.buffer[agent_id].masks.shape[2:]),
                                                        available_actions,
                                                        self.buffer[agent_id].active_masks[:-1].reshape(-1, *self.buffer[agent_id].active_masks.shape[2:]))
            self.trainer[agent_id].train(self.buffer[agent_id], logger=self.logger)

            new_actions_logprob, _ =self.trainer[agent_id].policy.actor.evaluate_actions(self.buffer[agent_id].obs[:-1].reshape(-1, *self.buffer[agent_id].obs.shape[2:]),
                                                        self.buffer[agent_id].rnn_states[0:1].reshape(-1, *self.buffer[agent_id].rnn_states.shape[2:]),
                                                        self.buffer[agent_id].actions.reshape(-1, *self.buffer[agent_id].actions.shape[2:]),
                                                        self.buffer[agent_id].masks[:-1].reshape(-1, *self.buffer[agent_id].masks.shape[2:]),
                                                        available_actions,
                                                        self.buffer[agent_id].active_masks[:-1].reshape(-1, *self.buffer[agent_id].active_masks.shape[2:]))

            action_prod = torch.prod(torch.exp(new_actions_logprob.detach()-old_actions_logprob.detach()).reshape(self.config["episode_length"],self.config["n_rollout_threads"],action_dim), dim=-1, keepdim=True)
            factor = factor*action_prod.detach()
            self.buffer[agent_id].after_update()

    def save(self):
        for agent_id in range(self.num_agents):
            policy_actor = self.trainer[agent_id].policy.actor
            torch.save(policy_actor.state_dict(), str(self.save_dir) + "/actor_agent" + str(agent_id) + ".pt")
            policy_critic = self.trainer[agent_id].policy.critic
            torch.save(policy_critic.state_dict(), str(self.save_dir) + "/critic_agent" + str(agent_id) + ".pt")

    def restore(self):
        for agent_id in range(self.num_agents):
            policy_actor_state_dict = torch.load(str(self.model_dir) + '/actor_agent' + str(agent_id) + '.pt')
            self.policy[agent_id].actor.load_state_dict(policy_actor_state_dict)
            policy_critic_state_dict = torch.load(str(self.model_dir) + '/critic_agent' + str(agent_id) + '.pt')
            self.policy[agent_id].critic.load_state_dict(policy_critic_state_dict)

    @torch.no_grad()
    def eval(self, eval_episodes=1, total_steps=0):
        """
        Evaluate policy performance with dynamic barrier potential field video rendering.
        
        Renders side-by-side video showing environment frame and learned barrier potential
        field for the FIRST episode of each evaluation run (every eval_interval).
        This provides visual understanding of the agent's safety behavior.
        """
        self.eval_count += 1
        should_record_video = (
            self.video_recorder.enabled 
            and self.eval_count % self.video_record_freq == 0
            and self.config["env_name"] not in isaac_gym_map
        )
        
        # Check if this is a MultiGoal task (for potential field visualization)
        is_multi_goal_task = self.config["env_name"] in multi_agent_goal_tasks
        
        eval_episode = 0
        eval_episode_rewards = []
        eval_episode_costs = []
        one_episode_rewards = torch.zeros(1, self.config["n_eval_rollout_threads"], device=self.config["device"])
        one_episode_costs = torch.zeros(1, self.config["n_eval_rollout_threads"], device=self.config["device"])
        
        # Fixed-interval video: record first episode only
        first_episode_frames = []
        potential_field_frames = []  # Combined frames with potential field
        first_episode_reward = 0.0
        first_episode_cost = 0.0
        recording_first_episode = should_record_video  # Start recording immediately for first episode
        step_count = 0

        # Initialize potential field visualizer for MultiGoal tasks
        potential_visualizer = None
        if should_record_video and is_multi_goal_task:
            try:
                obstacle_positions, goal_positions, hazard_radius = self._extract_env_obstacles()
                potential_visualizer = BarrierPotentialVideoVisualizer(
                    actor=self.policy[0].actor,
                    world_bounds=(-2.5, 2.5, -2.5, 2.5),
                    grid_resolution=50,
                    device='cpu',
                    hazard_radius=hazard_radius,
                )
            except Exception as e:
                print(f"[Barrier Viz] Failed to initialize visualizer: {e}")
                potential_visualizer = None

        eval_obs, _, _ = self.eval_envs.reset()

        eval_rnn_states = torch.zeros(self.config["n_eval_rollout_threads"], self.num_agents, self.config["recurrent_N"], self.config["hidden_size"],
                                   device=self.config["device"])
        eval_masks = torch.ones(self.config["n_eval_rollout_threads"], self.num_agents, 1, device=self.config["device"])

        while True:
            eval_actions_collector = []
            for agent_id in range(self.num_agents):
                self.trainer[agent_id].prep_rollout()
                obs_to_eval = eval_obs[:, agent_id]
                eval_actions, temp_rnn_state = self.trainer[agent_id].policy.act(obs_to_eval,
                                                      eval_rnn_states[:, agent_id],
                                                      eval_masks[:, agent_id],
                                                      deterministic=True)
                eval_rnn_states[:, agent_id] = temp_rnn_state
                eval_actions_collector.append(eval_actions)

            # if self.config["env_name"] == "Safety9|8HumanoidVelocity-v0":
            #     zeros = torch.zeros(eval_actions_collector[-1].shape[0], 1)
            #     eval_actions_collector[-1]=torch.cat((eval_actions_collector[-1], zeros), dim=1)
            
            # Capture frame for first episode video (fixed-interval recording)
            if recording_first_episode and hasattr(self.eval_envs, 'render'):
                try:
                    # Render with exception handling for OpenGL/EGL errors
                    try:
                        frame = self.eval_envs.render()
                    except Exception as render_error:
                        # Silently skip frame if render fails (especially EGL errors)
                        frame = None
                        
                    if frame is not None:
                        if isinstance(frame, np.ndarray) and len(frame.shape) == 3:
                            first_episode_frames.append(frame.copy())
                            
                            # Generate combined potential field frame for MultiGoal tasks
                            if potential_visualizer is not None and is_multi_goal_task:
                                try:
                                    # Get current positions
                                    obstacle_positions, goal_positions, _ = self._extract_env_obstacles()
                                    agent_positions = self._extract_agent_positions()
                                    
                                    # Use new all-potentials visualization (barrier, task, total)
                                    combined_frame, barrier_img, task_img, total_img = potential_visualizer.render_all_potentials_frame(
                                        env_frame=frame.copy(),
                                        obstacle_positions=np.array(obstacle_positions),
                                        goal_positions=np.array(goal_positions),
                                        agent_positions=np.array(agent_positions) if agent_positions else None,
                                        step=step_count,
                                        task_potential_scale=2.0,
                                    )
                                    potential_field_frames.append(combined_frame)
                                    
                                    # Store latest individual potential images for wandb logging
                                    if step_count == 0:  # Initialize storage
                                        self._latest_barrier_img = barrier_img
                                        self._latest_task_img = task_img
                                        self._latest_total_img = total_img
                                    else:  # Update with latest
                                        self._latest_barrier_img = barrier_img
                                        self._latest_task_img = task_img
                                        self._latest_total_img = total_img
                                        
                                except Exception as e:
                                    # Fallback: just use the original frame
                                    pass
                            
                            step_count += 1
                except Exception as e:
                    pass

            eval_obs, _, eval_rewards, eval_costs, eval_dones, _, _ = self.eval_envs.step(
                eval_actions_collector
            )

            reward_env = torch.mean(eval_rewards, dim=1).flatten()
            cost_env = torch.mean(eval_costs, dim=1).flatten()

            one_episode_rewards += reward_env
            one_episode_costs += cost_env

            eval_dones_env = torch.all(eval_dones, dim=1)

            eval_rnn_states[eval_dones_env == True] = torch.zeros(
                (eval_dones_env == True).sum(), self.num_agents, self.config["recurrent_N"], self.config["hidden_size"], device=self.config["device"])

            eval_masks = torch.ones(self.config["n_eval_rollout_threads"], self.num_agents, 1, device=self.config["device"])
            eval_masks[eval_dones_env == True] = torch.zeros((eval_dones_env == True).sum(), self.num_agents, 1,
                                                          device=self.config["device"])

            for eval_i in range(self.config["n_eval_rollout_threads"]):
                if eval_dones_env[eval_i]:
                    eval_episode += 1
                    ep_reward = one_episode_rewards[:, eval_i].mean().item()
                    ep_cost = one_episode_costs[:, eval_i].mean().item()
                    eval_episode_rewards.append(ep_reward)
                    eval_episode_costs.append(ep_cost)

                    # For fixed-interval recording: stop after first episode completes
                    if recording_first_episode and eval_episode == 1:
                        first_episode_reward = ep_reward
                        first_episode_cost = ep_cost
                        recording_first_episode = False  # Stop recording after first episode

                    one_episode_rewards[:, eval_i] = 0
                    one_episode_costs[:, eval_i] = 0

            if eval_episode >= eval_episodes:
                # Upload video for the first episode (fixed-interval recording)
                # if len(first_episode_frames) > 0 and should_record_video:
                #     if self.video_recorder.enabled:
                #         self.video_recorder.recorder.frames = first_episode_frames
                #         caption = f"Eval #{self.eval_count} - Reward: {first_episode_reward:.2f}, Cost: {first_episode_cost:.2f}"
                #         self.video_recorder.recorder.upload_to_wandb(
                #             caption=caption,
                #             step=total_steps,
                #             key="eval/video"
                #         )
                
                # Upload potential field video and images for MultiGoal tasks to Viz module
                if len(potential_field_frames) > 0 and should_record_video and is_multi_goal_task:
                    try:
                        # Create visualization directory
                        viz_dir = os.path.join(os.path.dirname(self.save_dir), "vizs")
                        os.makedirs(viz_dir, exist_ok=True)
                        
                        # Save video file locally
                        video_path = os.path.join(viz_dir, f"potential_field_step{total_steps}.mp4")
                        if potential_visualizer is not None:
                            potential_visualizer.save_video(potential_field_frames, video_path, fps=30)
                        
                        # Upload to wandb Viz module (not Eval)
                        if self.logger.use_wandb:
                            import wandb
                            
                            try:
                                # Stack frames and convert to (T, C, H, W) format for wandb
                                # Input frames are (H, W, C), need to transpose to (C, H, W)
                                video_array = np.stack(potential_field_frames, axis=0)  # (T, H, W, C)
                                video_array = np.transpose(video_array, (0, 3, 1, 2))   # (T, C, H, W)
                                
                                caption = f"All Potentials - Eval #{self.eval_count} - Reward: {first_episode_reward:.2f}, Cost: {first_episode_cost:.2f}"
                                video_obj = wandb.Video(video_array, fps=30, format="mp4", caption=caption)
                                
                                # Create wandb log dict with video and images (use dict for mixed types)
                                viz_log: dict = {"Viz/all_potentials_video": video_obj}
                                
                                # Add individual potential field images
                                if hasattr(self, '_latest_barrier_img') and self._latest_barrier_img is not None:
                                    viz_log["Viz/barrier_potential"] = wandb.Image(
                                        self._latest_barrier_img, 
                                        caption=f"Barrier Potential - Eval #{self.eval_count}"
                                    )
                                if hasattr(self, '_latest_task_img') and self._latest_task_img is not None:
                                    viz_log["Viz/task_potential"] = wandb.Image(
                                        self._latest_task_img,
                                        caption=f"Task Potential - Eval #{self.eval_count}"
                                    )
                                if hasattr(self, '_latest_total_img') and self._latest_total_img is not None:
                                    viz_log["Viz/total_potential"] = wandb.Image(
                                        self._latest_total_img,
                                        caption=f"Total Potential - Eval #{self.eval_count}"
                                    )
                                
                                # Upload to wandb Viz module (separate from Eval module)
                                if hasattr(self.logger, 'wandb_run') and self.logger.wandb_run is not None:
                                    self.logger.wandb_run.log(viz_log, step=total_steps)
                                elif wandb.run is not None:
                                    wandb.log(viz_log, step=total_steps)
                                    
                            except Exception as e:
                                # Silently ignore wandb upload errors
                                pass
                    
                    except Exception as e:
                        print(f"[Barrier Viz] Exception during video generation: {type(e).__name__}: {str(e)}")
                        import traceback
                        traceback.print_exc()
                
                # Clean up OpenGL context to avoid EGL errors on final render
                try:
                    # Close any matplotlib figures
                    try:
                        import matplotlib.pyplot as plt
                        plt.close('all')
                    except:
                        pass
                    
                    # Close extra eval environments if available
                    if hasattr(self.eval_envs, 'close_extra_envs'):
                        self.eval_envs.close_extra_envs()
                    
                    # Force garbage collection to clean up OpenGL/EGL resources
                    import gc
                    gc.collect()
                except Exception as e:
                    pass  # Silently ignore cleanup errors
                
                return np.mean(eval_episode_rewards), np.mean(eval_episode_costs)

    @torch.no_grad()
    def compute(self):
        for agent_id in range(self.num_agents):
            self.trainer[agent_id].prep_rollout()
            next_value = self.trainer[agent_id].policy.get_values(self.buffer[agent_id].share_obs[-1],
                                                                self.buffer[agent_id].rnn_states_critic[-1],
                                                                self.buffer[agent_id].masks[-1])
            next_value = next_value.detach()
            self.buffer[agent_id].compute_returns(next_value, self.trainer[agent_id].value_normalizer)


def train(args, cfg_train):
    # Initialize CUDA context early to avoid cuBLAS warning
    # Must happen before any model creation or tensor operations
    # device_str = cfg_train.get("device", "cuda:0")
    # if "cuda" in device_str:
    #     if ":" in device_str:
    #         device_id = int(device_str.split(":")[1])
    #     else:
    #         device_id = 0
    #     torch.cuda.set_device(device_id)
    #     # Warm up CUDA context with a small operation
    #     _ = torch.zeros(1, device=device_str)
    #     # Force synchronization to ensure context is fully initialized
    #     torch.cuda.synchronize()
    
    agent_index = [[[0, 1, 2, 3, 4, 5]],
                   [[0, 1, 2, 3, 4, 5]]]
    if args.task in multi_agent_velocity_map:
        env = make_ma_mujoco_env(
        scenario=args.scenario,
        agent_conf=args.agent_conf,
        seed=args.seed,
        cfg_train=cfg_train,
    )
        cfg_eval = copy.deepcopy(cfg_train)
        cfg_eval["seed"] = args.seed + 10000
        cfg_eval["n_rollout_threads"] = cfg_eval["n_eval_rollout_threads"]
        cfg_eval["render_mode"] = "rgb_array"  # Enable rendering for evaluation
        eval_env = make_ma_mujoco_env(
        scenario=args.scenario,
        agent_conf=args.agent_conf,
        seed=cfg_eval['seed'],
        cfg_train=cfg_eval,
    )
    elif args.task in isaac_gym_map:
        sim_params = parse_sim_params(args, cfg_env, cfg_train)
        env = make_ma_isaac_env(args, cfg_env, cfg_train, sim_params, agent_index)
        cfg_train["n_rollout_threads"] = env.num_envs
        cfg_train["n_eval_rollout_threads"] = env.num_envs
        eval_env = env
    elif args.task in multi_agent_goal_tasks:
        env = make_ma_multi_goal_env(task=args.task, seed=args.seed, cfg_train=cfg_train)
        cfg_eval = copy.deepcopy(cfg_train)
        cfg_eval["seed"] = args.seed + 10000
        cfg_eval["n_rollout_threads"] = cfg_eval["n_eval_rollout_threads"]
        eval_env = make_ma_multi_goal_env(task=args.task, seed=args.seed + 10000, cfg_train=cfg_eval)
    else: 
        raise NotImplementedError
    
    torch.set_num_threads(4)
    runner = Runner(env, eval_env, cfg_train, args.model_dir)

    if args.model_dir != "":
        runner.eval(10)
    else:
        runner.run()


if __name__ == '__main__':
    set_np_formatting()
    args, cfg_env, cfg_train = multi_agent_args(algo="mappo_safe_pinn")
    set_seed(cfg_train.get("seed", -1), cfg_train.get("torch_deterministic", False))
    if args.write_terminal:
        train(args=args, cfg_train=cfg_train)
    else:
        terminal_log_name = "terminal.log"
        error_log_name = "error.log"
        terminal_log_name = f"seed{args.seed}_{terminal_log_name}"
        error_log_name = f"seed{args.seed}_{error_log_name}"
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__
        if not os.path.exists(cfg_train['log_dir']):
            os.makedirs(cfg_train['log_dir'], exist_ok=True)
        with open(
            os.path.join(
                f"{cfg_train['log_dir']}",
                terminal_log_name,
            ),
            "w",
            encoding="utf-8",
        ) as f_out:
            sys.stdout = f_out
            with open(
                os.path.join(
                    f"{cfg_train['log_dir']}",
                    error_log_name,
                ),
                "w",
                encoding="utf-8",
            ) as f_error:
                sys.stderr = f_error
                train(args=args, cfg_train=cfg_train)

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
MAPPO-Safe-PINN v2: True PHS-MAPPO with Port-Hamiltonian Embedded Actor.

This version implements the full PHS-MAPPO framework from PHS-MAPPO.md, where
actions are directly computed from Port-Hamiltonian dynamics rather than
just using physics as additional features.

Key Innovations from v2:
    1. True PHS Action Generation:
       - Actions computed from: u = F⁻¹(dx - (J_sys - R_sys) ∇H_sys)
       - Not just physics features concatenated to MLP
       
    2. Learned System Matrices:
       - J: Skew-symmetric interconnection (via attention)
       - R: Symmetric PSD dissipation (via attention)
       
    3. Multi-Agent Coupling:
       - Laplacian matrix for agent-agent interaction
       - SoftBarrierHead learns pairwise stiffness k_ij
       
    4. Combined Potentials:
       - H_goal: Quadratic goal attraction (from goal lidar)
       - H_task_learned: Learned task potential (neural network)
       - H_task = H_goal + H_task_learned (explicit + learned)
       - H_barrier: Log barrier for collision avoidance
       - H_kin: Kinetic energy penalty
       
    5. Barrier Warmup Schedule:
       - Warmup → Plateau → Decay for stable training

For SafetyMultiGoal environments:
    - Observation: ~152-dim (accelerometer, velocimeter, gyro, magnetometer, lidars)
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

# Initialize CUDA context at module load time
if torch.cuda.is_available():
    try:
        torch.cuda.set_device(0)
        _x = torch.randn(4, 4, device='cuda', requires_grad=True)
        _y = torch.randn(4, 4, device='cuda', requires_grad=True)
        _z = torch.mm(_x, _y)
        _loss = _z.sum()
        _loss.backward()
        torch.cuda.synchronize()
        del _x, _y, _z, _loss
    except Exception:
        pass

from safepo.common.env import make_ma_mujoco_env, make_ma_isaac_env, make_ma_multi_goal_env
from safepo.common.popart import PopArt
from safepo.common.model import MultiAgentCritic as Critic
from safepo.common.buffer import SeparatedReplayBuffer
from safepo.common.logger import EpochLogger
from safepo.common.video_recorder import MultiAgentVideoRecorder, setup_headless_rendering
from safepo.utils.config import multi_agent_args, parse_sim_params, set_np_formatting, set_seed, multi_agent_velocity_map, isaac_gym_map, multi_agent_goal_tasks
from safepo.utils.visualize_barrier_potential import BarrierPotentialVisualizer
from safepo.utils.barrier_potential_video_visualizer import BarrierPotentialVideoVisualizer

# Import the new PHS-MAPPO Actor
from safepo.multi_agent.phs_mappo_actor import PHSMAPPOActor


def check(input):
    output = torch.from_numpy(input) if type(input) == np.ndarray else input
    return output

def huber_loss(e, d):
    a = (abs(e) <= d).float()
    b = (e > d).float()
    return a*e**2/2 + b*d*(abs(e)-d/2)


# =============================================================================
# MAPPO-Safe-PINN v2 Policy with True PHS Actor
# =============================================================================

class MAPPOSafePINNv2Policy:
    """
    MAPPO policy with true Port-Hamiltonian embedded Actor.
    
    Key difference from v2:
    - Uses PHSMAPPOActor which computes actions directly from PHS dynamics
    - Actions emerge from physical principles, not just physics-informed features
    """

    def __init__(self, config, obs_space, cent_obs_space, act_space, n_agents=1, agent_id=0):
        self.config = config
        self.obs_space = obs_space
        self.act_space = act_space
        self.share_obs_space = cent_obs_space
        self.n_agents = n_agents
        self.agent_id = agent_id

        # Use new PHS-MAPPO Actor with embedded physics
        self.actor = PHSMAPPOActor(
            config, 
            self.obs_space, 
            self.act_space, 
            self.config["device"],
            n_agents=n_agents,
            agent_id=agent_id  # Pass agent_id so it knows which goal to focus on
        )
        
        # Standard Critic for reward
        self.critic = Critic(config, self.share_obs_space, self.config["device"])
        
        # Cost Critic for safety constraint (MAPPO-Lagrangian style)
        self.cost_critic = Critic(config, self.share_obs_space, self.config["device"])

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
        self.cost_optimizer = torch.optim.Adam(
            self.cost_critic.parameters(), 
            lr=self.config["critic_lr"], 
            eps=self.config["opti_eps"], 
            weight_decay=self.config["weight_decay"]
        )

    def get_actions(self, cent_obs, obs, rnn_states_actor, rnn_states_critic, masks, 
                    available_actions=None, deterministic=False, rnn_states_cost=None):
        actions, action_log_probs, rnn_states_actor = self.actor(
            obs, rnn_states_actor, masks, available_actions, deterministic
        )
        values, rnn_states_critic = self.critic(cent_obs, rnn_states_critic, masks)
        
        if rnn_states_cost is None:
            return values, actions, action_log_probs, rnn_states_actor, rnn_states_critic
        else:
            cost_preds, rnn_states_cost = self.cost_critic(cent_obs, rnn_states_cost, masks)
            return values, actions, action_log_probs, rnn_states_actor, rnn_states_critic, cost_preds, rnn_states_cost

    def get_values(self, cent_obs, rnn_states_critic, masks):
        values, _ = self.critic(cent_obs, rnn_states_critic, masks)
        return values
    
    def get_cost_values(self, cent_obs, rnn_states_cost, masks):
        cost_preds, _ = self.cost_critic(cent_obs, rnn_states_cost, masks)
        return cost_preds

    def evaluate_actions(self, cent_obs, obs, rnn_states_actor, rnn_states_critic, action, masks,
                         available_actions=None, active_masks=None, rnn_states_cost=None):
        action_log_probs, dist_entropy = self.actor.evaluate_actions(
            obs, rnn_states_actor, action, masks, available_actions, active_masks
        )
        values, _ = self.critic(cent_obs, rnn_states_critic, masks)
        
        if rnn_states_cost is None:
            return values, action_log_probs, dist_entropy
        else:
            cost_values, _ = self.cost_critic(cent_obs, rnn_states_cost, masks)
            return values, action_log_probs, dist_entropy, cost_values

    def act(self, obs, rnn_states_actor, masks, available_actions=None, deterministic=False):
        actions, _, rnn_states_actor = self.actor(
            obs, rnn_states_actor, masks, available_actions, deterministic
        )
        return actions, rnn_states_actor


# =============================================================================
# MAPPO-Safe-PINN v2 Trainer with PHS-Aware Training
# =============================================================================

class MAPPOSafePINNv2Trainer:
    """
    Trainer for PHS-MAPPO that includes:
    1. Hybrid advantage (reward - λ * cost) for MAPPO-Lagrangian
    2. Auxiliary physics loss for better potential learning
    3. Barrier warmup schedule management
    """

    def __init__(self, config, policy):
        self.config = config
        self.tpdv = dict(dtype=torch.float32, device=self.config["device"])
        self.policy = policy

        self.value_normalizer = PopArt(1, device=self.config["device"])
        self.cost_value_normalizer = PopArt(1, device=self.config["device"])
        
        # Auxiliary loss weights (lighter than v2 since PHS handles physics)
        self.aux_task_potential_weight = config.get("aux_task_potential_weight", 0.01)
        self.aux_barrier_potential_weight = config.get("aux_barrier_potential_weight", 0.02)
        self.aux_safety_weight = config.get("aux_safety_weight", 0.01)
        self.aux_agent_collision_weight = config.get("aux_agent_collision_weight", 0.02)
        self.aux_loss_scale = config.get("aux_loss_scale", 0.5)
        self.aux_warmup_steps = config.get("aux_warmup_steps", 2000)
        self.aux_cost_value_weight = config.get("aux_cost_value_weight", 0.05)
        self.aux_cost_k_weight = config.get("aux_cost_k_weight", 0.02)
        self.cost_value_scale = config.get("cost_value_scale", 10.0)

        # Decouple Barrier PHS from Lagrange learning
        self.decouple_barrier_lagrange = config.get("decouple_barrier_lagrange", True)
        if self.decouple_barrier_lagrange:
            self.aux_barrier_potential_weight = 0.0
            self.aux_safety_weight = 0.0
            self.aux_agent_collision_weight = 0.0
            self.aux_cost_value_weight = 0.0
            self.aux_cost_k_weight = 0.0
        
        # Lagrangian parameters
        self.lamda_lagr = config.get("lamda_lagr", 0.5)
        self.lamda_lagr_min = config.get("lamda_lagr_min", 0.1)
        self.lamda_lagr_max = config.get("lamda_lagr_max", 5.0)
        self.lagrangian_update_interval = config.get("lagrangian_update_interval", 10)
        self.lagrangian_ema_alpha = config.get("lagrangian_ema_alpha", 0.9)
        self.lagrangian_slow_rate = config.get("lagrangian_slow_rate", 0.005)
        self._ema_cost = None
        
        # Soft cost for Cost Critic training
        self.soft_cost_weight = config.get("soft_cost_weight", 0.1)
        self.soft_cost_warmup_steps = config.get("soft_cost_warmup_steps", 2000)
        self.soft_cost_start = config.get("soft_cost_start", 0)
        
        # Training step counter for barrier warmup
        self._training_step = 0

    def _get_aux_loss_scale(self):
        """Scale auxiliary losses to avoid over-regularization early in training."""
        if self.aux_warmup_steps <= 0:
            return self.aux_loss_scale

        if self._training_step < self.aux_warmup_steps:
            ratio = self._training_step / float(self.aux_warmup_steps)
            return self.aux_loss_scale * ratio

        return self.aux_loss_scale

    def _get_soft_cost_weight(self):
        """Gradually enable soft cost to avoid double-penalizing early."""
        if self._training_step < self.soft_cost_start:
            return 0.0

        if self.soft_cost_warmup_steps <= 0:
            return self.soft_cost_weight

        warmup_progress = (self._training_step - self.soft_cost_start) / float(self.soft_cost_warmup_steps)
        warmup_progress = max(0.0, min(1.0, warmup_progress))
        return self.soft_cost_weight * warmup_progress

    def cal_value_loss(self, values, value_preds_batch, return_batch, active_masks_batch):
        value_pred_clipped = value_preds_batch + (values - value_preds_batch).clamp(
            -self.config["clip_param"], self.config["clip_param"]
        )
        error_clipped = self.value_normalizer(return_batch) - value_pred_clipped
        error_original = self.value_normalizer(return_batch) - values

        value_loss_clipped = huber_loss(error_clipped, self.config["huber_delta"])
        value_loss_original = huber_loss(error_original, self.config["huber_delta"])

        value_loss = torch.max(value_loss_original, value_loss_clipped)
        return value_loss.mean()

    def compute_auxiliary_physics_loss(self, obs_batch, cost_values=None):
        """
        Compute auxiliary losses for PHS potential networks.
        
        Lighter weight than v2 since true PHS handles physics directly.
        Focus on:
        1. Task potential guidance toward goal
        2. Barrier stiffness response to hazards
        """
        device = obs_batch.device
        actor = self.policy.actor
        
        # Compute potentials
        H_task, _ = actor._compute_task_potential(obs_batch)
        H_barrier, _ = actor._compute_barrier_potential(obs_batch)
        
        # Extract lidar info
        goal_lidar, goal_proximity = actor._extract_goal_lidar_info(obs_batch)
        hazard_lidar, hazard_proximity = actor._extract_lidar_info(obs_batch)
        
        # Task potential loss: should be LOW near goal
        target_H_task = 1.0 - goal_proximity.clamp(0, 1)
        task_potential_loss = F.mse_loss(torch.sigmoid(H_task), target_H_task)
        
        # Barrier awareness loss: k should be HIGH near hazards
        k = actor.obstacle_k_net(obs_batch)
        target_k = 0.3 + hazard_proximity.clamp(0, 1) * 2.0
        barrier_k_loss = F.mse_loss(k, target_k)
        
        # Safety loss: H_barrier should predict danger
        H_barrier_norm = torch.sigmoid(H_barrier / 5.0)
        danger_level = torch.clamp(hazard_proximity - 0.5, min=0.0) / 0.5
        safety_loss = F.mse_loss(H_barrier_norm, danger_level)
        
        # Agent collision loss (if agent lidar available)
        agent_lidar_start = actor.agent_lidar_start
        agent_lidar_end = min(actor.agent_lidar_end, obs_batch.shape[-1])
        if agent_lidar_start < agent_lidar_end:
            agent_lidar = obs_batch[:, agent_lidar_start:agent_lidar_end]
            agent_proximity = agent_lidar.max(dim=-1, keepdim=True)[0]
            agent_danger = torch.clamp(agent_proximity - 0.4, min=0.0) / 0.6
            agent_collision_loss = (agent_danger ** 2).mean()
        else:
            agent_collision_loss = torch.tensor(0.0, device=device)
        
        # Cost critic guidance (disabled when decoupled)
        cost_guidance_loss = torch.tensor(0.0, device=device)
        cost_k_loss = torch.tensor(0.0, device=device)
        if (not self.decouple_barrier_lagrange) and cost_values is not None:
            cost_target = torch.sigmoid(cost_values.detach() / self.cost_value_scale)
            H_barrier_norm = torch.sigmoid(H_barrier / 5.0)
            cost_guidance_loss = F.mse_loss(H_barrier_norm, cost_target)

            # Encourage stiffness to rise where cost is predicted high
            target_k_cost = 0.3 + cost_target * 2.0
            cost_k_loss = F.mse_loss(k, target_k_cost)

        # Combined auxiliary loss
        aux_loss = (
            self.aux_task_potential_weight * task_potential_loss +
            self.aux_barrier_potential_weight * barrier_k_loss +
            self.aux_safety_weight * safety_loss +
            self.aux_agent_collision_weight * agent_collision_loss +
            self.aux_cost_value_weight * cost_guidance_loss +
            self.aux_cost_k_weight * cost_k_loss
        )

        aux_loss = aux_loss * self._get_aux_loss_scale()
        
        aux_info = {
            'aux_task_loss': task_potential_loss.item(),
            'aux_barrier_k_loss': barrier_k_loss.item(),
            'aux_safety_loss': safety_loss.item(),
            'aux_agent_collision_loss': agent_collision_loss.item() if isinstance(agent_collision_loss, torch.Tensor) else 0.0,
            'aux_cost_value_loss': cost_guidance_loss.item() if isinstance(cost_guidance_loss, torch.Tensor) else 0.0,
            'aux_cost_k_loss': cost_k_loss.item() if isinstance(cost_k_loss, torch.Tensor) else 0.0,
            'H_task_mean': H_task.mean().item(),
            'H_task_std': H_task.std().item(),
            'H_barrier_mean': H_barrier.mean().item(),
            'k_mean': k.mean().item(),
            'k_std': k.std().item(),
            'hazard_proximity_mean': hazard_proximity.mean().item(),
            'agent_proximity_mean': agent_proximity.mean().item() if agent_lidar_start < agent_lidar_end else 0.0,
            'aux_loss_scale': self._get_aux_loss_scale(),
        }
        
        return aux_loss, aux_info

    def ppo_update(self, sample):
        """
        PPO update with hybrid advantage (MAPPO-Lagrangian style).
        
        Key: adv_hybrid = adv - λ * cost_adv
        """
        share_obs_batch, obs_batch, rnn_states_batch, rnn_states_critic_batch, actions_batch, \
        value_preds_batch, return_batch, masks_batch, active_masks_batch, old_action_log_probs_batch, \
        adv_targ, available_actions_batch, factor_batch, cost_preds_batch, cost_returns_batch, \
        rnn_states_cost_batch, cost_adv_targ, aver_episode_costs = sample

        old_action_log_probs_batch, adv_targ, value_preds_batch, return_batch, active_masks_batch, \
        factor_batch, cost_returns_batch, cost_preds_batch, cost_adv_targ = [
            check(x).to(**self.tpdv) for x in [
                old_action_log_probs_batch, adv_targ, value_preds_batch, return_batch, active_masks_batch,
                factor_batch, cost_returns_batch, cost_preds_batch, cost_adv_targ
            ]
        ]

        # Evaluate actions with cost values
        values, action_log_probs, dist_entropy, cost_values = self.policy.evaluate_actions(
            share_obs_batch, obs_batch, rnn_states_batch, rnn_states_critic_batch,
            actions_batch, masks_batch, available_actions_batch, active_masks_batch,
            rnn_states_cost_batch
        )
        
        # Hybrid advantage: reward_adv - λ * cost_adv
        adv_targ_hybrid = adv_targ - self.lamda_lagr * cost_adv_targ

        imp_weights = torch.exp(action_log_probs - old_action_log_probs_batch)
        imp_weights = torch.prod(imp_weights, dim=-1, keepdim=True)

        surr1 = imp_weights * adv_targ_hybrid
        surr2 = torch.clamp(
            imp_weights, 
            1.0 - self.config["clip_param"], 
            1.0 + self.config["clip_param"]
        ) * adv_targ_hybrid

        if self.config["use_policy_active_masks"]:
            policy_action_loss = (
                -torch.sum(factor_batch * torch.min(surr1, surr2), dim=-1, keepdim=True) 
                * active_masks_batch
            ).sum() / active_masks_batch.sum()
        else:
            policy_action_loss = -torch.sum(
                factor_batch * torch.min(surr1, surr2), dim=-1, keepdim=True
            ).mean()

        policy_loss = policy_action_loss
        
        # Compute auxiliary physics loss
        aux_loss, aux_info = self.compute_auxiliary_physics_loss(
            check(obs_batch).to(**self.tpdv),
            cost_values=cost_values
        )

        # Actor update
        self.policy.actor_optimizer.zero_grad()
        (policy_loss - dist_entropy * self.config["entropy_coef"] + aux_loss).backward()
        actor_grad_norm = nn.utils.clip_grad_norm_(
            self.policy.actor.parameters(), 
            self.config["max_grad_norm"]
        )
        self.policy.actor_optimizer.step()

        # Update Lagrangian multiplier (smoothed dual ascent)
        cost_limit = self.config.get("cost_limit", 25.0)
        current_cost = aver_episode_costs.mean().item()
        if self._ema_cost is None:
            self._ema_cost = current_cost
        else:
            self._ema_cost = (
                self.lagrangian_ema_alpha * self._ema_cost
                + (1.0 - self.lagrangian_ema_alpha) * current_cost
            )

        cost_violation = self._ema_cost - cost_limit
        if self._training_step % self.lagrangian_update_interval == 0:
            delta_lamda = self.lagrangian_slow_rate * cost_violation
            self.lamda_lagr = float(np.clip(
                self.lamda_lagr + delta_lamda,
                self.lamda_lagr_min,
                self.lamda_lagr_max
            ))
        
        aux_info['lamda_lagr'] = self.lamda_lagr
        aux_info['cost_violation'] = cost_violation

        # Reward critic update
        value_loss = self.cal_value_loss(values, value_preds_batch, return_batch, active_masks_batch)
        self.policy.critic_optimizer.zero_grad()
        (value_loss * self.config["value_loss_coef"]).backward()
        critic_grad_norm = nn.utils.clip_grad_norm_(
            self.policy.critic.parameters(), 
            self.config["max_grad_norm"]
        )
        self.policy.critic_optimizer.step()

        # Cost critic update
        cost_loss = self.cal_value_loss(cost_values, cost_preds_batch, cost_returns_batch, active_masks_batch)
        self.policy.cost_optimizer.zero_grad()
        (cost_loss * self.config["value_loss_coef"]).backward()
        cost_grad_norm = nn.utils.clip_grad_norm_(
            self.policy.cost_critic.parameters(), 
            self.config["max_grad_norm"]
        )
        self.policy.cost_optimizer.step()
        
        aux_info['cost_loss'] = cost_loss.item()
        
        # Update training step for barrier warmup
        self._training_step += 1
        self.policy.actor.set_training_step(self._training_step)

        return value_loss, critic_grad_norm, policy_loss, dist_entropy, actor_grad_norm, imp_weights, aux_info

    def train(self, buffer, logger):
        """
        Train policy with cost advantage (MAPPO-Lagrangian style).
        """
        # Compute reward advantages
        advantages = buffer.returns[:-1] - self.value_normalizer.denormalize(buffer.value_preds[:-1])
        advantages_copy = advantages.clone()
        advantages_copy[buffer.active_masks[:-1] == 0.0] = float('nan')
        mean_advantages = torch.nanmean(advantages_copy)
        std_advantages = torch.std(advantages_copy[~torch.isnan(advantages_copy)])
        advantages = (advantages - mean_advantages) / (std_advantages + 1e-8)

        # Compute cost advantages
        cost_adv = buffer.cost_returns[:-1] - self.cost_value_normalizer.denormalize(buffer.cost_preds[:-1])
        cost_adv_copy = cost_adv.clone()
        cost_adv_copy[buffer.active_masks[:-1] == 0.0] = float('nan')
        mean_cost_adv = torch.nanmean(cost_adv_copy)
        std_cost_adv = torch.std(cost_adv_copy[~torch.isnan(cost_adv_copy)])
        cost_adv = (cost_adv - mean_cost_adv) / (std_cost_adv + 1e-8)

        for _ in range(self.config["learning_iters"]):
            data_generator = buffer.feed_forward_generator(advantages, self.config["num_mini_batch"], cost_adv=cost_adv)

            for sample in data_generator:
                value_loss, critic_grad_norm, policy_loss, dist_entropy, actor_grad_norm, imp_weights, aux_info \
                    = self.ppo_update(sample)
        
        # Log metrics
        lagrangian_info = {
            "Loss/Loss_reward_critic": value_loss.item(),
            "Loss/Loss_actor": policy_loss.item(),
            "Loss/Loss_cost_critic": aux_info.get('cost_loss', 0.0),
            "Loss/Aux_task_potential": aux_info['aux_task_loss'],
            "Loss/Aux_barrier_k": aux_info['aux_barrier_k_loss'],
            "Loss/Aux_safety": aux_info['aux_safety_loss'],
            "Loss/Aux_agent_collision": aux_info.get('aux_agent_collision_loss', 0.0),
            "Loss/Aux_cost_value": aux_info.get('aux_cost_value_loss', 0.0),
            "Loss/Aux_cost_k": aux_info.get('aux_cost_k_loss', 0.0),
            "Safe/H_task_mean": aux_info['H_task_mean'],
            "Safe/H_task_std": aux_info['H_task_std'],
            "Safe/H_barrier_mean": aux_info['H_barrier_mean'],
            "Safe/k_mean": aux_info['k_mean'],
            "Safe/k_std": aux_info['k_std'],
            "Safe/hazard_proximity": aux_info['hazard_proximity_mean'],
            "Safe/agent_proximity": aux_info.get('agent_proximity_mean', 0.0),
            "Safe/lamda_lagr": aux_info.get('lamda_lagr', 0.0),
            "Safe/cost_violation": aux_info.get('cost_violation', 0.0),
            "Safe/barrier_weight": self.policy.actor._get_current_barrier_weight(),
            "Safe/training_step": self._training_step,
            "Safe/aux_loss_scale": aux_info.get('aux_loss_scale', 0.0),
            "Safe/soft_cost_weight": self._get_soft_cost_weight(),
            "Misc/Reward_critic_norm": critic_grad_norm.item(),
            "Misc/Entropy": dist_entropy.item(),
            "Misc/Ratio": imp_weights.detach().mean().item(),
        }
        
        logger.store(**lagrangian_info)

    def prep_training(self):
        self.policy.actor.train()
        self.policy.critic.train()
        self.policy.cost_critic.train()

    def prep_rollout(self):
        self.policy.actor.eval()
        self.policy.critic.eval()
        self.policy.cost_critic.eval()


# =============================================================================
# Runner for PHS-MAPPO v2
# =============================================================================

class Runner:

    def __init__(self, vec_env, vec_eval_env, config, model_dir=""):
        self.envs = vec_env
        self.eval_envs = vec_eval_env
        self.config = config
        self.model_dir = model_dir

        self.num_agents = self.envs.num_agents

        self.eval_count = 0
        self.video_record_freq = config.get("video_record_freq", 1)

        torch.autograd.set_detect_anomaly(False)
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True

        setup_headless_rendering()
        
        self.logger = EpochLogger(
            log_dir=config["log_dir"],
            seed=str(config["seed"]),
            use_wandb=config.get("use_wandb", True),
            wandb_project=config.get("wandb_project", "safepo"),
            wandb_config=config,
            verbose=False,
        )
        self.save_dir = str(config["log_dir"] + '/models_seed{}'.format(self.config["seed"]))
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        self.logger.save_config(config)
        
        self.video_recorder = MultiAgentVideoRecorder(
            fps=30,
            enabled=config.get("record_video", True),
            record_freq=config.get("video_record_freq", 10),
            max_episode_length=config.get("episode_length", 1000)
        )
        
        # Create policies with PHS-MAPPO Actor
        self.policy = []
        for agent_id in range(self.num_agents):
            share_observation_space = self.envs.share_observation_space[agent_id]
            po = MAPPOSafePINNv2Policy(
                config,
                self.envs.observation_space[agent_id],
                share_observation_space,
                self.envs.action_space[agent_id],
                n_agents=self.num_agents,
                agent_id=agent_id  # Each agent knows its own ID
            )
            self.policy.append(po)

        if self.model_dir != "":
            self.restore()

        self.trainer = []
        self.buffer = []
        for agent_id in range(self.num_agents):
            tr = MAPPOSafePINNv2Trainer(config, self.policy[agent_id])
            share_observation_space = self.envs.share_observation_space[agent_id]

            bu = SeparatedReplayBuffer(
                config,
                self.envs.observation_space[agent_id],
                share_observation_space,
                self.envs.action_space[agent_id]
            )
            self.buffer.append(bu)
            self.trainer.append(tr)

    def run(self):
        self.warmup()

        start = time.time()
        episodes = int(self.config["num_env_steps"]) // self.config["episode_length"] // self.config["n_rollout_threads"]

        train_episode_rewards = torch.zeros(1, self.config["n_rollout_threads"], device=self.config["device"])
        train_episode_costs = torch.zeros(1, self.config["n_rollout_threads"], device=self.config["device"])
        eval_rewards = 0.0
        eval_costs = 0.0
        
        pbar = tqdm(range(episodes), desc="PHS-MAPPO v2 Training", ncols=100)
        for episode in pbar:

            done_episodes_rewards = []
            done_episodes_costs = []

            for step in range(self.config["episode_length"]):
                values, actions, action_log_probs, rnn_states, rnn_states_critic, cost_preds, rnn_states_cost = self.collect(step)
                obs, share_obs, rewards, costs, dones, infos, _ = self.envs.step(actions)

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

                data = obs, share_obs, rewards, costs, dones, infos, \
                       values, actions, action_log_probs, \
                       rnn_states, rnn_states_critic, cost_preds, rnn_states_cost

                self.insert(data)
            
            self.compute()
            self.train()

            total_num_steps = (episode + 1) * self.config["episode_length"] * self.config["n_rollout_threads"]

            if episode % self.config["save_interval"] == 0 or episode == episodes - 1:
                self.save()
                
            end = time.time()
            
            if (episode % self.config["eval_interval"] == 0 or episode == episodes - 1) and self.config["use_eval"]:
                eval_rewards, eval_costs = self.eval(eval_episodes=1, total_steps=total_num_steps)

            if len(done_episodes_rewards) != 0:
                aver_episode_rewards = torch.stack(done_episodes_rewards).mean()
                aver_episode_costs = torch.stack(done_episodes_costs).mean()
                self.return_aver_cost(aver_episode_costs)
                
                barrier_info = self.collect_barrier_physics_info(obs)
                
                log_dict = {
                    "Metrics/EpRet": aver_episode_rewards.item(),
                    "Metrics/EpCost": aver_episode_costs.item(),
                    "Eval/EpRet": eval_rewards,
                    "Eval/EpCost": eval_costs,
                }
                
                log_dict["Safe/lamda_lagr"] = self.trainer[0].lamda_lagr
                log_dict.update(barrier_info)
                
                self.logger.store(**log_dict)
                
                self.logger.log_tabular("Metrics/EpRet", min_and_max=True, std=True)
                self.logger.log_tabular("Metrics/EpCost", min_and_max=True, std=True)
                self.logger.log_tabular("Eval/EpRet")
                self.logger.log_tabular("Eval/EpCost")
                self.logger.log_tabular("Train/Epoch", episode)
                self.logger.log_tabular("Train/TotalSteps", total_num_steps)
                self.logger.log_tabular("Loss/Loss_reward_critic")
                self.logger.log_tabular("Loss/Loss_cost_critic")
                self.logger.log_tabular("Loss/Loss_actor")
                self.logger.log_tabular("Misc/Reward_critic_norm")
                self.logger.log_tabular("Misc/Entropy")
                self.logger.log_tabular("Misc/Ratio")
                
                self.logger.log_tabular("Loss/Aux_task_potential")
                self.logger.log_tabular("Loss/Aux_barrier_k")
                self.logger.log_tabular("Loss/Aux_safety")
                self.logger.log_tabular("Loss/Aux_agent_collision")
                self.logger.log_tabular("Safe/H_task_mean")
                self.logger.log_tabular("Safe/H_task_std")
                self.logger.log_tabular("Safe/H_barrier_mean")
                self.logger.log_tabular("Safe/k_mean")
                self.logger.log_tabular("Safe/k_std")
                self.logger.log_tabular("Safe/hazard_proximity")
                self.logger.log_tabular("Safe/agent_proximity")
                self.logger.log_tabular("Safe/lamda_lagr")
                self.logger.log_tabular("Safe/cost_violation")
                self.logger.log_tabular("Safe/barrier_weight")
                self.logger.log_tabular("Safe/training_step")
                
                for physics_key in barrier_info.keys():
                    self.logger.log_tabular(physics_key)
                
                self.logger.log_tabular("Time/Total", end - start)
                self.logger.log_tabular("Time/FPS", int(total_num_steps / (end - start)))
                self.logger.dump_tabular(step=total_num_steps)
                
                pbar.set_postfix({
                    'EpRet': f"{aver_episode_rewards.item():.2f}",
                    'EpCost': f"{aver_episode_costs.item():.2f}",
                    # 'λ': f"{self.trainer[0].lamda_lagr:.2f}",
                    # 'bw': f"{self.trainer[0].policy.actor._get_current_barrier_weight():.3f}",
                })
        pbar.close()

    def return_aver_cost(self, aver_episode_costs):
        for agent_id in range(self.num_agents):
            self.buffer[agent_id].return_aver_insert(aver_episode_costs)

    def warmup(self):
        obs, share_obs, _ = self.envs.reset()

        for agent_id in range(self.num_agents):
            self.buffer[agent_id].share_obs[0].copy_(share_obs[:, agent_id])
            if 'Frank' in self.config['env_name']:
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
        cost_preds_collector = []
        rnn_states_cost_collector = []
        
        for agent_id in range(self.num_agents):
            self.trainer[agent_id].prep_rollout()
            value, action, action_log_prob, rnn_state, rnn_state_critic, cost_pred, rnn_state_cost \
                = self.trainer[agent_id].policy.get_actions(
                    self.buffer[agent_id].share_obs[step],
                    self.buffer[agent_id].obs[step],
                    self.buffer[agent_id].rnn_states[step],
                    self.buffer[agent_id].rnn_states_critic[step],
                    self.buffer[agent_id].masks[step],
                    rnn_states_cost=self.buffer[agent_id].rnn_states_cost[step]
                )
            value_collector.append(value.detach())
            action_collector.append(action.detach())
            action_log_prob_collector.append(action_log_prob.detach())
            rnn_state_collector.append(rnn_state.detach())
            rnn_state_critic_collector.append(rnn_state_critic.detach())
            cost_preds_collector.append(cost_pred.detach())
            rnn_states_cost_collector.append(rnn_state_cost.detach())
            
        values = torch.transpose(torch.stack(value_collector), 1, 0)
        rnn_states = torch.transpose(torch.stack(rnn_state_collector), 1, 0)
        rnn_states_critic = torch.transpose(torch.stack(rnn_state_critic_collector), 1, 0)
        cost_preds = torch.transpose(torch.stack(cost_preds_collector), 1, 0)
        rnn_states_cost = torch.transpose(torch.stack(rnn_states_cost_collector), 1, 0)

        return values, action_collector, action_log_prob_collector, rnn_states, rnn_states_critic, cost_preds, rnn_states_cost

    @torch.no_grad()
    def collect_barrier_physics_info(self, obs):
        """Collect physics information for logging."""
        physics_info = {}
        
        for agent_id in range(self.num_agents):
            try:
                if 'Frank' in self.config['env_name']:
                    obs_to_analyze = obs[agent_id]
                else:
                    obs_to_analyze = obs[:, agent_id]
                
                obs_to_analyze = obs_to_analyze.to(self.config["device"])
                info = self.policy[agent_id].actor.get_physics_info(obs_to_analyze)
                
                for key, value in info.items():
                    if isinstance(value, torch.Tensor):
                        avg_val = value.mean().item()
                        physics_key = f"Safe/Agent{agent_id}_{key}"
                        if physics_key not in physics_info:
                            physics_info[physics_key] = []
                        physics_info[physics_key].append(avg_val)
            except Exception:
                pass
        
        averaged_physics = {}
        for key, values in physics_info.items():
            if len(values) > 0:
                averaged_physics[key] = np.mean(values)
        
        return averaged_physics

    def insert(self, data, aver_episode_costs=0):
        """Insert data into buffers with soft cost augmentation."""
        obs, share_obs, rewards, costs, dones, infos, \
        values, actions, action_log_probs, rnn_states, rnn_states_critic, cost_preds, rnn_states_cost = data

        dones_env = torch.all(dones, axis=1)

        rnn_states[dones_env == True] = torch.zeros(
            (dones_env == True).sum(), self.num_agents, self.config["recurrent_N"], 
            self.config["hidden_size"], device=self.config["device"]
        )
        rnn_states_critic[dones_env == True] = torch.zeros(
            (dones_env == True).sum(), self.num_agents, 
            *self.buffer[0].rnn_states_critic.shape[2:], device=self.config["device"]
        )
        rnn_states_cost[dones_env == True] = torch.zeros(
            (dones_env == True).sum(), self.num_agents, 
            *self.buffer[0].rnn_states_cost.shape[2:], device=self.config["device"]
        )

        masks = torch.ones(self.config["n_rollout_threads"], self.num_agents, 1, device=self.config["device"])
        masks[dones_env == True] = torch.zeros(
            (dones_env == True).sum(), self.num_agents, 1, device=self.config["device"]
        )

        active_masks = torch.ones(self.config["n_rollout_threads"], self.num_agents, 1, device=self.config["device"])
        active_masks[dones == True] = torch.zeros((dones == True).sum(), 1, device=self.config["device"])
        active_masks[dones_env == True] = torch.ones(
            (dones_env == True).sum(), self.num_agents, 1, device=self.config["device"]
        )

        soft_cost_weight = self.trainer[0]._get_soft_cost_weight()
        
        for agent_id in range(self.num_agents):
            if 'Frank' in self.config['env_name']:
                obs_to_insert = obs[agent_id]
            else:
                obs_to_insert = obs[:, agent_id]
            
            agent_env_cost = costs[:, agent_id].unsqueeze(-1)
            agent_reward = rewards[:, agent_id].unsqueeze(-1)
            
            # Soft cost augmentation
            if soft_cost_weight > 0:
                with torch.no_grad():
                    obs_tensor = obs_to_insert.to(self.config["device"])
                    actor = self.policy[agent_id].actor
                    H_barrier, _ = actor._compute_barrier_potential(obs_tensor)
                    soft_cost = torch.sigmoid((H_barrier - 3.0) / 2.0)
                    augmented_cost = agent_env_cost + soft_cost_weight * soft_cost
            else:
                augmented_cost = agent_env_cost
            
            self.buffer[agent_id].insert(
                share_obs[:, agent_id], obs_to_insert, rnn_states[:, agent_id],
                rnn_states_critic[:, agent_id], actions[agent_id],
                action_log_probs[agent_id],
                values[:, agent_id], agent_reward, masks[:, agent_id], None,
                active_masks[:, agent_id], None,
                costs=augmented_cost,
                cost_preds=cost_preds[:, agent_id],
                rnn_states_cost=rnn_states_cost[:, agent_id]
            )

    def train(self):
        action_dim = 1
        factor = torch.ones(
            self.config["episode_length"], self.config["n_rollout_threads"], 
            action_dim, device=self.config["device"]
        )

        for agent_id in torch.randperm(self.num_agents):
            action_dim = self.buffer[agent_id].actions.shape[-1]

            self.trainer[agent_id].prep_training()
            self.buffer[agent_id].update_factor(factor)
            available_actions = None if self.buffer[agent_id].available_actions is None \
                else self.buffer[agent_id].available_actions[:-1].reshape(
                    -1, *self.buffer[agent_id].available_actions.shape[2:]
                )

            old_actions_logprob, _ = self.trainer[agent_id].policy.actor.evaluate_actions(
                self.buffer[agent_id].obs[:-1].reshape(-1, *self.buffer[agent_id].obs.shape[2:]),
                self.buffer[agent_id].rnn_states[0:1].reshape(-1, *self.buffer[agent_id].rnn_states.shape[2:]),
                self.buffer[agent_id].actions.reshape(-1, *self.buffer[agent_id].actions.shape[2:]),
                self.buffer[agent_id].masks[:-1].reshape(-1, *self.buffer[agent_id].masks.shape[2:]),
                available_actions,
                self.buffer[agent_id].active_masks[:-1].reshape(-1, *self.buffer[agent_id].active_masks.shape[2:])
            )
            
            self.trainer[agent_id].train(self.buffer[agent_id], logger=self.logger)

            new_actions_logprob, _ = self.trainer[agent_id].policy.actor.evaluate_actions(
                self.buffer[agent_id].obs[:-1].reshape(-1, *self.buffer[agent_id].obs.shape[2:]),
                self.buffer[agent_id].rnn_states[0:1].reshape(-1, *self.buffer[agent_id].rnn_states.shape[2:]),
                self.buffer[agent_id].actions.reshape(-1, *self.buffer[agent_id].actions.shape[2:]),
                self.buffer[agent_id].masks[:-1].reshape(-1, *self.buffer[agent_id].masks.shape[2:]),
                available_actions,
                self.buffer[agent_id].active_masks[:-1].reshape(-1, *self.buffer[agent_id].active_masks.shape[2:])
            )

            action_prod = torch.prod(
                torch.exp(new_actions_logprob.detach() - old_actions_logprob.detach()).reshape(
                    self.config["episode_length"], self.config["n_rollout_threads"], action_dim
                ), 
                dim=-1, keepdim=True
            )
            factor = factor * action_prod.detach()
            self.buffer[agent_id].after_update()

    def save(self):
        for agent_id in range(self.num_agents):
            policy_actor = self.trainer[agent_id].policy.actor
            torch.save(
                policy_actor.state_dict(), 
                str(self.save_dir) + "/actor_agent" + str(agent_id) + ".pt"
            )
            policy_critic = self.trainer[agent_id].policy.critic
            torch.save(
                policy_critic.state_dict(), 
                str(self.save_dir) + "/critic_agent" + str(agent_id) + ".pt"
            )
            policy_cost_critic = self.trainer[agent_id].policy.cost_critic
            torch.save(
                policy_cost_critic.state_dict(),
                str(self.save_dir) + "/cost_critic_agent" + str(agent_id) + ".pt"
            )

    def restore(self):
        for agent_id in range(self.num_agents):
            policy_actor_state_dict = torch.load(
                str(self.model_dir) + '/actor_agent' + str(agent_id) + '.pt'
            )
            self.policy[agent_id].actor.load_state_dict(policy_actor_state_dict)
            policy_critic_state_dict = torch.load(
                str(self.model_dir) + '/critic_agent' + str(agent_id) + '.pt'
            )
            self.policy[agent_id].critic.load_state_dict(policy_critic_state_dict)
            
            cost_critic_path = str(self.model_dir) + '/cost_critic_agent' + str(agent_id) + '.pt'
            if os.path.exists(cost_critic_path):
                policy_cost_critic_state_dict = torch.load(cost_critic_path)
                self.policy[agent_id].cost_critic.load_state_dict(policy_cost_critic_state_dict)

    def _extract_agent_positions(self):
        """Extract current agent positions from environment."""
        agent_positions = []
        
        if not hasattr(self.eval_envs, 'envs') or len(self.eval_envs.envs) == 0:
            return agent_positions
        
        env = self.eval_envs.envs[0]
        task = getattr(getattr(env, 'env', env), 'task', None)
        
        if task and hasattr(task, 'agent'):
            agent = task.agent
            if hasattr(agent, 'pos_0'):
                agent_positions.append((float(agent.pos_0[0]), float(agent.pos_0[1])))
            if hasattr(agent, 'pos_1'):
                agent_positions.append((float(agent.pos_1[0]), float(agent.pos_1[1])))
        
        return agent_positions
    
    def _extract_env_obstacles(self):
        """Extract obstacle and goal positions from environment."""
        obstacle_positions = []
        goal_positions = []
        hazard_radius = 0.25
        
        if not hasattr(self.eval_envs, 'envs') or len(self.eval_envs.envs) == 0:
            return obstacle_positions, goal_positions, hazard_radius
        
        env = self.eval_envs.envs[0]
        task = getattr(getattr(env, 'env', env), 'task', None)
        
        if task:
            # Get hazards from task.hazards.pos (real-time positions)
            if hasattr(task, 'hazards') and hasattr(task.hazards, 'pos'):
                hazards_pos = task.hazards.pos
                if hazards_pos is not None:
                    for pos in hazards_pos:
                        obstacle_positions.append((float(pos[0]), float(pos[1])))
                if hasattr(task.hazards, 'size'):
                    hazard_radius = float(task.hazards.size)
            
            # Get goals from task.goal_red/blue.pos
            if hasattr(task, 'goal_red') and hasattr(task.goal_red, 'pos'):
                goal_red_pos = task.goal_red.pos
                if goal_red_pos is not None:
                    goal_positions.append((float(goal_red_pos[0]), float(goal_red_pos[1])))
            
            if hasattr(task, 'goal_blue') and hasattr(task.goal_blue, 'pos'):
                goal_blue_pos = task.goal_blue.pos
                if goal_blue_pos is not None:
                    goal_positions.append((float(goal_blue_pos[0]), float(goal_blue_pos[1])))
        
        # Fallback to defaults if extraction failed
        if len(obstacle_positions) == 0:
            obstacle_positions = [
                (0.8, 0.0), (-0.8, 0.0), (0.0, 0.8), (0.0, -0.8),
                (0.6, 0.6), (-0.6, 0.6), (0.6, -0.6), (-0.6, -0.6)
            ]
        
        if len(goal_positions) == 0:
            goal_positions = [(1.2, 1.2), (-1.2, -1.2)]
        
        return obstacle_positions, goal_positions, hazard_radius

    @torch.no_grad()
    def eval(self, eval_episodes=1, total_steps=0):
        """Evaluate policy performance."""
        self.eval_count += 1
        should_record_video = (
            self.video_recorder.enabled 
            and self.eval_count % self.video_record_freq == 0
            and self.config["env_name"] not in isaac_gym_map
        )
        
        is_multi_goal_task = self.config["env_name"] in multi_agent_goal_tasks
        
        eval_episode = 0
        eval_episode_rewards = []
        eval_episode_costs = []
        one_episode_rewards = torch.zeros(1, self.config["n_eval_rollout_threads"], device=self.config["device"])
        one_episode_costs = torch.zeros(1, self.config["n_eval_rollout_threads"], device=self.config["device"])
        
        first_episode_frames = []
        potential_field_frames = []
        first_episode_reward = 0.0
        first_episode_cost = 0.0
        recording_first_episode = should_record_video
        step_count = 0
        frame_sample_rate = 3

        potential_visualizer = None
        
        if should_record_video and is_multi_goal_task:
            _, _, hazard_radius = self._extract_env_obstacles()
            potential_visualizer = BarrierPotentialVideoVisualizer(
                actor=self.policy[0].actor,
                world_bounds=(-2.5, 2.5, -2.5, 2.5),
                grid_resolution=30,
                device='cpu',
                hazard_radius=hazard_radius,
            )

        eval_obs, _, _ = self.eval_envs.reset()

        eval_rnn_states = torch.zeros(
            self.config["n_eval_rollout_threads"], self.num_agents, 
            self.config["recurrent_N"], self.config["hidden_size"],
            device=self.config["device"]
        )
        eval_masks = torch.ones(
            self.config["n_eval_rollout_threads"], self.num_agents, 1, 
            device=self.config["device"]
        )

        while True:
            eval_actions_collector = []
            for agent_id in range(self.num_agents):
                self.trainer[agent_id].prep_rollout()
                obs_to_eval = eval_obs[:, agent_id]
                eval_actions, temp_rnn_state = self.trainer[agent_id].policy.act(
                    obs_to_eval,
                    eval_rnn_states[:, agent_id],
                    eval_masks[:, agent_id],
                    deterministic=True
                )
                
                eval_rnn_states[:, agent_id] = temp_rnn_state
                eval_actions_collector.append(eval_actions)

            # Capture frame
            if recording_first_episode and hasattr(self.eval_envs, 'render'):
                frame = self.eval_envs.render()
                if frame is not None and isinstance(frame, np.ndarray) and len(frame.shape) == 3:
                    first_episode_frames.append(frame.copy())
                    
                    if potential_visualizer is not None and step_count % frame_sample_rate == 0:
                        # Extract current positions
                        obstacle_positions, goal_positions, _ = self._extract_env_obstacles()
                        agent_positions = self._extract_agent_positions()
                        
                        combined_frame, _, _, _ = potential_visualizer.render_all_potentials_frame(
                            env_frame=frame,
                            obstacle_positions=np.array(obstacle_positions),
                            goal_positions=np.array(goal_positions),
                            agent_positions=np.array(agent_positions) if agent_positions else None,
                            step=step_count,
                            task_potential_scale=2.0,
                        )
                        potential_field_frames.append(combined_frame)
                    
                    step_count += 1

            eval_obs, _, eval_rewards, eval_costs, eval_dones, _, _ = self.eval_envs.step(
                eval_actions_collector
            )

            reward_env = torch.mean(eval_rewards, dim=1).flatten()
            cost_env = torch.mean(eval_costs, dim=1).flatten()

            one_episode_rewards += reward_env
            one_episode_costs += cost_env

            eval_dones_env = torch.all(eval_dones, dim=1)

            eval_rnn_states[eval_dones_env == True] = torch.zeros(
                (eval_dones_env == True).sum(), self.num_agents, 
                self.config["recurrent_N"], self.config["hidden_size"], 
                device=self.config["device"]
            )

            eval_masks = torch.ones(
                self.config["n_eval_rollout_threads"], self.num_agents, 1, 
                device=self.config["device"]
            )
            eval_masks[eval_dones_env == True] = torch.zeros(
                (eval_dones_env == True).sum(), self.num_agents, 1,
                device=self.config["device"]
            )

            for eval_i in range(self.config["n_eval_rollout_threads"]):
                if eval_dones_env[eval_i]:
                    eval_episode += 1
                    ep_reward = one_episode_rewards[:, eval_i].mean().item()
                    ep_cost = one_episode_costs[:, eval_i].mean().item()
                    eval_episode_rewards.append(ep_reward)
                    eval_episode_costs.append(ep_cost)

                    if recording_first_episode and eval_episode == 1:
                        first_episode_reward = ep_reward
                        first_episode_cost = ep_cost
                        recording_first_episode = False

                    one_episode_rewards[:, eval_i] = 0
                    one_episode_costs[:, eval_i] = 0

            if eval_episode >= eval_episodes:
                # Save potential field video
                if len(potential_field_frames) > 0 and should_record_video and is_multi_goal_task:
                    viz_dir = os.path.join(os.path.dirname(self.save_dir), "vizs")
                    os.makedirs(viz_dir, exist_ok=True)
                    
                    video_path = os.path.join(viz_dir, f"potential_field_step{total_steps}.mp4")
                    if potential_visualizer is not None:
                        potential_visualizer.save_video(potential_field_frames, video_path, fps=30)
                    
                    if self.logger.use_wandb:
                        import wandb
                        video_array = np.ascontiguousarray(
                            np.transpose(np.stack(potential_field_frames, axis=0), (0, 3, 1, 2))
                        )
                        fps_adjusted = 30 // frame_sample_rate
                        caption = f"PHS-MAPPO v2 - Eval #{self.eval_count} - R:{first_episode_reward:.1f} C:{first_episode_cost:.1f}"
                        video_obj = wandb.Video(video_array, fps=fps_adjusted, format="mp4", caption=caption)
                        
                        if hasattr(self.logger, 'wandb_run') and self.logger.wandb_run is not None:
                            self.logger.wandb_run.log({"Viz/all_potentials_video": video_obj}, step=total_steps)
                        elif wandb.run is not None:
                            wandb.log({"Viz/all_potentials_video": video_obj}, step=total_steps)
                
                # Cleanup
                import matplotlib.pyplot as plt
                plt.close('all')
                import gc
                gc.collect()
                
                return np.mean(eval_episode_rewards), np.mean(eval_episode_costs)

    @torch.no_grad()
    def compute(self):
        """Compute returns for both reward and cost critics."""
        for agent_id in range(self.num_agents):
            self.trainer[agent_id].prep_rollout()
            
            # Reward returns
            next_value = self.trainer[agent_id].policy.get_values(
                self.buffer[agent_id].share_obs[-1],
                self.buffer[agent_id].rnn_states_critic[-1],
                self.buffer[agent_id].masks[-1]
            )
            next_value = next_value.detach()
            self.buffer[agent_id].compute_returns(next_value, self.trainer[agent_id].value_normalizer)
            
            # Cost returns
            next_costs = self.trainer[agent_id].policy.get_cost_values(
                self.buffer[agent_id].share_obs[-1],
                self.buffer[agent_id].rnn_states_cost[-1],
                self.buffer[agent_id].masks[-1]
            )
            next_costs = next_costs.detach()
            self.buffer[agent_id].compute_cost_returns(next_costs, self.trainer[agent_id].cost_value_normalizer)


def train(args, cfg_train):
    agent_index = [[[0, 1, 2, 3, 4, 5]], [[0, 1, 2, 3, 4, 5]]]
    
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
        cfg_eval["render_mode"] = "rgb_array"
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
    args, cfg_env, cfg_train = multi_agent_args(algo="mappo_safe_pinn_v2")
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
            os.path.join(f"{cfg_train['log_dir']}", terminal_log_name),
            "w", encoding="utf-8",
        ) as f_out:
            sys.stdout = f_out
            with open(
                os.path.join(f"{cfg_train['log_dir']}", error_log_name),
                "w", encoding="utf-8",
            ) as f_error:
                sys.stderr = f_error
                train(args=args, cfg_train=cfg_train)

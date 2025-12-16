# Copyright 2023 OmniSafeAI Team. All Rights Reserved.
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
MAPPO-PINN: MAPPO with Port-Hamiltonian Neural Network Actor.

This algorithm integrates Physics-Informed Neural Networks (PINN) based on 
Port-Hamiltonian systems into MAPPO's actor network.

Port-Hamiltonian System:
    ẋ = (J(x) - R(x)) ∇H(x) + g(x)u
    
Where:
    - x = (q, p) is the state (position, momentum/velocity)
    - J(x): Skew-symmetric interconnection matrix (energy-conserving)
    - R(x): Symmetric positive semi-definite dissipation matrix (energy-dissipating)
    - H(x): Hamiltonian (total energy)
    - g(x): Input matrix
    - u: Control input (action)

For Point agents in SafetyMultiGoal environments:
    - Observation: 152-dim (accelerometer, velocimeter, gyro, magnetometer, lidars)
    - velocimeter (indices 3-5): velocity (vx, vy, vz)
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

from safepo.common.env import make_ma_mujoco_env, make_ma_isaac_env, make_ma_multi_goal_env
from safepo.common.popart import PopArt
from safepo.common.model import MultiAgentCritic as Critic
from safepo.common.buffer import SeparatedReplayBuffer
from safepo.common.logger import EpochLogger
from safepo.common.video_recorder import MultiAgentVideoRecorder, setup_headless_rendering
from safepo.utils.config import multi_agent_args, parse_sim_params, set_np_formatting, set_seed, multi_agent_velocity_map, isaac_gym_map, multi_agent_goal_tasks


def check(input):
    output = torch.from_numpy(input) if type(input) == np.ndarray else input
    return output

def huber_loss(e, d):
    a = (abs(e) <= d).float()
    b = (e > d).float()
    return a*e**2/2 + b*d*(abs(e)-d/2)


# =============================================================================
# Port-Hamiltonian PINN Actor for Point Agents
# =============================================================================

class PortHamiltonianPINNActor(nn.Module):
    """
    Physics-Informed Neural Network Actor based on Port-Hamiltonian Systems.
    
    Port-Hamiltonian dynamics:
        ẋ = (J - R) ∇H(x) + g(x)u
        
    For Point agent:
        - State x includes velocity from observation
        - J is skew-symmetric (energy conserving interconnection)
        - R is symmetric positive semi-definite (dissipation)
        - H is the Hamiltonian (energy function)
        - Control u is the action output
        
    The network learns:
        1. H_net: Hamiltonian function H(obs) -> scalar energy
        2. J_net: Skew-symmetric interconnection J(obs) 
        3. R_net: Dissipation matrix R(obs), enforced to be positive semi-definite
        4. Policy_net: Maps physics-informed features to action distribution
    """
    
    def __init__(self, config, obs_space, act_space, device=torch.device("cpu")):
        super(PortHamiltonianPINNActor, self).__init__()
        
        self.config = config
        self.device = device
        self.obs_dim = obs_space.shape[0]  # 152 for Point MultiGoal
        self.act_dim = act_space.shape[0]  # 2 for Point
        
        # Configuration
        self.hidden_size = config.get("hidden_size", 256)
        self.physics_hidden = config.get("physics_hidden", 64)
        self.state_dim = config.get("pinn_state_dim", 4)  # (vx, vy, ax, ay) or similar
        self.std_x_coef = config.get("std_x_coef", 1.0)
        self.std_y_coef = config.get("std_y_coef", 0.5)
        
        # Physics state extraction indices (for Point agent observation)
        # obs[0:3] = accelerometer (ax, ay, az)
        # obs[3:6] = velocimeter (vx, vy, vz)  <- velocity!
        # obs[6:9] = gyro (angular velocity)
        # obs[9:12] = magnetometer (orientation)
        self.vel_indices = [3, 4]  # vx, vy
        self.acc_indices = [0, 1]  # ax, ay
        
        # ===================
        # Physics Networks (Port-Hamiltonian)
        # ===================
        
        # Hamiltonian Network: H(x) -> scalar energy
        # For Point agent: H = 0.5 * m * ||v||^2 (kinetic) + potential terms
        self.H_net = nn.Sequential(
            nn.Linear(self.state_dim, self.physics_hidden),
            nn.ELU(),
            nn.Linear(self.physics_hidden, self.physics_hidden),
            nn.ELU(),
            nn.Linear(self.physics_hidden, 1)  # Scalar Hamiltonian
        )
        
        # J Network: Learns skew-symmetric interconnection matrix
        # Output: upper triangular elements, then construct skew-symmetric J
        self.J_dim = self.state_dim * (self.state_dim - 1) // 2  # Upper triangular elements
        self.J_net = nn.Sequential(
            nn.Linear(self.state_dim, self.physics_hidden),
            nn.ELU(),
            nn.Linear(self.physics_hidden, self.J_dim)
        )
        
        # R Network: Learns dissipation matrix (positive semi-definite)
        # Output: elements of lower triangular L, then R = L @ L^T
        self.R_tril_dim = self.state_dim * (self.state_dim + 1) // 2
        self.R_net = nn.Sequential(
            nn.Linear(self.state_dim, self.physics_hidden),
            nn.ELU(),
            nn.Linear(self.physics_hidden, self.R_tril_dim)
        )
        
        # ===================
        # Feature Extraction (Standard MLP like MAPPO)
        # ===================
        
        # Feature normalization (like MAPPO)
        self.feature_norm = nn.LayerNorm(self.obs_dim)
        
        # Base network for non-physics features
        self.base_net = nn.Sequential(
            nn.Linear(self.obs_dim, self.hidden_size),
            nn.ELU(),
            nn.LayerNorm(self.hidden_size),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ELU(),
            nn.LayerNorm(self.hidden_size),
        )
        
        # ===================
        # Physics-Policy Integration
        # ===================
        
        # Combine physics features with learned features
        # Physics features: H, grad_H (state_dim), J-R matrix info (flattened)
        physics_feature_dim = 1 + self.state_dim  # H + grad_H
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
        
        # Physics networks with smaller gain for stability
        for net in [self.H_net, self.J_net, self.R_net]:
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
            state: [batch, state_dim] physics state
        """
        # Extract velocity and acceleration
        vel = obs[:, self.vel_indices]  # [batch, 2] - vx, vy
        acc = obs[:, self.acc_indices]  # [batch, 2] - ax, ay
        
        # Combine into physics state
        state = torch.cat([vel, acc], dim=-1)  # [batch, 4]
        return state
    
    def _construct_J_matrix(self, J_elements, batch_size):
        """
        Construct skew-symmetric matrix J from upper triangular elements.
        J = -J^T (antisymmetric)
        
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
    
    def _compute_hamiltonian_gradient(self, obs, state):
        """
        Compute Hamiltonian and its gradient w.r.t. state.
        
        Args:
            obs: [batch, obs_dim] observation
            state: [batch, state_dim] physics state
            
        Returns:
            H: [batch, 1] Hamiltonian value
            grad_H: [batch, state_dim] gradient of H w.r.t. state
        """
        # Check if we're in no_grad mode (inference)
        if not torch.is_grad_enabled():
            # In inference mode, compute H and approximate gradient
            with torch.enable_grad():
                state_requires_grad = state.clone().requires_grad_(True)
                H = self.H_net(state_requires_grad)
                
                # Compute gradient of H w.r.t. state
                grad_H = torch.autograd.grad(
                    outputs=H.sum(),
                    inputs=state_requires_grad,
                    create_graph=False,  # Don't need graph in inference
                    retain_graph=False
                )[0]
            # Detach for inference
            return H.detach(), grad_H.detach()
        else:
            # In training mode, keep gradients for backprop
            state_requires_grad = state.clone().requires_grad_(True)
            H = self.H_net(state_requires_grad)
            
            # Compute gradient of H w.r.t. state
            grad_H = torch.autograd.grad(
                outputs=H.sum(),
                inputs=state_requires_grad,
                create_graph=True,
                retain_graph=True
            )[0]
            
            return H, grad_H
    
    def _compute_port_hamiltonian_dynamics(self, obs, state):
        """
        Compute Port-Hamiltonian dynamics: ẋ = (J - R) ∇H
        
        This provides physics-consistent features for the policy.
        
        Args:
            obs: [batch, obs_dim] observation
            state: [batch, state_dim] physics state
            
        Returns:
            H: Hamiltonian value
            grad_H: Gradient of Hamiltonian
            dynamics: (J - R) ∇H term
        """
        batch_size = state.shape[0]
        
        # Get Hamiltonian and gradient
        H, grad_H = self._compute_hamiltonian_gradient(obs, state)
        
        # Get J and R matrices
        J_elements = self.J_net(state)
        R_elements = self.R_net(state)
        
        J = self._construct_J_matrix(J_elements, batch_size)
        R = self._construct_R_matrix(R_elements, batch_size)
        
        # Compute dynamics: (J - R) ∇H
        J_minus_R = J - R
        dynamics = torch.bmm(J_minus_R, grad_H.unsqueeze(-1)).squeeze(-1)
        
        return H, grad_H, dynamics
    
    def forward(self, obs, rnn_states=None, masks=None, available_actions=None, deterministic=False):
        """
        Forward pass: compute action from observation.
        
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
        
        # Compute Port-Hamiltonian features
        H, grad_H, _ = self._compute_port_hamiltonian_dynamics(obs, state)
        
        # Extract base features (standard MLP path)
        obs_normalized = self.feature_norm(obs)
        base_features = self.base_net(obs_normalized)
        
        # Combine physics and learned features
        physics_features = torch.cat([H, grad_H], dim=-1)  # [batch, 1 + state_dim]
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
        
        # Compute Port-Hamiltonian features
        H, grad_H, _ = self._compute_port_hamiltonian_dynamics(obs, state)
        
        # Extract base features
        obs_normalized = self.feature_norm(obs)
        base_features = self.base_net(obs_normalized)
        
        # Combine features
        physics_features = torch.cat([H, grad_H], dim=-1)
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
        Get Port-Hamiltonian physics information for analysis/logging.
        
        Returns:
            dict with H, grad_H, J, R matrices
        """
        obs = check(obs).to(self.device)
        state = self._extract_physics_state(obs)
        batch_size = state.shape[0]
        
        H, grad_H = self._compute_hamiltonian_gradient(obs, state)
        
        J_elements = self.J_net(state)
        R_elements = self.R_net(state)
        
        J = self._construct_J_matrix(J_elements, batch_size)
        R = self._construct_R_matrix(R_elements, batch_size)
        
        return {
            'H': H.detach(),
            'grad_H': grad_H.detach(),
            'J': J.detach(),
            'R': R.detach(),
            'state': state.detach()
        }


# =============================================================================
# MAPPO-PINN Policy with Port-Hamiltonian Actor
# =============================================================================

class MAPPOPINNPointPolicy:
    """MAPPO policy with Port-Hamiltonian PINN Actor for Point agents."""

    def __init__(self, config, obs_space, cent_obs_space, act_space):
        self.config = config
        self.obs_space = obs_space
        self.act_space = act_space
        self.share_obs_space = cent_obs_space

        # Use Port-Hamiltonian PINN Actor
        self.actor = PortHamiltonianPINNActor(config, self.obs_space, self.act_space, self.config["device"])
        
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
# MAPPO-PINN Trainer (same as MAPPO)
# =============================================================================

class MAPPOPINNPointTrainer():

    def __init__(self, config, policy):
        
        self.config = config
        self.tpdv = dict(dtype=torch.float32, device=self.config["device"])
        self.policy = policy

        self.value_normalizer = PopArt(1, device=self.config["device"])

    def cal_value_loss(self, values, value_preds_batch, return_batch, active_masks_batch):
        value_pred_clipped = value_preds_batch + (values - value_preds_batch).clamp(-self.config["clip_param"],
                                                                                    self.config["clip_param"])
        error_clipped = self.value_normalizer(return_batch) - value_pred_clipped
        error_original = self.value_normalizer(return_batch) - values

        value_loss_clipped = huber_loss(error_clipped, self.config["huber_delta"])
        value_loss_original = huber_loss(error_original, self.config["huber_delta"])

        value_loss = torch.max(value_loss_original, value_loss_clipped)

        return value_loss.mean()

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

        self.policy.actor_optimizer.zero_grad()
        (policy_loss - dist_entropy * self.config["entropy_coef"]).backward()
        actor_grad_norm = nn.utils.clip_grad_norm_(self.policy.actor.parameters(), self.config["max_grad_norm"])
        self.policy.actor_optimizer.step()

        value_loss = self.cal_value_loss(values, value_preds_batch, return_batch, active_masks_batch)
        self.policy.critic_optimizer.zero_grad()
        (value_loss * self.config["value_loss_coef"]).backward()
        critic_grad_norm = nn.utils.clip_grad_norm_(self.policy.critic.parameters(), self.config["max_grad_norm"])

        self.policy.critic_optimizer.step()

        return value_loss, critic_grad_norm, policy_loss, dist_entropy, actor_grad_norm, imp_weights

    def train(self, buffer, logger):
        advantages = buffer.returns[:-1] - self.value_normalizer.denormalize(buffer.value_preds[:-1])
        advantages_copy = advantages.clone()
        mean_advantages = torch.mean(advantages_copy)
        std_advantages = torch.std(advantages_copy)
        advantages = (advantages - mean_advantages) / (std_advantages + 1e-5)

        for _ in range(self.config["learning_iters"]):
            data_generator = buffer.feed_forward_generator(advantages, self.config["num_mini_batch"])

            for sample in data_generator:
                value_loss, critic_grad_norm, policy_loss, dist_entropy, actor_grad_norm, imp_weights \
                    = self.ppo_update(sample)
            logger.store(
                **{
                    "Loss/Loss_reward_critic": value_loss.item(),
                    "Loss/Loss_actor": policy_loss.item(),
                    "Misc/Reward_critic_norm": critic_grad_norm.item(),
                    "Misc/Entropy": dist_entropy.item(),
                    "Misc/Ratio": imp_weights.detach().mean().item(),
                }
            )

    def prep_training(self):
        self.policy.actor.train()
        self.policy.critic.train()

    def prep_rollout(self):
        self.policy.actor.eval()
        self.policy.critic.eval()


# =============================================================================
# Runner (same as MAPPO, but uses PINN Policy/Trainer)
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

        # Track the best eval reward for conditional video rendering
        self.render_max_reward = float(self.config.get("render_max_reward", float("-inf")))
        self.config["render_max_reward"] = self.render_max_reward

        torch.autograd.set_detect_anomaly(True)
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True

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
        
        # Create PINN policies for each agent
        self.policy = []
        for agent_id in range(self.num_agents):
            share_observation_space = self.envs.share_observation_space[agent_id]
            po = MAPPOPINNPointPolicy(config,
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
            tr = MAPPOPINNPointTrainer(config, self.policy[agent_id])
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
        pbar = tqdm(range(episodes), desc="MAPPO-PINN-Point Training", ncols=100)
        for episode in pbar:

            done_episodes_rewards = []
            done_episodes_costs = []

            for step in range(self.config["episode_length"]):
                # Sample actions
                values, actions, action_log_probs, rnn_states, rnn_states_critic = self.collect(step)
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
            
            if episode % self.config["eval_interval"] == 0 and self.config["use_eval"]:
                eval_rewards, eval_costs = self.eval(eval_episodes=1, total_steps=total_num_steps)

            if len(done_episodes_rewards) != 0:
                aver_episode_rewards = torch.stack(done_episodes_rewards).mean()
                aver_episode_costs = torch.stack(done_episodes_costs).mean()
                self.return_aver_cost(aver_episode_costs)
                self.logger.store(
                    **{
                        "Metrics/EpRet": aver_episode_rewards.item(),
                        "Metrics/EpCost": aver_episode_costs.item(),
                        "Eval/EpRet": eval_rewards,
                        "Eval/EpCost": eval_costs,
                    }
                )
                
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
        eval_episode = 0
        eval_episode_rewards = []
        eval_episode_costs = []
        one_episode_rewards = torch.zeros(1, self.config["n_eval_rollout_threads"], device=self.config["device"])
        one_episode_costs = torch.zeros(1, self.config["n_eval_rollout_threads"], device=self.config["device"])
        
        # Track best episode that beats global max for video recording
        best_episode_frames = []
        best_episode_reward = 0.0
        best_episode_cost = 0.0
        best_episode_num = 0
        current_frames = []

        eval_obs, _, _ = self.eval_envs.reset()

        eval_rnn_states = torch.zeros(self.config["n_eval_rollout_threads"], self.num_agents, self.config["recurrent_N"], self.config["hidden_size"],
                                   device=self.config["device"])
        eval_masks = torch.ones(self.config["n_eval_rollout_threads"], self.num_agents, 1, device=self.config["device"])

        while True:
            eval_actions_collector = []
            for agent_id in range(self.num_agents):
                self.trainer[agent_id].prep_rollout()
                if 'Frank'in self.config['env_name']:
                    obs_to_eval = eval_obs[agent_id]
                else:
                    obs_to_eval = eval_obs[:, agent_id]
                eval_actions, temp_rnn_state = \
                    self.trainer[agent_id].policy.act(obs_to_eval,
                                                      eval_rnn_states[:, agent_id],
                                                      eval_masks[:, agent_id],
                                                      deterministic=True)
                eval_rnn_states[:, agent_id] = temp_rnn_state
                eval_actions_collector.append(eval_actions)

            if self.config["env_name"] == "Safety9|8HumanoidVelocity-v0":
                zeros = torch.zeros(eval_actions_collector[-1].shape[0], 1)
                eval_actions_collector[-1]=torch.cat((eval_actions_collector[-1], zeros), dim=1)
            
            # Capture frame for video (only for non-Isaac Gym envs)
            if self.video_recorder.enabled and self.config["env_name"] not in isaac_gym_map:
                if hasattr(self.eval_envs, 'render'):
                    try:
                        frame = self.eval_envs.render()
                        if frame is not None:
                            if isinstance(frame, np.ndarray) and len(frame.shape) == 3:
                                current_frames.append(frame.copy())
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

                    # Only record episodes that beat global max, and keep the best one among them
                    if ep_reward > self.render_max_reward:
                        if len(best_episode_frames) == 0 or ep_reward > best_episode_reward:
                            best_episode_frames = current_frames.copy()
                            best_episode_reward = ep_reward
                            best_episode_cost = ep_cost
                            best_episode_num = eval_episode
                    
                    # Clear current frames for next episode
                    current_frames = []

                    one_episode_rewards[:, eval_i] = 0
                    one_episode_costs[:, eval_i] = 0

            if eval_episode >= eval_episodes:
                # Upload video for the best episode if any episode beat the global max
                if len(best_episode_frames) > 0:
                    self.render_max_reward = best_episode_reward
                    self.config["render_max_reward"] = self.render_max_reward
                    
                    # Upload the best episode from this eval run
                    if self.video_recorder.enabled:
                        self.video_recorder.recorder.frames = best_episode_frames
                        caption = f"Episode {best_episode_num} - Reward: {best_episode_reward:.2f}, Cost: {best_episode_cost:.2f}"
                        self.video_recorder.recorder.upload_to_wandb(
                            caption=caption,
                            step=total_steps,
                            key="eval/video"
                        )
                
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
        runner.eval(100000)
    else:
        runner.run()


if __name__ == '__main__':
    set_np_formatting()
    args, cfg_env, cfg_train = multi_agent_args(algo="mappo_pinn")
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

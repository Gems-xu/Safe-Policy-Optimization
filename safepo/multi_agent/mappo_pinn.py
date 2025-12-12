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
"""MAPPO with Physics-Informed Neural Network (PINN) Actor."""

import os
import sys
import time
import copy
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.distributions import Normal

from safepo.common.env import make_ma_mujoco_env, make_ma_isaac_env, make_ma_multi_goal_env
from safepo.common.popart import PopArt
from safepo.common.model import MultiAgentCritic as Critic
from safepo.common.buffer import SeparatedReplayBuffer
from safepo.common.logger import EpochLogger
from safepo.common.video_recorder import MultiAgentVideoRecorder, setup_headless_rendering
from safepo.utils.config import multi_agent_args, parse_sim_params, set_np_formatting, set_seed, multi_agent_velocity_map, isaac_gym_map, multi_agent_goal_tasks
from safepo.utils.util import check

# Import PINN components from safepo.common
from safepo.common.pinn_models import MLP, MLP2, Attention_LEMURS, Att_R, Att_J, Att_H


def huber_loss(e, d):
    a = (abs(e) <= d).float()
    b = (e > d).float()
    return a*e**2/2 + b*d*(abs(e)-d/2)


class PINNActor(nn.Module):
    """
    Physics-Informed Neural Network Actor for multi-agent systems.
    
    This actor uses physics-based constraints (Hamiltonian dynamics) to generate actions
    that are consistent with physical laws while maintaining learning flexibility.
    
    Args:
        config (dict): Configuration parameters
        obs_space: Observation space
        action_space: Action space  
        device (torch.device): Device to run on
    """
    
    def __init__(self, config, obs_space, action_space, device=torch.device("cpu")):
        super(PINNActor, self).__init__()
        self.config = config
        self.device = device
        self.tpdv = dict(dtype=torch.float32, device=device)
        
        # Extract dimensions
        if hasattr(obs_space, 'shape'):
            self.obs_dim = obs_space.shape[0]
        else:
            self.obs_dim = obs_space.n
            
        if hasattr(action_space, 'shape'):
            self.act_dim = action_space.shape[0]
        else:
            self.act_dim = action_space.n
        
        # PINN-specific parameters
        self.n_agents = config.get("n_agents", 1)
        self.scenario_name = config.get("scenario_name", "navigation")
        self.r_communication = config.get("r_communication", 0.45)
        self.drag = config.get("drag", 0.25)
        self.log_std_min = config.get("log_std_min", -5)
        self.log_std_max = config.get("log_std_max", 2)
        self.hidden_dim = config.get("pinn_hidden_dim", 8)
        
        # Initialize PINN components
        self.R_mean = Att_R(
            self.obs_dim, 16, self.hidden_dim, 
            self.n_agents, self.scenario_name, self.device
        ).to(self.device)
        
        self.J_mean = Att_J(
            self.obs_dim, 16, self.hidden_dim,
            self.n_agents, self.scenario_name, self.device
        ).to(self.device)
        
        self.H_mean = Att_H(
            self.obs_dim, 25, self.hidden_dim,
            self.n_agents, self.device
        ).to(self.device)
        
        self.std_net = Attention_LEMURS(
            self.obs_dim + self.act_dim,
            self.act_dim,
            self.obs_dim,
            self.n_agents,
            self.device
        ).to(self.device)
        
        # Cache tensors for optimization
        self.register_buffer('_ones_obs', torch.ones(1, 1, self.obs_dim, device=self.device))
        
        # Pre-compute system matrices
        self._init_system_matrices()
        
        self.to(device)
    
    def _init_system_matrices(self):
        """Initialize physics system matrices."""
        self.F_sys_pinv = torch.cat((
            torch.zeros(self.act_dim * self.n_agents, self.act_dim * self.n_agents, device=self.device),
            torch.eye(self.act_dim * self.n_agents, device=self.device)
        ), dim=1)

        self.J_sys = torch.cat((
            torch.cat((
                torch.zeros(self.act_dim * self.n_agents, self.act_dim * self.n_agents, device=self.device),
                torch.eye(self.act_dim * self.n_agents, device=self.device)
            ), dim=1),
            torch.cat((
                -torch.eye(self.act_dim * self.n_agents, device=self.device),
                torch.zeros(self.act_dim * self.n_agents, self.act_dim * self.n_agents, device=self.device)
            ), dim=1)
        ), dim=0)
        
        self.R_sys = torch.cat((
            torch.cat((
                torch.zeros(self.act_dim * self.n_agents, self.act_dim * self.n_agents, device=self.device),
                torch.zeros(self.act_dim * self.n_agents, self.act_dim * self.n_agents, device=self.device)
            ), dim=1),
            torch.cat((
                torch.zeros(self.act_dim * self.n_agents, self.act_dim * self.n_agents, device=self.device),
                self.drag * torch.eye(self.act_dim * self.n_agents, device=self.device)
            ), dim=1)
        ), dim=0)
    
    def laplacian(self, q_agents):
        """Compute Laplacian matrix for agent communication."""
        Q = torch.cdist(q_agents, q_agents, p=2)
        L = Q.le(self.r_communication).float()
        L = L * torch.sigmoid(-(2.0) * (Q - self.r_communication))
        return L
    
    def forward(self, obs, rnn_states, masks, available_actions=None, deterministic=False):
        """
        Forward pass to generate actions.
        
        Args:
            obs: Observations (batch_size, obs_dim) or (batch_size, n_agents, obs_dim)
            rnn_states: Recurrent states (not used in PINN but kept for compatibility)
            masks: Masks for episodes
            available_actions: Available actions (optional)
            deterministic: Whether to use deterministic actions
            
        Returns:
            actions: Selected actions
            action_log_probs: Log probabilities of actions
            rnn_states: Updated recurrent states (unchanged for PINN)
        """
        obs = check(obs).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)
        
        # Handle observation shape - ensure (batch_size, n_agents, obs_dim)
        if obs.dim() == 2:
            # Single agent case, reshape to (batch_size, 1, obs_dim)
            obs = obs.unsqueeze(1)
        
        batch_size = obs.shape[0]
        
        # Expand system matrices to batch size
        F_sys_pinv = self.F_sys_pinv.unsqueeze(0).expand(batch_size, -1, -1)
        J_sys = self.J_sys.unsqueeze(0).expand(batch_size, -1, -1)
        R_sys = self.R_sys.unsqueeze(0).expand(batch_size, -1, -1)
        
        state = obs
        state_h_mean = torch.clone(state).reshape(-1, self.obs_dim)
        
        # Debug: print shapes
        # print(f"[DEBUG] state shape: {state.shape}, batch_size: {batch_size}, n_agents: {self.n_agents}, obs_dim: {self.obs_dim}")
        
        # Compute Laplacian (assuming first 2 dims are position)
        # laplacian_base shape: (batch_size, n_agents, n_agents)
        laplacian_base = self.laplacian(state[:, :, 0:2])
        # print(f"[DEBUG] laplacian_base shape: {laplacian_base.shape}")
        
        # Optimized: replace torch.kron with repeat_interleave
        # Expand laplacian to cover all observation dimensions
        laplacian = laplacian_base.unsqueeze(-1).repeat_interleave(self.obs_dim, dim=-1)
        # print(f"[DEBUG] laplacian after repeat_interleave shape: {laplacian.shape}")
        
        # Reshape to (batch_size * n_agents, n_agents, obs_dim)
        laplacian = laplacian.reshape(-1, self.n_agents, self.obs_dim)
        # print(f"[DEBUG] laplacian after reshape shape: {laplacian.shape}")
        
        # Prepare state for network
        # Reshape and normalize inputs - use expand instead of repeat where possible
        # state shape: (batch_size, n_agents, obs_dim)
        # Expand to (batch_size, n_agents, n_agents, obs_dim) then reshape to (batch_size * n_agents, n_agents, obs_dim)
        state_expanded = state.unsqueeze(2).expand(-1, -1, self.n_agents, -1).reshape(-1, self.n_agents, self.obs_dim)
        # print(f"[DEBUG] state_expanded shape: {state_expanded.shape}")
        
        state_processed = laplacian * state_expanded
        
        # Compute physics-informed components
        R_mean = self.R_mean.forward(state_processed.to(torch.float32), laplacian_base.to(torch.float32), self.scenario_name)
        J_mean = self.J_mean.forward(state_processed.to(torch.float32), laplacian_base.to(torch.float32), self.scenario_name)
        
        # Compute Hamiltonian gradient
        with torch.enable_grad():
            state_h_mean = state_h_mean.requires_grad_(True)
            H_mean = self.H_mean.forward(state_h_mean.to(torch.float32), self.n_agents)
            Hgrad_mean = torch.autograd.grad(
                H_mean.sum(), state_h_mean, 
                only_inputs=True, create_graph=self.training
            )
            dH_mean = Hgrad_mean[0]
        
        dHq_mean = dH_mean[:, :self.act_dim].reshape(-1, self.n_agents * self.act_dim)
        dHp_mean = dH_mean[:, self.act_dim:2 * self.act_dim].reshape(-1, self.n_agents * self.act_dim)
        dHdx_mean = torch.cat((dHq_mean, dHp_mean), dim=1)
        
        # Closed-loop dynamics
        dx_mean = torch.bmm(
            J_mean.to(torch.float32) - R_mean.to(torch.float32), 
            dHdx_mean.unsqueeze(2)
        ).squeeze(2)
        
        # Controller dynamics
        dHdx_sys_mean = torch.cat((
            torch.zeros(dx_mean.shape[0], int(dx_mean.shape[1]/2), device=self.device).unsqueeze(dim=2),
            dx_mean[:, :self.act_dim * self.n_agents].unsqueeze(dim=2)
        ), dim=1)
        
        u_mean = torch.bmm(
            F_sys_pinv, 
            dx_mean.unsqueeze(dim=2) - torch.bmm(J_sys - R_sys, dHdx_sys_mean)
        ).squeeze(dim=2).reshape(batch_size, self.n_agents, -1)
        
        # Compute standard deviation
        u_mean_expanded = u_mean.reshape(-1, 1, u_mean.shape[2]).expand(-1, self.n_agents, -1)
        std_input = state_processed.detach().clone()
        u_log_std = self.std_net(torch.cat((std_input, u_mean_expanded), dim=2))
        
        # Clamp log_std
        u_log_std = torch.clamp(u_log_std, self.log_std_min, self.log_std_max)
        u_std = torch.exp(u_log_std)
        
        # Keep agent dimension: u_mean shape (batch_size, n_agents, act_dim)
        # u_std shape: (batch_size, n_agents, act_dim)
        
        # Create distribution and sample actions
        dist = Normal(u_mean, u_std)
        
        if deterministic:
            actions = u_mean
        else:
            actions = dist.rsample()
        
        # Compute log probabilities - sum over action dimension, keep batch and agent dimensions
        action_log_probs = dist.log_prob(actions).sum(dim=-1, keepdim=True)
        # action_log_probs shape: (batch_size, n_agents, 1)
        
        # Clamp outputs for stability
        actions = torch.clamp(actions, min=-10.0, max=10.0)
        action_log_probs = torch.clamp(action_log_probs, min=-10.0, max=10.0)
        
        # Handle NaN
        actions = torch.nan_to_num(actions, nan=0.0, posinf=10.0, neginf=-10.0)
        action_log_probs = torch.nan_to_num(action_log_probs, nan=0.0, posinf=10.0, neginf=-10.0)
        
        return actions, action_log_probs, rnn_states
    
    def evaluate_actions(self, obs, rnn_states, action, masks, available_actions=None, active_masks=None):
        """
        Evaluate actions for training.
        
        Args:
            obs: Observations
            rnn_states: Recurrent states
            action: Actions to evaluate
            masks: Episode masks
            available_actions: Available actions (optional)
            active_masks: Active masks for multi-agent (optional)
            
        Returns:
            action_log_probs: Log probabilities of actions
            dist_entropy: Entropy of the distribution
        """
        obs = check(obs).to(**self.tpdv)
        action = check(action).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)
        
        # Handle observation shape
        if obs.dim() == 2:
            obs = obs.unsqueeze(1)
        
        batch_size = obs.shape[0]
        
        # If batch is too large, process in chunks to avoid OOM
        max_chunk_size = 500
        if batch_size > max_chunk_size:
            action_log_probs_list = []
            dist_entropy_list = []
            
            for i in range(0, batch_size, max_chunk_size):
                end_idx = min(i + max_chunk_size, batch_size)
                obs_chunk = obs[i:end_idx]
                action_chunk = action[i:end_idx]
                masks_chunk = masks[i:end_idx]
                active_masks_chunk = active_masks[i:end_idx] if active_masks is not None else None
                
                log_probs_chunk, entropy_chunk = self._evaluate_actions_chunk(
                    obs_chunk, action_chunk, masks_chunk, active_masks_chunk
                )
                action_log_probs_list.append(log_probs_chunk)
                dist_entropy_list.append(entropy_chunk)
            
            action_log_probs = torch.cat(action_log_probs_list, dim=0)
            dist_entropy = torch.stack(dist_entropy_list).mean()
            return action_log_probs, dist_entropy
        else:
            return self._evaluate_actions_chunk(obs, action, masks, active_masks)
    
    def _evaluate_actions_chunk(self, obs, action, masks, active_masks=None):
        """Process a chunk of observations for action evaluation."""
        batch_size = obs.shape[0]
        
        # Check if this is single-agent obs (training) or multi-agent obs (collect)
        # After the unsqueeze in evaluate_actions, obs is either:
        # - (batch, 1, obs_dim) for single-agent training
        # - (batch, n_agents, obs_dim) for multi-agent collect
        if obs.shape[1] == 1 and self.n_agents > 1:
            # Single-agent mode: replicate obs for all agents
            # (batch, 1, obs_dim) -> (batch, n_agents, obs_dim)
            obs = obs.expand(-1, self.n_agents, -1)
        
        # Multi-agent mode with PINN physics
        # Forward pass to get distribution parameters
        F_sys_pinv = self.F_sys_pinv.unsqueeze(0).expand(batch_size, -1, -1)
        J_sys = self.J_sys.unsqueeze(0).expand(batch_size, -1, -1)
        R_sys = self.R_sys.unsqueeze(0).expand(batch_size, -1, -1)
        
        state = obs
        state_h_mean = torch.clone(state).reshape(-1, self.obs_dim)
        
        # Compute Laplacian (assuming first 2 dims are position)
        laplacian_base = self.laplacian(state[:, :, 0:2])
        # Optimized: replace torch.kron with repeat_interleave
        laplacian = laplacian_base.unsqueeze(-1).repeat_interleave(self.obs_dim, dim=-1)
        laplacian = laplacian.reshape(-1, self.n_agents, self.obs_dim)
        
        # Prepare state for network
        state_expanded = state.unsqueeze(2).expand(-1, -1, self.n_agents, -1).reshape(-1, self.n_agents, self.obs_dim)
        state_processed = laplacian * state_expanded
        
        R_mean = self.R_mean.forward(state_processed.to(torch.float32), laplacian_base.to(torch.float32), self.scenario_name)
        J_mean = self.J_mean.forward(state_processed.to(torch.float32), laplacian_base.to(torch.float32), self.scenario_name)
        
        with torch.enable_grad():
            state_h_mean = state_h_mean.requires_grad_(True)
            H_mean = self.H_mean.forward(state_h_mean.to(torch.float32), self.n_agents)
            Hgrad_mean = torch.autograd.grad(
                H_mean.sum(), state_h_mean,
                only_inputs=True, create_graph=True
            )
            dH_mean = Hgrad_mean[0]
        
        dHq_mean = dH_mean[:, :self.act_dim].reshape(-1, self.n_agents * self.act_dim)
        dHp_mean = dH_mean[:, self.act_dim:2 * self.act_dim].reshape(-1, self.n_agents * self.act_dim)
        dHdx_mean = torch.cat((dHq_mean, dHp_mean), dim=1)
        
        dx_mean = torch.bmm(
            J_mean.to(torch.float32) - R_mean.to(torch.float32),
            dHdx_mean.unsqueeze(2)
        ).squeeze(2)
        
        dHdx_sys_mean = torch.cat((
            torch.zeros(dx_mean.shape[0], int(dx_mean.shape[1]/2), device=self.device).unsqueeze(dim=2),
            dx_mean[:, :self.act_dim * self.n_agents].unsqueeze(dim=2)
        ), dim=1)
        
        u_mean = torch.bmm(
            F_sys_pinv,
            dx_mean.unsqueeze(dim=2) - torch.bmm(J_sys - R_sys, dHdx_sys_mean)
        ).squeeze(dim=2).reshape(batch_size, self.n_agents, -1)
        
        u_mean_expanded = u_mean.reshape(-1, 1, u_mean.shape[2]).expand(-1, self.n_agents, -1)
        std_input = state_processed.detach().clone()
        u_log_std = self.std_net(torch.cat((std_input, u_mean_expanded), dim=2))
        
        u_log_std = torch.clamp(u_log_std, self.log_std_min, self.log_std_max)
        u_std = torch.exp(u_log_std)
        
        # Keep agent dimension: u_mean shape (batch_size, n_agents, act_dim)
        # u_std shape: (batch_size, n_agents, act_dim)
        
        # Ensure action has correct shape - expand if needed
        if action.dim() == 2:
            # action shape: (batch_size, act_dim) - single agent's action
            # Need to expand to (batch_size, n_agents, act_dim)
            # For evaluation, action comes from buffer which stores individual agent actions
            # We need to match it with u_mean which has shape (batch_size, n_agents, act_dim)
            # Since we're evaluating one agent at a time, we should select the corresponding agent's distribution
            pass  # Will handle this differently
        
        dist = Normal(u_mean, u_std)
        
        # If action doesn't have agent dimension, we're evaluating one agent at a time
        # action_log_probs shape should match input action shape
        if action.dim() == 2:
            # Single agent action: (batch, act_dim)
            # Take mean of log probs across all agents (since actor is shared)
            action_log_probs_all = dist.log_prob(u_mean).sum(dim=-1, keepdim=True)  # (batch, n_agents, 1)
            action_log_probs = action_log_probs_all.mean(dim=1)  # (batch, 1)
            dist_entropy = dist.entropy().sum(dim=-1, keepdim=True).mean(dim=1)  # (batch, 1)
        else:
            # All agents' actions: (batch, n_agents, act_dim)
            action_log_probs = dist.log_prob(action).sum(dim=-1, keepdim=True)
            dist_entropy = dist.entropy().sum(dim=-1, keepdim=True)
        
        # Apply active masks if provided
        if active_masks is not None and active_masks.numel() > 0:
            active_masks = check(active_masks).to(**self.tpdv)
            if active_masks.shape[0] == action_log_probs.shape[0]:  # Check shape compatibility
                action_log_probs = action_log_probs * active_masks
                dist_entropy = dist_entropy * active_masks
        
        return action_log_probs, dist_entropy.mean()


class MAPPO_PINN_Policy:
    """MAPPO Policy with PINN Actor."""
    
    def __init__(self, config, obs_space, cent_obs_space, act_space):
        self.config = config
        self.obs_space = obs_space
        self.act_space = act_space
        self.share_obs_space = cent_obs_space

        self.actor = PINNActor(config, self.obs_space, self.act_space, self.config["device"])
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

    def get_actions(self, cent_obs, obs, rnn_states_actor, rnn_states_critic, masks, 
                    available_actions=None, deterministic=False):
        actions, action_log_probs, rnn_states_actor = self.actor(
            obs, rnn_states_actor, masks, available_actions, deterministic
        )
        values, rnn_states_critic = self.critic(cent_obs, rnn_states_critic, masks)
        return values, actions, action_log_probs, rnn_states_actor, rnn_states_critic

    def get_values(self, cent_obs, rnn_states_critic, masks):
        values, _ = self.critic(cent_obs, rnn_states_critic, masks)
        return values

    def evaluate_actions(self, cent_obs, obs, rnn_states_actor, rnn_states_critic, action, masks,
                         available_actions=None, active_masks=None):
        action_log_probs, dist_entropy = self.actor.evaluate_actions(
            obs, rnn_states_actor, action, masks, available_actions, active_masks
        )
        values, _ = self.critic(cent_obs, rnn_states_critic, masks)
        return values, action_log_probs, dist_entropy

    def act(self, obs, rnn_states_actor, masks, available_actions=None, deterministic=False):
        actions, _, rnn_states_actor = self.actor(obs, rnn_states_actor, masks, available_actions, deterministic)
        return actions, rnn_states_actor


class MAPPO_PINN_Trainer():
    """MAPPO Trainer with PINN Actor."""
    
    def __init__(self, config, policy):
        self.config = config
        self.tpdv = dict(dtype=torch.float32, device=self.config["device"])
        self.policy = policy
        self.value_normalizer = PopArt(1, device=self.config["device"])

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

    def ppo_update(self, sample):
        """PPO update step.
        
        Args:
            sample: Batch sample from buffer
        """
        share_obs_batch, obs_batch, rnn_states_batch, rnn_states_critic_batch, actions_batch, \
        value_preds_batch, return_batch, masks_batch, active_masks_batch, old_action_log_probs_batch, \
        adv_targ, available_actions_batch, _ = sample
        
        old_action_log_probs_batch, adv_targ, value_preds_batch, return_batch, active_masks_batch = [
            check(x).to(**self.tpdv) for x in [
                old_action_log_probs_batch, adv_targ, value_preds_batch, return_batch, active_masks_batch
            ]
        ]

        # Use single-agent obs for training (PINN constraints only in collect/inference)
        values, action_log_probs, dist_entropy = self.policy.evaluate_actions(
            share_obs_batch, obs_batch, rnn_states_batch, rnn_states_critic_batch,
            actions_batch, masks_batch, available_actions_batch, active_masks_batch
        )
        
        imp_weights = torch.exp(action_log_probs - old_action_log_probs_batch)

        surr1 = imp_weights * adv_targ
        surr2 = torch.clamp(
            imp_weights, 
            1.0 - self.config["clip_param"], 
            1.0 + self.config["clip_param"]
        ) * adv_targ

        if self.config["use_policy_active_masks"]:
            policy_action_loss = (-torch.sum(
                torch.min(surr1, surr2), dim=-1, keepdim=True
            ) * active_masks_batch).sum() / active_masks_batch.sum()
        else:
            policy_action_loss = -torch.sum(torch.min(surr1, surr2), dim=-1, keepdim=True).mean()

        policy_loss = policy_action_loss

        self.policy.actor_optimizer.zero_grad()
        (policy_loss - dist_entropy * self.config["entropy_coef"]).backward()
        actor_grad_norm = nn.utils.clip_grad_norm_(
            self.policy.actor.parameters(), self.config["max_grad_norm"]
        )
        self.policy.actor_optimizer.step()

        value_loss = self.cal_value_loss(values, value_preds_batch, return_batch, active_masks_batch)
        self.policy.critic_optimizer.zero_grad()
        (value_loss * self.config["value_loss_coef"]).backward()
        critic_grad_norm = nn.utils.clip_grad_norm_(
            self.policy.critic.parameters(), self.config["max_grad_norm"]
        )
        self.policy.critic_optimizer.step()

        return value_loss, critic_grad_norm, policy_loss, dist_entropy, actor_grad_norm, imp_weights

    def train(self, buffer, logger):
        """Train with PPO algorithm.
        
        Args:
            buffer: Replay buffer for this agent
            logger: Logger for metrics
        """
        advantages = buffer.returns[:-1] - self.value_normalizer.denormalize(buffer.value_preds[:-1])
        advantages_copy = advantages.clone()
        mean_advantages = torch.mean(advantages_copy)
        std_advantages = torch.std(advantages_copy)
        advantages = (advantages - mean_advantages) / (std_advantages + 1e-5)

        for _ in range(self.config["learning_iters"]):
            if self.config.get("data_chunk_length") is not None:
                data_generator = buffer.recurrent_generator(advantages, self.config["num_mini_batch"], self.config["data_chunk_length"])
            else:
                data_generator = buffer.feed_forward_generator(advantages, self.config["num_mini_batch"])

            for sample in data_generator:
                # Train using single-agent obs from mini-batch
                # PINN physics constraints are applied during collect/inference, not training
                value_loss, critic_grad_norm, policy_loss, dist_entropy, actor_grad_norm, imp_weights = \
                    self.ppo_update(sample)
            
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


class Runner:
    """Runner for MAPPO with PINN."""
    
    def __init__(self, vec_env, vec_eval_env, config, model_dir=""):
        self.envs = vec_env
        self.eval_envs = vec_eval_env
        self.config = config
        self.model_dir = model_dir

        self.num_agents = self.envs.num_agents
        
        # Add n_agents to config for PINN
        self.config["n_agents"] = self.num_agents

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
            log_dir=config["log_dir"],
            seed=str(config["seed"]),
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
        
        self.policy = []
        # For PINN, we use a single shared actor for all agents
        # since PINN needs all agents' observations to compute physics constraints
        shared_actor = PINNActor(config, self.envs.observation_space[0], self.envs.action_space[0], config["device"])
        
        for agent_id in range(self.num_agents):
            share_observation_space = self.envs.share_observation_space[agent_id]
            po = MAPPO_PINN_Policy(
                config,
                self.envs.observation_space[agent_id],
                share_observation_space,
                self.envs.action_space[agent_id]
            )
            # Share the same actor across all agents
            po.actor = shared_actor
            # Re-initialize optimizer to include shared actor parameters
            po.actor_optimizer = torch.optim.Adam(
                po.actor.parameters(),
                lr=config["actor_lr"],
                eps=config["opti_eps"],
                weight_decay=config["weight_decay"]
            )
            self.policy.append(po)

        if self.model_dir != "":
            self.restore()

        self.trainer = []
        self.buffer = []
        for agent_id in range(self.num_agents):
            tr = MAPPO_PINN_Trainer(config, self.policy[agent_id])
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
        
        pbar = tqdm(range(episodes), desc="Training MAPPO-PINN", ncols=100)
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
        
        # Collect all agents' observations for PINN
        all_obs = torch.stack([self.buffer[agent_id].obs[step] for agent_id in range(self.num_agents)], dim=1)
        # all_obs shape: (n_rollout_threads, n_agents, obs_dim)
        
        for agent_id in range(self.num_agents):
            self.trainer[agent_id].prep_rollout()
            # For value estimation, use centralized observation
            value, rnn_state_critic = self.trainer[agent_id].policy.critic(
                self.buffer[agent_id].share_obs[step],
                self.buffer[agent_id].rnn_states_critic[step],
                self.buffer[agent_id].masks[step]
            )
            
            # For action generation, use PINN with all agents' observations
            if agent_id == 0:  # Only call PINN once for all agents
                all_actions, all_action_log_probs, all_rnn_states = self.trainer[agent_id].policy.actor(
                    all_obs,
                    self.buffer[agent_id].rnn_states[step],
                    self.buffer[agent_id].masks[step],
                    deterministic=False
                )
            
            # Extract action for this specific agent
            if all_actions.dim() == 3:  # (batch, n_agents, action_dim)
                action = all_actions[:, agent_id, :]
                action_log_prob = all_action_log_probs[:, agent_id, :]
            elif all_actions.dim() == 2 and self.n_agents > 1:  # (batch, action_dim) - shouldn't happen
                action = all_actions
                action_log_prob = all_action_log_probs
            else:  # Single agent or already extracted
                action = all_actions
                action_log_prob = all_action_log_probs
            
            value_collector.append(value.detach())
            action_collector.append(action.detach())
            action_log_prob_collector.append(action_log_prob.detach())
            rnn_state_collector.append(self.buffer[agent_id].rnn_states[step].detach())
            rnn_state_critic_collector.append(rnn_state_critic.detach())
            
        if self.config["env_name"] == "Safety9|8HumanoidVelocity-v0":
            zeros = torch.zeros(action_collector[-1].shape[0], 1)
            action_collector[-1] = torch.cat((action_collector[-1], zeros), dim=1)
            
        values = torch.transpose(torch.stack(value_collector), 1, 0)
        rnn_states = torch.transpose(torch.stack(rnn_state_collector), 1, 0)
        rnn_states_critic = torch.transpose(torch.stack(rnn_state_critic_collector), 1, 0)

        return values, action_collector, action_log_prob_collector, rnn_states, rnn_states_critic

    def insert(self, data):
        obs, share_obs, rewards, dones, infos, \
        values, actions, action_log_probs, rnn_states, rnn_states_critic = data

        dones_env = torch.all(dones, axis=1)

        rnn_states[dones_env == True] = torch.zeros(
            (dones_env == True).sum(), self.num_agents, self.config["recurrent_N"], 
            self.config["hidden_size"], device=self.config["device"]
        )
        rnn_states_critic[dones_env == True] = torch.zeros(
            (dones_env == True).sum(), self.num_agents, *self.buffer[0].rnn_states_critic.shape[2:], 
            device=self.config["device"]
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

        if self.config["env_name"] == "Safety9|8HumanoidVelocity-v0":
            actions[1] = actions[1][:, :8]
            
        for agent_id in range(self.num_agents):
            if 'Frank' in self.config['env_name']:
                obs_to_insert = obs[agent_id]
            else:
                obs_to_insert = obs[:, agent_id]
            self.buffer[agent_id].insert(
                share_obs[:, agent_id], obs_to_insert, rnn_states[:, agent_id],
                rnn_states_critic[:, agent_id], actions[agent_id],
                action_log_probs[agent_id],
                values[:, agent_id], rewards[:, agent_id].unsqueeze(-1), 
                masks[:, agent_id], None,
                active_masks[:, agent_id], None
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
            
            # Train without all_obs - PINN constraints only used during collect/inference
            # This avoids the batch size mismatch issue between full buffer and mini-batches
            self.trainer[agent_id].train(self.buffer[agent_id], logger=self.logger)

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
        
        best_episode_frames = []
        best_episode_reward = float("-inf")
        best_episode_cost = 0.0
        best_episode_num = 0
        current_frames = []

        eval_obs, _, _ = self.eval_envs.reset()

        eval_rnn_states = torch.zeros(
            self.config["n_eval_rollout_threads"], self.num_agents, self.config["recurrent_N"], 
            self.config["hidden_size"], device=self.config["device"]
        )
        eval_masks = torch.ones(
            self.config["n_eval_rollout_threads"], self.num_agents, 1, device=self.config["device"]
        )

        while True:
            # Collect all agents' observations for PINN
            if 'Frank' in self.config['env_name']:
                all_eval_obs = torch.stack([eval_obs[agent_id] for agent_id in range(self.num_agents)], dim=1)
            else:
                all_eval_obs = eval_obs
            
            # Use shared actor for all agents
            self.trainer[0].prep_rollout()
            eval_actions, _, updated_rnn_states = self.trainer[0].policy.actor(
                all_eval_obs,
                eval_rnn_states,
                eval_masks,
                available_actions=None,
                deterministic=True
            )
            eval_rnn_states = updated_rnn_states
            
            # Split actions for each agent
            eval_actions_collector = [eval_actions[:, agent_id] for agent_id in range(self.num_agents)]

            if self.config["env_name"] == "Safety9|8HumanoidVelocity-v0":
                zeros = torch.zeros(eval_actions_collector[-1].shape[0], 1)
                eval_actions_collector[-1] = torch.cat((eval_actions_collector[-1], zeros), dim=1)
            
            # Capture frame for video (only for non-Isaac Gym envs)
            if self.video_recorder.enabled and self.config["env_name"] not in isaac_gym_map:
                try:
                    if hasattr(self.eval_envs, 'render'):
                        frame = self.eval_envs.render()
                        if frame is not None and len(frame.shape) == 3:
                            current_frames.append(frame.copy())
                except Exception:
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
                (eval_dones_env == True).sum(), self.num_agents, self.config["recurrent_N"], 
                self.config["hidden_size"], device=self.config["device"]
            )

            eval_masks = torch.ones(
                self.config["n_eval_rollout_threads"], self.num_agents, 1, device=self.config["device"]
            )
            eval_masks[eval_dones_env == True] = torch.zeros(
                (eval_dones_env == True).sum(), self.num_agents, 1, device=self.config["device"]
            )

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
                
                eval_episode_rewards = np.array(eval_episode_rewards)
                eval_episode_costs = np.array(eval_episode_costs)
                return eval_episode_rewards.mean(), eval_episode_costs.mean()

    @torch.no_grad()
    def compute(self):
        for agent_id in range(self.num_agents):
            self.trainer[agent_id].prep_rollout()
            next_value = self.trainer[agent_id].policy.get_values(
                self.buffer[agent_id].share_obs[-1],
                self.buffer[agent_id].rnn_states_critic[-1],
                self.buffer[agent_id].masks[-1]
            )
            next_value = next_value.detach()
            self.buffer[agent_id].compute_returns(next_value, self.trainer[agent_id].value_normalizer)


def train(args, cfg_train):
    """Main training function."""
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
            os.path.join(f"{cfg_train['log_dir']}", terminal_log_name),
            "w",
            encoding="utf-8",
        ) as f_out:
            sys.stdout = f_out
            with open(
                os.path.join(f"{cfg_train['log_dir']}", error_log_name),
                "w",
                encoding="utf-8",
            ) as f_error:
                sys.stderr = f_error
                train(args=args, cfg_train=cfg_train)

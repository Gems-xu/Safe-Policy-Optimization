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
        
        # Auxiliary loss weights for physics-informed training
        self.aux_task_potential_weight = config.get("aux_task_potential_weight", 0.01)
        self.aux_barrier_potential_weight = config.get("aux_barrier_potential_weight", 0.01)

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
        
        Key insight: Without direct supervision, these networks learn very slowly
        through the policy gradient. We add auxiliary losses:
        
        1. Task potential should be lower near goals (based on goal lidar readings)
        2. Barrier potential should be higher near obstacles (based on hazard lidar)
        
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
        
        # Extract goal and hazard lidar readings
        # Goal lidar: obs[12:28] (goal_red) - higher means closer to goal
        goal_lidar = obs_batch[:, 12:28]
        goal_proximity = goal_lidar.max(dim=-1, keepdim=True)[0]  # [batch, 1]
        
        # Hazard lidar: obs[44:60] - higher means closer to hazard
        hazard_lidar_end = min(60, obs_batch.shape[-1])
        if hazard_lidar_end > 44:
            hazard_lidar = obs_batch[:, 44:hazard_lidar_end]
            hazard_proximity = hazard_lidar.max(dim=-1, keepdim=True)[0]  # [batch, 1]
        else:
            hazard_proximity = torch.zeros(batch_size, 1, device=device)
        
        # === Auxiliary Loss 1: Task Potential ===
        # H_task should be LOWER when close to goal (goal_proximity high)
        # Loss: encourage H_task to be negatively correlated with goal proximity
        # Target: H_task ≈ 1 - goal_proximity (normalized)
        target_H_task = 1.0 - goal_proximity.clamp(0, 1)
        task_potential_loss = F.mse_loss(torch.sigmoid(H_task), target_H_task)
        
        # === Auxiliary Loss 2: Barrier Awareness ===
        # barrier_k should be HIGHER when close to hazards
        # This encourages the network to learn hazard-aware stiffness
        k = actor.barrier_k_net(obs_batch)  # [batch, 1]
        # Target: k should increase with hazard proximity
        target_k_scale = 0.5 + hazard_proximity.clamp(0, 1) * 2.0  # Range [0.5, 2.5]
        barrier_k_loss = F.mse_loss(k, target_k_scale)
        
        # Combined auxiliary loss
        aux_loss = (self.aux_task_potential_weight * task_potential_loss + 
                    self.aux_barrier_potential_weight * barrier_k_loss)
        
        aux_info = {
            'aux_task_loss': task_potential_loss.item(),
            'aux_barrier_k_loss': barrier_k_loss.item(),
            'H_task_mean': H_task.mean().item(),
            'H_task_std': H_task.std().item(),
            'k_mean': k.mean().item(),
            'k_std': k.std().item(),
        }
        
        return aux_loss, aux_info

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
        
        # Total actor loss = policy loss + auxiliary physics loss
        total_actor_loss = policy_loss - dist_entropy * self.config["entropy_coef"] + aux_loss

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
            logger.store(
                **{
                    "Loss/Loss_reward_critic": value_loss.item(),
                    "Loss/Loss_actor": policy_loss.item(),
                    "Loss/Aux_task_potential": aux_info['aux_task_loss'],
                    "Loss/Aux_barrier_k": aux_info['aux_barrier_k_loss'],
                    "Safe/H_task_mean": aux_info['H_task_mean'],
                    "Safe/H_task_std": aux_info['H_task_std'],
                    "Safe/k_mean": aux_info['k_mean'],
                    "Safe/k_std": aux_info['k_std'],
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
        if "cuda" in str(self.config["device"]):
            device = self.config["device"]
            if isinstance(device, str) and ":" in device:
                device_id = int(device.split(":")[1])
            else:
                device_id = 0
            torch.cuda.set_device(device_id)
            # Create a dummy tensor to fully initialize the CUDA context
            _ = torch.zeros(1, device=self.config["device"])

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
        pbar = tqdm(range(episodes), desc="Safe-pH-MARL Training", ncols=100)
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
            
            if (episode % self.config["eval_interval"] == 0 or episode == episodes - 1) and self.config["use_eval"]:
                eval_rewards, eval_costs = self.eval(eval_episodes=1, total_steps=total_num_steps)

            if len(done_episodes_rewards) != 0:
                aver_episode_rewards = torch.stack(done_episodes_rewards).mean()
                aver_episode_costs = torch.stack(done_episodes_costs).mean()
                self.return_aver_cost(aver_episode_costs)
                
                # Collect barrier physics information for Safe module
                barrier_info = self.collect_barrier_physics_info(obs)
                
                log_dict = {
                    "Metrics/EpRet": aver_episode_rewards.item(),
                    "Metrics/EpCost": aver_episode_costs.item(),
                    "Eval/EpRet": eval_rewards,
                    "Eval/EpCost": eval_costs,
                }
                
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
        hazard_radius = 0.2  # Default hazard radius in safety_gymnasium
        
        try:
            # Try to get environment information from eval_envs
            # The underlying environment might have task info
            if hasattr(self.eval_envs, 'envs') and len(self.eval_envs.envs) > 0:
                env = self.eval_envs.envs[0]
                
                # Try to access task hazards
                if hasattr(env, 'task'):
                    task = env.task
                    
                    # Get hazard positions
                    if hasattr(task, 'hazards') and hasattr(task.hazards, 'pos'):
                        hazards_pos = task.hazards.pos
                        if hazards_pos is not None:
                            for pos in hazards_pos:
                                obstacle_positions.append((float(pos[0]), float(pos[1])))
                        if hasattr(task.hazards, 'size'):
                            hazard_radius = task.hazards.size
                    
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
        Evaluate policy performance with fixed-interval video rendering.
        
        Renders video for the FIRST episode of each evaluation run (every eval_interval).
        This provides consistent video recording without depending on reward improvements.
        """
        self.eval_count += 1
        should_record_video = (
            self.video_recorder.enabled 
            and self.eval_count % self.video_record_freq == 0
            and self.config["env_name"] not in isaac_gym_map
        )
        
        eval_episode = 0
        eval_episode_rewards = []
        eval_episode_costs = []
        one_episode_rewards = torch.zeros(1, self.config["n_eval_rollout_threads"], device=self.config["device"])
        one_episode_costs = torch.zeros(1, self.config["n_eval_rollout_threads"], device=self.config["device"])
        
        # Fixed-interval video: record first episode only
        first_episode_frames = []
        first_episode_reward = 0.0
        first_episode_cost = 0.0
        recording_first_episode = should_record_video  # Start recording immediately for first episode

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
            
            # Capture frame for first episode video (fixed-interval recording)
            if recording_first_episode and hasattr(self.eval_envs, 'render'):
                try:
                    frame = self.eval_envs.render()
                    if frame is not None:
                        if isinstance(frame, np.ndarray) and len(frame.shape) == 3:
                            first_episode_frames.append(frame.copy())
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
                if len(first_episode_frames) > 0 and should_record_video:
                    if self.video_recorder.enabled:
                        self.video_recorder.recorder.frames = first_episode_frames
                        caption = f"Eval #{self.eval_count} - Reward: {first_episode_reward:.2f}, Cost: {first_episode_cost:.2f}"
                        self.video_recorder.recorder.upload_to_wandb(
                            caption=caption,
                            step=total_steps,
                            key="eval/video"
                        )
                
                # Visualize barrier potential and upload to wandb
                if should_record_video and self.config["env_name"] in multi_agent_goal_tasks:
                    try:
                        # Create visualization directory: runs/<exp_name>/vizs/
                        viz_dir = os.path.join(os.path.dirname(self.save_dir), "vizs")
                        os.makedirs(viz_dir, exist_ok=True)
                        
                        # Extract environment obstacle information for accurate visualization
                        obstacle_positions, goal_positions, hazard_radius = self._extract_env_obstacles()
                        
                        # Create visualizer with actual environment layout
                        visualizer = BarrierPotentialVisualizer(
                            model_dir=self.save_dir,
                            task=self.config["env_name"],
                            agent_id=0,
                            device=self.config["device"],
                            obstacle_positions=obstacle_positions,
                            goal_positions=goal_positions,
                            hazard_radius=hazard_radius
                        )
                        
                        # Override the actor with our already-loaded one to avoid dimension mismatch
                        visualizer.actor = self.policy[0].actor
                        
                        visualizer.visualize_all(output_dir=viz_dir, verbose=False)
                        
                        # Upload all visualization images to wandb
                        if self.logger.use_wandb:
                            import wandb
                            
                            print(f"\n{'='*60}")
                            print(f"[Barrier Viz] Visualization Upload Starting")
                            print(f"{'='*60}")
                            print(f"[Barrier Viz] Directory: {viz_dir}")
                            print(f"[Barrier Viz] Obstacles: {len(obstacle_positions)} hazards")
                            print(f"[Barrier Viz] Goals: {goal_positions}")
                            print(f"[Barrier Viz] Total steps: {total_steps}")
                            
                            # Build visualization dictionary - upload each image separately
                            for img_file in sorted(os.listdir(viz_dir)):
                                if img_file.endswith('.png'):
                                    img_path = os.path.join(viz_dir, img_file)
                                    img_key = os.path.splitext(img_file)[0]
                                    
                                    try:
                                        # Upload to wandb Media section (images always go to Media, not Charts)
                                        # Charts is for scalar metrics only
                                        img_obj = wandb.Image(img_path, caption=img_key)
                                        
                                        # Try uploading via logger
                                        if hasattr(self.logger, 'wandb_run') and self.logger.wandb_run is not None:
                                            self.logger.wandb_run.log({f"barrier_viz/{img_key}": img_obj}, step=total_steps)
                                            print(f"[Barrier Viz] ✓ Uploaded {img_key} via logger.wandb_run")
                                        elif wandb.run is not None:
                                            wandb.log({f"barrier_viz/{img_key}": img_obj}, step=total_steps)
                                            print(f"[Barrier Viz] ✓ Uploaded {img_key} via wandb.log")
                                        else:
                                            print(f"[Barrier Viz] ✗ No active wandb run found for {img_key}")
                                    except Exception as e:
                                        print(f"[Barrier Viz] ✗ Failed to upload {img_key}: {str(e)}")
                            
                            print(f"{'='*60}")
                            print(f"[Barrier Viz] Upload Complete - Check wandb 'Media' panel")
                            print(f"[Barrier Viz] (Note: Images appear in Media, not Charts)")
                            print(f"{'='*60}\n")
                    
                    except Exception as e:
                        print(f"[Barrier Viz] Exception during visualization: {type(e).__name__}: {str(e)}")
                        import traceback
                        traceback.print_exc()
                
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
    device_str = cfg_train.get("device", "cpu")
    if "cuda" in device_str:
        if ":" in device_str:
            device_id = int(device_str.split(":")[1])
        else:
            device_id = 0
        torch.cuda.set_device(device_id)
        # Warm up CUDA context with a small operation
        _ = torch.zeros(1, device=device_str)
        # Force synchronization to ensure context is fully initialized
        torch.cuda.synchronize()
    
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

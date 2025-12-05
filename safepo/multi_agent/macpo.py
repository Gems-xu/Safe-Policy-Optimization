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


import copy
import numpy as np
try: 
    import isaacgym
except:
    pass
import torch
import torch.nn as nn
import os
import sys
import time
from tqdm import tqdm

from safepo.common.env import make_ma_mujoco_env, make_ma_isaac_env, make_ma_multi_goal_env
from safepo.common.popart import PopArt
from safepo.common.model import MultiAgentActor as Actor, MultiAgentCritic as Critic
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

class MACPO_Policy():

    def __init__(self, config, obs_space, cent_obs_space, act_space):
        self.config = config
        self.obs_space = obs_space
        self.act_space = act_space
        self.share_obs_space = cent_obs_space

        self.actor = Actor(config, obs_space, act_space, self.config["device"])
        self.critic = Critic(config, cent_obs_space, self.config["device"])
        self.cost_critic = Critic(config, cent_obs_space, self.config["device"])

        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=self.config["actor_lr"], eps=self.config["opti_eps"], weight_decay=self.config["weight_decay"]
            )
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=self.config["critic_lr"], eps=self.config["opti_eps"], weight_decay=self.config["weight_decay"]
            )
        self.cost_optimizer = torch.optim.Adam(
            self.cost_critic.parameters(), lr=self.config["critic_lr"], eps=self.config["opti_eps"], weight_decay=self.config["weight_decay"]
            )

    def get_actions(self, cent_obs, obs, rnn_states_actor, rnn_states_critic, masks, available_actions=None,
                    deterministic=False, rnn_states_cost=None):
        actions, action_log_probs, rnn_states_actor = self.actor(obs, rnn_states_actor, masks, available_actions, deterministic)

        values, rnn_states_critic = self.critic(cent_obs, rnn_states_critic, masks)
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
        action_log_probs, dist_entropy, action_mu, action_std \
            = self.actor.evaluate_actions(obs, rnn_states_actor, action, masks, available_actions, active_masks)
        values, _ = self.critic(cent_obs, rnn_states_critic, masks)
        cost_values, _ = self.cost_critic(cent_obs, rnn_states_cost, masks)
        return values, action_log_probs, dist_entropy, cost_values, action_mu, action_std

    def act(self, obs, rnn_states_actor, masks, available_actions=None, deterministic=False):
        actions, _, rnn_states_actor = self.actor(obs, rnn_states_actor, masks, available_actions, deterministic)
        return actions, rnn_states_actor

class MACPO_Trainer():

    def __init__(self, config, policy):
        
        self.policy = policy
        self.config = config

        self.value_normalizer = PopArt(1, device=self.config["device"])
        self.tpdv = dict(dtype=torch.float32, device=self.config["device"])

    def cal_value_loss(self, values, value_preds_batch, return_batch, active_masks_batch):
        value_pred_clipped = value_preds_batch + (values - value_preds_batch).clamp(-self.config["clip_param"],
                                                                                    self.config["clip_param"])
        error_clipped = self.value_normalizer(return_batch) - value_pred_clipped
        error_original = self.value_normalizer(return_batch) - values

        value_loss_clipped = huber_loss(error_clipped, self.config["huber_delta"])
        value_loss_original = huber_loss(error_original, self.config["huber_delta"])

        value_loss = torch.max(value_loss_original, value_loss_clipped)

        return value_loss.mean()

    def flat_grad(self, grads):
        grad_flatten = []
        for grad in grads:
            if grad is None:
                continue
            grad_flatten.append(grad.view(-1))
        grad_flatten = torch.cat(grad_flatten)
        return grad_flatten

    def flat_hessian(self, hessians):
        hessians_flatten = []
        for hessian in hessians:
            if hessian is None:
                continue
            hessians_flatten.append(hessian.contiguous().view(-1))
        hessians_flatten = torch.cat(hessians_flatten).data
        return hessians_flatten

    def flat_params(self, model):
        params = []
        for param in model.parameters():
            params.append(param.data.view(-1))
        params_flatten = torch.cat(params)
        return params_flatten

    def update_model(self, model, new_params):
        index = 0
        for params in model.parameters():
            params_length = len(params.view(-1))
            new_param = new_params[index: index + params_length]
            new_param = new_param.view(params.size())
            params.data.copy_(new_param)
            index += params_length

    def kl_divergence(self, obs, rnn_states, action, masks, available_actions, active_masks, new_actor, old_actor):

        _, _, mu, std = new_actor.evaluate_actions(obs, rnn_states, action, masks, available_actions, active_masks)
        _, _, mu_old, std_old = old_actor.evaluate_actions(obs, rnn_states, action, masks, available_actions,
                                                           active_masks)
        logstd = torch.log(std)
        mu_old = mu_old.detach()
        std_old = std_old.detach()
        logstd_old = torch.log(std_old)

        kl = logstd_old - logstd + (std_old.pow(2) + (mu_old - mu).pow(2)) / \
             (1e-8 + 2.0 * std.pow(2)) - 0.5

        return kl.sum(1, keepdim=True)

    def conjugate_gradient(self, actor, obs, rnn_states, action, masks, available_actions, active_masks, b, nsteps,
                           residual_tol=1e-10):
        x = torch.zeros(b.size()).to(device=self.config["device"])
        r = b.clone()
        p = b.clone()
        rdotr = torch.dot(r, r)
        for _ in range(nsteps):
            _Avp = self.fisher_vector_product(actor, obs, rnn_states, action, masks, available_actions, active_masks, p)
            alpha = rdotr / (torch.dot(p, _Avp)+1e-8)
            x += alpha * p
            r -= alpha * _Avp
            new_rdotr = torch.dot(r, r)
            betta = new_rdotr / rdotr
            p = r + betta * p
            rdotr = new_rdotr
            if rdotr < residual_tol:
                break
        return x

    def fisher_vector_product(self, actor, obs, rnn_states, action, masks, available_actions, active_masks, p):
        p.detach()
        kl = self.kl_divergence(obs, rnn_states, action, masks, available_actions, active_masks, new_actor=actor,
                                old_actor=actor).mean()
        kl_grad = torch.autograd.grad(kl, actor.parameters(), create_graph=True, allow_unused=True)
        kl_grad = self.flat_grad(kl_grad)

        kl_grad_p = (kl_grad * p).sum()
        kl_hessian_p = torch.autograd.grad(kl_grad_p, actor.parameters(), allow_unused=True)
        kl_hessian_p = self.flat_hessian(kl_hessian_p)

        return kl_hessian_p + 0.1 * p

    def trpo_update(self, sample):
        
        share_obs_batch, obs_batch, rnn_states_batch, rnn_states_critic_batch, actions_batch, \
        value_preds_batch, return_batch, masks_batch, active_masks_batch, old_action_log_probs_batch, \
        adv_targ, available_actions_batch, factor_batch, cost_preds_batch, cost_returns_barch, rnn_states_cost_batch, \
        cost_adv_targ, aver_episode_costs = sample

        old_action_log_probs_batch, adv_targ, value_preds_batch, return_batch, active_masks_batch, factor_batch, \
        cost_returns_barch, cost_preds_batch, cost_adv_targ = [
            check(x).to(**self.tpdv) for x in [
                old_action_log_probs_batch, adv_targ, value_preds_batch, return_batch, active_masks_batch, factor_batch, \
                    cost_returns_barch, cost_preds_batch, cost_adv_targ
                    ]
        ]

        values, action_log_probs, dist_entropy, cost_values, action_mu, action_std = self.policy.evaluate_actions(
            share_obs_batch, obs_batch, rnn_states_batch, rnn_states_critic_batch, actions_batch, \
            masks_batch, available_actions_batch, active_masks_batch, rnn_states_cost_batch
            )
            
        value_loss = self.cal_value_loss(values, value_preds_batch, return_batch, active_masks_batch)
        self.policy.critic_optimizer.zero_grad()
        (value_loss * self.config["value_loss_coef"]).backward()
        critic_grad_norm = nn.utils.clip_grad_norm_(self.policy.critic.parameters(), self.config["max_grad_norm"])
        self.policy.critic_optimizer.step()

        cost_loss = self.cal_value_loss(cost_values, cost_preds_batch, cost_returns_barch, active_masks_batch)
        self.policy.cost_optimizer.zero_grad()
        (cost_loss * self.config["value_loss_coef"]).backward()
        cost_grad_norm = nn.utils.clip_grad_norm_(self.policy.cost_critic.parameters(), self.config["max_grad_norm"])

        self.policy.cost_optimizer.step()


        rescale_constraint_val = (aver_episode_costs.mean() - self.config["cost_limit"]) * (1 - self.config["gamma"])

        if rescale_constraint_val == 0:
            rescale_constraint_val = 1e-8

        ratio = torch.exp(action_log_probs - old_action_log_probs_batch)
        ratio = torch.prod(ratio, dim=-1, keepdim=True)

        reward_loss = torch.sum(ratio * factor_batch * adv_targ, dim=-1, keepdim=True).mean()
        reward_loss = - reward_loss
        reward_loss_grad = torch.autograd.grad(reward_loss, self.policy.actor.parameters(), retain_graph=True,
                                               allow_unused=True)
        reward_loss_grad = self.flat_grad(reward_loss_grad)

        cost_loss = torch.sum(ratio * factor_batch * (cost_adv_targ), dim=-1, keepdim=True).mean()
        cost_loss_grad = torch.autograd.grad(cost_loss, self.policy.actor.parameters(), retain_graph=True,
                                             allow_unused=True)
        cost_loss_grad = self.flat_grad(cost_loss_grad)
        B_cost_loss_grad = cost_loss_grad.unsqueeze(0)
        B_cost_loss_grad = self.flat_grad(B_cost_loss_grad)

        g_step_dir = self.conjugate_gradient(
            self.policy.actor, obs_batch, rnn_states_batch, actions_batch, masks_batch,\
            available_actions_batch, active_masks_batch, reward_loss_grad.data, nsteps=self.config["conjugate_gradient_iters"]
        )  
        b_step_dir = self.conjugate_gradient(
            self.policy.actor, obs_batch, rnn_states_batch, actions_batch, masks_batch,\
            available_actions_batch, active_masks_batch, B_cost_loss_grad.data, nsteps=self.config["conjugate_gradient_iters"]
        )  

        q_coef = (reward_loss_grad * g_step_dir).sum(0, keepdim=True)  
        r_coef = (reward_loss_grad * b_step_dir).sum(0, keepdim=True)  
        s_coef = (cost_loss_grad * b_step_dir).sum(0, keepdim=True)  

        fraction = self.config["step_fraction"] 
        loss_improve = 0

        B_cost_loss_grad_dot = torch.dot(B_cost_loss_grad, B_cost_loss_grad)
        if (torch.dot(B_cost_loss_grad, B_cost_loss_grad)) <= 1e-8 and rescale_constraint_val < 0:
            b_step_dir = torch.tensor(0)
            r_coef = torch.tensor(0)
            s_coef = torch.tensor(0)
            positive_Cauchy_value = torch.tensor(0)
            whether_recover_policy_value = torch.tensor(0)
            optim_case = 4
        else:
            r_coef = (reward_loss_grad * b_step_dir).sum(0, keepdim=True)  
            s_coef = (cost_loss_grad * b_step_dir).sum(0, keepdim=True)  
            if r_coef == 0:
                r_coef = 1e-8
            if s_coef == 0:
                s_coef = 1e-8
            positive_Cauchy_value = (
                        q_coef - (r_coef ** 2) / (1e-8 + s_coef))  
            whether_recover_policy_value = 2 * self.config["target_kl"] - (
                    rescale_constraint_val ** 2) / (
                                                       1e-8 + s_coef)
            if rescale_constraint_val < 0 and whether_recover_policy_value < 0:
                optim_case = 3
            elif rescale_constraint_val < 0 and whether_recover_policy_value >= 0:
                optim_case = 2
            elif rescale_constraint_val >= 0 and whether_recover_policy_value >= 0:
                optim_case = 1
            else:
                optim_case = 0
        if whether_recover_policy_value == 0:
            whether_recover_policy_value = 1e-8

        if optim_case in [3, 4]:
            lam = torch.sqrt(
                (q_coef / (2 * self.config["target_kl"])))
            nu = torch.tensor(0)  # v_coef = 0
        elif optim_case in [1, 2]:
            LA, LB = [0, r_coef / rescale_constraint_val], [r_coef / rescale_constraint_val, np.inf]
            LA, LB = (LA, LB) if rescale_constraint_val < 0 else (LB, LA)
            proj = lambda x, L: max(L[0], min(L[1], x))
            lam_a = proj(torch.sqrt(positive_Cauchy_value / whether_recover_policy_value), LA)
            lam_b = proj(torch.sqrt(q_coef / (torch.tensor(2 * self.config["target_kl"]))), LB)

            f_a = lambda lam: -0.5 * (positive_Cauchy_value / (
                        1e-8 + lam) + whether_recover_policy_value * lam) - r_coef * rescale_constraint_val / (
                                          1e-8 + s_coef)
            f_b = lambda lam: -0.5 * (q_coef / (1e-8 + lam) + 2 * self.config["target_kl"] * lam)
            lam = lam_a if f_a(lam_a) >= f_b(lam_b) else lam_b
            nu = max(0, lam * rescale_constraint_val - r_coef) / (1e-8 + s_coef)
        else:
            lam = torch.tensor(0)
            nu = torch.sqrt(torch.tensor(2 * self.config["target_kl"]) / (1e-8 + s_coef))

        x_a = (1. / (lam + 1e-8)) * (g_step_dir + nu * b_step_dir)
        x_b = (nu * b_step_dir)
        x = x_a if optim_case > 0 else x_b

        reward_loss = reward_loss.detach()
        cost_loss = cost_loss.detach()
        params = self.flat_params(self.policy.actor)

        old_actor = Actor(self.policy.config,
                            self.policy.obs_space,
                            self.policy.act_space,
                            self.config["device"])
        self.update_model(old_actor, params)

        expected_improve = -torch.dot(x, reward_loss_grad).sum(0, keepdim=True)
        expected_improve = expected_improve.detach()

        flag = False
        fraction_coef = self.config["fraction_coef"]
        for i in range(self.config["searching_steps"]):
            x_norm = torch.norm(x)
            if x_norm > 0.5:
                x = x * 0.5 / x_norm

            new_params = params - fraction_coef * (fraction**i) * x
            self.update_model(self.policy.actor, new_params)
            values, action_log_probs, dist_entropy, new_cost_values, action_mu, action_std = self.policy.evaluate_actions(
                share_obs_batch, obs_batch, rnn_states_batch, rnn_states_critic_batch,\
                actions_batch, masks_batch, available_actions_batch, active_masks_batch, rnn_states_cost_batch
            )

            ratio = torch.exp(action_log_probs - old_action_log_probs_batch)
            ratio = torch.prod(ratio, dim=-1, keepdim=True)

            new_reward_loss = torch.sum(ratio * factor_batch * adv_targ, dim=-1, keepdim=True).mean()
            new_cost_loss = torch.sum(ratio * factor_batch * cost_adv_targ, dim=-1, keepdim=True).mean()

            new_reward_loss = new_reward_loss.detach()
            new_reward_loss = -new_reward_loss
            new_cost_loss = new_cost_loss.detach()
            loss_improve = new_reward_loss - reward_loss

            kl = self.kl_divergence(
                obs_batch, rnn_states_batch, actions_batch, masks_batch,\
                available_actions_batch, active_masks_batch, new_actor=self.policy.actor, old_actor=old_actor
            ).mean()

            if ((kl < self.config["target_kl"]) and (loss_improve < 0 if optim_case > 1 else True)
                    and (new_cost_loss.mean() - cost_loss.mean() <= max(-rescale_constraint_val, 0))):
                flag = True
                break
            expected_improve *= fraction

        if not flag:
            params = self.flat_params(old_actor)
            self.update_model(self.policy.actor, params)

        return value_loss, critic_grad_norm, kl, loss_improve, expected_improve, dist_entropy, ratio, cost_loss, cost_grad_norm, whether_recover_policy_value, cost_preds_batch, cost_returns_barch, B_cost_loss_grad, lam, nu, g_step_dir, b_step_dir, x, action_mu, action_std, B_cost_loss_grad_dot

    def train(self, buffer, logger, agent_id):
        advantages = buffer.returns[:-1] - self.value_normalizer.denormalize(buffer.value_preds[:-1])

        advantages_copy = advantages.clone()
        mean_advantages = torch.mean(advantages_copy)
        std_advantages = torch.std(advantages_copy)
        advantages = (advantages - mean_advantages) / (std_advantages + 1e-5)

        cost_adv = buffer.cost_returns[:-1] - self.value_normalizer.denormalize(buffer.cost_preds[:-1])

        cost_adv_copy = cost_adv.clone()
        mean_cost_adv = cost_adv_copy.mean()
        std_cost_adv = cost_adv_copy.std()
        cost_adv = (cost_adv - mean_cost_adv) / (std_cost_adv + 1e-5)

        # Accumulate losses across mini-batches
        train_info = {
            "value_loss": 0,
            "cost_loss": 0,
            "loss_improve": 0,
            "expected_improve": 0,
            "critic_grad_norm": 0,
            "cost_grad_norm": 0,
            "dist_entropy": 0,
            "imp_weights": 0,
            "kl": 0,
        }
        num_updates = 0
        
        data_generator = buffer.feed_forward_generator(advantages, self.config["num_mini_batch"], cost_adv=cost_adv)
        for sample in data_generator:
            value_loss, critic_grad_norm, kl, loss_improve, expected_improve, dist_entropy, imp_weights, cost_loss, cost_grad_norm, whether_recover_policy_value, cost_preds_batch, cost_returns_barch, B_cost_loss_grad, lam, nu, g_step_dir, b_step_dir, x, action_mu, action_std, B_cost_loss_grad_dot \
                = self.trpo_update(sample)
            
            # Accumulate losses
            train_info["value_loss"] += value_loss.item()
            train_info["cost_loss"] += cost_loss.item()
            train_info["loss_improve"] += loss_improve.item()
            train_info["expected_improve"] += expected_improve.item()
            train_info["critic_grad_norm"] += critic_grad_norm.item()
            train_info["cost_grad_norm"] += cost_grad_norm.item()
            train_info["dist_entropy"] += dist_entropy.item()
            train_info["imp_weights"] += imp_weights.detach().mean().item()
            train_info["kl"] += kl.detach().item()
            num_updates += 1
        
        # Average over mini-batches
        for k in train_info:
            train_info[k] /= num_updates
        
        # Return train_info for aggregation across agents (don't log per-agent metrics)
        return train_info

    def prep_training(self):
        self.policy.actor.train()
        self.policy.critic.train()
        self.policy.cost_critic.train()

    def prep_rollout(self):
        self.policy.actor.eval()
        self.policy.critic.eval()
        self.policy.cost_critic.eval()


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
        self.policy = []
        for agent_id in range(self.num_agents):
            share_observation_space = self.envs.share_observation_space[agent_id]
            po = MACPO_Policy(config,
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
            tr = MACPO_Trainer(config, self.policy[agent_id])
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
        step = 0
        pbar = tqdm(range(episodes), desc="Training", ncols=100)
        for episode in pbar:

            done_episodes_rewards = []
            done_episodes_costs = []
            episode_steps_list = []  # Track steps for each completed episode

            for env_step in range(self.config["episode_length"]):
                # Sample actions
                values, actions, action_log_probs, rnn_states, rnn_states_critic, cost_preds, \
                rnn_states_cost = self.collect(env_step)
                obs, share_obs, rewards, costs, dones, infos, _ = self.envs.step(actions)

                dones_env = torch.all(dones, dim=1)

                reward_env = torch.mean(rewards, dim=1).flatten()
                cost_env = torch.mean(costs, dim=1).flatten()

                train_episode_rewards += reward_env
                train_episode_costs += cost_env
                step += self.config["n_rollout_threads"]

                for t in range(self.config["n_rollout_threads"]):
                    if dones_env[t]:
                        done_episodes_rewards.append(train_episode_rewards[:, t].clone())
                        done_episodes_costs.append(train_episode_costs[:, t].clone())
                        episode_steps_list.append(step)
                        train_episode_rewards[:, t] = 0
                        train_episode_costs[:, t] = 0

                done_episodes_costs_aver = train_episode_costs.mean()
                data = obs, share_obs, rewards, costs, dones, infos, \
                       values, actions, action_log_probs, \
                       rnn_states, rnn_states_critic, cost_preds, rnn_states_cost, done_episodes_costs_aver

                self.insert(data)
            self.compute()
            self.train()

            if (episode % self.config["save_interval"] == 0 or episode == episodes - 1):
                self.save()
                
            end = time.time()
            
            # Evaluate if needed
            should_eval = episode % self.config["eval_interval"] == 0 and self.config["use_eval"]
            if should_eval:
                eval_rewards, eval_costs = self.eval(eval_episodes=1, total_steps=step)

            # Store each completed episode separately with its corresponding step
            for ep_ret, ep_cost, ep_steps in zip(done_episodes_rewards, done_episodes_costs, episode_steps_list):
                self.return_aver_cost(ep_cost)
                self.logger.store(
                    **{
                        "Metrics/EpRet": ep_ret.item(),
                        "Metrics/EpCost": ep_cost.item(),
                    }
                )
            
            if len(done_episodes_rewards) != 0:
                # Log training metrics
                self.logger.log_tabular("Metrics/EpRet", min_and_max=True, std=True)
                self.logger.log_tabular("Metrics/EpCost", min_and_max=True, std=True)
                # Only log eval metrics when eval was actually performed
                if should_eval:
                    self.logger.log_tabular("Eval/EpRet", eval_rewards)
                    self.logger.log_tabular("Eval/EpCost", eval_costs)
                    # self.logger.log_tabular("Eval/render_max_reward", self.render_max_reward)
                self.logger.log_tabular("Train/Epoch", episode)
                self.logger.log_tabular("Train/Step", step)
                self.logger.log_tabular("Loss/Loss_reward_critic")
                self.logger.log_tabular("Loss/Loss_cost_critic")
                self.logger.log_tabular("Loss/Loss_actor_improve")
                self.logger.log_tabular("Loss/Loss_actor_expected_improve")
                self.logger.log_tabular("Misc/Reward_critic_norm")
                self.logger.log_tabular("Misc/Cost_critic_norm")
                self.logger.log_tabular("Misc/Entropy")
                self.logger.log_tabular("Misc/Ratio")
                self.logger.log_tabular("Misc/KL")
                self.logger.log_tabular("Time/Total", end - start)
                self.logger.log_tabular("Time/FPS", int(step / (end - start)))
                self.logger.dump_tabular(step=step)
                
                # Update tqdm progress bar with key metrics (use mean)
                aver_episode_rewards = torch.stack(done_episodes_rewards).mean()
                aver_episode_costs = torch.stack(done_episodes_costs).mean()
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
        cost_preds_collector = []
        rnn_states_cost_collector = []

        for agent_id in range(self.num_agents):
            self.trainer[agent_id].prep_rollout()
            value, action, action_log_prob, rnn_state, rnn_state_critic, cost_pred, rnn_state_cost \
                = self.trainer[agent_id].policy.get_actions(self.buffer[agent_id].share_obs[step],
                                                            self.buffer[agent_id].obs[step],
                                                            self.buffer[agent_id].rnn_states[step],
                                                            self.buffer[agent_id].rnn_states_critic[step],
                                                            self.buffer[agent_id].masks[step],
                                                            rnn_states_cost=self.buffer[agent_id].rnn_states_cost[step])
            value_collector.append(value.detach())
            action_collector.append(action.detach())
            action_log_prob_collector.append(action_log_prob.detach())
            rnn_state_collector.append(rnn_state.detach())
            rnn_state_critic_collector.append(rnn_state_critic.detach())
            cost_preds_collector.append(cost_pred.detach())
            rnn_states_cost_collector.append(rnn_state_cost.detach())
        if self.config["env_name"] == "Safety9|8HumanoidVelocity-v0":
            zeros = torch.zeros(action_collector[-1].shape[0], 1)
            action_collector[-1]=torch.cat((action_collector[-1], zeros), dim=1)
        values = torch.transpose(torch.stack(value_collector), 1, 0)
        rnn_states = torch.transpose(torch.stack(rnn_state_collector), 1, 0)
        rnn_states_critic = torch.transpose(torch.stack(rnn_state_critic_collector), 1, 0)
        cost_preds = torch.transpose(torch.stack(cost_preds_collector), 1, 0)
        rnn_states_cost = torch.transpose(torch.stack(rnn_states_cost_collector), 1, 0)

        return values, action_collector, action_log_prob_collector, rnn_states, rnn_states_critic, cost_preds, rnn_states_cost

    def insert(self, data, aver_episode_costs=0):
        aver_episode_costs = aver_episode_costs
        obs, share_obs, rewards, costs, dones, infos, \
        values, actions, action_log_probs, rnn_states, rnn_states_critic, cost_preds, rnn_states_cost, done_episodes_costs_aver  = data

        dones_env = torch.all(dones, axis=1)

        rnn_states[dones_env == True] = torch.zeros(
            (dones_env == True).sum(), self.num_agents, self.config["recurrent_N"], self.config["hidden_size"], device=self.config["device"])
        rnn_states_critic[dones_env == True] = torch.zeros(
            (dones_env == True).sum(), self.num_agents, *self.buffer[0].rnn_states_critic.shape[2:], device=self.config["device"])
        rnn_states_cost[dones_env == True] = torch.zeros(
            ((dones_env == True).sum(), self.num_agents, *self.buffer[0].rnn_states_cost.shape[2:]), device=self.config["device"])

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
                                         active_masks[:, agent_id], None, costs=costs[:, agent_id].unsqueeze(-1),
                                         cost_preds=cost_preds[:, agent_id],
                                         rnn_states_cost=rnn_states_cost[:, agent_id], done_episodes_costs_aver=done_episodes_costs_aver, aver_episode_costs=aver_episode_costs)

    def train(self):
        action_dim = 1
        factor = torch.ones(self.config["episode_length"], self.config["n_rollout_threads"], action_dim, device=self.config["device"])

        # Accumulate training info across agents
        avg_train_info = {
            "value_loss": 0,
            "cost_loss": 0,
            "loss_improve": 0,
            "expected_improve": 0,
            "critic_grad_norm": 0,
            "cost_grad_norm": 0,
            "dist_entropy": 0,
            "imp_weights": 0,
            "kl": 0,
        }
        
        for agent_id in torch.randperm(self.num_agents):
            action_dim=self.buffer[agent_id].actions.shape[-1]

            self.trainer[agent_id].prep_training()
            self.buffer[agent_id].update_factor(factor)
            available_actions = None if self.buffer[agent_id].available_actions is None \
                else self.buffer[agent_id].available_actions[:-1].reshape(-1, *self.buffer[agent_id].available_actions.shape[2:])

            old_actions_logprob, _, _, _ = self.trainer[agent_id].policy.actor.evaluate_actions(
                self.buffer[agent_id].obs[:-1].reshape(-1, *self.buffer[agent_id].obs.shape[2:]),
                self.buffer[agent_id].rnn_states[0:1].reshape(-1, *self.buffer[agent_id].rnn_states.shape[2:]),
                self.buffer[agent_id].actions.reshape(-1, *self.buffer[agent_id].actions.shape[2:]),
                self.buffer[agent_id].masks[:-1].reshape(-1, *self.buffer[agent_id].masks.shape[2:]),
                available_actions,
                self.buffer[agent_id].active_masks[:-1].reshape(-1, *self.buffer[agent_id].active_masks.shape[2:]))
            
            train_info = self.trainer[agent_id].train(self.buffer[agent_id], logger=self.logger, agent_id=agent_id)
            
            # Accumulate for average
            for k in avg_train_info:
                avg_train_info[k] += train_info[k]

            new_actions_logprob, _, _, _ = self.trainer[agent_id].policy.actor.evaluate_actions(
                self.buffer[agent_id].obs[:-1].reshape(-1, *self.buffer[agent_id].obs.shape[2:]),
                self.buffer[agent_id].rnn_states[0:1].reshape(-1, *self.buffer[agent_id].rnn_states.shape[2:]),
                self.buffer[agent_id].actions.reshape(-1, *self.buffer[agent_id].actions.shape[2:]),
                self.buffer[agent_id].masks[:-1].reshape(-1, *self.buffer[agent_id].masks.shape[2:]),
                available_actions,
                self.buffer[agent_id].active_masks[:-1].reshape(-1, *self.buffer[agent_id].active_masks.shape[2:]))

            action_prod = torch.prod(torch.exp(new_actions_logprob.detach()-old_actions_logprob.detach()).reshape(self.config["episode_length"],self.config["n_rollout_threads"],action_dim), dim=-1, keepdim=True)
            factor = factor*action_prod.detach()
            self.buffer[agent_id].after_update()
        
        # Store average losses across all agents
        for k in avg_train_info:
            avg_train_info[k] /= self.num_agents
        
        self.logger.store(
            add_value=False,
            **{
                "Loss/Loss_reward_critic": avg_train_info["value_loss"],
                "Loss/Loss_cost_critic": avg_train_info["cost_loss"],
                "Loss/Loss_actor_improve": avg_train_info["loss_improve"],
                "Loss/Loss_actor_expected_improve": avg_train_info["expected_improve"],
                "Misc/Reward_critic_norm": avg_train_info["critic_grad_norm"],
                "Misc/Cost_critic_norm": avg_train_info["cost_grad_norm"],
                "Misc/Entropy": avg_train_info["dist_entropy"],
                "Misc/Ratio": avg_train_info["imp_weights"],
                "Misc/KL": avg_train_info["kl"],
            }
        )

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
    def eval(self, eval_episodes=1, total_steps=None):
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
                (eval_dones_env == True).sum(), 
                self.num_agents, 
                self.config["recurrent_N"], 
                self.config["hidden_size"], 
                device=self.config["device"])

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

            next_costs = self.trainer[agent_id].policy.get_cost_values(self.buffer[agent_id].share_obs[-1],
                                                                       self.buffer[agent_id].rnn_states_cost[-1],
                                                                       self.buffer[agent_id].masks[-1])
            next_costs = next_costs.detach()
            self.buffer[agent_id].compute_cost_returns(next_costs, self.trainer[agent_id].value_normalizer)

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
    args, cfg_env, cfg_train = multi_agent_args(algo="macpo")
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

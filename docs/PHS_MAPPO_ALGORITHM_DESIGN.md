# PHS-MAPPO 算法设计说明文档

## Port-Hamiltonian System Embedded Multi-Agent Proximal Policy Optimization

**版本**: v8.1  
**作者**: Gems Team  
**日期**: 2026年1月

---

## 目录

1. [算法概述](#1-算法概述)
2. [Port-Hamiltonian 系统理论基础](#2-port-hamiltonian-系统理论基础)
3. [Barrier 势函数设计](#3-barrier-势函数设计)
4. [PHS Actor 网络架构](#4-phs-actor-网络架构)
5. [与 MAPPO 的结合](#5-与-mappo-的结合)
6. [训练流程](#6-训练流程)
7. [版本演进历史](#7-版本演进历史)

---

## 1. 算法概述

### 1.1 核心思想

PHS-MAPPO 是一种将 **Port-Hamiltonian 系统 (PHS)** 物理结构嵌入到多智能体强化学习中的创新算法。其核心思想是：

- **物理先验 (Physics Prior)**: 利用 Hamiltonian 力学中的能量守恒和耗散结构，为策略网络提供物理归纳偏置
- **安全约束 (Safety Constraints)**: 通过 Barrier 势函数实现障碍物规避的软约束
- **多智能体协调 (Multi-Agent Coordination)**: 使用 Laplacian 矩阵建模智能体间的耦合关系

### 1.2 关键创新

```
传统 MAPPO:      obs → MLP → action
PHS-MAPPO v8.1:  obs → [state_encoder + PHS_features] → policy_net → action
```

**v8.1 架构创新**:
1. **完全可学习策略网络**: `policy_net` 是完整的 MLP，具有完全的梯度流
2. **PHS 作为输入特征**: PHS 梯度作为物理信息输入，而非输出混合
3. **标准 RL 优化**: 保持与标准 PPO 相同的优化流程

### 1.3 架构对比

| 组件 | 标准 MAPPO | PHS-MAPPO v8.1 |
|------|------------|----------------|
| 状态编码 | MLP | state_encoder (145K 参数) |
| 策略网络 | MLP (110K) | policy_net (462K 参数) |
| 物理信息 | 无 | PHS 梯度特征 |
| 总参数 | ~130K | ~890K |

---

## 2. Port-Hamiltonian 系统理论基础

### 2.1 PHS 基本方程

Port-Hamiltonian 系统描述了能量在系统中的流动和耗散：

$$
\dot{x} = (J - R) \nabla H(x) + F u
$$

其中：
- $x \in \mathbb{R}^n$: 系统状态 (本项目中 $x = [v_x, v_y, a_x, a_y]$)
- $H(x)$: Hamiltonian 函数 (总能量)
- $J \in \mathbb{R}^{n \times n}$: **互连矩阵** (反对称，$J = -J^T$)
- $R \in \mathbb{R}^{n \times n}$: **耗散矩阵** (对称正半定，$R = R^T \succeq 0$)
- $F \in \mathbb{R}^{n \times m}$: 控制输入矩阵
- $u \in \mathbb{R}^m$: 控制输入 (动作)

### 2.2 能量分解

总 Hamiltonian 分解为多个势函数：

$$
H_{total} = \underbrace{H_{goal} + H_{task\_learned}}_{H_{task}} + H_{kin} + H_{barrier\_obs} + H_{barrier\_agent}
$$

| 势函数 | 物理意义 | 计算方式 |
|--------|----------|----------|
| $H_{goal}$ | 目标吸引势 | $\frac{1}{2} k_{goal} \cdot d_{goal}^2$ |
| $H_{task\_learned}$ | 学习到的任务势 | 神经网络 `H_task_net` |
| $H_{kin}$ | 动能 | $\frac{1}{2} \|v\|^2$ |
| $H_{barrier\_obs}$ | 障碍物排斥势 | 可学习指数 Barrier |
| $H_{barrier\_agent}$ | 智能体间排斥势 | 对数 Barrier |

### 2.3 基础系统矩阵

对于 2D 运动 ($dim = 2$)，系统矩阵定义为：

**互连矩阵 J** (标准 Hamiltonian 形式):
$$
J_{sys} = \begin{bmatrix} 0 & I_2 \\ -I_2 & 0 \end{bmatrix}
$$

**耗散矩阵 R** (阻尼):
$$
R_{sys} = \begin{bmatrix} 0 & 0 \\ 0 & \gamma I_2 \end{bmatrix}
$$

其中 $\gamma$ 是阻尼系数 (默认 0.1)。

---

## 3. Barrier 势函数设计

### 3.1 障碍物 Barrier (可学习指数 Barrier)

**v5.1 版本创新**: 使用可学习的指数 Barrier 结构

$$
H_{barrier\_obs} = k(s) \cdot \text{scale} \cdot \frac{\exp(\alpha(s) \cdot p_{shifted}) - 1}{\exp(\alpha(s)) - 1}
$$

其中：
- $p_{shifted} = \max(0, \frac{p_{max} - \tau(s)}{1 - \tau(s)})$：移位后的危险度
- $p_{max} = \max(\text{hazard\_lidar})$：最大障碍物接近度
- $k(s)$：**可学习刚度** (由 `obstacle_k_net` 预测)
- $\alpha(s)$：**可学习形状参数** (由 `barrier_shape_net` 预测)
- $\tau(s)$：**可学习激活阈值** (由 `barrier_shape_net` 预测)

**网络结构**:

```python
# 刚度网络
obstacle_k_net = Sequential(
    Linear(obs_dim, hidden//2), ELU(),
    Linear(hidden//2, 1), Softplus()
)

# 形状网络
barrier_shape_net = Sequential(
    Linear(obs_dim, hidden//4), ELU(),
    Linear(hidden//4, 2), Tanh()  # 输出 [alpha_mod, threshold_mod]
)
```

**参数范围**:
- 基础 $\alpha = 4.0$，可调范围 $[2.8, 5.2]$
- 基础阈值 $\tau = 0.75$，可调范围 $[0.65, 0.85]$
- 刚度 $k \in [0.5, 1.5]$

### 3.2 智能体间 Barrier (对数 Barrier)

使用 `SoftBarrierHead` 学习成对刚度系数：

$$
H_{barrier\_agent} = -\sum_{i<j} k_{ij}(s_i, s_j) \cdot \log(d_{ij} + \epsilon)
$$

**SoftBarrierHead 结构**:

```python
class SoftBarrierHead(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        # 共享状态编码器
        self.mlp_shared = Sequential(
            Linear(input_dim, hidden_dim), ELU(),
            Linear(hidden_dim, hidden_dim//2), ELU()
        )
        
        # 成对刚度预测器
        self.mlp_k = Sequential(
            Linear(hidden_dim, hidden_dim//2), ELU(),
            Linear(hidden_dim//2, 1)
        )
        
        # 可学习平滑参数
        self.log_smoothness = Parameter(tensor(0.0))
```

**计算流程**:
1. 编码每个智能体状态: $z_i = \text{mlp\_shared}(s_i)$
2. 成对展开: $z_{ij} = [z_i; z_j]$
3. 预测刚度: $k_{ij} = \text{softplus}(\text{mlp\_k}(z_{ij})) \cdot \text{smoothness}$
4. 邻接掩码: $k_{ij} = k_{ij} \cdot \text{adj}_{ij}$

### 3.3 Barrier 权重调度

**v7.4 策略**: 无预热，立即激活

```python
def _get_current_barrier_weight(self):
    step = self._training_step
    
    if step < barrier_decay_start:  # Phase 1: 全强度
        return barrier_weight_max  # 默认 1.5
    else:  # Phase 2: 缓慢衰减
        decay_factor = exp(-(step - barrier_decay_start) / 2000.0)
        target = barrier_weight * barrier_decay_rate
        return target + (barrier_weight_max - target) * decay_factor
```

**理由**: 早期训练中不使用预热，可以防止智能体学到碰撞行为。

---

## 4. PHS Actor 网络架构

### 4.1 整体架构 (v8.1)

```
输入 obs [batch, obs_dim]
       ↓
┌──────────────────────────────────────┐
│         Feature Normalization         │
└──────────────────────────────────────┘
       ↓
┌──────────────────────────────────────┐
│     State Encoder (4层 MLP)          │
│     obs_dim → 256 → 256 → 256 → 128  │
│     参数: 145,024                     │
└──────────────────────────────────────┘
       ↓
   state_features [batch, 128]
       ↓
┌──────────────────────────────────────┐
│       Policy Network (4层 MLP)        │ ← 主策略网络
│     128 → 512 → 512 → 256 → 2        │
│     参数: 462,594                     │
└──────────────────────────────────────┘
       ↓
   policy_output [batch, 2]
       ↓
┌──────────────────────────────────────┐
│     Residual MLP + 探索噪声           │
│     参数: 16,770                      │
└──────────────────────────────────────┘
       ↓
   u_body [forward, turn]
       ↓
┌──────────────────────────────────────┐
│    Differential Drive Conversion     │ (仅 Car 智能体)
│    [forward, turn] → [left, right]   │
└──────────────────────────────────────┘
       ↓
   action [batch, 2]
```

### 4.2 关键模块

**状态编码器 (State Encoder)**:
```python
state_encoder = Sequential(
    Linear(obs_dim, 256), ELU(),
    Linear(256, 256), ELU(),
    Linear(256, 256), ELU(),
    Linear(256, physics_hidden)  # 128
)
```

**策略网络 (Policy Net) - v8.1 核心**:
```python
policy_net = Sequential(
    Linear(physics_hidden, 512), ELU(),
    Linear(512, 512), ELU(),
    Linear(512, 256), ELU(),
    Linear(256, act_dim)  # 2
)
```

**标准差网络 (Std Net)**:
```python
std_net = Sequential(
    Linear(physics_hidden + act_dim, hidden//2), ELU(),
    Linear(hidden//2, act_dim)
)
```

### 4.3 智能体类型检测

```python
def detect_agent_type(config, obs_dim):
    env_name = config.get("env_name", "")
    
    if "Car" in env_name:
        return "car", 24  # base_sensor_dim = 24
    elif "Point" in env_name:
        return "point", 12  # base_sensor_dim = 12
    else:
        # 基于观测维度推断
        # Car: 176 dims, Point: 152 dims (MultiGoal Level1)
        if obs_dim >= 170:
            return "car", 24
        else:
            return "point", 12
```

### 4.4 差分驱动转换 (Car 专用)

Car 智能体使用差分驱动，需要将 `[forward, turn]` 转换为 `[left_wheel, right_wheel]`:

```python
def differential_drive_conversion(u_body):
    forward = u_body[:, 0:1]
    turn = u_body[:, 1:2]
    
    turn_mix = 0.6  # 转向混合系数
    left_wheel = forward + turn_mix * turn
    right_wheel = forward - turn_mix * turn
    
    return torch.cat([left_wheel, right_wheel], dim=-1)
```

**物理解释**:
- `left_wheel = right_wheel`: 直行
- `left_wheel > right_wheel`: 右转
- `left_wheel < right_wheel`: 左转

---

## 5. 与 MAPPO 的结合

### 5.1 Policy 类

```python
class MAPPOSafePINNv2Policy:
    def __init__(self, config, obs_space, cent_obs_space, act_space, 
                 n_agents=1, agent_id=0):
        # PHS-MAPPO Actor (替代标准 Actor)
        self.actor = PHSMAPPOActor(
            config, obs_space, act_space, device,
            n_agents=n_agents,
            agent_id=agent_id  # 每个智能体知道自己的 ID
        )
        
        # 标准 Critic (价值函数)
        self.critic = Critic(config, cent_obs_space, device)
        
        # Cost Critic (安全约束)
        self.cost_critic = Critic(config, cent_obs_space, device)
```

### 5.2 MAPPO-Lagrangian 混合优势

使用混合优势函数结合奖励和安全约束：

$$
A_{hybrid} = A_{reward} - \lambda \cdot A_{cost}
$$

其中 $\lambda$ 是 Lagrangian 乘子，自适应更新：

```python
def update_lagrangian(aver_episode_costs, cost_limit, lamda_lagr):
    cost_violation = aver_episode_costs - cost_limit
    
    if cost_violation > 2.0:
        # 超过限制，增加惩罚
        lamda_lagr += lagr_rate * cost_violation * 0.1
    elif cost_violation < -5.0:
        # 远低于限制，减少惩罚
        lamda_lagr -= lagr_rate * abs(cost_violation) * 0.05
    
    return clamp(lamda_lagr, lamda_lagr_min, lamda_lagr_max)
```

### 5.3 辅助物理损失

训练时添加轻量级辅助损失：

```python
aux_loss = (
    aux_task_potential_weight * task_potential_loss +      # 0.01
    aux_barrier_potential_weight * barrier_k_loss +        # 0.02
    aux_safety_weight * safety_loss +                      # 0.01
    aux_agent_collision_weight * agent_collision_loss      # 0.02
)
```

**损失定义**:
- `task_potential_loss`: $\text{MSE}(\sigma(H_{task}), 1 - p_{goal})$
- `barrier_k_loss`: $\text{MSE}(k, 0.3 + 2 \cdot p_{hazard})$
- `safety_loss`: $\text{MSE}(\sigma(H_{barrier}/5), \text{danger\_level})$

### 5.4 软 Cost 增强

在 buffer 插入时增强 cost 信号：

```python
# 使用 barrier potential 作为软 cost
H_barrier, _ = actor._compute_barrier_potential(obs)
soft_cost = sigmoid((H_barrier - 3.0) / 2.0)
augmented_cost = env_cost + soft_cost_weight * soft_cost
```

---

## 6. 训练流程

### 6.1 完整训练循环

```
1. Warmup: 重置环境，初始化 buffer

2. 数据收集 (Rollout):
   for step in episode_length:
       values, actions, log_probs, cost_preds = collect(step)
       obs, rewards, costs, dones = env.step(actions)
       insert_to_buffer(data)

3. 计算回报:
   compute_returns(reward_critic)
   compute_cost_returns(cost_critic)

4. 训练更新:
   for _ in learning_iters:
       for batch in data_generator:
           # 计算混合优势
           adv_hybrid = adv_reward - λ * adv_cost
           
           # PPO 更新
           policy_loss = ppo_clip_loss(adv_hybrid)
           aux_loss = compute_aux_physics_loss(obs)
           
           # 更新 Actor
           actor_loss = policy_loss - entropy_coef * entropy + aux_loss
           actor_optimizer.step()
           
           # 更新 Critics
           critic_optimizer.step()
           cost_critic_optimizer.step()
           
           # 更新 Lagrangian
           update_lagrangian()
           
           # 更新 barrier warmup
           actor.set_training_step(step)

5. 评估与保存:
   if eval_interval:
       eval_rewards, eval_costs = eval()
       save_models()
```

### 6.2 关键超参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `actor_lr` | 9e-5 | Actor 学习率 |
| `critic_lr` | 5e-3 | Critic 学习率 |
| `entropy_coef` | 0.02 | 熵正则化系数 |
| `clip_param` | 0.2 | PPO 裁剪参数 |
| `learning_iters` | 5 | 每轮训练迭代次数 |
| `lamda_lagr` | 0.5 | Lagrangian 乘子初始值 |
| `barrier_weight_max` | 1.5 | Barrier 最大权重 |
| `cost_limit` | 25.0 | 每 episode 安全成本限制 |

---

## 7. 版本演进历史

### 7.1 版本对比

| 版本 | 核心特征 | 问题 |
|------|----------|------|
| v7.x | PHS 梯度直接输出 | **梯度被 detach**，只有 4 个可学习参数 |
| v8.0 | Policy Net + PHS 混合 | 混合公式复杂，优化困难 |
| **v8.1** | **纯 Policy Net，PHS 作为特征** | ✅ 梯度流畅，标准 RL 优化 |

### 7.2 v7.x 的致命缺陷

```python
# v7.x 问题代码
goal_grad = goal_grad.detach()  # ❌ 梯度被截断！
barrier_grad = barrier_grad.detach()

action = k_goal * goal_grad + k_barrier * barrier_grad
# 只有 k_goal, k_barrier 能学习 (4 个参数)
```

### 7.3 v8.1 的修复

```python
# v8.1 正确实现
policy_output = self.policy_net(state_features)  # 完整 MLP
residual = self.residual_mlp(state_features)
residual_w = sigmoid(self.residual_weight) * 0.3

u_body = policy_output + residual_w * residual  # 直接输出，无混合
```

**梯度流验证**:
```
policy_net gradient magnitude: 19,591.78 ✅
```

---

## 附录：参数量统计

```
=== v8.1 Network Structure ===
state_encoder: 145,024 params
policy_net:    462,594 params  ← 主要可学习参数
phs_gain_net:   17,282 params  (legacy, 未使用)
residual_mlp:   16,770 params
std_net:        17,026 params
weight params:       2 params

Total active params: 658,698
```

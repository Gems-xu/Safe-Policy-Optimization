# SafetyCarMultiGoal1-v0 环境设计说明文档

## 多智能体安全导航环境

**版本**: v1.0  
**基于**: Safety Gymnasium  
**日期**: 2026年1月

---

## 目录

1. [环境概述](#1-环境概述)
2. [任务设计](#2-任务设计)
3. [Car 智能体详解](#3-car-智能体详解)
4. [观测空间](#4-观测空间)
5. [动作空间](#5-动作空间)
6. [奖励与安全约束](#6-奖励与安全约束)
7. [多智能体扩展](#7-多智能体扩展)
8. [环境配置](#8-环境配置)

---

## 1. 环境概述

### 1.1 任务描述

`SafetyCarMultiGoal1-v0` 是一个多智能体安全强化学习环境，其中**两个 Car 智能体**需要：

1. **导航到各自的目标**: Agent 0 → 红色目标 (Goal Red)，Agent 1 → 蓝色目标 (Goal Blue)
2. **避开障碍物**: 8 个 Hazards (接触产生 cost)
3. **避免碰撞**: 智能体之间不能相撞

### 1.2 环境层级

| 层级 | 名称 | 障碍物配置 |
|------|------|-----------|
| Level 0 | MultiGoalLevel0 | 仅目标，无障碍 |
| **Level 1** | **MultiGoalLevel1** | **8 个 Hazards + 1 个 Vase (无 cost)** |
| Level 2 | MultiGoalLevel2 | 10 个 Hazards + 10 个 Vases (有 cost) |

### 1.3 支持的智能体类型

```
SafetyPointMultiGoal{0,1,2}-v0   # Point 机器人
SafetyCarMultiGoal{0,1,2}-v0    # Car 机器人
SafetyRacecarMultiGoal{0,1,2}-v0 # Racecar
SafetyDoggoMultiGoal{0,1,2}-v0  # 四足机器人
SafetyAntMultiGoal{0,1,2}-v0    # Ant 机器人
```

---

## 2. 任务设计

### 2.1 空间布局

```
        ┌─────────────────────────────┐
        │                             │
        │    ●    (hazard)    ●      │
        │         ★ Goal Red         │
        │    ●              ●        │
        │                             │
        │    ●    🚗 Car 0  ●        │
        │              🚗 Car 1       │
        │    ●              ●        │
        │         ★ Goal Blue        │
        │    ●              ●        │
        │                             │
        └─────────────────────────────┘
        
        范围: [-1.5, 1.5] × [-1.5, 1.5]
```

### 2.2 Level 1 配置

```python
class MultiGoalLevel1(MultiGoalLevel0):
    def __init__(self, config):
        super().__init__(config)
        
        # 场景范围
        self.placements_conf.extents = [-1.5, -1.5, 1.5, 1.5]
        
        # 添加 8 个危险区域
        self._add_geoms(Hazards(num=8, keepout=0.18))
        
        # 添加 1 个花瓶 (无 cost)
        self._add_free_geoms(Vases(num=1, is_constrained=False))
        
        # 碰撞成本
        self.contact_other_cost = 1.0
```

### 2.3 目标设计

**Goal Red (Agent 0 的目标)**:
- 颜色: 红色
- keepout: 0.305 (与其他物体的最小距离)
- reward_distance: 1.0 (距离奖励系数)
- reward_goal: 1.0 (到达目标奖励)
- size: 目标半径，用于判断是否到达

**Goal Blue (Agent 1 的目标)**:
- 颜色: 蓝色
- 其他参数与 Goal Red 相同

### 2.4 障碍物设计

**Hazards (危险区域)**:
- 数量: 8
- keepout: 0.18 (放置时的安全距离)
- 接触成本: 1.0 (每次接触产生 cost=1)
- 视觉: 红色圆柱

**Vases (花瓶)** (Level 1):
- 数量: 1
- is_constrained: False (无 cost)
- 可被推动

---

## 3. Car 智能体详解

### 3.1 物理结构

Car 是一个具有**差分驱动 (Differential Drive)** 的三轮机器人：

```
         ┌───────────┐
         │           │
    ◉────┤  Car Body ├────◉   ← 两个独立驱动轮
         │           │
         └─────┬─────┘
               │
               ◎              ← 后部自由滚轮 (ball joint)
```

**关键特性**:
- **两个独立驱动轮**: 左轮和右轮可以独立控制扭矩
- **自由滚动后轮**: 通过 ball joint 连接，提供稳定性
- **非完整约束**: 不能侧向移动，只能通过差分驱动转向

### 3.2 运动学模型

**差分驱动方程**:

$$
v = \frac{v_L + v_R}{2}
$$

$$
\omega = \frac{v_R - v_L}{W}
$$

其中：
- $v$: 线速度
- $\omega$: 角速度
- $v_L, v_R$: 左右轮速度
- $W$: 轮距

**动作到运动的映射**:
```python
# 给定动作 [left_torque, right_torque]
# 相等扭矩 → 直行
# left > right → 右转
# left < right → 左转
```

### 3.3 传感器配置

Car 智能体配备以下传感器：

| 传感器 | 维度 | 说明 |
|--------|------|------|
| accelerometer | 3 | 线加速度 (x, y, z) |
| velocimeter | 3 | 线速度 (x, y, z) |
| gyro | 3 | 角速度 (roll, pitch, yaw) |
| magnetometer | 3 | 朝向 (cos θ, sin θ, 0) |
| **ballangvel_rear** | 3 | 后轮 ball joint 角速度 |
| **ballquat_rear** | 9 | 后轮 ball joint 四元数 (3x3 展开) |

**Car 特有传感器**:
- `ballangvel_rear`: 后部自由轮的角速度，反映车辆转向状态
- `ballquat_rear`: 后部自由轮的姿态四元数，展开为 9 维

**基础传感器维度对比**:
```
Point: accelerometer(3) + velocimeter(3) + gyro(3) + magnetometer(3) = 12
Car:   12 + ballangvel_rear(3) + ballquat_rear(9) = 24
```

### 3.4 Lidar 配置

每个智能体配备多组 Lidar 传感器：

| Lidar 类型 | Bins | 范围 | 用途 |
|------------|------|------|------|
| goal_red_lidar | 16 | 360° | Agent 0 探测红色目标 |
| goal_blue_lidar | 16 | 360° | Agent 1 探测蓝色目标 |
| hazards_lidar | 16 | 360° | 探测危险区域 |
| vases_lidar | 16 | 360° | 探测花瓶 |
| agents_lidar | 16 | 360° | 探测其他智能体 |

**Lidar 工作原理**:
- 每个 Lidar 有 16 个 bins，均匀分布在 360°
- bin 0 指向智能体前方，逆时针增加
- 值范围 [0, 1]：0 表示无检测，1 表示接触

---

## 4. 观测空间

### 4.1 Car 观测结构 (176 维)

```
观测向量结构 (多智能体扩展后):

索引范围      内容                    维度
────────────────────────────────────────────
[0:3]        accelerometer           3
[3:6]        velocimeter             3
[6:9]        gyro                    3
[9:12]       magnetometer            3
[12:15]      ballangvel_rear         3    ← Car 特有
[15:24]      ballquat_rear           9    ← Car 特有
────────────────────────────────────────────
[24:40]      goal_red_lidar          16   ← Agent 0 主目标
[40:56]      goal_blue_lidar         16   ← Agent 1 主目标
[56:72]      hazards_lidar           16
[72:88]      vases_lidar             16
[88:104]     agents_lidar            16   ← 其他智能体位置
────────────────────────────────────────────
[104:120]    accelerometer1          3    ← 扩展传感器
[120:136]    velocimeter1            3
[136:152]    gyro1                   3
[152:168]    magnetometer1           3
[168:176]    其他/填充               8
────────────────────────────────────────────
总计:                                176 维
```

### 4.2 观测归一化

环境包装器对观测进行 Z-score 归一化：

```python
def _get_obs(self):
    obs_n = []
    for agent in self.possible_agents:
        obs = self._last_obs_dict[agent]
        # Z-score 归一化
        obs_normed = (obs - np.mean(obs)) / (np.std(obs) + 1e-8)
        obs_n.append(obs_normed.astype(np.float32))
    return obs_n
```

**注意**: 归一化后 Lidar 值可能为负（低于均值）或正（高于均值）。

### 4.3 共享观测 (Centralized Critic)

用于集中式 Critic 的共享观测：

```python
share_obs_size = obs_size * num_agents  # 176 * 2 = 352

def _get_share_obs(self):
    # 拼接所有智能体的观测
    all_obs = [self._last_obs_dict[agent] for agent in self.possible_agents]
    concat_obs = np.concatenate(all_obs)  # [352]
    concat_obs_normed = (concat_obs - np.mean(concat_obs)) / (np.std(concat_obs) + 1e-8)
    
    # 每个智能体获得相同的共享观测
    return [concat_obs_normed] * self.num_agents
```

---

## 5. 动作空间

### 5.1 Car 动作空间

```python
action_space = Box(low=-1.0, high=1.0, shape=(2,), dtype=float64)
```

| 动作索引 | 含义 | 范围 |
|----------|------|------|
| action[0] | 左轮扭矩 | [-1, 1] |
| action[1] | 右轮扭矩 | [-1, 1] |

### 5.2 动作效果

```python
# 直行
action = [1.0, 1.0]   # 前进
action = [-1.0, -1.0] # 后退

# 原地转向
action = [1.0, -1.0]  # 顺时针旋转
action = [-1.0, 1.0]  # 逆时针旋转

# 转弯行驶
action = [1.0, 0.5]   # 左转前进
action = [0.5, 1.0]   # 右转前进
```

### 5.3 与 Point 的对比

| 智能体 | 动作含义 | 控制方式 |
|--------|----------|----------|
| Point | [forward, turn] | 直接速度控制 |
| **Car** | **[left_wheel, right_wheel]** | **差分驱动** |

**PHS-MAPPO 的处理**:
```python
# PHS 计算的是 body frame 动作 [forward, turn]
# 需要转换为 Car 的差分驱动 [left, right]

forward = u_body[:, 0:1]
turn = u_body[:, 1:2]

turn_mix = 0.6
left_wheel = forward + turn_mix * turn
right_wheel = forward - turn_mix * turn
```

---

## 6. 奖励与安全约束

### 6.1 奖励函数

**每步奖励**:
$$
r_t = (d_{t-1} - d_t) \cdot \text{reward\_distance}
$$

其中 $d_t$ 是智能体到其目标的距离。

**到达目标奖励**:
$$
r_{goal} = \text{reward\_goal} = 1.0
$$

**典型奖励量级**:
- 每步距离奖励: ~0.002 (非常稀疏)
- 静止不动: 0
- 到达目标: +1.0

### 6.2 安全成本 (Cost)

**碰撞成本**:
```python
cost = 0.0

# 接触 Hazard
if contact_hazard:
    cost += contact_hazard_cost  # 1.0

# 接触其他智能体
if contact_other_agent:
    cost += contact_other_cost   # 1.0

# Level 2: 接触 Vase
if level == 2 and contact_vase:
    cost += vase_cost            # 1.0
```

**成本限制**:
- 典型限制: `cost_limit = 25.0` 每 episode
- PHS-MAPPO 使用 Lagrangian 方法约束累积成本

### 6.3 Episode 终止条件

```python
# 时间截止
if step >= max_episode_steps:
    truncated = True

# 两个目标都到达 (可选)
if goal_achieved[0] and goal_achieved[1]:
    terminated = True
```

---

## 7. 多智能体扩展

### 7.1 PettingZoo 接口

环境使用 PettingZoo 风格的多智能体接口：

```python
# 智能体标识
possible_agents = ['agent_0', 'agent_1']

# 观测和动作空间按智能体索引
observation_space = {
    'agent_0': Box(..., shape=(176,)),
    'agent_1': Box(..., shape=(176,))
}

action_space = {
    'agent_0': Box(-1, 1, shape=(2,)),
    'agent_1': Box(-1, 1, shape=(2,))
}
```

### 7.2 MultiGoalEnv 包装器

`MultiGoalEnv` 类将 PettingZoo 接口转换为 MAPPO 所需的格式：

```python
class MultiGoalEnv:
    def reset(self):
        obs_dict, _ = self.env.reset()
        return (
            self._get_obs(),           # [obs_agent0, obs_agent1]
            self._get_share_obs(),     # [share_obs] * 2
            self._get_avail_actions()  # [[1,1], [1,1]]
        )
    
    def step(self, actions):
        # actions: [action_agent0, action_agent1]
        action_dict = {
            'agent_0': actions[0],
            'agent_1': actions[1]
        }
        
        obs, rewards, costs, terms, truncs, infos = self.env.step(action_dict)
        
        return (
            self._get_obs(),
            self._get_share_obs(),
            [rewards['agent_0'], rewards['agent_1']],
            [costs['agent_0'], costs['agent_1']],
            [terms['agent_0'] or truncs['agent_0'], ...],
            infos,
            self._get_avail_actions()
        )
```

### 7.3 向量化环境

使用 `ShareDummyVecEnv` 或 `ShareSubprocVecEnv` 并行化：

```python
# 数据形状
obs.shape = [n_envs, n_agents, obs_dim]        # [8, 2, 176]
share_obs.shape = [n_envs, n_agents, share_dim] # [8, 2, 352]
rewards.shape = [n_envs, n_agents]              # [8, 2]
costs.shape = [n_envs, n_agents]                # [8, 2]
dones.shape = [n_envs, n_agents]                # [8, 2]

# 动作格式 (step 输入)
actions = [action_agent0, action_agent1]
# 其中 action_agent_i.shape = [n_envs, act_dim]
```

### 7.4 智能体协调

**目标分配**:
- Agent 0 → Goal Red (固定分配)
- Agent 1 → Goal Blue (固定分配)

**Lidar 分配**:
- Agent 0 使用 `goal_red_lidar` 作为主要导航信息
- Agent 1 使用 `goal_blue_lidar` 作为主要导航信息
- 两者都可以看到 `agents_lidar` 来避免相互碰撞

**PHS-MAPPO 中的处理**:
```python
# 每个智能体有自己的 actor
actor_0 = PHSMAPPOActor(..., agent_id=0)  # 使用 goal_red_lidar
actor_1 = PHSMAPPOActor(..., agent_id=1)  # 使用 goal_blue_lidar

# Lidar 索引根据 agent_id 确定
if agent_id == 0:
    goal_lidar_start = 24   # goal_red
    goal_lidar_end = 40
else:
    goal_lidar_start = 40   # goal_blue
    goal_lidar_end = 56
```

---

## 8. 环境配置

### 8.1 创建环境

```python
from safepo.common.env import make_ma_multi_goal_env

cfg_train = {
    'n_rollout_threads': 8,      # 并行环境数
    'n_eval_rollout_threads': 1, # 评估环境数
    'device': 'cuda:0',
    'render_width': 1024,        # 渲染宽度
    'render_height': 1024,       # 渲染高度
    'camera_name': 'fixedfar',   # 俯视相机
}

env = make_ma_multi_goal_env(
    task='SafetyCarMultiGoal1-v0',
    seed=42,
    cfg_train=cfg_train
)
```

### 8.2 环境属性

```python
# 智能体数量
env.num_agents  # 2

# 观测空间
env.observation_space[0].shape  # (176,)

# 共享观测空间
env.share_observation_space[0].shape  # (352,)

# 动作空间
env.action_space[0].shape  # (2,)
env.action_space[0].low    # -1.0
env.action_space[0].high   # 1.0
```

### 8.3 训练配置示例

```yaml
# config.yaml
env_name: SafetyCarMultiGoal1-v0
algorithm_name: mappo_safe_pinn_v8
num_env_steps: 10000000
episode_length: 1000
n_rollout_threads: 20

# 网络配置
hidden_size: 256
physics_hidden: 128

# 安全约束
cost_limit: 25.0
lamda_lagr: 0.5
barrier_weight_max: 1.5

# Barrier 参数
r_collision: 0.17
barrier_epsilon: 0.06
```

### 8.4 渲染与可视化

```python
# 获取渲染帧
frame = env.render()  # numpy array [H, W, 3]

# 相机选项
camera_options = {
    'fixedfar': '俯视全局视角',
    'fixednear': '近距离俯视',
    'agent_0': '跟随 Agent 0',
    'agent_1': '跟随 Agent 1',
}
```

---

## 附录：维度速查表

### Car 观测维度 (176)

| 范围 | 传感器 | 维度 |
|------|--------|------|
| [0:12] | 基础传感器 | 12 |
| [12:24] | Ball joint | 12 |
| [24:40] | Goal Red Lidar | 16 |
| [40:56] | Goal Blue Lidar | 16 |
| [56:72] | Hazards Lidar | 16 |
| [72:88] | Vases Lidar | 16 |
| [88:104] | Agents Lidar | 16 |
| [104:176] | 扩展传感器 | 72 |

### Point 观测维度 (152)

| 范围 | 传感器 | 维度 |
|------|--------|------|
| [0:12] | 基础传感器 | 12 |
| [12:28] | Goal Red Lidar | 16 |
| [28:44] | Goal Blue Lidar | 16 |
| [44:60] | Hazards Lidar | 16 |
| [60:76] | Vases Lidar | 16 |
| [76:92] | Agents Lidar | 16 |
| [92:152] | 扩展传感器 | 60 |

---

## 参考资料

- [Safety Gymnasium 文档](https://www.safety-gymnasium.com/)
- [PettingZoo 多智能体 API](https://pettingzoo.farama.org/)
- [MuJoCo 物理引擎](https://mujoco.org/)

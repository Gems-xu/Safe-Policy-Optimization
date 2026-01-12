# MAPPO-Safe-PINN 算法框架说明

## 概述

MAPPO-Safe-PINN (Multi-Agent PPO with Safe Physics-Informed Neural Network) 是一种基于**端口哈密顿系统 (Port-Hamiltonian Systems, PHS)** 的多智能体安全强化学习算法。该算法将物理系统的能量概念引入强化学习，通过**势能函数**来实现目标导航和障碍物避让。

### 核心思想

```
智能体行为 = 任务势能引导 + 障碍势能排斥 + 耗散稳定
```

智能体如同一个"能量球"在势能场中运动：
- **任务势能 H_task**：在目标位置最低，引导智能体向目标移动
- **障碍势能 H_barrier**：在障碍物位置最高，排斥智能体远离危险

---

## 算法架构

### 1. 端口哈密顿动力学

系统动力学方程：
$$\dot{x} = (J(x) - R(x)) \nabla H_{total}(x)$$

其中：
- $H_{total} = H_{task} + H_{barrier}$：总势能
- $J$：反对称互联矩阵（产生垂直于梯度的力，用于绕行）
- $R$：正半定耗散矩阵（能量耗散，保证稳定性）
- $\nabla H_{total}$：总势能梯度

**物理意义**：
- $J$ 矩阵产生"陀螺力"，帮助智能体绕开障碍物而不是正面撞击
- $R$ 矩阵提供阻尼，防止智能体无限振荡

### 2. 势能函数设计

#### 2.1 任务势能 H_task（吸引子）

```python
H_task = NeuralNetwork(observation)  # 由神经网络学习
```

**目标**：学习一个势能场，使目标位置为全局最小值

$$\nabla H_{task} \rightarrow \text{指向目标的方向}$$

训练信号：
- 当智能体接近目标时，H_task 应该减小
- 由辅助损失函数监督：`task_potential_loss = H_task * goal_proximity`

#### 2.2 障碍势能 H_barrier（排斥子）— 核心创新

**v9.2: 指数型障碍势能**

```python
H_barrier = k * (exp(α * proximity) - 1)
```

其中：
- `proximity ∈ [0, 1]`：激光雷达读数，0=远，1=碰撞
- `α = 5.0`：指数增长率
- `k`：由神经网络学习的自适应刚度系数

**势能增长曲线**：

```
proximity | H_barrier
----------|----------
   0.0    |    0.00   (安全区)
   0.3    |    3.48
   0.5    |   11.18
   0.7    |   32.12   (警告区)
   0.9    |   89.02   (危险区)
   1.0    |  147.41   (碰撞边界)
```

**为什么用指数增长？**

1. **边界处柔和**：远离障碍物时，势能几乎为零，不影响正常导航
2. **中心处陡峭**：接近障碍物时，势能快速增大，产生强烈排斥力
3. **梯度清晰**：$\nabla H_{barrier} = k \cdot \alpha \cdot e^{\alpha \cdot p}$，梯度方向明确

**与其他方法对比**：

| 方法 | 公式 | 边界行为 | 中心行为 |
|------|------|----------|----------|
| 线性 | $k \cdot p$ | 弱 | 弱 |
| 二次 | $k \cdot p^2$ | 弱 | 中等 |
| 倒数 | $k/(1-p)$ | 弱 | 奇异点 |
| **指数** | $k(e^{\alpha p}-1)$ | **0** | **快速增大** |

---

## 防死锁机制

### 问题描述

当智能体被多个障碍物包围时，所有方向的排斥力可能相互抵消，导致智能体"卡住"不动。

### 解决方案：垂直逃逸方向

```python
# 检测是否被包围
high_readings = (lidar > 0.4).sum()
is_surrounded = (high_readings > 5)

# 计算垂直逃逸方向（旋转90度）
escape_x = -obstacle_dir_y
escape_y = obstacle_dir_x

# 混合正常避障和逃逸方向
final_dir = (1 - 0.4*is_surrounded) * obstacle_dir + 0.4*is_surrounded * escape_dir
```

**效果**：
- 正常情况：直接远离障碍物
- 被包围时：添加40%垂直分量，帮助智能体"滑过"障碍物间隙

### 走廊检测

当智能体在两个障碍物之间的通道中时，不应过度逃逸：

```python
# 检测前后是否畅通
front_clear = lidar[0:2].max() < 0.5
back_clear = lidar[7:9].max() < 0.5
in_corridor = front_clear and back_clear and high_readings > 2

# 走廊中减少逃逸混合
escape_blend *= (1 - 0.5 * in_corridor)
```

---

## 训练流程

### 1. PPO 策略更新

```python
advantage = reward_advantage - λ * cost_advantage
policy_loss = -min(ratio * advantage, clip(ratio) * advantage)
```

**Lagrangian 乘子 λ**：
- 当 cost > cost_limit 时增大，惩罚不安全行为
- 当 cost < cost_limit 时减小，允许更多探索

### 2. 物理辅助损失

```python
aux_loss = (
    w_task * task_potential_loss +      # H_task 应在目标处最小
    w_barrier * barrier_potential_loss + # H_barrier 应在障碍处最大
    w_safety * safety_loss +             # 安全违规惩罚
    w_agent * agent_collision_loss       # 智能体间碰撞惩罚
)
```

### 3. 软成本增强

环境的碰撞成本是**稀疏的**（只在碰撞时为1）。为帮助 Cost Critic 学习预测危险：

```python
soft_cost = sigmoid((H_barrier - 3) / 2)  # 基于障碍势能的连续成本
augmented_cost = env_cost + 0.1 * soft_cost
```

---

## 网络结构

```
观测 (152维)
├── 加速度计 [0:3]
├── 速度计 [3:6]      → 物理状态提取
├── 陀螺仪 [6:9]
├── 磁力计 [9:12]
├── 目标激光雷达 [12:44]   → H_task 计算
├── 障碍激光雷达 [44:60]   → H_barrier 计算
├── 花瓶激光雷达 [60:76]
└── 其他智能体激光雷达 [76:92] → 智能体避碰

                    ┌─────────────────┐
观测 ──────────────→│  特征提取网络    │
                    │  (MLP 256-256)  │
                    └────────┬────────┘
                             │
        ┌────────────────────┼────────────────────┐
        ↓                    ↓                    ↓
┌───────────────┐   ┌───────────────┐   ┌───────────────┐
│   H_task 网络  │   │  H_barrier 网络│   │   J/R 网络     │
│  (学习目标势能) │   │ (学习障碍刚度k) │   │ (学习物理矩阵)  │
└───────┬───────┘   └───────┬───────┘   └───────┬───────┘
        │                   │                   │
        └───────────────────┼───────────────────┘
                            ↓
                    ┌───────────────┐
                    │  策略集成层    │
                    │  (物理+特征)   │
                    └───────┬───────┘
                            ↓
                    ┌───────────────┐
                    │   动作输出     │
                    │  (前进, 转向)  │
                    └───────────────┘
```

---

## 关键参数 (v9.2)

| 参数 | 值 | 说明 |
|------|-----|------|
| `barrier_alpha` | 5.0 | 指数增长率 |
| `barrier_k_scale` | 2.0 | 刚度缩放因子 |
| `barrier_gradient_scale` | 2.0 | 梯度缩放因子 |
| `soft_cost_weight` | 0.1 | 软成本权重 |
| `lamda_lagr` | 0.5 | 初始 Lagrangian 乘子 |
| `lamda_lagr_max` | 5.0 | 最大 Lagrangian 乘子 |
| `aux_barrier_weight` | 0.02 | 障碍势能辅助损失权重 |

---

## 可视化

障碍势能场可视化（H_barrier 热力图）：

```
          高势能 ████████ 
                ████████
           障碍物 ██████
                ████████
          高势能 ████████
                  │
    智能体 ○ ←────┘ 被排斥远离
                  
          低势能区（安全）
```

---

## 算法优势

1. **物理可解释性**：势能函数有明确物理意义，易于理解和调试
2. **安全保证**：指数型障碍势能在碰撞边界提供强大排斥力
3. **避免死锁**：垂直逃逸机制帮助智能体穿过狭窄通道
4. **多智能体协调**：智能体间也产生障碍势能，自动避让
5. **Lagrangian 约束**：自适应调节安全-性能权衡

---

## 使用方法

```bash
# 训练
CUDA_VISIBLE_DEVICES=0 uv run safepo/multi_agent/mappo_safe_pinn.py \
    --task SafetyCarMultiGoal1-v0

# 评估（带渲染）
uv run safepo/multi_agent/mappo_safe_pinn.py \
    --task SafetyCarMultiGoal1-v0 \
    --model-dir runs/mappo_safe_pinn/xxx \
    --render
```

---

## 版本历史

| 版本 | 主要改进 |
|------|----------|
| v8.0 | 添加 Cost Critic |
| v8.6 | 软成本机制 |
| v9.0 | 简化架构，移除过度惩罚 |
| **v9.2** | **指数型障碍势能 + 防死锁机制** |

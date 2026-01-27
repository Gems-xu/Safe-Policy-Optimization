# Barrier PHS Actor 的奖励/成本学习机制说明

本文说明当前 `mappo_safe_pinn_v2` 中的 Barrier PHS Actor 如何同时利用 **reward** 与 **cost** 信号学习安全性的物理结构，并与 MAPPO-Lagrangian 主训练流程协同。

> 适用版本：PHS-MAPPO Actor v8.x（当前仓库实现）

---

## 1. 整体训练信号结构概览

Actor 的学习信号来自三条路径：

1. **PPO 主目标（Reward 方向）**
   - 基于优势函数 `adv`（由奖励回报和 reward critic 估计得到）。
   - 直接驱动策略网络提升任务回报。

2. **Lagrangian 约束（Cost 方向）**
   - 混合优势：`adv_hybrid = adv - λ * cost_adv`。
   - `λ` 由 cost 违规程度自适应调整，抑制高风险策略。

3. **Barrier PHS 辅助损失（结构性安全）**
   - 直接塑形 Barrier 物理势能的结构，使其与成本预测一致。

---

## 2. Actor 内部结构（与 Barrier 相关部分）

Actor 的动作由策略网络生成，但引入了 Barrier 相关物理结构：

- **Barrier 势能 `H_barrier`**：由 `hazard_lidar` 驱动的障碍势能。
- **Barrier 刚度 `k`**：由 `obstacle_k_net(obs)` 学习，决定障碍排斥强度。
- **Barrier 权重**：带 warmup → plateau → decay 的时间调度。

这意味着：
- 动作学习不直接依赖手工控制律；
- Barrier 网络通过损失与 cost 对齐进行“物理结构学习”。

---

## 3. Reward 信号如何影响 Barrier

Barrier 不直接由 reward 回传，但 reward 会通过 PPO 主目标影响策略输出，从而间接影响 Actor 对环境的分布与 Barrier 训练样本：

- 当策略更靠近目标、减少无意义运动时，`hazard_lidar` 分布变化；
- Barrier 的训练样本会逐步聚焦在有效轨迹附近。

因此：**reward 信号是“行为层”的主驱动，Barrier 的学习样本依赖 reward 导向的轨迹分布。**

---

## 4. Cost 信号如何驱动 Barrier 学习

当前实现中，Barrier 学习来自两类 cost 引导损失：

### 4.1 Barrier 势能与 Cost critic 对齐

Cost critic 预测 `cost_values`，用于监督 Barrier 势能大小：

- 目标：`H_barrier` 越大 → cost 越高
- 损失形式：

```
aux_cost_value_loss = MSE(sigmoid(H_barrier / 5), sigmoid(cost_values / cost_value_scale))
```

这使得 Barrier 势能逐渐拟合“cost 空间结构”。

---

### 4.2 Barrier 刚度 `k` 与 Cost critic 对齐

Barrier 刚度用于决定排斥强度，当前通过 cost 预测进行软监督：

```
aux_cost_k_loss = MSE(k, 0.3 + 2.0 * sigmoid(cost_values / cost_value_scale))
```

当 cost 预测高时，`k` 被推动变大，增强安全排斥能力。

---

## 5. Cost 与 Barrier 的协同效果

整体上，Barrier PHS 的“安全性学习”来自两个来源：

- **成本约束（Lagrangian）**：影响策略优化方向
- **成本对齐（Aux Loss）**：塑形 Barrier 势能结构

因此它具备双重监督：

| 信号来源 | 作用位置 | 作用效果 |
|---|---|---|
| cost_adv (Lagrangian) | PPO 策略目标 | 抑制高风险动作 |
| cost_values (Aux) | Barrier 势能/刚度 | 学习安全物理结构 |

---

## 6. 相关参数（可调）

当前 Barrier 学习相关超参：

- `aux_cost_value_weight`：cost 对齐势能的权重
- `aux_cost_k_weight`：cost 引导刚度的权重
- `cost_value_scale`：cost 预测归一化尺度
- `barrier_weight/barrier_weight_max`：Barrier 势能在总 Hamiltonian 中的权重
- `barrier_warmup_steps`：Barrier 逐步激活

如果成本仍偏高，可提高 `aux_cost_value_weight` 与 `aux_cost_k_weight`，或略增 `barrier_weight_max`。

---

## 7. 训练过程中的监控建议

建议重点监控以下指标：

- `Metrics/EpRet` 与 `Metrics/EpCost`
- `Loss/Aux_cost_value`
- `Loss/Aux_cost_k`
- `Safe/H_barrier_mean`
- `Safe/k_mean`

判断标准：
- `Aux_cost_value` 应逐渐下降；
- `EpCost` 下降但 `EpRet` 保持在可接受区间（10+）。

---

## 8. 总结

当前 Barrier PHS Actor 通过 **reward 引导轨迹分布**、**cost critic 对齐势能结构** 以及 **Lagrangian 约束策略** 三重机制学习安全性，最终实现“物理结构 + 强化学习”的协同优化。

如需更强安全性，可进一步提高 cost 对齐损失与 Barrier 权重；如需更高回报，可适当减弱这些项以释放探索空间。

# Barrier PHS Actor 的奖励/成本机制说明（固定 Barrier 结构）

本文说明当前 `mappo_safe_pinn_v2` 中的 Barrier PHS Actor 如何在 **固定 Barrier 结构** 的前提下，通过 reward/cost 信号与 MAPPO-Lagrangian 协同优化。

> 适用版本：PHS-MAPPO Actor v8.x（当前仓库实现）

---

## 1. 整体训练信号结构概览

Actor 的学习信号来自两条路径（Barrier 结构固定，不再学习）：

1. **PPO 主目标（Reward 方向）**
   - 基于优势函数 `adv`（由奖励回报和 reward critic 估计得到）。
   - 直接驱动策略网络提升任务回报。

2. **Lagrangian 约束（Cost 方向）**
   - 混合优势：`adv_hybrid = adv - λ * cost_adv`。
   - `λ` 由 cost 违规程度自适应调整，抑制高风险策略。

3. **Barrier PHS 固定结构（物理先验）**
   - 障碍势能不再学习，只作为动作生成的物理约束项。

---

## 2. Actor 内部结构（与 Barrier 相关部分）

Actor 的动作由 **Barrier PHS 约束端口** 生成，策略网络只提供期望状态变化：

- **Barrier 势能 `H_barrier`**：由 `hazard_lidar` 驱动的障碍势能（固定结构）。
- **Barrier 刚度 `k`**：固定常量（不再学习）。
- **Barrier 权重**：带 warmup → plateau → decay 的时间调度。

这意味着：
- 动作显式通过 Barrier PHS 结构生成；
- Barrier 结构作为物理先验参与动作生成，但不再学习。

---

## 3. Reward 信号如何影响 Barrier

Barrier 不直接由 reward 回传，但 reward 会通过 PPO 主目标影响策略输出，从而间接影响 Actor 对环境的分布与 Barrier 训练样本：

- 当策略更靠近目标、减少无意义运动时，`hazard_lidar` 分布变化；
- Barrier 的训练样本会逐步聚焦在有效轨迹附近。

因此：**reward 信号是“行为层”的主驱动，Barrier 的学习样本依赖 reward 导向的轨迹分布。**

---

## 4. Cost 信号如何影响策略（Barrier 固定）

由于 Barrier 结构固定，cost 信号不再用于 Barrier 结构学习，而是通过 Lagrange 约束影响策略：

- 混合优势：`adv_hybrid = adv - λ * cost_adv`
- 违规时提高 `λ`，抑制高风险策略

---

## 5. Cost 与 Barrier 的协同效果

整体上，Barrier PHS 的安全性来自两个来源：

- **成本约束（Lagrangian）**：影响策略优化方向
- **固定 Barrier 物理先验**：直接参与动作生成

---

## 6. 相关参数（可调）

当前 Barrier PHS 相关超参：

- `barrier_weight/barrier_weight_max`：Barrier 势能在总 Hamiltonian 中的权重
- `barrier_warmup_steps`：Barrier 逐步激活
- `obstacle_barrier_k/alpha/threshold/scale`：固定障碍势能结构
- `agent_barrier_k`：固定 agent-agent 势能刚度
- `phs_goal_guidance_weight/phs_barrier_guidance_weight`：动作生成中的目标/避障引导系数

如果成本仍偏高，可提高 `barrier_weight_max` 或 `agent_barrier_k`，并加快 `λ` 更新速率。

---

## 7. 训练过程中的监控建议

建议重点监控以下指标：

- `Metrics/EpRet` 与 `Metrics/EpCost`
- `Safe/H_barrier_mean`
- `Safe/k_mean`

判断标准：
- `EpCost` 下降但 `EpRet` 保持在可接受区间（10+）。

---

## 8. 总结

当前 Barrier PHS Actor 通过 **Barrier 固定结构** 与 **Lagrangian 约束策略** 形成安全约束，同时由 reward 驱动任务完成。

如需更强安全性，可提高 Barrier 权重或加快 `λ` 更新；如需更高回报，可适当减弱 Barrier 权重以释放探索空间。

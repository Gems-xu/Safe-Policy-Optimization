# MAPPO-Lagrange：reward/critic学习机制、与MAPPO差异、以及Barrier PHS融合方式

本文回答三个问题：
1) MAPPO-Lagrange 如何通过 reward 与 critic 学习？
2) 与标准 MAPPO 有何不同？
3) 如何将 Barrier PHS 融入 MAPPO-Lagrange 框架？

---

## 1. MAPPO-Lagrange 如何通过 reward 与 critic 学习

MAPPO-Lagrange 是在 MAPPO 基础上引入 **成本约束** 的多智能体 PPO 变体。它包含两个价值网络：

- **Reward Critic**：估计奖励回报 $V_r$。
- **Cost Critic**：估计成本回报 $V_c$。

学习流程概括为：

1. **Reward 优势**
   $$A_r = R_t - V_r$$
   其中 $R_t$ 为奖励回报（GAE或蒙特卡洛）。

2. **Cost 优势**
   $$A_c = C_t - V_c$$
   其中 $C_t$ 为成本回报。

3. **Lagrange 混合优势**
   $$A_{hybrid} = A_r - \lambda A_c$$

4. **策略更新目标**
   PPO 目标与 MAPPO 相同，但用 $A_{hybrid}$ 替代 $A_r$：
   $$\mathcal{L}_{policy} = \mathbb{E}[\min(r_t A_{hybrid}, \text{clip}(r_t)A_{hybrid})]$$

5. **Lagrange 乘子更新**
   $$\lambda \leftarrow \text{clip}(\lambda + \eta (\bar{C} - C_{limit}))$$
   若平均成本超标，$\lambda$ 增大，使策略更保守；反之减小。

因此，MAPPO-Lagrange 同时从 reward 与 cost 学习，并在策略目标中权衡二者。

---

## 2. 与标准 MAPPO 的关键差异

| 维度 | MAPPO | MAPPO-Lagrange |
|---|---|---|
| 价值网络 | 仅 Reward Critic | Reward Critic + Cost Critic |
| 优势函数 | $A_r$ | $A_r - \lambda A_c$ |
| 目标 | 最大化 reward | reward 与 cost 约束平衡 |
| 额外超参 | 无 | $C_{limit}, \lambda$ 更新率等 |
| 行为倾向 | 更激进探索 | 在安全约束下保守规划 |

简单理解：MAPPO 只优化回报；MAPPO-Lagrange 在“回报最大化 + 成本控制”之间动态权衡。

---

## 3. 如何将 Barrier PHS 融入 MAPPO-Lagrange 框架

Barrier PHS 的作用是将“安全结构”嵌入 Actor 里，从两条路径融合进 MAPPO-Lagrange：

### 3.1 结构性融入（Actor 内部）
Barrier PHS 提供安全势能与刚度结构，使策略具备“物理安全先验”。

- `H_barrier`：障碍势能，描述风险场
- `k`：势能刚度，控制排斥强度
- `barrier_weight`：势能权重，带 warmup/decay 调度

**Barrier PHS 可学习的关键参数与网络：**

- `obstacle_k_net(obs)`：输出刚度系数 `k`，控制排斥强度（可学习）。
- `barrier_shape_net(obs)`：输出形状参数（可学习），调节：
   - 激活阈值（危险开始生效的距离）
   - 斜率/陡峭度（势能增长快慢）
- `H_barrier` 的整体幅值由 `k` 与形状参数共同决定。

**可学习参数清单（核心）：**

1. 刚度网络权重（`obstacle_k_net`）
2. 形状网络权重（`barrier_shape_net`）
3. Barrier 相关辅助网络权重（如 SoftBarrierHead 用于 agent-agent 约束）

> 说明：`barrier_weight/barrier_weight_max` 是超参调度，不是网络参数。

Actor 的动作生成由策略网络主导，但 PHS 提供结构化安全信息。

---

### 3.2 学习性融入（Cost Critic 引导）
MAPPO-Lagrange 已有 cost critic，可直接用于训练 Barrier PHS：

- 让 Barrier 势能结构与 cost 预测对齐：
  $$\mathcal{L}_{cost\_value} = \text{MSE}(\sigma(H_{barrier}), \sigma(V_c))$$

- 让刚度 $k$ 随 cost 增大而增强：
  $$\mathcal{L}_{cost\_k} = \text{MSE}(k, 0.3 + 2\sigma(V_c))$$

这意味着 Barrier PHS 不仅依赖手工规则，也通过 **cost critic 学会安全结构**。

**完整训练流程（Barrier 相关）：**

1. 从环境采样得到 `obs`、`reward`、`cost`。
2. cost critic 预测 `V_c`（成本价值）。
3. 计算 `H_barrier` 与 `k`（由 `obstacle_k_net`/`barrier_shape_net` 输出）。
4. 通过 `L_cost_value` 与 `L_cost_k` 将 `H_barrier/k` 对齐到 `V_c`。
5. Barrier 辅助损失与 PPO 目标一起反传到 Actor（cost critic 仅作为监督信号）。

---

### 3.4 解耦后的 Barrier PHS：如何学习、如何影响 Actor

当前实现中引入**严格解耦**（Barrier 与 Lagrange 互不“夺权”），其核心变化如下：

**A. Barrier PHS 的学习方式（解耦版）**

- Barrier 相关网络（如 `obstacle_k_net`、`barrier_shape_net`、`H_barrier_head`）**冻结**，不再从 reward/cost critic 接收梯度。
- cost critic 信号仅作为**标量风险指示**用于 $
\lambda$ 的平滑更新或 Barrier 激活 gating（均 stop-gradient）。
- 因此，Barrier PHS 在解耦模式下**不再进行基于 cost 的结构学习**，只保留“物理先验”。

**B. Barrier PHS 如何影响 Actor（解耦版）**

- Barrier 仅以**无梯度安全先验**的形式影响动作：

$$u = u_{policy} + w_{phs} \cdot (-\nabla H_{barrier})$$

其中：
- $u_{policy}$ 来自策略网络；
- $-\nabla H_{barrier}$ 为 Hamiltonian 安全梯度（stop-gradient）；
- $w_{phs}$ 是可调的先验权重（如 `phs_prior_weight`）。

这确保：
- **Barrier 只“约束瞬时动作可行域”**，不参与 reward/cost 反传；
- **Lagrange 只“约束长期成本”**，不修改 Barrier 形状或刚度；
- 二者分工明确，避免安全信号之间相互竞争。

---

### 3.3 完整融合逻辑总结

1. **PPO 主目标**：用 $A_r - \lambda A_c$ 更新策略
2. **Cost Critic**：训练成本价值，驱动约束
3. **Barrier PHS**：
   - 作为 Actor 的结构先验
   - 通过 cost critic 对齐损失学习物理安全结构

这种融合使得：
- MAPPO-Lagrange 保证安全约束；
- Barrier PHS 强化安全结构表达；
- 策略在满足安全的前提下仍能追求高回报。

---

## 4. 结论

- MAPPO-Lagrange 的核心是 **双 critic + Lagrange 约束**；
- MAPPO 仅优化奖励，而 MAPPO-Lagrange 同时约束成本；
- Barrier PHS 可以通过 **Actor 结构嵌入 + cost critic 对齐损失** 无缝融入该框架。

如果需要更强安全性，提高 cost 对齐损失与 barrier 权重；若回报下降过多，降低这些权重以释放探索空间。

---

## 5. 当前实现中的 R 矩阵与 H_barrier 学习细节

### 5.1 R 矩阵（耗散矩阵）如何组成与学习？

- 代码中存在 `R_net`（AttentionLEMURS），用于输出扁平化的 $R$ 矩阵参数，理论上用于构造**对称正半定耗散矩阵**：
   $$R = A A^T + \epsilon I$$
- 但在**当前 v8.x 的实际策略前向路径中，动作由 policy_net 直接生成**，并未使用 $R$ 或 $J$ 来形成控制律。
- 也就是说：
   - `R_net` **存在但未参与当前 actor 的控制输出**；
   - 仅保留作为结构化物理模块的占位或兼容组件。

若需要让 $R$ 真正学习并进入控制律，需要恢复/加入基于 $J,R$ 的 PHS 动力学控制路径。

---

### 5.2 障碍势能 $H_{barrier}$ 的结构如何学习？

当前障碍势能由“刚度 + 形状参数”共同决定：

1. **刚度网络** `obstacle_k_net(obs)`：输出 $k$，控制势能幅值。
2. **形状网络** `barrier_shape_net(obs)`：输出 2 个形状参数，分别调节：
    - 激活阈值（危险距离起始点）
    - 势能陡峭度（增长速度）

完整结构：

$$H_{barrier} = k \cdot \text{scale} \cdot \frac{\exp(\alpha \cdot s) - 1}{\exp(\alpha) - 1}$$

其中 $s$ 为“阈值归一化后的危险度”。

**学习方式分两种模式：**

- **耦合模式（旧版）：**
   - 通过 cost critic 对齐损失更新 $k$ 与形状参数；
   - Barrier 会被“训练成”符合成本结构。

- **解耦模式（当前默认）：**
   - `obstacle_k_net` 与 `barrier_shape_net` **冻结**；
   - Barrier 作为“物理先验”，不再从 reward/cost 获取梯度；
   - 学习只发生在 policy_net，Barrier 仅提供安全梯度先验。

因此，是否“学习 H_barrier 结构”，取决于是否开启解耦。

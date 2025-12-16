**MAPPO-PINN 算法说明**

- **概述**: MAPPO-PINN 是将物理信息神经网络（PINN，基于端口-哈密顿系统 Port-Hamiltonian）嵌入到多智能体概率策略优化（MAPPO）中，使 actor 在遵循物理结构（能量守恒/耗散）的同时学习策略。该实现见 [safepo/multi_agent/mappo_pinn.py](safepo/multi_agent/mappo_pinn.py)。

**核心思想**
- **物理先验**: 使用 Hamiltonian H(x)、互连矩阵 J(x)（反对称）和耗散矩阵 R(x)（正半定）建模动力学，提取 (J - R) ∇H(x) 特征供策略网络使用。
- **混合表示**: 将 PINN 产生的物理特征（H, ∇H 等）与常规模型提取的特征拼接，输入到策略头以输出动作分布。
- **可解释性与稳定性**: 通过结构化约束（如 R=L L^T）保证耗散矩阵 PSD，从而提高数值稳定性与物理一致性。

**关键组件**
- **Actor (Port-Hamiltonian PINN)**:
  - **状态提取**: 从观测中抽取速度与加速度（示例中使用 obs 的 vx, vy 与 ax, ay）构成 physics state。
  - **H_net**: MLP 输出标量哈密顿量 H(x)。
  - **J_net**: 输出上三角元素，构造反对称矩阵 J。
  - **R_net**: 输出下三角 L 元素，构造 R = L L^T（保证正半定），并对对角作 softplus 以避免数值问题。
  - **动力学特征**: 计算 (J - R) ∇H 作为物理特征，连同 H 与 ∇H 一并输入策略整合层。
  - **动作输出**: 均值通过 MLP，方差为可学习 log_std，采用高斯分布（可选确定性输出）。
- **Critic**: 复用常规模型 `MultiAgentCritic`（见代码引用），并使用 PopArt 对价值进行归一化/反归一化。
- **Trainer (MAPPOPINNPointTrainer)**: 基于 PPO 的训练流程，包含策略/价值损失、剪切比率、熵正则化、梯度裁剪与 Adam 优化器。
- **Runner**: 负责环境交互、采样、buffer 管理、评估与模型保存（actor/critic 各自保存为 .pt）。同时集成视频录制与条件上传。

**PINN Actor 实现要点**
- **J 矩阵构造**: J_net 输出上三角（不含对角），在构造中同时写入负对称项以满足 J = -J^T。
- **R 矩阵构造**: R_net 输出 L 的下三角元素，先对对角做 softplus，再计算 R = L L^T，确保 PSD 和数值稳定。
- **哈密顿梯度**: 训练时使用 Autograd 计算 grad_H（create_graph=True），推理时可在 no_grad 模式下使用 enable_grad 临时计算并 detach。
- **数值技巧**: 对物理网络采用较小的初始化 gain（0.01）以提升训练稳定性；对 value 使用 PopArt；对梯度做 clip_norm。

**损失与训练流程**
- **策略损失**: PPO 剪切目标：surr = min(ratio * adv, clip(ratio) * adv)，求和/平均后取负作为策略损失。
- **价值损失**: 使用 Hubber 风格的损失（huber_loss）对 PopArt 归一化后的 returns 与 value 预测做截断后的误差比较，最后取两者较大值的均值。
- **熵项**: 使用动作分布的熵作为正则，按系数加回到策略目标中以促探索。
- **优化器**: Actor / Critic 分别用 Adam，超参来自 config（lr、eps、weight_decay），并在更新前做 zero_grad，更新后做梯度裁剪。
- **训练流程**: 按 MAPPO 的多智能体流程：warmup -> rollout 采样 -> buffer 计算 returns -> 多轮 mini-batches PPO 更新 -> 保存/评估。

**评估与保存**
- **保存**: 每个 agent 的 actor 与 critic 参数分别保存为 `actor_agent{ID}.pt` 与 `critic_agent{ID}.pt` 到日志目录下的 models 子目录。
- **评估**: Runner 提供 eval 接口，支持渲染帧集合并在提升全局最好回报时上传视频（集成 wandb）。

**常用配置要点 / 调优建议**
- **物理网络规模**: PINN 的隐藏层（`physics_hidden`）通常比策略主干小（示例为 64），方便训练稳定性。
- **初始化**: 对 H/J/R 网络使用较小的正交初始化 gain（示例中为 0.01），可防止初期过大梯度扰动。
- **状态选择**: 确认观测中速度/加速度的索引（示例使用 `[3,4]` 和 `[0,1]`），不同环境需调整 `vel_indices`/`acc_indices`。
- **方差参数化**: log_std 可设置/冻结为常数或训练；若不稳定，可先固定或缩小初始值。
- **数值稳定性**: R 的对角使用 softplus + 小常数，避免奇异矩阵导致动力学特征发散。

**运行示例**
- 直接通过脚本启动（脚本内使用 `multi_agent_args` 解析参数）：

```bash
python safepo/multi_agent/mappo_pinn.py --scenario <SCENARIO> --task <TASK> --seed 0
```

- 常见调参字段（位于 config/命令行解析后）：
  - **actor_lr / critic_lr**: 优化器学习率
  - **hidden_size**: 策略主干隐藏单元数
  - **physics_hidden**: PINN 隐藏单元数
  - **pinn_state_dim**: PINN 状态维度（示例 4）
  - **episode_length / n_rollout_threads**: 采样长度与并行环境数
  - **use_wandb / log_dir**: 日志与可视化配置

**参考实现**
- 主实现文件: [safepo/multi_agent/mappo_pinn.py](safepo/multi_agent/mappo_pinn.py)
- 本文档已保存为: [docs/mappo_pinn_summary.md](docs/mappo_pinn_summary.md)

---
简短结论：该实现通过把端口-哈密顿结构嵌入 actor，向策略网络提供物理一致性特征（H, ∇H, (J-R)∇H），有助于在动力学相关任务中提升鲁棒性与可解释性。若需要，我可以：
- 将文档扩展为中文+英文版对照；
- 在仓库中添加单元测试 / 简单示例训练脚本；
- 根据具体环境调整 `vel_indices`/`acc_indices` 并验证数值稳定性。

# 障碍势能可视化工具使用指南

## 概述

本工具用于可视化 **MAPPO-Safe-PINN** 算法学习到的障碍势能场和任务势能场。

### 理论背景

系统的总势能为：
```
H_total(x) = H_task(x;θ) + H_barrier(x;φ)
```

- **H_task**: 任务势能，在目标位置达到最小值（吸引器）
- **H_barrier**: 障碍势能，在障碍物附近趋向无穷大（排斥器）

详见 `Barrier_PHS.md`。

---

## 使用方法

### Python API

```python
from safepo.utils.visualize_barrier_potential import BarrierPotentialVisualizer

# 创建可视化器
visualizer = BarrierPotentialVisualizer(
    model_dir='runs/multi_goal/models_seed0',
    task='SafetyPointMultiGoal1-v0',
    agent_id=0,
    device='cpu'
)

# 生成2D热力图
visualizer.visualize_potential_2d(resolution=100, save_path='vizs')

# 生成3D曲面图
visualizer.visualize_potential_3d(resolution=50, potential_type='barrier', save_path='vizs')

# 生成所有可视化
visualizer.visualize_all(output_dir='vizs')
```

---

## 输出文件

可视化图片默认保存在 `vizs/` 目录:

```
vizs/
├── potential_fields_2d.png          # 2D热力图
├── potential_surface_3d_barrier.png # 3D障碍势能
├── potential_surface_3d_task.png    # 3D任务势能
├── potential_surface_3d_total.png   # 3D总势能
├── gradient_vector_field.png        # 梯度向量场
└── potential_slice_y0.0.png         # 1D剖面
```

---

## 如何解读结果

### ✅ 良好的障碍势能

- 障碍物/边界附近能量急剧升高（红色/热色）
- 安全区域能量低且平坦
- 梯度箭头指向远离障碍

### ✅ 良好的任务势能

- 目标位置能量最低（蓝色谷底）
- 远离目标能量逐渐升高
- 形成明确的吸引梯度

### ⚠️ 常见问题

**问题1: 障碍势能在目标位置也很高**
- 原因：网络混淆了目标激光雷达和障碍激光雷达
- 解决：调整训练配置，确保网络能区分目标和障碍

**问题2: 任务势能完全平坦**
- 原因：任务势能网络未学到有效梯度
- 解决：降低 `barrier_k_scale`，增加训练步数

---

## 训练中集成可视化

可视化已集成到 `mappo_safe_pinn.py` 训练代码中，在每次 eval 时自动生成并上传到 WandB：

```python
# 训练时自动可视化
python safepo/multi_agent/mappo_safe_pinn.py \
    --task SafetyCarMultiGoal1-v0 \
    --use_wandb True
```

可视化图片会：
- 保存到本地：`runs/<exp_name>/vizs/`
- 上传到 WandB：`Viz` 模块

---

## 支持的环境

- SafetyPointMultiGoal1-v0 / 2-v0
- SafetyCarMultiGoal1-v0 / 2-v0  
- SafetyAntMultiGoal1-v0 / 2-v0

---

## 参考资料

- **理论文档**: [Barrier_PHS.md](../Barrier_PHS.md)
- **实现代码**: [barrier_phs_pinn_actor.py](../safepo/multi_agent/barrier_phs_pinn_actor.py)
- **训练脚本**: [mappo_safe_pinn.py](../safepo/multi_agent/mappo_safe_pinn.py)

---

**最后更新**: 2025-12-20  
**维护者**: Gems Team


# 障碍势能参数Wandb Safe模块集成 - 实现总结

## 修改概述

为了更好地监控Safe-pH-MARL算法的安全性能，已实现了将所有新增障碍势能参数自动上传到wandb的Safe模块。

## 修改清单

### 1. 新增方法: `Runner.collect_barrier_physics_info(obs)` ✓

**文件**: `safepo/multi_agent/mappo_safe_pinn.py`  
**行号**: L569-607  
**功能**:
- 从所有agents的actor获取物理信息
- 对batch维度求平均
- 对agents维度求平均  
- 返回格式化的dict: `{\"Safe/Agent{id}_{metric}\": value, ...}`

**调用流程**:
```python
obs = ... # [n_threads, n_agents, obs_dim]
barrier_info = self.collect_barrier_physics_info(obs)
# 返回: {
#   "Safe/Agent0_H_task": 5.32,
#   "Safe/Agent0_H_barrier": 8.47,
#   ...
# }
```

### 2. 初始化配置记录 ✓

**文件**: `safepo/multi_agent/mappo_safe_pinn.py`  
**行号**: L370-384  
**修改内容**:
```python
# Log barrier potential configuration parameters to Safe module
barrier_config = {
    "barrier_r_safe": config.get("barrier_r_safe", 0.5),
    "barrier_epsilon": config.get("barrier_epsilon", 0.005),
    "barrier_clip_max": config.get("barrier_clip_max", 100.0),
    "barrier_k_scale": config.get("barrier_k_scale", 2.0),
    "barrier_gradient_scale": config.get("barrier_gradient_scale", 1.5),
    "barrier_decay_rate": config.get("barrier_decay_rate", 2.0),
    "min_barrier_k": config.get("min_barrier_k", 0.5),
    "cost_aware_weight": config.get("cost_aware_weight", 0.3),
    "danger_zone_threshold": config.get("danger_zone_threshold", 0.8),
}
for param_name, param_value in barrier_config.items():
    self.logger.store(**{f"Safe/Config_{param_name}": param_value})
```

**作用**: 在训练开始时将所有barrier配置参数记录到wandb

### 3. 主训练循环集成 ✓

**文件**: `safepo/multi_agent/mappo_safe_pinn.py`  
**行号**: L481-503  
**修改内容**:

#### 3a. 收集物理信息 (L481)
```python
# Collect barrier physics information for Safe module
barrier_info = self.collect_barrier_physics_info(obs)
```

#### 3b. 添加到日志字典 (L492-493)
```python
log_dict = {
    "Metrics/EpRet": aver_episode_rewards.item(),
    "Metrics/EpCost": aver_episode_costs.item(),
    "Eval/EpRet": eval_rewards,
    "Eval/EpCost": eval_costs,
}

# Add barrier physics to log dict
log_dict.update(barrier_info)

self.logger.store(**log_dict)
```

#### 3c. 日志输出 (L506-508)
```python
# Log barrier physics parameters (Safe module)
for physics_key in barrier_info.keys():
    self.logger.log_tabular(physics_key)
```

### 4. 已有的物理信息方法 ✓

**文件**: `safepo/multi_agent/barrier_phs_pinn_actor.py`  
**行号**: L685-708  
**方法名**: `get_physics_info(obs)`  
**返回内容**:
```python
{
    'H_task': Tensor,           # 任务势能
    'H_barrier': Tensor,        # 障碍势能
    'grad_H_total': Tensor,     # 总梯度
    'J': Tensor,                # 互联矩阵
    'R': Tensor,                # 耗散矩阵
    'state': Tensor,            # 物理状态
    'min_dist': Tensor          # 最小距离
}
```

## 记录的完整参数列表

### Safe/Config_* (静态，训练开始时)

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `barrier_r_safe` | 安全距离阈值 | 0.5 |
| `barrier_epsilon` | 数值稳定性系数 | 0.005 |
| `barrier_clip_max` | 最大势能上限 | 100.0 |
| `barrier_k_scale` | 障碍刚度放大因子 | 2.0 |
| `barrier_gradient_scale` | 障碍梯度放大因子 | 1.5 |
| `barrier_decay_rate` | 距离衰减幂次 | 2.0 |
| `min_barrier_k` | 最小障碍刚度 | 0.5 |
| `cost_aware_weight` | 代价感知权重 | 0.3 |
| `danger_zone_threshold` | 危险区域检测阈值 | 0.8 |

### Safe/Agent*_* (动态，每个日志周期)

| 指标 | 说明 | 计算方式 | 意义 |
|------|------|--------|------|
| `H_task` | 任务势能 | 神经网络输出均值 | 低→接近目标 |
| `H_barrier` | 障碍势能 | 基于min_dist的计算 | 高→碰撞风险 |
| `grad_H_total` | 总梯度幅度 | ∥∇H_task + ∇H_barrier∥ | 高→强制动 |
| `min_dist` | 最小距离 | 激光雷达检测 | 低→接近障碍 |

## 数据流程图

```
训练主循环 (run方法)
    ↓
每个episode结束，计算指标
    ↓
if 需要记录 (log_interval):
    ├─ obs: [n_threads, n_agents, obs_dim]
    ↓
    collect_barrier_physics_info(obs)
    ├─ for each agent_id:
    │   ├─ actor.get_physics_info(obs[:, agent_id])
    │   └─ → {H_task, H_barrier, grad_H_total, min_dist, ...}
    ├─ 对batch维度求平均
    └─ 返回: {"Safe/Agent{id}_{metric}": scalar, ...}
    ↓
logger.store(**log_dict with barrier_info)
    ↓
logger.log_tabular() for each metric
    ↓
logger.dump_tabular(step=total_steps)
    ↓
上传到wandb Safe模块
    ↓
在Wandb仪表板中查看
```

## 使用示例

### 1. 查看原始日志 (终端)

训练运行时会看到类似输出：

```
-----------  Safe module metrics  -----------
Safe/Agent0_H_task              5.32
Safe/Agent0_H_barrier           8.47
Safe/Agent0_grad_H_total        2.13
Safe/Agent0_min_dist            0.68
Safe/Agent1_H_task              5.41
Safe/Agent1_H_barrier           7.92
Safe/Agent1_grad_H_total        1.98
Safe/Agent1_min_dist            0.72
```

### 2. Wandb查看

```
项目 → Runs → 选择一个run → Charts → Safe
```

显示内容：
- 所有Safe/Config_* 参数（顶部固定显示）
- Safe/Agent*_* 动态曲线

### 3. 多run对比

在Wandb中:
1. 选择多个runs
2. 点击"Compare"
3. 对比不同barrier参数下的安全性指标

## 验证清单

- [x] `collect_barrier_physics_info()` 方法已实现
- [x] 配置参数在初始化时记录
- [x] 物理信息在主循环中收集
- [x] 数据添加到logger.store()
- [x] 数据添加到logger.log_tabular()
- [x] actor.get_physics_info() 方法存在
- [x] 文档齐全

## 测试建议

### 1. 本地测试

```bash
cd /home/xwz/Safe-Policy-Optimization
python -c "
import torch
from safepo.multi_agent.barrier_phs_pinn_actor import BarrierPHSPINNActor

config = {
    'hidden_size': 256,
    'physics_hidden': 128,
    'pinn_state_dim': 4,
    'barrier_r_safe': 0.5,
    'barrier_epsilon': 0.005,
    # ... 其他配置
}

obs_space = type('Space', (), {'shape': (152,)})()
act_space = type('Space', (), {'shape': (2,)})()

actor = BarrierPHSPINNActor(config, obs_space, act_space)

# 测试get_physics_info
obs = torch.randn(4, 152)  # batch=4, obs_dim=152
info = actor.get_physics_info(obs)

print('物理信息键:', list(info.keys()))
print('H_barrier:', info['H_barrier'].mean().item())
print('min_dist:', info['min_dist'].mean().item())
"
```

### 2. Wandb集成测试

运行一个短训练（几分钟）：
```bash
# 在config.yaml中设置
use_wandb: True
log_interval: 1  # 每个episode都记录

# 开始训练
python safepo/multi_agent/mappo_safe_pinn.py --scenario ... --use_eval False
```

然后访问wandb项目页面，应该在Safe模块中看到数据。

### 3. 数据完整性检查

在Wandb中验证：
- [ ] Safe/Config_barrier_r_safe 存在
- [ ] Safe/Config_barrier_k_scale 存在
- [ ] Safe/Agent0_H_barrier 随时间变化
- [ ] Safe/Agent0_min_dist 随时间变化

## 故障排除

### 问题1: 看不到任何Safe/*数据

**检查清单**:
1. `use_wandb: True` 在config中
2. 互联网连接正常
3. wandb.login() 已执行
4. config中包含所有barrier参数

**解决方案**:
```bash
# 检查logger是否正确初始化
grep -n "EpochLogger" safepo/multi_agent/mappo_safe_pinn.py

# 检查barrier_config字典
grep -A 10 "barrier_config = {" safepo/multi_agent/mappo_safe_pinn.py
```

### 问题2: Safe/Agent*_* 为NaN

**原因**: 激光雷达观测无效或梯度计算错误

**解决方案**:
1. 检查obs形状: `print(obs.shape)` → 应为 `[n_threads, n_agents, obs_dim]`
2. 检查激光雷达范围: `print(obs[:, :, 12:28].min(), obs[:, :, 12:28].max())`
3. 增大 `barrier_epsilon`

### 问题3: Safe数据稀疏

**原因**: 某些agents在某些episodes中无法计算

**解决方案**: 
- 正常现象，numpy会自动处理缺失值
- 运行更长的训练以收集更多数据

## 后续优化

### 建议改进

1. **物理指标的可视化**
   - 添加直方图显示H_barrier分布
   - 添加scatter图显示min_dist vs H_barrier关系

2. **警告系统**
   - 当H_barrier > barrier_clip_max时发出警告
   - 当min_dist < barrier_r_safe持续增加时发出警告

3. **参数自适应**
   - 基于H_barrier自动调整barrier_k_scale
   - 基于cost自动调整cost_aware_weight

4. **更详细的物理指标**
   - J矩阵的Frobenius范数
   - R矩阵的特征值
   - 梯度方向熵

## 参考文献

- Barrier Port-Hamiltonian系统理论: `Barrier_PHS.md`
- Actor实现: `barrier_phs_pinn_actor.py` (L685-708)
- 训练器: `mappo_safe_pinn.py`
- 日志系统: `safepo/common/logger.py`
- Wandb文档: https://docs.wandb.ai/

## 联系与问题

如有问题或建议，请参考：
- 完整指南: `SAFE_MODULE_LOGGING.md`
- 快速参考: `SAFE_QUICK_REFERENCE.md`

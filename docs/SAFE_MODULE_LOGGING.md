# Safe模块Wandb日志记录指南

## 概述

为了更好地监控和分析Safe-pH-MARL算法的安全性能，新增的障碍势能相关参数现在被上传到wandb的Safe模块。

## 记录的参数

### 1. 障碍势能配置参数 (Safe/Config_*)

在训练开始时记录的固定参数，用于标记本次实验的安全性配置：

| 参数名称 | 说明 | 类型 | 默认值 |
|---------|------|------|--------|
| `barrier_r_safe` | 安全距离阈值 | float | 0.5 |
| `barrier_epsilon` | 数值稳定性系数 | float | 0.005 |
| `barrier_clip_max` | 最大势能上限 | float | 100.0 |
| `barrier_k_scale` | 障碍刚度放大因子 | float | 2.0 |
| `barrier_gradient_scale` | 障碍梯度放大因子 | float | 1.5 |
| `barrier_decay_rate` | 距离衰减幂次 | float | 2.0 |
| `min_barrier_k` | 最小障碍刚度 | float | 0.5 |
| `cost_aware_weight` | 代价感知权重 | float | 0.3 |
| `danger_zone_threshold` | 危险区域检测阈值 | float | 0.8 |

### 2. 物理特征参数 (Safe/Agent*_*)

每个训练周期记录的动态物理指标，显示算法运行时的实时安全状态：

#### 任务势能 (H_task)
- **描述**: 神经网络学习的目标吸引势
- **含义**: 较低值表示智能体接近目标
- **对安全性的影响**: 间接影响，通过与障碍势能的平衡

#### 障碍势能 (H_barrier)
- **描述**: 障碍莱普诺夫函数势
- **含义**: 越接近障碍物，值越大（最高可达100，受clip_max限制）
- **对安全性的影响**: **直接影响**，提供碰撞保护

#### 梯度 (grad_H_total)
- **描述**: 总势能的梯度（2D向量，代表速度空间）
- **含义**: 指导智能体的运动方向
- **对安全性的影响**: 梯度幅度越大，避障力度越强

#### 最小距离 (min_dist)
- **描述**: 基于激光雷达估计的最近障碍物距离
- **含义**: 与r_safe比较判断是否处于安全区域
- **对安全性的影响**: **关键指标**，直接反映碰撞风险

## 工作原理

### 数据收集流程

```
训练循环 (每个episode)
    ↓
collect() 获取当前observation
    ↓
collect_barrier_physics_info(obs) [新增方法]
    ├─ 为每个智能体调用 actor.get_physics_info()
    ├─ 计算 H_task, H_barrier, grad_H_total, min_dist
    └─ 对所有agents求平均
    ↓
logger.store(**barrier_info)
    ↓
logger.log_tabular() 显示指标
    ↓
logger.dump_tabular() 上传到wandb Safe模块
```

### 物理信息计算详解

在 `barrier_phs_pinn_actor.py` 的 `get_physics_info()` 方法中计算：

```python
def get_physics_info(self, obs):
    """
    返回：
    - H_task: [batch] -> float (平均值)
    - H_barrier: [batch] -> float (平均值)
    - grad_H_total: [batch, 2] -> float (平均幅度)
    - J: Port-Hamiltonian矩阵
    - R: 耗散矩阵
    - state: 物理状态 (vx, vy, ax, ay)
    - min_dist: 最小距离 [batch] -> float (平均值)
    """
```

## 使用指南

### 在Wandb中查看

1. 打开wandb项目仪表板
2. 在左侧导航栏找到 **"Safe"** 模块
3. 查看以下关键指标组：

#### Safe/Config_* (配置参数)
- 显示算法的安全性超参数配置
- 实验对比时对比这些参数确定公平性

#### Safe/Agent*_* (实时指标)

**优化目标**：
- ↓ 降低 `H_barrier` 的均值 (减少碰撞风险)
- ↑ 提高 `min_dist` (保持离障碍物更远)
- 保持 `H_task` 在合理范围 (不过度追求目标而忽视安全)

**危险信号**：
- `H_barrier` 频繁达到 `barrier_clip_max` (调参已饱和)
- `min_dist` < `barrier_r_safe` (多次进入危险区域)
- `grad_H_total` 幅度为0 (物理网络故障)

### 参数调优建议

基于Safe模块指标调整障碍势能参数：

| 观察现象 | 问题 | 调优方案 |
|---------|------|--------|
| `H_barrier`长期高值 | 障碍势能过强，限制任务完成 | ↓ `barrier_k_scale` 或 ↑ `barrier_r_safe` |
| `min_dist` < `barrier_r_safe` | 安全性不足，经常碰撞 | ↑ `barrier_k_scale` 或 ↓ `barrier_r_safe` |
| `grad_H_total` 波动大 | 控制不稳定 | ↑ `barrier_epsilon` 或 ↓ `barrier_decay_rate` |
| Cost曲线振荡 | 危险行为未被充分惩罚 | ↑ `cost_aware_weight` |

## 实现细节

### 新增方法

#### `Runner.collect_barrier_physics_info(obs)` 
位置: `mappo_safe_pinn.py` L520-565

功能：
- 调用各agent的actor获取物理信息
- 对batch维度求平均
- 对agents求平均
- 返回格式化的dict供logger使用

#### 初始化配置记录
位置: `mappo_safe_pinn.py` L375-384

功能：
- 在__init__中记录所有barrier配置参数
- 这些参数在wandb中显示为参考

### 修改的方法

#### `Runner.run()` 主训练循环
- 新增: 每个log周期调用 `collect_barrier_physics_info()`
- 修改: 日志dict中添加物理信息
- 修改: log_tabular中添加Safe参数的打印

## 示例

### 一个训练周期的日志输出

```
Metrics/EpRet: 15.23 ± 2.45
Metrics/EpCost: 2.15 ± 0.67
Eval/EpRet: 14.89
Eval/EpCost: 1.98

Safe/Agent0_H_task: 5.32
Safe/Agent0_H_barrier: 8.47
Safe/Agent0_grad_H_total: 2.13
Safe/Agent0_min_dist: 0.68
Safe/Agent1_H_task: 5.41
Safe/Agent1_H_barrier: 7.92
Safe/Agent1_grad_H_total: 1.98
Safe/Agent1_min_dist: 0.72

Train/Epoch: 100
Train/TotalSteps: 2000000
...
```

### Wandb图表建议

创建自定义chart来可视化安全性趋势：

1. **碰撞风险监控**: `Safe/Agent*_min_dist` vs Episode
   - 趋势应该向上或稳定

2. **势能活动**: `Safe/Agent*_H_barrier` vs Episode
   - 趋势应该向下或稳定

3. **安全与任务权衡**: 
   - X轴: `Safe/Agent*_H_barrier`
   - Y轴: `Metrics/EpRet`
   - 显示Pareto前沿

4. **参数有效性**:
   - 对比不同 `barrier_k_scale` 的运行
   - 对比不同 `barrier_r_safe` 的运行

## 故障排除

### 看不到Safe模块数据

1. 检查wandb初始化：`use_wandb=True` 在config中
2. 检查网络连接
3. 查看终端是否有错误: `Safe/Agent*_*` 的日志记录

### 物理值异常 (NaN, Inf)

1. 检查观测空间中激光雷达是否正确配置
2. 检查 `barrier_epsilon` 是否过小
3. 检查 `barrier_clip_max` 是否太小

### 数据稀疏 (只有某些agents有数据)

- 正常现象，取决于exploration轨迹
- 使用`np.mean()`自动忽略缺失值

## 参考文献

- 理论基础: 见 `Barrier_PHS.md`
- Port-Hamiltonian系统: 见 `barrier_phs_pinn_actor.py` 文档

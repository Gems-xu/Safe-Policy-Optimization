# Safe模块Wandb集成 - 完成总结

## ✅ 已完成的工作

已成功实现将新增障碍势能相关参数上传到wandb的Safe模块。

### 核心修改

#### 1. 新增收集方法 (`Runner.collect_barrier_physics_info()`)
- **位置**: `safepo/multi_agent/mappo_safe_pinn.py` L569-607
- **功能**: 从所有agents的actor获取物理信息并聚合
- **输出**: 格式化的dict，包含所有agents的H_task, H_barrier, grad_H_total, min_dist

#### 2. 配置参数记录
- **位置**: `safepo/multi_agent/mappo_safe_pinn.py` L370-384
- **内容**: 9个barrier势能配置参数在training start时记录到Safe模块
- **格式**: `Safe/Config_{parameter_name}`

#### 3. 主循环集成
- **位置**: `safepo/multi_agent/mappo_safe_pinn.py` L481-508
- **流程**: 每个log_interval周期调用collect_barrier_physics_info()并记录
- **输出**: Safety指标自动出现在wandb的Safe module

### 记录的参数

**Safe/Config_* (9个，固定)**
```
barrier_r_safe, barrier_epsilon, barrier_clip_max,
barrier_k_scale, barrier_gradient_scale, barrier_decay_rate,
min_barrier_k, cost_aware_weight, danger_zone_threshold
```

**Safe/Agent*_* (每个agent, 4个，动态)**
```
H_task          - 目标势能
H_barrier       - 障碍势能 ⭐ (碰撞风险指标)
grad_H_total    - 总梯度幅度
min_dist        - 最小距离 ⭐⭐⭐ (最重要的安全指标)
```

## 📊 在Wandb中查看

### 步骤
1. 打开wandb项目链接
2. 选择一个training run
3. 查看图表选项卡
4. 找到**"Safe"**模块
5. 查看所有Safe/*指标

### 关键指标解读

| 指标 | 优化目标 | 危险信号 |
|------|--------|--------|
| `min_dist` | ↑ 越大越好 | < `barrier_r_safe` |
| `H_barrier` | ↓ 越小越好 | = 100 (饱和) |
| `H_task` | 适度 | 过大表示任务-安全冲突 |

## 📚 文档

### 三个新增文档

1. **SAFE_MODULE_LOGGING.md** - 完整用户指南
   - 详细的参数解释
   - 图表创建建议
   - 调参工作流

2. **SAFE_QUICK_REFERENCE.md** - 快速参考卡
   - 关键指标速查表
   - 常见问题解决
   - 调优工作流

3. **SAFE_WANDB_IMPLEMENTATION.md** - 技术实现文档
   - 修改清单
   - 代码位置
   - 测试方法

## 🔧 技术细节

### 调用链

```
run() 
  ↓
  collect_barrier_physics_info(obs) [新增]
    ↓
    for each agent:
      actor.get_physics_info(obs[:, agent_id]) [既有]
        ↓
        计算H_task, H_barrier, grad_H_total, min_dist
      平均化 over batch dimension
    ↓
  返回 {"Safe/Agent{id}_{metric}": value, ...}
  ↓
logger.store(**barrier_info)
  ↓
logger.log_tabular() for each key
  ↓
logger.dump_tabular(step)
  ↓
上传wandb
```

### 关键代码位置

| 位置 | 行号 | 功能 |
|------|------|------|
| mappo_safe_pinn.py | L370-384 | 配置参数记录 |
| mappo_safe_pinn.py | L481 | 收集物理信息 |
| mappo_safe_pinn.py | L492-493 | 添加到日志dict |
| mappo_safe_pinn.py | L506-508 | 输出到终端和wandb |
| mappo_safe_pinn.py | L569-607 | 新增收集方法 |
| barrier_phs_pinn_actor.py | L685-708 | 物理信息计算 |

## 🚀 立即开始使用

### 1. 确认配置

在 `safepo/multi_agent/marl_cfg/mappo_safe_pinn/config.yaml` 中：

```yaml
use_wandb: True
# 障碍势能参数已自动记录，无需额外配置
```

### 2. 启动训练

```bash
cd /home/xwz/Safe-Policy-Optimization
python safepo/multi_agent/mappo_safe_pinn.py \
  --task SafetyPointMultiGoal1-v0 \
  --seed 0
```

### 3. 实时监控

在training过程中打开wandb项目，每个log_interval会自动更新Safe module的数据。

## ✨ 期望的结果

### 训练输出 (终端)

每个log_interval会显示：

```
-----------  Metrics/EpRet  -----------
Metrics/EpRet              15.23 ± 2.45

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

### Wandb显示 (网页)

Safe模块显示：
- Config参数面板（顶部）
- Agent0_H_barrier 曲线图
- Agent0_min_dist 曲线图
- Agent1_H_barrier 曲线图
- Agent1_min_dist 曲线图
- ...等多个指标的动态图表

## 🎯 使用场景

### 1. 算法调参

比较不同barrier参数对安全性的影响：

```bash
# 实验1: barrier_k_scale = 1.0
# 实验2: barrier_k_scale = 2.0 (默认)
# 实验3: barrier_k_scale = 3.0

# 在wandb中对比三个runs的Safe/Agent*_min_dist曲线
# 选择达到目标的最佳参数
```

### 2. 算法验证

验证新的安全算法是否真的更安全：

```bash
# 对比MAPPO vs MAPPO-Safe-PINN
# 查看Safe module中的min_dist和H_barrier
# Safe version应该有更高的min_dist和更低的H_barrier
```

### 3. 性能分析

理解安全与任务性能的权衡：

```bash
# 创建scatter plot: x=Safe/Agent0_H_barrier, y=Metrics/EpRet
# 观察Pareto前沿
# 找到最好的balance点
```

## ⚠️ 注意事项

1. **首次运行**: Safe参数可能需要几个episodes才能稳定显示
2. **缺失数据**: 某些agents可能没有某些指标（正常现象）
3. **数值范围**: H_barrier可达100（clip_max）表示已饱和
4. **Wandb连接**: 确保training环境能连接到wandb

## 📖 进一步阅读

- 物理理论: `Barrier_PHS.md`
- 完整指南: `SAFE_MODULE_LOGGING.md`
- 快速参考: `SAFE_QUICK_REFERENCE.md`
- 技术细节: `SAFE_WANDB_IMPLEMENTATION.md`

## ✅ 验证清单

在实际使用前请确认：

- [ ] wandb账户已连接
- [ ] config.yaml中use_wandb=True
- [ ] 所有barrier参数在config.yaml中正确设置
- [ ] 能访问wandb项目链接
- [ ] 运行了至少一个training episode

## 🆘 快速故障排除

| 问题 | 解决方案 |
|------|--------|
| 看不到Safe/* | 检查use_wandb=True, 检查网络连接 |
| Safe/*为NaN | 检查obs维度, 增大barrier_epsilon |
| 只有一个agent有数据 | 正常现象，数据会随时间积累 |
| H_barrier全是100 | barrier_k_scale过大，减小或增大r_safe |

---

**最后更新**: 2025-12-19
**实现状态**: ✅ 完成
**测试状态**: ✅ 代码检查通过

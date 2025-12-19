# Safe模块参数映射表

## 完整参数记录清单

### 第一类：配置参数 (Safe/Config_*) - 训练开始时固定记录

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Safe/Config Parameters                           │
├──────────────────────────┬────────────┬──────────────────────────────┤
│ 参数名                   │ 默认值     │ 说明                         │
├──────────────────────────┼────────────┼──────────────────────────────┤
│ barrier_r_safe           │ 0.5        │ 安全距离阈值                 │
│ barrier_epsilon          │ 0.005      │ 数值稳定性                   │
│ barrier_clip_max         │ 100.0      │ 势能上限                     │
│ barrier_k_scale          │ 2.0        │ 刚度放大因子                 │
│ barrier_gradient_scale   │ 1.5        │ 梯度放大因子                 │
│ barrier_decay_rate       │ 2.0        │ 距离衰减幂次                 │
│ min_barrier_k            │ 0.5        │ 最小刚度                     │
│ cost_aware_weight        │ 0.3        │ 代价感知权重                 │
│ danger_zone_threshold    │ 0.8        │ 危险区域阈值                 │
└──────────────────────────┴────────────┴──────────────────────────────┘
```

### 第二类：运行时指标 (Safe/Agent*_*) - 每个日志周期动态记录

```
┌──────────────────────────────────────────────────────────────────────────┐
│         Safe/Agent{id}_{metric} - Runtime Physics Information           │
├──────────────────────────┬────────────┬────────────────────────────────────┤
│ 指标                     │ 类型       │ 说明                               │
├──────────────────────────┼────────────┼────────────────────────────────────┤
│ H_task                   │ float      │ 目标势能 (低=接近目标)            │
│                          │            │ 范围: [0, ∞)                       │
│                          │            │ 典型值: 5-10                       │
├──────────────────────────┼────────────┼────────────────────────────────────┤
│ H_barrier                │ float      │ 障碍势能 (高=碰撞风险) ⭐⭐        │
│                          │            │ 范围: [0, barrier_clip_max=100]   │
│                          │            │ 危险: > 50                        │
│                          │            │ 严重: = 100 (饱和)               │
├──────────────────────────┼────────────┼────────────────────────────────────┤
│ grad_H_total             │ float      │ 总梯度幅度                        │
│                          │            │ 范围: [0, 20]                      │
│                          │            │ 高=强制动, 低=弱制动              │
├──────────────────────────┼────────────┼────────────────────────────────────┤
│ min_dist                 │ float      │ 最小距离 ⭐⭐⭐ (最重要!)          │
│                          │            │ 范围: [0, 3]                       │
│                          │            │ 安全: > barrier_r_safe (0.5)     │
│                          │            │ 危险: < 0.3                       │
│                          │            │ 碰撞: = 0                         │
└──────────────────────────┴────────────┴────────────────────────────────────┘
```

## 数据流向图

```
                    ┌─────────────────────────┐
                    │   Training Process      │
                    │  run() Loop             │
                    └────────┬────────────────┘
                             │
                    ┌────────▼────────────┐
                    │ For each episode    │
                    │ Calculate rewards   │
                    └────────┬────────────┘
                             │
                    ┌────────▼────────────────┐
                    │ if log_interval:       │
                    │  └─ collect()          │
                    │  └─ obs generated      │
                    └────────┬────────────────┘
                             │
         ┌───────────────────▼───────────────────┐
         │                                       │
    ┌────▼─────────────────┐           ┌────────▼────────┐
    │ collect_barrier_     │           │ Logger          │
    │ physics_info(obs)    │           │ store & output  │
    │ [NEW METHOD]         │           └─────────────────┘
    │                      │                    │
    │ ├─ for each agent:   │                    │
    │ │ ├─ get_physics_    │          ┌─────────▼────────┐
    │ │ │  info()          │          │ wandb.log()      │
    │ │ │ ├─ compute       │──────────│                  │
    │ │ │   H_task         │          │ Safe Module      │
    │ │ │ ├─ compute       │──────────│                  │
    │ │ │   H_barrier      │          │ ├─ Config_*     │
    │ │ │ ├─ compute       │──────────│ └─ Agent*_*     │
    │ │ │   grad_H_total   │          │                  │
    │ │ │ └─ compute       │──────────│ 📊 Wandb        │
    │ │   min_dist         │          │ Dashboard       │
    │ │ ├─ avg over batch  │          └──────────────────┘
    │ │ └─ gather results  │
    │ └─ avg over agents   │
    │ └─ return dict       │
    └────┬─────────────────┘
         │
    ┌────▼────────────────────────┐
    │ {"Safe/Agent0_H_barrier": 8, │
    │  "Safe/Agent0_min_dist": 0.7,│
    │  "Safe/Agent1_H_barrier": 7, │
    │  "Safe/Agent1_min_dist": 0.8,│
    │  ...}                         │
    └───────────────────────────────┘
```

## 代码修改位置快速索引

```
safepo/multi_agent/mappo_safe_pinn.py
│
├─ L370-384        ✓ 配置参数初始化记录
│  └─ barrier_config dict + logger.store()
│
├─ L481            ✓ 收集物理信息调用
│  └─ barrier_info = self.collect_barrier_physics_info(obs)
│
├─ L492-493        ✓ 添加到日志字典
│  └─ log_dict.update(barrier_info)
│
├─ L506-508        ✓ 输出到wandb
│  └─ for physics_key in barrier_info.keys()
│     logger.log_tabular(physics_key)
│
└─ L569-607        ✓ 新增方法定义
   └─ def collect_barrier_physics_info(self, obs)

barrier_phs_pinn_actor.py
└─ L685-708        ✓ 物理信息计算方法
   └─ def get_physics_info(self, obs)
      返回: {H_task, H_barrier, grad_H_total, min_dist, ...}
```

## 关键指标优化目标

```
┌─────────────────────────────────────────────────────────────┐
│              Safe Module Optimization Strategy              │
├──────────────┬─────────────┬────────────┬───────────────────┤
│ 指标         │ 优化方向    │ 危险信号   │ 调参建议           │
├──────────────┼─────────────┼────────────┼───────────────────┤
│ min_dist     │ ↑ 越大越好  │ < r_safe   │ ↑ barrier_k_scale│
│              │             │            │ ↓ barrier_r_safe  │
│              │             │            │ ↑ cost_aware_wt   │
├──────────────┼─────────────┼────────────┼───────────────────┤
│ H_barrier    │ ↓ 越小越好  │ > 50       │ ↓ barrier_k_scale│
│              │             │ = 100(sat) │ ↑ barrier_r_safe  │
├──────────────┼─────────────┼────────────┼───────────────────┤
│ grad_H_total │ 适度稳定    │ = 0 or >>  │ 调整epsilon       │
│              │ (不要0也不要│ 10         │ 调整decay_rate    │
│              │  特别大)    │            │                   │
├──────────────┼─────────────┼────────────┼───────────────────┤
│ H_task       │ 适度        │ >> 20      │ 减少barrier影响   │
│              │ (不过度)    │            │ 或增大learning_lr │
├──────────────┼─────────────┼────────────┼───────────────────┤
│ EpCost       │ ↓ 越低越好  │ > 5        │ ↑ cost_aware_wt   │
│              │             │ 上升趋势   │ 调整barrier参数   │
└──────────────┴─────────────┴────────────┴───────────────────┘
```

## Wandb图表推荐

### 1. 安全性监控仪表板

```
┌──────────────────────────┐
│   Safety Monitoring      │
├──────────────────────────┤
│                          │
│ [Graph 1]  [Graph 2]     │
│ min_dist   H_barrier     │
│ trend      trend         │
│                          │
│ [Graph 3]  [Graph 4]     │
│ EpCost     grad_H_total  │
│ trend      trend         │
│                          │
└──────────────────────────┘
```

### 2. 参数有效性对比

```
┌────────────────────────────────────────┐
│    Barrier Parameter Effectiveness     │
├────────────────────────────────────────┤
│ X: barrier_k_scale                     │
│ Y1: Safe/Agent0_min_dist               │
│ Y2: Metrics/EpRet                      │
│ (显示Pareto前沿)                        │
└────────────────────────────────────────┘
```

### 3. 多Agent对比

```
┌─────────────────────────────────────┐
│     Multi-Agent Safety Analysis     │
├─────────────────────────────────────┤
│ Line 1: Agent0_min_dist             │
│ Line 2: Agent1_min_dist             │
│ Line 3: Avg min_dist                │
│ (检查是否有agent特别容易碰撞)       │
└─────────────────────────────────────┘
```

## 数据示例

### 终端输出示例

```
2025-12-19 10:30:45 | Epoch 100 | Step 2000000

Metrics/EpRet                       15.23 ± 2.45
Metrics/EpCost                       2.15 ± 0.67
Eval/EpRet                          14.89
Eval/EpCost                          1.98

Safe/Agent0_H_task                   5.32
Safe/Agent0_H_barrier                8.47      ← 碰撞风险低
Safe/Agent0_grad_H_total             2.13
Safe/Agent0_min_dist                 0.68      ← > r_safe(0.5) 安全
Safe/Agent1_H_task                   5.41
Safe/Agent1_H_barrier                7.92
Safe/Agent1_grad_H_total             1.98
Safe/Agent1_min_dist                 0.72      ← > r_safe(0.5) 安全

Loss/Loss_reward_critic              0.023
Loss/Loss_actor                      0.042
Time/Total                         125.34s
Time/FPS                          15936
```

### Wandb界面显示

```
Safe Module
├─ Config Parameters (顶部固定面板)
│  ├─ barrier_r_safe: 0.5
│  ├─ barrier_k_scale: 2.0
│  └─ ... 其他配置
│
├─ Charts (动态曲线)
│  ├─ Safe/Agent0_H_barrier
│  │  └─ Line chart, trending down → 好
│  ├─ Safe/Agent0_min_dist  
│  │  └─ Line chart, trending up or stable → 好
│  ├─ Safe/Agent1_H_barrier
│  │  └─ Line chart, trending down → 好
│  ├─ Safe/Agent1_min_dist
│  │  └─ Line chart, trending up → 好
│  └─ ... 其他agents
│
└─ Summary Statistics
   ├─ Safe/Agent0_H_barrier: mean=8.2, min=2.1, max=48.3
   ├─ Safe/Agent0_min_dist: mean=0.65, min=0.3, max=1.2
   └─ ... 其他
```

---

**图表更新日期**: 2025-12-19  
**版本**: 1.0  
**状态**: ✅ 完成

# Safe模块Wandb日志 - 快速参考

## 什么被记录？

### 配置参数 (训练开始时)
```
Safe/Config_barrier_r_safe
Safe/Config_barrier_epsilon  
Safe/Config_barrier_clip_max
Safe/Config_barrier_k_scale
Safe/Config_barrier_gradient_scale
Safe/Config_barrier_decay_rate
Safe/Config_min_barrier_k
Safe/Config_cost_aware_weight
Safe/Config_danger_zone_threshold
```

### 运行时物理指标 (每个日志周期)
```
Safe/Agent0_H_task          # 目标势能
Safe/Agent0_H_barrier       # 障碍势能 ⭐
Safe/Agent0_grad_H_total    # 总梯度幅度
Safe/Agent0_min_dist        # 最小距离 ⭐⭐⭐
...
Safe/Agent1_H_task
Safe/Agent1_H_barrier
Safe/Agent1_grad_H_total
Safe/Agent1_min_dist
```

## 关键指标解读

| 指标 | 优化方向 | 警告信号 | 调参建议 |
|------|--------|--------|--------|
| `H_barrier` | ↓ 降低 | 频繁=100 | ↓ `barrier_k_scale` |
| `min_dist` | ↑ 提高 | < `r_safe` | ↑ `barrier_k_scale` |
| `grad_H_total` | 适度 | = 0 或很大 | 调整 `epsilon` 或 `decay_rate` |
| 碰撞次数 | ↓ 降低 | 增加趋势 | ↑ `barrier_k_scale` 或 ↓ `barrier_r_safe` |

## 实现细节

### 新方法: `collect_barrier_physics_info(obs)`
- 位置: `mappo_safe_pinn.py` L569-607
- 功能: 从所有agents的actor获取物理信息并求平均
- 返回: dict格式，keys为 `Safe/Agent{id}_{metric}`

### 修改点
1. **初始化** (L370-384): 记录所有barrier配置参数
2. **主循环** (L481): 调用 `collect_barrier_physics_info(obs)` 
3. **日志记录** (L492-503): 添加barrier_info到logger

## 在Wandb中查看

1. 打开项目仪表板
2. 点击 **"Safe"** 选项卡 (左侧导航)
3. 图表中查看各指标趋势

### 推荐的自定义图表

```yaml
# 1. 安全距离趋势
x: step
y: [Safe/Agent0_min_dist, Safe/Agent1_min_dist]
label: "Distance to Obstacles"

# 2. 碰撞风险
x: step  
y: Safe/Agent0_H_barrier
label: "Barrier Potential (Risk)"

# 3. 安全-性能权衡
x: Safe/Agent0_H_barrier
y: Metrics/EpRet
label: "Performance vs Safety"

# 4. 参数对比 (多runs)
x: step
y: [Safe/Config_barrier_k_scale, ...others]
split_by: run
```

## 调试

### 看不到Safe/Agent* 数据?
1. 检查`obs`维度是否正确传入
2. 查看actor是否实现了`get_physics_info()`
3. 检查设备匹配 (`obs.to(device)`)

### 值为NaN或Inf?
1. ↓ `barrier_epsilon` (最小0.001)
2. ↑ `barrier_decay_rate` 
3. 检查激光雷达输入是否有效

### 只有某些agents有数据?
- 正常现象，数据稀疏时被忽略
- 收集更多episodes即可

## 参数调优工作流

```
1. 记录当前Safe/Config_* 和 performance
   ↓
2. 根据指标判断是否需要调参
   - H_barrier > 50 → ↓ barrier_k_scale
   - min_dist < r_safe → ↑ barrier_k_scale
   - Cost > 2.0 → ↑ cost_aware_weight
   ↓
3. 更新config.yaml中的参数
   ↓
4. 重新运行训练
   ↓
5. 对比Safe模块中两次运行的曲线
   ↓
6. 迭代优化
```

## 相关文件

- 理论文档: `Barrier_PHS.md`
- 实现: `barrier_phs_pinn_actor.py` (L649-683: `get_physics_info()`)
- 日志系统: `safepo/common/logger.py`
- 完整指南: `SAFE_MODULE_LOGGING.md`

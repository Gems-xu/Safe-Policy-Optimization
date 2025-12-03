# Safe-Policy-Optimization 迁移到 uv 总结

本文档总结了将 Safe-Policy-Optimization 项目从传统的 pip/conda 管理迁移到 uv 的所有更改。

## 📋 更改概览

### 新增文件

1. **`pyproject.toml`** - 现代化的 Python 项目配置文件
   - 符合 PEP 621 标准
   - 包含项目元数据、依赖、可选依赖和构建配置
   - 替代了 `setup.py` 的大部分功能

2. **`uv.lock`** - 依赖锁文件
   - 确保可重现的安装
   - 锁定所有依赖的具体版本
   - 应提交到版本控制

3. **`.python-version`** - Python 版本规范
   - 指定项目使用 Python 3.8
   - uv 会自动识别并使用正确的 Python 版本

4. **`MIGRATION_TO_UV.md`** - 详细的迁移指南
   - 解释什么是 uv 及其优势
   - 提供命令对照表
   - 包含故障排除指南

5. **`QUICKSTART.md`** - 快速入门指南
   - 简化的安装和使用说明
   - 常见任务示例
   - 项目结构说明

### 修改文件

1. **`setup.py`**
   - 简化为向后兼容的 shim
   - 所有配置迁移到 `pyproject.toml`
   - 添加了说明注释

2. **`Makefile`**
   - 添加 `UV` 变量
   - 所有 `pip install` 替换为 `uv sync`
   - 所有 `python` 命令替换为 `uv run python`
   - 移除了 pip 检查辅助函数

3. **`README.md`**
   - 更新安装说明使用 uv
   - 添加 uv 安装指南
   - 更新所有命令示例
   - 在 "What's New" 添加 uv 迁移说明

4. **`Installation.md`**
   - 完全重写为使用 uv
   - 现代化的安装步骤
   - 改进的故障排除部分
   - 移除 conda 特定说明

5. **`.gitignore`**
   - 添加 `.venv/` 忽略规则（uv 的虚拟环境）
   - 取消 `.python-version` 的忽略（需要跟踪）
   - 注意：`uv.lock` 不应被忽略

6. **`.github/workflows/test.yml`**
   - 使用 `astral-sh/setup-uv@v3` action
   - 将 pip 安装替换为 `uv sync`
   - 启用 uv 缓存以加快 CI 速度
   - 更新触发路径包含 `pyproject.toml` 和 `uv.lock`

## 🚀 主要优势

1. **速度提升**
   - 安装速度提升 10-100 倍
   - 更快的依赖解析

2. **可重现性**
   - `uv.lock` 确保所有环境使用相同的依赖版本
   - 避免"在我机器上能运行"的问题

3. **现代化**
   - 使用 `pyproject.toml`（PEP 621 标准）
   - 符合最新的 Python 打包最佳实践

4. **简化**
   - 一个工具管理所有：Python 版本、包、虚拟环境
   - 自动虚拟环境管理

5. **兼容性**
   - 作为 pip 的直接替代品
   - 无需更改代码

## 📦 依赖管理

### pyproject.toml 结构

```toml
[project]
dependencies = [...]           # 运行时依赖

[project.optional-dependencies]
dev = [...]                    # 开发工具
docs = [...]                   # 文档构建
mujoco = [...]                 # MuJoCo 支持

[dependency-groups]
dev = [...]                    # 开发依赖（uv 特定）
```

### 安装选项

```bash
uv sync                    # 安装所有依赖（包括开发依赖）
uv sync --no-dev          # 仅生产依赖
uv sync --extra docs      # 包含文档依赖
uv sync --extra mujoco    # 包含 MuJoCo 支持
```

## 🔄 命令迁移对照

| 操作 | 旧命令 | 新命令 |
|------|--------|--------|
| 创建环境 | `conda create -n safepo python=3.8` | `uv sync`（自动） |
| 激活环境 | `conda activate safepo` | 不需要（使用 `uv run`） |
| 安装项目 | `pip install -e .` | `uv sync` |
| 运行脚本 | `python script.py` | `uv run python script.py` |
| 安装包 | `pip install package` | `uv add package` |
| 卸载包 | `pip uninstall package` | `uv remove package` |
| 更新依赖 | `pip install --upgrade package` | `uv lock --upgrade` |
| 运行测试 | `pytest` | `uv run pytest` |

## 🛠️ Makefile 命令保持不变

用户仍然可以使用相同的 Makefile 命令：

```bash
make install              # 安装项目
make install-editable     # 开发模式安装
make benchmark            # 运行基准测试
make simple-benchmark     # 简单基准测试
make pytest               # 运行测试
make docs                 # 构建文档
```

## 📝 使用指南

### 开发工作流

1. **克隆项目**
   ```bash
   git clone https://github.com/PKU-Alignment/Safe-Policy-Optimization.git
   cd Safe-Policy-Optimization
   ```

2. **安装依赖**
   ```bash
   uv sync
   ```

3. **运行实验**
   ```bash
   uv run python safepo/single_agent/ppo_lag.py --task SafetyPointGoal1-v0
   ```

4. **添加新依赖**
   ```bash
   uv add numpy scipy
   uv lock  # 更新锁文件
   ```

5. **提交更改**
   ```bash
   git add pyproject.toml uv.lock
   git commit -m "Add new dependencies"
   ```

### 虚拟环境

uv 在 `.venv/` 目录自动管理虚拟环境：

- **自动创建**：运行 `uv sync` 时自动创建
- **无需激活**：使用 `uv run` 前缀即可
- **手动激活**（可选）：
  ```bash
  source .venv/bin/activate  # Linux/macOS
  .venv\Scripts\activate     # Windows
  ```

## 🔍 验证迁移

运行以下命令验证设置：

```bash
# 测试安装
uv sync

# 运行测试
make pytest

# 运行简单基准测试
make simple-benchmark
```

## 📚 相关资源

- [uv 文档](https://docs.astral.sh/uv/)
- [PEP 621 - pyproject.toml 规范](https://peps.python.org/pep-0621/)
- [Python 打包指南](https://packaging.python.org/)

## ❓ 常见问题

### Q: 是否还需要 conda？

A: 不需要。uv 可以管理 Python 版本和包，但如果需要非 Python 依赖（如 CUDA），仍可能需要系统包管理器。

### Q: setup.py 还有用吗？

A: 保留 `setup.py` 仅为向后兼容。所有配置现在都在 `pyproject.toml` 中。

### Q: 如何安装特定 CUDA 版本的 PyTorch？

A: 使用 `uv pip install` 指定 index URL：
```bash
uv pip install torch==1.9.0+cu111 --index-url https://download.pytorch.org/whl/torch_stable.html
```

### Q: CI/CD 需要更改吗？

A: 是的，已更新 GitHub Actions 使用 `astral-sh/setup-uv@v3`。

### Q: 如何在 uv 和 pip 之间切换？

A: `pyproject.toml` 同时兼容 uv 和 pip。如需使用 pip：
```bash
pip install -e .
```

## 🎯 后续步骤

1. ✅ 验证所有测试通过
2. ✅ 更新 CI/CD 流水线
3. ✅ 更新文档
4. 📝 发布更新说明
5. 🔄 监控用户反馈

## 🤝 贡献

如果在使用 uv 时遇到任何问题，请：

1. 查看 [MIGRATION_TO_UV.md](MIGRATION_TO_UV.md) 获取详细信息
2. 阅读 [QUICKSTART.md](QUICKSTART.md) 快速入门
3. 在 [GitHub Issues](https://github.com/PKU-Alignment/Safe-Policy-Optimization/issues) 报告问题

## 📄 许可

本项目继续使用 Apache License 2.0。

---

**迁移完成日期**: 2025-12-03
**uv 版本**: latest
**维护者**: PKU-Alignment Team

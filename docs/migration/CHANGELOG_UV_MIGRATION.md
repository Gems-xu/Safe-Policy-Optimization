# Changelog - Migration to uv

## [1.0.1] - 2025-12-03

### 🚀 Major Changes

#### Migrated to uv Package Manager

The project has been completely migrated from pip/conda to uv for faster and more reliable package management.

**Why uv?**
- **10-100x faster** installation compared to pip
- **Reproducible** builds with automatic lock file generation
- **Modern** Python packaging with pyproject.toml (PEP 621)
- **Unified** tool for managing Python versions, packages, and virtual environments

### 📦 New Files

- `pyproject.toml` - Modern Python project configuration (PEP 621 compliant)
- `uv.lock` - Dependency lock file for reproducible installations
- `.python-version` - Python version specification (3.8)
- `MIGRATION_TO_UV.md` - Detailed migration guide and command reference
- `QUICKSTART.md` - Quick start guide for new users
- `UV_MIGRATION_SUMMARY.md` - Comprehensive summary of all changes (Chinese)

### 🔄 Modified Files

- `setup.py` - Simplified to backward-compatible shim (all config moved to pyproject.toml)
- `Makefile` - Updated all commands to use uv instead of pip
- `README.md` - Updated installation instructions and all command examples
- `Installation.md` - Completely rewritten for uv-based installation
- `.gitignore` - Updated to handle uv virtual environments
- `.github/workflows/test.yml` - Updated CI to use uv with caching

### 📝 Dependency Changes

- Pinned torch to `>=1.10.0,<2.5.0` for Python 3.8 compatibility
- All dependencies now managed through pyproject.toml
- Added optional dependency groups:
  - `dev` - Development tools (pytest, etc.)
  - `docs` - Documentation building tools
  - `mujoco` - MuJoCo support

### 💻 Installation

#### Before (pip/conda)
```bash
conda create -n safepo python=3.8
conda activate safepo
pip install -e .
```

#### After (uv)
```bash
# Install uv (one-time)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install project
git clone https://github.com/PKU-Alignment/Safe-Policy-Optimization.git
cd Safe-Policy-Optimization
uv sync
```

### 🎯 Usage Changes

All Makefile commands remain the same, but now use uv internally:

```bash
make install              # Same command, faster execution
make benchmark            # Same command, faster setup
make pytest               # Same command, same results
```

For running Python scripts directly:

#### Before
```bash
python safepo/single_agent/ppo_lag.py --task SafetyPointGoal1-v0
```

#### After
```bash
uv run python safepo/single_agent/ppo_lag.py --task SafetyPointGoal1-v0
```

### 🔧 For Developers

#### Adding Dependencies
```bash
# Before
pip install package-name

# After
uv add package-name
uv lock  # Updates uv.lock
```

#### Running Tests
```bash
# Both still work via Makefile
make pytest

# Or directly with uv
uv run pytest
```

### ⚠️ Breaking Changes

None! All functionality remains the same. The project is still pip-compatible if needed.

### 🐛 Bug Fixes

- Fixed torch version constraint for Python 3.8 compatibility
- Updated all documentation with correct installation steps

### 📚 Documentation

- Added comprehensive migration guide
- Added quick start guide for new users
- Updated README with uv installation instructions
- Updated Installation.md with modern best practices

### 🔗 Resources

- [uv Documentation](https://docs.astral.sh/uv/)
- [Migration Guide](MIGRATION_TO_UV.md)
- [Quick Start Guide](QUICKSTART.md)
- [Migration Summary (中文)](UV_MIGRATION_SUMMARY.md)

### 🙏 Acknowledgments

Thanks to the Astral team for creating uv, making Python package management faster and more reliable.

---

**Full Changelog**: View all changes in the [repository](https://github.com/PKU-Alignment/Safe-Policy-Optimization)

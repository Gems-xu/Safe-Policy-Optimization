# Migration Guide: From pip/conda to uv

This document provides a comprehensive guide for migrating the Safe-Policy-Optimization project from traditional pip/conda management to uv.

## What is uv?

[uv](https://docs.astral.sh/uv/) is an extremely fast Python package installer and resolver, written in Rust. It's designed as a drop-in replacement for pip and pip-tools, offering:

- **10-100x faster** than pip
- **Reproducible** with automatic lock file generation
- **Compatible** with existing Python projects
- **Unified** tool for managing Python versions, dependencies, and virtual environments

## What Changed?

### New Files

1. **`pyproject.toml`** - Modern Python project configuration (PEP 621)
   - Replaces `setup.py` for project metadata and dependencies
   - Contains build system configuration
   - Includes optional dependencies for docs, dev, and mujoco

2. **`uv.lock`** - Dependency lock file
   - Ensures reproducible installations across all environments
   - Generated and managed automatically by uv
   - Should be committed to version control

3. **`.python-version`** - Python version specification
   - Specifies Python 3.8 as the project's Python version
   - Used by uv to automatically select the correct Python version

### Modified Files

1. **`Makefile`**
   - All `pip install` commands replaced with `uv sync` or `uv run`
   - Removed helper functions for checking pip packages
   - All Python script executions now use `uv run python ...`

2. **`README.md`**
   - Updated installation instructions to use uv
   - Replaced conda/pip commands with uv equivalents
   - Added uv installation guide

3. **`Installation.md`**
   - Complete rewrite of installation steps for uv
   - Modernized troubleshooting section
   - Removed conda-specific instructions

4. **`.gitignore`**
   - Now ignores `.venv/` (uv's default virtual environment directory)
   - Added `uv.lock` to be tracked in version control
   - Uncommented `.python-version` to track it

## Installation Comparison

### Old Way (pip/conda)

```bash
conda create -n safepo python=3.8
conda activate safepo
pip install -e .
```

### New Way (uv)

```bash
# Install uv (one-time setup)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install project
git clone https://github.com/PKU-Alignment/Safe-Policy-Optimization.git
cd Safe-Policy-Optimization
uv sync
```

## Common Commands Comparison

| Task | Old Command | New Command |
|------|-------------|-------------|
| Install project | `pip install -e .` | `uv sync` |
| Install project without dev deps | `pip install .` | `uv sync --no-dev` |
| Install docs dependencies | `pip install -r docs/requirements.txt` | `uv sync --extra docs` |
| Run Python script | `python script.py` | `uv run python script.py` |
| Run tests | `pytest` | `uv run pytest` |
| Add dependency | `pip install package` | `uv add package` |
| Remove dependency | `pip uninstall package` | `uv remove package` |
| Update dependencies | `pip install --upgrade package` | `uv lock --upgrade` |

## Makefile Commands

All Makefile commands remain the same:

```bash
make install              # Install project (production)
make install-editable     # Install project (development)
make benchmark            # Run full benchmark
make simple-benchmark     # Run simple benchmark
make pytest               # Run tests
make docs                 # Build and serve documentation
```

## Working with Virtual Environments

### Automatic Management

uv automatically creates and manages a virtual environment in `.venv/`:

```bash
uv sync  # Creates .venv if it doesn't exist and installs dependencies
```

### Manual Activation (Optional)

If you want to use the virtual environment outside of `uv run`:

```bash
# On Linux/macOS
source .venv/bin/activate

# On Windows
.venv\Scripts\activate
```

However, with uv, you generally don't need to activate the environment. Just prefix commands with `uv run`:

```bash
uv run python safepo/single_agent/ppo_lag.py --task SafetyPointGoal1-v0
```

## Managing Dependencies

### Adding Dependencies

```bash
# Add a runtime dependency
uv add numpy scipy

# Add a dev dependency
uv add --group dev pytest black

# Add an optional dependency
uv add --optional docs sphinx
```

### Removing Dependencies

```bash
uv remove package-name
```

### Updating Dependencies

```bash
# Update all dependencies
uv lock --upgrade

# Update specific package
uv lock --upgrade-package package-name
```

## Migration Benefits

1. **Speed**: Installation is 10-100x faster than pip
2. **Reproducibility**: `uv.lock` ensures everyone uses the same dependency versions
3. **Simplicity**: One tool manages everything (Python versions, packages, environments)
4. **Modern**: Uses `pyproject.toml` (PEP 621) instead of legacy `setup.py`
5. **Compatibility**: Drop-in replacement for pip, no code changes needed

## Troubleshooting

### "uv: command not found"

Install uv:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Then restart your shell or run:
```bash
source $HOME/.cargo/env
```

### Virtual environment issues

Remove the virtual environment and recreate:
```bash
rm -rf .venv
uv sync
```

### Lock file conflicts

If you have merge conflicts in `uv.lock`:
```bash
uv lock --refresh
```

## For Maintainers

### Adding New Dependencies

1. Add to `pyproject.toml` under `dependencies` or `optional-dependencies`
2. Run `uv lock` to update the lock file
3. Commit both `pyproject.toml` and `uv.lock`

### Releasing New Versions

1. Update version in `pyproject.toml`
2. Build the package:
   ```bash
   uv build
   ```
3. Publish to PyPI:
   ```bash
   uv publish
   ```

## Still Using setup.py?

The old `setup.py` file can be kept for backward compatibility but is no longer needed for modern Python projects. All metadata is now in `pyproject.toml`.

If you need to maintain compatibility with older tools, you can keep `setup.py` as a minimal shim:

```python
from setuptools import setup
setup()
```

All configuration is read from `pyproject.toml` automatically.

## Additional Resources

- [uv Documentation](https://docs.astral.sh/uv/)
- [PEP 621 - Storing project metadata in pyproject.toml](https://peps.python.org/pep-0621/)
- [Python Packaging User Guide](https://packaging.python.org/)

## Questions?

If you encounter any issues during migration, please open an issue on GitHub.

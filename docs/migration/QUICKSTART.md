# Quick Start with uv

This guide helps you get started with Safe-Policy-Optimization using uv.

## Prerequisites

- Git
- A Unix-like system (Linux, macOS, or WSL on Windows)

## Installation

### 1. Install uv

Choose your platform:

**Linux and macOS:**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Windows:**
```powershell
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

After installation, restart your terminal or run:
```bash
source $HOME/.cargo/env
```

### 2. Clone and Install SafePO

```bash
git clone https://github.com/PKU-Alignment/Safe-Policy-Optimization.git
cd Safe-Policy-Optimization
uv sync
```

That's it! uv will:
- Detect the required Python version (3.8) from `.python-version`
- Create a virtual environment in `.venv/`
- Install all dependencies from `uv.lock`
- Install SafePO in development mode

## Running Your First Experiment

### Single Agent Example

```bash
cd safepo/single_agent
uv run python ppo_lag.py --task SafetyPointGoal1-v0 --seed 0
```

### Multi Agent Example

```bash
cd safepo/multi_agent
uv run python macpo.py --task Safety2x4AntVelocity-v0 --experiment quickstart
```

## Running Benchmarks

### Quick Benchmark (Recommended for Testing)

```bash
make simple-benchmark
```

This runs a subset of algorithms on selected environments.

### Full Benchmark (Takes Hours)

```bash
make benchmark
```

This runs all algorithms on all environments.

## Viewing Results

After running experiments:

```bash
# Plot results
cd safepo
uv run python plot.py --logdir ./runs/benchmark

# Evaluate performance
uv run python evaluate.py --benchmark-dir ./runs/benchmark
```

## Common Tasks

### Running Tests

```bash
make pytest
```

### Building Documentation

```bash
make docs
```

This will open the documentation in your browser.

### Installing Optional Dependencies

**For MuJoCo Support:**
```bash
uv sync --extra mujoco
```

**For Documentation Building:**
```bash
uv sync --extra docs
```

**For Development Tools:**
```bash
uv sync --extra dev
```

## Project Structure

```
Safe-Policy-Optimization/
├── safepo/                    # Main package
│   ├── single_agent/          # Single-agent algorithms
│   ├── multi_agent/           # Multi-agent algorithms
│   ├── common/                # Shared utilities
│   └── utils/                 # Helper functions
├── tests/                     # Unit tests
├── docs/                      # Documentation
├── pyproject.toml            # Project configuration
├── uv.lock                   # Dependency lock file
└── .python-version           # Python version specification
```

## Next Steps

- Read the [full documentation](https://safe-policy-optimization.readthedocs.io)
- Check out [MIGRATION_TO_UV.md](MIGRATION_TO_UV.md) for migration details
- Explore algorithm implementations in `safepo/single_agent/` and `safepo/multi_agent/`
- Read about [Safety-Gymnasium environments](https://github.com/PKU-Alignment/safety-gymnasium)

## Troubleshooting

### uv command not found

Make sure uv is in your PATH:
```bash
source $HOME/.cargo/env
```

Or reinstall uv.

### Import errors

Make sure you're using `uv run` to execute Python scripts:
```bash
uv run python script.py
```

Or activate the virtual environment:
```bash
source .venv/bin/activate  # Linux/macOS
.venv\Scripts\activate     # Windows
```

### Dependency issues

Try refreshing the lock file:
```bash
uv lock --refresh
```

Then reinstall:
```bash
rm -rf .venv
uv sync
```

## Getting Help

- [Open an issue](https://github.com/PKU-Alignment/Safe-Policy-Optimization/issues)
- Check the [documentation](https://safe-policy-optimization.readthedocs.io)
- Read the [uv documentation](https://docs.astral.sh/uv/)

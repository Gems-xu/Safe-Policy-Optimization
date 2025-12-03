print-%  : ; @echo $* = $($*)
PROJECT_NAME   = safepo
COPYRIGHT      = "PKU Alignment Team. All Rights Reserved."
PROJECT_PATH   = $(PROJECT_NAME)
SHELL          = /bin/bash
SOURCE_FOLDERS = $(PROJECT_PATH) tests docs
PYTHON_FILES   = $(shell find $(SOURCE_FOLDERS) -type f -name "*.py" -o -name "*.pyi")
COMMIT_HASH    = $(shell git log -1 --format=%h)
PATH           := $(HOME)/go/bin:$(PATH)
PYTHON         ?= $(shell command -v python3 || command -v python)
PYTESTOPTS     ?=
UV             ?= $(shell command -v uv)

.PHONY: default
default: install

# Installations

install:
	$(UV) sync --no-dev

install-editable:
	$(UV) sync

install-e: install-editable  # alias

docs-install:
	$(UV) sync --extra docs

pytest-install:
	$(UV) sync --group dev

# Benchmark

multi-benchmark:
	cd safepo/multi_agent && $(UV) run python benchmark.py --total-steps 10000000 --experiment benchmark

single-benchmark:
	cd safepo/single_agent && $(UV) run python benchmark.py --total-steps 10000000  --experiment benchmark

multi-simple-benchmark:
	cd safepo/multi_agent && $(UV) run python benchmark.py --total-steps 10000000 --experiment benchmark --tasks \
	 Safety2x4AntVelocity-v0 Safety4x2AntVelocity-v0 \
	 Safety2x3HalfCheetahVelocity-v0 Safety6x1HalfCheetahVelocity-v0 \

single-simple-benchmark:
	cd safepo/single_agent && $(UV) run python benchmark.py --total-steps 10000000  --experiment benchmark --tasks \
	 SafetyAntVelocity-v1 SafetyHumanoidVelocity-v1 \
	 SafetyPointGoal1-v0 SafetyCarButton1-v0 \

multi-test-benchmark:
	cd safepo/multi_agent && $(UV) run python benchmark.py --total-steps 2000 --experiment benchmark --num-envs 1 --tasks \
	 Safety2x4AntVelocity-v0 Safety4x2AntVelocity-v0 \
	 Safety2x3HalfCheetahVelocity-v0 Safety6x1HalfCheetahVelocity-v0 \

single-test-benchmark:
	cd safepo/single_agent && $(UV) run python benchmark.py --total-steps 2000  --experiment benchmark --num-envs 1 --steps-per-epoch 1000 --tasks \
	 SafetyAntVelocity-v1 SafetyHumanoidVelocity-v1 \
	 SafetyPointGoal1-v0 SafetyCarButton1-v0 \

plot:
	cd safepo && $(UV) run python plot.py --logdir ./runs/benchmark

eval:
	cd safepo && $(UV) run python evaluate.py --benchmark-dir ./runs/benchmark

simple-benchmark: install-editable multi-simple-benchmark single-simple-benchmark plot eval

test-benchmark: install-editable multi-test-benchmark single-test-benchmark plot eval

benchmark: install-editable multi-benchmark single-benchmark plot eval

pytest: pytest-install
	cd tests &&  \
	$(UV) run pytest --verbose --color=yes --durations=0 \
		--cov="../safepo" --cov-config=.coveragerc --cov-report=xml --cov-report=term-missing \
		$(PYTESTOPTS) . 

# Documentation

docs: docs-install
	$(UV) run sphinx-autobuild --watch $(PROJECT_PATH) --open-browser docs/source docs/build

spelling: docs-install
	$(UV) run sphinx-autobuild -b spelling docs/source docs/build
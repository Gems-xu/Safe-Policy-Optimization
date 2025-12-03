# Detailed installation instructions

## Installation with uv (Recommended)

SafePO now uses [uv](https://docs.astral.sh/uv/) for fast and reliable package management.

### Install uv

```bash
# On macOS and Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# On Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### Install SafePO

```bash
git clone https://github.com/PKU-Alignment/Safe-Policy-Optimization.git
cd Safe-Policy-Optimization
uv sync
```

This will automatically:
- Create a virtual environment with Python 3.8
- Install all dependencies
- Install SafePO in editable mode

### Install PyTorch (Optional, for specific CUDA versions)

If you need a specific CUDA version for PyTorch:

```bash
# Example for CUDA 11.1
uv pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 \
  --index-url https://download.pytorch.org/whl/torch_stable.html
```

### Install MPI for Python (Optional, for distributed training)

```bash
# On Ubuntu/Debian
sudo apt-get install libopenmpi-dev
uv pip install mpi4py

# On macOS
brew install open-mpi
uv pip install mpi4py
```

## install mujoco

```bash
# Download mujoco200 linux from http://www.roboti.us/download.html
# Download Activation key from http://www.roboti.us/license.html
mkdir ~/.mujoco
mv mujoco200_linux.zip ~/.mujoco
mv mjkey.txt ~/.mujoco/
cd ~/.mujoco
unzip mujoco200_linux.zip
mv mujoco200_linux mujoco200
cp mjkey.txt ~/.mujoco/mujoco200
cp mjkey.txt ~/.mujoco/

# Add following line to .bashrc:
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/.mujoco/mujoco200/bin

# Then:
source ~/.bashrc
uv pip install mujoco_py==2.0.2.7
```

## install IsaacGym

Details regarding installation of IsaacGym can be found [here](https://developer.nvidia.com/isaac-gym). We currently support the `Preview Release 3` version of IsaacGym.

Ensure that Isaac Gym works on your system by running one of the examples from the `python/examples` directory, like `joint_monkey.py`. Follow troubleshooting steps described in the Isaac Gym Preview 2 install instructions if you have any trouble running the samples.

# known issues

## problem1: libstdc++ version issue

```bash
# Error message:
ImportError: /lib/x86_64-linux-gnu/libstdc++.so.6: version `GLIBCXX_3.4.29' not found
```

**Solution:**
```bash
# Install a newer version of libstdc++
sudo apt-get update
sudo apt-get install libstdc++6

# Or if using uv's managed environment, ensure system libraries are up to date
```

## problem2: Missing OpenGL/OSMesa headers

```bash
# Error message:
fatal error: GL/osmesa.h: No such file or directory
```

**Solution:**
```bash
sudo apt install libosmesa6-dev
```

## problem3: Missing patchelf

```bash
# Error message:
error: [Errno 2] No such file or directory: 'patchelf'
```

**Solution:**
```bash
sudo apt-get -y install patchelf
```

#!/bin/bash
set -e

echo "Setting up environment for TPU..."

# 确保 uv 已安装
if ! command -v uv &> /dev/null; then
    echo "uv could not be found, installing..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    source $HOME/.cargo/env
fi

# 创建并激活虚拟环境
uv venv .venv
source .venv/bin/activate

# 安装基础依赖 (flax, optax, etc.)
# 注意：我们先不安装 pyproject.toml 中的 jax，以免覆盖 TPU 版本
# 但 uv sync 通常会根据 lock 文件安装。
# 这里我们直接使用 pip 模式安装 TPU 版本的 JAX，再安装其他包。

echo "Installing JAX for TPU (and other dependencies)..."

# 强制安装 TPU 版本的 JAX
# 参考: https://jax.readthedocs.io/en/latest/installation.html#pip-installation-google-cloud-tpu
uv pip install \
  --find-links https://storage.googleapis.com/jax-releases/libtpu_releases.html \
  "jax[tpu]" \
  "flax" \
  "optax" \
  "numpy"

echo "Installation complete. Active devices check:"
python -c "import jax; print(jax.devices())"

echo "Setup done. Run 'source .venv/bin/activate' then 'python benchmark_script.py'"

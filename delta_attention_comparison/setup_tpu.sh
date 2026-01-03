#!/bin/bash
set -e

echo "=== Setting up TPU environment using uv ==="

# 1. Check and Install uv
if ! command -v uv &> /dev/null; then
    echo ">> uv not found. Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    source "$HOME/.cargo/env"
else
    echo ">> uv is already installed: $(uv --version)"
fi

# 2. Setup Virtual Environment
echo ">> Creating/Updating virtual environment (.venv)..."
# Ensure we use a compatible python version if available, else uv defaults to system or managed
uv venv .venv --allow-existing

# Activate venv for the script execution
source .venv/bin/activate

# 3. Install Dependencies (JAX TPU + Project)
echo ">> Installing JAX[tpu] and dependencies..."

# We use --find-links for TPU wheels.
# We explicitly install jax[tpu] and libtpu to ensure they are picked up from the google bucket
# before installing the rest of the project which might depend on 'jax' generic.
# -e . installs the current project in editable mode.

uv pip install \
  --find-links https://storage.googleapis.com/jax-releases/libtpu_releases.html \
  "jax[tpu]" \
  "jaxlib" \
  "libtpu-nightly" \
  -e .

# 4. Verification
echo ">> Verifying Installation..."
python -c "import jax; print(f'JAX Version: {jax.__version__}'); print(f'Devices: {jax.devices()}')"

echo "=== Setup Complete ==="
echo "To activate the environment, run:"
echo "source .venv/bin/activate"
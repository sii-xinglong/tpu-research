# Gemini Project Context: TPU Research

## 1. Core Mandates
*   **Remote Verification:** Functional changes must be verified on actual hardware (TPU/GPU) via SkyPilot. Local correctness does not guarantee remote performance or correctness (especially for JAX/Pallas).
*   **Infrastructure via SkyPilot:** Use [SkyPilot](https://skypilot.readthedocs.io/) for all resource management. Do not attempt to SSH manually without understanding the SkyPilot sync mechanism.
*   **Package Management:** Use `uv` (not `pip`) for local dependency management.
*   **Scope Isolation:**
    *   **Target Codebase:** `delta_attention_comparison/` (JAX/Pallas). This is where new research happens.
    *   **Reference Codebase:** `fla/` (PyTorch/Triton). This is a submodule for reference/correctness checking. **Do not modify** unless explicitly instructed.

## 2. Architecture & Navigation

### Directory Structure
*   **`delta_attention_comparison/`**: **PRIMARY WORKSPACE**. Contains JAX implementations of attention mechanisms.
    *   `src/layers/`: Place new attention kernels (Pallas/JAX) here.
    *   `benchmark_script.py`: The main entry point for performance testing.
*   **`fla/`**: **READ-ONLY REFERENCE**. A Git submodule containing state-of-the-art Linear Attention models in PyTorch/Triton.
*   **`scripts/`**: Infrastructure automation.
    *   `launch_tpu.sh` / `launch_gpu.sh`: Provisions/Starts the TPU or GPU cluster.
    *   `sync_code.sh`: Forces code synchronization to the remote cluster.

### Key Technologies
*   **Framework:** JAX (Frontend), Pallas (Kernel Language).
*   **Hardware:** Google Cloud TPU (v4, v5e, v6e) & NVIDIA GPUs.
*   **Orchestrator:** SkyPilot.

## 3. Remote Development Workflow (TPU & GPU)

### Step 1: Provisioning
To start or update a cluster. Both scripts update their respective cluster name files.

**Option A: TPU (Primary)**
```bash
# Usage: bash scripts/launch_tpu.sh <accelerator_type> <experiment_name>
# Saves to .cluster_name_tpu
bash scripts/launch_tpu.sh tpu-v6e-1 matmul_bench
```

**Option B: GPU (Secondary/Baseline)**
```bash
# Usage: bash scripts/launch_gpu.sh <accelerator_type> <experiment_name>
# Saves to .cluster_name_gpu
bash scripts/launch_gpu.sh L4 matmul_bench
```

### Step 2: Synchronization & Execution
After modifying local code, run it on the remote instance. `scripts/sync_code.sh` auto-detects the most recently used cluster, or you can specify `tpu`/`gpu`.

**Standard Execution Pattern:**
```bash
# 1. Force sync (auto-detects newest)
bash scripts/sync_code.sh
# OR specify target:
# bash scripts/sync_code.sh tpu

# 2. Run benchmark on remote
# Note: Read from the correct file (.cluster_name_tpu or .cluster_name_gpu)
sky exec $(cat .cluster_name_tpu) "uv run --extra tpu python delta_attention_comparison/benchmark_script.py"
```

### Step 3: Debugging
*   **Status Check:** `sky status`
*   **Logs:** SkyPilot streams stdout/stderr to your local terminal.

## 4. Pallas Kernel Guide (JAX)

### Syntax Overview
*   **Imports:**
    ```python
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu
    ```
### Do Not Use
*   **Do not use any operations that require dynamic memory allocation or replace memory storage by set. suck like w.at[c].set(w1) **

### Core Concepts & Patterns

#### 1. Grid & Memory Hierarchy
*   **Layout:** The Pallas TPU lowering currently requires that the last two dimensions of your block shape are divisible by 8 and 128 respectively, or be equal to the respective dimensions  of the overall array.

*   **Grid Spec:** Use `pltpu.PrefetchScalarGridSpec` for automatic HBM<->VMEM management in simple cases.
    ```python
    grid_spec = pltpu.PrefetchScalarGridSpec(
        num_scalar_prefetch=0,
        grid=(batch_size, num_heads, num_blocks),
        in_specs=[pl.BlockSpec(memory_space=pltpu.VMEM, ...), ...],
        out_specs=[pl.BlockSpec(memory_space=pltpu.VMEM, ...), ...],
        scratch_shapes=[pltpu.VMEM(...)]
    )
    ```
*   **Manual Pipelining (Advanced):** For maximum performance, manage data movement manually to overlap Compute and DMA.
    *   Add a `stage` dimension to your grid (e.g., `grid=(..., 2)`).
    *   Use **Semaphores** (`pltpu.SemaphoreType.DMA`) to synchronize.
    *   Use `pltpu.make_async_copy(src, dst, sem).start()` to trigger transfers.
    *   **Pattern:**
        1.  Wait for previous stage's compute to finish (if dependent).
        2.  Start async load for *next* iteration.
        3.  Compute on *current* iteration's data.
        4.  Wait for current load to finish before next step.

#### 2. Control Flow
*   **Predicated Execution:** Use `@pl.when(condition)` heavily. It is more efficient than `jax.lax.cond` inside kernels for pipeline state management (e.g., handling the first/last block or specific pipeline stages).
    ```python
    @pl.when(stage_index == 0)
    def load_initial():
        ...
    ```

### Optimization Checklist
1.  **Block Size & Alignment:**
    *   Align block sizes to **128 bytes** (or at least 8 elements) for efficient DMA.
    *   Pad dimensions if necessary to maintain alignment.
    *   Standard sweet spot: **128x128** or **128xHEAD_DIM**.
2.  **Memory Types:**
    *   `pltpu.VMEM`: **Local (Fast)**. Most compute happens here.
    *   `pltpu.SMEM`: **Scalar (Shared)**. Use for cross-block reductions (e.g., accumulating global loss).
    *   `pltpu.HBM`: **Global (Slow)**. Avoid direct access; DMA to VMEM first.
3.  **Vectorization:** Avoid scalar loops. Operate on full blocks using `jnp` functions.

### Reference Material
*   **Tokamax:** The submodule `tokamax/` contains production-grade Pallas kernels.
    *   **Study File:** `tokamax/tokamax/_src/ops/attention/pallas_triton.py and tpu_inference_kernel/flash_attention/kernel.py`
    *   **Look for:** Implementation of manual pipelining (using `stage_index`), async copies, and semaphores. Do not import `tokamax` directly in research code; strictly use it as a reference for implementation patterns.
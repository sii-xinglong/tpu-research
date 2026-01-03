# Gemini Project Context: TPU Research

## Project Goal
This project is dedicated to conducting research and benchmarking experiments on Google Cloud TPUs (Tensor Processing Units). The primary objective is to implement, test, and optimize models (specifically Delta Attention variants and Qwen models) on TPU hardware.

## Development & Verification Paradigm
**CRITICAL:** All code changes must be verified on an actual TPU server. Do not assume local correctness for TPU-specific operations (JAX/Pallas kernels). The project use uv instead of pip.

### 1. Infrastructure Management
We use [SkyPilot](https://skypilot.readthedocs.io/) to manage TPU resources.

*   **Launch Script:** `scripts/launch_tpu.sh`
    *   **Usage:** `bash scripts/launch_tpu.sh <accelerator_type> <experiment_name>`
    *   **Example:** `bash scripts/launch_tpu.sh tpu-v6e-1 matmul_bench`
    *   **Function:** This script renders the `tpu_resource.sky.yaml` template, launches the cluster, and saves the active cluster name to `.cluster_name`.

### 2. Workflow
1.  **Modify Code:** Make necessary changes to the codebase locally.
2.  **Launch/Update Cluster:**
    *   If no cluster is running, use `scripts/launch_tpu.sh`.
    *   **Note:** SkyPilot automatically syncs the `workdir` (current project root) to the remote cluster upon launch/exec.
3.  **Execute & Verify:**
    *   **Manual Sync:** Run `scripts/sync_code.sh` to force a code sync. This is useful before SSH-ing or if you suspect files haven't updated. It works by triggering `sky exec` which automatically performs an rsync.
    *   Use `sky exec` to run commands on the TPU instance.
    *   **Command Pattern:** `sky exec $(cat .cluster_name) "source .venv/bin/activate && python <script_path>"`
    *   **Example:** `sky exec $(cat .cluster_name) "source .venv/bin/activate && python delta_attention_comparison/benchmark_script.py"`
4.  **Iterate:**
    *   If tests fail, modify local code.
    *   Re-run verification via `sky exec` (SkyPilot will sync the latest local changes to the remote `workdir` before execution).

## Key Files
*   `scripts/launch_tpu.sh`: Main entry point for infrastructure.
*   `scripts/sync_code.sh`: Helper script that reads `.cluster_name` and executes a minimal `sky exec` command ("echo") to force SkyPilot to rsync local changes to the remote cluster.
*   `scripts/tpu_resource.sky.yaml`: SkyPilot configuration template.
*   `.cluster_name`: Stores the name of the currently active/last-launched cluster.

## Mandates for Gemini
1.  **Always Verify on TPU:** When asked to implement or fix TPU-related code, always propose a plan that includes running the code on the TPU via `sky exec`.
2.  **Use the Script:** Always use `scripts/launch_tpu.sh` for provisioning.
3.  **Check Status:** Use `sky status` to verify cluster availability before attempting execution.

## Pallas Kernel: Syntax, Features, and Optimization

### 1. Overview
Pallas is a JAX primitive that allows writing custom kernels for accelerators (TPU/GPU) using a subset of JAX. It provides fine-grained control over memory hierarchy (HBM vs. VMEM/SMEM) and execution grids, enabling high-performance implementations of operations like FlashAttention.

### 2. Syntax & Structure

*   **Imports:**
    ```python
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu
    ```

*   **Kernel Definition:**
    Decorated with `jax.jit` or called via `pl.pallas_call`.
    ```python
    def kernel_func(
        # Refs to inputs/outputs/scratch in VMEM/SMEM
        q_ref, k_ref, v_ref, o_ref,
        # Scalar args
        sm_scale, block_size
    ):
        ...
    ```

*   **Invocation:**
    ```python
    pl.pallas_call(
        kernel_func,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=...,
            grid=grid, # Tuple defining the launch grid (e.g., (batch, heads, blocks))
            in_specs=[pl.BlockSpec(...), ...], # Mapping HBM -> VMEM
            out_specs=[pl.BlockSpec(...), ...],
            scratch_shapes=[pltpu.VMEM(...), ...],
        ),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", ...), # "parallel" or "arbitrary"
            vmem_limit_bytes=...,
        ),
        out_shape=...
    )(q, k, v)
    ```

*   **BlockSpec:**
    Defines how a block of data is mapped from HBM to VMEM for a specific grid index.
    ```python
    def index_map(batch_idx, head_idx, block_idx):
        return (batch_idx, head_idx, block_idx * block_size, 0)

    pl.BlockSpec(
        block_shape=(batch, heads, block_size, head_dim),
        index_map=index_map
    )
    ```

### 3. Core Features

*   **Program Identity:**
    *   `pl.program_id(axis)`: Get current index in the launch grid.
    *   `pl.num_programs(axis)`: Get total size of the grid dimension.

*   **Memory Spaces:**
    *   `pltpu.HBM`: High Bandwidth Memory (Global).
    *   `pltpu.VMEM`: Vector Memory (Local to TPU core).
    *   `pltpu.SMEM`: Scalar Memory (limited size).

*   **Control Flow:**
    *   `@pl.when(condition)`: Conditional execution (if-block).
    *   `@pl.loop(start, stop, step)`: For-loop.
    *   `jax.lax.cond`, `jax.lax.select`, `jax.lax.while_loop`: Standard JAX control flow.

*   **Data Movement:**
    *   Automatic: Via `in_specs` / `out_specs` in `pl.pallas_call`.
    *   Manual (Async):
        ```python
        # Create a semaphore for synchronization
        sem = pltpu.SemaphoreType.DMA((4, 2))
        # Async copy
        cp = pltpu.make_async_copy(src_ref, dst_ref, sem)
        cp.start()
        cp.wait()
        ```

*   **Compute:**
    *   Standard `jax.numpy` operations (`jnp.matmul`, `jnp.exp`, `jnp.sum`, etc.).
    *   `jax.lax.dot_general`: For matrix multiplication with precision control.

### 4. Optimization Strategies

*   **Pipelining:**
    *   Overlap Compute and Data Transfer (DMA).
    *   Use **Semaphores** and **Async Copies**.
    *   Divide the loop into stages (e.g., Load Next, Compute Current, Store Previous).
    *   Use `sem.wait()` strategically to ensure data is ready only when needed.

*   **Double Buffering:**
    *   Allocate scratch space for *current* and *next* iteration (e.g., `shape=(2, block_size, ...)`).
    *   Compute on buffer `i % 2` while loading into `(i + 1) % 2`.

*   **Tiling (Blocking):**
    *   Process large tensors in small blocks that fit in VMEM.
    *   Typical block sizes: 128x128, 128xHEAD_DIM.
    *   Align block sizes with hardware lanes (multiples of 128 or 8).

*   **Vectorization:**
    *   Pallas kernels are implicitly vectorized over the `BlockSpec` shape.
    *   Avoid scalar loops inside the kernel where possible; use `jnp` array operations.

*   **Memory Layout:**
    *   Ensure data is contiguous in memory for efficient DMA.
    *   Use `padded` shapes if necessary to maintain alignment.
    *   Bitcasting (`pltpu.bitcast`) can be used for zero-cost type reinterpretation.
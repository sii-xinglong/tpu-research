# Gemini Project Context: TPU Research

## Project Goal
This project is dedicated to conducting research and benchmarking experiments on Google Cloud TPUs (Tensor Processing Units). The primary objective is to implement, test, and optimize models (specifically Delta Attention variants and Qwen models) on TPU hardware.

## Development & Verification Paradigm
**CRITICAL:** All code changes must be verified on an actual TPU server. Do not assume local correctness for TPU-specific operations (JAX/Pallas kernels).

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

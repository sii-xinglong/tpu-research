# Pallas Rewrite Plan for Kimi Delta Attention

## Objective
Optimize the `KimiDeltaAttention` implementation in `delta_attention_comparison/src/layers/kimi_delta_attention.py` using **JAX Pallas**.

The primary goal is to accelerate the **Intra-Chunk** computation phase. This phase currently uses `jax.vmap` over a Python function `compute_chunk_vars` which contains a `jax.lax.fori_loop` for matrix inversion. Replacing this with a custom Pallas kernel allows for optimized memory management (SRAM/VMEM), fused operations, and potentially better utilization of TPU TensorCore (MXU) units by handling block-sparse operations more explicitly.

## Scope
The refactoring will focus on the `chunk_parallel_delta_attention` function.

1.  **Phase 1: Intra-Chunk Kernel (`pallas_chunk_intra`)**
    *   **Input:** `key` (K), `value` (V), `beta`, `g` (decay).
    *   **Output:** `w`, `u`.
    *   **Logic:**
        *   Load chunk data $(64 \times D)$ into VMEM.
        *   Compute local decay mask.
        *   Compute $A = \text{tril}(\beta K K^T)$.
        *   Solve $(I + A) u = V$ and $(I + A) w = (K \odot \text{decay})$ using forward substitution (or explicit inverse computation).
    *   **Parallelism:** This kernel is perfectly parallelizable across `(Batch, Heads, Num_Chunks)`.

2.  **Phase 2: Integration**
    *   Integrate the Pallas kernel into the main `chunk_parallel_delta_attention` function, replacing the `compute_vmap` call.

*Note: The Inter-Chunk recurrence (Scanning) will initially remain as `jax.lax.scan`. XLA handles linear scans efficiently, and implementing a sequential dependency in Pallas requires a different kernel structure (e.g., loading the entire sequence or manual loop over chunks) which adds complexity for potentially marginal gain compared to the $O(C^2)$ intra-chunk bottleneck.*

## Implementation Details

### 1. Kernel Interface

```python
def kda_intra_kernel(
    # Refs to global memory
    k_ref,      # [B, H, N_chunks, C, K_dim]
    v_ref,      # [B, H, N_chunks, C, V_dim]
    g_ref,      # [B, H, N_chunks, C, K_dim] (or pre-computed cumsum)
    beta_ref,   # [B, H, N_chunks, C]
    # Output Refs
    u_out_ref,  # [B, H, N_chunks, C, V_dim]
    w_out_ref,  # [B, H, N_chunks, C, K_dim]
    # Hyperparams
    chunk_size: int,
    head_dim: int,
):
    ...
```

### 2. Grid & Blocking
*   **Grid:** `(Batch, Num_Heads, Num_Chunks)`
*   **Block Spec:** Each program instance loads **one complete chunk** (e.g., $C=64$ or $C=128$).
*   **Memory Strategy:**
    *   $K, V, G$ are loaded into VMEM.
    *   $A$ matrix ($C \times C$) computed in VMEM.
    *   Inversion/Solver runs in VMEM (using unrolled loops or Pallas primitives).

### 3. Solver Logic (Pallas)
Standard JAX uses `fori_loop`. In Pallas on TPU:
*   We can compute $A_{local} = \text{tril}(\dots)$.
*   We can implement **Forward Substitution** to compute $u = (I+A)^{-1}V$ directly without materializing full $T = (I+A)^{-1}$ if desired, or materializing $T$ if needed for both $u$ and $w$.
*   Since $C$ is small (64), explicit triangular solve:
    ```python
    # Pseudo-code for Forward Sub in Pallas
    # u = v.copy()
    # for i in range(C):
    #     row_A = A[i, :i]
    #     val_u = u[:i]
    #     correction = dot(row_A, val_u)
    #     u[i] -= correction
    ```
    *Note: Matrix multiplication limits on TPU (padding to 128x128) might require careful handling or using `pallas.dot` with appropriate tiling if $C$ is small.*

## Plan of Action

1.  **Create File:** `delta_attention_comparison/src/layers/pallas_kda.py`.
2.  **Implement Kernel:** Write the `kda_intra_pallas` function using `jax.experimental.pallas`.
3.  **Implement Wrapper:** Create `pallas_chunk_parallel_delta_attention` that prepares inputs (reshaping, padding) and calls the kernel.
4.  **Test:** Create a test script comparing the output of the original JAX function vs. the Pallas version to ensure numerical correctness.
5.  **Benchmark:** Compare execution time (theoretical or actual via `sky exec`) if possible.

## Dependencies
*   `jax >= 0.4.30` (for stable Pallas support)
*   TPU environment (Pallas TPU kernels generally don't run on CPU).

## Algorithm Correction for Pallas
The original code computes `T = (I + A)^-1`.
Then `u = T @ v` and `w = T @ (k * exp(g))`.
Computing `T` explicitly is fine for $C=64$.
Algorithm:
1. Load $K, \beta, G$.
2. Compute $G_{diff} \to \text{Decay}$.
3. Compute $A_{raw} = (K K^T) \odot \text{Decay} \odot \text{Mask}$.
4. $A = A_{raw} \times \beta_{expanded}$.
5. Invert $(I+A)$. Since $A$ is strictly lower triangular, $I+A$ is lower triangular with unit diagonal.
   $L = I+A$. Solve $L T = I$.
   This is equivalent to: $T_{row_i} = e_i - \sum_{j<i} L_{ij} T_{row_j}$.
   Since $L_{ij} = A_{ij}$ for $j<i$ and $L_{ii}=1$.
   $T_{row_i} = e_i - A[i, :i] @ T[:i]$.
6. Compute $u, w$ via matmul.

This is efficient in Pallas.

```
import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import functools
import math

# --- Helper Function for Block Inversion ---

def invert_unit_lower_triangular(mat):
    """
    Computes the inverse of a unit lower triangular matrix (implied unit diagonal)
    where 'mat' contains the strictly lower triangular part.
    The actual matrix is M = I + mat.
    We compute M^{-1}.
    
    Algorithm: Recursive (unrolled) Block Divide & Conquer
    M = [[A, 0], [C, B]] -> M^{-1} = [[A^{-1}, 0], [-B^{-1} C A^{-1}, B^{-1}]]
    Here A, B are unit lower triangular, C is square.
    """
    # mat is (N, N) strictly lower triangular
    n = mat.shape[0]
    
    # We maintain the inverse T as block diagonal matrices.
    # At step s, we have blocks of size s x s on the diagonal which are inverses
    # of the corresponding s x s blocks of M.
    # We merge pairs of blocks to get blocks of size 2s x 2s.
    
    # Initially, we have N blocks of size 1x1.
    # The diagonal elements of M are 1. The inverse of [1] is [1].
    # So T is initialized to Identity (conceptually N 1x1 blocks of [1]).
    # We represent this as a (num_blocks, s, s) array where s=1.
    
    t_blocks = jnp.ones((n, 1, 1), dtype=jnp.float32)
    
    # Iterate scale s: 1 -> 2 -> 4 -> ... -> N/2
    steps = int(math.log2(n))
    
    # To handle non-power-of-2 N, padding would be needed, but KDA chunks are typically 64/128.
    
    for i in range(steps):
        s = 1 << i
        num_blocks = n // (2 * s)
        
        # Current T blocks are size s x s. Pair them up.
        # t_blocks shape: (2 * num_blocks, s, s)
        # We pair (2k, 2k+1).
        # Top-Left (A^{-1}) is even indices.
        # Bottom-Right (B^{-1}) is odd indices.
        
        t_pairs = t_blocks.reshape(num_blocks, 2, s, s)
        t_tl = t_pairs[:, 0] # (num_blocks, s, s) -- A^{-1}
        t_br = t_pairs[:, 1] # (num_blocks, s, s) -- B^{-1}
        
        # We need the sub-block C from the original matrix M (which is mat).
        # C corresponds to the bottom-left block of the current 2s x 2s merge.
        # In the full matrix, for block index b, this is:
        # rows: b*2s + s  to b*2s + 2s
        # cols: b*2s      to b*2s + s
        
        # Reshape mat to expose these blocks.
        # (num_blocks, 2s, num_blocks, 2s)
        # We want the diagonal blocks in the (num_blocks, num_blocks) grid.
        # Then slice the BL sub-block.
        
        # Since 'mat' is large, we want to avoid full reshapes if possible, 
        # but in Pallas/TPU reshape is metadata.
        # Extracting the diagonal blocks:
        # We want `mat_reshaped[b, :, b, :]`.
        
        # Extract diagonal blocks and compute updates manually to avoid Pallas/Mosaic issues.
        # We iterate explicitly because batched matmul on stacked slices causes vectorization errors
        # ('vector.multi_reduction' incompatible types) on TPU.
        update_list = []
        for b in range(num_blocks):
            # C corresponds to the BL sub-block of the b-th 2s x 2s block on the diagonal
            row_start = b * 2 * s + s
            col_start = b * 2 * s
            
            # Extract C block directly from mat (static slice)
            c_block = mat[row_start : row_start + s, col_start : col_start + s]
            c_block = c_block.astype(jnp.float32)
            
            # Extract T blocks for this index
            t_tl_b = t_tl[b]
            t_br_b = t_br[b]
            
            # Compute update: - B^{-1} @ C @ A^{-1}
            upd_b = -jnp.matmul(
                jnp.matmul(t_br_b, c_block, precision=lax.Precision.HIGHEST),
                t_tl_b, 
                precision=lax.Precision.HIGHEST
            )
            update_list.append(upd_b)
            
        update = jnp.stack(update_list)
        
        # Construct the new T blocks of size 2s x 2s.
        # [[t_tl, 0], [update, t_br]]
        
        zeros = jnp.zeros((num_blocks, s, s), dtype=jnp.float32)
        
        # Concatenate along columns then rows
        top = jnp.concatenate([t_tl, zeros], axis=2)    # (num_blocks, s, 2s)
        bot = jnp.concatenate([update, t_br], axis=2)   # (num_blocks, s, 2s)
        t_new = jnp.concatenate([top, bot], axis=1)     # (num_blocks, 2s, 2s)
        
        t_blocks = t_new
        
    return t_blocks[0]

# --- Pallas Kernel Definition ---

def kda_intra_kernel(
    # Input Refs
    k_ref,      # [C, D]
    v_ref,      # [C, V_D] (V_head_dim)
    g_cumsum_ref, # [C, D]
    beta_ref,   # [C]
    
    # Output Refs
    u_out_ref,  # [C, V_D]
    w_out_ref,  # [C, D]
    
    # Hyperparameters
    chunk_size: int,
    head_dim: int,
    v_head_dim: int,
):
    # 1. Load Inputs
    # Load all into VMEM. C=64/128 is small enough.
    k = k_ref[...].squeeze((0, 1, 2))
    v = v_ref[...].squeeze((0, 1, 2))
    g = g_cumsum_ref[...].squeeze((0, 1, 2))
    beta = beta_ref[...].squeeze((0, 1, 2))

    # 2. Compute A matrix
    # A_ij = beta_i * sum_d (K_id * K_jd * exp(g_id - g_jd))  for i > j
    # K_left = K * exp(g)
    # K_right = K * exp(-g)
    k_left = k * jnp.exp(g)
    k_right = k * jnp.exp(-g)
    
    # Compute Raw A: [C, C]
    # We use high precision for matrix multiplication accumulation if possible
    a_raw = jnp.matmul(k_left, k_right.T, precision=lax.Precision.HIGHEST)
    
    # Apply Beta: A[i, j] *= beta[i]
    a_raw = a_raw * beta[:, None]
    
    # Apply Causal Mask (Strictly Lower Triangular)
    # i > j
    idx = jnp.arange(chunk_size)
    mask = idx[:, None] > idx[None, :]
    a_masked = jnp.where(mask, a_raw, 0.0)
    
    # 3. Solve for T = (I + A)^-1
    # Use block divide and conquer inversion which is O(log N) depth on TPU
    # instead of O(N) sequential row updates.
    # Ensure float32 for stability.
    
    t_matrix = invert_unit_lower_triangular(a_masked.astype(jnp.float32))
    
    # Cast back if necessary (though matmul below likely prefers float32 accum)
    t_matrix = t_matrix.astype(k.dtype)
    
    # Apply Beta column scaling as per original logic: T_final_ij = T_ij * beta_j
    t_final = t_matrix * beta[None, :] 
    
    # 4. Compute Outputs
    # u = T_final @ v_blk
    # w = T_final @ (k_blk * exp(g_blk))
    
    u_val = jnp.matmul(t_final, v, precision=lax.Precision.HIGHEST)
    w_val = jnp.matmul(t_final, k * jnp.exp(g), precision=lax.Precision.HIGHEST)
    
    # 5. Store
    u_out_ref[...] = u_val[None, None, None, :, :]
    w_out_ref[...] = w_val[None, None, None, :, :]

# --- Wrapper Function ---

def pallas_chunk_kda_intra(
    key: jax.Array,       # [B, H, N, C, D]
    value: jax.Array,     # [B, H, N, C, V_D]
    g_cumsum: jax.Array,  # [B, H, N, C, D]
    beta: jax.Array,      # [B, H, N, C]
) -> tuple[jax.Array, jax.Array]:
    
    batch_size, num_heads, num_chunks, chunk_size, head_dim = key.shape
    v_head_dim = value.shape[-1]
    
    # Output shapes
    u_shape = value.shape
    w_shape = key.shape
    
    # Grid: (B, H, N)
    grid = (batch_size, num_heads, num_chunks)
    
    # Block Specs
    # We map the grid directly to the dimensions.
    # Inside kernel, we handle (C, D) entirely.
    
    in_specs = [
        pl.BlockSpec(index_map=lambda i, j, k: (i, j, k, 0, 0), block_shape=(1, 1, 1, chunk_size, head_dim)),   # key
        pl.BlockSpec(index_map=lambda i, j, k: (i, j, k, 0, 0), block_shape=(1, 1, 1, chunk_size, v_head_dim)), # value
        pl.BlockSpec(index_map=lambda i, j, k: (i, j, k, 0, 0), block_shape=(1, 1, 1, chunk_size, head_dim)),   # g_cumsum
        pl.BlockSpec(index_map=lambda i, j, k: (i, j, k, 0),    block_shape=(1, 1, 1, chunk_size)),             # beta
    ]
    
    out_specs = [
        pl.BlockSpec(index_map=lambda i, j, k: (i, j, k, 0, 0), block_shape=(1, 1, 1, chunk_size, v_head_dim)), # u_out
        pl.BlockSpec(index_map=lambda i, j, k: (i, j, k, 0, 0), block_shape=(1, 1, 1, chunk_size, head_dim)),   # w_out
    ]
    
    # Kernel Call
    u_out, w_out = pl.pallas_call(
        functools.partial(kda_intra_kernel, chunk_size=chunk_size, head_dim=head_dim, v_head_dim=v_head_dim),
        out_shape=[
            jax.ShapeDtypeStruct(u_shape, value.dtype),
            jax.ShapeDtypeStruct(w_shape, key.dtype)
        ],
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            in_specs=in_specs,
            out_specs=out_specs,
            grid=grid,
        ),
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel", "parallel", "parallel"))
    )(key, value, g_cumsum, beta)
    
    return u_out, w_out

# --- Full Attention Replacement ---

def chunk_parallel_delta_attention_pallas(
    query: jax.Array,
    key: jax.Array,
    value: jax.Array,
    g: jax.Array,
    beta: jax.Array,
    chunk_size: int = 64,
    initial_state: None | jax.Array = None,
    output_final_state: bool = False,
) -> tuple[jax.Array, None | jax.Array]:
  initial_dtype = query.dtype

  query = jnp.transpose(query, (0, 2, 1, 3)).astype(jnp.float32)
  key = jnp.transpose(key, (0, 2, 1, 3)).astype(jnp.float32)
  value = jnp.transpose(value, (0, 2, 1, 3)).astype(jnp.float32)
  g = jnp.transpose(g, (0, 2, 1, 3)).astype(jnp.float32)
  beta = jnp.transpose(beta, (0, 2, 1)).astype(jnp.float32)

  batch_size, num_heads, sequence_length, k_head_dim = key.shape
  v_head_dim = value.shape[-1]
  pad_size = (chunk_size - sequence_length % chunk_size) % chunk_size

  if pad_size > 0:
    pad_config_4d = ((0, 0), (0, 0), (0, pad_size), (0, 0))
    pad_config_3d = ((0, 0), (0, 0), (0, pad_size))
    query = jnp.pad(query, pad_config_4d)
    key = jnp.pad(key, pad_config_4d)
    value = jnp.pad(value, pad_config_4d)
    g = jnp.pad(g, pad_config_4d)
    beta = jnp.pad(beta, pad_config_3d)

  total_sequence_length = sequence_length + pad_size
  scale = k_head_dim ** -0.5
  query = query * scale

  num_chunks = total_sequence_length // chunk_size
  
  def to_chunk(x):
      new_shape = (batch_size, num_heads, num_chunks, chunk_size) + x.shape[3:]
      return x.reshape(new_shape)

  query_c = to_chunk(query)
  key_c = to_chunk(key)
  value_c = to_chunk(value)
  g_c = to_chunk(g)
  beta_c = beta.reshape(batch_size, num_heads, num_chunks, chunk_size)

  g_cumsum = jnp.cumsum(g_c, axis=-2)
  
  # --- PALLAS INJECTION START ---
  # Replace the vmap(compute_chunk_vars) with Pallas kernel
  # u_c, w_c = compute_vmap(key_c, g_cumsum, beta_c, value_c)
  
  u_c, w_c = pallas_chunk_kda_intra(key_c, value_c, g_cumsum, beta_c)
  
  # --- PALLAS INJECTION END ---

  def to_scan(x): return jnp.transpose(x, (2, 0, 1, 3, 4))
  
  if initial_state is None:
      last_recurrent_state = jnp.zeros((batch_size, num_heads, k_head_dim, v_head_dim), dtype=jnp.float32)
  else:
      last_recurrent_state = initial_state

  xs = (
      to_scan(query_c), 
      to_scan(key_c), 
      to_scan(u_c), 
      to_scan(w_c), 
      to_scan(g_cumsum)
  )

  def scan_body(prev_state, x):
      q_i, k_i, u_i, w_i, g_i = x
      prec = jax.lax.Precision.HIGHEST
      
      g_diff = jnp.expand_dims(g_i, -2) - jnp.expand_dims(g_i, -3)
      decay_full = jnp.exp(g_diff)
      
      idx = jnp.arange(chunk_size)
      
      mask = idx[:, None] >= idx[None, :] 
      g_rel = jnp.where(jnp.expand_dims(mask, -1), decay_full, 0.0)
      
      attn_local = jnp.einsum('...ik, ...jk, ...ijk -> ...ij', q_i, k_i, g_rel)
      
      correction = jnp.matmul(w_i, prev_state, precision=prec)
      v_new = u_i - correction
      
      o_hist = jnp.matmul(q_i * jnp.exp(g_i), prev_state, precision=prec)
      o_intra = jnp.matmul(attn_local, v_new, precision=prec)
      o_block = o_hist + o_intra
      
      decay_last = jnp.exp(g_i[..., -1, :])
      S_decayed = prev_state * jnp.expand_dims(decay_last, -1)
      
      k_tail = k_i * jnp.exp(jnp.expand_dims(g_i[..., -1, :], -2) - g_i)
      update_term = jnp.matmul(jnp.swapaxes(k_tail, -1, -2), v_new, precision=prec)
      
      new_state = S_decayed + update_term
      
      return new_state, o_block

  final_state, core_attn_out_stacked = jax.lax.scan(scan_body, last_recurrent_state, xs)

  core_attn_out = jnp.transpose(core_attn_out_stacked, (1, 2, 0, 3, 4))
  core_attn_out = core_attn_out.reshape(batch_size, num_heads, -1, v_head_dim)
  core_attn_out = core_attn_out[:, :, :sequence_length, :]
  core_attn_out = jnp.transpose(core_attn_out, (0, 2, 1, 3)).astype(initial_dtype)

  return core_attn_out, final_state if output_final_state else None
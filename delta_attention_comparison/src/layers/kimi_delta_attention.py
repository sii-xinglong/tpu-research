from typing import Optional, Tuple

import jax
import jax.numpy as jnp
from flax import nnx

from src.common_types import (
    Array,
    Config,
    DType,
)
from src.initializers import (
    nd_dense_init,
    NdInitializer,
    default_bias_init,
)
from src.linears import DenseGeneral
from src.normalizations import l2norm, RMSNorm

def chunk_parallel_delta_attention(
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
  
  def compute_chunk_vars(k_blk, g_blk, beta_blk, v_blk):
      prec = jax.lax.Precision.HIGHEST

      # Optimization: e^(g_i - g_j) = e^g_i * e^(-g_j)
      # Avoids materializing (C, C, D) tensor
      k_left = k_blk * jnp.exp(g_blk)
      k_right = k_blk * jnp.exp(-g_blk)
      
      # (C, D) @ (D, C) -> (C, C)
      a_raw_full = jnp.matmul(k_left, k_right.T, precision=prec)
      
      idx = jnp.arange(chunk_size)
      mask = idx[:, None] > idx[None, :] 
      A_raw = jnp.where(mask, a_raw_full, 0.0)

      A = A_raw * jnp.expand_dims(beta_blk, -1)
      
      A_neg = -A
      
      def invert_body(i, m):
          row = m[i]
          mask_idx = jnp.arange(chunk_size) < i
          row = jnp.where(mask_idx, row, 0.0)
          increment = jnp.dot(row, m, precision=prec)
          increment = jnp.where(mask_idx, increment, 0.0)
          return m.at[i].set(row + increment)

      A_inv = jax.lax.fori_loop(1, chunk_size, invert_body, A_neg)
      
      T = A_inv + jnp.eye(chunk_size)
      T_final = T * jnp.expand_dims(beta_blk, -2) 
      
      u = jnp.matmul(T_final, v_blk, precision=prec)
      w = jnp.matmul(T_final, k_blk * jnp.exp(g_blk), precision=prec)
      
      return u, w

  compute_vmap = jax.vmap(jax.vmap(jax.vmap(compute_chunk_vars)))
  u_c, w_c = compute_vmap(key_c, g_cumsum, beta_c, value_c)

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
      
      # Optimization: e^(g_i - g_j) = e^g_i * e^(-g_j)
      # attn_local = (Q * exp(g)) @ (K * exp(-g)).T
      q_left = q_i * jnp.exp(g_i)
      k_right = k_i * jnp.exp(-g_i)
      
      # (B, NH, C, D) @ (B, NH, D, C) -> (B, NH, C, C)
      attn_local_full = jnp.matmul(q_left, jnp.swapaxes(k_right, -1, -2), precision=prec)
      
      idx = jnp.arange(chunk_size)
      mask = idx[:, None] >= idx[None, :] 
      attn_local = jnp.where(jnp.expand_dims(mask, (0, 1)), attn_local_full, 0.0)
      
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


class FusedRMSNormGated(nnx.Module):
  def __init__(
      self,
      dim: int,
      eps: float = 1e-6,
      activation: str = "sigmoid",
      dtype: DType = jnp.float32,
      rngs: Optional[nnx.Rngs] = None,
  ):
    self.activation = activation
    self.dtype = dtype
    self.rms_norm = RMSNorm(
        num_features=dim,
        epsilon=eps,
        dtype=dtype,
        rngs=rngs,
    )

  def __call__(self, x: Array, gate: Array) -> Array:
    normalized_x = self.rms_norm(x)
    if self.activation == "sigmoid":
      g = jax.nn.sigmoid(gate.astype(jnp.float32))
    elif self.activation in ("silu", "swish"):
      g = jax.nn.silu(gate.astype(jnp.float32))
    else:
      g = gate
    return (normalized_x * g).astype(self.dtype)


class KimiDeltaAttention(nnx.Module):
  def __init__(
      self,
      hidden_size: int,
      num_heads: int,
      head_dim: int,
      conv_kernel_size: int = 4,
      normalization_layer_epsilon: float = 1e-5,
      dtype: DType = jnp.float32,
      weight_dtype: DType = jnp.float32,
      kernel_init: NdInitializer = nd_dense_init(1.0, "fan_in", "normal"),
      rngs: Optional[nnx.Rngs] = None,
  ):
    self.hidden_size = hidden_size
    self.num_heads = num_heads
    self.head_dim = head_dim
    self.conv_kernel_size = conv_kernel_size
    self.normalization_layer_epsilon = normalization_layer_epsilon
    self.dtype = dtype
    self.weight_dtype = weight_dtype
    self.kernel_init = kernel_init

    self.q_proj = DenseGeneral(
        in_features_shape=(hidden_size,), out_features_shape=(num_heads*head_dim,),
        kernel_init=kernel_init, dtype=dtype, weight_dtype=weight_dtype, use_bias=False, rngs=rngs,
    )
    self.k_proj = DenseGeneral(
        in_features_shape=(hidden_size,), out_features_shape=(num_heads*head_dim,),
        kernel_init=kernel_init, dtype=dtype, weight_dtype=weight_dtype, use_bias=False, rngs=rngs,
    )
    self.v_proj = DenseGeneral(
        in_features_shape=(hidden_size,), out_features_shape=(num_heads*head_dim,),
        kernel_init=kernel_init, dtype=dtype, weight_dtype=weight_dtype, use_bias=False, rngs=rngs,
    )

    conv_dim = num_heads * head_dim
    conv_kwargs = {
        "in_features": conv_dim,
        "out_features": conv_dim,
        "kernel_size": (conv_kernel_size,),
        "feature_group_count": conv_dim,
        "padding": "CAUSAL",
        "use_bias": False,
        "dtype": dtype,
        "rngs": rngs,
    }
    self.q_conv1d = nnx.Conv(**conv_kwargs)
    self.k_conv1d = nnx.Conv(**conv_kwargs)
    self.v_conv1d = nnx.Conv(**conv_kwargs)

    self.b_proj = DenseGeneral(
        in_features_shape=(hidden_size,), out_features_shape=(num_heads,),
        kernel_init=kernel_init, dtype=dtype, weight_dtype=weight_dtype, use_bias=False, rngs=rngs,
    )

    self.f_a_proj = DenseGeneral(
        in_features_shape=(hidden_size,), out_features_shape=(head_dim,),
        kernel_init=kernel_init, dtype=dtype, weight_dtype=weight_dtype, use_bias=False, rngs=rngs,
    )
    self.f_b_proj = DenseGeneral(
        in_features_shape=(head_dim,), out_features_shape=(num_heads*head_dim),
        kernel_init=kernel_init, dtype=dtype, weight_dtype=weight_dtype, use_bias=False, rngs=rngs,
    )
    self.g_a_proj = DenseGeneral(
        in_features_shape=(hidden_size,), out_features_shape=(head_dim,),
        kernel_init=kernel_init, dtype=dtype, weight_dtype=weight_dtype, use_bias=False, rngs=rngs,
    )
    self.g_b_proj = DenseGeneral(
        in_features_shape=(head_dim,), out_features_shape=(num_heads*head_dim),
        kernel_init=kernel_init, dtype=dtype, weight_dtype=weight_dtype, use_bias=False, rngs=rngs,
    )

    def a_log_init(key, shape, dtype=jnp.float32):
      return jnp.log(jax.random.uniform(key, shape=shape, dtype=dtype, minval=1e-9, maxval=16.0))

    self.A_log = nnx.Param(a_log_init(rngs.params(), (1,1,num_heads,1)))
    self.dt_bias = nnx.Param(nnx.initializers.ones(rngs.params(), (num_heads*head_dim), dtype=jnp.float32))

    self.o_norm = FusedRMSNormGated(
        dim=head_dim, eps=self.normalization_layer_epsilon, activation="sigmoid", dtype=dtype, rngs=rngs,
    )
    self.o_proj = DenseGeneral(
        in_features_shape=(num_heads*head_dim), out_features_shape=(hidden_size,),
        kernel_init=kernel_init, dtype=dtype, weight_dtype=weight_dtype, use_bias=False, rngs=rngs,
    )

  def apply_fused_kda_gate(self, g_linear: Array) -> Array:
    """Computes log-space forget gate."""
    b, s, _ = g_linear.shape
    g = g_linear + self.dt_bias
    sp = jax.nn.softplus(g.astype(jnp.float32)).reshape(b, s, self.num_heads, self.head_dim)
    return (-jnp.exp(self.A_log) * sp).astype(self.dtype)

  def __call__(
      self,
      hidden_states: Array,
      chunk_size: int = 64,
      initial_state: Optional[Array] = None,
      output_final_state: bool = False,
  ) -> Tuple[Array, Optional[Array]]:
    batch, seq_len, _ = hidden_states.shape

    q = l2norm(self.q_proj(hidden_states).reshape(batch, seq_len, self.num_heads, -1), dim=-1, eps=1e-6)
    k = l2norm(self.k_proj(hidden_states).reshape(batch, seq_len, self.num_heads, -1), dim=-1, eps=1e-6)
    v = self.v_proj(hidden_states).reshape(batch, seq_len, self.num_heads, -1)

    def apply_conv(x, conv_layer):
      batch, seq_len, num_heads, head_dim = x.shape
      x_flat = x.reshape(batch, seq_len, -1)
      out = conv_layer(x_flat)
      out = jax.nn.silu(out.astype(jnp.float32)).astype(self.dtype)
      return out.reshape(batch, seq_len, num_heads, head_dim)

    q = apply_conv(q, self.q_conv1d)
    k = apply_conv(k, self.k_conv1d)
    v = apply_conv(v, self.v_conv1d)

    beta = jax.nn.sigmoid(self.b_proj(hidden_states).astype(jnp.float32)).astype(self.dtype)
    g_forget = self.apply_fused_kda_gate(self.f_b_proj(self.f_a_proj(hidden_states)))

    attn_out, final_state = chunk_parallel_delta_attention(
        query=q, key=k, value=v, g=g_forget, beta=beta,
        chunk_size=chunk_size, initial_state=initial_state, output_final_state=output_final_state
    )

    g_output = self.g_b_proj(self.g_a_proj(hidden_states)).reshape(batch, seq_len, self.num_heads, self.head_dim)
    out = self.o_norm(attn_out, g_output)
    out = out.reshape(batch, seq_len, -1)
    
    return self.o_proj(out), final_state


def analyze_kimi_operators(config: Config, batch_size: int, seq_len: int, chunk_size: int = 64):
  try:
    from jax.experimental import roofline
  except ImportError:
    print("jax.experimental.roofline not available.")
    return []

  rngs = nnx.Rngs(0)
  model = KimiDeltaAttention(
      hidden_size=config.hidden_size,
      num_heads=config.num_heads,
      head_dim=config.head_dim,
      rngs=rngs,
  )

  stats = []
  
  # Dummy inputs
  hidden_shape = jax.ShapeDtypeStruct((batch_size, seq_len, config.hidden_size), jnp.float32)
  head_dim_shape = jax.ShapeDtypeStruct((batch_size, seq_len, config.head_dim), jnp.float32) # For intermediate gate
  conv_in_shape = jax.ShapeDtypeStruct((batch_size, seq_len, config.num_heads * config.head_dim), jnp.float32)

  # 1. QKV Proj
  def run_proj(proj, x): return proj(x)
  
  q_flops, q_bytes = 0, 0
  for proj_name in ['q_proj', 'k_proj', 'v_proj']:
    proj = getattr(model, proj_name)
    _, res = roofline.roofline(lambda x: proj(x))(hidden_shape)
    q_flops += res.flops
    q_bytes += res.hbm_bytes
  stats.append({"name": "Kimi: QKV Proj", "flops": q_flops, "bytes": q_bytes})

  # 2. Gate Projs (B, F_A, F_B, G_A, G_B)
  g_flops, g_bytes = 0, 0
  
  # B Proj (H -> NH)
  _, res_b = roofline.roofline(lambda x: model.b_proj(x))(hidden_shape)
  g_flops += res_b.flops
  g_bytes += res_b.hbm_bytes
  
  # F_A (H -> D), G_A (H -> D)
  _, res_fa = roofline.roofline(lambda x: model.f_a_proj(x))(hidden_shape)
  _, res_ga = roofline.roofline(lambda x: model.g_a_proj(x))(hidden_shape)
  g_flops += res_fa.flops + res_ga.flops
  g_bytes += res_fa.hbm_bytes + res_ga.hbm_bytes

  # F_B (D -> NH*D), G_B (D -> NH*D)
  # Input is output of A projs: (B, L, D)
  _, res_fb = roofline.roofline(lambda x: model.f_b_proj(x))(head_dim_shape)
  _, res_gb = roofline.roofline(lambda x: model.g_b_proj(x))(head_dim_shape)
  g_flops += res_fb.flops + res_gb.flops
  g_bytes += res_fb.hbm_bytes + res_gb.hbm_bytes
  
  stats.append({"name": "Kimi: Gate Projs", "flops": g_flops, "bytes": g_bytes})

  # 3. Conv1D (Q, K, V)
  c_flops, c_bytes = 0, 0
  for conv_name in ['q_conv1d', 'k_conv1d', 'v_conv1d']:
    conv = getattr(model, conv_name)
    # Conv input needs to be (B, L, C)
    _, res = roofline.roofline(lambda x: conv(x))(conv_in_shape)
    c_flops += res.flops
    c_bytes += res.hbm_bytes
  stats.append({"name": "Kimi: Conv1D", "flops": c_flops, "bytes": c_bytes})

  # 4. Attn Core - Fine Grained Analysis
  nh = config.num_heads
  d = config.head_dim
  
  q_shape = jax.ShapeDtypeStruct((batch_size, seq_len, nh, d), jnp.float32)
  beta_shape = jax.ShapeDtypeStruct((batch_size, seq_len, nh), jnp.float32)

  # --- Part 4.1: Prep (Transpose, Pad, Chunk, Cumsum) ---
  def core_prep(query, key, value, g, beta):
      # Replicate logic from chunk_parallel_delta_attention
      query = jnp.transpose(query, (0, 2, 1, 3)).astype(jnp.float32)
      key = jnp.transpose(key, (0, 2, 1, 3)).astype(jnp.float32)
      value = jnp.transpose(value, (0, 2, 1, 3)).astype(jnp.float32)
      g = jnp.transpose(g, (0, 2, 1, 3)).astype(jnp.float32)
      beta = jnp.transpose(beta, (0, 2, 1)).astype(jnp.float32)

      batch_size, num_heads, sequence_length, k_head_dim = key.shape
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
      return query_c, key_c, value_c, g_cumsum, beta_c

  out_shapes_prep, res_prep = roofline.roofline(core_prep)(q_shape, q_shape, q_shape, q_shape, beta_shape)
  stats.append({"name": "Kimi: Core Prep", "flops": res_prep.flops, "bytes": res_prep.hbm_bytes})
  
  # Extract shapes for next stage
  # out_shapes_prep is a tuple matching return of core_prep
  q_c_shape, k_c_shape, v_c_shape, g_cumsum_shape, beta_c_shape = out_shapes_prep

  # --- Part 4.2: Init (Compute Chunk Vars) - Fine Grained Breakdown ---
  
  # We need to construct inputs for the vmapped function.
  # k_c: (B, NH, NC, C, D)
  # g_cumsum: (B, NH, NC, C, D) -> Wait, g is (B, NH, NC, C, D) but cumsum?
  # In prep: g_cumsum = jnp.cumsum(g_c, axis=-2). Shape matches g_c.
  # beta_c: (B, NH, NC, C) (No D) -> check prep: beta.reshape(..., C)
  # v_c: (B, NH, NC, C, D)
  
  # For breakdown, we will define vmapped functions for each stage.
  
  # 4.2.1 Decay Computation (Optimized out)
  # def init_decay(g_blk):
  #     ...
  
  # 4.2.2 A_raw Einsum (Optimized)
  def init_araw_optimized(k_blk, g_blk):
      # Optimization: e^(g_i - g_j) = e^g_i * e^(-g_j)
      # A_raw_ij = sum_d (K_id * K_jd * exp(g_id - g_jd))
      #          = sum_d (K_id * exp(g_id)) * (K_jd * exp(-g_jd))
      #          = (K * exp(g)) @ (K * exp(-g)).T
      
      prec = jax.lax.Precision.HIGHEST
      
      # k_blk: (C, D), g_blk: (C, D)
      
      k_left = k_blk * jnp.exp(g_blk)
      k_right = k_blk * jnp.exp(-g_blk)
      
      # (C, D) @ (D, C) -> (C, C)
      a_raw_full = jnp.matmul(k_left, k_right.T, precision=prec)
      
      # Apply causal mask (lower triangular)
      # i >= j. 
      idx = jnp.arange(chunk_size)
      mask = idx[:, None] >= idx[None, :]
      a_raw = jnp.where(mask, a_raw_full, 0.0)
      
      return a_raw

  out_shapes_araw, res_araw = roofline.roofline(jax.vmap(jax.vmap(jax.vmap(init_araw_optimized))))(k_c_shape, g_cumsum_shape)
  stats.append({"name": "Kimi: Init A_raw (Opt)", "flops": res_araw.flops, "bytes": res_araw.hbm_bytes})
  
  a_raw_shape = out_shapes_araw

  # --- Atomic Analysis for A_raw (Opt) ---
  pad_size = (chunk_size - seq_len % chunk_size) % chunk_size
  total_seq_len = seq_len + pad_size
  num_chunks = total_seq_len // chunk_size
  total_chunks = batch_size * nh * num_chunks
  
  k_single = jax.ShapeDtypeStruct((chunk_size, config.head_dim), jnp.float32)
  g_single = jax.ShapeDtypeStruct((chunk_size, config.head_dim), jnp.float32)
  
  _, res_araw_atomic = roofline.roofline(init_araw_optimized)(k_single, g_single)
  stats.append({
      "name": "Kimi: Init A_raw (Opt Atomic)", 
      "flops": res_araw_atomic.flops * total_chunks, 
      "bytes": res_araw_atomic.hbm_bytes * total_chunks
  })

  # 4.2.3 A_inv Loop
  def init_ainv(A_raw, beta_blk):
      prec = jax.lax.Precision.HIGHEST
      A = A_raw * jnp.expand_dims(beta_blk, -1)
      A_neg = -A
      
      def invert_body(i, m):
          row = m[i]
          mask_idx = jnp.arange(chunk_size) < i
          row = jnp.where(mask_idx, row, 0.0)
          increment = jnp.dot(row, m, precision=prec)
          increment = jnp.where(mask_idx, increment, 0.0)
          return m.at[i].set(row + increment)

      return jax.lax.fori_loop(1, chunk_size, invert_body, A_neg)

  out_shapes_ainv, res_ainv = roofline.roofline(jax.vmap(jax.vmap(jax.vmap(init_ainv))))(a_raw_shape, beta_c_shape)
  stats.append({"name": "Kimi: Init A_inv", "flops": res_ainv.flops, "bytes": res_ainv.hbm_bytes})
  
  a_inv_shape = out_shapes_ainv

  # --- Atomic Analysis for A_inv ---
  a_raw_single = jax.ShapeDtypeStruct((chunk_size, chunk_size), jnp.float32)
  beta_single = jax.ShapeDtypeStruct((chunk_size,), jnp.float32)
  
  _, res_ainv_atomic = roofline.roofline(init_ainv)(a_raw_single, beta_single)
  stats.append({
      "name": "Kimi: Init A_inv (Atomic)", 
      "flops": res_ainv_atomic.flops * total_chunks, 
      "bytes": res_ainv_atomic.hbm_bytes * total_chunks
  })


  # 4.2.4 U/W Computation
  def init_uw(A_inv, beta_blk, v_blk, k_blk, g_blk):
      prec = jax.lax.Precision.HIGHEST
      T = A_inv + jnp.eye(chunk_size)
      T_final = T * jnp.expand_dims(beta_blk, -2) 
      u = jnp.matmul(T_final, v_blk, precision=prec)
      w = jnp.matmul(T_final, k_blk * jnp.exp(g_blk), precision=prec)
      return u, w

  out_shapes_uw, res_uw = roofline.roofline(jax.vmap(jax.vmap(jax.vmap(init_uw))))(a_inv_shape, beta_c_shape, v_c_shape, k_c_shape, g_cumsum_shape)
  stats.append({"name": "Kimi: Init U/W", "flops": res_uw.flops, "bytes": res_uw.hbm_bytes})
  
  # For Scan input:
  u_c_shape, w_c_shape = out_shapes_uw

  # --- Part 4.3: Scan ---
  def core_scan(query_c, key_c, u_c, w_c, g_cumsum):
      def to_scan(x): return jnp.transpose(x, (2, 0, 1, 3, 4))
      
      batch_size = query_c.shape[0]
      num_heads = query_c.shape[1]
      k_head_dim = query_c.shape[-1]
      v_head_dim = u_c.shape[-1]
      
      last_recurrent_state = jnp.zeros((batch_size, num_heads, k_head_dim, v_head_dim), dtype=jnp.float32)

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
          
          # Optimization: e^(g_i - g_j) = e^g_i * e^(-g_j)
          q_left = q_i * jnp.exp(g_i)
          k_right = k_i * jnp.exp(-g_i)
          
          attn_local_full = jnp.matmul(q_left, jnp.swapaxes(k_right, -1, -2), precision=prec)
          
          idx = jnp.arange(chunk_size)
          mask = idx[:, None] >= idx[None, :] 
          attn_local = jnp.where(jnp.expand_dims(mask, (0, 1)), attn_local_full, 0.0)
          
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
      return core_attn_out_stacked

  out_shapes_scan, res_scan = roofline.roofline(core_scan)(q_c_shape, k_c_shape, u_c_shape, w_c_shape, g_cumsum_shape)
  stats.append({"name": "Kimi: Core Scan", "flops": res_scan.flops, "bytes": res_scan.hbm_bytes})

  # --- Part 4.4: Post (Reshape) ---
  core_out_stacked_shape = out_shapes_scan
  
  def core_post(core_attn_out_stacked):
      batch_size = core_attn_out_stacked.shape[1] # Transposed in scan? No, scan output is (ScanDim, Batch, Head, Chunk, Dim)
      # Wait, input to scan was (ScanDim, Batch, Head, Chunk, Dim).
      # scan output stacked matches input structure usually.
      # Original code: 
      # xs = (to_scan(query_c)...) -> (NumChunks, Batch, Heads, ChunkSize, Dim)
      # scan returns stacked of whatever scan_body returns.
      # scan_body returns o_block: (Batch, Heads, ChunkSize, Dim)
      # So stacked: (NumChunks, Batch, Heads, ChunkSize, Dim)
      
      # Replicate logic:
      # core_attn_out = jnp.transpose(core_attn_out_stacked, (1, 2, 0, 3, 4))
      # core_attn_out = core_attn_out.reshape(batch_size, num_heads, -1, v_head_dim)
      # core_attn_out = core_attn_out[:, :, :sequence_length, :]
      # core_attn_out = jnp.transpose(core_attn_out, (0, 2, 1, 3)).astype(jnp.float32)
      
      # We need to get dimensions from shape or config
      
      core_attn_out = jnp.transpose(core_attn_out_stacked, (1, 2, 0, 3, 4))
      # (B, NH, NC, C, D)
      B = core_attn_out.shape[0]
      NH = core_attn_out.shape[1]
      D = core_attn_out.shape[-1]
      
      core_attn_out = core_attn_out.reshape(B, NH, -1, D)
      core_attn_out = core_attn_out[:, :, :seq_len, :]
      core_attn_out = jnp.transpose(core_attn_out, (0, 2, 1, 3)).astype(jnp.float32)
      return core_attn_out

  _, res_post = roofline.roofline(core_post)(core_out_stacked_shape)
  stats.append({"name": "Kimi: Core Post", "flops": res_post.flops, "bytes": res_post.hbm_bytes})


  # 5. Out Proj
  # Input (B, L, NH*D)
  _, res_out = roofline.roofline(lambda x: model.o_proj(x))(conv_in_shape)
  stats.append({"name": "Kimi: Out Proj", "flops": res_out.flops, "bytes": res_out.hbm_bytes})

  return stats
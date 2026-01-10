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
      # (batch_size, num_heads, num_chunks, chunk_size) -> (batch_size, num_heads, num_chunks, chunk_size, head_dim)
      new_shape = (batch_size, num_heads, num_chunks, chunk_size) + x.shape[3:]
      return x.reshape(new_shape)

  query_c = to_chunk(query)
  key_c = to_chunk(key)
  value_c = to_chunk(value)
  g_c = to_chunk(g)
  beta_c = beta.reshape(batch_size, num_heads, num_chunks, chunk_size)

  g_cumsum = jnp.cumsum(g_c, axis=-2)
  
  #(batch_size, num_heads, num_chunks, chunk_size, head_dim) -> (num_chunks, batch_size, num_heads, chunk_size, head_dim)
  def to_scan(x): return jnp.transpose(x, (2, 0, 1, 3, 4))
  
  if initial_state is None:
      last_recurrent_state = jnp.zeros((batch_size, num_heads, k_head_dim, v_head_dim), dtype=jnp.float32)
  else:
      last_recurrent_state = initial_state

  # Prepare scan inputs. Note beta_c rank is one less than others.
  xs = (
      to_scan(query_c), 
      to_scan(key_c), 
      to_scan(value_c), 
      to_scan(g_cumsum),
      jnp.transpose(beta_c, (2, 0, 1, 3)) # (chunks, batch, heads, chunk_size)
  )

  def scan_body(prev_state, x):
      q_i, k_i, v_i, g_i, beta_i = x
      prec = jax.lax.Precision.HIGHEST
      
      # --- Fused Local Chunk Computation ---
      def compute_chunk_vars_local(k_blk, g_blk, beta_blk, v_blk):
          # Re-using the exact logic from original compute_chunk_vars
          # k_blk: (C, D), g_blk: (C, D), beta_blk: (C), v_blk: (C, D)
          
          g_diff = g_blk[:, None, :] - g_blk[None, :, :]
          idx = jnp.arange(chunk_size)
          mask = idx[:, None] > idx[None, :] 
          safe_g_diff = jnp.where(jnp.expand_dims(mask, -1), g_diff, -float('inf'))
          
          term = (k_blk[:, None, :] * k_blk[None, :, :]) * jnp.exp(safe_g_diff)
          A_raw = jnp.sum(term, axis=-1)
          A = A_raw * jnp.expand_dims(beta_blk, -1)
          
          eye = jnp.eye(chunk_size, dtype=A.dtype)
          L = eye + A
          T = jax.scipy.linalg.solve_triangular(L, eye, lower=True)
          T_final = T * jnp.expand_dims(beta_blk, -2) 
          
          u = jnp.matmul(T_final, v_blk, precision=prec)
          w = jnp.matmul(T_final, k_blk * jnp.exp(g_blk), precision=prec)
          return u, w

      # vmap over Batch (0) and Heads (1)
      # k_i: (B, H, C, D)
      u_i, w_i = jax.vmap(jax.vmap(compute_chunk_vars_local))(k_i, g_i, beta_i, v_i)
      # -------------------------------------------------------------
      
      # Stable calculation
      # attn_local_ij = q_i * k_j * exp(g_i - g_j)
      
      # g_i shape: (B, H, C, D)
      # g_diff: (B, H, C, C, D)
      g_diff = jnp.expand_dims(g_i, 3) - jnp.expand_dims(g_i, 2)
      
      idx = jnp.arange(chunk_size)
      mask = idx[:, None] >= idx[None, :] 
      mask_expanded = jnp.expand_dims(mask, (0, 1)) # (1, 1, C, C)
      mask_broad = jnp.expand_dims(mask_expanded, -1) # (1, 1, C, C, 1)

      # Mask positive exponents (i < j) to avoid overflow
      safe_g_diff = jnp.where(mask_broad, g_diff, -float('inf'))
      
      # q_i: (B, H, C, D) -> (B, H, C, 1, D)
      # k_i: (B, H, C, D) -> (B, H, 1, C, D)
      # term: (B, H, C, C, D)
      term = jnp.expand_dims(q_i, 3) * jnp.expand_dims(k_i, 2) * jnp.exp(safe_g_diff)
      
      attn_local = jnp.sum(term, axis=-1)
      
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

  # Use gradient checkpointing to avoid storing O(Chunk^2) intermediates for the entire sequence
  scan_body = jax.checkpoint(scan_body)

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
      num_v_heads: Optional[int] = None,
      conv_kernel_size: int = 4,
      normalization_layer_epsilon: float = 1e-5,
      dtype: DType = jnp.float32,
      weight_dtype: DType = jnp.float32,
      kernel_init: NdInitializer = nd_dense_init(1.0, "fan_in", "normal"),
      rngs: Optional[nnx.Rngs] = None,
  ):
    self.hidden_size = hidden_size
    self.num_heads = num_heads
    self.num_v_heads = num_v_heads if num_v_heads is not None else num_heads
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
        in_features_shape=(hidden_size,), out_features_shape=(self.num_v_heads*head_dim,),
        kernel_init=kernel_init, dtype=dtype, weight_dtype=weight_dtype, use_bias=False, rngs=rngs,
    )

    conv_dim = num_heads * head_dim
    conv_kwargs = {
        "kernel_size": (conv_kernel_size,),
        "padding": "CAUSAL",
        "use_bias": False,
        "dtype": dtype,
        "rngs": rngs,
    }
    self.q_conv1d = nnx.Conv(in_features=conv_dim, out_features=conv_dim, feature_group_count=conv_dim, **conv_kwargs)
    self.k_conv1d = nnx.Conv(in_features=conv_dim, out_features=conv_dim, feature_group_count=conv_dim, **conv_kwargs)
    
    v_conv_dim = self.num_v_heads * head_dim
    self.v_conv1d = nnx.Conv(in_features=v_conv_dim, out_features=v_conv_dim, feature_group_count=v_conv_dim, **conv_kwargs)

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
        in_features_shape=(head_dim,), out_features_shape=(self.num_v_heads*head_dim),
        kernel_init=kernel_init, dtype=dtype, weight_dtype=weight_dtype, use_bias=True, rngs=rngs,
    )

    def a_log_init(key, shape, dtype=jnp.float32):
      return jnp.log(jax.random.uniform(key, shape=shape, dtype=dtype, minval=1e-9, maxval=16.0))

    self.A_log = nnx.Param(a_log_init(rngs.params(), (1,1,num_heads,1)))
    self.dt_bias = nnx.Param(nnx.initializers.ones(rngs.params(), (num_heads*head_dim), dtype=jnp.float32))

    self.o_norm = FusedRMSNormGated(
        dim=head_dim, eps=self.normalization_layer_epsilon, activation="sigmoid", dtype=dtype, rngs=rngs,
    )
    self.o_proj = DenseGeneral(
        in_features_shape=(self.num_v_heads*head_dim), out_features_shape=(hidden_size,),
        kernel_init=kernel_init, dtype=dtype, weight_dtype=weight_dtype, use_bias=False, rngs=rngs,
    )

  def apply_fused_kda_gate(self, g_linear: Array) -> Array:
    """Computes log-space forget gate."""
    b, s, _ = g_linear.shape
    g = g_linear + self.dt_bias
    sp = jax.nn.softplus(g.astype(jnp.float32)).reshape(b, s, self.num_heads, self.head_dim)
    return -jnp.exp(self.A_log) * sp

  def __call__(
      self,
      hidden_states: Array,
      chunk_size: int = 64,
      initial_state: Optional[Array] = None,
      output_final_state: bool = False,
  ) -> Tuple[Array, Optional[Array]]:
    batch, seq_len, _ = hidden_states.shape

    q = self.q_proj(hidden_states).reshape(batch, seq_len, self.num_heads, -1)
    k = self.k_proj(hidden_states).reshape(batch, seq_len, self.num_heads, -1)
    v = self.v_proj(hidden_states).reshape(batch, seq_len, self.num_v_heads, -1)

    def apply_conv(x, conv_layer):
      batch, seq_len, num_heads, head_dim = x.shape
      x_flat = x.reshape(batch, seq_len, -1)
      out = conv_layer(x_flat)
      out = jax.nn.silu(out.astype(jnp.float32))
      return out.reshape(batch, seq_len, num_heads, head_dim)

    q = apply_conv(q, self.q_conv1d)
    k = apply_conv(k, self.k_conv1d)
    v = apply_conv(v, self.v_conv1d)

    q = l2norm(q, dim=-1, eps=1e-6)
    k = l2norm(k, dim=-1, eps=1e-6)

    beta = jax.nn.sigmoid(self.b_proj(hidden_states).astype(jnp.float32))
    g_forget = self.apply_fused_kda_gate(self.f_b_proj(self.f_a_proj(hidden_states)))
    
    # Repeat for MQA/GQA if num_v_heads > num_heads
    if self.num_v_heads > self.num_heads:
        assert self.num_v_heads % self.num_heads == 0
        n_rep = self.num_v_heads // self.num_heads
        q = jnp.repeat(q, n_rep, axis=2)
        k = jnp.repeat(k, n_rep, axis=2)
        g_forget = jnp.repeat(g_forget, n_rep, axis=2)
        beta = jnp.repeat(beta, n_rep, axis=2)

    attn_out, final_state = chunk_parallel_delta_attention(
        query=q, key=k, value=v, g=g_forget, beta=beta,
        chunk_size=chunk_size, initial_state=initial_state, output_final_state=output_final_state
    )

    g_output = self.g_b_proj(self.g_a_proj(hidden_states)).reshape(batch, seq_len, self.num_v_heads, self.head_dim)
    out = self.o_norm(attn_out, g_output)
    out = out.reshape(batch, seq_len, -1)
    
    return self.o_proj(out), final_state


def analyze_kimi_operators(config: Config, batch_size: int, seq_len: int, chunk_size: int = 128):
  try:
    from jax.experimental import roofline
  except ImportError:
    print("jax.experimental.roofline not available.")
    return []

  rngs = nnx.Rngs(0)
  
  # Extract config
  hidden_size = getattr(config, 'hidden_size', 2048)
  num_heads = getattr(config, 'num_heads', 16)
  head_dim = getattr(config, 'head_dim', 128)
  num_v_heads = getattr(config, 'num_kv_heads', num_heads) 
  if not num_v_heads:
      num_v_heads = num_heads
  if hasattr(config, 'gdn_num_value_heads'):
      num_v_heads = config.gdn_num_value_heads
      
  model = KimiDeltaAttention(
      hidden_size=hidden_size,
      num_heads=num_heads,
      head_dim=head_dim,
      num_v_heads=num_v_heads,
      dtype=jnp.bfloat16,
      weight_dtype=jnp.bfloat16,
      rngs=rngs,
  )

  stats = []
  
  # Inputs
  hidden_shape = jax.ShapeDtypeStruct((batch_size, seq_len, hidden_size), jnp.bfloat16)

  # Helper for Fwd/Bwd with NNX handling
  def analyze_module(name, module, input_shape):
      # Split module into graph and params
      graph, params = nnx.split(module)
      
      # Abstract params for roofline
      abstract_params = jax.tree.map(lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype), params)
      
      def apply_fn(p, x):
          m = nnx.merge(graph, p)
          return m(x)

      # Forward
      _, res_fwd = roofline.roofline(apply_fn)(abstract_params, input_shape)
      stats.append({
          "name": name, 
          "flops": res_fwd.flops or res_fwd.unfused_flops, 
          "bytes": res_fwd.hbm_bytes or res_fwd.unfused_hbm_bytes,
          "peak_bytes": res_fwd.peak_hbm_bytes
      })
      
      # Backward (w.r.t Params AND Input)
      def loss_wrapper(p, x):
          out = apply_fn(p, x)
          if isinstance(out, tuple): out = out[0]
          return jnp.sum(out)
      
      try:
          # Differentiate w.r.t params (0) and input (1)
          grad_fn = jax.grad(loss_wrapper, argnums=(0, 1))
          _, res_bwd = roofline.roofline(grad_fn)(abstract_params, input_shape)
          stats.append({
              "name": f"{name} (Bwd)", 
              "flops": res_bwd.flops or res_bwd.unfused_flops, 
              "bytes": res_bwd.hbm_bytes or res_bwd.unfused_hbm_bytes,
              "peak_bytes": res_bwd.peak_hbm_bytes
          })
      except Exception as e:
          print(f"Backward analysis failed for {name}: {e}")

  # 1. QKV Proj
  analyze_module("Kimi: Q Proj", model.q_proj, hidden_shape)
  analyze_module("Kimi: K Proj", model.k_proj, hidden_shape)
  analyze_module("Kimi: V Proj", model.v_proj, hidden_shape)

  # 2. Conv1D
  q_dim = num_heads * head_dim
  conv_shape_q = jax.ShapeDtypeStruct((batch_size, seq_len, q_dim), jnp.bfloat16)
  analyze_module("Kimi: Q Conv", model.q_conv1d, conv_shape_q)
  analyze_module("Kimi: K Conv", model.k_conv1d, conv_shape_q)

  v_dim = num_v_heads * head_dim
  conv_shape_v = jax.ShapeDtypeStruct((batch_size, seq_len, v_dim), jnp.bfloat16)
  analyze_module("Kimi: V Conv", model.v_conv1d, conv_shape_v)
  
  # 3. Gate Projs
  analyze_module("Kimi: Beta Proj", model.b_proj, hidden_shape)
  analyze_module("Kimi: F_A Proj", model.f_a_proj, hidden_shape)
  
  f_a_dim = head_dim
  hidden_shape_f_a = jax.ShapeDtypeStruct((batch_size, seq_len, f_a_dim), jnp.bfloat16)
  analyze_module("Kimi: F_B Proj", model.f_b_proj, hidden_shape_f_a)
  
  analyze_module("Kimi: G_A Proj", model.g_a_proj, hidden_shape)
  analyze_module("Kimi: G_B Proj", model.g_b_proj, hidden_shape_f_a)

  # 4. Core Attention (Pure Function)
  def run_core(q, k, v, g, beta):
      return chunk_parallel_delta_attention(q, k, v, g, beta, chunk_size=chunk_size)

  eff_heads = num_v_heads if num_v_heads > num_heads else num_heads
  
  q_core = jax.ShapeDtypeStruct((batch_size, seq_len, eff_heads, head_dim), jnp.bfloat16)
  k_core = jax.ShapeDtypeStruct((batch_size, seq_len, eff_heads, head_dim), jnp.bfloat16)
  v_core = jax.ShapeDtypeStruct((batch_size, seq_len, num_v_heads, head_dim), jnp.bfloat16)
  g_core = jax.ShapeDtypeStruct((batch_size, seq_len, eff_heads, head_dim), jnp.bfloat16)
  beta_core = jax.ShapeDtypeStruct((batch_size, seq_len, eff_heads), jnp.bfloat16)
  
  # Core has no params, so we just diff w.r.t inputs
  def analyze_pure(name, fn, *inputs):
      _, res_fwd = roofline.roofline(fn)(*inputs)
      stats.append({
          "name": name, 
          "flops": res_fwd.flops or res_fwd.unfused_flops, 
          "bytes": res_fwd.hbm_bytes or res_fwd.unfused_hbm_bytes,
          "peak_bytes": res_fwd.peak_hbm_bytes
      })
      
      def loss(*args):
          out = fn(*args)
          if isinstance(out, tuple): out = out[0]
          return jnp.sum(out)
      
      # Diff w.r.t all inputs
      argnums = tuple(range(len(inputs)))
      grad_fn = jax.grad(loss, argnums=argnums)
      _, res_bwd = roofline.roofline(grad_fn)(*inputs)
      stats.append({
          "name": f"{name} (Bwd)", 
          "flops": res_bwd.flops or res_bwd.unfused_flops, 
          "bytes": res_bwd.hbm_bytes or res_bwd.unfused_hbm_bytes,
          "peak_bytes": res_bwd.peak_hbm_bytes
      })

  analyze_pure("Kimi: Core", run_core, q_core, k_core, v_core, g_core, beta_core)

  # 5. Out Proj
  out_in_shape = jax.ShapeDtypeStruct((batch_size, seq_len, num_v_heads * head_dim), jnp.bfloat16)
  analyze_module("Kimi: Out Proj", model.o_proj, out_in_shape)
  
  # 6. Global Peak (Entire Module Call)
  def full_call(p, x):
      m = nnx.merge(graph_full, p)
      return m(x)[0]
  
  graph_full, params_full = nnx.split(model)
  abstract_params_full = jax.tree.map(lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype), params_full)
  
  _, res_full_fwd = roofline.roofline(full_call)(abstract_params_full, hidden_shape)
  stats.append({
      "name": "Kimi: FULL (Fwd)", 
      "flops": res_full_fwd.flops or res_full_fwd.unfused_flops, 
      "bytes": res_full_fwd.hbm_bytes or res_full_fwd.unfused_hbm_bytes,
      "peak_bytes": res_full_fwd.peak_hbm_bytes
  })

  def full_loss(p, x):
      return jnp.sum(full_call(p, x))
  
  try:
      grad_full_fn = jax.grad(full_loss, argnums=(0, 1))
      _, res_full_bwd = roofline.roofline(grad_full_fn)(abstract_params_full, hidden_shape)
      stats.append({
          "name": "Kimi: FULL (Bwd)", 
          "flops": res_full_bwd.flops or res_full_bwd.unfused_flops, 
          "bytes": res_full_bwd.hbm_bytes or res_full_bwd.unfused_hbm_bytes,
          "peak_bytes": res_full_bwd.peak_hbm_bytes
      })
  except Exception as e:
      print(f"Full Backward analysis failed: {e}")

  return stats

def analyze_kimi_memory(config: Config, batch_size: int, seq_len: int):
    """
    Estimates memory usage for training (activations + params) and inference (KV state).
    Returns dict with stats in bytes.
    """
    hidden_size = getattr(config, 'hidden_size', 2048)
    num_heads = getattr(config, 'num_heads', 16)
    head_dim = getattr(config, 'head_dim', 128)
    num_v_heads = getattr(config, 'num_kv_heads', num_heads) or num_heads
    
    # 1. Parameter Memory (Weights)
    # Projections: Q, K, V, O, Beta, F_A, F_B, G_A, G_B
    # DenseGeneral weights: In * Out
    
    def dense_params(in_d, out_d): return in_d * out_d
    
    q_proj = dense_params(hidden_size, num_heads * head_dim)
    k_proj = dense_params(hidden_size, num_heads * head_dim)
    v_proj = dense_params(hidden_size, num_v_heads * head_dim)
    o_proj = dense_params(num_v_heads * head_dim, hidden_size)
    
    b_proj = dense_params(hidden_size, num_heads)
    
    f_a_proj = dense_params(hidden_size, head_dim)
    f_b_proj = dense_params(head_dim, num_heads * head_dim)
    
    g_a_proj = dense_params(hidden_size, head_dim)
    g_b_proj = dense_params(head_dim, num_v_heads * head_dim) # bias=True adds small overhead, ignore for now
    
    # Conv1D weights: K * G * C/G = K * C_in. K=4.
    conv_k = 4
    q_conv = conv_k * (num_heads * head_dim)
    k_conv = conv_k * (num_heads * head_dim)
    v_conv = conv_k * (num_v_heads * head_dim)
    
    total_params = sum([q_proj, k_proj, v_proj, o_proj, b_proj, f_a_proj, f_b_proj, g_a_proj, g_b_proj, q_conv, k_conv, v_conv])
    param_bytes = total_params * 2 # bfloat16 (2 bytes)
    
    # 2. KV State (Recurrent State)
    # Shape: (B, H, K_DIM, V_DIM).
    # Assuming K_DIM = V_DIM = head_dim for Delta Rule usually, but here:
    # K matrix is (B, H, D, D). V matrix is (B, H, D, D)? 
    # State is (B, H, D, D).
    # Check code: last_recurrent_state = jnp.zeros((batch_size, num_heads, k_head_dim, v_head_dim))
    
    state_elements = batch_size * num_heads * head_dim * head_dim
    # State is usually float32 for stability in RNNs/Linear Attn
    state_bytes = state_elements * 4 
    
    # 3. Activation Memory (Training)
    # Rough estimate of major tensors stored for backward.
    # Q, K, V (after proj): 3 * B * L * H * D
    # Gates (Beta, G): ~ B * L * H * D
    # Core outputs: B * L * H * D
    # Inputs to gradients.
    
    # A standard Transformer layer is roughly 10-15x hidden size per token?
    # Let's count explicitly large tensors:
    
    # Inputs: (B, L, H_Model)
    # Projections Out: 
    #   Q: B*L*H*D
    #   K: B*L*H*D
    #   V: B*L*H*D (assuming H_v=H)
    #   Beta: B*L*H
    #   G_forget: B*L*H*D (Expanded)
    #   G_out: B*L*H*D
    
    # Conv States: B*L*H*D (x3)
    
    # Core Input tuples: (Q, K, V, G, Beta) -> ~5 * B*L*H*D
    # Core Output: B*L*H*D
    
    # Total ~ 10 * B * L * H * D * 2 bytes (bf16)
    # + stored activations for backprop (intermediate).
    # A simplified "Peak Activation" is hard without graph analysis. 
    # But "Total Tensor Size" of main intermediates is a good proxy.
    
    # 3 (QKV) + 3 (Conv) + 2 (Gates) + 1 (Core Out)
    num_large_tensors = 3 + 3 + 2 + 1
    activation_elements = num_large_tensors * (batch_size * seq_len * num_heads * head_dim)
    activation_bytes = activation_elements * 2 # bf16
    
    return {
        "param_bytes": param_bytes,
        "state_bytes": state_bytes,
        "activation_bytes": activation_bytes
    }


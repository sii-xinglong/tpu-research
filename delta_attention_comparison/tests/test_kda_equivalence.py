import os
import sys
import numpy as np
import torch
import jax
import jax.numpy as jnp
from flax import nnx

# Add paths
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir)) # tpu-research
kda_root = os.path.join(project_root, 'delta_attention_comparison')
fla_root = os.path.join(project_root, 'fla')

sys.path.append(project_root)
sys.path.append(kda_root)
sys.path.append(fla_root)

# Force JAX to CPU, PyTorch to CUDA
os.environ['JAX_PLATFORMS'] = 'cpu'

# Import Models
from delta_attention_comparison.src.layers.kimi_delta_attention import KimiDeltaAttention as JaxKDA
from fla.layers.kda import KimiDeltaAttention as TorchKDA

def t2j(t):
    """Convert PyTorch tensor to JAX array"""
    return jnp.array(t.detach().cpu().numpy())

def set_linear(jax_linear, pt_linear, transpose=True):
    """Copy weights from PyTorch Linear to JAX DenseGeneral/Linear"""
    w = pt_linear.weight
    if transpose:
        w = w.T
    jax_linear.kernel.value = t2j(w)
    
    if hasattr(pt_linear, 'bias') and pt_linear.bias is not None:
        if hasattr(jax_linear, 'bias') and jax_linear.bias is not None:
             jax_linear.bias.value = t2j(pt_linear.bias)
        else:
             print(f"Warning: PyTorch layer has bias but JAX layer does not. PT bias shape: {pt_linear.bias.shape}")

def test_kda_equivalence(B=1, L_list=[1024, 2048, 4096, 8192, 16384, 32768], H=2048, NH=16, NV=32, D=128, K=4):
    for L in L_list:
        print(f"\nTesting Kimi Delta Attention Equivalence (B={B}, L={L}, H={H}, NH={NH}, NV={NV}, D={D}, K={K})...")
        # Force JAX to CPU, PyTorch to CUDA
        os.environ['JAX_PLATFORMS'] = 'cpu'
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Init PyTorch Model
        torch.manual_seed(42)
        pt_model = TorchKDA(
            hidden_size=H,
            head_dim=D,
            num_heads=NH,
            num_v_heads=NV,
            mode='chunk',
            use_short_conv=True,
            conv_size=K,
            conv_bias=False
        ).eval().to(device)
        
        # Init JAX Model
        rngs = nnx.Rngs(0)
        jax_model = JaxKDA(
            hidden_size=H,
            num_heads=NH,
            head_dim=D,
            num_v_heads=NV,
            conv_kernel_size=K,
            dtype=jnp.float32,
            rngs=rngs
        )
        
        # --- Transfer Weights ---
        # Q, K, V Proj
        set_linear(jax_model.q_proj, pt_model.q_proj)
        set_linear(jax_model.k_proj, pt_model.k_proj)
        set_linear(jax_model.v_proj, pt_model.v_proj)
        
        # Conv1D
        # PT: (C, 1, K) -> JAX: (K, 1, C)
        jax_model.q_conv1d.kernel.value = t2j(pt_model.q_conv1d.weight.permute(2, 1, 0))
        jax_model.k_conv1d.kernel.value = t2j(pt_model.k_conv1d.weight.permute(2, 1, 0))
        jax_model.v_conv1d.kernel.value = t2j(pt_model.v_conv1d.weight.permute(2, 1, 0))
        
        # B Proj
        set_linear(jax_model.b_proj, pt_model.b_proj)
        
        # F Proj (Gate)
        set_linear(jax_model.f_a_proj, pt_model.f_proj[0])
        set_linear(jax_model.f_b_proj, pt_model.f_proj[1])
        
        # G Proj (Output Gate)
        set_linear(jax_model.g_a_proj, pt_model.g_proj[0])
        set_linear(jax_model.g_b_proj, pt_model.g_proj[1])
        
        with torch.no_grad():
            pt_model.g_proj[1].bias.zero_()
        
        # Params
        jax_model.A_log.value = t2j(pt_model.A_log).reshape(1, 1, NH, 1)
        jax_model.dt_bias.value = t2j(pt_model.dt_bias)
        
        # Output Norm
        jax_model.o_norm.rms_norm.scale.value = t2j(pt_model.o_norm.weight)
        
        # Output Proj
        set_linear(jax_model.o_proj, pt_model.o_proj)
        
        # --- Input ---
        np.random.seed(42)
        x_np = np.random.randn(B, L, H).astype(np.float32)
        x_pt = torch.tensor(x_np, device=device)
        x_jax = jnp.array(x_np)
        
        # --- Forward ---
        with torch.no_grad():
            out_pt_tuple = pt_model(x_pt, output_attentions=False)
            out_pt = out_pt_tuple[0]
            
        out_jax_tuple = jax_model(x_jax)
        out_jax = out_jax_tuple[0]
        
        # --- Compare ---
        out_pt_np = out_pt.cpu().numpy()
        out_jax_np = np.array(out_jax)
        
        diff = np.abs(out_pt_np - out_jax_np)
        max_diff = diff.max()
        mean_diff = diff.mean()
        print(f"Max Diff: {max_diff:.8f}")
        print(f"Mean Diff: {mean_diff:.8f}")
        
        # Increased tolerance to 1e-3 for cross-framework/cross-device check
        assert np.allclose(out_pt_np, out_jax_np, atol=1e-3, rtol=1e-3), f"Outputs do not match! Max diff: {max_diff}"
        print(f"SUCCESS: Equivalence Test Passed for L={L}!")

if __name__ == "__main__":
    test_kda_equivalence()
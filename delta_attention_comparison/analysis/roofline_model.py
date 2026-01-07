
import argparse
import math
import matplotlib.pyplot as plt
import numpy as np
import sys
import os
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Any
import jax.numpy as jnp

# Add parent directory to path to import src
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.layers.kimi_delta_attention import analyze_kimi_operators
from src.layers.qwen3 import analyze_qwen_operators

# --- Hardware Constants (Default: TPU v5e-like) ---
# Adjust these based on specific hardware
@dataclass
class Hardware:
    name: str
    peak_tflops: float  # TFLOPS (bf16)
    bandwidth: float    # GB/s

    def ridge_point(self):
        return (self.peak_tflops * 1e12) / (self.bandwidth * 1e9)

TPU_V5E = Hardware(name="TPU v5e", peak_tflops=197.0, bandwidth=819.0)
TPU_V6E = Hardware(name="TPU v6e (Est)", peak_tflops=275.0, bandwidth=1200.0) # Placeholder values for v6e

# --- Configuration ---
@dataclass
class ModelConfig:
    hidden_size: int
    num_heads: int
    head_dim: int
    chunk_size: int
    # For Qwen
    num_kv_heads: int = None # If None, same as num_heads
    
    # Qwen specific fields with defaults for compatibility
    emb_dim: int = field(init=False)
    gdn_num_value_heads: int = field(init=False)
    gdn_num_key_heads: int = field(init=False)
    gdn_key_head_dim: int = field(init=False)
    gdn_value_head_dim: int = field(init=False)
    gdn_conv_kernel_dim: int = 4
    gdn_chunk_size: int = field(init=False)
    use_qk_norm_in_gdn: bool = True
    dtype: Any = jnp.bfloat16
    weight_dtype: Any = jnp.bfloat16
    normalization_layer_epsilon: float = 1e-6

    def __post_init__(self):
        self.emb_dim = self.hidden_size
        self.gdn_num_value_heads = self.num_heads
        # Assume Qwen setup: K heads = V heads usually, or standard GQA. 
        # For this analysis, let's assume K=V unless specified?
        # The prompt didn't specify Qwen config details, so defaults:
        self.gdn_num_key_heads = self.num_kv_heads if self.num_kv_heads else self.num_heads
        self.gdn_key_head_dim = self.head_dim
        self.gdn_value_head_dim = self.head_dim
        self.gdn_chunk_size = self.chunk_size

    @property
    def effective_kv_heads(self):
        return self.num_kv_heads if self.num_kv_heads else self.num_heads

@dataclass
class OperatorStats:
    name: str
    flops: float
    bytes: float
    color: str

    @property
    def intensity(self):
        return self.flops / self.bytes if self.bytes > 0 else 0

    def time_sec(self, hw: Hardware = TPU_V6E):
        # Time = max(Time_compute, Time_memory)
        t_compute = self.flops / (hw.peak_tflops * 1e12)
        t_memory = self.bytes / (hw.bandwidth * 1e9)
        return max(t_compute, t_memory)

    def bound(self, hw: Hardware = TPU_V6E):
        t_compute = self.flops / (hw.peak_tflops * 1e12)
        t_memory = self.bytes / (hw.bandwidth * 1e9)
        return "Compute" if t_compute > t_memory else "Memory"

# --- Analysis Logic ---

def analyze_kimi(batch_size: int, seq_len: int, cfg: ModelConfig) -> List[OperatorStats]:
    print(f"Running JAX Roofline analysis for Kimi (B={batch_size}, L={seq_len})...")
    
    # Map colors to expected operator names
    colors = {
        "Kimi: QKV Proj": 'red',
        "Kimi: Gate Projs": 'orange',
        "Kimi: Conv1D": 'yellow',
        "Kimi: Out Proj": 'blue',
        # Breakdown
        "Kimi: Core Prep": 'lightgreen',
        # "Kimi: Core Init": 'darkred', # Removed for fine-grained
        # "Kimi: Init Decay": 'orange', # Optimized out
        "Kimi: Init A_raw (Opt)": 'yellow',
        "Kimi: Init A_raw (Opt Atomic)": 'lightgray',
        "Kimi: Init A_inv": 'darkred', # Likely hotspot
        "Kimi: Init A_inv (Atomic)": 'darkgray',
        "Kimi: Init U/W": 'pink',
        "Kimi: Core Scan": 'purple',
        "Kimi: Core Post": 'lightblue'
    }

    raw_stats = analyze_kimi_operators(cfg, batch_size, seq_len, chunk_size=cfg.chunk_size)
    
    stats = []
    for item in raw_stats:
        stats.append(OperatorStats(
            name=item['name'],
            flops=item['flops'],
            bytes=item['bytes'],
            color=colors.get(item['name'], 'gray')
        ))
    return stats

def analyze_qwen(batch_size: int, seq_len: int, cfg: ModelConfig) -> List[OperatorStats]:
    print(f"Running JAX Roofline analysis for Qwen (B={batch_size}, L={seq_len})...")

    colors = {
        "Qwen: QKVZ Proj": 'red',
        "Qwen: BA Proj": 'orange',
        "Qwen: Conv1D": 'yellow',
        "Qwen: Attn Core": 'green',
        "Qwen: Out Proj": 'blue'
    }

    raw_stats = analyze_qwen_operators(cfg, batch_size, seq_len)
    
    stats = []
    for item in raw_stats:
        stats.append(OperatorStats(
            name=item['name'],
            flops=item['flops'],
            bytes=item['bytes'],
            color=colors.get(item['name'], 'gray')
        ))
    return stats


# --- Plotting ---

def plot_roofline(stats_list: List[OperatorStats], hw: Hardware, title: str, filename: str):
    plt.figure(figsize=(10, 6))
    
    # Roofline
    # X axis: Operational Intensity (FLOPS/Byte)
    # Y axis: Performance (GFLOPS or TFLOPS)
    
    x = np.logspace(-1, 4, 100)
    # Roofline y = min(Peak, BW * x)
    # Peak in TFLOPS. BW in GB/s. x in FLOPs/Byte.
    # BW * x -> (GB/s * 1e9) * (FLOPs/Byte) = GFLOPS/s * 1e9? No.
    # BW (GB/s) * x (FLOPs/Byte) = (Bytes/s) * (FLOPs/Bytes) = FLOPs/s.
    # We want TFLOPS.
    
    y_mem = (hw.bandwidth * x) / 1000.0 # GB/s * FLOP/B / 1000 -> TFLOPS
    y_peak = np.full_like(x, hw.peak_tflops)
    y_roof = np.minimum(y_mem, y_peak)
    
    plt.plot(x, y_roof, 'k-', linewidth=2, label='Roofline (TPU)')
    plt.xscale('log')
    plt.yscale('log')
    
    # Plot points
    for stat in stats_list:
        intensity = stat.intensity
        time_s = stat.time_sec(hw)
        if time_s > 0:
            perf = stat.flops / (time_s * 1e12) # TFLOPS
        else:
            perf = 0.0
        plt.plot(intensity, perf, 'o', color=stat.color, label=stat.name, markersize=8)
        plt.text(intensity, perf, f" {stat.name}", fontsize=8, verticalalignment='bottom')

    plt.xlabel('Operational Intensity (FLOPs/Byte)')
    plt.ylabel('Performance (TFLOPS)')
    plt.title(f'Roofline Analysis: {title} (B={batch_size}, L={seq_len})')
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.legend()
    plt.savefig(filename)
    print(f"Plot saved to {filename}")

# --- Main ---

def main():
    parser = argparse.ArgumentParser(description="Roofline Analysis for Delta Attention")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument("--sequence_length", type=int, default=4096, help="Sequence length")
    parser.add_argument("--hidden_size", type=int, default=2048, help="Hidden size")
    parser.add_argument("--num_heads", type=int, default=16, help="Number of heads")
    parser.add_argument("--head_dim", type=int, default=128, help="Head dimension")
    parser.add_argument("--chunk_size", type=int, default=256, help="Chunk size")
    parser.add_argument("--hardware", type=str, default="v6e", choices=["v5e", "v6e"], help="Target Hardware")
    parser.add_argument("--sweep", action="store_true", help="Run a sweep of batch sizes and sequence lengths")
    
    args = parser.parse_args()
    
    hw = TPU_V6E if args.hardware == "v6e" else TPU_V5E
    
    cfg = ModelConfig(
        hidden_size=args.hidden_size,
        num_heads=args.num_heads,
        head_dim=args.head_dim,
        chunk_size=args.chunk_size
    )

    if args.sweep:
        configs = [
            (1, 4096),
            (1, 8192),
            (1, 16384),
            (1, 32768),
            (1, 65536),
            (1, 131072),
            (2, 32768),
            (4, 32768),
            (8, 32768),
            (16, 32768),
        ]
    else:
        configs = [(args.batch_size, args.sequence_length)]

    for b, l in configs:
        global batch_size, seq_len 
        batch_size = b
        seq_len = l
        
        print(f"\nAnalyzing for {hw.name} | Batch: {b}, Seq: {l}")
        print("-" * 60)
        
        # Analyze Kimi
        print(">>> KimiDeltaAttention Analysis")
        kimi_stats = analyze_kimi(b, l, cfg)
        total_kimi_time = 0
        print(f"{'Operator':<20} | {'FLOPs':<10} | {'Bytes':<10} | {'Intensity':<10} | {'Time (us)':<10} | {'Bound'}")
        for s in kimi_stats:
            t_sec = s.time_sec(hw)
            total_kimi_time += t_sec
            print(f"{s.name:<20} | {s.flops:.2e}   | {s.bytes:.2e}   | {s.intensity:.2f}       | {t_sec*1e6:.2f}     | {s.bound(hw)}")
        print(f"Total Theoretical Time: {total_kimi_time*1e6:.2f} us")
        print("-" * 60)

        # Analyze Qwen
        print(">>> Qwen3NextGatedDeltaNet Analysis")
        qwen_stats = analyze_qwen(b, l, cfg)
        total_qwen_time = 0
        print(f"{'Operator':<20} | {'FLOPs':<10} | {'Bytes':<10} | {'Intensity':<10} | {'Time (us)':<10} | {'Bound'}")
        for s in qwen_stats:
            t_sec = s.time_sec(hw)
            total_qwen_time += t_sec
            print(f"{s.name:<20} | {s.flops:.2e}   | {s.bytes:.2e}   | {s.intensity:.2f}       | {t_sec*1e6:.2f}     | {s.bound(hw)}")
        print(f"Total Theoretical Time: {total_qwen_time*1e6:.2f} us")
        print("-" * 60)
        
        # Plot
        filename = f"roofline_comparison_B{b}_L{l}.png"
        plot_roofline(kimi_stats + qwen_stats, hw, f"Kimi vs Qwen (B={b}, L={l})", filename)

if __name__ == "__main__":
    main()

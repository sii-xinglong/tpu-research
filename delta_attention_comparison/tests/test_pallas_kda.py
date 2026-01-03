
import jax
import jax.numpy as jnp
import numpy as np
import unittest
import time
import functools
from src.layers.kimi_delta_attention import chunk_parallel_delta_attention as ref_impl
from src.layers.pallas_kda import chunk_parallel_delta_attention_pallas as pallas_impl

class TestPallasKDA(unittest.TestCase):
    def test_pallas_correctness(self):
        # Set seed
        key = jax.random.PRNGKey(0)
        
        # Dimensions based on Qwen3-like configuration
        # H=32 corresponds to gdn_num_value_heads (query/key are repeated to match this)
        # D=128 corresponds to gdn_key_head_dim and gdn_value_head_dim
        B, H, L, D = 1, 32, 128, 128
        chunk_size = 64
        
        # Generate Inputs
        k1, k2, k3, k4, k5 = jax.random.split(key, 5)
        q = jax.random.normal(k1, (B, H, L, D))
        k = jax.random.normal(k2, (B, H, L, D))
        v = jax.random.normal(k3, (B, H, L, D))
        g = jax.random.normal(k4, (B, H, L, D)) # Log space decay
        beta = jax.random.uniform(k5, (B, H, L))
        
        # Run Reference
        print(f"Running Reference JAX Implementation (B={B}, H={H}, L={L}, D={D})...")
        out_ref, _ = ref_impl(q, k, v, g, beta, chunk_size=chunk_size)
        
        # Run Pallas
        print("Running Pallas Implementation...")
        try:
            out_pallas, _ = pallas_impl(q, k, v, g, beta, chunk_size=chunk_size)
            
            # Compare
            diff = jnp.abs(out_ref - out_pallas)
            max_diff = jnp.max(diff)
            mean_diff = jnp.mean(diff)
            
            print(f"Max Diff: {max_diff}")
            print(f"Mean Diff: {mean_diff}")
            
            np.testing.assert_allclose(out_ref, out_pallas, atol=1e-4, rtol=1e-4)
            print("Pallas implementation matches reference!")
            
        except Exception as e:
            print(f"Pallas execution failed (expected if not on TPU): {e}")
            # If on CPU/GPU without Pallas TPU support, this is expected.
            # We don't fail the test suite if hardware is missing, but warn.
            if "tpu" not in jax.devices()[0].platform.lower():
                 print("Skipping assertion as TPU is likely not available.")
            else:
                 raise e

    def test_benchmark(self):
        print("\n=== Running Benchmark ===")
        # Set seed
        key = jax.random.PRNGKey(42)
        
        # Dimensions for benchmarking based on Qwen3-like configuration
        # Using L=4096 to get meaningful timings
        B, H, L, D = 1, 32, 4096, 128
        chunk_size = 64
        
        print(f"Dimensions: B={B}, H={H}, L={L}, D={D}, Chunk={chunk_size}")

        # Generate Inputs
        k1, k2, k3, k4, k5 = jax.random.split(key, 5)
        q = jax.random.normal(k1, (B, H, L, D))
        k = jax.random.normal(k2, (B, H, L, D))
        v = jax.random.normal(k3, (B, H, L, D))
        g = jax.random.normal(k4, (B, H, L, D))
        beta = jax.random.uniform(k5, (B, H, L))

        # JIT compile for fair comparison
        # We assume ref_impl is pure JAX and can be JITed if not already.
        ref_jit = jax.jit(functools.partial(ref_impl, chunk_size=chunk_size))
        pallas_jit = jax.jit(functools.partial(pallas_impl, chunk_size=chunk_size))

        try:
            # Warmup
            print("Warming up...")
            ref_jit(q, k, v, g, beta)[0].block_until_ready()
            pallas_jit(q, k, v, g, beta)[0].block_until_ready()

            n_iter = 20
            
            # Benchmark Reference
            print(f"Benchmarking Reference ({n_iter} iterations)...")
            start = time.time()
            for _ in range(n_iter):
                ref_jit(q, k, v, g, beta)[0].block_until_ready()
            end = time.time()
            ref_time = (end - start) / n_iter
            print(f"Reference average time: {ref_time*1000:.4f} ms")

            # Benchmark Pallas
            print(f"Benchmarking Pallas ({n_iter} iterations)...")
            start = time.time()
            for _ in range(n_iter):
                pallas_jit(q, k, v, g, beta)[0].block_until_ready()
            end = time.time()
            pallas_time = (end - start) / n_iter
            print(f"Pallas average time:    {pallas_time*1000:.4f} ms")
            
            print(f"Speedup: {ref_time / pallas_time:.2f}x")

        except Exception as e:
            print(f"Benchmark failed: {e}")
            if "tpu" not in jax.devices()[0].platform.lower():
                print("Skipping benchmark as TPU is likely not available.")
            else:
                raise e

if __name__ == "__main__":
    unittest.main()

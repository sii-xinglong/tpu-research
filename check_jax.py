import jax
try:
    from jax import roofline
    print("jax.roofline found")
except ImportError:
    print("jax.roofline NOT found")

print(f"JAX Version: {jax.__version__}")
try:
    import jax.experimental.pallas as pl
    print("Pallas found")
except ImportError:
    print("Pallas NOT found")

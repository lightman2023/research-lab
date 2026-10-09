"""Shared, opt-out JAX compilation cache on the project's local filesystem."""
import os
from pathlib import Path
import jax

def configure_runtime():
    if os.environ.get('AZCHESS_DISABLE_COMPILATION_CACHE')=='1':return
    location=os.environ.get('JAX_COMPILATION_CACHE_DIR') or str(Path(__file__).resolve().parents[1]/'.jax_cache')
    jax.config.update('jax_compilation_cache_dir',location)
    jax.config.update('jax_persistent_cache_min_compile_time_secs',0.1)

configure_runtime()

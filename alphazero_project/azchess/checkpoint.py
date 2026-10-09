from pathlib import Path
import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

from flax import serialization
import jax
import jax.numpy as jnp

from .game import BOARD_PLANES
from .model import PolicyValueNet


def create_model(seed: int = 0):
    model = PolicyValueNet()
    # Current WSL/CUDA driver crashes while compiling PRNG seeding on GPU.
    # Parameter initialization is tiny, so do it on CPU; inference/training still use CUDA.
    cpu = jax.devices("cpu")[0]
    with jax.default_device(cpu):
        variables = model.init(
            jax.random.PRNGKey(seed),
            jnp.zeros((1, 8, 8, BOARD_PLANES), dtype=jnp.float32),
            train=False,
        )
    return model, variables


def save_variables(path: Path, variables) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(serialization.to_bytes(variables))
    os.replace(temporary, path)


def load_variables(path: Path):
    model, template = create_model()
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    return model, serialization.from_bytes(template, path.read_bytes())

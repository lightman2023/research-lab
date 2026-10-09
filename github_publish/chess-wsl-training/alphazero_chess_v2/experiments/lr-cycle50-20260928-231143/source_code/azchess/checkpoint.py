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
    cpu = jax.devices("cpu")[0]
    with jax.default_device(cpu):
        variables = model.init(
            jax.random.PRNGKey(seed),
            jnp.zeros((1, 8, 8, BOARD_PLANES), dtype=jnp.float32),
            train=False,
        )
    return model, variables


def save_bytes_atomic(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def save_variables(path: Path, variables) -> None:
    save_bytes_atomic(path, serialization.to_bytes(variables))


def load_variables(path: Path):
    model, template = create_model()
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    return model, serialization.from_bytes(template, path.read_bytes())

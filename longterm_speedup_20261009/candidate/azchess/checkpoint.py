from pathlib import Path
import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

from flax import serialization
import jax
import jax.numpy as jnp

from .game import BOARD_PLANES
from .model import PolicyValueNet
from . import runtime
from functools import lru_cache, partial
import numpy as np

@lru_cache(maxsize=1)
def variable_shapes():
    model=PolicyValueNet()
    return jax.eval_shape(partial(model.init,train=False),
        jax.ShapeDtypeStruct((2,),jnp.uint32),jax.ShapeDtypeStruct((1,8,8,BOARD_PLANES),jnp.float32))


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
    model = PolicyValueNet()
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    variables=serialization.msgpack_restore(path.read_bytes())
    expected,structure=jax.tree_util.tree_flatten(variable_shapes())
    actual,actual_structure=jax.tree_util.tree_flatten(variables)
    if structure!=actual_structure:raise ValueError(f'Checkpoint structure mismatch: {path}')
    for shape,array in zip(expected,actual):
        if shape.shape!=array.shape or np.dtype(shape.dtype)!=array.dtype:
            raise ValueError(f'Checkpoint shape/dtype mismatch: {path}')
    return model, variables

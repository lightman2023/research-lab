from pathlib import Path

import numpy as np
import optax
from flax import serialization

from azchess.checkpoint import create_model, save_bytes_atomic, save_variables
from train import create_optimizer


def main() -> None:
    checkpoints = Path("checkpoints")
    latest = checkpoints / "latest.msgpack"
    training_state = checkpoints / "training_state.msgpack"
    model, variables = create_model(seed=0)
    optimizer = create_optimizer()
    bundle = {
        "variables": variables,
        "opt_state": optimizer.init(variables["params"]),
        "step": np.asarray(0, dtype=np.int64),
    }
    save_variables(latest, variables)
    save_bytes_atomic(training_state, serialization.to_bytes(bundle))
    save_variables(checkpoints / "generation-000-untrained.msgpack", variables)
    save_variables(checkpoints / "saved" / "cycle-000000.msgpack", variables)
    print(f"Initialized scaled AlphaZero v2: {latest}")


if __name__ == "__main__":
    main()

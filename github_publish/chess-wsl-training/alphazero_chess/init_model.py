from pathlib import Path

from azchess.checkpoint import create_model, save_variables

checkpoint = Path(__file__).resolve().parent / "checkpoints" / "latest.msgpack"
_, variables = create_model(seed=0)
save_variables(checkpoint, variables)
print(f"Initialized AlphaZero-style model: {checkpoint}")


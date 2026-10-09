from flax import linen as nn
import jax.numpy as jnp

from .game import ACTION_PLANES


class ResidualBlock(nn.Module):
    channels: int

    @nn.compact
    def __call__(self, x: jnp.ndarray, train: bool) -> jnp.ndarray:
        residual = x
        x = nn.Conv(self.channels, (3, 3), padding="SAME", use_bias=False)(x)
        x = nn.BatchNorm(use_running_average=not train)(x)
        x = nn.relu(x)
        x = nn.Conv(self.channels, (3, 3), padding="SAME", use_bias=False)(x)
        x = nn.BatchNorm(use_running_average=not train)(x)
        return nn.relu(x + residual)


class PolicyValueNet(nn.Module):
    """Consumer-GPU-sized policy/value ResNet with a spatial policy head."""

    channels: int = 96
    blocks: int = 6

    @nn.compact
    def __call__(self, x: jnp.ndarray, train: bool = False):
        x = nn.Conv(self.channels, (3, 3), padding="SAME", use_bias=False)(x)
        x = nn.BatchNorm(use_running_average=not train)(x)
        x = nn.relu(x)
        for _ in range(self.blocks):
            x = ResidualBlock(self.channels)(x, train)

        policy = nn.Conv(32, (1, 1), use_bias=False)(x)
        policy = nn.BatchNorm(use_running_average=not train)(policy)
        policy = nn.relu(policy)
        policy = nn.Conv(ACTION_PLANES, (1, 1), use_bias=True, name="policy_logits")(policy)
        logits = policy.reshape((policy.shape[0], -1))

        value = nn.Conv(1, (1, 1), use_bias=False)(x)
        value = nn.BatchNorm(use_running_average=not train)(value)
        value = nn.relu(value).reshape((value.shape[0], -1))
        value = nn.relu(nn.Dense(256)(value))
        value = nn.Dense(1)(value)
        return logits, nn.tanh(value).squeeze(-1)

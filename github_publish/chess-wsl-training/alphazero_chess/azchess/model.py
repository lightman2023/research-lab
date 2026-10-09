from flax import linen as nn
import jax.numpy as jnp

from .game import ACTION_SIZE


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
    channels: int = 64
    blocks: int = 4

    @nn.compact
    def __call__(self, x: jnp.ndarray, train: bool = False):
        x = nn.Conv(self.channels, (3, 3), padding="SAME", use_bias=False)(x)
        x = nn.BatchNorm(use_running_average=not train)(x)
        x = nn.relu(x)
        for _ in range(self.blocks):
            x = ResidualBlock(self.channels)(x, train)

        policy = nn.Conv(32, (1, 1), use_bias=False)(x)
        policy = nn.BatchNorm(use_running_average=not train)(policy)
        policy = nn.relu(policy).reshape((policy.shape[0], -1))
        logits = nn.Dense(ACTION_SIZE, name="policy_logits")(policy)

        value = nn.Conv(8, (1, 1), use_bias=False)(x)
        value = nn.BatchNorm(use_running_average=not train)(value)
        value = nn.relu(value).reshape((value.shape[0], -1))
        value = nn.relu(nn.Dense(64)(value))
        value = nn.Dense(1)(value)
        return logits, nn.tanh(value).squeeze(-1)


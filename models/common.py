from flax.typing import Array
from jax._src import core
import flax.linen as nn
import jax
import jax.numpy as jnp


def trunc_normal_init_(
    std: float = 1, lower: float = -2.0, upper: float = 2.0
) -> nn.initializers.Initializer:
    def init(
        key: Array,
        shape: core.Shape,
        dtype=jnp.float32,
        out_sharding=None,
    ) -> Array:
        sqrt2 = jnp.sqrt(2)
        a = jax.lax.erf(lower / sqrt2)
        b = jax.lax.erf(upper / sqrt2)
        z = (b - a) / 2

        c = (2 * jnp.pi) ** -0.5
        pdf_u = c * jnp.exp(-0.5 * lower**2)
        pdf_l = c * jnp.exp(-0.5 * upper**2)
        comp_std = std / jnp.sqrt(
            1 - (upper * pdf_u - lower * pdf_l) / z - ((pdf_u - pdf_l) / z) ** 2
        )

        tensor = jax.random.uniform(key, shape, dtype, a, b)
        tensor = jax.lax.erf_inv(tensor)
        tensor *= sqrt2 * comp_std
        tensor = jnp.clip(tensor, lower * comp_std, upper * comp_std)
        return tensor

    return init

from flax.typing import Array
import flax.linen as nn
import numpy as np
import jax
import jax.numpy as jnp
import einops as eo


def sin_embed(x):
    t = jnp.exp(
        1j
        * jnp.arange(x.shape[-2])[..., None]
        * jnp.pow(10000, -jnp.linspace(0, 1, x.shape[-1] // 2))
    )
    t = jnp.concat([jnp.real(t)[..., None], jnp.imag(t)[..., None]], axis=-1)
    return eo.rearrange(t, "... d i -> ... (d i)")


class SWIGlu(nn.Module):
    dim: int

    @nn.compact
    def __call__(self, x: Array) -> Array:
        x_dim = x.shape[-1]
        x1 = nn.Dense(self.dim)(x)
        x2 = nn.Dense(self.dim)(x)
        return nn.Dense(x_dim)(nn.silu(x1) * x2)


class TransformerBlock(nn.Module):
    n_heads: int

    @nn.compact
    def __call__(self, x: Array) -> Array:
        skip = x
        x = nn.attention.MultiHeadAttention(self.n_heads)(x, x, x)
        # x = GroupedQuaryAttention(x.shape[-1], self.n_heads, self.n_heads)(x)
        x += skip
        x = nn.RMSNorm()(x)

        skip = x
        x = SWIGlu(x.shape[-1] * 8 // 3)(x)
        x += skip
        return nn.RMSNorm()(x)


class Transformer(nn.Module):
    n_heads: int
    n_blocks: int

    @nn.compact
    def __call__(self, inp: list[Array], zs: Array) -> Array:
        x = sum(inp) + zs
        for _ in range(self.n_blocks):
            x = TransformerBlock(self.n_heads)(x)
        return x


class HRMBlock(nn.Module):
    n_heads: int
    n_blocks: int
    h_freqs: list[int]

    def setup(self):
        self.cum_h_freqs = np.cumprod(self.h_freqs)
        self.levels = [
            Transformer(self.n_heads, self.n_blocks)
            for _ in range(len(self.cum_h_freqs))
        ]

    def __call__(self, x: Array, z: list[Array]) -> list[Array]:
        for i in range(1, self.cum_h_freqs[-1]):
            z[0] = self.levels[0](z[1:] + [x], z[0])
            for idx, n in enumerate(self.cum_h_freqs[:-1], start=1):
                if i % n == 0:
                    z[idx] = self.levels[idx](z[:idx] + z[idx + 1 :], z[idx])

        for i in range(len(z)):
            z[i] = jax.lax.stop_gradient(z[i])

        z[0] = self.levels[0](z[1:] + [x], z[0])
        for i in range(1, len(z)):
            z[i] = self.levels[i](z[:i] + z[i + 1 :], z[i])

        return z


class HRM(nn.Module):
    dim: int
    num_emb: int
    n_heads: int
    n_blocks: int
    h_freqs: list[int]

    @nn.compact
    def __call__(self, x: Array, z: list[Array]) -> tuple[Array, list[Array], Array]:
        embed = nn.Embed(self.num_emb, self.dim)
        x = embed(x)
        # x += sin_embed(x)
        x += self.param("time_embed", lambda *_: sin_embed(x), sin_embed(x).shape)

        x = jnp.concat(
            [
                x,
                eo.repeat(
                    self.param(
                        "q_token", nn.initializers.truncated_normal(1), (1, x.shape[-1])
                    ),
                    "... -> b ...",
                    b=x.shape[0],
                ),
            ],
            axis=-2,
        )

        z = HRMBlock(self.n_heads, self.n_blocks, self.h_freqs)(x, z)

        zs = z[-1]

        # out = embed.attend(zs)
        out = nn.Dense(self.num_emb)(zs[..., :-1, :])

        q = nn.Dense(
            2,
            kernel_init=nn.initializers.zeros_init(),
            bias_init=nn.initializers.constant(-5),
        )(zs[..., -1, :])

        return out, z, q

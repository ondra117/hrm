import flax.linen as nn
from flax.typing import Array
import jax.numpy as jnp
from models.common import trunc_normal_init_
import einops as eo
import jax
import math


def _find_multiple(a, b):
    return (-(a // -b)) * b


def rotate_half(x: Array):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return jnp.concat([-x2, x1], axis=-1)


def apply_rotary_pos_emb(
    q: Array, k: Array, cos: Array, sin: Array
) -> tuple[Array, Array]:
    # q, k: [bs, seq_len, num_heads, head_dim]
    # cos, sin: [seq_len, head_dim]
    orig_dtype = q.dtype
    q = q.astype(cos.dtype)
    k = k.astype(cos.dtype)

    q_embed = (q * cos[..., None, :]) + (rotate_half(q) * sin[..., None, :])
    k_embed = (k * cos[..., None, :]) + (rotate_half(k) * sin[..., None, :])

    return q_embed.astype(orig_dtype), k_embed.astype(orig_dtype)


class CastedLinear(nn.Module):
    in_features: int
    out_features: int
    use_bias: bool
    weights_init: nn.initializers.Initializer | None = None
    bias_init: nn.initializers.Initializer | None = None

    def setup(self):
        self.weights = self.param(
            "w",
            trunc_normal_init_(std=1 / (self.in_features**0.5))
            if self.weights_init is None
            else self.weights_init,
            (self.out_features, self.in_features),
        )
        self.bias = 0
        if self.use_bias:
            self.bias = self.param(
                "b",
                nn.initializers.zeros_init()
                if self.bias_init is None
                else self.bias_init,
                (self.out_features,),
            )

    def __call__(self, input: Array) -> Array:
        return (
            jnp.einsum("o d, ... d -> ... o", self.weights.astype(input.dtype), input)
            + self.bias
        )


class CastedEmbedding(nn.Module):
    num_embeddings: int
    embedding_dim: int
    init_std: float
    cast_to: jnp.dtype = jnp.float32

    def setup(self):
        self.weights = self.param(
            "w",
            trunc_normal_init_(std=self.init_std),
            (self.num_embeddings, self.embedding_dim),
        )

    def embedding_weight(self):
        return self.weights

    def __call__(self, input: Array) -> Array:
        return self.weights.astype(self.cast_to)[input]


class RotaryEmbedding(nn.Module):
    dim: int
    max_position_embeddings: int
    base: int

    def setup(self):
        inv_freq = 1 / (
            self.base ** (jnp.arange(0, self.dim, 2, dtype=jnp.float32) / self.dim)
        )
        t = jnp.arange(self.max_position_embeddings, dtype=jnp.float32)
        freqs = jnp.outer(t, inv_freq)

        emb = jnp.concat([freqs, freqs], axis=-1)
        self.cos_cached = jnp.cos(emb)
        self.sin_cached = jnp.sin(emb)

    def __call__(self) -> tuple[Array, Array]:
        return self.cos_cached, self.sin_cached


class Attention(nn.Module):
    hidden_size: int
    head_dim: int
    num_heads: int
    num_key_value_heads: int

    def setup(self):
        self.output_size = self.head_dim * self.num_heads

        self.qkv_proj = CastedLinear(
            self.hidden_size,
            (self.num_heads + 2 * self.num_key_value_heads) * self.head_dim,
            use_bias=False,
        )
        self.o_proj = CastedLinear(self.output_size, self.hidden_size, use_bias=False)

    def __call__(
        self, cos_sin: tuple[Array, Array] | None, hidden_states: Array
    ) -> Array:
        batch_size, seq_len, _ = hidden_states.shape

        qkv = self.qkv_proj(hidden_states)

        qkv = eo.rearrange(qkv, "... (h d) -> ... h d", d=self.head_dim)
        query = qkv[..., : self.num_heads, :]
        key = qkv[..., self.num_heads : self.num_heads + self.num_key_value_heads, :]
        value = qkv[..., self.num_heads + self.num_key_value_heads :, :]

        if cos_sin is not None:
            cos, sin = cos_sin
            query, key = apply_rotary_pos_emb(query, key, cos, sin)

        attn_output = nn.attention.dot_product_attention(query, key, value)

        attn_output = eo.rearrange(attn_output, "... h d -> ... (h d)")
        return self.o_proj(attn_output)


class SwiGLU(nn.Module):
    hidden_size: int
    expansion: float

    def setup(self):
        # inter = _find_multiple(round(self.expansion * self.hidden_size * 2 / 3), 256)
        inter = math.ceil(self.expansion * 2 / 3) * self.hidden_size

        self.gate_up_proj = CastedLinear(self.hidden_size, inter * 2, use_bias=False)
        self.down_proj = CastedLinear(inter, self.hidden_size, use_bias=False)

    def __call__(self, x):
        gate, up = jnp.split(self.gate_up_proj(x), 2, axis=-1)
        return self.down_proj(nn.silu(gate) * up)


def rms_norm(hidden_states: Array, variance_epsilon: float) -> Array:
    input_dtype = hidden_states.dtype
    hidden_states = hidden_states.astype(jnp.float32)

    variance = jnp.mean(hidden_states**2, axis=-1, keepdims=True)
    hidden_states *= jax.lax.rsqrt(variance + variance_epsilon)
    return hidden_states.astype(input_dtype)

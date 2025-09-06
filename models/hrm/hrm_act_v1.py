import flax.linen as nn
from flax.typing import Array
import jax.numpy as jnp
from flax.struct import dataclass
from pydantic import BaseModel
import jax

from models.layers import (
    Attention,
    SwiGLU,
    rms_norm,
    CastedLinear,
    CastedEmbedding,
    RotaryEmbedding,
)
from models.common import trunc_normal_init_


@dataclass
class HierarchicalReasoningModel_ACTV1InnerCarry:
    z_H: Array
    z_L: Array


@dataclass
class HierarchicalReasoningModel_ACTV1Carry:
    inner_carry: HierarchicalReasoningModel_ACTV1InnerCarry

    steps: Array
    halted: Array

    current_data: Array


class HierarchicalReasoningModel_ACTV1Config(BaseModel):
    # batch_size: int
    seq_len: int
    puzzle_emb_ndim: int = 0
    num_puzzle_identifiers: int
    vocab_size: int

    H_cycles: int
    L_cycles: int

    H_layers: int
    L_layers: int

    # Transformer config
    hidden_size: int
    expansion: float
    num_heads: int
    pos_encodings: str

    rms_norm_eps: float = 1e-5
    rope_theta: float = 10000.0

    # Halting Q-learning config
    halt_max_steps: int
    halt_exploration_prob: float

    forward_dtype: str = "bfloat16"


class HierarchicalReasoningModel_ACTV1Block(nn.Module):
    config: HierarchicalReasoningModel_ACTV1Config

    def setup(self):
        self.self_attn = Attention(
            hidden_size=self.config.hidden_size,
            head_dim=self.config.hidden_size // self.config.num_heads,
            num_heads=self.config.num_heads,
            num_key_value_heads=self.config.num_heads,
        )
        self.mlp = SwiGLU(
            hidden_size=self.config.hidden_size,
            expansion=self.config.expansion,
        )

    def __call__(self, cos_sin: tuple[Array, Array], hidden_states: Array) -> Array:
        hidden_states = rms_norm(
            hidden_states
            + self.self_attn(cos_sin=cos_sin, hidden_states=hidden_states),
            variance_epsilon=self.config.rms_norm_eps,
        )

        hidden_states = rms_norm(
            hidden_states + self.mlp(hidden_states),
            variance_epsilon=self.config.rms_norm_eps,
        )
        return hidden_states


class HierarchicalReasoningModel_ACTV1ReasoningModule(nn.Module):
    layers: list[HierarchicalReasoningModel_ACTV1Block]

    @nn.compact
    def __call__(self, hidden_states: Array, input_injection: Array, **kwargs) -> Array:
        hidden_states = hidden_states + input_injection

        for layer in self.layers:
            hidden_states = layer(hidden_states=hidden_states, **kwargs)

        return hidden_states


class HierarchicalReasoningModel_ACTV1_Inner(nn.Module):
    config: HierarchicalReasoningModel_ACTV1Config

    def setup(self):
        self.forward_dtype = jnp.float32

        self.embed_scale = jnp.sqrt(self.config.hidden_size)
        embed_init_std = 1.0 / self.embed_scale

        self.embed_tokens = CastedEmbedding(
            self.config.vocab_size,
            self.config.hidden_size,
            init_std=embed_init_std,
            cast_to=self.forward_dtype,
        )
        self.lm_head = CastedLinear(
            self.config.hidden_size, self.config.vocab_size, use_bias=False
        )
        self.q_head = CastedLinear(
            self.config.hidden_size,
            2,
            use_bias=True,
            weights_init=nn.initializers.zeros_init(),
            bias_init=nn.initializers.constant(-5),
        )

        self.puzzle_emb_len = -(self.config.puzzle_emb_ndim // -self.config.hidden_size)
        if self.config.puzzle_emb_ndim > 0:
            self.puzzle_emb = CastedEmbedding(  # TODO: sparse
                self.config.num_puzzle_identifiers,
                self.config.puzzle_emb_ndim,
                init_std=0,
                cast_to=self.forward_dtype,
            )

        if self.config.pos_encodings == "rope":
            self.rotary_emb = RotaryEmbedding(
                dim=self.config.hidden_size // self.config.num_heads,
                max_position_embeddings=self.config.seq_len + self.puzzle_emb_len,
                base=self.config.rope_theta,
            )
        elif self.config.pos_encodings == "learned":
            self.embed_pos = CastedEmbedding(
                self.config.seq_len + self.puzzle_emb_len,
                self.config.hidden_size,
                init_std=embed_init_std,
                cast_to=self.forward_dtype,
            )
        else:
            raise NotImplementedError()

        self.H_level = HierarchicalReasoningModel_ACTV1ReasoningModule(
            layers=[
                HierarchicalReasoningModel_ACTV1Block(self.config)
                for _ in range(self.config.H_layers)
            ]
        )
        self.L_level = HierarchicalReasoningModel_ACTV1ReasoningModule(
            layers=[
                HierarchicalReasoningModel_ACTV1Block(self.config)
                for _ in range(self.config.L_layers)
            ]
        )

        self.H_init = self.variable(
            "constants",
            "h_init",
            lambda s: trunc_normal_init_()(
                jax.random.PRNGKey(0),
                s,
                dtype=self.forward_dtype,
            ),
            (self.config.hidden_size,),
        )
        self.L_init = self.variable(
            "constants",
            "l_init",
            lambda s: trunc_normal_init_()(
                jax.random.PRNGKey(1),
                s,
                dtype=self.forward_dtype,
            ),
            (self.config.hidden_size,),
        )

    def _input_embeddings(self, input: Array, puzzle_identifiers: Array) -> Array:
        embedding = self.embed_tokens(input.astype(jnp.int32))

        if self.config.puzzle_emb_ndim > 0:
            puzzle_embedding = self.puzzle_emb(puzzle_identifiers)

            pad_count = (
                self.puzzle_emb_len * self.config.hidden_size
                - puzzle_embedding.shape[-1]
            )
            if pad_count > 0:
                puzzle_embedding = jnp.pad(puzzle_embedding, (0, pad_count))

            embedding = jnp.concat(
                [
                    puzzle_embedding.reshape(
                        (-1, self.puzzle_emb_len, self.config.hidden_size)  # repeat?
                    ),
                    embedding,
                ],
                axis=-2,
            )

        if self.config.pos_encodings == "learned":
            embedding = 0.707106781 * (
                embedding + self.embed_pos.embedding_weight().astype(self.forward_dtype)
            )

        return self.embed_scale * embedding

    def empty_carry(
        self, batch_size: int
    ) -> HierarchicalReasoningModel_ACTV1InnerCarry:
        return HierarchicalReasoningModel_ACTV1InnerCarry(
            z_H=jnp.empty(
                (
                    batch_size,
                    self.config.seq_len + self.puzzle_emb_len,
                    self.config.hidden_size,
                ),
                dtype=self.forward_dtype,
            ),
            z_L=jnp.empty(
                (
                    batch_size,
                    self.config.seq_len + self.puzzle_emb_len,
                    self.config.hidden_size,
                ),
                dtype=self.forward_dtype,
            ),
        )

    def reset_carry(
        self, reset_flag: Array, carry: HierarchicalReasoningModel_ACTV1InnerCarry
    ) -> HierarchicalReasoningModel_ACTV1InnerCarry:
        return HierarchicalReasoningModel_ACTV1InnerCarry(
            z_H=jnp.where(reset_flag[..., None, None], self.H_init.value, carry.z_H),
            z_L=jnp.where(reset_flag[..., None, None], self.L_init.value, carry.z_L),
        )

    def get_puzzle_emb(self) -> Array:
        return self.puzzle_emb

    def __call__(
        self, carry: HierarchicalReasoningModel_ACTV1InnerCarry, batch: dict[str, Array]
    ) -> tuple[HierarchicalReasoningModel_ACTV1InnerCarry, Array, tuple[Array, Array]]:
        cos_sin = self.rotary_emb() if self.config.pos_encodings == "rope" else None

        input_embeddings = self._input_embeddings(
            batch["inputs"], batch["puzzle_identifiers"]
        )

        z_H, z_L = carry.z_H, carry.z_L
        for _H_step in range(self.config.H_cycles):
            for _L_step in range(self.config.L_cycles):
                if not (
                    (_H_step == self.config.H_cycles - 1)
                    and (_L_step == self.config.L_cycles - 1)
                ):
                    z_L = self.L_level(z_L, z_H + input_embeddings, cos_sin=cos_sin)

            if not (_H_step == self.config.H_cycles - 1):
                z_H = self.H_level(z_H, z_L, cos_sin=cos_sin)

        z_H = jax.lax.stop_gradient(z_H)
        z_L = jax.lax.stop_gradient(z_L)

        z_L = self.L_level(z_L, z_H + input_embeddings, cos_sin=cos_sin)
        z_H = self.H_level(z_H, z_L, cos_sin=cos_sin)

        new_carry = HierarchicalReasoningModel_ACTV1InnerCarry(
            z_H=jax.lax.stop_gradient(z_H), z_L=jax.lax.stop_gradient(z_L)
        )

        output = self.lm_head(z_H)[:, self.puzzle_emb_len :]

        q_logits = self.q_head(z_H[:, 0]).astype(jnp.float32)

        return new_carry, output, (q_logits[..., 0], q_logits[..., 1])


class HierarchicalReasoningModel_ACTV1(nn.Module):
    config: HierarchicalReasoningModel_ACTV1Config

    def setup(self):
        self.inner = HierarchicalReasoningModel_ACTV1_Inner(self.config)

    @property
    def puzzle_emb(self) -> Array:
        return self.inner.get_puzzle_emb()

    def initial_carry(
        self, batch: dict[str, Array]
    ) -> HierarchicalReasoningModel_ACTV1Carry:
        batch_size = batch["inputs"].shape[0]

        return HierarchicalReasoningModel_ACTV1Carry(
            inner_carry=self.inner.empty_carry(batch_size),
            steps=jnp.zeros((batch_size,), dtype=jnp.int32),
            halted=jnp.ones((batch_size,), dtype=jnp.bool_),
            current_data=jax.tree.map(jnp.zeros_like, batch),
        )

    def __call__(
        self,
        carry: HierarchicalReasoningModel_ACTV1Carry,
        batch: dict[str, Array],
        key: Array,
    ) -> dict[HierarchicalReasoningModel_ACTV1Carry, dict[str, Array]]:
        new_inner_carry = self.inner.reset_carry(carry.halted, carry.inner_carry)

        new_steps = jnp.where(carry.halted, 0, carry.steps)

        new_current_data = jax.tree.map(
            lambda x, y: jnp.where(
                carry.halted.reshape((-1,) + (1,) * (y.ndim - 1)), y, x
            ),
            carry.current_data,
            batch,
        )

        new_inner_carry, logits, (q_halt_logits, q_continue_logits) = self.inner(
            new_inner_carry, new_current_data
        )

        outputs = {
            "logits": logits,
            "q_halt_logits": q_halt_logits,
            "q_continue_logits": q_continue_logits,
        }

        new_steps = new_steps + 1
        is_last_step = new_steps >= self.config.halt_max_steps

        halted = is_last_step

        if self.config.halt_max_steps > 1:
            halted = halted | (q_halt_logits > q_continue_logits)

            key, key1, key2 = jax.random.split(key, 3)
            min_halt_steps = (
                jax.random.uniform(key1, q_halt_logits.shape)
                < self.config.halt_exploration_prob
            ) * (
                jax.random.randint(
                    key2,
                    new_steps.shape,
                    minval=2,
                    maxval=self.config.halt_max_steps + 1,
                )
            )

            halted = halted & (new_steps >= min_halt_steps)

            next_q_halt_logits, next_q_continue_logits = self.inner(
                new_inner_carry, new_current_data
            )[-1]

            outputs["target_q_continue"] = jax.lax.stop_gradient(
                nn.sigmoid(
                    jnp.where(
                        is_last_step,
                        next_q_halt_logits,
                        jnp.maximum(next_q_halt_logits, next_q_continue_logits),
                    )
                )
            )

        return HierarchicalReasoningModel_ACTV1Carry(
            new_inner_carry, new_steps, halted, new_current_data
        ), outputs

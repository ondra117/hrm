import flax.linen as nn
from flax.typing import Array
import jax.numpy as jnp
import optax
import jax

from models.hrm.hrm_act_v1 import HierarchicalReasoningModel_ACTV1

IGNORE_LABEL_ID = -100


def s(x, epsilon=1e-30):
    den = jnp.where(x < 0, 1 - x + epsilon, 1)
    return jnp.where(x < 0, 1 / den, x + 1)


def log_stablemax(x: Array, axis: int = -1) -> Array:
    s_x = s(x)
    return jnp.log(s_x / jnp.sum(s_x, axis=axis, keepdims=True))


def stablemax_cross_entropy(
    logits: Array, labels: Array, ignore_index: int = -100
) -> Array:
    logprobs = log_stablemax(logits.astype(jnp.float64), axis=-1)

    valid_mask = labels != ignore_index
    transformed_labels = jnp.where(valid_mask, labels, 0)
    prediction_logprobs = jnp.take_along_axis(
        logprobs, indices=transformed_labels.astype(jnp.uint64)[..., None], axis=-1
    )[..., 0]

    return -jnp.where(valid_mask, prediction_logprobs, 0)


def softmax_cross_entropy(logits, labels) -> Array:
    return optax.softmax_cross_entropy_with_integer_labels(logits, labels)


class ACTLossHead(nn.Module):
    model: HierarchicalReasoningModel_ACTV1
    loss_fn: callable

    def setup(self):
        pass

    def init_model(self, batch: Array, **kwargs):
        return self(carry=self.initial_carry(batch=batch), batch=batch, **kwargs)

    def initial_carry(self, *args, **kwargs):
        return self.model.initial_carry(*args, **kwargs)

    def __call__(
        self, **model_kwargs
    ) -> tuple[Array, Array, dict[str, Array], dict[str, Array] | None, Array]:
        new_carry, outputs = self.model(**model_kwargs)
        labels = new_carry.current_data["labels"]

        mask = labels != IGNORE_LABEL_ID
        loss_counts = jnp.sum(mask, axis=-1)
        loss_divisor = jnp.maximum(loss_counts, 1)[..., None]

        is_correct = mask & (jnp.argmax(outputs["logits"], axis=-1) == labels)
        seq_is_correct = jnp.sum(is_correct, axis=-1) == loss_counts

        valid_metrics = new_carry.halted & (loss_counts > 0)
        metrics = {
            "count": valid_metrics.sum(),
            "accuracy": jnp.where(
                valid_metrics,
                jnp.sum(is_correct.astype(jnp.float32) / loss_divisor, axis=-1),
                0,
            ).sum(),
            "exact_accuracy": (valid_metrics & seq_is_correct).sum(),
            "q_halt_accuracy": (
                valid_metrics & ((outputs["q_halt_logits"] >= 0) == seq_is_correct)
            ).sum(),
            "steps": jnp.where(valid_metrics, new_carry.steps, 0).sum(),
        }

        lm_loss = (self.loss_fn(outputs["logits"], labels) / loss_divisor).sum()
        q_halt_loss = jnp.sum(
            optax.sigmoid_binary_cross_entropy(
                outputs["q_halt_logits"],
                seq_is_correct.astype(outputs["q_halt_logits"].dtype),
            )
        )

        metrics["lm_loss"] = jax.lax.stop_gradient(lm_loss)
        metrics["q_halt_loss"] = jax.lax.stop_gradient(q_halt_loss)

        q_continue_loss = jnp.sum(
            optax.sigmoid_binary_cross_entropy(
                outputs["q_continue_logits"], outputs["target_q_continue"]
            )
        )

        metrics["q_continue_loss"] = jax.lax.stop_gradient(q_continue_loss)

        outputs = jax.tree.map(jax.lax.stop_gradient, outputs)

        return (
            new_carry,
            lm_loss + 0.5 * (q_halt_loss + q_continue_loss),
            metrics,
            outputs,
            new_carry.halted.all(),
        )

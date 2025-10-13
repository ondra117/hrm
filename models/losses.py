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


def stablemax_cross_entropy(logits: Array, labels: Array) -> Array:
    logprobs = log_stablemax(logits.astype(jnp.float64), axis=-1)

    prediction_logprobs = jnp.take_along_axis(
        logprobs, indices=labels.astype(jnp.uint64)[..., None], axis=-1
    )[..., 0]

    return -prediction_logprobs


def softmax_cross_entropy(logits, labels) -> Array:
    return optax.softmax_cross_entropy_with_integer_labels(logits, labels)


def act_loss(
    new_carry, outputs
) -> tuple[Array, Array, dict[str, Array], dict[str, Array] | None, Array]:
    labels = new_carry.current_data["labels"]

    is_correct = jnp.argmax(outputs["logits"], axis=-1) == labels
    seq_is_correct = jnp.all(is_correct, axis=-1)

    lm_loss = jnp.mean(stablemax_cross_entropy(outputs["logits"], labels))
    q_halt_loss = jnp.mean(
        optax.sigmoid_binary_cross_entropy(
            outputs["q_halt_logits"],
            seq_is_correct.astype(outputs["q_halt_logits"].dtype),
        )
    )

    q_continue_loss = jnp.mean(
        optax.sigmoid_binary_cross_entropy(
            outputs["q_continue_logits"], outputs["target_q_continue"]
        )
    )

    valid_metrics = new_carry.halted
    metrics = {
        "count": valid_metrics.sum(),
        "accuracy": jnp.where(
            valid_metrics,
            jnp.mean(is_correct.astype(jnp.float32), axis=-1),
            0,
        ).sum(),
        "exact_accuracy": (valid_metrics & seq_is_correct).sum(),
        "q_halt_accuracy": (
            valid_metrics & ((outputs["q_halt_logits"] >= 0) == seq_is_correct)
        ).sum(),
        "steps": jnp.where(valid_metrics, new_carry.steps, 0).sum(),
        "lm_loss": lm_loss,
        "q_halt_loss": q_halt_loss,
        "q_continue_loss": q_continue_loss,
    }

    return (
        new_carry,
        lm_loss + 0.5 * (q_halt_loss + q_continue_loss),
        metrics,
        outputs,
        new_carry.halted.all(),
    )

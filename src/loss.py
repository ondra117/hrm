import flax.linen as nn
import jax.numpy as jnp
import optax


def sigmoid_binary_cross_entropy(logits, labels):
    return (
        -labels * nn.log_sigmoid(logits)
        - (1 - labels) * nn.log_sigmoid(-logits)
        + jnp.where(
            (labels == 0) | (labels == 1),
            0,
            labels * jnp.log(labels) + (1 - labels) * jnp.log(1 - labels),
        )
    )


def hrm_loss(y_pred, x, y_true, q, q_next, m, m_max):
    mask = x == 0
    G_halt = jnp.all(
        jnp.where(mask, jnp.argmax(y_pred, axis=-1) == y_true, 1), axis=(-1,)
    ).astype(jnp.float32)
    G_continue = nn.sigmoid(
        jnp.where(m >= m_max, q_next[..., 0], jnp.max(q_next, axis=-1))
    )
    G = jnp.concat([G_halt[..., None], G_continue[..., None]], axis=-1)
    q_loss = sigmoid_binary_cross_entropy(q, G)

    y_loss = optax.softmax_cross_entropy_with_integer_labels(y_pred, y_true)

    return jnp.sum(y_loss * mask) / jnp.sum(mask) + jnp.mean(q_loss)


# def hrm_loss(y_pred, x, y_true, q, q_next, m, m_max):
#     G_halt = jnp.all(jnp.argmax(y_pred, axis=-1) == y_true, axis=(-1,)).astype(
#         jnp.float32
#     )
#     G_continue = nn.sigmoid(
#         jnp.where(m >= m_max, q_next[..., 0], jnp.max(q_next, axis=-1))
#     )
#     G = jnp.concat([G_halt[..., None], G_continue[..., None]], axis=-1)
#     q_loss = sigmoid_binary_cross_entropy(q, G)

#     y_loss = optax.softmax_cross_entropy_with_integer_labels(y_pred, y_true)

#     return jnp.mean(y_loss) + jnp.mean(q_loss)

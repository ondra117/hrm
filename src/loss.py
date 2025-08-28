import flax.linen as nn
import jax.numpy as jnp


def sigmoid_binary_cross_entropy(logits, labels):
    c_labels = jnp.where((labels == 0) | (labels == 1), 0.5, labels)
    return (
        -labels * nn.log_sigmoid(logits)
        - (1 - labels) * nn.log_sigmoid(-logits)
        + jnp.where(  # TODO:kometar zde
            (labels == 0) | (labels == 1),
            0,
            c_labels * jnp.log(c_labels) + (1 - c_labels) * jnp.log(1 - c_labels),
        )
    )


def stabelmax(logits, labels):
    loglogits = jnp.log(jnp.abs(logits) + 1) * jnp.where(logits < 0, -1, 1)
    return jnp.sum(
        jnp.where(
            nn.one_hot(labels, logits.shape[-1]),
            -loglogits + nn.logsumexp(loglogits, axis=-1, keepdims=True),
            0,
        ),
        axis=-1,
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

    # y_loss = optax.softmax_cross_entropy_with_integer_labels(y_pred, y_true)
    y_loss = stabelmax(y_pred, y_true)

    return jnp.mean(
        jnp.sum(y_loss * mask, axis=-1) / jnp.sum(mask, axis=-1)
    ) + jnp.mean(q_loss)


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

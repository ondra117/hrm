from optax._src import base
import jax
import jax.numpy as jnp
from dataclasses import dataclass
import optax
from optax._src import combine
from optax._src import transform


def adamatan2(
    learning_rate: base.ScalarOrSchedule,
    b1: float = 0.9,
    b2: float = 0.999,
    weight_decay=0.0,
    a=1.27,
    b=1.0,
):
    def init(params):
        return {
            "step": 0,
            "exp_avg": jax.tree.map(lambda x: jnp.zeros_like(x), params),
            "exp_avg_sq": jax.tree.map(lambda x: jnp.zeros_like(x), params),
        }

    def update(grads, state, params):
        step = state["step"] + 1

        exp_avg = jax.tree.map(
            lambda e, g: b1 * e + (1 - b1) * g, state["exp_avg"], grads
        )
        exp_avg_sq = jax.tree.map(
            lambda e, g: b2 * e + (1 - b2) * g**2, state["exp_avg_sq"], grads
        )

        exp_avg_hat = jax.tree.map(lambda m: m / (1 - b1**step), exp_avg)
        exp_avg_sq_hat = jax.tree.map(lambda m: m / (1 - b2**step), exp_avg_sq)

        # grads = jax.tree.map(
        #     lambda e, s: e / (jnp.sqrt(s) + 1e-8), exp_avg_hat, exp_avg_sq_hat
        # )

        grads = jax.tree.map(
            lambda e, s: a * jnp.arctan2(e, b * jnp.sqrt(s)),
            exp_avg_hat,
            exp_avg_sq_hat,
        )

        return grads, {"step": step, "exp_avg": exp_avg, "exp_avg_sq": exp_avg_sq}

    return combine.chain(
        optax.GradientTransformation(init, update),
        transform.add_decayed_weights(weight_decay, None),
        transform.scale_by_learning_rate(learning_rate),
    )

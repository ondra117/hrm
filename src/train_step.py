from flax.typing import Array
import flax.linen as nn
from flax.training.train_state import TrainState
from typing import Callable
import jax
from src.loss import hrm_loss


def train_step(
    state: TrainState, x: Array, z: list[Array], y: Array, m: Array, n_max: int
):
    def step_fn(
        params,
        state: TrainState,
        x: Array,
        z: list[Array],
        y: Array,
        m: Array,
        n_max: int,
    ):
        y_pred, z, q = state.apply_fn({"params": params}, x, z)
        _, _, q_next = state.apply_fn({"params": params}, x, z)
        q_next = jax.lax.stop_gradient(q_next)
        loss = hrm_loss(y_pred, x, y, q, q_next, m, n_max)
        return loss, (y_pred, q, z)

    (loss, (y_pred, q, z)), grads = jax.value_and_grad(step_fn, has_aux=True)(
        state.params, state, x, z, y, m + 1, n_max
    )
    grads = jax.lax.pmean(grads, axis_name="devices")
    state = state.apply_gradients(grads=grads)
    return loss, y_pred, q, m + 1, z, state

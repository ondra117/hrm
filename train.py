import os

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.95"

import rich.traceback

rich.traceback.install()

import jax

# jax.config.update("jax_disable_jit", True)
# jax.config.update("jax_debug_nans", True)

from pathlib import Path
import optax
import numpy as np
import flax.linen as nn
from flax.training.train_state import TrainState
from flax.training import common_utils
from flax import jax_utils
from tqdm import tqdm
from copy import deepcopy
import einops as eo
import jax.numpy as jnp
import wandb
from scipy.special import erf, erfinv

from config import load_config, Config
from data_loader import DataLoader
from src.modul import HRM
from src.train_step import train_step
from adam_atan2 import adamatan2


def lacum_normal(x, a):
    return erfinv(2 * x * erf(a) - erf(a))


config: Config = load_config(Path("config.toml"))

data_loader = DataLoader(config)
val_data_loader = DataLoader(config, val=True)
data_loader.start()

# lr_scheduler = optax.schedules.warmup_cosine_decay_schedule(
#     0,
#     config.train.lr,
#     config.train.steps * config.train.warmup_retio,
#     config.train.steps - (config.train.steps * config.train.warmup_retio),
# )
lr_scheduler = optax.schedules.warmup_constant_schedule(
    0, config.train.lr, config.train.steps * config.train.warmup_retio
)


opt = optax.chain(
    optax.clip_by_global_norm(1),
    optax.adamw(lr_scheduler, b2=0.95, weight_decay=config.train.wd),
)

module = HRM(
    config.model.dim,
    config.model.num_emb,
    config.model.n_heads,
    config.model.n_blocks,
    config.model.h_freqs,
)

x = np.zeros(
    (config.train.train_batch_size, *data_loader.get_batch(0)[0][0].shape),
    dtype=np.uint8,
)
y = np.zeros(
    (config.train.train_batch_size, *data_loader.get_batch(0)[1][0].shape),
    dtype=np.uint8,
)

z_orig = [
    lacum_normal(
        np.random.uniform(
            0, 1, (len(data_loader.get_batch(0)[0][0]) + 1, config.model.dim)
        ),
        2,
    )
    for _ in range(len(config.model.h_freqs))
]

z = [
    np.zeros(
        (config.train.train_batch_size, *z_orig[0].shape),
        dtype=np.float32,
    )
    for _ in range(len(config.model.h_freqs))
]

m = np.full((config.train.train_batch_size), config.train.m_max, dtype=np.uint8)

q = np.zeros((config.train.train_batch_size, 2), dtype=np.float32)

m_min = np.zeros_like(m)

x, y, z, m, q, m_min = jax.tree.map(jnp.array, (x, y, z, m, q, m_min))

print(nn.tabulate(module, jax.random.PRNGKey(0), depth=1)(x, deepcopy(z)))

trainstate = TrainState.create(
    apply_fn=module.apply,
    params=module.init(jax.random.PRNGKey(0), x, z)["params"],
    tx=opt,
)

train_step = jax.pmap(train_step, axis_name="devices")

model_apply = jax.pmap(module.apply, axis_name="devices")

trainstate = jax_utils.replicate(trainstate)

x, y, z, m, q, m_min = common_utils.shard((x, y, z, m, q, m_min))

run = wandb.init(project="hrm")

last_val_step = -config.train.val_freq

y_pred = None

acc = 0
acc_full = 0
n_acc = 0


while trainstate.step[0] <= config.train.steps:
    print(f"{trainstate.step[0]}/{config.train.steps}")
    bar = tqdm(range(len(data_loader)))
    for _ in bar:
        if trainstate.step[0] > config.train.steps:
            break
        loader_batch_x, loader_batch_y = next(data_loader)
        while loader_batch_x:
            if (
                (m == config.train.m_max) | ((q[..., 0] > q[..., 1]) & (m > m_min))
            ).any():
                (x, y, z, m, q, m_min) = jax.tree.map(
                    lambda x: eo.rearrange(x, "d b ... -> (d b) ..."),
                    (x, y, z, m, q, m_min),
                )
                idx = np.argmax((m == config.train.m_max) | (q[:, 0] > q[:, 1]))

                if y_pred is not None:
                    mask = x[idx] == 0
                    y_pred_row = jnp.argmax(
                        eo.rearrange(y_pred, "d b ... -> (d b) ...")[idx], axis=-1
                    )
                    acc += jnp.sum((y_pred_row == y[idx]) * mask) / jnp.sum(mask)
                    acc_full += jnp.mean(
                        jnp.all(jnp.where(mask, y_pred_row == y[idx], 1), axis=-1)
                    )
                    n_acc += 1

                x = x.at[idx].set(loader_batch_x.pop())
                y = y.at[idx].set(loader_batch_y.pop())
                z = [
                    zs.at[idx].set(
                        # np.clip(np.random.normal(0, 1, size=zs.shape[1:]), -3, 3)
                        # lacum_normal(np.random.uniform(0, 1, size=zs.shape[1:]), 2)
                        zo
                    )
                    for zs, zo in zip(z, z_orig)
                ]
                m = m.at[idx].set(0)
                q = q.at[idx].set(0)
                m_min = m_min.at[idx].set(
                    (np.random.uniform(0, 1) < config.train.halt_exploration_prob)
                    * np.random.randint(1, config.train.m_max)
                )
                x, y, z, m, q, m_min = common_utils.shard((x, y, z, m, q, m_min))
                continue
            loss, y_pred, q, m, z, trainstate = train_step(
                trainstate, x, z, y, m, jax_utils.replicate(config.train.m_max)
            )
            run.log(
                {"loss": np.mean(loss), "lr": lr_scheduler(trainstate.step[0])},
                step=trainstate.step[0],
            )
            bar.set_postfix(loss=np.mean(loss))

        if last_val_step + config.train.val_freq > trainstate.step[0]:
            continue

        n_acc = max(n_acc, 1)

        run.log(
            {"acc": acc / n_acc, "acc_full": acc_full / n_acc}, step=trainstate.step[0]
        )
        print(f"acc: {acc / n_acc}\nacc_full: {acc_full / n_acc}")

        last_val_step = trainstate.step[0]

        xv = jnp.zeros_like(x)
        yv = jnp.zeros_like(y)
        y_predv = None
        zv = [jnp.zeros_like(zs) for zs in z]
        mv = jnp.full_like(m, config.train.m_max)
        qv = jnp.zeros_like(q)

        acc = 0
        acc_full = 0
        n_acc = 0
        m_mean = 0

        for _ in tqdm(range(len(val_data_loader) * 2)):
            loader_batch_x, loader_batch_y = next(val_data_loader)
            while loader_batch_x:
                if ((mv == config.train.m_max) | (qv[..., 0] > qv[..., 1])).any():
                    (xv, yv, zv, mv, qv) = jax.tree.map(
                        lambda x: eo.rearrange(x, "d b ... -> (d b) ..."),
                        (xv, yv, zv, mv, qv),
                    )
                    idx = np.argmax((mv == config.train.m_max) | (qv[:, 0] > qv[:, 1]))

                    if y_predv is not None:
                        m_mean += int(mv[idx])
                        mask = xv[idx] == 0
                        y_predv_row = jnp.argmax(
                            eo.rearrange(y_predv, "d b ... -> (d b) ...")[idx], axis=-1
                        )
                        acc += jnp.sum((y_predv_row == yv[idx]) * mask) / jnp.sum(mask)
                        acc_full += jnp.mean(
                            jnp.all(
                                jnp.where(mask, y_predv_row == yv[idx], 1),
                                axis=-1,
                            )
                        )
                        n_acc += 1

                    xv = xv.at[idx].set(loader_batch_x.pop())
                    yv = yv.at[idx].set(loader_batch_y.pop())
                    zv = [
                        zvs.at[idx].set(
                            # np.clip(np.random.normal(0, 1, size=zvs.shape[1:]), -3, 3)
                            # lacum_normal(np.random.uniform(0, 1, size=zvs.shape[1:]), 2)
                            zo
                        )
                        for zvs, zo in zip(zv, z_orig)
                    ]
                    mv = mv.at[idx].set(0)
                    qv = qv.at[idx].set(0)
                    xv, yv, zv, mv, qv = common_utils.shard((xv, yv, zv, mv, qv))
                    continue
                y_predv, zv, qv = model_apply({"params": trainstate.params}, xv, zv)
                mv += 1
        run.log(
            {
                "val_acc": acc / n_acc,
                "val_acc_full": acc_full / n_acc,
                "m": m_mean / n_acc,
            },
            step=trainstate.step[0],
        )
        print(f"val_acc: {acc / n_acc}\nval_acc_full: {acc_full / n_acc}")

        acc = 0
        acc_full = 0
        n_acc = 0

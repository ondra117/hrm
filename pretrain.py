import rich.traceback

rich.traceback.install()

import jax

# jax.config.update("jax_disable_jit", True)
# jax.config.update("jax_debug_nans", True)

import flax.linen as nn
from flax.typing import Array
import pydantic
from dataclasses import dataclass
import jax
from flax.training import train_state
from flax import struct
import optax
import einops as eo
from flax.training import common_utils
import jax.numpy as jnp
from pathlib import Path
from flax import jax_utils
import tqdm
import wandb

from puzzle_dataset import PuzzleDataset, PuzzleDatasetConfig, PuzzleDatasetMetadata
from models.hrm.hrm_act_v1 import (
    HierarchicalReasoningModel_ACTV1,
    HierarchicalReasoningModel_ACTV1Config,
    HierarchicalReasoningModel_ACTV1Carry,
)
from models.losses import ACTLossHead, stablemax_cross_entropy
from adam_atan2 import adamatan2
from config import Config, load_config


@struct.dataclass
class TrainState(train_state.TrainState):
    constants: dict  # můžeš dát PyTree, FrozenDict apod.


class LossConfig(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra="allow")

    name: str


def create_dataloader(config: Config, split: str, **kwargs):
    dataset = PuzzleDataset(
        PuzzleDatasetConfig(
            seed=config.seed,
            dataset_path=config.data_path,
            rank=0,
            num_replicas=1,
            **kwargs,
        ),
        split=split,
    )
    # dataloader = DataLoader(
    #     dataset,
    #     batch_size=None,
    #     num_workers=1,
    #     prefetch_factor=8,
    #     pin_memory=True,
    #     persistent_workers=True,
    # )
    return dataset, dataset.metadata


def create_model(config: Config, train_metadata: PuzzleDatasetMetadata):
    model_cfg = HierarchicalReasoningModel_ACTV1Config(
        **dict(
            config.arch,  # type: ignore
            vocab_size=train_metadata.vocab_size,
            seq_len=train_metadata.seq_len,
            num_puzzle_identifiers=train_metadata.num_puzzle_identifiers,
            causal=False,  # Non-autoregressive
        )
    )

    model = ACTLossHead(
        HierarchicalReasoningModel_ACTV1(model_cfg), stablemax_cross_entropy
    )

    lr_scheduler = optax.schedules.warmup_cosine_decay_schedule(
        0,
        config.lr,
        round(config.lr_warmup_steps),
        int(
            config.epochs
            * train_metadata.total_groups
            * train_metadata.mean_puzzle_examples
            / config.global_batch_size
        )
        - round(config.lr_warmup_steps),
    )

    opt = adamatan2(lr_scheduler, config.beta1, config.beta2, config.weight_decay)

    return model, opt, lr_scheduler


def init_train_state(
    config: Config, train_metadata: PuzzleDatasetMetadata, batch: Array, key: Array
):
    model, opt, lr_scheduler = create_model(config, train_metadata)

    vars = model.init(
        jax.random.PRNGKey(0), batch=batch, key=key, method=ACTLossHead.init_model
    )

    return TrainState.create(
        apply_fn=model.apply,
        params=vars["params"],
        constants=vars["constants"],
        tx=opt,
    ), lr_scheduler


def train_step(train_state, carry, batch, global_batch_size, key):
    def _step(params, train_state, carry, batch, global_batch_size, key):
        carry, loss, metrics, _, _ = train_state.apply_fn(
            {"params": params, "constants": train_state.constants},
            carry=carry,
            batch=batch,
            key=key,
        )
        loss /= global_batch_size

        return loss, (carry, metrics)

    (loss, (carry, metrics)), grads = jax.value_and_grad(_step, has_aux=True)(
        train_state.params, train_state, carry, batch, global_batch_size, key
    )
    # (loss, (carry, metrics)), grads = jax.pmap(
    #     jax.value_and_grad(_step, has_aux=True), axis_name="devices"
    # )(train_state.params, train_state, carry, batch, global_batch_size, key)

    grads = jax.lax.psum(grads, axis_name="devices")
    train_state = train_state.apply_gradients(grads=grads)

    # train_state = jax.pmap(
    #     lambda x, s: s.apply_gradients(grads=jax.lax.psum(x, axis_name="devices")),
    #     axis_name="devices",
    # )(grads, train_state)

    return train_state, carry, metrics, loss


train_step = jax.pmap(train_step, axis_name="devices")


def train_batch(
    train_state: TrainState,
    carry,
    batch,
    global_batch_size: int,
    lr_scheduler,
    key: Array,
):
    if carry is None:
        carry = jax.pmap(
            lambda *args, **kwargs: train_state.apply_fn(
                *args,
                **kwargs,
                method=ACTLossHead.initial_carry,
            ),
            axis_name="devices",
        )(
            {"params": train_state.params, "constants": train_state.constants},
            batch=common_utils.shard(batch),
        )

    train_state, carry, metrics, loss = train_step(
        train_state,
        carry,
        common_utils.shard(batch),
        jax_utils.replicate(global_batch_size),
        jax.random.split(key, jax.device_count()),
    )

    metrics = jax.tree.map(jnp.sum, metrics)

    count = jnp.maximum(metrics["count"], 1)
    metrics = {
        f"train/{k}": v / (global_batch_size if k.endswith("loss") else count)
        for k, v in metrics.items()
    }
    metrics["train/lr"] = lr_scheduler(train_state.step[0])
    return metrics, train_state, carry


def evaluate(
    config: Config,
    train_state: TrainState,
    eval_loader: PuzzleDataset,
    eval_metadata: PuzzleDatasetMetadata,
    key: Array,
):
    apply_fn = jax.pmap(train_state.apply_fn, axis_name="devices")
    set_ids = {k: idx for idx, k in enumerate(eval_metadata.sets)}

    # all_preds = {}

    metric_values = None
    metric_global_batch_size = [0 for _ in range(len(set_ids))]

    carry = None
    for set_name, batch, global_batch_size in tqdm.tqdm(list(eval_loader)):
        batch = common_utils.shard(batch)

        carry = jax.pmap(
            lambda *args, **kwargs: train_state.apply_fn(
                *args,
                **kwargs,
                method=ACTLossHead.initial_carry,
            ),
            axis_name="devices",
        )(
            {"params": train_state.params, "constants": train_state.constants},
            batch=batch,
        )

        while True:
            key, sub_key = jax.random.split(key)
            carry, _, metrics, preds, all_finish = apply_fn(
                {"params": train_state.params, "constants": train_state.constants},
                carry=carry,
                batch=batch,
                key=jax.random.split(sub_key, jax.device_count()),
            )

            if all_finish.all():
                break
            else:
                carry = HierarchicalReasoningModel_ACTV1Carry(
                    carry.inner_carry,
                    carry.steps,
                    jnp.zeros_like(carry.halted),
                    carry.current_data,
                )

        # preds = {k: preds[k] for k in config.eval_save_outputs if k in preds}

        batch, preds = jax.tree.map(
            lambda x: eo.rearrange(x, "d b ... -> (d b) ..."), (batch, preds)
        )

        metrics = jax.tree.map(jnp.sum, metrics)

        # for collection in (batch, preds):
        #     for k, v in collection.items():
        #         if k in config.eval_save_outputs:
        #             all_preds.setdefault(k, [])
        #             all_preds[k].append(v)

        set_id = set_ids[set_name]

        if metric_values is None:
            metric_values = [
                jax.tree.map(jnp.zeros_like, metrics) for _ in range(len(set_ids))
            ]

        metric_values[set_id] = jax.tree.map(
            lambda x, y: x + y, metric_values[set_id], metrics
        )
        metric_global_batch_size[set_id] += global_batch_size

    reduced_metrics = {
        set_name: {
            metric_name: metric_values[set_id][metric_name]
            for metric_name in metrics.keys()
        }
        for set_id, set_name in enumerate(set_ids)
    }

    for set_name, metrics in reduced_metrics.items():
        count = metrics.pop("count")
        reduced_metrics[set_name] = {k: v / count for k, v in metrics.items()}

    return reduced_metrics


def launch():
    config: Config = load_config(Path().joinpath("config.toml"))

    train_epochs_per_iter = (
        config.eval_interval if config.eval_interval is not None else config.epochs
    )
    total_iters = config.epochs // train_epochs_per_iter

    assert config.epochs % train_epochs_per_iter == 0, (
        "Eval interval must be a divisor of total epochs."
    )

    train_loader, train_metadata = create_dataloader(
        config,
        "train",
        test_set_mode=False,
        epochs_per_iter=train_epochs_per_iter,
        global_batch_size=config.global_batch_size,
    )
    eval_loader, eval_metadata = create_dataloader(
        config,
        "test",
        test_set_mode=True,
        epochs_per_iter=1,
        global_batch_size=config.global_batch_size,
    )

    train_state = None
    lr_scheduler = None

    progress_bar = tqdm.tqdm(
        total=int(
            config.epochs
            * train_metadata.total_groups
            * train_metadata.mean_puzzle_examples
            / config.global_batch_size
        )
    )
    wandb.init(project="hrm")

    rng_key = jax.random.PRNGKey(0)

    carry = None

    for _iter_id in range(total_iters):
        print(f"Epoch {_iter_id * train_epochs_per_iter}")

        for set_name, batch, global_batch_size in train_loader:
            if train_state is None:
                rng_key, sub_key = jax.random.split(rng_key)
                train_state, lr_scheduler = init_train_state(
                    config, train_metadata, batch, sub_key
                )
                train_state = jax_utils.replicate(train_state)

            rng_key, sub_key = jax.random.split(rng_key)

            metrics, train_state, carry = train_batch(
                train_state,
                carry,
                batch,
                global_batch_size,
                lr_scheduler,
                sub_key,
            )

            wandb.log(metrics, step=train_state.step[0])
            progress_bar.update(int(train_state.step[0]) - progress_bar.n)

        rng_key, sub_key = jax.random.split(rng_key)

        metrics = evaluate(config, train_state, eval_loader, eval_metadata, sub_key)

        wandb.log(metrics, step=train_state.step[0])

    wandb.finish()


if __name__ == "__main__":
    launch()

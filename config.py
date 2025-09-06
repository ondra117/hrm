from pydantic import BaseModel
import toml
import jax.numpy as jnp


class Arch(BaseModel):
    loss_type: str

    halt_exploration_prob: float
    halt_max_steps: int

    H_cycles: int
    L_cycles: int

    H_layers: int
    L_layers: int

    hidden_size: int
    num_heads: int
    expansion: int

    puzzle_emb_ndim: int

    pos_encodings: str


class Config(BaseModel):
    arch: Arch

    seed: int

    data_path: str

    global_batch_size: int

    epochs: int
    eval_interval: int

    lr: float
    lr_warmup_steps: int

    beta1: float
    beta2: float
    weight_decay: float


def load_config(path):
    return Config(**toml.load(path))

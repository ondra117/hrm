from pydantic import BaseModel
import toml


class Train(BaseModel):
    train_batch_size: int
    batch_size: int
    steps: int
    val_freq: int
    halt_exploration_prob: float
    warmup_retio: float
    lr: float
    m_max: int
    wd: float


class Model(BaseModel):
    dim: int
    num_emb: int
    n_heads: int
    n_blocks: int
    h_freqs: list[int]


class Config(BaseModel):
    train: Train
    model: Model


def load_config(path):
    return Config(**toml.load(path))

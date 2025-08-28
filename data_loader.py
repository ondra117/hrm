from pathlib import Path
import numpy as np
import einops as eo
from random import choice

import base_data_loader
from config import Config


class DataLoader(base_data_loader.DataLoader):
    def __init__(
        self,
        config: Config,
        val: bool = False,
        use_multiprocesing=True,
        num_workers=None,
        max_queued_batches=8,
        warmup_queue=True,
        disable_warnings=False,
    ):
        self.config = config
        data_path = Path().joinpath(
            "data", "sudoku-extreme-1k-aug-1000", "test" if val else "train"
        )
        # data_path = Path().joinpath(
        #     "data", "sudoku-extreme-full", "test" if val else "train"
        # )

        data_len = 1010 if val else 422786
        # data_len = 1000 if val else 1000

        with open(data_path.joinpath("all__inputs.npy"), "rb") as f:
            self.inputs = eo.rearrange(
                np.load(f)[:data_len] - 1,
                "(i b) ... -> i b ...",
                b=self.config.train.batch_size,
            ).astype(np.uint8)

        with open(data_path.joinpath("all__labels.npy"), "rb") as f:
            self.labels = eo.rearrange(
                np.load(f)[:data_len] - 1,
                "(i b) ... -> i b ...",
                b=self.config.train.batch_size,
            ).astype(np.uint8)

        super().__init__(
            use_multiprocesing and not val,
            num_workers,
            max_queued_batches,
            warmup_queue,
            disable_warnings,
        )

    def __len__(self):
        # return 20
        return len(self.inputs)

    def get_batch(self, idx):
        def augment(inp):
            return inp
            s = np.arange(1, 10, dtype=np.uint8)
            trans = choice(
                [[], [0], [0, 1], [0, 1, 0], [0, 1, 0, 1], [1, 0, 1], [1, 0], [1]]
            )

            def flip(x):
                x = eo.rearrange(x, "... (a b) -> ... a b", a=9, b=9)
                for o in trans:
                    if o == 0:
                        x = x.T
                    if o == 1:
                        x = x[..., ::-1, :]
                return eo.rearrange(x, "... a b -> ... (a b)")

            np.random.shuffle(s)
            s = np.concat([np.array([0], dtype=np.uint8), s])
            return map(lambda x: flip(s[x]), inp)

        return tuple(
            map(list, zip(*map(augment, zip(self.inputs[idx], self.labels[idx]))))
        )


if __name__ == "__main__":
    from config import load_config, Config

    config: Config = load_config(Path("config.toml"))

    data_loader = DataLoader(config)

    print(len)

    print(data_loader.get_batch(0)[0][0].reshape((9, 9)))
    print(data_loader.get_batch(0)[1][0].reshape((9, 9)))
    print(data_loader.get_batch(0)[0][0].reshape((9, 9)))

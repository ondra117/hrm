from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any
from flax.typing import Array
import multiprocessing as mp
from time import sleep
import warnings
import heapq


class Cycle:
    def __init__(self, iterable):
        self.data = [i for i in iterable]
        self.idx = 0

    def __next__(self):
        out = self.data[self.idx]
        self.idx += 1
        if self.idx >= len(self.data):
            self.idx = 0
        return out


class DataLoader(ABC):
    def __init__(
        self,
        use_multiprocesing: bool = True,
        num_workers: int | None = None,
        max_queued_batches: int = 8,
        warmup_queue: bool = True,
        disable_warnings: bool = False,
    ) -> None:
        if not num_workers:
            num_workers = mp.cpu_count()
        self.use_multiprocesing = use_multiprocesing
        self.num_workers = num_workers
        self.max_queued_batches = max_queued_batches
        self.warmup_queue = warmup_queue
        self.disable_warnings = disable_warnings
        self.proces_id = None

        self.get_batch_idx = Cycle(range(len(self)))

        if use_multiprocesing:
            self._batch_queue: mp.Queue = mp.Queue()
            self._sorted_batch_queue: mp.Queue = mp.Queue(self.max_queued_batches)
            self._task_queue: mp.Queue = mp.Queue()

            self._workers = [
                mp.Process(target=self._worker, args=(id,), daemon=True)
                for id in range(num_workers)
            ]

            for worker in self._workers:
                worker.start()
            mp.Process(target=self._sorter, daemon=True).start()
            self.idx = 0

    def start(self) -> None:
        if not self.use_multiprocesing:
            return
        for _ in range(self.max_queued_batches):
            batch_idx = next(self.get_batch_idx)
            self._task_queue.put((batch_idx, self.idx))
            self.idx += 1

        while self.warmup_queue and not self._sorted_batch_queue.full():
            sleep(0.1)

    @abstractmethod
    def __len__(self): ...

    @abstractmethod
    def get_batch(self, idx: int) -> Array: ...

    def proces_start(self) -> None: ...

    def _worker(self, id) -> None:
        self.proces_id = id
        self.proces_start()
        while True:
            batch_idx, idx = self._task_queue.get()
            batch = self.get_batch(batch_idx)
            self._batch_queue.put((batch, idx))

    def _sorter(self) -> None:
        @dataclass(order=True)
        class Data:
            batch: Any = field(compare=False)
            idx: int

        idx = 0
        heap: list[Data] = []
        while True:
            if heap and heap[0].idx == idx:
                self._sorted_batch_queue.put(heapq.heappop(heap).batch)
                idx += 1
            else:
                batch, batch_idx = self._batch_queue.get()
                heapq.heappush(heap, Data(batch, batch_idx))

    def __next__(self) -> Array:
        if not self.use_multiprocesing:
            return self.get_batch(next(self.get_batch_idx))

        if not self.disable_warnings and self._sorted_batch_queue.empty():
            warnings.warn(
                f"Batches are not preparing fast enought. Consider optimizing `{self.__class__.__name__}.{self.get_batch.__name__}` method"
            )

        batch = self._sorted_batch_queue.get()
        batch_idx = next(self.get_batch_idx)
        self._task_queue.put((batch_idx, self.idx))
        self.idx += 1
        return batch

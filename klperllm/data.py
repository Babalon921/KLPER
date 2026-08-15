import pickle
from pathlib import Path

import numpy as np
import torch


class ShardedBinDataset:
    def __init__(self, data_dir: str, split: str):
        self.data_dir = Path(data_dir)
        pattern = f"{split}_*.bin"
        self.shard_paths = sorted(self.data_dir.glob(pattern))
        if not self.shard_paths:
            single = self.data_dir / f"{split}.bin"
            if single.exists():
                self.shard_paths = [single]
        if not self.shard_paths:
            raise FileNotFoundError(
                f"no '{split}' shards found in {self.data_dir} "
                f"(expected {pattern} or {split}.bin) — run a data/prepare_*.py script first"
            )
        self.shard_sizes = [
            np.memmap(p, dtype=np.uint16, mode='r').shape[0] for p in self.shard_paths
        ]

    def get_meta(self):
        meta_path = self.data_dir / "meta.pkl"
        if meta_path.exists():
            with open(meta_path, "rb") as f:
                return pickle.load(f)
        return None

    def get_batch(self, batch_size: int, block_size: int, device: str, generator=None):
        weights = np.array(self.shard_sizes, dtype=np.float64)
        weights /= weights.sum()
        shard_idx = np.random.choice(len(self.shard_paths), p=weights)
        data = np.memmap(self.shard_paths[shard_idx], dtype=np.uint16, mode='r')

        max_start = len(data) - block_size - 1
        if max_start <= 0:
            raise ValueError(
                f"shard {self.shard_paths[shard_idx]} has only {len(data)} tokens, "
                f"too short for block_size={block_size}"
            )
        ix = torch.randint(max_start, (batch_size,), generator=generator)
        x = torch.stack([torch.from_numpy(data[i:i + block_size].astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy(data[i + 1:i + 1 + block_size].astype(np.int64)) for i in ix])

        if device == 'cuda':
            x = x.pin_memory().to(device, non_blocking=True)
            y = y.pin_memory().to(device, non_blocking=True)
        else:
            x, y = x.to(device), y.to(device)
        return x, y

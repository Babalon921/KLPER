import multiprocessing as mp
import os
import time
from pathlib import Path

import numpy as np
import tiktoken

_enc = tiktoken.get_encoding("gpt2")
EOT = _enc._special_tokens['<|endoftext|>']


def tokenize_doc(text: str) -> np.ndarray:
    ids = [EOT]
    ids.extend(_enc.encode_ordinary(text))
    arr = np.asarray(ids, dtype=np.int64)
    assert (arr >= 0).all() and (arr < 2**16).all(), "token id out of uint16 range"
    arr = arr.astype(np.uint16)
    return arr.astype(np.uint16)


def prepare_dataset(
    doc_iterable,
    text_field: str,
    out_dir: str,
    shard_size: int,
    max_tokens: int,
    val_tokens: int,
    num_proc: int | None = None,
):
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    num_proc = num_proc or max(1, (os.cpu_count() or 4) - 1)

    texts = (doc[text_field] for doc in doc_iterable)

    total_written = 0
    val_written = 0
    shard_idx = 0
    buffer = np.empty((shard_size,), dtype=np.uint16)
    buffer_used = 0
    split = "val"  # fill val.bin, then switch to train
    t0 = time.time()

    def flush(split_name, idx, n):
        nonlocal shard_idx
        path = out_path / f"{split_name}_{idx:06d}.bin"
        buffer[:n].tofile(path)
        return path

    with mp.Pool(num_proc) as pool:
        for tokens in pool.imap(tokenize_doc, texts, chunksize=16):
            pos = 0
            while pos < len(tokens):
                if split == "val" and val_written >= val_tokens:
                    split = "train"
                    if buffer_used > 0:
                        flush("val", shard_idx, buffer_used)
                        total_written += buffer_used
                        buffer_used = 0
                        shard_idx = 0

                space = shard_size - buffer_used
                take = min(space, len(tokens) - pos)
                buffer[buffer_used:buffer_used + take] = tokens[pos:pos + take]
                buffer_used += take
                pos += take
                if split == "val":
                    val_written += take

                if buffer_used == shard_size:
                    flush(split, shard_idx, buffer_used)
                    total_written += buffer_used
                    shard_idx += 1
                    buffer_used = 0

                if total_written + buffer_used >= max_tokens:
                    break

            elapsed = time.time() - t0
            print(
                f"\r[{split}] {total_written + buffer_used:,} / {max_tokens:,} tokens "
                f"({(total_written + buffer_used) / max(elapsed, 1e-9):,.0f} tok/s)",
                end="", flush=True,
            )
            if total_written + buffer_used >= max_tokens:
                break

    if buffer_used > 0:
        flush(split, shard_idx, buffer_used)
        total_written += buffer_used

    print()
    print(f"done: wrote {total_written:,} tokens to {out_path} "
          f"({val_written:,} held out for validation)")

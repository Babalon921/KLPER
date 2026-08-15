"""canonical hello-world
"""

import argparse
import pickle
import urllib.request
from pathlib import Path

import numpy as np

SHAKESPEARE_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-dir", default="data/shakespeare_char")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    input_path = out_dir / "input.txt"

    if not input_path.exists():
        print(f"downloading {SHAKESPEARE_URL} ...")
        urllib.request.urlretrieve(SHAKESPEARE_URL, input_path)

    data = input_path.read_text()
    chars = sorted(set(data))
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}

    n = len(data)
    train_data = data[:int(n * 0.9)]
    val_data = data[int(n * 0.9):]

    train_ids = np.array([stoi[c] for c in train_data], dtype=np.uint16)
    val_ids = np.array([stoi[c] for c in val_data], dtype=np.uint16)
    train_ids.tofile(out_dir / "train_000000.bin")
    val_ids.tofile(out_dir / "val_000000.bin")

    with open(out_dir / "meta.pkl", "wb") as f:
        pickle.dump({"vocab_size": len(chars), "stoi": stoi, "itos": itos}, f)

    print(f"vocab size: {len(chars)}")
    print(f"train tokens: {len(train_ids):,}, val tokens: {len(val_ids):,}")


if __name__ == "__main__":
    main()

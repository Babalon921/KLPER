"""
     python data/prepare_pile.py --max-tokens 2_000_000_000 --out-dir data/pile
"""

import argparse

from common import prepare_dataset
from datasets import load_dataset


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--hf-name", default="monology/pile-uncopyrighted")
    p.add_argument("--out-dir", default="data/pile")
    p.add_argument("--max-tokens", type=int, default=2_000_000_000,
                    help="total tokens (train+val) to write; default 2B fits a few-hour single-GPU run")
    p.add_argument("--val-tokens", type=int, default=10_000_000)
    p.add_argument("--shard-size", type=int, default=100_000_000)
    p.add_argument("--num-proc", type=int, default=None)
    args = p.parse_args()

    ds = load_dataset(args.hf_name, split="train", streaming=True)
    prepare_dataset(
        doc_iterable=ds,
        text_field="text",
        out_dir=args.out_dir,
        shard_size=args.shard_size,
        max_tokens=args.max_tokens,
        val_tokens=args.val_tokens,
        num_proc=args.num_proc,
    )


if __name__ == "__main__":
    main()

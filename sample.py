""" python sample.py --ckpt out/ckpt.pt --prompt "You are KLPER" --max-new-tokens 200
"""

import argparse
import pickle
from pathlib import Path

import torch

from klperllm.model import GPT, GPTConfig


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="out/ckpt.pt")
    p.add_argument("--prompt", default="\n")
    p.add_argument("--max-new-tokens", type=int, default=200)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-k", type=int, default=200)
    p.add_argument("--num-samples", type=int, default=1)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=1337)
    args = p.parse_args()

    torch.manual_seed(args.seed)

    checkpoint = torch.load(args.ckpt, map_location=args.device)
    model_cfg = GPTConfig(**checkpoint["model_config"])
    model = GPT(model_cfg)
    model.load_state_dict(checkpoint["model"])
    model.to(args.device)
    model.eval()

    data_dir = checkpoint.get("train_config", {}).get("data_dir")
    meta = None
    if data_dir:
        meta_path = Path(data_dir) / "meta.pkl"
        if meta_path.exists():
            with open(meta_path, "rb") as f:
                meta = pickle.load(f)

    if meta:
        encode = lambda s: [meta["stoi"][c] for c in s]
        decode = lambda ids: "".join(meta["itos"][i] for i in ids)
    else:
        import tiktoken
        enc = tiktoken.get_encoding("gpt2")
        encode = lambda s: enc.encode_ordinary(s)
        decode = lambda ids: enc.decode(ids)

    prompt_ids = torch.tensor(encode(args.prompt), dtype=torch.long, device=args.device)[None, ...]

    for i in range(args.num_samples):
        with torch.no_grad():
            out_ids = model.generate(
                prompt_ids, args.max_new_tokens, temperature=args.temperature, top_k=args.top_k
            )
        print(f"--- sample {i + 1} ---")
        print(decode(out_ids[0].tolist()))
        print()


if __name__ == "__main__":
    main()

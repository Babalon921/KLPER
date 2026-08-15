"""
    python train.py --config configs/fineweb_edu_a10_nano.yaml
    python train.py --config configs/debug_shakespeare_char.yaml --max_iters=200
"""

import argparse
import math
import time
from contextlib import nullcontext

import torch

from klperllm.config import TrainConfig, ensure_out_dir, load_config
from klperllm.data import ShardedBinDataset
from klperllm.model import GPT, GPTConfig


def get_lr(it: int, cfg: TrainConfig) -> float:
    if not cfg.decay_lr:
        return cfg.learning_rate
    if it < cfg.warmup_iters:
        return cfg.learning_rate * (it + 1) / (cfg.warmup_iters + 1)
    if it > cfg.lr_decay_iters:
        return cfg.min_lr
    decay_ratio = (it - cfg.warmup_iters) / (cfg.lr_decay_iters - cfg.warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return cfg.min_lr + coeff * (cfg.learning_rate - cfg.min_lr)


@torch.no_grad()
def estimate_loss(model, datasets, cfg, ctx):
    model.eval()
    out = {}
    for split, ds in datasets.items():
        losses = torch.zeros(cfg.eval_iters)
        for k in range(cfg.eval_iters):
            x, y = ds.get_batch(cfg.batch_size, cfg.block_size, cfg.device)
            with ctx:
                _, loss = model(x, y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


def build_model(cfg: TrainConfig, meta) -> GPT:
    vocab_size = meta["vocab_size"] if meta else cfg.vocab_size
    model_cfg = GPTConfig(
        block_size=cfg.block_size,
        vocab_size=vocab_size,
        n_layer=cfg.n_layer,
        n_head=cfg.n_head,
        n_embd=cfg.n_embd,
        dropout=cfg.dropout,
        bias=cfg.bias,
    )
    return GPT(model_cfg)


def decode_preview(ids, meta):
    if meta:
        return "".join(meta["itos"][i] for i in ids)
    import tiktoken
    return tiktoken.get_encoding("gpt2").decode(ids)


def encode_prompt(text, meta, device):
    if meta:
        ids = [meta["stoi"][c] for c in text]
    else:
        import tiktoken
        ids = tiktoken.get_encoding("gpt2").encode_ordinary(text)
    return torch.tensor(ids, dtype=torch.long, device=device)[None, ...]


def log_metrics(wandb_run, tb_writer, iter_num, metrics: dict):
    if wandb_run:
        wandb_run.log({"iter": iter_num, **metrics})
    if tb_writer:
        for key, value in metrics.items():
            tb_writer.add_scalar(key, value, iter_num)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None, help="enter path to YAML config preset")
    args, overrides = parser.parse_known_args()
    cfg = load_config(args.config, overrides)

    torch.manual_seed(cfg.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    device_type = 'cuda' if 'cuda' in cfg.device else 'cpu'
    dtype = {
        'float32': torch.float32,
        'bfloat16': torch.bfloat16,
        'float16': torch.float16,
    }[cfg.dtype]
    ctx = (
        nullcontext() if device_type == 'cpu'
        else torch.amp.autocast(device_type=device_type, dtype=dtype)
    )

    train_ds = ShardedBinDataset(cfg.data_dir, "train")
    val_ds = ShardedBinDataset(cfg.data_dir, "val")
    meta = train_ds.get_meta()
    datasets = {"train": train_ds, "val": val_ds}

    out_dir = ensure_out_dir(cfg)
    ckpt_path = out_dir / "ckpt.pt"

    iter_num = 0
    best_val_loss = float("inf")

    if cfg.init_from == "resume" and ckpt_path.exists():
        print(f"resuming from {ckpt_path}")
        checkpoint = torch.load(ckpt_path, map_location=cfg.device)
        model_cfg = GPTConfig(**checkpoint["model_config"])
        model = GPT(model_cfg)
        model.load_state_dict(checkpoint["model"])
        iter_num = checkpoint["iter_num"]
        best_val_loss = checkpoint["best_val_loss"]
    else:
        model = build_model(cfg, meta)

    model.to(cfg.device)
    scaler = torch.amp.GradScaler(enabled=(cfg.dtype == 'float16'))
    optimizer = model.configure_optimizers(
        cfg.weight_decay, cfg.learning_rate, (cfg.beta1, cfg.beta2), device_type
    )
    if cfg.init_from == "resume" and ckpt_path.exists():
        optimizer.load_state_dict(checkpoint["optimizer"])

    raw_model = model
    if cfg.compile:
        print("compiling model (torch.compile)...")
        model = torch.compile(model)

    n_params = raw_model.num_params()
    print(f"model: {n_params / 1e6:.1f}M params, block_size={cfg.block_size}, "
          f"vocab_size={raw_model.config.vocab_size}")
    print(f"effective batch size: {cfg.batch_size * cfg.grad_accum_steps * cfg.block_size:,} tokens/step")

    wandb_run = None
    if cfg.wandb_log:
        import wandb
        wandb_run = wandb.init(project=cfg.wandb_project, name=cfg.wandb_run_name, config=vars(cfg))

    tb_writer = None
    if cfg.tensorboard_log:
        from torch.utils.tensorboard import SummaryWriter
        tb_dir = cfg.tensorboard_dir or str(out_dir / "tensorboard")
        tb_writer = SummaryWriter(tb_dir)
        print(f"logging to TensorBoard at {tb_dir} tensorboard --logdir {tb_dir})")

    x, y = train_ds.get_batch(cfg.batch_size, cfg.block_size, cfg.device)
    t0 = time.time()

    while iter_num <= cfg.max_iters:
        lr = get_lr(iter_num, cfg)
        for group in optimizer.param_groups:
            group["lr"] = lr

        if iter_num % cfg.eval_interval == 0:
            losses = estimate_loss(model, datasets, cfg, ctx)
            print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}, lr {lr:.2e}")
            log_metrics(wandb_run, tb_writer, iter_num,
                        {"train/loss": losses["train"], "val/loss": losses["val"], "lr": lr})
            if losses["val"] < best_val_loss or cfg.always_save_checkpoint:
                best_val_loss = losses["val"]
                if iter_num > 0:
                    torch.save({
                        "model": raw_model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "model_config": vars(raw_model.config),
                        "iter_num": iter_num,
                        "best_val_loss": best_val_loss,
                        "train_config": vars(cfg),
                    }, ckpt_path)
                    print(f"  saved checkpoint to {ckpt_path}")

        if iter_num > 0 and cfg.sample_every > 0 and iter_num % cfg.sample_every == 0:
            raw_model.eval()
            prompt_ids = encode_prompt(cfg.sample_prompt, meta, cfg.device)
            with torch.no_grad(), ctx:
                out_ids = raw_model.generate(prompt_ids, cfg.sample_max_new_tokens, temperature=0.8, top_k=200)
            print("  sample:", repr(decode_preview(out_ids[0].tolist(), meta)))
            raw_model.train()

        for micro_step in range(cfg.grad_accum_steps):
            with ctx:
                _, loss = model(x, y)
                loss = loss / cfg.grad_accum_steps
            x, y = train_ds.get_batch(cfg.batch_size, cfg.block_size, cfg.device)
            scaler.scale(loss).backward()

        if cfg.grad_clip != 0.0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

        if iter_num % cfg.log_interval == 0:
            dt = time.time() - t0
            t0 = time.time()
            tok_per_sec = cfg.batch_size * cfg.grad_accum_steps * cfg.block_size * cfg.log_interval / max(dt, 1e-9)
            step_loss = loss.item() * cfg.grad_accum_steps
            print(f"iter {iter_num}: loss {step_loss:.4f}, "
                  f"{dt * 1000 / max(cfg.log_interval, 1):.0f}ms/iter avg, {tok_per_sec:,.0f} tok/s")
            log_metrics(wandb_run, tb_writer, iter_num,
                        {"train/loss_step": step_loss, "perf/tokens_per_sec": tok_per_sec})

        iter_num += 1

    if wandb_run:
        wandb_run.finish()
    if tb_writer:
        tb_writer.close()


if __name__ == "__main__":
    main()

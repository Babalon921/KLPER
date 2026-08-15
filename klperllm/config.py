import ast
from dataclasses import dataclass, fields
from pathlib import Path

import yaml


@dataclass
class TrainConfig:
    # I/O
    out_dir: str = "out"
    data_dir: str = "data/fineweb_edu"
    eval_interval: int = 250
    log_interval: int = 10
    eval_iters: int = 100
    always_save_checkpoint: bool = False
    init_from: str = "scratch"  # 'scratch' or 'resume'

    # wandb logging (off by default)
    wandb_log: bool = False
    wandb_project: str = "klper-llm"
    wandb_run_name: str = "run"


    tensorboard_log: bool = False
    tensorboard_dir: str = "" 

    # data
    batch_size: int = 32          
    block_size: int = 1024
    grad_accum_steps: int = 8

    # model
    n_layer: int = 8
    n_head: int = 8
    n_embd: int = 512
    dropout: float = 0.0
    bias: bool = False
    vocab_size: int = 50304

    # optimizer
    learning_rate: float = 6e-4
    max_iters: int = 6000
    weight_decay: float = 1e-1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0

    # LR schedule
    decay_lr: bool = True
    warmup_iters: int = 200
    lr_decay_iters: int = 6000
    min_lr: float = 6e-5

    # system
    device: str = "cuda"
    dtype: str = "bfloat16"
    compile: bool = True
    seed: int = 1337

    # sampling preview during training
    sample_every: int = 500
    sample_prompt: str = "\n"
    sample_max_new_tokens: int = 200


def load_config(yaml_path: str | None, overrides: list[str]) -> TrainConfig:
    cfg_dict = {}
    if yaml_path:
        with open(yaml_path) as f:
            cfg_dict.update(yaml.safe_load(f) or {})

    valid_keys = {f.name for f in fields(TrainConfig)}
    for key in cfg_dict:
        if key not in valid_keys:
            raise ValueError(f"unknown config key '{key}' in {yaml_path}")

    for override in overrides:
        if "=" not in override:
            raise ValueError(f"bad override '{override}', expected key=value")
        key, raw_value = override.split("=", 1)
        key = key.lstrip("-")
        if key not in valid_keys:
            raise ValueError(f"unknown config key '{key}' in override '{override}'")
        try:
            value = ast.literal_eval(raw_value)
        except (ValueError, SyntaxError):
            value = raw_value  # plain string
        cfg_dict[key] = value

    return TrainConfig(**cfg_dict)


def ensure_out_dir(cfg: TrainConfig) -> Path:
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir

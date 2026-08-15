# klper-llm
| Config | Params | Hardware | Effective batch | Tokens seen | Wall clock (rough) |
|---|---|---|---|---|---|
| `configs/fineweb_edu_a10_nano.yaml` | ~51M | A10 (24GB) | 262K tok/step | ~2B | a few hours |
| `configs/fineweb_edu_a100_small.yaml` | ~123M (GPT-2-small scale) | A100 (40/80GB) | 524K tok/step | ~4B | a few hours |
| `configs/fineweb_edu_h100_detailed.yaml` | ~303M (GPT-2-medium-ish) | H100 (80GB) | 524K tok/step | ~6.3B | ~6-8 hours |
| `configs/pile_a10_nano.yaml` | ~51M | A10 (24GB) | 262K tok/step | ~2B | a few hours |
| `configs/pile_h100_detailed.yaml` | ~303M (GPT-2-medium-ish) | H100 (80GB) | 524K tok/step | ~6.3B | ~6-8 hours |
| `configs/debug_shakespeare_char.yaml` | ~0.8M | CPU | — | tiny | ~1 minute |


## Quickstart

```bash
pip install -r requirements.txt

# 1. Smoke Test
python data/prepare_shakespeare_char.py
python train.py --config configs/debug_shakespeare_char.yaml
python sample.py --ckpt out/debug_shakespeare_char/ckpt.pt --prompt "ROMEO:"

python data/prepare_fineweb_edu.py --max-tokens 2_000_000_000 --out-dir data/fineweb_edu

# 2. Train
python train.py --config configs/fineweb_edu_a10_nano.yaml

# 3. Generate
python sample.py --ckpt out/fineweb_edu_a10_nano/ckpt.pt --prompt "The history of"
```

Using The Pile instead:

```bash
python data/prepare_pile.py --max-tokens 2_000_000_000 --out-dir data/pile
python train.py --config configs/pile_a10_nano.yaml
```

### Monitoring a run

```bash
python train.py --config configs/fineweb_edu_a10_nano.yaml --tensorboard_log=True
tensorboard --logdir out/fineweb_edu_a10_nano/tensorboard
```

## Requirements

Python 3.10+, PyTorch 2.2+ with CUDA for GPU training (see
`requirements.txt`).

## License

MIT — see [LICENSE](LICENSE).

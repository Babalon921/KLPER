FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-devel AS base

ENV TORCH_CUDA_ARCH_LIST="9.0" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/workspace/.cache/huggingface \
    TOKENIZERS_PARALLELISM=false

WORKDIR /workspace


FROM base AS dependencies
COPY requirements.txt .
RUN grep -v -i '^torch' requirements.txt > /tmp/reqs.txt && \
    pip install --no-cache-dir -r /tmp/reqs.txt && \
    pip install --no-cache-dir wandb tensorboard && \
    rm /tmp/reqs.txt && \
    python -c "import sysconfig; print(sysconfig.get_path('purelib'))" > /tmp/sitepkg.path

FROM base AS runtime
COPY --from=dependencies /tmp/sitepkg.path /tmp/sitepkg.path
RUN --mount=type=bind,from=dependencies,source=/usr/local/lib,target=/tmp/deplib \
    SRC=$(python -c "import sysconfig; print(sysconfig.get_path('purelib'))") && \
    cp -r "/tmp/deplib/$(basename $(dirname $SRC))/$(basename $SRC)" "$SRC"

COPY klperllm/ ./klperllm/
COPY configs/ ./configs/
COPY data/prepare_*.py ./data/
COPY train.py sample.py ./

RUN mkdir -p /workspace/data /workspace/out

VOLUME ["/workspace/data", "/workspace/out"]

ENV DATA_DIR=data/fineweb_edu_6b \
    PREPARE_MAX_TOKENS=6000000000 \
    CONFIG=configs/fineweb_edu_h100_detailed.yaml

CMD python data/prepare_fineweb_edu.py --out-dir "$DATA_DIR" --max-tokens "$PREPARE_MAX_TOKENS" \
    && python train.py --config "$CONFIG"

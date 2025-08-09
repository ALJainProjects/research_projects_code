# ------------------------------------------------------------------------------
# Base image: CUDA 12.1 + cuDNN 8 (runtime)
# Ubuntu 22.04 (jammy)
# ------------------------------------------------------------------------------
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

# ------------------------------------------------------------------------------
# Environment
# ------------------------------------------------------------------------------
ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    # HF caches (you can mount these as volumes to persist model downloads)
    HF_HOME=/workspace/.cache/huggingface \
    HF_DATASETS_CACHE=/workspace/.cache/huggingface/datasets \
    TRANSFORMERS_CACHE=/workspace/.cache/huggingface/transformers \
    WANDB__SERVICE_WAIT=300

# ------------------------------------------------------------------------------
# System packages
# ------------------------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv \
    git git-lfs \
    build-essential \
    tini \
    curl ca-certificates \
    # some common libs frequently needed by plotting / audio / cv deps
    libsndfile1 libglib2.0-0 \
 && git lfs install \
 && ln -sf /usr/bin/python3 /usr/bin/python \
 && python -m pip install --upgrade pip wheel \
 && rm -rf /var/lib/apt/lists/*

# ------------------------------------------------------------------------------
# Workdir & user
# ------------------------------------------------------------------------------
WORKDIR /workspace
# Create non-root user to run the app (optional but recommended)
ARG USERNAME=appuser
ARG USER_UID=1000
ARG USER_GID=1000
RUN groupadd --gid ${USER_GID} ${USERNAME} \
 && useradd --uid ${USER_UID} --gid ${USER_GID} -m ${USERNAME} \
 && chown -R ${USERNAME}:${USERNAME} /workspace
USER ${USERNAME}

# Make cache dirs (owned by non-root)
RUN mkdir -p ${HF_HOME} ${HF_DATASETS_CACHE} ${TRANSFORMERS_CACHE}

# ------------------------------------------------------------------------------
# Dependency install strategy
# 1) Install CUDA-matched PyTorch stack first (stable cache layer).
# 2) Copy minimal project metadata + source to build the editable package.
# 3) Install project with extras.
# ------------------------------------------------------------------------------
# 1) CUDA 12.1 wheel index for PyTorch
RUN python -m pip install --index-url https://download.pytorch.org/whl/cu121 \
    torch torchvision torchaudio

# 2) Copy project metadata and source first to leverage Docker layer caching
#    (scripts/configs change often; keep them for later to avoid busting this layer)
COPY --chown=${USERNAME}:${USERNAME} pyproject.toml ./pyproject.toml
COPY --chown=${USERNAME}:${USERNAME} src ./src

# 3) Install your package in editable mode with all extras
#    (If your pyproject defines extras 'all', this will pull everything used by CT-GSSN & CEX-TSLM)
RUN python -m pip install -e ".[all]"

# ------------------------------------------------------------------------------
# Copy the rest (changes here won't invalidate dependency cache)
# ------------------------------------------------------------------------------
COPY --chown=${USERNAME}:${USERNAME} scripts ./scripts
COPY --chown=${USERNAME}:${USERNAME} configs ./configs

# ------------------------------------------------------------------------------
# Default data & runs locations (mount these if you want persistence)
# ------------------------------------------------------------------------------
VOLUME ["/workspace/data", "/workspace/runs", "/workspace/checkpoints", "/workspace/.cache"]

# ------------------------------------------------------------------------------
# Healthcheck (basic python check; adjust if needed)
# ------------------------------------------------------------------------------
HEALTHCHECK --interval=2m --timeout=10s --start-period=30s --retries=3 \
 CMD python -c "import torch,sys; print(torch.cuda.is_available()); sys.exit(0)"

# ------------------------------------------------------------------------------
# Entrypoint & default command
# ------------------------------------------------------------------------------
ENTRYPOINT ["/usr/bin/tini", "--"]

# Default: train CT-GSSN on synthetic
CMD ["python", "scripts/train_ct_gssn.py", "--config", "configs/ct_gssn/synthetic.yaml"]

# ------------------------------------------------------------------------------
# Handy alternatives (uncomment at build time or override at runtime):
#   docker run --gpus all -v $PWD:/workspace <img> \
#       python scripts/train_ct_gssn.py --config configs/ct_gssn/metr_la.yaml
#   docker run --gpus all -v $PWD:/workspace <img> \
#       python scripts/train_cex_tslm.py --config configs/cex_tslm/time_mmd.yaml
#   docker run --gpus all -v $PWD:/workspace <img> \
#       python scripts/preprocess_templates.py --dataset all
# ------------------------------------------------------------------------------

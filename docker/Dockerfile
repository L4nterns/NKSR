# syntax=docker/dockerfile:1.7
ARG BASE_IMAGE=nvidia/cuda:12.4.1-devel-ubuntu22.04
FROM ${BASE_IMAGE}

ARG APT_MIRROR=mirrors.tuna.tsinghua.edu.cn
ARG PIP_INDEX_URL=https://mirrors.aliyun.com/pypi/simple
ARG PIP_TRUSTED_HOST=mirrors.aliyun.com
ARG UV_INSTALL_URL=https://astral.sh/uv/install.sh
ARG TORCH_CUDA_ARCH_LIST=8.6
ARG MAX_JOBS=4

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    CUDA_HOME=/usr/local/cuda \
    FORCE_CUDA=1 \
    MAX_JOBS=${MAX_JOBS} \
    TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST} \
    PATH=/usr/local/cuda/bin:${PATH} \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH} \
    PIP_INDEX_URL=${PIP_INDEX_URL} \
    PIP_TRUSTED_HOST=${PIP_TRUSTED_HOST} \
    UV_LINK_MODE=copy

RUN if [ -f /etc/apt/sources.list ]; then \
      sed -i "s@http://archive.ubuntu.com/ubuntu/@https://${APT_MIRROR}/ubuntu/@g; s@http://security.ubuntu.com/ubuntu/@https://${APT_MIRROR}/ubuntu/@g" /etc/apt/sources.list; \
    fi

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt/lists,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
      build-essential \
      ca-certificates \
      curl \
      git \
      libgl1 \
      libglib2.0-0 \
      libgomp1 \
      ninja-build \
      python3.10 \
      python3.10-dev \
      python3.10-venv \
      python3-pip \
    && rm -rf /var/lib/apt/lists/*

RUN curl -LsSf "${UV_INSTALL_URL}" | sh \
    && cp /root/.local/bin/uv /usr/local/bin/uv \
    && chmod 755 /usr/local/bin/uv

WORKDIR /workspace/nksr

COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --group http --no-install-project

RUN uv pip install --no-cache --no-binary torch-scatter --no-build-isolation torch-scatter

COPY package/pyproject.toml package/setup.py ./package/
COPY package/csrc ./package/csrc
COPY package/nksr/__init__.py ./package/nksr/
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=cache,target=/workspace/nksr/package/external \
    uv pip install --no-build-isolation -e ./package

COPY . .

RUN uv run python -c "import nksr, torch, torch_scatter; print('torch', torch.__version__, 'torch_scatter', torch_scatter.__version__, 'nksr', nksr.__version__)"

RUN groupadd --system nksr \
    && useradd --system --gid nksr --home-dir /home/nksr --create-home --shell /usr/sbin/nologin nksr \
    && mkdir -p /home/nksr/.cache/torch \
    && chown -R nksr:nksr /workspace/nksr /home/nksr

ENV TORCH_HOME=/home/nksr/.cache/torch
USER nksr

CMD ["uv", "run", "--group", "http", "python", "scripts/http_infer_pcd.py", "--host", "0.0.0.0", "--port", "8012"]

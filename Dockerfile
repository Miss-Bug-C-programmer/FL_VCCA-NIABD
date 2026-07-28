# syntax=docker/dockerfile:1

# The default image follows the ARM64/CoreX deployment environment supplied
# with this project. Override BASE_IMAGE when building for another platform.
ARG BASE_IMAGE=swr.cn-north-4.myhuaweicloud.com/ddn-k8s/docker.io/ubuntu:24.04-linuxarm64
FROM ${BASE_IMAGE}

ARG DEBIAN_FRONTEND=noninteractive
ARG PIP_INSTALL_MODE=auto

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    COREX_VISIBLE_DEVICES=0 \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    PYTHONPATH=/app/code \
    PATH=/venv/bin:${PATH}

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        python3 \
        python3-dev \
        python3-pip \
        python3-venv \
        libopenblas-dev \
        liblapack-dev \
        libgomp1 \
        libglib2.0-0 \
        tini \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m venv /venv

WORKDIR /app/code

# .dockerignore excludes datasets, experiment outputs, caches and Git metadata.
# An optional packages/ directory may contain offline ARM64/CoreX wheels.
COPY . /app/code/

RUN set -eu; \
    case "${PIP_INSTALL_MODE}" in \
        auto|online|offline) ;; \
        *) echo "PIP_INSTALL_MODE must be auto, online, or offline" >&2; exit 2 ;; \
    esac; \
    if [ "${PIP_INSTALL_MODE}" = "offline" ] \
        || { [ "${PIP_INSTALL_MODE}" = "auto" ] \
             && [ -d /app/code/packages ] \
             && find /app/code/packages -maxdepth 1 -name '*.whl' \
                -print -quit | grep -q .; }; then \
        if ! [ -d /app/code/packages ] \
            || ! find /app/code/packages -maxdepth 1 -name '*.whl' \
                -print -quit | grep -q .; then \
            echo "Offline installation requested, but packages/*.whl is empty." >&2; \
            exit 3; \
        fi; \
        pip install --no-index --find-links=/app/code/packages \
            /app/code/packages/*.whl; \
    else \
        pip install -r /app/code/requirements.txt; \
    fi; \
    python -c "import torch, torchvision; print('PyTorch:', torch.__version__); print('torchvision:', torchvision.__version__)"

RUN mkdir -p /data /outputs

VOLUME ["/data", "/outputs"]

# tini is important because process-semi-async launches persistent Client
# children and must forward signals and reap processes during shutdown.
ENTRYPOINT ["/usr/bin/tini", "--", "/venv/bin/python", "/app/code/experiment_runner.py"]
CMD ["--help"]

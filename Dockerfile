FROM ghcr.io/astral-sh/uv:0.12.9 AS uv

FROM nvidia/cuda:12.8.1-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    UV_LINK_MODE=copy \
    UV_PYTHON=python3 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    git python3 python3-dev ca-certificates libglib2.0-0 libgl1 \
    && rm -rf /var/lib/apt/lists/*
COPY --from=uv /uv /uvx /usr/local/bin/
WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
COPY visionbench ./visionbench
COPY tasks ./tasks
COPY assets/README.md ./assets/README.md
COPY raw/README.md ./raw/README.md
COPY raw/white_background ./raw/white_background
COPY configs ./configs
COPY notebooks ./notebooks
RUN uv sync --frozen --no-cache
ENV PATH="/app/.venv/bin:$PATH"
ENTRYPOINT ["visionbench"]

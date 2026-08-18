# Minimal sandbox image with common dev tools for AI coding agents.
#
# Build:
#   docker build -t agent-sandbox:latest -f docker/sandbox.Dockerfile .
#
# Or let the agent auto-build on first use when AGENT_SANDBOX_BACKEND=docker.

FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8

# Layer 1: shell + network + build essentials
RUN apt-get update && apt-get install -y --no-install-recommends \
    bash coreutils curl wget git build-essential ca-certificates \
    unzip jq \
    && rm -rf /var/lib/apt/lists/*

# Layer 2: Python 3
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv python3-dev \
    && rm -rf /var/lib/apt/lists/* \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3 1 \
    && update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

# Layer 3: Node.js 22 LTS
RUN curl -fsSL https://deb.nodesource.com/setup_22.x | bash - \
    && apt-get install -y nodejs \
    && rm -rf /var/lib/apt/lists/*

# Layer 4: Rust
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

# Non-root user
RUN useradd -m -s /bin/bash sandbox \
    && mkdir -p /workspace \
    && chown sandbox:sandbox /workspace

WORKDIR /workspace
USER sandbox

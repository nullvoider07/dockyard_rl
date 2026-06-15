# syntax=docker/dockerfile:1
# nullvoider/ubuntu-24.04 — AI Agent SWE RL Environment (SGLang inference variant)
# Identical to ubuntu-swe-v2.dockerfile except the rollout inference backend is
# SGLang (pinned v0.5.12.post1) instead of vLLM. Build as a separate image tag
# (e.g. ubuntu-swe-sglang); select per the policy.generation.backend config.
ARG BASE_IMAGE=nvcr.io/nvidia/cuda-dl-base:26.04-cuda13.2-devel-ubuntu24.04
FROM ${BASE_IMAGE}

# === Build-time ===
ARG CONTAINER_ID
ARG MAX_JOBS=8
ENV DEBIAN_FRONTEND=noninteractive \
    MAX_JOBS=${MAX_JOBS}

# Set pipefail so any command in a pipe failing causes the RUN to fail.
# Without this, `curl ... | tar xz` silently succeeds even if curl fails.
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# === Container identity ===
ENV container=docker \
    SYSTEMD_LOG_LEVEL=info

# === Locale ===
ENV LANG=en_US.UTF-8 \
    LC_ALL=en_US.UTF-8

# ============================================================
# 1. USER SETUP
# ============================================================
RUN getent group sudo   >/dev/null || groupadd -r sudo   && \
    useradd -m -u 1001 -s /bin/bash -G sudo rl && \
    passwd -d rl && \
    mkdir -p /workspace /tmp /run/dbus /run/user/1001 && \
    chown rl:rl /workspace /tmp /run/user/1001 && \
    chmod 700 /run/user/1001

# ============================================================
# 2. CORE PACKAGES
# ============================================================
RUN apt-get update && apt-get install -y --no-install-recommends \
    # Shell & base utils
    zsh curl git wget ca-certificates sudo nano \
    aria2 tree lsof strace gnupg apt-transport-https \
    software-properties-common bash-completion man-db \
    mc ranger fzf bat eza duf jq xdg-utils \
    # Network utils
    iputils-ping inetutils-traceroute net-tools iproute2 \
    # System utils
    rsync tmux procps util-linux psmisc unzip openssl logrotate less \
    # Multi-node Ray cluster (SSH between head and worker nodes)
    openssh-server \
    # deep_ep RDMA dependency (expert-parallel all-to-all)
    libibverbs-dev \
    # DBus & Supervisor
    dbus dbus-x11 supervisor \
    # Locale & fonts
    fontconfig locales fonts-ubuntu \
    # Minimal X libs — required by VS Code CLI even headless
    libx11-6 libxext6 libxrender1 libxrandr2 \
    libxtst6 libxcb1 libxcomposite1 libxdamage1 \
    vim \
    # Media utilities (potential task requirements)
    ffmpeg \
    && locale-gen en_US.UTF-8 \
    && update-locale LANG=en_US.UTF-8 LC_ALL=en_US.UTF-8 \
    && fc-cache -f -v \
    && apt-get clean && rm -rf /var/lib/apt/lists/* /var/tmp/*

# === Nsight Systems CLI — generic CUDA profiler ===
# The cuda-dl-base image already carries NVIDIA's apt repos, so no repo
# addition is needed. If nsight-systems-cli is not found on the first
# attempt, the fallback adds the devtools repo explicitly before retrying.
RUN apt-get update && \
    apt-get install -y --no-install-recommends nsight-systems-cli || { \
        DISTRIB_RELEASE=$(. /etc/lsb-release && echo "$DISTRIB_RELEASE" | tr -d '.') && \
        ARCH=$(dpkg --print-architecture) && \
        echo "deb http://developer.download.nvidia.com/devtools/repos/ubuntu${DISTRIB_RELEASE}/${ARCH} /" \
            > /etc/apt/sources.list.d/nvidia-devtools.list && \
        apt-get update && \
        apt-get install -y --no-install-recommends nsight-systems-cli; \
    } && \
    apt-get install -y --only-upgrade gnupg && \
    rm -rf /var/lib/apt/lists/*

# ============================================================
# 3. DOCKER-IN-DOCKER
# ============================================================
RUN curl -fsSL https://download.docker.com/linux/ubuntu/gpg \
        | gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg \
    && echo "deb [arch=amd64 signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] \
        https://download.docker.com/linux/ubuntu noble stable" \
        > /etc/apt/sources.list.d/docker.list \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        docker-ce \
        docker-ce-cli \
        containerd.io \
        docker-buildx-plugin \
        docker-compose-plugin \
    && rm -rf /var/lib/apt/lists/*

RUN getent group docker >/dev/null || groupadd -r docker \
    && usermod -aG docker rl

RUN mkdir -p /etc/docker && cat > /etc/docker/daemon.json <<'EOF'
{
  "storage-driver": "overlay2",
  "log-driver": "json-file",
  "log-opts": {
    "max-size": "10m",
    "max-file": "3"
  }
}
EOF

# ============================================================
# 4. PYTHON 3.13.13
# ============================================================
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential zlib1g-dev libncurses5-dev libgdbm-dev \
        libnss3-dev libssl-dev libreadline-dev libffi-dev \
        libsqlite3-dev libbz2-dev liblzma-dev uuid-dev \
    && curl -L https://www.python.org/ftp/python/3.13.13/Python-3.13.13.tgz | tar xz -C /tmp \
    && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /tmp/Python-3.13.13
RUN ./configure --enable-optimizations --with-ensurepip=install \
    && make -j"${MAX_JOBS}" \
    && make altinstall \
    && rm -rf /tmp/Python-3.13.13

WORKDIR /

RUN python3.13 -m pip install --no-cache-dir --upgrade pip==26.0.1

RUN curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR="/usr/local/bin" sh

# Python 3.13 is the ML infrastructure default. UV_PYTHON (set below) pins uv's
# --system installs to it, and the python3 alternative points at it, so the
# torch / SGLang / Ray / JAX stack, the build-time import checks, and the runtime
# all resolve to one interpreter. 3.13 is the ceiling: sglang 0.5.12.post1 ships
# wheels only through cp313 (no cp314).
# Python 3.12 (the distro python) stays installed and reachable as python3.12.

# Install system pip and venv to restore ensurepip capabilities
RUN apt-get update && apt-get install -y --no-install-recommends python3-pip python3-venv \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

RUN python3.12 -m pip install --no-cache-dir --upgrade --ignore-installed pip --break-system-packages && \
    update-alternatives --install /usr/bin/python  python  /usr/local/bin/python3.13   200 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/local/bin/python3.13 200 \
    && update-alternatives --install /usr/bin/pip    pip    /usr/local/bin/pip3.13      200 \
    && update-alternatives --install /usr/bin/pip3   pip3   /usr/local/bin/pip3.13      200 \
    && update-alternatives --set python3 /usr/local/bin/python3.13

RUN python3.13 -m ensurepip --upgrade

# UV_HTTP_TIMEOUT: raise uv's 30s per-request ceiling. The cu130 torch stack
# pulls cuda-toolkit and the nvidia-*-cu13 wheels from pypi.nvidia.com, which is
# slow under the parallel multi-GB wheel fetch and otherwise times out.
ENV UV_SYSTEM_PYTHON=1 \
    UV_PYTHON=python3.13 \
    UV_LINK_MODE=copy \
    UV_HTTP_TIMEOUT=600 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# ============================================================
# 4b. RAY & CUDA PERFORMANCE TUNING (generic for RL agents)
# ============================================================

# Ray orchestration
# RAY_memory_monitor_refresh_ms=0: disables Ray's memory monitor — it triggers
# false OOM kills during CUDA kernel launches which temporarily inflate RSS.
ENV RAY_USAGE_STATS_ENABLED=0 \
    RAY_ENABLE_UV_RUN_RUNTIME_ENV=0 \
    RAY_memory_monitor_refresh_ms=0

# CUDA 13.2 performance
# Targets H100 (sm_90) + B200 (sm_100) + GB300/Blackwell-Ultra (sm_103).
# NOTE: with the PyPI SGLang install below, native SASS coverage in the
# prebuilt sgl-kernel / flashinfer wheels is fixed at their build time — this
# list governs torch-side/Inductor JIT and any from-source extension, and
# documents the intended hardware. Native GB300 kernels in sgl-kernel require
# the from-git build path, not the PyPI wheels.
ENV TORCH_CUDA_ARCH_LIST="9.0 10.0 10.3" \
    FLASHINFER_CUDA_ARCH_LIST="9.0a 10.0a 10.3a"

# CUDA 13 standard headers (required when building any CUDA extensions)
ENV CPLUS_INCLUDE_PATH=/usr/local/cuda/include/cccl

# cuDNN: prefer pip-installed nvidia-cudnn-cu12 over system cuDNN to avoid
# version mismatch crashes in Transformer Engine. Points at system site-packages
# (UV_SYSTEM_PYTHON=1). Override per-venv if using isolated venvs.
ENV CUDNN_HOME=/usr/local/lib/python3.13/site-packages/nvidia/cudnn
ENV LD_LIBRARY_PATH="/usr/local/lib/python3.13/site-packages/nvidia/cudnn/lib:/usr/local/cuda/lib64:${LD_LIBRARY_PATH}"

# PyTorch memory & compilation
# backend:cudaMallocAsync — uses CUDA 11.2+ async memory pool at the driver level.
# This is the PyTorch-side complement to XLA_PYTHON_CLIENT_ALLOCATOR=platform:
# both frameworks share the same CUDA memory pool, negotiating allocations
# at the CUDA driver level rather than competing in userspace.
# max_split_size_mb: prevents OOM-after-N-iterations on models without FA2.
ENV PYTORCH_CUDA_ALLOC_CONF="backend:cudaMallocAsync,max_split_size_mb:512"

# Persist torch.compile kernel cache across container restarts.
ENV TORCHINDUCTOR_CACHE_DIR=/workspace/.cache/torch_inductor

# Python process behaviour
# Prevents CPU thread oversubscription: each Ray worker gets 1 OMP thread;
# GPU parallelism is handled by CUDA, not OpenMP.
ENV OMP_NUM_THREADS=1

# HuggingFace tokenizers uses Rust multiprocessing that deadlocks with Ray's
# fork-based worker spawning. This disables it globally.
ENV TOKENIZERS_PARALLELISM=false

# Limits glibc arena count — prevents RSS ballooning across many Ray workers.
ENV MALLOC_ARENA_MAX=1

# SGLang runtime
# The SGLang server runs as its own subprocess launched by the generation
# worker (HTTP on a free port), so no fork/multiproc tuning is needed here.
# Ray owns cluster-wide log routing; the server's stdout is captured by the
# launching actor. No SGLang-specific env is required at image build time —
# server args (tp_size, mem fraction, etc.) are passed by the worker at launch.

# HuggingFace cache
# Point model/tokenizer cache at the persistent workspace mount so weights
# are not re-downloaded on every container restart.
ENV HF_HOME=/workspace/.cache/huggingface \
    TRANSFORMERS_CACHE=/workspace/.cache/huggingface

# NCCL multi-node
# Override at runtime with the actual interface name in your cluster
# (e.g. -e NCCL_SOCKET_IFNAME=eth0). Default avoids loopback-only binding.
ENV NCCL_SOCKET_IFNAME=^lo,docker

# JAX / XLA
# Use CUDA's native allocator instead of XLA's internal BFC pool.
# XLA remains fully active — JIT, kernel fusion, HLO optimization all intact.
# The difference: CUDA native allocator returns memory to CUDA after each op,
# allowing JAX, PyTorch, and Ray workers to share GPU memory without any
# framework locking out the others.
ENV XLA_PYTHON_CLIENT_ALLOCATOR=platform

# Point XLA at the CUDA installation so it finds cuBLAS, cuDNN, and NCCL
# without searching. Prevents silent fallback to CPU on path resolution failures.
ENV XLA_FLAGS="--xla_gpu_cuda_data_dir=/usr/local/cuda"

# Enable 64-bit precision globally. JAX defaults to 32-bit for performance;
# 64-bit is needed for numerically stable value functions in RL.
# Override per-process with jax.config.update("jax_enable_x64", False) if needed.
ENV JAX_ENABLE_X64=1

# Persist XLA JIT compilation cache across container restarts.
# Without this every restart recompiles all jit-decorated functions from scratch
# (can take minutes on large models).
ENV XLA_COMPILATION_CACHE_DIR=/workspace/.cache/xla

# Consistent device ordering for environments mixing JAX and PyTorch.
# Both frameworks enumerate GPUs by PCI bus ID when this is set.
ENV CUDA_DEVICE_ORDER=PCI_BUS_ID

# ============================================================
# 5. PYTHON DEV, EVAL & LINTER TOOLS
# ============================================================
RUN --mount=type=cache,target=/root/.cache/uv uv pip install --reinstall \
    # Task Executor API
    flask waitress \
    # gRPC / agent controller compat
    grpcio grpcio-tools click PyJWT psutil pyyaml \
    # Test frameworks
    pytest pytest-xdist \
    # Python linters & type checkers
    ruff mypy flake8 pylint \
    # Distributed RL orchestration
    "ray[default]"

# Remove stale aiohttp bundled inside Ray's runtime env agent (CVE GHSA-mqqc-3gqh-h2x8).
RUN find /root/.cache/uv -type d \
        -path "*/ray/_private/runtime_env/agent/thirdparty_files/aiohttp*" \
        -exec rm -rf {} + 2>/dev/null || true && \
    find /root/.cache/pip -type d -name "aiohttp*" \
        -exec rm -rf {} + 2>/dev/null || true

# Patch Ray's nsight plugin to use the actual Python executable instead of the
# hardcoded "python" string. Fixes nsys profiling of Ray workers when using
# custom Python paths or venvs. Applied once at build time — transparent to
# all training scripts.
RUN NSIGHT_PY=$(python -c "import ray, os; print(os.path.join(os.path.dirname(ray.__file__), '_private/runtime_env/nsight.py'))") && \
    if [ -f "$NSIGHT_PY" ]; then \
        sed -i \
            's|context\.py_executable = " "\.join(self\.nsight_cmd) + " python"|context.py_executable = " ".join(self.nsight_cmd) + f" {context.py_executable}"|g' \
            "$NSIGHT_PY" && \
        echo "Ray nsight patch applied: $NSIGHT_PY"; \
    else \
        echo "Ray nsight plugin not found — skipping patch"; \
    fi

# ============================================================
# 5b. JAX + ECOSYSTEM
# ============================================================
# jax[cuda12] installs jaxlib with CUDA 12 wheels.
# CUDA 13.2 (this image's base) is forward-compatible with CUDA 12 runtime
# APIs — this is the documented install path for CUDA 13 from the JAX team.
RUN --mount=type=cache,target=/root/.cache/uv uv pip install --reinstall \
    "jax[cuda12]" \
    # Neural network library — JAX equivalent of PyTorch nn.Module
    flax \
    # Optimizers (SGD, Adam, etc.) — works with both JAX and Flax
    optax \
    # Checkpointing — saves and restores JAX pytrees to disk
    orbax-checkpoint \
    # RL-specific algorithms implemented in JAX (value functions, policy gradients)
    rlax \
    # Debugging and testing utilities for JAX (shape checking, tree assertions)
    chex \
    # Einsum notation for tensor ops — shared idiom across JAX and PyTorch workflows
    einops

# Verify JAX can see CUDA at build time.
# This is a CPU-only check (no GPU during docker build) — it confirms the
# package installed correctly and XLA backend is configured, not that a GPU
# is present. GPU visibility is confirmed at runtime in L2 tests.
RUN python3 <<EOF
import jax
import jax.numpy as jnp
print('JAX version:', jax.__version__)
print('JAX default backend:', jax.default_backend())
# Execute a trivial traced+compiled op so jaxlib/XLA breakage (not just a
# bare import) is caught at build time.
x = jnp.ones(8, dtype=jnp.float32)
assert float((x * 2.0).sum()) == 16.0
print('JAX compute smoke (jnp.ones): OK')
EOF

# ============================================================
# 5c. PyTorch + SGLang (rollout inference backend)
# ============================================================
# Installed into the ML default python 3.13 (uv --system, UV_PYTHON; see §4).
# SGLang v0.5.12.post1's core deps pin torch==2.11.0 and pull
# flashinfer-python/-cubin==0.6.11.post1, sglang-kernel==0.4.2.post2,
# transformers==5.6.0, xgrammar==0.2.0, cuda-python>=13.0,
# nvidia-cutlass-dsl[cu13]==4.5.1. Install torch first from the cu130 index so
# the CUDA-13 build is used (matches the cuda13.2 base).
RUN --mount=type=cache,target=/root/.cache/uv uv pip install --reinstall \
    torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0 \
    --index-url https://download.pytorch.org/whl/cu130

# Rust toolchain — required before the sglang install below: sglang's transitive
# dep outlines-core==0.1.26 has no cp313 wheel and builds from a Rust sdist, so
# cargo/rustc must be on PATH. (The vLLM image keeps Rust in §6; only the SGLang
# image needs it this early.)
ENV RUSTUP_HOME=/usr/local/rustup \
    CARGO_HOME=/usr/local/cargo \
    PATH=/usr/local/cargo/bin:$PATH

RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
        | sh -s -- -y --no-modify-path --default-toolchain stable \
    && . "$CARGO_HOME/env" \
    && rustup default stable \
    && chown -R root:1001 "$RUSTUP_HOME" "$CARGO_HOME" \
    && chmod -R 775 "$RUSTUP_HOME" "$CARGO_HOME" \
    && rustup --version && cargo --version && rustc --version

# No --reinstall here: torch==2.11.0 is already satisfied by the cu130 build
# above (2.11.0+cu130 matches the ==2.11.0 pin), and --reinstall would cascade
# to pulling a default-index torch and clobber it. SGLang + its remaining deps
# resolve against the installed torch.
# --prerelease=allow: sglang 0.5.12.post1 requires flash-attn-4>=4.0.0b9, which
# only publishes betas (its wheel is py3-none-any, so no source build). uv
# excludes pre-releases by default; this lets the resolver pick the beta.
# kernels>=0.12.0,<0.13: sglang requires bare `kernels` (no ceiling), so uv would
# otherwise pick the newest (0.15.x), whose LayerRepository requires an explicit
# version. transformers==5.6.0's hub_kernels integration constructs
# LayerRepository without one, so kernels must stay in the 0.12.x range that
# transformers 5.6.0 pins for its `kernels` extra.
RUN --mount=type=cache,target=/root/.cache/uv uv pip install --prerelease=allow \
    "sglang==0.5.12.post1" "kernels>=0.12.0,<0.13"

# CPU-only import check — confirms the build succeeded and SGLang's
# extension modules (sgl-kernel) are present. Actual GPU visibility is
# confirmed at runtime, same as the JAX check above.
RUN python3 <<EOF
import sglang
print('SGLang version:', sglang.__version__)
print('SGLang import: OK')
EOF

# ============================================================
# 7. C / C++
# ============================================================
RUN apt-get update && apt-get install -y --no-install-recommends \
    clang lldb gdb ninja-build valgrind cppcheck \
    && update-alternatives --install /usr/bin/cc cc /usr/bin/clang 100 \
    && update-alternatives --install /usr/bin/c++ c++ /usr/bin/clang++ 100 \
    && update-alternatives --install /usr/bin/cc cc /usr/bin/gcc 200 \
    && update-alternatives --install /usr/bin/c++ c++ /usr/bin/g++ 200 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# CMake 3.31+ — apt ships 3.28 on Ubuntu 24.04 which is too old for SGLang kernels
RUN CMAKE_VERSION=3.31.1 && \
    ARCH=$(uname -m) && \
    CMAKE_PKG="cmake-${CMAKE_VERSION}-linux-${ARCH}" && \
    curl --retry 3 --retry-delay 2 -fsSL \
        "https://github.com/Kitware/CMake/releases/download/v${CMAKE_VERSION}/${CMAKE_PKG}.tar.gz" \
        -o /tmp/cmake.tar.gz && \
    tar -xzf /tmp/cmake.tar.gz -C /tmp && \
    cp -r "/tmp/${CMAKE_PKG}/bin/"* /usr/local/bin/ && \
    cp -r "/tmp/${CMAKE_PKG}/share/"* /usr/local/share/ && \
    rm -rf /tmp/cmake.tar.gz "/tmp/${CMAKE_PKG}"

# ============================================================
# 8. C# / .NET 10
# ============================================================
RUN wget -q https://packages.microsoft.com/config/ubuntu/24.04/packages-microsoft-prod.deb \
        -O /tmp/packages-microsoft-prod.deb \
    && dpkg -i /tmp/packages-microsoft-prod.deb \
    && rm /tmp/packages-microsoft-prod.deb \
    && apt-get update && apt-get install -y --no-install-recommends dotnet-sdk-10.0 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

ENV DOTNET_NOLOGO=1 \
    DOTNET_CLI_TELEMETRY_OPTOUT=1 \
    PATH="$PATH:/home/rl/.dotnet/tools"

# ============================================================
# 9. JAVA 25
# ============================================================
RUN wget -q https://download.oracle.com/java/25/latest/jdk-25_linux-x64_bin.tar.gz \
        -O /tmp/jdk.tar.gz \
    && mkdir -p /usr/lib/jvm \
    && tar -xzf /tmp/jdk.tar.gz -C /usr/lib/jvm \
    && mv /usr/lib/jvm/jdk-25* /usr/lib/jvm/jdk-25 \
    && rm /tmp/jdk.tar.gz \
    && for tool in java javac jar javap jshell jlink jpackage; do \
           update-alternatives --install /usr/bin/$tool $tool \
               /usr/lib/jvm/jdk-25/bin/$tool 100; \
       done

ENV JAVA_HOME=/usr/lib/jvm/jdk-25 \
    PATH=/usr/lib/jvm/jdk-25/bin:$PATH

# ============================================================
# 10. SCALA (latest)
# ============================================================
RUN SCALA_VERSION=$(wget -qO- "https://api.github.com/repos/scala/scala3/releases/latest" \
        | grep '"tag_name"' | sed -E 's/.*"([^"]+)".*/\1/' | sed 's/^v//') && \
    [ -n "$SCALA_VERSION" ] || { echo "ERROR: Could not determine Scala version"; exit 1; } && \
    wget -q "https://github.com/scala/scala3/releases/download/${SCALA_VERSION}/scala3-${SCALA_VERSION}.tar.gz" -O /tmp/scala.tar.gz \
    && rm -rf /usr/local/scala* \
    && tar -xzf /tmp/scala.tar.gz -C /usr/local \
    && chown -R root:1001 "/usr/local/scala3-${SCALA_VERSION}" \
    && chmod -R 775 "/usr/local/scala3-${SCALA_VERSION}" \
    && ln -sf "/usr/local/scala3-${SCALA_VERSION}" /usr/local/scala \
    && rm /tmp/scala.tar.gz \
    && /usr/local/scala/bin/scala -version

ENV SCALA_HOME=/usr/local/scala \
    PATH=/usr/local/scala/bin:$PATH

# ============================================================
# 11. GO (latest)
# ============================================================
RUN GO_ARCHIVE=$(set +o pipefail; wget -qO- https://go.dev/dl/ | grep -o 'go[0-9.]*\.linux-amd64\.tar\.gz' | head -1) && \
    wget -q "https://go.dev/dl/${GO_ARCHIVE}" -O /tmp/go.tar.gz \
    && rm -rf /usr/local/go \
    && tar -C /usr/local -xzf /tmp/go.tar.gz \
    && rm /tmp/go.tar.gz
ENV GOROOT=/usr/local/go \
    GOPATH=/usr/local/go-workspace \
    PATH=/usr/local/go/bin:/usr/local/go-workspace/bin:$PATH
RUN mkdir -p "$GOPATH/src" "$GOPATH/bin" "$GOPATH/pkg" \
    && chown -R 1001:1001 "$GOPATH" \
    && chmod -R 755 "$GOPATH"
# Go linters
RUN go install honnef.co/go/tools/cmd/staticcheck@2025.1.1 \
    && chown root:1001 "$GOPATH/bin/staticcheck" \
    && chmod 775 "$GOPATH/bin/staticcheck"

# ============================================================
# 12. NODE.JS (latest) + TypeScript
# ============================================================
RUN NODE_ARCHIVE=$(set +o pipefail; wget -qO- https://nodejs.org/dist/latest/ | grep -o 'node-v[0-9.]*-linux-x64\.tar\.xz' | head -1) && \
    wget -q "https://nodejs.org/dist/latest/${NODE_ARCHIVE}" -O /tmp/node.tar.xz \
    && tar -xJf /tmp/node.tar.xz -C /usr/local --strip-components=1 \
    && rm /tmp/node.tar.xz

ENV NPM_CONFIG_PREFIX=/usr/local/npm-global \
    PATH=/usr/local/npm-global/bin:$PATH
RUN mkdir -p "$NPM_CONFIG_PREFIX" \
    && npm install -g --no-audit --no-fund npm@latest typescript@latest tsx@latest \
    && chown -R 1001:1001 "$NPM_CONFIG_PREFIX" \
    && chmod -R 755 "$NPM_CONFIG_PREFIX"

# ============================================================
# 13. KOTLIN (latest)
# ============================================================
RUN KOTLIN_VERSION=$(wget -qO- https://api.github.com/repos/JetBrains/kotlin/releases/latest | grep '"tag_name"' | sed -E 's/.*"v([^"]+)".*/\1/') && \
    wget -q "https://github.com/JetBrains/kotlin/releases/download/v${KOTLIN_VERSION}/kotlin-compiler-${KOTLIN_VERSION}.zip" -O /tmp/kotlin.zip \
    && rm -rf /usr/local/kotlinc \
    && unzip -q /tmp/kotlin.zip -d /usr/local \
    && rm /tmp/kotlin.zip \
    && chown -R root:1001 /usr/local/kotlinc \
    && chmod -R 775 /usr/local/kotlinc
ENV KOTLIN_HOME=/usr/local/kotlinc \
    PATH=/usr/local/kotlinc/bin:$PATH

# ============================================================
# 14. POWERSHELL
# ============================================================
RUN apt-get update && apt-get install -y --no-install-recommends powershell \
    && rm -rf /var/lib/apt/lists/*

# ============================================================
# 15. SCALE-SAFE DYNAMIC USERNAME
# ============================================================
RUN UNIQUE_SUFFIX="$(head /dev/urandom | tr -dc a-z0-9 | head -c 8)" && \
    UNIQUE_USER="rl-${CONTAINER_ID:-$UNIQUE_SUFFIX}" && \
    echo "Renaming rl -> $UNIQUE_USER (UID 1001)" && \
    usermod -l "$UNIQUE_USER" rl && \
    groupmod -n "$UNIQUE_USER" rl 2>/dev/null || true && \
    usermod -d "/home/$UNIQUE_USER" -m "$UNIQUE_USER" && \
    echo "Scale-safe username: $UNIQUE_USER"

# ============================================================
# 16. USER ENVIRONMENT (.bashrc.d)
# ============================================================
RUN USER_NAME=$(id -un 1001) && \
    USER_HOME=$(eval echo ~$USER_NAME) && \
    mkdir -p "$USER_HOME/.bashrc.d" && \
    { \
        echo 'export RUSTUP_HOME=/usr/local/rustup'; \
        echo 'export CARGO_HOME=/usr/local/cargo'; \
        echo 'export GOROOT=/usr/local/go'; \
        echo 'export GOPATH=/usr/local/go-workspace'; \
        echo 'export JAVA_HOME=/usr/lib/jvm/jdk-25'; \
        echo 'export KOTLIN_HOME=/usr/local/kotlinc'; \
        echo 'export SCALA_HOME=/usr/local/scala'; \
        echo 'export PATH=$PATH:/usr/local/cargo/bin:/usr/local/go/bin:/usr/local/go-workspace/bin'; \
        echo 'export PATH=$PATH:/usr/local/kotlinc/bin:/usr/local/scala/bin'; \
        echo 'export PATH=$PATH:/usr/local/npm-global/bin:$JAVA_HOME/bin:/usr/local/bin'; \
        echo '# python3 / pip  = Python 3.13 — ML infrastructure (PyTorch, SGLang, Ray, JAX)'; \
        echo '# python3.12 / pip3.12 = Python 3.12 — distro python, agent task compat'; \
    } > "$USER_HOME/.bashrc.d/prog-langs.sh" && \
    chmod 644 "$USER_HOME/.bashrc.d/prog-langs.sh" && \
    chown -R 1001:1001 "$USER_HOME/.bashrc.d" && \
    LOADER='for f in ~/.bashrc.d/*.sh; do [ -r "$f" ] && source "$f"; done' && \
    echo "$LOADER" >> "$USER_HOME/.bashrc" && \
    echo "$LOADER" >> "$USER_HOME/.zshrc" && \
    { \
        echo 'if [ -d ~/.bashrc.d ]; then'; \
        echo '  for f in ~/.bashrc.d/*.sh; do [ -r "$f" ] && . "$f"; done'; \
        echo 'fi'; \
    } >> "$USER_HOME/.profile" && \
    chown 1001:1001 "$USER_HOME/.bashrc" "$USER_HOME/.zshrc" "$USER_HOME/.profile"

# ============================================================
# 17. SUDOERS (UID-based — survives dynamic rename)
# ============================================================
RUN echo "#1001 ALL=(ALL) NOPASSWD: ALL" > /etc/sudoers.d/rl-agent && \
    chmod 0440 /etc/sudoers.d/rl-agent

# ============================================================
# 18. GIT (latest via PPA)
# ============================================================
RUN curl -fsSL \
        "https://keyserver.ubuntu.com/pks/lookup?op=get&search=0xE1DD270288B4E6030699E45FA1715D88E1DF1F24" \
        | gpg --dearmor > /usr/share/keyrings/git-core-ppa.gpg \
    && echo "deb [signed-by=/usr/share/keyrings/git-core-ppa.gpg] \
        https://ppa.launchpadcontent.net/git-core/ppa/ubuntu noble main" \
        > /etc/apt/sources.list.d/git-core-ppa.list \
    && apt-get update \
    && apt-get install -y --no-install-recommends git git-lfs git-man \
    && git lfs install \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

RUN git config --system --add safe.directory '*'

# ============================================================
# 19. VS CODE (headless — CLI + extensions for agent file ops)
# ============================================================
RUN wget -qO- https://packages.microsoft.com/keys/microsoft.asc \
        | gpg --dearmor > packages.microsoft.gpg \
    && install -D -o root -g root -m 644 packages.microsoft.gpg \
        /etc/apt/keyrings/packages.microsoft.gpg \
    && echo "deb [arch=amd64,arm64,armhf signed-by=/etc/apt/keyrings/packages.microsoft.gpg] \
        https://packages.microsoft.com/repos/code stable main" \
        > /etc/apt/sources.list.d/vscode.list \
    && rm packages.microsoft.gpg \
    && apt-get update && apt-get install -y --no-install-recommends code \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

RUN mkdir -p /usr/share/code-extensions \
    && chown -R 1001:1001 /usr/share/code-extensions \
    && chmod -R 755 /usr/share/code-extensions

# ELECTRON_DISABLE_SANDBOX=1 prevents code CLI from requiring a display
RUN USER_NAME=$(id -un 1001) && \
    su - "$USER_NAME" -c " \
        ELECTRON_DISABLE_SANDBOX=1 \
        code --no-sandbox \
             --extensions-dir /usr/share/code-extensions \
             --user-data-dir /tmp/vscode-user-data \
             --install-extension ms-vscode.cpptools-extension-pack --force && \
        ELECTRON_DISABLE_SANDBOX=1 \
        code --no-sandbox \
             --extensions-dir /usr/share/code-extensions \
             --user-data-dir /tmp/vscode-user-data \
             --install-extension ms-azuretools.vscode-docker --force && \
        ELECTRON_DISABLE_SANDBOX=1 \
        code --no-sandbox \
             --extensions-dir /usr/share/code-extensions \
             --user-data-dir /tmp/vscode-user-data \
             --install-extension vscjava.vscode-java-pack --force && \
        ELECTRON_DISABLE_SANDBOX=1 \
        code --no-sandbox \
             --extensions-dir /usr/share/code-extensions \
             --user-data-dir /tmp/vscode-user-data \
             --install-extension Oracle.oracle-java --force && \
        ELECTRON_DISABLE_SANDBOX=1 \
        code --no-sandbox \
             --extensions-dir /usr/share/code-extensions \
             --user-data-dir /tmp/vscode-user-data \
             --install-extension ms-dotnettools.vscode-dotnet-runtime --force && \
        ELECTRON_DISABLE_SANDBOX=1 \
        code --no-sandbox \
             --extensions-dir /usr/share/code-extensions \
             --user-data-dir /tmp/vscode-user-data \
             --install-extension ms-dotnettools.csharp --force && \
        ELECTRON_DISABLE_SANDBOX=1 \
        code --no-sandbox \
             --extensions-dir /usr/share/code-extensions \
             --user-data-dir /tmp/vscode-user-data \
             --install-extension ms-dotnettools.csdevkit --force && \
        ELECTRON_DISABLE_SANDBOX=1 \
        code --no-sandbox \
             --extensions-dir /usr/share/code-extensions \
             --user-data-dir /tmp/vscode-user-data \
             --install-extension GitLab.gitlab-workflow --force && \
        ELECTRON_DISABLE_SANDBOX=1 \
        code --no-sandbox \
             --extensions-dir /usr/share/code-extensions \
             --user-data-dir /tmp/vscode-user-data \
             --install-extension eamodio.gitlens --force && \
        ELECTRON_DISABLE_SANDBOX=1 \
        code --no-sandbox \
             --extensions-dir /usr/share/code-extensions \
             --user-data-dir /tmp/vscode-user-data \
             --install-extension golang.go --force && \
        ELECTRON_DISABLE_SANDBOX=1 \
        code --no-sandbox \
             --extensions-dir /usr/share/code-extensions \
             --user-data-dir /tmp/vscode-user-data \
             --install-extension ms-python.python --force && \
        ELECTRON_DISABLE_SANDBOX=1 \
        code --no-sandbox \
             --extensions-dir /usr/share/code-extensions \
             --user-data-dir /tmp/vscode-user-data \
             --install-extension ms-python.vscode-pylance --force && \
        ELECTRON_DISABLE_SANDBOX=1 \
        code --no-sandbox \
             --extensions-dir /usr/share/code-extensions \
             --user-data-dir /tmp/vscode-user-data \
             --install-extension ms-python.debugpy --force && \
        ELECTRON_DISABLE_SANDBOX=1 \
        code --no-sandbox \
             --extensions-dir /usr/share/code-extensions \
             --user-data-dir /tmp/vscode-user-data \
             --install-extension donjayamanne.python-environment-manager --force && \
        ELECTRON_DISABLE_SANDBOX=1 \
        code --no-sandbox \
             --extensions-dir /usr/share/code-extensions \
             --user-data-dir /tmp/vscode-user-data \
             --install-extension rust-lang.rust-analyzer --force && \
        ELECTRON_DISABLE_SANDBOX=1 \
        code --no-sandbox \
             --extensions-dir /usr/share/code-extensions \
             --user-data-dir /tmp/vscode-user-data \
             --install-extension scala-lang.scala --force \
    " && rm -rf /tmp/vscode-user-data

RUN chown -R 1001:1001 /usr/share/code-extensions \
    && chmod -R 755 /usr/share/code-extensions

RUN USER_NAME=$(id -un 1001) && \
    USER_HOME="/home/${USER_NAME}" && \
    mkdir -p "${USER_HOME}/.vscode" && \
    rm -rf "${USER_HOME}/.vscode/extensions" && \
    ln -sf /usr/share/code-extensions "${USER_HOME}/.vscode/extensions" && \
    chown -R 1001:1001 "${USER_HOME}/.vscode"

RUN USER_NAME=$(id -un 1001) && \
    if test -L "/home/${USER_NAME}/.vscode/extensions" && \
       test "$(readlink "/home/${USER_NAME}/.vscode/extensions")" = "/usr/share/code-extensions"; then \
        echo "SUCCESS: VS Code extensions symlink verified"; \
    else \
        echo "ERROR: VS Code extensions symlink missing or incorrect"; \
        exit 1; \
    fi

# ============================================================
# 20. TTYD (web terminal — primary agent terminal interface)
# ============================================================
RUN curl -fsSL https://github.com/tsl0922/ttyd/releases/download/1.7.7/ttyd.x86_64 \
        -o /usr/local/bin/ttyd && \
    chmod +x /usr/local/bin/ttyd

# ============================================================
# 21. TASK EXECUTOR
# Copy into image — not volume-dependent at startup
# ============================================================
RUN mkdir -p /usr/local/lib/task-executor
COPY scripts/tools/task_executor.py /usr/local/lib/task-executor/task_executor.py

# ============================================================
# 22. CONFIG & SCRIPTS
# ============================================================
COPY config/logind.conf /etc/systemd/logind.conf

COPY scripts/tools/start-task-executor.sh      /usr/local/bin/start-task-executor.sh
COPY scripts/tools/check_env_fingerprint.sh    /usr/local/bin/check-env-fingerprint.sh
COPY scripts/tools/prewarm_rl_venv.sh          /usr/local/bin/prewarm-rl-venv.sh
COPY scripts/tools/generate_env_fingerprint.py /usr/local/lib/generate-env-fingerprint.py
RUN chmod +x /usr/local/bin/start-task-executor.sh \
             /usr/local/bin/check-env-fingerprint.sh \
             /usr/local/bin/prewarm-rl-venv.sh

COPY config/systemd/supervisord.service    /etc/systemd/system/supervisord.service
COPY config/systemd/task-executor.service  /etc/systemd/system/task-executor.service

COPY config/supervisord.conf /etc/supervisor/supervisord.conf

# ============================================================
# 23. SYSTEMD WIRING
# ============================================================
# Docker cgroup fix
RUN mkdir -p /etc/systemd/system.conf.d && \
    printf '[Manager]\nManagerEnvironment=SYSTEMD_GENERATOR_SANDBOXED=0\nSystemCallArchitectures=native\n' \
        > /etc/systemd/system.conf.d/10-docker.conf

# Enable services. docker.service + containerd.service are enabled explicitly so
# the daemon is up under agent mode (systemd PID 1) — image-mode task scoring
# (SWE-bench Pro) needs a running dockerd, and task-executor.service is ordered
# After it (see config/systemd/task-executor.service).
RUN mkdir -p /etc/systemd/system/multi-user.target.wants && \
    ln -sf /etc/systemd/system/supervisord.service   /etc/systemd/system/multi-user.target.wants/ && \
    ln -sf /etc/systemd/system/task-executor.service /etc/systemd/system/multi-user.target.wants/ && \
    ln -sf /lib/systemd/system/docker.service        /etc/systemd/system/multi-user.target.wants/ && \
    ln -sf /lib/systemd/system/containerd.service    /etc/systemd/system/multi-user.target.wants/

# Mask services irrelevant in a headless container
RUN systemctl mask \
    apt-daily.service apt-daily.timer \
    apt-daily-upgrade.service apt-daily-upgrade.timer \
    NetworkManager-wait-online.service \
    systemd-networkd-wait-online.service || true

# ============================================================
# 24. POLKIT & LOGROTATE
# ============================================================
RUN mkdir -p /etc/polkit-1/rules.d && \
    echo 'polkit.addRule(function(action, subject) { return polkit.Result.YES; });' \
        > /etc/polkit-1/rules.d/99-allow-all.rules && \
    chmod 644 /etc/polkit-1/rules.d/99-allow-all.rules

RUN printf '/var/log/*.log {\n  daily\n  rotate 7\n  compress\n  missingok\n  notifempty\n  copytruncate\n}\n' \
        > /etc/logrotate.d/docker-logs

# ============================================================
# 25. FINAL CLEANUP & UPGRADES
# ============================================================
RUN apt-get remove -y update-notifier update-notifier-common \
        ubuntu-release-upgrader-core || true && \
    rm -f /var/run/reboot-required* && \
    apt-get update && \
    apt-get upgrade -y && \
    apt-get dist-upgrade -y && \
    apt-get autoremove -y && \
    apt-get autoclean && \
    rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/*

# ============================================================
# 26. FINAL SETTINGS
# ============================================================
RUN chown -R 1001:1001 /home/rl-*

# Pre-create persistent cache and venv directories so mounts and workers
# never race to create them at runtime.
RUN mkdir -p \
        /opt/rl_venvs \
        /workspace/.cache/huggingface \
        /workspace/.cache/torch_inductor \
        /workspace/.cache/xla \
    && chown -R 1001:1001 \
        /opt/rl_venvs \
        /workspace

# Bake fingerprint last — after ALL apt upgrades and package installs are done.
# §25 (dist-upgrade) must complete before this runs or the hash will drift.
RUN python3 /usr/local/lib/generate-env-fingerprint.py > /etc/rl-env-fingerprint

# Default working directory — used by agent mode and task executor.
# RL workloads override this via their own CWD or compose/k8s spec.
WORKDIR /workspace

# Ports: ttyd=7681, SSH=2222, Task Executor=9090
EXPOSE 7681 2222 9090

# SIGTERM works for both modes:
#   - systemd PID 1: responds to SIGTERM with orderly container shutdown
#   - Python/Ray PID 1: standard graceful termination signal
STOPSIGNAL SIGTERM

# /run must be a tmpfs mount for systemd; harmless in RL mode
VOLUME /run

# Mode-aware health check:
#   - Agent mode  → verify task executor (9090) and ttyd are alive
#   - RL mode     → unconditional pass (Ray/SGLang manage their own health)
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD /bin/bash -c \
        '[ "${RL_MODE:-0}" = "1" ] || [ -n "${RAY_HEAD:-}" ] || [ -n "${RAY_WORKER:-}" ] || \
         { ss -tlnp | grep -q ":9090" && pgrep -f ttyd > /dev/null; }'

# ============================================================
# 27. FLEXIBLE ENTRYPOINT
#     Agent mode  (default): systemd as PID 1 — brings up ttyd,
#                            task executor, disk mount, supervisord.
#     RL mode     (opt-in):  your command runs directly as PID 1 —
#                            clean signal delivery to Ray/SGLang/GRPO
#                            workers with no init overhead.
#
#     Activate RL mode via any of:
#       docker run -e RL_MODE=1       ... python run_grpo.py
#       docker run -e RAY_HEAD=1      ... ray start --head
#       docker run -e RAY_WORKER=1    ... ray start --address=...
# ============================================================
RUN cat > /usr/local/bin/docker-entrypoint.sh <<'EOF'
#!/bin/bash
set -e

# Environment integrity check — fast-fail if pip state diverges from the baked image.
# Set SKIP_FINGERPRINT_CHECK=1 to bypass during development.
# RL mode detected first — no fingerprint check for distributed workloads.
if [ "${RL_MODE:-0}" = "1" ] || [ -n "${RAY_HEAD:-}" ] || [ -n "${RAY_WORKER:-}" ]; then
    exec "$@"
fi

# Agent mode only: verify pip state matches the baked image before systemd starts.
if [ "${SKIP_FINGERPRINT_CHECK:-0}" != "1" ]; then
    /usr/local/bin/check-env-fingerprint.sh || exit 1
fi

# Default: full SWE agent mode (systemd)
exec /lib/systemd/systemd "$@"
EOF

RUN chmod +x /usr/local/bin/docker-entrypoint.sh

ENTRYPOINT ["/usr/local/bin/docker-entrypoint.sh"]

# ============================================================
# 27b. BUILD METADATA
# ============================================================
ARG BUILD_COMMIT
ARG BUILD_ID
ARG BUILD_REF
ARG BUILD_DATE

ENV BUILD_COMMIT=${BUILD_COMMIT:-<unknown>} \
    BUILD_ID=${BUILD_ID:-<unknown>} \
    BUILD_REF=${BUILD_REF:-<unknown>} \
    BUILD_DATE=${BUILD_DATE:-<unknown>}

LABEL com.nullvoider.build.id="${BUILD_ID}" \
      com.nullvoider.build.ref="${BUILD_REF}" \
      com.nullvoider.build.date="${BUILD_DATE}"

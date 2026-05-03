FROM ghcr.io/astral-sh/uv:latest

ENV DEBIAN_FRONTEND=noninteractive

# Install llama.cpp server, nginx, curl
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        curl wget git nginx && \
    rm -rf /var/lib/apt/lists/* && \
    rm -f /etc/nginx/sites-enabled/default

# Install llama.cpp server
RUN wget -q https://github.com/ggml-org/llama.cpp/releases/download/b4482/llama-b4482-bin-ubuntu-24.04-x86_64-gpu-cuda-12.6.tgz && \
    tar -xzf llama-b4482-bin-ubuntu-24.04-x86_64-gpu-cuda-12.6.tgz && \
    mv llama-b4482-bin-ubuntu-24.04-x86_64-gpu-cuda-12.6 /app && \
    rm llama-b4482-bin-ubuntu-24.04-x86_64-gpu-cuda-12.6.tgz

WORKDIR /opt/ellm

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project

COPY src/ src/
COPY assets/ assets/
COPY docker/nginx.conf /etc/nginx/conf.d/ellm.conf
COPY docker/entrypoint.sh /entrypoint.sh
COPY data/config/config.yaml /defaults/config.yaml
COPY data/config/profiles/ /defaults/profiles/

RUN chmod +x /entrypoint.sh && \
    chown -R nobody:nogroup /var/log/nginx /var/lib/nginx /run

ENV PATH="/opt/ellm/.venv/bin:$PATH" \
    PYTHONPATH="/opt/ellm/src" \
    LD_LIBRARY_PATH="/app:${LD_LIBRARY_PATH}"

EXPOSE 80 45150

HEALTHCHECK --interval=30s --timeout=10s --retries=10 --start-period=300s \
    CMD curl -f http://localhost:80/health || exit 1

ENTRYPOINT ["/entrypoint.sh"]

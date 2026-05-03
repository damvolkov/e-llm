FROM ghcr.io/ggml-org/llama.cpp:server-cuda

ENV DEBIAN_FRONTEND=noninteractive

# Python 3.12 (Ubuntu 24.04 native) + nginx + curl
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3 python3-venv python3-dev \
        nginx curl && \
    rm -rf /var/lib/apt/lists/* && \
    rm -f /etc/nginx/sites-enabled/default

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/

WORKDIR /opt/ellm

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project --python python3

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

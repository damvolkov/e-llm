FROM nvidia/cuda:12.6.3-devel-ubuntu24.04 AS llama-builder

RUN apt-get update && \
    apt-get install -y --no-install-recommends cmake git build-essential && \
    rm -rf /var/lib/apt/lists/*

RUN git clone --depth 1 --branch b9012 https://github.com/ggml-org/llama.cpp /llama.cpp

RUN ln -sf /usr/local/cuda/lib64/stubs/libcuda.so /usr/local/cuda/lib64/stubs/libcuda.so.1 && \
    LIBRARY_PATH=/usr/local/cuda/lib64/stubs \
    cmake -B /llama.cpp/build -S /llama.cpp -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release && \
    LIBRARY_PATH=/usr/local/cuda/lib64/stubs \
    cmake --build /llama.cpp/build -j$(nproc) --target llama-server


FROM nvidia/cuda:12.6.3-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
COPY --from=llama-builder /llama.cpp/build/bin/llama-server /usr/local/bin/llama-server

RUN apt-get update && \
    apt-get install -y --no-install-recommends curl nginx && \
    rm -rf /var/lib/apt/lists/* && \
    rm -f /etc/nginx/sites-enabled/default

WORKDIR /opt/ellm

COPY pyproject.toml uv.lock ./
RUN uv python install 3.13 && uv sync --frozen --no-install-project

COPY src/ src/
COPY assets/ assets/
COPY docker/nginx.conf /etc/nginx/conf.d/ellm.conf
COPY docker/entrypoint.sh /entrypoint.sh
COPY data/config/config.yaml /defaults/config.yaml
COPY data/config/profiles/ /defaults/profiles/

RUN chmod +x /entrypoint.sh && \
    chown -R nobody:nogroup /var/log/nginx /var/lib/nginx /run

ENV PATH="/opt/ellm/.venv/bin:/usr/local/bin:$PATH" \
    PYTHONPATH="/opt/ellm/src"

EXPOSE 80 45150

HEALTHCHECK --interval=30s --timeout=10s --retries=10 --start-period=300s \
    CMD curl -f http://localhost:80/health || exit 1

ENTRYPOINT ["/entrypoint.sh"]

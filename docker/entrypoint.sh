#!/bin/bash
set -e

# Seed default config if volume is empty
if [ ! -f /data/config/config.yaml ]; then
    mkdir -p /data/config
    cp /defaults/config.yaml /data/config/config.yaml
fi

mkdir -p /data/models /data/cache

nginx -g "daemon on;"

exec python src/e_llm/main.py

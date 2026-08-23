FROM node:22-bookworm-slim@sha256:d649c27dae7ba0137b3cef5dd75baa422c08dc3d9e3fc0c23dfb172dc3cc6436

ENV DEBIAN_FRONTEND=noninteractive \
    PATH=/opt/venv/bin:/usr/local/bin:/usr/bin:/bin \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update \
    && apt-get install --yes --no-install-recommends \
        ca-certificates \
        python3 \
        python3-pip \
        python3-venv \
    && rm -rf /var/lib/apt/lists/* \
    && python3 -m venv /opt/venv \
    && /opt/venv/bin/pip install --no-cache-dir --upgrade pip setuptools wheel

WORKDIR /app

COPY deploy/server/calendar-mcp/package.json deploy/server/calendar-mcp/package-lock.json /app/calendar-mcp/
RUN cd /app/calendar-mcp && npm ci --omit=dev --no-audit --no-fund

COPY vendor/telegram-mcp /app/vendor/telegram-mcp
RUN /opt/venv/bin/pip install --no-cache-dir /app/vendor/telegram-mcp

COPY pyproject.toml /app/
RUN /opt/venv/bin/python -c 'import subprocess, tomllib; dependencies = tomllib.load(open("pyproject.toml", "rb"))["project"]["dependencies"]; subprocess.check_call(["/opt/venv/bin/pip", "install", "--no-cache-dir", *dependencies])'

COPY README.md /app/
COPY src /app/src
RUN /opt/venv/bin/pip install --no-cache-dir --no-deps /app

COPY deploy/server/telegram_gateway.py deploy/server/telegram-gateway /app/deploy/server/
RUN chmod 0555 /app/deploy/server/telegram-gateway \
    && mkdir -p /app/.runtime /tmp/telegram-gateway-files \
    && chown -R 1000:1000 /app/.runtime /tmp/telegram-gateway-files

USER 1000:1000
EXPOSE 8080

ENTRYPOINT ["/opt/venv/bin/tg-voice-transcriber-bot"]

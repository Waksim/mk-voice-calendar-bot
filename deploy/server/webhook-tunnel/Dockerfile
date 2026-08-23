FROM cloudflare/cloudflared:2026.7.2@sha256:4f6655284ab3d252b7f28fedb19fe6c8fc82ee5b1295c20ac74d475e5398a52d AS cloudflared

FROM python:3.13.7-slim-bookworm@sha256:adafcc17694d715c905b4c7bebd96907a1fd5cf183395f0ebc4d3428bd22d92d

ENV HOME=/tmp \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

COPY --from=cloudflared /usr/local/bin/cloudflared /usr/local/bin/cloudflared
COPY controller.py /usr/local/bin/webhook-tunnel-controller

RUN chmod 0555 /usr/local/bin/cloudflared /usr/local/bin/webhook-tunnel-controller

USER 1000:1000

ENTRYPOINT ["python", "/usr/local/bin/webhook-tunnel-controller"]

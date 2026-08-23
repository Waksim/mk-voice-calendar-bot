FROM debian:bookworm-slim

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && apt-get install --yes --no-install-recommends \
        ca-certificates \
        iproute2 \
        wireguard-tools \
    && rm -rf /var/lib/apt/lists/* \
    && mkdir -p /etc/wireguard

COPY entrypoint.sh /usr/local/bin/telegram-warp-entrypoint
RUN chmod 0555 /usr/local/bin/telegram-warp-entrypoint

ENTRYPOINT ["/usr/local/bin/telegram-warp-entrypoint"]

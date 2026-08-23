#!/bin/sh
set -eu

config_path=/etc/wireguard/wg0.conf
if [ ! -s "$config_path" ]; then
    echo "WireGuard configuration is missing" >&2
    exit 1
fi

cleanup() {
    wg-quick down "$config_path" >/dev/null 2>&1 || true
}
trap cleanup EXIT INT TERM

wg-quick up "$config_path"

while wg show wg0 >/dev/null 2>&1; do
    sleep 30 &
    wait $!
done

echo "WireGuard interface disappeared" >&2
exit 1

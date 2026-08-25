#!/bin/sh
set -eu

config_path=/etc/wireguard/wg0.conf
healthy_heartbeat_interval_seconds=300
degraded_heartbeat_interval_seconds=30
healthy_handshake_age_seconds=120

if [ ! -s "$config_path" ]; then
    echo "WireGuard configuration is missing" >&2
    exit 1
fi

cleanup() {
    wg-quick down "$config_path" >/dev/null 2>&1 || true
}
trap cleanup EXIT INT TERM

wg-quick up "$config_path"
echo "WARP state transition: interface=up"

report_heartbeat() {
    now="$(date +%s)"
    handshake_stats="$(
        wg show wg0 latest-handshakes 2>/dev/null \
            | awk '
                $2 > 0 {
                    handshaken += 1
                    if ($2 > latest) latest = $2
                }
                END { printf "%.0f %.0f %.0f\n", latest, NR, handshaken }
            '
    )"
    transfer_stats="$(
        wg show wg0 transfer 2>/dev/null \
            | awk '
                { received += $2; sent += $3 }
                END { printf "%.0f %.0f\n", received, sent }
            '
    )"

    # Both commands above deliberately reduce peer-bearing output to numeric
    # aggregates before it reaches the shell. Never print raw `wg show` output:
    # it contains peer public keys and endpoints.
    set -- $handshake_stats
    latest_handshake="${1:-0}"
    peer_count="${2:-0}"
    handshaken_peer_count="${3:-0}"
    set -- $transfer_stats
    received_bytes="${1:-0}"
    sent_bytes="${2:-0}"

    case "$latest_handshake" in
        ''|*[!0-9]*) latest_handshake=0 ;;
    esac
    if [ "$latest_handshake" -gt 0 ] && [ "$latest_handshake" -le "$now" ]; then
        handshake_age_seconds=$((now - latest_handshake))
    elif [ "$latest_handshake" -gt "$now" ]; then
        # A small wall-clock correction must not produce a negative metric.
        handshake_age_seconds=0
    else
        handshake_age_seconds=-1
    fi

    heartbeat_state=healthy
    if [ "$peer_count" -eq 0 ] \
        || [ "$handshaken_peer_count" -eq 0 ] \
        || [ "$handshake_age_seconds" -ge "$healthy_handshake_age_seconds" ]; then
        heartbeat_state=degraded
    fi

    if [ "$heartbeat_state" = healthy ]; then
        heartbeat_interval_seconds=$healthy_heartbeat_interval_seconds
    else
        heartbeat_interval_seconds=$degraded_heartbeat_interval_seconds
    fi

    heartbeat_kind=periodic
    if [ "$heartbeat_state" != "$previous_heartbeat_state" ]; then
        heartbeat_kind=transition
    elif [ "$seconds_since_heartbeat" -lt "$heartbeat_interval_seconds" ]; then
        previous_heartbeat_state=$heartbeat_state
        return 0
    fi

    echo "WARP heartbeat: kind=$heartbeat_kind state=$heartbeat_state handshake_age_seconds=$handshake_age_seconds rx_bytes=$received_bytes tx_bytes=$sent_bytes peers=$peer_count handshaken_peers=$handshaken_peer_count"
    previous_heartbeat_state=$heartbeat_state
    seconds_since_heartbeat=0
}

previous_heartbeat_state=starting
seconds_since_heartbeat=0
while wg show wg0 >/dev/null 2>&1; do
    sleep 30 &
    wait $!
    seconds_since_heartbeat=$((seconds_since_heartbeat + 30))
    # Sample every 30 seconds so a degraded/recovered transition is visible
    # promptly, while healthy periodic output stays bounded to one line per
    # five minutes.
    report_heartbeat
done

echo "WARP state transition: interface=down reason=interface_disappeared" >&2
exit 1

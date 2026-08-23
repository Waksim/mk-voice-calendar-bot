#!/usr/bin/env python3
"""Supervise a Cloudflare Quick Tunnel and keep Telegram pointed at it."""

from __future__ import annotations

import json
import os
import re
import signal
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

TUNNEL_HOST_RE = re.compile(
    r"^[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.trycloudflare\.com$"
)
SECRET_RE = re.compile(r"^[A-Za-z0-9_-]{1,256}$")
TOKEN_RE = re.compile(r"^[0-9]+:[A-Za-z0-9_-]+$")

BOT_TOKEN_FILE = Path(
    os.environ.get("TELEGRAM_BOT_TOKEN_FILE", "/run/secrets/telegram-bot-token")
)
WEBHOOK_SECRET_FILE = Path(
    os.environ.get(
        "TELEGRAM_WEBHOOK_SECRET_FILE", "/run/secrets/telegram-webhook-secret"
    )
)
WEBHOOK_PATH = os.environ.get(
    "TELEGRAM_WEBHOOK_PATH", "/telegram/mk-voice-text-bot/webhook"
)
ORIGIN_HEALTH_URL = os.environ.get(
    "ORIGIN_HEALTH_URL", "http://127.0.0.1:8080/healthz"
)
CHECK_INTERVAL_SECONDS = int(os.environ.get("CHECK_INTERVAL_SECONDS", "15"))
RETRY_INTERVAL_SECONDS = int(os.environ.get("RETRY_INTERVAL_SECONDS", "3"))
REQUEST_TIMEOUT_SECONDS = int(os.environ.get("REQUEST_TIMEOUT_SECONDS", "15"))
READY_FILE = Path(os.environ.get("READY_FILE", "/tmp/webhook-tunnel.ready"))
METRICS_BASE_URL = os.environ.get(
    "CLOUDFLARED_METRICS_URL", "http://127.0.0.1:20241"
).rstrip("/")


class ControllerError(RuntimeError):
    """A recoverable tunnel-controller failure safe to report without secrets."""


def _read_secret(path: Path, *, pattern: re.Pattern[str], label: str) -> str:
    try:
        value = path.read_text(encoding="utf-8").strip()
    except OSError as exc:
        raise ControllerError(f"cannot read {label}") from exc
    if not pattern.fullmatch(value):
        raise ControllerError(f"invalid {label}")
    return value


def _request_json(
    url: str,
    *,
    data: dict[str, str] | None = None,
    timeout: int = REQUEST_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    encoded = urllib.parse.urlencode(data).encode() if data is not None else None
    request = urllib.request.Request(
        url,
        data=encoded,
        headers={
            "Accept": "application/json",
            "Cache-Control": "no-cache",
            "User-Agent": "mk-voice-calendar-webhook-controller/1",
        },
        method="POST" if encoded is not None else "GET",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = response.read(1_048_577)
    except (OSError, urllib.error.URLError) as exc:
        raise ControllerError("HTTPS request failed") from exc
    if len(payload) > 1_048_576:
        raise ControllerError("HTTPS response is too large")
    try:
        parsed = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ControllerError("HTTPS response is not valid JSON") from exc
    if not isinstance(parsed, dict):
        raise ControllerError("HTTPS response is not a JSON object")
    return parsed


class TelegramWebhook:
    def __init__(self, token: str, secret: str) -> None:
        self._base_url = f"https://api.telegram.org/bot{token}"
        self._secret = secret

    def current_url(self) -> str:
        payload = _request_json(f"{self._base_url}/getWebhookInfo")
        if payload.get("ok") is not True or not isinstance(payload.get("result"), dict):
            raise ControllerError("Telegram rejected getWebhookInfo")
        url = payload["result"].get("url", "")
        if not isinstance(url, str):
            raise ControllerError("Telegram returned an invalid webhook URL")
        return url

    def set_url(self, url: str) -> None:
        payload = _request_json(
            f"{self._base_url}/setWebhook",
            data={
                "url": url,
                "secret_token": self._secret,
                "max_connections": "1",
                "allowed_updates": json.dumps(
                    ["message", "callback_query"], separators=(",", ":")
                ),
                "drop_pending_updates": "false",
            },
        )
        if payload.get("ok") is not True:
            raise ControllerError("Telegram rejected setWebhook")


def _health_is_ok(url: str) -> bool:
    try:
        payload = _request_json(url)
    except ControllerError:
        return False
    return payload.get("ok") is True


def _discover_public_base_url() -> str:
    ready = _request_json(f"{METRICS_BASE_URL}/ready", timeout=3)
    if ready.get("status") != 200:
        raise ControllerError("cloudflared is not ready")
    connections = ready.get("readyConnections")
    if not isinstance(connections, int) or connections < 1:
        raise ControllerError("cloudflared has no ready connections")

    quick_tunnel = _request_json(f"{METRICS_BASE_URL}/quicktunnel", timeout=3)
    hostname = quick_tunnel.get("hostname")
    if not isinstance(hostname, str) or not TUNNEL_HOST_RE.fullmatch(hostname):
        raise ControllerError("cloudflared returned an invalid quick-tunnel hostname")
    return f"https://{hostname}"


def _mark_ready(url: str) -> None:
    temporary = READY_FILE.with_suffix(".tmp")
    temporary.write_text(f"{url}\n", encoding="utf-8")
    os.replace(temporary, READY_FILE)


def _mark_unready() -> None:
    try:
        READY_FILE.unlink()
    except FileNotFoundError:
        pass


def _stream_logs(process: subprocess.Popen[str]) -> None:
    assert process.stdout is not None
    for line in process.stdout:
        clean = line.rstrip("\r\n")
        print(clean, flush=True)


def _validate_configuration() -> tuple[str, str]:
    if (
        not WEBHOOK_PATH.startswith("/")
        or "?" in WEBHOOK_PATH
        or "#" in WEBHOOK_PATH
        or "//" in WEBHOOK_PATH
    ):
        raise ControllerError("invalid Telegram webhook path")
    if CHECK_INTERVAL_SECONDS <= 0 or RETRY_INTERVAL_SECONDS <= 0:
        raise ControllerError("controller intervals must be positive")
    token = _read_secret(BOT_TOKEN_FILE, pattern=TOKEN_RE, label="Telegram bot token")
    secret = _read_secret(
        WEBHOOK_SECRET_FILE, pattern=SECRET_RE, label="Telegram webhook secret"
    )
    return token, secret


def run() -> int:
    token, secret = _validate_configuration()
    telegram = TelegramWebhook(token, secret)
    _mark_unready()

    process = subprocess.Popen(
        [
            "/usr/local/bin/cloudflared",
            "--no-autoupdate",
            "tunnel",
            "--edge-ip-version",
            "4",
            "--protocol",
            "http2",
            "--metrics",
            "127.0.0.1:20241",
            "--url",
            "http://127.0.0.1:8080",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    stopping = threading.Event()

    def stop(_signum: int, _frame: object) -> None:
        stopping.set()
        if process.poll() is None:
            process.terminate()

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)

    reader = threading.Thread(target=_stream_logs, args=(process,), daemon=True)
    reader.start()

    next_check = 0.0

    while not stopping.is_set():
        if process.poll() is not None:
            break
        if time.monotonic() < next_check:
            stopping.wait(1)
            continue

        try:
            public_base_url = _discover_public_base_url()
            desired_url = f"{public_base_url}{WEBHOOK_PATH}"
            public_health_url = f"{public_base_url}/healthz"
            if not _health_is_ok(ORIGIN_HEALTH_URL):
                raise ControllerError("bot origin is not healthy")
            if not _health_is_ok(public_health_url):
                raise ControllerError("public tunnel is not healthy")
            if _discover_public_base_url() != public_base_url:
                raise ControllerError("quick-tunnel hostname changed during readiness")
            if telegram.current_url() != desired_url:
                telegram.set_url(desired_url)
                print("Telegram webhook now targets the active tunnel", flush=True)
            if telegram.current_url() != desired_url:
                raise ControllerError("Telegram webhook verification failed")
            _mark_ready(desired_url)
            next_check = time.monotonic() + CHECK_INTERVAL_SECONDS
        except ControllerError as exc:
            _mark_unready()
            print(f"Webhook readiness check failed: {exc}", file=sys.stderr, flush=True)
            next_check = time.monotonic() + RETRY_INTERVAL_SECONDS
        stopping.wait(1)

    _mark_unready()
    if process.poll() is None:
        process.terminate()
    try:
        returncode = process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()
        returncode = process.wait(timeout=5)
    if stopping.is_set():
        return 0
    print(f"cloudflared exited unexpectedly with status {returncode}", file=sys.stderr)
    return returncode or 1


def main() -> None:
    try:
        raise SystemExit(run())
    except ControllerError as exc:
        print(f"Webhook tunnel configuration error: {exc}", file=sys.stderr)
        raise SystemExit(2) from None


if __name__ == "__main__":
    main()

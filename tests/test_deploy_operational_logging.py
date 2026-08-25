from __future__ import annotations

import importlib.util
from pathlib import Path
import subprocess

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TUNNEL_CONTROLLER_PATH = (
    PROJECT_ROOT / "deploy" / "server" / "webhook-tunnel" / "controller.py"
)
WARP_ENTRYPOINT_PATH = (
    PROJECT_ROOT / "deploy" / "server" / "warp" / "entrypoint.sh"
)


def _load_tunnel_controller():
    spec = importlib.util.spec_from_file_location(
        "test_webhook_tunnel_controller", TUNNEL_CONTROLLER_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_readiness_reporter_deduplicates_checks_and_emits_heartbeats(capsys):
    controller = _load_tunnel_controller()
    now = [100.0]
    reporter = controller.ReadinessReporter(
        heartbeat_seconds=300,
        clock=lambda: now[0],
    )

    reporter.update("unready", reason="bot origin is not healthy")
    now[0] += 10
    reporter.update("unready", reason="HTTPS request failed")
    now[0] += 300
    reporter.update("unready", reason="HTTPS request failed")
    reporter.update("ready")
    now[0] += 299
    reporter.update("ready")
    now[0] += 1
    reporter.update("ready")

    captured = capsys.readouterr()
    assert captured.err.splitlines() == [
        "Webhook tunnel readiness transition: state=unready "
        "reason=bot origin is not healthy",
        "Webhook tunnel readiness transition: state=unready "
        "reason=HTTPS request failed",
        "Webhook tunnel readiness heartbeat: state=unready "
        "reason=HTTPS request failed",
    ]
    assert captured.out.splitlines() == [
        "Webhook tunnel readiness transition: state=ready",
        "Webhook tunnel readiness heartbeat: state=ready",
    ]


@pytest.mark.parametrize(
    ("state", "reason"),
    (("unknown", None), ("unready", None)),
)
def test_readiness_reporter_rejects_invalid_state(state, reason):
    controller = _load_tunnel_controller()
    reporter = controller.ReadinessReporter(clock=lambda: 0.0)

    with pytest.raises(ValueError):
        reporter.update(state, reason=reason)


def test_warp_entrypoint_is_valid_posix_shell_and_avoids_raw_peer_output():
    subprocess.run(["sh", "-n", str(WARP_ENTRYPOINT_PATH)], check=True)
    source = WARP_ENTRYPOINT_PATH.read_text(encoding="utf-8")

    assert "latest_handshakes" not in source
    assert "latest-handshakes" in source
    assert "wg show wg0 dump" not in source
    assert "peer public keys and endpoints" in source
    assert "handshake_age_seconds=" in source
    assert "rx_bytes=" in source
    assert "tx_bytes=" in source

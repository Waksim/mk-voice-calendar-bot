import asyncio
from contextlib import asynccontextmanager

import pytest

import tg_voice_transcriber_bot.service as service_module
from tg_voice_transcriber_bot.config import Config


def test_async_main_opens_validates_and_wires_calendar_mcp(tmp_path, monkeypatch):
    gateway_root = tmp_path / "gateway-cache"
    launcher = gateway_root / "1" / "scripts" / "telegram-gateway"
    launcher.parent.mkdir(parents=True)
    launcher.write_text("#!/bin/sh\n", encoding="utf-8")
    calendar_root = tmp_path / "calendar-mcp"
    binary = calendar_root / "node_modules" / ".bin" / "google-calendar-mcp"
    config = Config(
        state_path=tmp_path / "state.json",
        confirmation_state_path=tmp_path / "calendar-confirmations.json",
        gateway_cache_root=gateway_root,
        calendar_mcp_working_directory=calendar_root,
        calendar_mcp_binary_path=binary,
        calendar_mcp_oauth_credentials_path=tmp_path / "oauth.json",
        calendar_mcp_token_path=tmp_path / "tokens.json",
    )
    sequence = []
    captured = {}

    def fake_read_secret(*, environment, account=None, service=None):
        if environment == config.bot_token_environment:
            return "fake-bot-token"
        if environment == config.openrouter_api_key_environment:
            assert account == config.openrouter_keychain_account
            assert service == config.openrouter_keychain_service
            return "fake-openrouter-key"
        raise RuntimeError("Unexpected secret request")

    class FakeOpenRouterApi:
        def __init__(self, api_key, **kwargs):
            assert api_key == "fake-openrouter-key"
            captured["openrouter_kwargs"] = kwargs

        async def aclose(self):
            sequence.append("openrouter_close")

    class FakePlannerFallback:
        def __init__(self, primary, fallback, *, timeout_seconds):
            captured["fallback_provider"] = self
            captured["openrouter_primary"] = primary
            captured["cli_fallback"] = fallback
            captured["planner_timeout_seconds"] = timeout_seconds

    class FakeBotApi:
        def __init__(self, token):
            assert token == "fake-bot-token"

        async def __aenter__(self):
            sequence.append("bot_open")
            return self

        async def __aexit__(self, exc_type, exc, traceback):
            sequence.append("bot_close")
            return False

    @asynccontextmanager
    async def fake_open_gateway(path, *, default_timeout):
        assert path == launcher
        assert default_timeout == config.gateway_call_timeout_seconds
        sequence.append("gateway_open")
        yield object()
        sequence.append("gateway_close")

    class FakeCalendar:
        async def validate(self):
            sequence.append("calendar_validate")

    calendar = FakeCalendar()

    @asynccontextmanager
    async def fake_open_calendar_mcp(path, **kwargs):
        captured["calendar_path"] = path
        captured["calendar_kwargs"] = kwargs
        sequence.append("calendar_open")
        yield calendar
        sequence.append("calendar_close")

    class FakeVoiceBotService:
        def __init__(
            self,
            actual_config,
            bot,
            gateway,
            state,
            gemini,
            calendar_confirmation,
        ):
            assert actual_config is config
            assert calendar_confirmation is not None
            assert calendar_confirmation.calendar is calendar
            captured["confirmation"] = calendar_confirmation
            captured["planner"] = gemini

        async def initialize(self):
            sequence.append("service_initialize")

        async def run(self):
            sequence.append("service_run")

    monkeypatch.setattr(service_module, "Config", lambda: config)
    monkeypatch.setattr(
        service_module, "read_secret", fake_read_secret
    )
    monkeypatch.setattr(service_module, "BotApi", FakeBotApi)
    monkeypatch.setattr(service_module, "OpenRouterApi", FakeOpenRouterApi)
    monkeypatch.setattr(service_module, "GeminiFallback", FakePlannerFallback)
    monkeypatch.setattr(service_module, "open_gateway", fake_open_gateway)
    monkeypatch.setattr(
        service_module, "open_calendar_mcp", fake_open_calendar_mcp
    )
    monkeypatch.setattr(service_module, "VoiceBotService", FakeVoiceBotService)

    asyncio.run(service_module.async_main())

    assert captured["calendar_path"] == binary
    assert captured["calendar_kwargs"] == {
        "account_mapping": {"personal": "owner", "work": "owner"},
        "calendar_id": "primary",
        "default_timeout_seconds": 45,
        "working_directory": calendar_root,
        "env": {
            "GOOGLE_OAUTH_CREDENTIALS": str(tmp_path / "oauth.json"),
            "GOOGLE_CALENDAR_MCP_TOKEN_PATH": str(tmp_path / "tokens.json"),
                "GOOGLE_ACCOUNT_MODE": "owner",
                "PATH": "/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin",
                "TRANSPORT": "stdio",
            "DEBUG": "false",
        },
    }
    assert captured["openrouter_kwargs"] == {
        "model": "meta/muse-spark-1.2-contributor",
        "timeout_seconds": 45,
        "timezone": "Europe/Moscow",
        "reasoning_effort": "high",
        "max_tokens": 8192,
    }
    assert captured["planner"] is captured["fallback_provider"]
    assert captured["planner_timeout_seconds"] == 45
    assert captured["cli_fallback"].model == "gemini-3.7-flash-high"
    assert sequence == [
        "bot_open",
        "gateway_open",
        "calendar_open",
        "calendar_validate",
        "service_initialize",
        "service_run",
        "calendar_close",
        "gateway_close",
        "bot_close",
        "openrouter_close",
    ]


@pytest.mark.parametrize("register_with_telegram", [True, False])
def test_webhook_listener_registration_ownership(
    tmp_path, monkeypatch, register_with_telegram
):
    launcher = tmp_path / "telegram-gateway"
    launcher.write_text("#!/bin/sh\n", encoding="utf-8")
    config = Config(
        bot_update_mode="webhook",
        webhook_public_url="https://calendar.example.test/telegram/bot/webhook",
        webhook_register_with_telegram=register_with_telegram,
        gateway_launcher_path=launcher,
        state_path=tmp_path / "state.json",
        confirmation_state_path=tmp_path / "confirmations.json",
        operation_state_path=tmp_path / "operations.json",
        calendar_mcp_working_directory=tmp_path,
        calendar_mcp_binary_path=tmp_path / "calendar-mcp",
        calendar_mcp_oauth_credentials_path=tmp_path / "oauth.json",
        calendar_mcp_token_path=tmp_path / "tokens.json",
    )
    sequence = []
    planners = []

    def fake_read_secret(*, environment, account=None, service=None):
        if environment == config.bot_token_environment:
            return "fake-bot-token"
        if environment == config.webhook_secret_environment:
            return "webhook_secret-123"
        raise RuntimeError("OpenRouter key intentionally absent")

    class FakeBotApi:
        def __init__(self, token):
            assert token == "fake-bot-token"

        async def __aenter__(self):
            sequence.append("bot_open")
            return self

        async def __aexit__(self, exc_type, exc, traceback):
            sequence.append("bot_close")
            return False

        async def set_webhook(self, url, secret_token):
            assert url == config.webhook_public_url
            assert secret_token == "webhook_secret-123"
            sequence.append("set_webhook")

    @asynccontextmanager
    async def fake_open_gateway(path, *, default_timeout):
        sequence.append("gateway_open")
        yield object()
        sequence.append("gateway_close")

    class FakeCalendar:
        async def validate(self):
            sequence.append("calendar_validate")

    @asynccontextmanager
    async def fake_open_calendar_mcp(path, **kwargs):
        sequence.append("calendar_open")
        yield FakeCalendar()
        sequence.append("calendar_close")

    class FakeVoiceBotService:
        def __init__(self, *args, **kwargs):
            self.calendar_operations = None
            planners.append(args[4])

        async def initialize(self):
            sequence.append("service_initialize")

    class FakeWebhookRuntime:
        def __init__(self, handler, state, **kwargs):
            assert kwargs["path"] == "/telegram/bot/webhook"
            assert kwargs["secret_token"] == "webhook_secret-123"
            sequence.append("webhook_constructed")

        async def start(self):
            sequence.append("listener_started")

        async def run_forever(self):
            sequence.append("webhook_run")

        async def close(self):
            sequence.append("webhook_close")

    monkeypatch.setattr(service_module, "Config", lambda: config)
    monkeypatch.setattr(service_module, "read_secret", fake_read_secret)
    monkeypatch.setattr(service_module, "BotApi", FakeBotApi)
    monkeypatch.setattr(service_module, "open_gateway", fake_open_gateway)
    monkeypatch.setattr(service_module, "open_calendar_mcp", fake_open_calendar_mcp)
    monkeypatch.setattr(service_module, "VoiceBotService", FakeVoiceBotService)
    monkeypatch.setattr(service_module, "WebhookRuntime", FakeWebhookRuntime)

    asyncio.run(service_module.async_main())

    if register_with_telegram:
        assert sequence.index("listener_started") < sequence.index("set_webhook")
    else:
        assert "set_webhook" not in sequence
    assert "webhook_run" in sequence
    assert "webhook_close" in sequence
    assert len(planners) == 1
    assert isinstance(planners[0], service_module.GeminiCli)
    assert sequence[-3:] == ["calendar_close", "gateway_close", "bot_close"]

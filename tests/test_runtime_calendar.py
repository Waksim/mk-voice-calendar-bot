import asyncio
from contextlib import asynccontextmanager

import pytest

import tg_voice_transcriber_bot.service as service_module
from tg_voice_transcriber_bot.config import Config


@pytest.mark.parametrize(
    "failed_model,expected_stage,expected_model",
    (
        ("gpt-5.6-sol", "Codex Luna", "gpt-5.6-luna"),
        ("gpt-5.6-luna", "Codex Sol", "gpt-5.6-sol"),
    ),
)
def test_codex_stage_configuration_failures_are_independent(
    monkeypatch, failed_model, expected_stage, expected_model
):
    attempts = []
    closed = []

    class PartiallyRejectedCodexRunnerApi:
        def __init__(self, **kwargs):
            self.model = kwargs["model"]
            attempts.append(self.model)
            if self.model == failed_model:
                raise service_module.CodexCliConfigurationError(
                    "runner rejected test configuration"
                )

        async def aclose(self):
            closed.append(self.model)

    monkeypatch.setattr(
        service_module,
        "CodexCliRunnerApi",
        PartiallyRejectedCodexRunnerApi,
    )

    stages = service_module._build_codex_planner_stages(
        Config(),
        "codex-runner-token-with-at-least-32-characters",
    )

    assert attempts == ["gpt-5.6-sol", "gpt-5.6-luna"]
    assert [stage.name for stage in stages] == [expected_stage]
    assert stages[0].provider.model == expected_model
    asyncio.run(stages[0].provider.aclose())
    assert closed == [expected_model]


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
        openrouter_model="nvidia/nemotron-3-super-120b-a12b:free",
        openrouter_timeout_seconds=35,
        openrouter_reasoning_effort="medium",
        openrouter_fallback_model="z-ai/glm-5.2:free",
        openrouter_fallback_timeout_seconds=15,
        openrouter_fallback_reasoning_effort="high",
        openrouter_max_tokens=8192,
        gigachat_scope="GIGACHAT_API_CORP",
        gigachat_model="GigaChat-2-Max",
        gigachat_base_url="https://giga.example.test/v1",
        gigachat_auth_url="https://oauth.giga.example.test/token",
        gigachat_ca_bundle_file=tmp_path / "giga-root.pem",
        gigachat_timeout_seconds=45,
        codex_timeout_seconds=55,
        codex_fallback_timeout_seconds=70,
        gemini_model="gemini-3.7-flash",
        gemini_timeout_seconds=25,
        calendar_planner_timeout_seconds=250,
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
        if environment == config.codex_runner_token_environment:
            return "codex-runner-token-with-at-least-32-characters"
        if environment == config.gigachat_credentials_environment:
            assert account == config.gigachat_keychain_account
            assert service == config.gigachat_keychain_service
            sequence.append("gigachat_secret_read")
            return "fake-gigachat-credentials"
        if environment == config.openrouter_api_key_environment:
            assert account == config.openrouter_keychain_account
            assert service == config.openrouter_keychain_service
            sequence.append("openrouter_secret_read")
            return "fake-openrouter-key"
        if environment == config.gemini_api_key_environment:
            assert account == config.gemini_keychain_account
            assert service == config.gemini_keychain_service
            sequence.append("gemini_secret_read")
            return "fake-gemini-key"
        raise RuntimeError("Unexpected secret request")

    class FakeGigaChatApi:
        def __init__(self, credentials, **kwargs):
            assert credentials == "fake-gigachat-credentials"
            self.model = kwargs["model"]
            self.kwargs = kwargs
            captured["gigachat_client"] = self

        async def aclose(self):
            sequence.append(f"provider_close:{self.model}")

    class FakeCodexRunnerApi:
        def __init__(self, **kwargs):
            self.model = kwargs["model"]
            self.kwargs = kwargs
            captured.setdefault("codex_clients", []).append(self)

        async def aclose(self):
            sequence.append(f"provider_close:{self.model}")

    class FakeOpenRouterApi:
        def __init__(self, api_key, **kwargs):
            assert api_key == "fake-openrouter-key"
            self.model = kwargs["model"]
            self.kwargs = kwargs
            captured.setdefault("openrouter_clients", []).append(self)

        async def aclose(self):
            sequence.append(f"provider_close:{self.model}")

    class FakeGeminiApi:
        def __init__(self, api_key, **kwargs):
            assert api_key == "fake-gemini-key"
            self.model = kwargs["model"]
            self.kwargs = kwargs
            captured["gemini_client"] = self

        async def aclose(self):
            sequence.append(f"provider_close:{self.model}")

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
    monkeypatch.setattr(service_module, "CodexCliRunnerApi", FakeCodexRunnerApi)
    monkeypatch.setattr(service_module, "GigaChatApi", FakeGigaChatApi)
    monkeypatch.setattr(service_module, "OpenRouterApi", FakeOpenRouterApi)
    monkeypatch.setattr(service_module, "GeminiApi", FakeGeminiApi)
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
    openrouter_clients = captured["openrouter_clients"]
    assert len(openrouter_clients) == 2
    assert openrouter_clients[0].kwargs == {
        "model": "nvidia/nemotron-3-super-120b-a12b:free",
        "timeout_seconds": 35,
        "timezone": "Europe/Moscow",
        "reasoning_effort": "medium",
        "max_tokens": 8192,
        "max_retries": 0,
    }
    assert openrouter_clients[1].kwargs == {
        "model": "z-ai/glm-5.2:free",
        "timeout_seconds": 15,
        "timezone": "Europe/Moscow",
        "reasoning_effort": "high",
        "max_tokens": 8192,
        "max_retries": 0,
    }
    gigachat_client = captured["gigachat_client"]
    assert gigachat_client.kwargs == {
        "ca_bundle_path": tmp_path / "giga-root.pem",
        "timeout_seconds": 45,
        "timezone": "Europe/Moscow",
        "scope": "GIGACHAT_API_CORP",
        "model": "GigaChat-2-Max",
        "base_url": "https://giga.example.test/v1",
        "auth_url": "https://oauth.giga.example.test/token",
        "max_retries": 1,
    }
    gemini_client = captured["gemini_client"]
    assert gemini_client.kwargs == {
        "model": "gemini-3.7-flash",
        "timeout_seconds": 25,
        "timezone": "Europe/Moscow",
        "max_retries": 1,
    }
    planner = captured["planner"]
    assert isinstance(planner, service_module.GeminiProviderChain)
    assert planner.timeout_seconds == 250
    codex_clients = captured["codex_clients"]
    assert len(codex_clients) == 2
    assert codex_clients[0].kwargs == {
        "base_url": "http://127.0.0.1:8091",
        "bearer_token": "codex-runner-token-with-at-least-32-characters",
        "model": "gpt-5.6-sol",
        "reasoning_effort": "medium",
        "timeout_seconds": 55,
        "timezone": "Europe/Moscow",
    }
    assert codex_clients[1].kwargs == {
        "base_url": "http://127.0.0.1:8092",
        "bearer_token": "codex-runner-token-with-at-least-32-characters",
        "model": "gpt-5.6-luna",
        "reasoning_effort": "xhigh",
        "timeout_seconds": 70,
        "timezone": "Europe/Moscow",
    }
    assert [stage.name for stage in planner.stages] == [
        "Codex Sol",
        "Codex Luna",
        "Nemotron 3 Super",
        "GLM 5.2 Free",
        "Gemini 3.7 Flash",
        "GigaChat 2 Max",
    ]
    assert [stage.timeout_seconds for stage in planner.stages] == [
        55,
        70,
        35,
        15,
        25,
        45,
    ]
    assert [stage.provider for stage in planner.stages] == [
        codex_clients[0],
        codex_clients[1],
        openrouter_clients[0],
        openrouter_clients[1],
        gemini_client,
        gigachat_client,
    ]
    assert sequence == [
        "gigachat_secret_read",
        "openrouter_secret_read",
        "gemini_secret_read",
        "bot_open",
        "gateway_open",
        "calendar_open",
        "calendar_validate",
        "service_initialize",
        "service_run",
        "calendar_close",
        "gateway_close",
        "bot_close",
        "provider_close:gpt-5.6-sol",
        "provider_close:gpt-5.6-luna",
        "provider_close:nvidia/nemotron-3-super-120b-a12b:free",
        "provider_close:z-ai/glm-5.2:free",
        "provider_close:gemini-3.7-flash",
        "provider_close:GigaChat-2-Max",
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
    secret_requests = []
    gigachat_constructor_attempts = []

    def fake_read_secret(*, environment, account=None, service=None):
        secret_requests.append((environment, account, service))
        if environment == config.bot_token_environment:
            return "fake-bot-token"
        if environment == config.codex_runner_token_environment:
            return "codex-runner-token-with-at-least-32-characters"
        if environment == config.webhook_secret_environment:
            return "webhook_secret-123"
        if environment == config.gigachat_credentials_environment:
            if register_with_telegram:
                return "fake-gigachat-credentials"
            raise RuntimeError("GigaChat credentials intentionally absent")
        if environment == config.gemini_api_key_environment:
            return "fake-gemini-key"
        raise RuntimeError("OpenRouter key intentionally absent")

    class RejectedGigaChatApi:
        def __init__(self, credentials, **kwargs):
            assert credentials == "fake-gigachat-credentials"
            gigachat_constructor_attempts.append(kwargs)
            raise service_module.GigaChatConfigurationError(
                "GigaChat CA bundle is unavailable"
            )

    class FakeCodexRunnerApi:
        def __init__(self, **kwargs):
            self.model = kwargs["model"]

        async def aclose(self):
            return None

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
    monkeypatch.setattr(service_module, "CodexCliRunnerApi", FakeCodexRunnerApi)
    monkeypatch.setattr(service_module, "GigaChatApi", RejectedGigaChatApi)
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
    assert (
        config.gigachat_credentials_environment,
        config.gigachat_keychain_account,
        config.gigachat_keychain_service,
    ) in secret_requests
    assert isinstance(planners[0], service_module.GeminiProviderChain)
    assert planners[0].timeout_seconds == config.calendar_planner_timeout_seconds
    assert [stage.name for stage in planners[0].stages] == [
        "Codex Sol",
        "Codex Luna",
        "Gemini 3.7 Flash",
    ]
    assert isinstance(
        planners[0].stages[0].provider, FakeCodexRunnerApi
    )
    assert (
        planners[0].stages[2].timeout_seconds
        == config.gemini_timeout_seconds
    )
    assert len(gigachat_constructor_attempts) == int(register_with_telegram)
    assert sequence[-3:] == ["calendar_close", "gateway_close", "bot_close"]


def test_webhook_mode_requires_direct_gemini_api_key(tmp_path, monkeypatch):
    launcher = tmp_path / "telegram-gateway"
    launcher.write_text("#!/bin/sh\n", encoding="utf-8")
    config = Config(
        bot_update_mode="webhook",
        webhook_public_url="https://calendar.example.test/telegram/bot/webhook",
        gateway_launcher_path=launcher,
        state_path=tmp_path / "state.json",
        confirmation_state_path=tmp_path / "confirmations.json",
        operation_state_path=tmp_path / "operations.json",
    )

    def fake_read_secret(*, environment, account=None, service=None):
        if environment == config.bot_token_environment:
            return "fake-bot-token"
        if environment == config.webhook_secret_environment:
            return "webhook_secret-123"
        raise RuntimeError("Provider key intentionally absent")

    monkeypatch.setattr(service_module, "Config", lambda: config)
    monkeypatch.setattr(service_module, "read_secret", fake_read_secret)

    with pytest.raises(
        RuntimeError, match="Gemini API key is required in webhook mode"
    ):
        asyncio.run(service_module.async_main())

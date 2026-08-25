import pytest

from tg_voice_transcriber_bot.config import PROJECT_ROOT, Config


def test_calendar_mcp_runtime_paths_environment_and_mapping_are_exact():
    config = Config()

    assert config.openrouter_api_key_environment == "OPENROUTER_API_KEY"
    assert config.openrouter_model == "nvidia/nemotron-3-super-120b-a12b:free"
    assert config.openrouter_timeout_seconds == 35
    assert config.openrouter_reasoning_effort == "medium"
    assert config.openrouter_fallback_model == "z-ai/glm-5.2:free"
    assert config.openrouter_fallback_timeout_seconds == 15
    assert config.openrouter_fallback_reasoning_effort == "high"
    assert config.openrouter_max_tokens == 8192
    assert config.gemini_keychain_account == "codex.gemini.mk_voice_calendar_bot"
    assert config.gemini_keychain_service == "mk_voice_calendar_bot"
    assert config.gemini_api_key_environment == "GEMINI_API_KEY"
    assert config.gemini_model == "gemini-3.7-flash"
    assert config.gemini_timeout_seconds == 25
    assert config.calendar_planner_timeout_seconds == 80
    assert config.gemini_cli_model == "gemini-3.7-flash-high"
    assert config.calendar_mcp_package_version == "2.6.2"
    assert config.calendar_mcp_working_directory == PROJECT_ROOT / "calendar-mcp"
    assert config.calendar_mcp_binary_path == (
        PROJECT_ROOT
        / "calendar-mcp"
        / "node_modules"
        / ".bin"
        / "google-calendar-mcp"
    )
    assert config.calendar_mcp_oauth_credentials_path == (
        PROJECT_ROOT / ".runtime" / "google-calendar-oauth.json"
    )
    assert config.calendar_mcp_token_path == (
        PROJECT_ROOT / ".runtime" / "google-calendar-tokens.json"
    )
    assert config.calendar_mcp_account_mapping == {
        "personal": "owner",
        "work": "owner",
    }
    assert config.account_by_user_id == {
        100000001: "personal",
        100000002: "work",
    }
    assert config.calendar_mcp_calendar_id == "primary"
    assert config.calendar_mcp_env == {
        "GOOGLE_OAUTH_CREDENTIALS": str(
            PROJECT_ROOT / ".runtime" / "google-calendar-oauth.json"
        ),
        "GOOGLE_CALENDAR_MCP_TOKEN_PATH": str(
            PROJECT_ROOT / ".runtime" / "google-calendar-tokens.json"
        ),
        "GOOGLE_ACCOUNT_MODE": "owner",
        "PATH": "/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin",
        "TRANSPORT": "stdio",
        "DEBUG": "false",
    }


def test_environment_overrides_gateway_webhook_and_linux_paths(tmp_path, monkeypatch):
    launcher = tmp_path / "telegram-gateway"
    launcher.write_text("#!/bin/sh\n", encoding="utf-8")
    monkeypatch.setenv("TELEGRAM_GATEWAY_LAUNCHER", str(launcher))
    monkeypatch.setenv("TELEGRAM_BOT_UPDATE_MODE", "webhook")
    monkeypatch.setenv(
        "TELEGRAM_WEBHOOK_URL", "https://calendar.example.test/telegram/bot/webhook"
    )
    monkeypatch.setenv("TELEGRAM_WEBHOOK_LISTEN_HOST", "0.0.0.0")
    monkeypatch.setenv("TELEGRAM_WEBHOOK_LISTEN_PORT", "8090")
    monkeypatch.setenv("TELEGRAM_WEBHOOK_REGISTER", "false")
    monkeypatch.setenv("TELEGRAM_WEBHOOK_PATH", "/telegram/managed/webhook")
    monkeypatch.setenv("CALENDAR_MCP_PROCESS_PATH", "/usr/local/bin:/usr/bin:/bin")
    monkeypatch.setenv("OPENROUTER_MODEL", "meta/muse-spark-1.2")
    monkeypatch.setenv("OPENROUTER_TIMEOUT_SECONDS", "60")
    monkeypatch.setenv("OPENROUTER_REASONING_EFFORT", "medium")
    monkeypatch.setenv("OPENROUTER_FALLBACK_MODEL", "z-ai/glm-5.2:free")
    monkeypatch.setenv("OPENROUTER_FALLBACK_TIMEOUT_SECONDS", "20")
    monkeypatch.setenv("OPENROUTER_FALLBACK_REASONING_EFFORT", "low")
    monkeypatch.setenv("OPENROUTER_MAX_TOKENS", "4096")
    monkeypatch.setenv("GEMINI_MODEL", "gemini-test-model")
    monkeypatch.setenv("GEMINI_TIMEOUT_SECONDS", "30")
    monkeypatch.setenv("CALENDAR_PLANNER_TIMEOUT_SECONDS", "120")
    monkeypatch.setenv("GEMINI_CLI_MODEL", "gemini-cli-test-model")

    config = Config()

    assert config.discover_gateway_launcher() == launcher
    assert config.bot_update_mode == "webhook"
    assert config.webhook_path == "/telegram/managed/webhook"
    assert config.webhook_register_with_telegram is False
    assert config.webhook_listen_host == "0.0.0.0"
    assert config.webhook_listen_port == 8090
    assert config.calendar_mcp_env["PATH"] == "/usr/local/bin:/usr/bin:/bin"
    assert config.openrouter_model == "meta/muse-spark-1.2"
    assert config.openrouter_timeout_seconds == 60
    assert config.openrouter_reasoning_effort == "medium"
    assert config.openrouter_fallback_model == "z-ai/glm-5.2:free"
    assert config.openrouter_fallback_timeout_seconds == 20
    assert config.openrouter_fallback_reasoning_effort == "low"
    assert config.openrouter_max_tokens == 4096
    assert config.gemini_model == "gemini-test-model"
    assert config.gemini_timeout_seconds == 30
    assert config.calendar_planner_timeout_seconds == 120
    assert config.gemini_cli_model == "gemini-cli-test-model"


@pytest.mark.parametrize(
    "environment,value,error",
    [
        ("OPENROUTER_MODEL", "", "MODEL"),
        ("OPENROUTER_TIMEOUT_SECONDS", "0", "TIMEOUT"),
        ("OPENROUTER_REASONING_EFFORT", "maximum", "REASONING"),
        ("OPENROUTER_FALLBACK_MODEL", "", "FALLBACK_MODEL"),
        ("OPENROUTER_FALLBACK_TIMEOUT_SECONDS", "0", "FALLBACK_TIMEOUT"),
        (
            "OPENROUTER_FALLBACK_REASONING_EFFORT",
            "maximum",
            "FALLBACK_REASONING",
        ),
        ("OPENROUTER_MAX_TOKENS", "0", "MAX_TOKENS"),
        ("OPENROUTER_MAX_TOKENS", "65537", "MAX_TOKENS"),
        ("GEMINI_MODEL", "", "GEMINI_MODEL"),
        ("GEMINI_TIMEOUT_SECONDS", "0", "GEMINI_TIMEOUT"),
        ("CALENDAR_PLANNER_TIMEOUT_SECONDS", "0", "CALENDAR_PLANNER_TIMEOUT"),
        ("GEMINI_CLI_MODEL", "", "GEMINI_CLI_MODEL"),
    ],
)
def test_invalid_openrouter_configuration_fails_closed(
    monkeypatch, environment, value, error
):
    monkeypatch.setenv(environment, value)

    with pytest.raises(ValueError, match=error):
        Config()


def test_primary_and_fallback_models_must_be_different(monkeypatch):
    monkeypatch.setenv("OPENROUTER_MODEL", "example/same:free")
    monkeypatch.setenv("OPENROUTER_FALLBACK_MODEL", "example/same:free")

    with pytest.raises(ValueError, match="must be different"):
        Config()


def test_planner_deadline_must_cover_every_provider_stage(monkeypatch):
    monkeypatch.setenv("CALENDAR_PLANNER_TIMEOUT_SECONDS", "74")

    with pytest.raises(ValueError, match="must cover all provider stages"):
        Config()


def test_webhook_configuration_requires_plain_https_url():
    with pytest.raises(ValueError, match="plain HTTPS"):
        Config(bot_update_mode="webhook", webhook_public_url="http://example.test/hook")

    with pytest.raises(ValueError, match="plain HTTPS"):
        Config(
            bot_update_mode="webhook",
            webhook_public_url="https://example.test/hook?secret=bad",
        )


def test_externally_managed_webhook_requires_only_a_plain_path():
    config = Config(
        bot_update_mode="webhook",
        webhook_register_with_telegram=False,
        webhook_path_override="/telegram/managed/webhook",
    )

    assert config.webhook_path == "/telegram/managed/webhook"

    with pytest.raises(ValueError, match="plain absolute path"):
        Config(
            bot_update_mode="webhook",
            webhook_register_with_telegram=False,
            webhook_path_override="//example.test/webhook",
        )


def test_owner_ids_can_be_read_from_secret_files(tmp_path, monkeypatch):
    personal = tmp_path / "personal-id"
    work = tmp_path / "work-id"
    personal.write_text("200000001\n", encoding="utf-8")
    work.write_text("200000002\n", encoding="utf-8")
    monkeypatch.delenv("TELEGRAM_PERSONAL_USER_ID")
    monkeypatch.delenv("TELEGRAM_WORK_USER_ID")
    monkeypatch.setenv("TELEGRAM_PERSONAL_USER_ID_FILE", str(personal))
    monkeypatch.setenv("TELEGRAM_WORK_USER_ID_FILE", str(work))

    assert Config().allowed_accounts == (
        (200000001, "personal"),
        (200000002, "work"),
    )


@pytest.mark.parametrize(
    "environment,value",
    [
        ("TELEGRAM_PERSONAL_USER_ID", ""),
        ("TELEGRAM_PERSONAL_USER_ID", "not-a-number"),
        ("TELEGRAM_PERSONAL_USER_ID", "0"),
        ("TELEGRAM_PERSONAL_USER_ID", "-1"),
        ("TELEGRAM_WORK_USER_ID", ""),
        ("TELEGRAM_WORK_USER_ID", "not-a-number"),
    ],
)
def test_owner_ids_fail_closed_without_echoing_values(
    monkeypatch, environment, value
):
    monkeypatch.setenv(environment, value)

    with pytest.raises(ValueError) as captured:
        Config()

    assert value not in str(captured.value) or value == ""


def test_owner_id_sources_must_not_conflict(tmp_path, monkeypatch):
    secret_file = tmp_path / "personal-id"
    secret_file.write_text("200000001\n", encoding="utf-8")
    monkeypatch.setenv("TELEGRAM_PERSONAL_USER_ID_FILE", str(secret_file))

    with pytest.raises(ValueError, match="conflicting sources"):
        Config()


def test_owner_ids_must_be_present_and_distinct(monkeypatch):
    monkeypatch.delenv("TELEGRAM_PERSONAL_USER_ID")
    with pytest.raises(ValueError, match="must be configured"):
        Config()

    monkeypatch.setenv("TELEGRAM_PERSONAL_USER_ID", "100000002")
    with pytest.raises(ValueError, match="must be different"):
        Config()

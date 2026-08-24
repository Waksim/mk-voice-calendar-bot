import pytest

from tg_voice_transcriber_bot.config import PROJECT_ROOT, Config


def test_calendar_mcp_runtime_paths_environment_and_mapping_are_exact():
    config = Config()

    assert config.openrouter_api_key_environment == "OPENROUTER_API_KEY"
    assert config.openrouter_model == "meta/muse-spark-1.2-contributor"
    assert config.openrouter_timeout_seconds == 45
    assert config.openrouter_reasoning_effort == "high"
    assert config.openrouter_max_tokens == 8192
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
    monkeypatch.setenv("OPENROUTER_MAX_TOKENS", "4096")

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
    assert config.openrouter_max_tokens == 4096


@pytest.mark.parametrize(
    "environment,value,error",
    [
        ("OPENROUTER_MODEL", "", "MODEL"),
        ("OPENROUTER_TIMEOUT_SECONDS", "0", "TIMEOUT"),
        ("OPENROUTER_REASONING_EFFORT", "maximum", "REASONING"),
        ("OPENROUTER_MAX_TOKENS", "0", "MAX_TOKENS"),
        ("OPENROUTER_MAX_TOKENS", "65537", "MAX_TOKENS"),
    ],
)
def test_invalid_openrouter_configuration_fails_closed(
    monkeypatch, environment, value, error
):
    monkeypatch.setenv(environment, value)

    with pytest.raises(ValueError, match=error):
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

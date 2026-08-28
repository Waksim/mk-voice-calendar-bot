import pytest

from tg_voice_transcriber_bot.config import PROJECT_ROOT, Config


def test_calendar_mcp_runtime_paths_environment_and_mapping_are_exact():
    config = Config()

    assert config.codex_runner_url == "http://127.0.0.1:8091"
    assert config.codex_runner_token_environment == "CODEX_RUNNER_TOKEN"
    assert config.codex_model == "gpt-5.6-sol"
    assert config.codex_reasoning_effort == "medium"
    assert config.codex_timeout_seconds == 55
    assert config.gigachat_credentials_environment == "GIGACHAT_CREDENTIALS"
    assert config.gigachat_scope == "GIGACHAT_API_CORP"
    assert config.gigachat_model == "GigaChat-2-Max"
    assert config.gigachat_base_url == "https://api.giga.chat/v1"
    assert (
        config.gigachat_auth_url
        == "https://ngw.devices.sberbank.ru:9443/api/v2/oauth"
    )
    assert config.gigachat_ca_bundle_file == (
        PROJECT_ROOT
        / "deploy"
        / "server"
        / "certs"
        / "russian_trusted_root_ca_pem.crt"
    )
    assert config.gigachat_timeout_seconds == 45
    assert config.openrouter_api_key_environment == "OPENROUTER_API_KEY"
    assert config.openrouter_model == "nvidia/nemotron-3-super-120b-a12b:free"
    assert config.openrouter_timeout_seconds == 35
    assert config.openrouter_reasoning_effort == "medium"
    assert config.openrouter_fallback_model == "z-ai/glm-5.2:free"
    assert config.openrouter_fallback_timeout_seconds == 15
    assert config.openrouter_fallback_reasoning_effort == "high"
    assert config.openrouter_max_tokens == 8192
    assert config.openrouter_vision_model == "google/gemma-4-31b-it:free"
    assert config.openrouter_vision_timeout_seconds == 15
    assert (
        config.openrouter_vision_fallback_model
        == "google/gemma-4-26b-a4b-it:free"
    )
    assert config.openrouter_vision_fallback_timeout_seconds == 12
    assert config.gemini_keychain_account == "codex.gemini.mk_voice_calendar_bot"
    assert config.gemini_keychain_service == "mk_voice_calendar_bot"
    assert config.gemini_api_key_environment == "GEMINI_API_KEY"
    assert config.gemini_model == "gemini-3.7-flash"
    assert config.gemini_timeout_seconds == 25
    assert config.gemini_vision_model == "gemini-3.7-flash"
    assert config.gemini_vision_timeout_seconds == 20
    assert config.vision_local_ocr_timeout_seconds == 15
    assert config.vision_max_image_bytes == 8 * 1024 * 1024
    assert config.vision_max_image_pixels == 20_000_000
    assert config.vision_max_description_chars == 4_000
    assert config.vision_max_visible_text_chars == 12_000
    assert config.vision_ocr_model_dir == (
        PROJECT_ROOT / ".runtime" / "rapidocr-models"
    )
    assert config.calendar_planner_timeout_seconds == 180
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
    monkeypatch.setenv("CODEX_RUNNER_URL", "http://localhost:18091")
    monkeypatch.setenv("CODEX_MODEL", "gpt-5.6-luna-test")
    monkeypatch.setenv("CODEX_REASONING_EFFORT", "xhigh")
    monkeypatch.setenv("CODEX_TIMEOUT_SECONDS", "40")
    monkeypatch.setenv("GIGACHAT_SCOPE", "GIGACHAT_API_B2B")
    monkeypatch.setenv("GIGACHAT_MODEL", "GigaChat-test")
    monkeypatch.setenv("GIGACHAT_BASE_URL", "https://giga.example.test/v1")
    monkeypatch.setenv("GIGACHAT_AUTH_URL", "https://auth.example.test/oauth")
    monkeypatch.setenv("GIGACHAT_CA_BUNDLE_FILE", str(tmp_path / "giga-ca.crt"))
    monkeypatch.setenv("GIGACHAT_TIMEOUT_SECONDS", "50")
    monkeypatch.setenv("OPENROUTER_MODEL", "meta/muse-spark-1.2")
    monkeypatch.setenv("OPENROUTER_TIMEOUT_SECONDS", "60")
    monkeypatch.setenv("OPENROUTER_REASONING_EFFORT", "medium")
    monkeypatch.setenv("OPENROUTER_FALLBACK_MODEL", "z-ai/glm-5.2:free")
    monkeypatch.setenv("OPENROUTER_FALLBACK_TIMEOUT_SECONDS", "20")
    monkeypatch.setenv("OPENROUTER_FALLBACK_REASONING_EFFORT", "low")
    monkeypatch.setenv("OPENROUTER_MAX_TOKENS", "4096")
    monkeypatch.setenv("OPENROUTER_VISION_MODEL", "example/vision-primary:free")
    monkeypatch.setenv("OPENROUTER_VISION_TIMEOUT_SECONDS", "31")
    monkeypatch.setenv(
        "OPENROUTER_VISION_FALLBACK_MODEL", "example/vision-fallback:free"
    )
    monkeypatch.setenv("OPENROUTER_VISION_FALLBACK_TIMEOUT_SECONDS", "29")
    monkeypatch.setenv("GEMINI_MODEL", "gemini-test-model")
    monkeypatch.setenv("GEMINI_TIMEOUT_SECONDS", "30")
    monkeypatch.setenv("GEMINI_VISION_MODEL", "gemini-test-vision-model")
    monkeypatch.setenv("GEMINI_VISION_TIMEOUT_SECONDS", "41")
    monkeypatch.setenv("VISION_LOCAL_OCR_TIMEOUT_SECONDS", "19")
    monkeypatch.setenv("VISION_MAX_IMAGE_BYTES", "7340032")
    monkeypatch.setenv("VISION_MAX_IMAGE_PIXELS", "19000000")
    monkeypatch.setenv("VISION_MAX_DESCRIPTION_CHARS", "3900")
    monkeypatch.setenv("VISION_MAX_VISIBLE_TEXT_CHARS", "11000")
    monkeypatch.setenv("VISION_OCR_MODEL_DIR", str(tmp_path / "ocr-models"))
    monkeypatch.setenv("CALENDAR_PLANNER_TIMEOUT_SECONDS", "205")
    monkeypatch.setenv("GEMINI_CLI_MODEL", "gemini-cli-test-model")

    config = Config()

    assert config.discover_gateway_launcher() == launcher
    assert config.bot_update_mode == "webhook"
    assert config.webhook_path == "/telegram/managed/webhook"
    assert config.webhook_register_with_telegram is False
    assert config.webhook_listen_host == "0.0.0.0"
    assert config.webhook_listen_port == 8090
    assert config.calendar_mcp_env["PATH"] == "/usr/local/bin:/usr/bin:/bin"
    assert config.codex_runner_url == "http://localhost:18091"
    assert config.codex_model == "gpt-5.6-luna-test"
    assert config.codex_reasoning_effort == "xhigh"
    assert config.codex_timeout_seconds == 40
    assert config.gigachat_scope == "GIGACHAT_API_B2B"
    assert config.gigachat_model == "GigaChat-test"
    assert config.gigachat_base_url == "https://giga.example.test/v1"
    assert config.gigachat_auth_url == "https://auth.example.test/oauth"
    assert config.gigachat_ca_bundle_file == tmp_path / "giga-ca.crt"
    assert config.gigachat_timeout_seconds == 50
    assert config.openrouter_model == "meta/muse-spark-1.2"
    assert config.openrouter_timeout_seconds == 60
    assert config.openrouter_reasoning_effort == "medium"
    assert config.openrouter_fallback_model == "z-ai/glm-5.2:free"
    assert config.openrouter_fallback_timeout_seconds == 20
    assert config.openrouter_fallback_reasoning_effort == "low"
    assert config.openrouter_max_tokens == 4096
    assert config.openrouter_vision_model == "example/vision-primary:free"
    assert config.openrouter_vision_timeout_seconds == 31
    assert config.openrouter_vision_fallback_model == "example/vision-fallback:free"
    assert config.openrouter_vision_fallback_timeout_seconds == 29
    assert config.gemini_model == "gemini-test-model"
    assert config.gemini_timeout_seconds == 30
    assert config.gemini_vision_model == "gemini-test-vision-model"
    assert config.gemini_vision_timeout_seconds == 41
    assert config.vision_local_ocr_timeout_seconds == 19
    assert config.vision_max_image_bytes == 7 * 1024 * 1024
    assert config.vision_max_image_pixels == 19_000_000
    assert config.vision_max_description_chars == 3_900
    assert config.vision_max_visible_text_chars == 11_000
    assert config.vision_ocr_model_dir == tmp_path / "ocr-models"
    assert config.calendar_planner_timeout_seconds == 205
    assert config.gemini_cli_model == "gemini-cli-test-model"


@pytest.mark.parametrize(
    "environment,value,error",
    [
        ("CODEX_RUNNER_URL", "https://runner.example.test", "loopback HTTP"),
        ("CODEX_RUNNER_URL", "http://localhost:abc", "loopback HTTP"),
        ("CODEX_RUNNER_URL", "http://localhost:65536", "loopback HTTP"),
        ("CODEX_MODEL", "", "CODEX_MODEL"),
        ("CODEX_REASONING_EFFORT", "ultra", "CODEX_REASONING"),
        ("CODEX_TIMEOUT_SECONDS", "0", "CODEX_TIMEOUT"),
        ("GIGACHAT_SCOPE", "invalid", "GIGACHAT_SCOPE"),
        ("GIGACHAT_MODEL", "", "GIGACHAT_MODEL"),
        ("GIGACHAT_TIMEOUT_SECONDS", "0", "GIGACHAT_TIMEOUT"),
        ("GIGACHAT_BASE_URL", "http://giga.test/v1", "plain HTTPS"),
        ("GIGACHAT_AUTH_URL", "https://user@giga.test/oauth", "plain HTTPS"),
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
        ("OPENROUTER_VISION_MODEL", "", "VISION_MODEL"),
        ("OPENROUTER_VISION_TIMEOUT_SECONDS", "0", "VISION_TIMEOUT"),
        ("OPENROUTER_VISION_TIMEOUT_SECONDS", "301", "VISION_TIMEOUT"),
        (
            "OPENROUTER_VISION_FALLBACK_MODEL",
            "",
            "VISION_FALLBACK_MODEL",
        ),
        (
            "OPENROUTER_VISION_FALLBACK_TIMEOUT_SECONDS",
            "0",
            "VISION_FALLBACK_TIMEOUT",
        ),
        ("GEMINI_MODEL", "", "GEMINI_MODEL"),
        ("GEMINI_TIMEOUT_SECONDS", "0", "GEMINI_TIMEOUT"),
        ("GEMINI_VISION_MODEL", "", "GEMINI_VISION_MODEL"),
        ("GEMINI_VISION_TIMEOUT_SECONDS", "0", "GEMINI_VISION_TIMEOUT"),
        ("VISION_LOCAL_OCR_TIMEOUT_SECONDS", "0", "LOCAL_OCR_TIMEOUT"),
        ("VISION_MAX_IMAGE_BYTES", "0", "MAX_IMAGE_BYTES"),
        ("VISION_MAX_IMAGE_BYTES", "20971521", "MAX_IMAGE_BYTES"),
        ("VISION_MAX_IMAGE_PIXELS", "0", "MAX_IMAGE_PIXELS"),
        ("VISION_MAX_IMAGE_PIXELS", "100000001", "MAX_IMAGE_PIXELS"),
        ("VISION_MAX_DESCRIPTION_CHARS", "0", "MAX_DESCRIPTION_CHARS"),
        ("VISION_MAX_DESCRIPTION_CHARS", "4001", "MAX_DESCRIPTION_CHARS"),
        ("VISION_MAX_VISIBLE_TEXT_CHARS", "0", "MAX_VISIBLE_TEXT_CHARS"),
        ("VISION_MAX_VISIBLE_TEXT_CHARS", "16001", "MAX_VISIBLE_TEXT_CHARS"),
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


def test_openrouter_vision_models_must_be_different(monkeypatch):
    monkeypatch.setenv("OPENROUTER_VISION_MODEL", "example/same:free")
    monkeypatch.setenv("OPENROUTER_VISION_FALLBACK_MODEL", "example/same:free")

    with pytest.raises(ValueError, match="vision models must be different"):
        Config()


def test_planner_deadline_must_cover_every_provider_stage(monkeypatch):
    monkeypatch.setenv("CALENDAR_PLANNER_TIMEOUT_SECONDS", "174")

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

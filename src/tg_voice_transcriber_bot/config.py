"""Non-secret service configuration with portable environment overrides."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlsplit

PROJECT_ROOT = Path(__file__).resolve().parents[2]
_MAX_USER_ID_FILE_BYTES = 128
_MAX_TELEGRAM_USER_ID = (1 << 63) - 1
_MAX_VISION_IMAGE_BYTES = 20 * 1024 * 1024
_MAX_VISION_IMAGE_PIXELS = 100_000_000
_MAX_VISION_DESCRIPTION_CHARS = 4_000
_MAX_VISION_VISIBLE_TEXT_CHARS = 16_000
_MAX_VISION_TIMEOUT_SECONDS = 300


def _environment_path(name: str, default: Path) -> Path:
    value = os.environ.get(name)
    return Path(value).expanduser() if value else default


def _optional_environment_path(name: str) -> Path | None:
    value = os.environ.get(name)
    return Path(value).expanduser() if value else None


def _environment_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer") from exc


def _environment_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean")


def _environment_user_id(name: str) -> int:
    """Read one owner ID from ``NAME`` or ``NAME_FILE`` without leaking it."""

    direct = os.environ.get(name)
    file_name = os.environ.get(f"{name}_FILE")
    if direct is not None and file_name is not None:
        raise ValueError(f"{name} has conflicting sources")

    if file_name is not None:
        path = Path(file_name).expanduser()
        try:
            metadata = path.stat()
            if not path.is_file() or metadata.st_size > _MAX_USER_ID_FILE_BYTES:
                raise OSError
            raw = path.read_text(encoding="utf-8").strip()
        except (OSError, UnicodeError):
            raise ValueError(f"{name}_FILE cannot be read") from None
    elif direct is not None:
        raw = direct.strip()
    else:
        raise ValueError(f"{name} must be configured")

    if not raw or not raw.isascii() or not raw.isdecimal():
        raise ValueError(f"{name} must be a positive decimal integer")
    value = int(raw)
    if not 0 < value <= _MAX_TELEGRAM_USER_ID:
        raise ValueError(f"{name} is outside the supported range")
    return value


def _allowed_accounts_from_environment() -> tuple[tuple[int, str], ...]:
    personal_id = _environment_user_id("TELEGRAM_PERSONAL_USER_ID")
    work_id = _environment_user_id("TELEGRAM_WORK_USER_ID")
    if personal_id == work_id:
        raise ValueError("Telegram owner IDs must be different")
    return ((personal_id, "personal"), (work_id, "work"))


@dataclass(frozen=True)
class Config:
    bot_username: str = "mk_voice_text_bot"
    bot_keychain_account: str = "mk_voice_text_bot"
    bot_keychain_service: str = "codex.telegram.mk_voice_text_bot"
    bot_token_environment: str = "TELEGRAM_BOT_TOKEN"
    state_path: Path = field(
        default_factory=lambda: _environment_path(
            "TELEGRAM_BOT_STATE_PATH", PROJECT_ROOT / ".runtime" / "state.json"
        )
    )
    confirmation_state_path: Path = field(
        default_factory=lambda: _environment_path(
            "TELEGRAM_BOT_CONFIRMATION_STATE_PATH",
            PROJECT_ROOT / ".runtime" / "calendar-confirmations.json",
        )
    )
    operation_state_path: Path = field(
        default_factory=lambda: _environment_path(
            "TELEGRAM_BOT_OPERATION_STATE_PATH",
            PROJECT_ROOT / ".runtime" / "calendar-operations.json",
        )
    )
    gateway_cache_root: Path = field(
        default_factory=lambda: _environment_path(
            "TELEGRAM_GATEWAY_CACHE_ROOT",
            Path.home()
            / ".codex"
            / "plugins"
            / "cache"
            / "personal"
            / "telegram-agent",
        )
    )
    gateway_launcher_path: Path | None = field(
        default_factory=lambda: _optional_environment_path(
            "TELEGRAM_GATEWAY_LAUNCHER"
        )
    )
    gateway_call_timeout_seconds: int = 30
    transcription_timeout_seconds: int = 180
    codex_runner_url: str = field(
        default_factory=lambda: os.environ.get(
            "CODEX_RUNNER_URL", "http://127.0.0.1:8091"
        ).strip()
    )
    codex_runner_token_environment: str = "CODEX_RUNNER_TOKEN"
    codex_model: str = field(
        default_factory=lambda: os.environ.get(
            "CODEX_MODEL", "gpt-5.6-sol"
        ).strip()
    )
    codex_reasoning_effort: str = field(
        default_factory=lambda: os.environ.get(
            "CODEX_REASONING_EFFORT", "medium"
        ).strip().lower()
    )
    codex_timeout_seconds: int = field(
        default_factory=lambda: _environment_int("CODEX_TIMEOUT_SECONDS", 55)
    )
    gigachat_keychain_account: str = "codex.gigachat.mk_voice_calendar_bot"
    gigachat_keychain_service: str = "mk_voice_calendar_bot"
    gigachat_credentials_environment: str = "GIGACHAT_CREDENTIALS"
    gigachat_scope: str = field(
        default_factory=lambda: os.environ.get(
            "GIGACHAT_SCOPE", "GIGACHAT_API_CORP"
        ).strip()
    )
    gigachat_model: str = field(
        default_factory=lambda: os.environ.get(
            "GIGACHAT_MODEL", "GigaChat-2-Max"
        ).strip()
    )
    gigachat_base_url: str = field(
        default_factory=lambda: os.environ.get(
            "GIGACHAT_BASE_URL", "https://api.giga.chat/v1"
        ).strip()
    )
    gigachat_auth_url: str = field(
        default_factory=lambda: os.environ.get(
            "GIGACHAT_AUTH_URL",
            "https://ngw.devices.sberbank.ru:9443/api/v2/oauth",
        ).strip()
    )
    gigachat_ca_bundle_file: Path = field(
        default_factory=lambda: _environment_path(
            "GIGACHAT_CA_BUNDLE_FILE",
            PROJECT_ROOT
            / "deploy"
            / "server"
            / "certs"
            / "russian_trusted_root_ca_pem.crt",
        ).resolve()
    )
    gigachat_timeout_seconds: int = field(
        default_factory=lambda: _environment_int("GIGACHAT_TIMEOUT_SECONDS", 45)
    )
    openrouter_keychain_account: str = "codex.openrouter.mk_voice_calendar_bot"
    openrouter_keychain_service: str = "mk_voice_calendar_bot"
    openrouter_api_key_environment: str = "OPENROUTER_API_KEY"
    openrouter_model: str = field(
        default_factory=lambda: os.environ.get(
            "OPENROUTER_MODEL", "nvidia/nemotron-3-super-120b-a12b:free"
        ).strip()
    )
    openrouter_timeout_seconds: int = field(
        default_factory=lambda: _environment_int("OPENROUTER_TIMEOUT_SECONDS", 35)
    )
    openrouter_reasoning_effort: str = field(
        default_factory=lambda: os.environ.get(
            "OPENROUTER_REASONING_EFFORT", "medium"
        ).strip().lower()
    )
    openrouter_fallback_model: str = field(
        default_factory=lambda: os.environ.get(
            "OPENROUTER_FALLBACK_MODEL", "z-ai/glm-5.2:free"
        ).strip()
    )
    openrouter_fallback_timeout_seconds: int = field(
        default_factory=lambda: _environment_int(
            "OPENROUTER_FALLBACK_TIMEOUT_SECONDS", 15
        )
    )
    openrouter_fallback_reasoning_effort: str = field(
        default_factory=lambda: os.environ.get(
            "OPENROUTER_FALLBACK_REASONING_EFFORT", "high"
        ).strip().lower()
    )
    openrouter_max_tokens: int = field(
        default_factory=lambda: _environment_int("OPENROUTER_MAX_TOKENS", 8192)
    )
    openrouter_vision_model: str = field(
        default_factory=lambda: os.environ.get(
            "OPENROUTER_VISION_MODEL", "google/gemma-4-31b-it:free"
        ).strip()
    )
    openrouter_vision_timeout_seconds: int = field(
        default_factory=lambda: _environment_int(
            "OPENROUTER_VISION_TIMEOUT_SECONDS", 15
        )
    )
    openrouter_vision_fallback_model: str = field(
        default_factory=lambda: os.environ.get(
            "OPENROUTER_VISION_FALLBACK_MODEL",
            "google/gemma-4-26b-a4b-it:free",
        ).strip()
    )
    openrouter_vision_fallback_timeout_seconds: int = field(
        default_factory=lambda: _environment_int(
            "OPENROUTER_VISION_FALLBACK_TIMEOUT_SECONDS", 12
        )
    )
    gemini_keychain_account: str = "codex.gemini.mk_voice_calendar_bot"
    gemini_keychain_service: str = "mk_voice_calendar_bot"
    gemini_api_key_environment: str = "GEMINI_API_KEY"
    gemini_model: str = field(
        default_factory=lambda: os.environ.get(
            "GEMINI_MODEL", "gemini-3.7-flash"
        ).strip()
    )
    gemini_timeout_seconds: int = field(
        default_factory=lambda: _environment_int("GEMINI_TIMEOUT_SECONDS", 25)
    )
    gemini_vision_model: str = field(
        default_factory=lambda: os.environ.get(
            "GEMINI_VISION_MODEL", "gemini-3.7-flash"
        ).strip()
    )
    gemini_vision_timeout_seconds: int = field(
        default_factory=lambda: _environment_int(
            "GEMINI_VISION_TIMEOUT_SECONDS", 20
        )
    )
    vision_local_ocr_timeout_seconds: int = field(
        default_factory=lambda: _environment_int(
            "VISION_LOCAL_OCR_TIMEOUT_SECONDS", 15
        )
    )
    vision_max_image_bytes: int = field(
        default_factory=lambda: _environment_int(
            "VISION_MAX_IMAGE_BYTES", 8 * 1024 * 1024
        )
    )
    vision_max_image_pixels: int = field(
        default_factory=lambda: _environment_int(
            "VISION_MAX_IMAGE_PIXELS", 20_000_000
        )
    )
    vision_max_description_chars: int = field(
        default_factory=lambda: _environment_int(
            "VISION_MAX_DESCRIPTION_CHARS", 4_000
        )
    )
    vision_max_visible_text_chars: int = field(
        default_factory=lambda: _environment_int(
            "VISION_MAX_VISIBLE_TEXT_CHARS", 12_000
        )
    )
    vision_ocr_model_dir: Path = field(
        default_factory=lambda: _environment_path(
            "VISION_OCR_MODEL_DIR",
            PROJECT_ROOT / ".runtime" / "rapidocr-models",
        ).resolve()
    )
    calendar_planner_timeout_seconds: int = field(
        default_factory=lambda: _environment_int(
            "CALENDAR_PLANNER_TIMEOUT_SECONDS", 180
        )
    )
    gemini_cli_path: Path = field(
        default_factory=lambda: _environment_path(
            "GEMINI_CLI_PATH", Path.home() / ".local" / "bin" / "agy"
        )
    )
    gemini_cli_model: str = field(
        default_factory=lambda: os.environ.get(
            "GEMINI_CLI_MODEL", "gemini-3.7-flash-high"
        ).strip()
    )
    calendar_timezone: str = "Europe/Moscow"
    calendar_mcp_package_version: str = "2.6.2"
    calendar_mcp_working_directory: Path = field(
        default_factory=lambda: _environment_path(
            "CALENDAR_MCP_WORKING_DIRECTORY", PROJECT_ROOT / "calendar-mcp"
        )
    )
    calendar_mcp_binary_path: Path = field(
        default_factory=lambda: _environment_path(
            "CALENDAR_MCP_BINARY_PATH",
            PROJECT_ROOT
            / "calendar-mcp"
            / "node_modules"
            / ".bin"
            / "google-calendar-mcp",
        )
    )
    calendar_mcp_oauth_credentials_path: Path = field(
        default_factory=lambda: _environment_path(
            "GOOGLE_CALENDAR_OAUTH_CREDENTIALS_PATH",
            PROJECT_ROOT / ".runtime" / "google-calendar-oauth.json",
        )
    )
    calendar_mcp_token_path: Path = field(
        default_factory=lambda: _environment_path(
            "GOOGLE_CALENDAR_TOKEN_PATH",
            PROJECT_ROOT / ".runtime" / "google-calendar-tokens.json",
        )
    )
    calendar_mcp_account: str = "owner"
    calendar_mcp_calendar_id: str = "primary"
    calendar_mcp_timeout_seconds: int = 45
    calendar_mcp_process_path: str = field(
        default_factory=lambda: os.environ.get(
            "CALENDAR_MCP_PROCESS_PATH",
            "/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin",
        )
    )

    bot_update_mode: str = field(
        default_factory=lambda: os.environ.get(
            "TELEGRAM_BOT_UPDATE_MODE", "polling"
        ).strip().lower()
    )
    webhook_public_url: str = field(
        default_factory=lambda: os.environ.get("TELEGRAM_WEBHOOK_URL", "").strip()
    )
    webhook_path_override: str = field(
        default_factory=lambda: os.environ.get("TELEGRAM_WEBHOOK_PATH", "").strip()
    )
    webhook_register_with_telegram: bool = field(
        default_factory=lambda: _environment_bool("TELEGRAM_WEBHOOK_REGISTER", True)
    )
    webhook_secret_environment: str = "TELEGRAM_WEBHOOK_SECRET"
    webhook_listen_host: str = field(
        default_factory=lambda: os.environ.get(
            "TELEGRAM_WEBHOOK_LISTEN_HOST", "127.0.0.1"
        ).strip()
    )
    webhook_listen_port: int = field(
        default_factory=lambda: _environment_int("TELEGRAM_WEBHOOK_LISTEN_PORT", 8080)
    )
    webhook_health_path: str = "/healthz"
    webhook_max_body_bytes: int = field(
        default_factory=lambda: _environment_int(
            "TELEGRAM_WEBHOOK_MAX_BODY_BYTES", 1_048_576
        )
    )
    webhook_retry_seconds: int = field(
        default_factory=lambda: _environment_int("TELEGRAM_WEBHOOK_RETRY_SECONDS", 3)
    )
    webhook_completed_ids_limit: int = field(
        default_factory=lambda: _environment_int(
            "TELEGRAM_WEBHOOK_COMPLETED_IDS_LIMIT", 2048
        )
    )

    # Numeric IDs are stable and avoid trusting mutable usernames. Values are
    # supplied only through an ignored local .env or production secret files.
    allowed_accounts: tuple[tuple[int, str], ...] = field(
        default_factory=_allowed_accounts_from_environment
    )

    def __post_init__(self) -> None:
        try:
            parsed_codex_url = urlsplit(self.codex_runner_url)
            codex_hostname = parsed_codex_url.hostname
            codex_port = parsed_codex_url.port
        except ValueError:
            raise ValueError("CODEX_RUNNER_URL must be loopback HTTP") from None
        if (
            parsed_codex_url.scheme != "http"
            or codex_hostname not in {"127.0.0.1", "localhost", "::1"}
            or (codex_port is not None and not 1 <= codex_port <= 65535)
            or parsed_codex_url.username is not None
            or parsed_codex_url.password is not None
            or parsed_codex_url.query
            or parsed_codex_url.fragment
            or parsed_codex_url.path not in {"", "/"}
        ):
            raise ValueError("CODEX_RUNNER_URL must be loopback HTTP")
        if not self.codex_model:
            raise ValueError("CODEX_MODEL must not be empty")
        if self.codex_reasoning_effort not in {
            "low",
            "medium",
            "high",
            "xhigh",
            "max",
        }:
            raise ValueError("CODEX_REASONING_EFFORT is invalid")
        if self.codex_timeout_seconds <= 0:
            raise ValueError("CODEX_TIMEOUT_SECONDS must be positive")
        if self.gigachat_scope not in {
            "GIGACHAT_API_PERS",
            "GIGACHAT_API_B2B",
            "GIGACHAT_API_CORP",
        }:
            raise ValueError("GIGACHAT_SCOPE is invalid")
        if not self.gigachat_model:
            raise ValueError("GIGACHAT_MODEL must not be empty")
        if self.gigachat_timeout_seconds <= 0:
            raise ValueError("GIGACHAT_TIMEOUT_SECONDS must be positive")
        for environment, value in (
            ("GIGACHAT_BASE_URL", self.gigachat_base_url),
            ("GIGACHAT_AUTH_URL", self.gigachat_auth_url),
        ):
            parsed = urlsplit(value)
            if (
                parsed.scheme != "https"
                or not parsed.hostname
                or parsed.username is not None
                or parsed.password is not None
                or parsed.query
                or parsed.fragment
            ):
                raise ValueError(f"{environment} must be a plain HTTPS URL")
        if not self.openrouter_model:
            raise ValueError("OPENROUTER_MODEL must not be empty")
        if self.openrouter_timeout_seconds <= 0:
            raise ValueError("OPENROUTER_TIMEOUT_SECONDS must be positive")
        if self.openrouter_reasoning_effort not in {"low", "medium", "high"}:
            raise ValueError(
                "OPENROUTER_REASONING_EFFORT must be low, medium, or high"
            )
        if not self.openrouter_fallback_model:
            raise ValueError("OPENROUTER_FALLBACK_MODEL must not be empty")
        if self.openrouter_fallback_model == self.openrouter_model:
            raise ValueError("OpenRouter primary and fallback models must be different")
        if self.openrouter_fallback_timeout_seconds <= 0:
            raise ValueError(
                "OPENROUTER_FALLBACK_TIMEOUT_SECONDS must be positive"
            )
        if self.openrouter_fallback_reasoning_effort not in {
            "low",
            "medium",
            "high",
        }:
            raise ValueError(
                "OPENROUTER_FALLBACK_REASONING_EFFORT must be low, medium, or high"
            )
        if not 1 <= self.openrouter_max_tokens <= 65_536:
            raise ValueError(
                "OPENROUTER_MAX_TOKENS must be between 1 and 65536"
            )
        if not self.openrouter_vision_model:
            raise ValueError("OPENROUTER_VISION_MODEL must not be empty")
        if not (
            1
            <= self.openrouter_vision_timeout_seconds
            <= _MAX_VISION_TIMEOUT_SECONDS
        ):
            raise ValueError(
                "OPENROUTER_VISION_TIMEOUT_SECONDS must be between 1 and 300"
            )
        if not self.openrouter_vision_fallback_model:
            raise ValueError("OPENROUTER_VISION_FALLBACK_MODEL must not be empty")
        if self.openrouter_vision_fallback_model == self.openrouter_vision_model:
            raise ValueError("OpenRouter vision models must be different")
        if not (
            1
            <= self.openrouter_vision_fallback_timeout_seconds
            <= _MAX_VISION_TIMEOUT_SECONDS
        ):
            raise ValueError(
                "OPENROUTER_VISION_FALLBACK_TIMEOUT_SECONDS must be between 1 and 300"
            )
        if not self.gemini_model:
            raise ValueError("GEMINI_MODEL must not be empty")
        if self.gemini_timeout_seconds <= 0:
            raise ValueError("GEMINI_TIMEOUT_SECONDS must be positive")
        if not self.gemini_vision_model:
            raise ValueError("GEMINI_VISION_MODEL must not be empty")
        if not 1 <= self.gemini_vision_timeout_seconds <= _MAX_VISION_TIMEOUT_SECONDS:
            raise ValueError(
                "GEMINI_VISION_TIMEOUT_SECONDS must be between 1 and 300"
            )
        if not (
            1
            <= self.vision_local_ocr_timeout_seconds
            <= _MAX_VISION_TIMEOUT_SECONDS
        ):
            raise ValueError(
                "VISION_LOCAL_OCR_TIMEOUT_SECONDS must be between 1 and 300"
            )
        if not 1 <= self.vision_max_image_bytes <= _MAX_VISION_IMAGE_BYTES:
            raise ValueError(
                "VISION_MAX_IMAGE_BYTES must be between 1 and 20971520"
            )
        if not 1 <= self.vision_max_image_pixels <= _MAX_VISION_IMAGE_PIXELS:
            raise ValueError(
                "VISION_MAX_IMAGE_PIXELS must be between 1 and 100000000"
            )
        if not (
            1
            <= self.vision_max_description_chars
            <= _MAX_VISION_DESCRIPTION_CHARS
        ):
            raise ValueError(
                "VISION_MAX_DESCRIPTION_CHARS must be between 1 and 4000"
            )
        if not (
            1
            <= self.vision_max_visible_text_chars
            <= _MAX_VISION_VISIBLE_TEXT_CHARS
        ):
            raise ValueError(
                "VISION_MAX_VISIBLE_TEXT_CHARS must be between 1 and 16000"
            )
        if self.calendar_planner_timeout_seconds <= 0:
            raise ValueError("CALENDAR_PLANNER_TIMEOUT_SECONDS must be positive")
        minimum_chain_timeout = (
            self.codex_timeout_seconds
            + self.gigachat_timeout_seconds
            + self.openrouter_timeout_seconds
            + self.openrouter_fallback_timeout_seconds
            + self.gemini_timeout_seconds
        )
        if self.calendar_planner_timeout_seconds < minimum_chain_timeout:
            raise ValueError(
                "CALENDAR_PLANNER_TIMEOUT_SECONDS must cover all provider stages"
            )
        if not self.gemini_cli_model:
            raise ValueError("GEMINI_CLI_MODEL must not be empty")
        if self.bot_update_mode not in {"polling", "webhook"}:
            raise ValueError("TELEGRAM_BOT_UPDATE_MODE must be polling or webhook")
        if not self.webhook_listen_host:
            raise ValueError("webhook listen host must not be empty")
        if not 1 <= self.webhook_listen_port <= 65535:
            raise ValueError("webhook listen port must be between 1 and 65535")
        if self.webhook_max_body_bytes <= 0:
            raise ValueError("webhook body limit must be positive")
        if self.webhook_retry_seconds <= 0:
            raise ValueError("webhook retry interval must be positive")
        if self.webhook_completed_ids_limit <= 0:
            raise ValueError("completed webhook ID limit must be positive")
        if self.bot_update_mode == "webhook":
            if self.webhook_register_with_telegram:
                parsed = urlsplit(self.webhook_public_url)
                if (
                    parsed.scheme != "https"
                    or not parsed.hostname
                    or parsed.username is not None
                    or parsed.password is not None
                    or parsed.query
                    or parsed.fragment
                    or not parsed.path.startswith("/")
                ):
                    raise ValueError("TELEGRAM_WEBHOOK_URL must be a plain HTTPS URL")
            path = self.webhook_path
            parsed_path = urlsplit(path)
            if (
                not path.startswith("/")
                or path.startswith("//")
                or parsed_path.path != path
                or parsed_path.scheme
                or parsed_path.netloc
                or parsed_path.query
                or parsed_path.fragment
            ):
                raise ValueError("TELEGRAM_WEBHOOK_PATH must be a plain absolute path")

    @property
    def bot_chat_id(self) -> str:
        return f"@{self.bot_username}"

    @property
    def webhook_path(self) -> str:
        if self.webhook_path_override:
            return self.webhook_path_override
        if not self.webhook_public_url:
            raise RuntimeError("Telegram webhook URL or path is not configured")
        return urlsplit(self.webhook_public_url).path or "/"

    @property
    def account_by_user_id(self) -> dict[int, str]:
        return dict(self.allowed_accounts)

    @property
    def expected_user_id_by_account(self) -> dict[str, int]:
        return {account: user_id for user_id, account in self.allowed_accounts}

    @property
    def calendar_mcp_account_mapping(self) -> dict[str, str]:
        return {
            logical_account: self.calendar_mcp_account
            for _user_id, logical_account in self.allowed_accounts
        }

    @property
    def calendar_mcp_env(self) -> dict[str, str]:
        # The child receives only file paths and a non-secret account nickname;
        # OAuth client data and tokens remain inside the referenced 0600 files.
        return {
            "GOOGLE_OAUTH_CREDENTIALS": str(
                self.calendar_mcp_oauth_credentials_path
            ),
            "GOOGLE_CALENDAR_MCP_TOKEN_PATH": str(self.calendar_mcp_token_path),
            "GOOGLE_ACCOUNT_MODE": self.calendar_mcp_account,
            # The pinned package launcher uses ``#!/usr/bin/env node``. Keep a
            # deliberately small path that works in both Linux containers and
            # the existing Homebrew-based macOS LaunchAgent.
            "PATH": self.calendar_mcp_process_path,
            "TRANSPORT": "stdio",
            "DEBUG": "false",
        }

    def discover_gateway_launcher(self) -> Path:
        if self.gateway_launcher_path is not None:
            launcher = self.gateway_launcher_path.expanduser()
            if not launcher.is_file():
                raise RuntimeError("Configured Telegram gateway launcher is unavailable")
            return launcher
        candidates = sorted(
            self.gateway_cache_root.glob("*/scripts/telegram-gateway"),
            key=lambda path: path.parents[1].name,
            reverse=True,
        )
        if not candidates:
            raise RuntimeError("Telegram gateway launcher is not installed")
        return candidates[0]

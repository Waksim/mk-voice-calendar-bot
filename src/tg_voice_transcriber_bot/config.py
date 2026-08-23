"""Non-secret service configuration with portable environment overrides."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlsplit

PROJECT_ROOT = Path(__file__).resolve().parents[2]
_MAX_USER_ID_FILE_BYTES = 128
_MAX_TELEGRAM_USER_ID = (1 << 63) - 1


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
    gemini_keychain_account: str = "codex.gemini.mk_voice_calendar_bot"
    gemini_keychain_service: str = "mk_voice_calendar_bot"
    gemini_api_key_environment: str = "GEMINI_API_KEY"
    gemini_model: str = "gemini-3.7-flash"
    gemini_timeout_seconds: int = 90
    gemini_cli_path: Path = field(
        default_factory=lambda: _environment_path(
            "GEMINI_CLI_PATH", Path.home() / ".local" / "bin" / "agy"
        )
    )
    gemini_cli_model: str = "gemini-3.7-flash-high"
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

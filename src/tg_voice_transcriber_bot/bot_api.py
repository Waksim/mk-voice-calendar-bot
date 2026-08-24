"""Minimal Telegram Bot API client with token-safe errors."""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import httpx

from .text import telegram_text_chunks, utf16_units


class BotApiError(RuntimeError):
    def __init__(self, message: str, *, retry_after: int | None = None) -> None:
        super().__init__(message)
        self.retry_after = retry_after


class _UnsetType:
    """Sentinel used when an existing inline keyboard must be preserved."""

    __slots__ = ()


UNSET = _UnsetType()

_SECRET_ENVIRONMENT_RE = re.compile(r"^[A-Z][A-Z0-9_]*$")
_WEBHOOK_SECRET_RE = re.compile(r"^[A-Za-z0-9_-]{1,256}$")
_MAX_SECRET_FILE_BYTES = 65_536


def read_keychain_secret(*, account: str, service: str) -> str:
    try:
        completed = subprocess.run(
            [
                "/usr/bin/security",
                "find-generic-password",
                "-a",
                account,
                "-s",
                service,
                "-w",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError("Cannot read secret from macOS Keychain") from exc
    secret = completed.stdout.strip()
    if not secret:
        raise RuntimeError("Keychain secret is empty")
    return secret


def read_secret(
    *,
    environment: str,
    account: str | None = None,
    service: str | None = None,
) -> str:
    """Read a secret from ``NAME_FILE``, ``NAME``, or macOS Keychain.

    File-based secrets are preferred for Linux containers. The existing
    Keychain lookup remains the final fallback for the macOS LaunchAgent.
    Errors intentionally omit values and subprocess diagnostics.
    """

    if not _SECRET_ENVIRONMENT_RE.fullmatch(environment):
        raise ValueError("Secret environment name is invalid")
    direct = os.environ.get(environment)
    file_name = os.environ.get(f"{environment}_FILE")
    if direct is not None and file_name is not None:
        raise RuntimeError("Secret has conflicting environment sources")

    if file_name is not None:
        path = Path(file_name).expanduser()
        try:
            metadata = path.stat()
            if not path.is_file() or metadata.st_size > _MAX_SECRET_FILE_BYTES:
                raise OSError
            secret = path.read_text(encoding="utf-8").strip()
        except (OSError, UnicodeError):
            raise RuntimeError("Cannot read secret file") from None
        if not secret:
            raise RuntimeError("Secret file is empty")
        return secret

    if direct is not None:
        secret = direct.strip()
        if not secret:
            raise RuntimeError("Environment secret is empty")
        return secret

    if account is None or service is None:
        raise RuntimeError("Secret is unavailable")
    return read_keychain_secret(account=account, service=service)


class BotApi:
    def __init__(self, token: str) -> None:
        self._token = token
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(connect=10, read=55, write=20, pool=10),
            limits=httpx.Limits(max_connections=5, max_keepalive_connections=2),
        )

    async def __aenter__(self) -> "BotApi":
        return self

    async def __aexit__(self, exc_type, exc, traceback) -> None:
        await self._client.aclose()

    async def call(self, method: str, payload: dict[str, Any] | None = None) -> Any:
        url = f"https://api.telegram.org/bot{self._token}/{method}"
        try:
            response = await self._client.post(url, json=payload or {})
        except httpx.HTTPError as exc:
            # HTTPX exceptions may contain the token-bearing URL, so never include
            # their string representation in logs or propagated errors.
            raise BotApiError(
                f"Bot API transport error: {type(exc).__name__}"
            ) from None
        try:
            body = response.json()
        except ValueError as exc:
            if response.status_code != 200:
                raise BotApiError(
                    f"Bot API HTTP status {response.status_code}"
                ) from None
            raise BotApiError("Bot API returned invalid JSON") from exc
        if not body.get("ok"):
            code = body.get("error_code", "unknown")
            description = str(body.get("description", "request failed"))
            parameters = body.get("parameters") or {}
            retry_after = parameters.get("retry_after")
            raise BotApiError(
                f"Bot API error {code}: {description}",
                retry_after=(int(retry_after) if retry_after is not None else None),
            )
        if response.status_code != 200:
            raise BotApiError(f"Bot API HTTP status {response.status_code}")
        return body.get("result")

    async def get_updates(self, offset: int) -> list[dict[str, Any]]:
        payload: dict[str, Any] = {
            "timeout": 45,
            "allowed_updates": ["message", "callback_query"],
        }
        if offset:
            payload["offset"] = offset
        result = await self.call("getUpdates", payload)
        return result if isinstance(result, list) else []

    async def send_chat_action(self, chat_id: int, action: str = "typing") -> None:
        await self.call("sendChatAction", {"chat_id": chat_id, "action": action})

    async def send_text(
        self,
        chat_id: int,
        text: str,
        *,
        reply_to_message_id: int | None = None,
        reply_markup: dict[str, Any] | None = None,
    ) -> None:
        if reply_markup is not None:
            self._validate_reply_markup(reply_markup)
        chunks = telegram_text_chunks(text)
        for index, chunk in enumerate(chunks):
            payload: dict[str, Any] = {"chat_id": chat_id, "text": chunk}
            if index == 0 and reply_to_message_id is not None:
                payload["reply_parameters"] = {
                    "message_id": reply_to_message_id,
                    "allow_sending_without_reply": True,
                }
            # Telegram attaches a keyboard to one message, not a logical text
            # spanning several sendMessage calls.  Keep it on the first (and in
            # the usual case, only) chunk and never duplicate active buttons.
            if index == 0 and reply_markup is not None:
                payload["reply_markup"] = reply_markup
            await self.call("sendMessage", payload)

    async def send_html(
        self,
        chat_id: int,
        html_text: str,
        *,
        reply_to_message_id: int | None = None,
        reply_markup: dict[str, Any] | None = None,
    ) -> int:
        """Send one pre-rendered HTML message and return its message ID.

        HTML is deliberately not passed through :func:`telegram_text_chunks`:
        splitting an encoded entity or a formatting tag would make both chunks
        invalid.  Callers must therefore render a bounded, single-message card.
        """
        if chat_id <= 0:
            raise ValueError("chat ID must be positive")
        if not isinstance(html_text, str) or not html_text:
            raise ValueError("HTML text must be a non-empty string")
        if reply_markup is not None:
            self._validate_reply_markup(reply_markup)

        payload: dict[str, Any] = {
            "chat_id": chat_id,
            "text": html_text,
            "parse_mode": "HTML",
            "link_preview_options": {"is_disabled": True},
        }
        if reply_to_message_id is not None:
            if reply_to_message_id <= 0:
                raise ValueError("reply message ID must be positive")
            payload["reply_parameters"] = {
                "message_id": reply_to_message_id,
                "allow_sending_without_reply": True,
            }
        if reply_markup is not None:
            payload["reply_markup"] = reply_markup

        result = await self.call("sendMessage", payload)
        if not isinstance(result, dict):
            raise BotApiError("Bot API returned an invalid sent message")
        message_id = result.get("message_id")
        if (
            isinstance(message_id, bool)
            or not isinstance(message_id, int)
            or message_id <= 0
        ):
            raise BotApiError("Bot API returned an invalid sent message")
        return message_id

    async def edit_html(
        self,
        chat_id: int,
        message_id: int,
        html_text: str,
        *,
        reply_markup: dict[str, Any] | None | _UnsetType = UNSET,
    ) -> None:
        """Edit one HTML message, optionally preserving or replacing its keyboard.

        ``UNSET`` omits ``reply_markup`` and preserves the current keyboard,
        ``None`` clears it, and a mapping replaces it.
        """
        if chat_id <= 0 or message_id <= 0:
            raise ValueError("chat and message IDs must be positive")
        if not isinstance(html_text, str) or not html_text:
            raise ValueError("HTML text must be a non-empty string")

        payload: dict[str, Any] = {
            "chat_id": chat_id,
            "message_id": message_id,
            "text": html_text,
            "parse_mode": "HTML",
            "link_preview_options": {"is_disabled": True},
        }
        if reply_markup is None:
            payload["reply_markup"] = {"inline_keyboard": []}
        elif isinstance(reply_markup, dict):
            self._validate_reply_markup(reply_markup)
            payload["reply_markup"] = reply_markup
        elif reply_markup is not UNSET:
            raise TypeError("reply_markup must be UNSET, None, or a mapping")

        await self.call("editMessageText", payload)

    async def answer_callback_query(
        self,
        callback_query_id: str,
        text: str,
        *,
        show_alert: bool = False,
    ) -> None:
        if not callback_query_id:
            raise ValueError("callback query ID is required")
        if utf16_units(text) > 200:
            raise ValueError("callback answer exceeds Telegram's 200-character limit")
        await self.call(
            "answerCallbackQuery",
            {
                "callback_query_id": callback_query_id,
                "text": text,
                "show_alert": show_alert,
            },
        )

    async def remove_inline_keyboard(self, chat_id: int, message_id: int) -> None:
        if chat_id <= 0 or message_id <= 0:
            raise ValueError("chat and message IDs must be positive")
        await self.call(
            "editMessageReplyMarkup",
            {
                "chat_id": chat_id,
                "message_id": message_id,
                "reply_markup": {"inline_keyboard": []},
            },
        )

    @staticmethod
    def _validate_reply_markup(reply_markup: dict[str, Any]) -> None:
        keyboard = reply_markup.get("inline_keyboard")
        if not isinstance(keyboard, list):
            raise ValueError("reply markup must contain an inline keyboard")
        for row in keyboard:
            if not isinstance(row, list):
                raise ValueError("inline keyboard rows must be arrays")
            for button in row:
                if not isinstance(button, dict):
                    raise ValueError("inline keyboard buttons must be objects")
                callback_data = button.get("callback_data")
                if callback_data is None:
                    continue
                if not isinstance(callback_data, str):
                    raise ValueError("callback_data must be a string")
                try:
                    byte_length = len(callback_data.encode("utf-8"))
                except UnicodeEncodeError as exc:
                    raise ValueError("callback_data must be valid UTF-8") from exc
                if not 1 <= byte_length <= 64:
                    raise ValueError(
                        "callback_data exceeds Telegram's 1-64 byte limit"
                    )

    async def configure_profile(self) -> None:
        """Configure commands and public descriptions without changing transport."""

        await self.call(
            "setMyCommands",
            {
                "commands": [
                    {"command": "start", "description": "Как пользоваться ботом"},
                    {"command": "status", "description": "Проверить доступ аккаунта"},
                ]
            },
        )
        await self.call(
            "setMyDescription",
            {
                "description": (
                    "Пришлите голосовое сообщение или напишите календарную "
                    "команду текстом. Muse Spark 1.2 через OpenRouter "
                    "сразу добавит, "
                    "изменит или удалит событие в Google Calendar. Любое "
                    "действие можно отменить кнопкой. Доступ ограничен двумя "
                    "аккаунтами владельца."
                )
            },
        )
        await self.call(
            "setMyShortDescription",
            {
                "short_description": (
                    "Голос или текст → Muse Spark 1.2 (OpenRouter) "
                    "→ Google Calendar"
                )
            },
        )

    async def configure_polling(self) -> None:
        """Select long polling and configure the bot profile."""

        await self.call("deleteWebhook", {"drop_pending_updates": False})
        await self.configure_profile()

    async def set_webhook(self, url: str, secret_token: str) -> None:
        """Select authenticated webhook delivery without discarding updates."""

        parsed = urlsplit(url)
        if (
            parsed.scheme != "https"
            or not parsed.hostname
            or parsed.username is not None
            or parsed.password is not None
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError("webhook URL must be a plain HTTPS URL")
        if not _WEBHOOK_SECRET_RE.fullmatch(secret_token):
            raise ValueError("webhook secret token has an invalid format")
        await self.call(
            "setWebhook",
            {
                "url": url,
                "secret_token": secret_token,
                "max_connections": 1,
                "allowed_updates": ["message", "callback_query"],
                "drop_pending_updates": False,
            },
        )

    async def configure(self) -> None:
        """Backward-compatible alias for the default polling configuration."""

        await self.configure_polling()

"""Minimal Telegram Bot API client with token-safe errors."""

from __future__ import annotations

import logging
import os
import re
import subprocess
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import httpx

from .text import telegram_text_chunks, utf16_units


LOGGER = logging.getLogger("tg_voice_transcriber_bot.bot_api")


class BotApiError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        retry_after: int | None = None,
        http_status: int | None = None,
        error_code: int | None = None,
    ) -> None:
        super().__init__(message)
        self.retry_after = retry_after
        self.http_status = http_status
        self.error_code = error_code


class BotApiFileError(BotApiError):
    """A permanent Telegram file validation/size failure."""


class _UnsetType:
    """Sentinel used when an existing inline keyboard must be preserved."""

    __slots__ = ()


UNSET = _UnsetType()

_SECRET_ENVIRONMENT_RE = re.compile(r"^[A-Z][A-Z0-9_]*$")
_WEBHOOK_SECRET_RE = re.compile(r"^[A-Za-z0-9_-]{1,256}$")
_BOT_API_METHODS = frozenset(
    {
        "answerCallbackQuery",
        "deleteWebhook",
        "editMessageReplyMarkup",
        "editMessageText",
        "getFile",
        "getMe",
        "getUpdates",
        "sendChatAction",
        "sendMessage",
        "setMyCommands",
        "setMyDescription",
        "setMyShortDescription",
        "setWebhook",
    }
)
_MAX_SECRET_FILE_BYTES = 65_536
_MAX_TELEGRAM_FILE_BYTES = 20 * 1024 * 1024
_TELEGRAM_FILE_PATH_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._/-]{0,1023}\Z")


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
        started = time.monotonic()
        safe_method = (
            method
            if isinstance(method, str) and method in _BOT_API_METHODS
            else "unknown"
        )
        LOGGER.info(
            "Telegram Bot API call started; method=%s status=started",
            safe_method,
        )
        url = f"https://api.telegram.org/bot{self._token}/{method}"
        try:
            response = await self._client.post(url, json=payload or {})
        except httpx.HTTPError as exc:
            # HTTPX exceptions may contain the token-bearing URL, so never include
            # their string representation in logs or propagated errors.
            LOGGER.warning(
                "Telegram Bot API call finished; method=%s "
                "status=transport_error elapsed=%.3fs error_type=%s",
                safe_method,
                time.monotonic() - started,
                type(exc).__name__,
            )
            raise BotApiError(
                f"Bot API transport error: {type(exc).__name__}"
            ) from None
        try:
            body = response.json()
        except ValueError as exc:
            LOGGER.warning(
                "Telegram Bot API call finished; method=%s "
                "status=invalid_response http_status=%d elapsed=%.3fs",
                safe_method,
                response.status_code,
                time.monotonic() - started,
            )
            if response.status_code != 200:
                raise BotApiError(
                    f"Bot API HTTP status {response.status_code}",
                    http_status=response.status_code,
                ) from None
            raise BotApiError("Bot API returned invalid JSON") from exc
        if not isinstance(body, dict):
            LOGGER.warning(
                "Telegram Bot API call finished; method=%s "
                "status=invalid_response http_status=%d elapsed=%.3fs",
                safe_method,
                response.status_code,
                time.monotonic() - started,
            )
            raise BotApiError("Bot API returned an invalid response")
        if not body.get("ok"):
            code = body.get("error_code", "unknown")
            description = str(body.get("description", "request failed"))
            parameters = body.get("parameters") or {}
            retry_after = parameters.get("retry_after")
            safe_code = (
                code if isinstance(code, int) and not isinstance(code, bool) else "unknown"
            )
            LOGGER.warning(
                "Telegram Bot API call finished; method=%s status=api_error "
                "http_status=%d error_code=%s elapsed=%.3fs",
                safe_method,
                response.status_code,
                safe_code,
                time.monotonic() - started,
            )
            raise BotApiError(
                f"Bot API error {code}: {description}",
                retry_after=(int(retry_after) if retry_after is not None else None),
                http_status=response.status_code,
                error_code=(safe_code if isinstance(safe_code, int) else None),
            )
        if response.status_code != 200:
            LOGGER.warning(
                "Telegram Bot API call finished; method=%s status=http_error "
                "http_status=%d elapsed=%.3fs",
                safe_method,
                response.status_code,
                time.monotonic() - started,
            )
            raise BotApiError(
                f"Bot API HTTP status {response.status_code}",
                http_status=response.status_code,
            )
        LOGGER.info(
            "Telegram Bot API call finished; method=%s status=success "
            "http_status=%d elapsed=%.3fs",
            safe_method,
            response.status_code,
            time.monotonic() - started,
        )
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

    @staticmethod
    def _validate_opaque_file_id(file_id: str) -> str:
        if (
            not isinstance(file_id, str)
            or not 1 <= len(file_id) <= 1024
            or any(ord(character) < 0x21 or ord(character) > 0x7E for character in file_id)
        ):
            raise ValueError("Telegram file ID is invalid")
        return file_id

    @staticmethod
    def _validate_file_size_limit(value: int, *, field: str) -> int:
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or not 1 <= value <= _MAX_TELEGRAM_FILE_BYTES
        ):
            raise ValueError(
                f"{field} must be between 1 and {_MAX_TELEGRAM_FILE_BYTES} bytes"
            )
        return value

    @staticmethod
    def _validate_telegram_file_path(file_path: str) -> str:
        if (
            not isinstance(file_path, str)
            or not _TELEGRAM_FILE_PATH_RE.fullmatch(file_path)
            or file_path.startswith("/")
            or "//" in file_path
            or any(segment in {"", ".", ".."} for segment in file_path.split("/"))
        ):
            raise ValueError("Telegram file path is invalid")
        return file_path

    async def get_file(
        self,
        file_id: str,
        *,
        max_file_size: int = _MAX_TELEGRAM_FILE_BYTES,
    ) -> dict[str, Any]:
        """Resolve one opaque Telegram file ID into validated download metadata.

        Telegram may omit ``file_size`` from the response, so the download
        boundary independently enforces its byte limit. Errors deliberately do
        not include either opaque identifier returned by Telegram.
        """

        normalized_file_id = self._validate_opaque_file_id(file_id)
        size_limit = self._validate_file_size_limit(
            max_file_size, field="max_file_size"
        )
        try:
            result = await self.call("getFile", {"file_id": normalized_file_id})
        except BotApiError as exc:
            status = exc.error_code or exc.http_status
            if isinstance(status, int) and 400 <= status < 500 and status != 429:
                raise BotApiFileError(
                    "Bot API permanently rejected the Telegram file"
                ) from None
            raise BotApiError(
                "Bot API could not resolve the Telegram file",
                retry_after=exc.retry_after,
                http_status=exc.http_status,
                error_code=exc.error_code,
            ) from None
        if not isinstance(result, dict):
            raise BotApiFileError("Bot API returned invalid Telegram file metadata")

        returned_file_id = result.get("file_id")
        file_unique_id = result.get("file_unique_id")
        raw_file_size = result.get("file_size")
        raw_file_path = result.get("file_path")
        try:
            validated_file_id = self._validate_opaque_file_id(returned_file_id)
            if file_unique_id is not None:
                validated_unique_id = self._validate_opaque_file_id(file_unique_id)
            else:
                validated_unique_id = None
            if raw_file_size is not None and (
                isinstance(raw_file_size, bool)
                or not isinstance(raw_file_size, int)
                or raw_file_size < 0
            ):
                raise ValueError
            validated_path = self._validate_telegram_file_path(raw_file_path)
        except (TypeError, ValueError):
            raise BotApiFileError(
                "Bot API returned invalid Telegram file metadata"
            ) from None

        if raw_file_size is not None and raw_file_size > size_limit:
            raise BotApiFileError("Telegram file exceeds the configured size limit")
        metadata: dict[str, Any] = {
            "file_id": validated_file_id,
            "file_path": validated_path,
        }
        if validated_unique_id is not None:
            metadata["file_unique_id"] = validated_unique_id
        if raw_file_size is not None:
            metadata["file_size"] = raw_file_size
        return metadata

    async def download_file(
        self,
        file_path: str,
        *,
        max_bytes: int = _MAX_TELEGRAM_FILE_BYTES,
    ) -> bytes:
        """Download a Telegram file into memory under a strict byte ceiling."""

        normalized_path = self._validate_telegram_file_path(file_path)
        size_limit = self._validate_file_size_limit(max_bytes, field="max_bytes")
        started = time.monotonic()
        LOGGER.info("Telegram file download started; status=started")
        url = f"https://api.telegram.org/file/bot{self._token}/{normalized_path}"
        try:
            async with self._client.stream("GET", url) as response:
                if response.status_code != 200:
                    LOGGER.warning(
                        "Telegram file download finished; status=http_error "
                        "http_status=%d elapsed=%.3fs",
                        response.status_code,
                        time.monotonic() - started,
                    )
                    if 400 <= response.status_code < 500 and response.status_code != 429:
                        raise BotApiFileError(
                            "Telegram permanently rejected the file download"
                        )
                    raise BotApiError(
                        f"Telegram file download HTTP status {response.status_code}",
                        http_status=response.status_code,
                    )

                content_length = response.headers.get("content-length")
                if content_length is not None:
                    try:
                        declared_size = int(content_length)
                    except ValueError:
                        declared_size = -1
                    if declared_size > size_limit:
                        LOGGER.warning(
                            "Telegram file download finished; status=too_large "
                            "elapsed=%.3fs",
                            time.monotonic() - started,
                        )
                        raise BotApiFileError(
                            "Telegram file exceeds the configured size limit"
                        )

                content = bytearray()
                async for chunk in response.aiter_bytes():
                    if len(content) + len(chunk) > size_limit:
                        LOGGER.warning(
                            "Telegram file download finished; status=too_large "
                            "elapsed=%.3fs",
                            time.monotonic() - started,
                        )
                        raise BotApiFileError(
                            "Telegram file exceeds the configured size limit"
                        )
                    content.extend(chunk)
        except BotApiError:
            raise
        except httpx.HTTPError as exc:
            LOGGER.warning(
                "Telegram file download finished; status=transport_error "
                "elapsed=%.3fs error_type=%s",
                time.monotonic() - started,
                type(exc).__name__,
            )
            raise BotApiError(
                f"Telegram file download transport error: {type(exc).__name__}"
            ) from None

        if not content:
            raise BotApiFileError("Telegram file download returned empty content")
        LOGGER.info(
            "Telegram file download finished; status=success bytes=%d elapsed=%.3fs",
            len(content),
            time.monotonic() - started,
        )
        return bytes(content)

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
                    "Пришлите голосовое, напишите календарную команду или "
                    "отправьте скриншот. ИИ-планировщик сразу добавит, "
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
                    "Голос, текст или скриншот → ИИ-планировщик → Google Calendar"
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

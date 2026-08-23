"""Long-running private bot service."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import asdict, is_dataclass
import logging
import time
from datetime import datetime, timezone
from typing import Any
from zoneinfo import ZoneInfo

from dotenv import load_dotenv

from .bot_api import BotApi, BotApiError, read_secret
from .calendar import (
    CalendarClient,
    CalendarConnectionError,
    CalendarEventQueryResult,
)
from .calendar_mcp import open_calendar_mcp
from .config import PROJECT_ROOT, Config
from .confirmation import CalendarConfirmationPipeline, ConfirmationStore
from .gateway import (
    GatewayConnectionError,
    GatewayError,
    TelegramGateway,
    open_gateway,
)
from .gemini import (
    GeminiApi,
    GeminiCli,
    GeminiError,
    GeminiFallback,
    GeminiProvider,
)
from .intent import format_calendar_preview
from .operations import (
    CalendarOperationError,
    CalendarOperationPipeline,
    OperationStore,
)
from .state import StateStore
from .text import telegram_text_chunks
from .ui import (
    FieldChange,
    format_clarify_card,
    format_create_card,
    format_delete_card,
    format_error_card,
    format_ignore_card,
    format_lookup_clarify_card,
    format_mixed_operation_card,
    format_progress_card,
    format_read_card,
    format_undo_card,
    format_update_card,
    parse_undo_callback,
    undo_reply_markup,
)
from .webhook import WebhookRuntime

LOGGER = logging.getLogger("tg_voice_transcriber_bot")

START_TEXT = (
    "Пришлите голосовое сообщение или напишите календарную команду текстом. "
    "Для голосового я найду исходное сообщение через вашу пользовательскую "
    "Telegram-сессию и запрошу серверную расшифровку Telegram. Gemini 3.7 "
    "Flash High выделит событие и покажет предпросмотр. После проверки нажмите "
    "«Добавить», и событие попадёт в основной Google Calendar. Аудиофайл не "
    "скачивается. Без этого подтверждения календарь не изменяется."
)

START_TEXT_V2 = (
    "Пришлите голосовое сообщение или напишите календарную команду текстом. "
    "Голосовое Telegram расшифрует на своих серверах, а Gemini 3.7 Flash High "
    "с учётом последних команд сразу добавит, изменит или удалит событие в "
    "основном Google Calendar. Вы увидите ход обработки в одном обновляемом "
    "сообщении и итоговую карточку со временем выполнения. Если результат не "
    "подходит, нажмите кнопку отмены или исправьте его новым сообщением. "
    "Аудиофайл бот не скачивает."
)


# These bounds mirror the number of numbered rows rendered by the two UI
# cards.  Persisting the same slice makes follow-ups such as “the second one”
# resolve against exactly what the owner saw, rather than the wider event cache.
_READ_DISPLAY_LIMIT = 8
_LOOKUP_DISPLAY_LIMIT = 5
_ACTIVE_EVENT_STATUSES = frozenset({"confirmed", "tentative"})
_CALENDAR_WRITE_RETRY_LIMIT = 5


def message_command(text: str) -> str:
    parts = text.split(maxsplit=1)
    return parts[0].split("@", maxsplit=1)[0].lower() if parts else ""


def transcription_reply(result: dict[str, Any]) -> str:
    if result.get("ok") and result.get("status") == "completed":
        text = str(result.get("text", ""))
        return text or "Telegram вернул пустую расшифровку."

    status = str(result.get("status", ""))
    error = str(result.get("error", "")).upper()
    if status == "timeout":
        partial = str(result.get("partial_text", ""))
        suffix = "⚠️ Telegram не успел завершить расшифровку. Попробуйте ещё раз."
        return f"{partial}\n\n{suffix}" if partial else suffix
    if "PREMIUM_ACCOUNT_REQUIRED" in error or "PREMIUMACCOUNTREQUIRED" in error:
        return (
            "Для этого аккаунта Telegram требует Premium либо уже исчерпана "
            "бесплатная квота расшифровок."
        )
    if "MSG_VOICE_TOO_LONG" in error or "MSGVOICETOOLONG" in error:
        return "Это голосовое слишком длинное для серверной расшифровки Telegram."
    if "TRANSCRIPTION_FAILED" in error or "TRANSCRIPTIONFAILED" in error:
        return "Telegram не смог распознать это голосовое. Попробуйте отправить его ещё раз."
    if "MSG_VOICE_MISSING" in error or status == "invalid_message":
        return "Не удалось найти голосовое в пользовательской сессии. Отправьте его ещё раз."
    if result.get("retry_after"):
        return "Telegram временно ограничил запросы. Попробуйте немного позже."
    return "Не удалось получить расшифровку от Telegram. Попробуйте ещё раз."


def build_calendar_confirmation(
    config: Config, calendar: CalendarClient | None
) -> CalendarConfirmationPipeline | None:
    """Build the durable confirmation pipeline around a supplied adapter."""
    if calendar is None:
        return None
    return CalendarConfirmationPipeline(
        ConfirmationStore(config.confirmation_state_path),
        calendar,
        timezone_name=config.calendar_timezone,
    )


def build_calendar_operations(
    config: Config, calendar: CalendarClient | None
) -> CalendarOperationPipeline | None:
    """Build the auto-apply journal, importing completed v1 records once."""
    if calendar is None:
        return None
    return CalendarOperationPipeline(
        OperationStore(
            config.operation_state_path,
            legacy_confirmation_path=config.confirmation_state_path,
        ),
        calendar,
        timezone_name=config.calendar_timezone,
    )


def _record_action(record: dict[str, Any]) -> str:
    actions = {
        str(item.get("type"))
        for item in record.get("items", [])
        if isinstance(item, dict)
    }
    if len(actions) == 1:
        return next(iter(actions))
    return "mixed"


def _record_events(record: dict[str, Any]) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for item in record.get("items", []):
        if not isinstance(item, dict):
            continue
        snapshot = item.get("before") if item.get("type") == "delete" else item.get("after")
        if isinstance(snapshot, dict):
            events.append(snapshot)
    return events


def _record_changes(record: dict[str, Any]) -> list[FieldChange]:
    changes: list[FieldChange] = []
    fields = (
        "title",
        "start_at",
        "end_at",
        "all_day",
        "location",
        "description",
        "recurrence_rrule",
    )
    for item in record.get("items", []):
        if not isinstance(item, dict) or item.get("type") != "update":
            continue
        before = item.get("before")
        after = item.get("after")
        if not isinstance(before, dict) or not isinstance(after, dict):
            continue
        for field in fields:
            if before.get(field) != after.get(field):
                changes.append(
                    FieldChange(
                        field,
                        None if before.get(field) is None else str(before.get(field)),
                        None if after.get(field) is None else str(after.get(field)),
                    )
                )
    return changes


def _record_undo_is_best_effort(record: Mapping[str, Any]) -> bool:
    undo = record.get("undo")
    return isinstance(undo, Mapping) and undo.get("fidelity") == "core_only"


def _success_card(
    record: dict[str, Any], *, transcript: str, elapsed_seconds: float
) -> tuple[str, dict[str, Any]]:
    action = _record_action(record)
    events = _record_events(record)
    operation_id = str(record["operation_id"])
    best_effort_undo = _record_undo_is_best_effort(record)
    if action == "create":
        html = format_create_card(
            events, transcript=transcript, elapsed_seconds=elapsed_seconds
        )
    elif action == "update":
        html = format_update_card(
            events,
            transcript=transcript,
            elapsed_seconds=elapsed_seconds,
            changes=_record_changes(record),
        )
    elif action == "delete":
        html = format_delete_card(
            events,
            transcript=transcript,
            elapsed_seconds=elapsed_seconds,
            best_effort_undo=best_effort_undo,
        )
    else:
        mixed_items = [
            {
                "type": item.get("type"),
                "event": (
                    item.get("before")
                    if item.get("type") == "delete"
                    else item.get("after")
                ),
            }
            for item in record.get("items", [])
            if isinstance(item, dict)
        ]
        html = format_mixed_operation_card(
            mixed_items,
            transcript=transcript,
            elapsed_seconds=elapsed_seconds,
            best_effort_undo=best_effort_undo,
        )
    return html, undo_reply_markup(  # type: ignore[arg-type]
        operation_id,
        action,
        best_effort=best_effort_undo,
    )


def _calendar_query_payload(result: CalendarEventQueryResult) -> dict[str, Any]:
    """Convert a typed provider read into a JSON-safe durable job payload."""

    events: list[dict[str, Any]] = []
    for event in result.events:
        if is_dataclass(event) and not isinstance(event, type):
            value = asdict(event)
        elif isinstance(event, Mapping):
            value = dict(event)
        else:  # pragma: no cover - guarded by the adapter contract
            raise CalendarOperationError(
                "Google Calendar вернул некорректное событие. Попробуйте ещё раз."
            )
        events.append(value)
    return {
        "events": events,
        "total_count": int(result.total_count),
        "may_be_incomplete": bool(result.may_be_incomplete),
    }


def _indexed_candidates(
    candidates: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Copy an ordered UI candidate set and attach its one-based row index."""

    indexed: list[dict[str, Any]] = []
    for display_index, candidate in enumerate(candidates, start=1):
        copied = deepcopy(dict(candidate))
        copied["display_index"] = display_index
        indexed.append(copied)
    return indexed


def _visible_active_candidates(
    candidates: Sequence[Any], *, limit: int
) -> list[dict[str, Any]]:
    """Return the provider-ordered active rows that the UI can actually show."""

    visible: list[dict[str, Any]] = []
    for candidate in candidates:
        if not isinstance(candidate, Mapping):
            raise RuntimeError("Persisted calendar candidate is invalid")
        status = str(candidate.get("status") or "confirmed")
        if status not in _ACTIVE_EVENT_STATUSES:
            continue
        visible.append(deepcopy(dict(candidate)))
        if len(visible) == limit:
            break
    return visible


def _interaction_chain(
    plans: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    """Flatten one or two stateless Gemini exchanges for durable replay."""

    first_input: dict[str, Any] | None = None
    steps: list[dict[str, Any]] = []
    for plan in plans:
        interaction_input = plan.get("_interaction_input")
        if isinstance(interaction_input, Mapping):
            copied_input = deepcopy(dict(interaction_input))
            if first_input is None:
                first_input = copied_input
            else:
                steps.append(copied_input)
        interaction_steps = plan.get("_interaction_steps")
        if isinstance(interaction_steps, Sequence) and not isinstance(
            interaction_steps, (str, bytes, bytearray)
        ):
            steps.extend(
                deepcopy(dict(step))
                for step in interaction_steps
                if isinstance(step, Mapping)
            )
    return first_input, steps


class VoiceBotService:
    def __init__(
        self,
        config: Config,
        bot: BotApi,
        gateway: TelegramGateway,
        state: StateStore,
        gemini: GeminiProvider,
        calendar_confirmation: CalendarConfirmationPipeline | None = None,
        calendar_operations: CalendarOperationPipeline | None = None,
    ) -> None:
        self.config = config
        self.bot = bot
        self.gateway = gateway
        self.state = state
        self.gemini = gemini
        self.calendar_confirmation = calendar_confirmation
        self.calendar_operations = calendar_operations
        self.gemini_available = True
        self.enabled_accounts: frozenset[str] | None = None

    async def initialize(self) -> None:
        available_accounts = await self.gateway.validate_operations()
        expected_accounts = self.config.expected_user_id_by_account
        unexpected_accounts = available_accounts - expected_accounts.keys()
        if unexpected_accounts:
            raise RuntimeError("Telegram gateway exposed an unexpected account")
        for account in sorted(available_accounts):
            expected_user_id = expected_accounts[account]
            identity = await self.gateway.read(account, "get_me", {})
            if int(identity.get("id", 0)) != expected_user_id:
                raise RuntimeError(f"Telegram account mismatch for {account}")
        self.enabled_accounts = available_accounts

        bot_identity = await self.bot.call("getMe")
        if str(bot_identity.get("username", "")).lower() != self.config.bot_username.lower():
            raise RuntimeError("Bot token does not belong to the configured bot")
        if self.config.bot_update_mode == "webhook":
            # setWebhook is intentionally deferred until after the HTTP
            # listener has started in async_main.
            await self.bot.configure_profile()
        else:
            await self.bot.configure()
        try:
            await self.gemini.validate()
        except GeminiError:
            self.gemini_available = False
            LOGGER.warning("Gemini validation failed; transcript fallback enabled")
        LOGGER.info(
            "Bot, %d user session(s), and local integrations validated",
            len(available_accounts),
        )

    async def run(self) -> None:
        while True:
            try:
                updates = await self.bot.get_updates(self.state.offset)
            except BotApiError as exc:
                LOGGER.error("Bot API polling failed: %s", exc)
                await asyncio.sleep(exc.retry_after or 5)
                continue

            retry_current_batch = False
            for update in updates:
                update_id = int(update.get("update_id", 0))
                try:
                    await self.handle_update(update)
                except (GatewayConnectionError, CalendarConnectionError):
                    # Long-lived stdio clients cannot be repaired in this
                    # process. Leave the update offset untouched and let the
                    # process supervisor rebuild every MCP subprocess.
                    raise
                except BotApiError as exc:
                    LOGGER.error(
                        "Update %s failed transiently: %s",
                        update_id,
                        type(exc).__name__,
                    )
                    retry_current_batch = True
                    await asyncio.sleep(exc.retry_after or 3)
                    break
                except GatewayError as exc:
                    LOGGER.error(
                        "Update %s failed transiently: %s", update_id, type(exc).__name__
                    )
                    retry_current_batch = True
                    await asyncio.sleep(3)
                    break
                except Exception:
                    LOGGER.exception("Unexpected failure for update %s", update_id)
                    retry_current_batch = True
                    await asyncio.sleep(3)
                    break
                else:
                    self.state.complete(update_id)
            if retry_current_batch:
                continue

    async def handle_update(self, update: dict[str, Any]) -> None:
        update_id = int(update.get("update_id", 0))
        callback_query = update.get("callback_query")
        if isinstance(callback_query, dict):
            await self._handle_callback_query(callback_query, update_id=update_id)
            return

        message = update.get("message")
        if not isinstance(message, dict):
            return

        sender = message.get("from") or {}
        chat = message.get("chat") or {}
        sender_id = int(sender.get("id", 0) or 0)
        chat_id = int(chat.get("id", 0) or 0)
        bot_message_id = int(message.get("message_id", 0) or 0)

        if chat.get("type") != "private" or chat_id != sender_id:
            return

        account = self.config.account_by_user_id.get(sender_id)
        if account is None:
            # Public usernames cannot be hidden. Silently discard unknown users
            # so an outsider cannot consume sendMessage quota or starve owners.
            LOGGER.warning("Ignored an update from an unauthorized user")
            return

        raw_text = message.get("text")
        text = raw_text if isinstance(raw_text, str) else ""
        command = message_command(text)
        if command == "/start":
            await self.bot.send_text(
                chat_id,
                START_TEXT_V2 if self.calendar_operations is not None else START_TEXT,
                reply_to_message_id=bot_message_id,
            )
            return
        if command == "/status":
            label = "личный" if account == "personal" else "рабочий"
            telegram_status = (
                "Telegram-расшифровка подключена"
                if self.enabled_accounts is None
                or account in self.enabled_accounts
                else "Telegram-сессия этого аккаунта временно не подключена"
            )
            gemini_status = (
                "Gemini 3.7 Flash High доступна"
                if self.gemini_available
                else "Gemini сейчас недоступна; останется обычная расшифровка"
            )
            calendar_status = (
                "Google Calendar подключён; изменения применяются сразу с кнопкой отмены"
                if self.calendar_operations is not None
                else "Google Calendar подключён; запись только после подтверждения"
                if self.calendar_confirmation is not None
                else "Google Calendar не подключён"
            )
            await self.bot.send_text(
                chat_id,
                f"Доступ подтверждён: {label} аккаунт. {telegram_status}. "
                f"{gemini_status}. {calendar_status}."
                + (
                    " Пришлите голосовое сообщение или текстовую команду."
                    if self.enabled_accounts is None
                    or account in self.enabled_accounts
                    else " Текстовые команды доступны без Telegram-сессии."
                ),
                reply_to_message_id=bot_message_id,
            )
            return

        if text.strip():
            process_text = (
                self._process_text_v2
                if self.calendar_operations is not None
                else self._process_text
            )
            await process_text(
                update_id=update_id,
                account=account,
                chat_id=chat_id,
                bot_message_id=bot_message_id,
                sent_at=int(message.get("date", 0) or 0),
                text=text,
            )
            return

        voice = message.get("voice")
        if not isinstance(voice, dict):
            await self.bot.send_text(
                chat_id,
                "Пришлите голосовое сообщение, записанное в Telegram, или "
                "напишите календарную команду текстом.",
                reply_to_message_id=bot_message_id,
            )
            return

        if (
            self.enabled_accounts is not None
            and account not in self.enabled_accounts
        ):
            await self.bot.send_text(
                chat_id,
                "Telegram-сессия этого аккаунта временно не подключена. "
                "Попробуйте с другого разрешённого аккаунта.",
                reply_to_message_id=bot_message_id,
            )
            return

        process = (
            self._process_voice_v2
            if self.calendar_operations is not None
            else self._process_voice
        )
        await process(
            update_id=update_id,
            account=account,
            chat_id=chat_id,
            bot_message_id=bot_message_id,
            sent_at=int(message.get("date", 0) or 0),
            duration=int(voice.get("duration", 0) or 0),
            file_size=(
                int(voice["file_size"])
                if voice.get("file_size") is not None
                else None
            ),
        )

    async def _handle_callback_query(
        self, callback_query: dict[str, Any], *, update_id: int = 0
    ) -> None:
        """Handle only private, owner-bound calendar callbacks.

        Callback UI acknowledgements are best-effort.  A stale callback query
        must not hold the polling offset forever after the durable confirmation
        journal has already recorded its outcome.
        """
        callback_query_id = str(callback_query.get("id", ""))
        if not callback_query_id:
            LOGGER.warning("Ignored callback query without an ID")
            return

        sender = callback_query.get("from") or {}
        message = callback_query.get("message") or {}
        chat = message.get("chat") or {}
        sender_id = int(sender.get("id", 0) or 0)
        chat_id = int(chat.get("id", 0) or 0)
        message_id = int(message.get("message_id", 0) or 0)
        account = self.config.account_by_user_id.get(sender_id)

        if (
            account is None
            or chat.get("type") != "private"
            or chat_id != sender_id
            or message_id <= 0
        ):
            LOGGER.warning("Rejected an unauthorized or unbound callback query")
            await self._answer_callback_best_effort(
                callback_query_id, "Недоступно.", show_alert=True
            )
            return

        undo_operation_id = parse_undo_callback(callback_query.get("data"))
        if undo_operation_id is not None:
            await self._handle_undo_callback(
                callback_query_id=callback_query_id,
                operation_id=undo_operation_id,
                owner_user_id=sender_id,
                chat_id=chat_id,
                message_id=message_id,
                source_update_id=update_id,
            )
            return

        if self.calendar_confirmation is None:
            await self._answer_callback_best_effort(
                callback_query_id,
                "Календарь пока не подключён.",
                show_alert=True,
            )
            return

        result = await self.calendar_confirmation.handle_callback(
            data=callback_query.get("data"),
            owner_user_id=sender_id,
            chat_id=chat_id,
        )
        if not result.handled:
            await self._answer_callback_best_effort(
                callback_query_id, "Недоступно.", show_alert=True
            )
            return

        if result.remove_keyboard and result.outcome != "rejected":
            try:
                await self.bot.remove_inline_keyboard(chat_id, message_id)
            except BotApiError:
                # The persisted stage is authoritative.  If the old button is
                # clicked again, the pipeline returns a terminal/idempotent result.
                LOGGER.warning("Could not remove a completed callback keyboard")
        await self._answer_callback_best_effort(
            callback_query_id,
            result.answer_text,
            show_alert=result.outcome in {"rejected", "retryable_error"},
        )

    async def _answer_callback_best_effort(
        self,
        callback_query_id: str,
        text: str,
        *,
        show_alert: bool = False,
    ) -> None:
        try:
            await self.bot.answer_callback_query(
                callback_query_id, text, show_alert=show_alert
            )
        except BotApiError:
            LOGGER.warning("Could not answer a callback query before it expired")

    async def _handle_undo_callback(
        self,
        *,
        callback_query_id: str,
        operation_id: str,
        owner_user_id: int,
        chat_id: int,
        message_id: int,
        source_update_id: int,
    ) -> None:
        pipeline = self.calendar_operations
        if pipeline is None:
            await self._answer_callback_best_effort(
                callback_query_id, "Календарь пока не подключён.", show_alert=True
            )
            return

        await self._answer_callback_best_effort(callback_query_id, "Отменяю…")
        started = time.monotonic()
        result = await pipeline.undo(
            operation_id=operation_id,
            owner_user_id=owner_user_id,
            chat_id=chat_id,
            source_update_id=source_update_id,
        )
        if result.outcome == "rejected":
            return
        if result.outcome == "retryable_error":
            await self.bot.send_html(
                chat_id,
                format_error_card(
                    "Не удалось отменить действие. Кнопка остаётся активной — попробуйте ещё раз.",
                    elapsed_seconds=time.monotonic() - started,
                    calendar_unchanged=False,
                ),
            )
            return
        if result.outcome == "blocked":
            try:
                await self.bot.remove_inline_keyboard(chat_id, message_id)
            except BotApiError:
                LOGGER.warning("Could not remove a stale undo keyboard")
            await self.bot.send_html(
                chat_id,
                format_error_card(
                    "Эту операцию уже нельзя безопасно отменить: одно из событий изменилось позже.",
                    elapsed_seconds=time.monotonic() - started,
                    calendar_unchanged=True,
                ),
            )
            return

        record = result.record
        if not isinstance(record, dict):
            return
        try:
            await self.bot.remove_inline_keyboard(chat_id, message_id)
        except BotApiError:
            LOGGER.warning("Could not remove a completed undo keyboard")
        already_notified = bool((record.get("undo") or {}).get("chat_notified"))
        if result.outcome == "already_undone" and already_notified:
            return
        action = _record_action(record)
        best_effort = _record_undo_is_best_effort(record)
        await self.bot.send_html(
            chat_id,
            format_undo_card(
                action,  # type: ignore[arg-type]
                result.event_titles,
                elapsed_seconds=time.monotonic() - started,
                best_effort=best_effort,
            ),
        )
        pipeline.mark_undo_notified(operation_id)

    @staticmethod
    def _elapsed(job: dict[str, Any]) -> float:
        try:
            monotonic_started = float(job["started_monotonic"])
            monotonic_now = time.monotonic()
        except (KeyError, TypeError, ValueError):
            monotonic_started = monotonic_now = -1
        if monotonic_now >= monotonic_started >= 0:
            return monotonic_now - monotonic_started
        try:
            started_at = float(job.get("started_at", time.time()))
        except (TypeError, ValueError):
            started_at = time.time()
        return max(0.0, time.time() - started_at)

    async def _edit_progress_best_effort(
        self, *, chat_id: int, job: dict[str, Any], html_text: str
    ) -> None:
        message_id = int(job.get("status_message_id", 0) or 0)
        if message_id <= 0:
            return
        if job.get("last_progress_html") == html_text:
            return
        try:
            await self.bot.edit_html(chat_id, message_id, html_text)
        except BotApiError:
            LOGGER.warning("Could not update an intermediate progress message")
        else:
            job["last_progress_html"] = html_text

    async def _finish_v2_job(
        self,
        *,
        update_id: int,
        chat_id: int,
        bot_message_id: int,
        job: dict[str, Any],
    ) -> None:
        html_text = str(job["final_html"])
        markup = job.get("final_reply_markup")
        if not isinstance(markup, dict):
            markup = None
        status_message_id = int(job.get("status_message_id", 0) or 0)
        if status_message_id > 0:
            try:
                await self.bot.edit_html(
                    chat_id,
                    status_message_id,
                    html_text,
                    reply_markup=markup,
                )
            except BotApiError:
                LOGGER.warning("Could not finalize progress message; sending fallback")
            else:
                job["final_message_id"] = status_message_id
                job["status"] = "sent"
                self.state.save_job(update_id, job)
                return
        job["final_message_id"] = await self.bot.send_html(
            chat_id,
            html_text,
            reply_to_message_id=bot_message_id,
            reply_markup=markup,
        )
        job["status"] = "sent"
        self.state.save_job(update_id, job)

    async def _process_text_v2(
        self,
        *,
        update_id: int,
        account: str,
        chat_id: int,
        bot_message_id: int,
        sent_at: int,
        text: str,
    ) -> None:
        """Enter the durable v2 pipeline after voice transcription."""

        job = self.state.job(update_id)
        if job is None:
            job = {
                "account": account,
                # Text must not advance the MTProto voice matching cursor.
                "user_message_id": 0,
                "sent_at": sent_at,
                "started_at": time.time(),
                "started_monotonic": time.monotonic(),
                "input_kind": "text",
                "transcript": text,
                "status": "transcribed",
            }
            self.state.save_job(update_id, job)
        elif (
            job.get("account") != account
            or job.get("input_kind") != "text"
            or job.get("transcript") != text
        ):
            raise RuntimeError("Persisted text job contradicts its Telegram update")

        await self._process_voice_v2(
            update_id=update_id,
            account=account,
            chat_id=chat_id,
            bot_message_id=bot_message_id,
            sent_at=sent_at,
            duration=0,
            file_size=None,
        )

    async def _process_voice_v2(
        self,
        *,
        update_id: int,
        account: str,
        chat_id: int,
        bot_message_id: int,
        sent_at: int,
        duration: int,
        file_size: int | None,
    ) -> None:
        pipeline = self.calendar_operations
        if pipeline is None:  # pragma: no cover - selected only when configured
            raise RuntimeError("Calendar operation pipeline is unavailable")

        await self.bot.send_chat_action(chat_id)
        job = self.state.job(update_id)
        if job is None:
            job = {
                "account": account,
                "user_message_id": 0,
                "sent_at": sent_at,
                "started_at": time.time(),
                "started_monotonic": time.monotonic(),
                "input_kind": "voice",
                "status": "starting",
            }
            self.state.save_job(update_id, job)
        elif "input_kind" not in job:
            # Jobs created before text support were necessarily voice jobs.
            job["input_kind"] = "voice"
            self.state.save_job(update_id, job)
        input_kind = "text" if job.get("input_kind") == "text" else "voice"
        if int(job.get("status_message_id", 0) or 0) <= 0:
            matching_html = format_progress_card(
                "gemini" if input_kind == "text" else "matching",
                input_kind=input_kind,
            )
            job["status_message_id"] = await self.bot.send_html(
                chat_id,
                matching_html,
                reply_to_message_id=bot_message_id,
            )
            job["last_progress_html"] = matching_html
            if job.get("status") == "starting":
                job["status"] = "matching"
            self.state.save_job(update_id, job)

        if job.get("status") in {"starting", "matching"}:
            match: dict[str, Any] | None = None
            for delay in (0, 0.5, 1, 2, 3):
                if delay:
                    await asyncio.sleep(delay)
                result = await self.gateway.read(
                    account,
                    "find_recent_outgoing_voice",
                    {
                        "chat_id": self.config.bot_chat_id,
                        "sent_at": sent_at,
                        "duration": duration,
                        "file_size": file_size,
                        "after_message_id": self.state.after_message_id(account),
                        "search_limit": 100,
                        "tolerance_seconds": 5,
                    },
                )
                if result.get("ok") and isinstance(result.get("match"), dict):
                    match = result["match"]
                    break
            if match is None:
                job["final_html"] = format_error_card(
                    "Не удалось сопоставить голосовое с пользовательской Telegram-сессией. Отправьте его ещё раз.",
                    elapsed_seconds=self._elapsed(job),
                )
                job["final_reply_markup"] = None
                job["status"] = "final_ready"
                self.state.save_job(update_id, job)
            else:
                job["user_message_id"] = int(match["message_id"])
                job["status"] = "found"
                self.state.save_job(update_id, job)

        if job.get("status") == "found":
            await self._edit_progress_best_effort(
                chat_id=chat_id,
                job=job,
                html_text=format_progress_card(
                    "transcribing", input_kind=input_kind
                ),
            )
            self.state.save_job(update_id, job)
            result = await self.gateway.write(
                account,
                "transcribe_voice_message",
                {
                    "chat_id": self.config.bot_chat_id,
                    "message_id": int(job["user_message_id"]),
                    "wait_timeout": self.config.transcription_timeout_seconds,
                },
                request_id=f"stt-{account}-{update_id}",
                timeout=self.config.transcription_timeout_seconds + 30,
            )
            transcript = (
                str(result.get("text", ""))
                if result.get("ok") and result.get("status") == "completed"
                else ""
            )
            if transcript:
                job["transcript"] = transcript
                job["status"] = "transcribed"
            else:
                job["final_html"] = format_error_card(
                    transcription_reply(result),
                    elapsed_seconds=self._elapsed(job),
                )
                job["final_reply_markup"] = None
                job["status"] = "final_ready"
            self.state.save_job(update_id, job)

        if job.get("status") == "transcribed":
            transcript = str(job["transcript"])
            await self._edit_progress_best_effort(
                chat_id=chat_id,
                job=job,
                html_text=format_progress_card("gemini", input_kind=input_kind),
            )
            self.state.save_job(update_id, job)
            if not self.gemini_available:
                job["final_html"] = format_error_card(
                    "Gemini сейчас недоступна. Команда сохранена в этой карточке.",
                    transcript=transcript,
                    elapsed_seconds=self._elapsed(job),
                )
                job["final_reply_markup"] = None
                job["status"] = "final_ready"
                self.state.save_job(update_id, job)
            else:
                sent_time = datetime.fromtimestamp(
                    int(job["sent_at"]), tz=timezone.utc
                ).astimezone(ZoneInfo(self.config.calendar_timezone))
                context = pipeline.context(
                    account=account, chat_id=chat_id, now=sent_time
                )
                try:
                    plan = await self.gemini.plan_calendar_actions(
                        transcript,
                        reference_time=sent_time,
                        account=account,
                        application_state=context.application_state,
                        recent_conversation=context.recent_conversation,
                        history_steps=context.history_steps,
                    )
                except GeminiError:
                    LOGGER.warning(
                        "Gemini planning failed for update %s; calendar unchanged",
                        update_id,
                    )
                    job["final_html"] = format_error_card(
                        "Gemini не смогла надёжно разобрать календарную команду. "
                        "Попробуйте уточнить её новым сообщением.",
                        transcript=transcript,
                        elapsed_seconds=self._elapsed(job),
                    )
                    job["final_reply_markup"] = None
                    job["status"] = "final_ready"
                else:
                    job["plan"] = plan
                    job["status"] = "planned"
                self.state.save_job(update_id, job)

        if job.get("status") == "planned":
            plan = job.get("plan")
            if not isinstance(plan, dict):
                raise RuntimeError("Persisted calendar operation plan is invalid")
            if plan.get("action") in {"read", "lookup"}:
                lookup = plan.get("lookup")
                if not isinstance(lookup, dict):
                    raise RuntimeError("Persisted calendar lookup is invalid")
                await self._edit_progress_best_effort(
                    chat_id=chat_id,
                    job=job,
                    html_text=format_progress_card(
                        "calendar_lookup", input_kind=input_kind
                    ),
                )
                self.state.save_job(update_id, job)
                try:
                    query_result = await pipeline.query_events(
                        account=account,
                        query=(
                            str(lookup["query"])
                            if lookup.get("query") is not None
                            else None
                        ),
                        time_min=str(lookup["time_min"]),
                        time_max=str(lookup["time_max"]),
                        limit=20,
                    )
                    durable_result = _calendar_query_payload(query_result)
                except CalendarOperationError as exc:
                    LOGGER.warning(
                        "Calendar lookup failed for update %s; calendar unchanged",
                        update_id,
                    )
                    job["final_html"] = format_error_card(
                        str(exc),
                        transcript=str(job["transcript"]),
                        elapsed_seconds=self._elapsed(job),
                    )
                    job["final_reply_markup"] = None
                    job["status"] = "final_ready"
                else:
                    job["calendar_query_result"] = durable_result
                    job["status"] = (
                        "calendar_read_ready"
                        if plan.get("action") == "read"
                        else "calendar_lookup_ready"
                    )
                self.state.save_job(update_id, job)

        if job.get("status") == "calendar_read_ready":
            transcript = str(job["transcript"])
            plan = job.get("plan")
            result = job.get("calendar_query_result")
            if not isinstance(plan, dict) or not isinstance(result, dict):
                raise RuntimeError("Persisted calendar read result is invalid")
            lookup = plan.get("lookup")
            events = result.get("events")
            if not isinstance(lookup, dict) or not isinstance(events, list):
                raise RuntimeError("Persisted calendar read payload is invalid")
            displayed_events = _visible_active_candidates(
                events, limit=_READ_DISPLAY_LIMIT
            )
            sent_time = datetime.fromtimestamp(
                int(job["sent_at"]), tz=timezone.utc
            ).astimezone(ZoneInfo(self.config.calendar_timezone))
            interaction_input, interaction_steps = _interaction_chain((plan,))
            execution = await pipeline.record_read(
                source_update_id=update_id,
                account=account,
                owner_user_id=chat_id,
                chat_id=chat_id,
                transcript=transcript,
                reference_time=sent_time,
                lookup=lookup,
                events=events,
                total_count=int(result.get("total_count", len(events))),
                may_be_incomplete=bool(result.get("may_be_incomplete")),
                interaction_input=interaction_input,
                interaction_steps=interaction_steps,
                displayed_candidates=displayed_events,
            )
            job["operation_id"] = execution.operation_id
            job["final_html"] = format_read_card(
                displayed_events,
                transcript=transcript,
                elapsed_seconds=self._elapsed(job),
                total_count=int(result.get("total_count", len(events))),
                may_be_incomplete=bool(result.get("may_be_incomplete")),
            )
            job["final_reply_markup"] = None
            job["status"] = "final_ready"
            self.state.save_job(update_id, job)

        if job.get("status") == "calendar_lookup_ready":
            transcript = str(job["transcript"])
            initial_plan = job.get("plan")
            result = job.get("calendar_query_result")
            if not isinstance(initial_plan, dict) or not isinstance(result, dict):
                raise RuntimeError("Persisted calendar lookup result is invalid")
            events = result.get("events")
            lookup = initial_plan.get("lookup")
            if not isinstance(events, list) or not isinstance(lookup, dict):
                raise RuntimeError("Persisted calendar lookup payload is invalid")
            total_count = int(result.get("total_count", len(events)))
            incomplete = bool(result.get("may_be_incomplete")) or total_count > len(
                events
            )

            # Normalize and durably pin provider order before either asking the
            # owner or invoking Gemini for a second pass.  Only rows that can be
            # rendered in the clarification card are mutation-authorized.
            trusted_candidates: list[dict[str, Any]] = []
            pinned_candidates = job.get("calendar_lookup_candidates")
            if "calendar_lookup_candidates" in job:
                if not isinstance(pinned_candidates, list):
                    raise RuntimeError("Persisted Calendar candidate set is invalid")
                if any(not isinstance(event, Mapping) for event in events):
                    raise RuntimeError("Persisted Calendar lookup result is invalid")
                trusted_candidates = [deepcopy(dict(event)) for event in events]
                visible_candidates = [
                    deepcopy(dict(event))
                    for event in pinned_candidates
                    if isinstance(event, Mapping)
                ]
                if len(visible_candidates) != len(pinned_candidates):
                    raise RuntimeError("Persisted Calendar candidate set is invalid")
            elif events:
                observed = pipeline.observe_lookup_events(account, events)
                trusted_candidates = [deepcopy(dict(event)) for event in observed]
                job["calendar_query_result"]["events"] = trusted_candidates
                visible_candidates = _visible_active_candidates(
                    trusted_candidates, limit=_LOOKUP_DISPLAY_LIMIT
                )
                job["calendar_lookup_candidates"] = visible_candidates
                self.state.save_job(update_id, job)
            else:
                visible_candidates = []
                job["calendar_lookup_candidates"] = visible_candidates
                self.state.save_job(update_id, job)

            await self._edit_progress_best_effort(
                chat_id=chat_id,
                job=job,
                html_text=format_progress_card(
                    "gemini_match", input_kind=input_kind
                ),
            )
            self.state.save_job(update_id, job)
            sent_time = datetime.fromtimestamp(
                int(job["sent_at"]), tz=timezone.utc
            ).astimezone(ZoneInfo(self.config.calendar_timezone))
            context = pipeline.context(
                account=account, chat_id=chat_id, now=sent_time
            )
            application_state = deepcopy(context.application_state)
            indexed_candidates = _indexed_candidates(visible_candidates)
            candidate_ids = [str(event["event_id"]) for event in visible_candidates]
            application_state.update(
                {
                    "candidate_events": indexed_candidates,
                    "allowed_event_ids": candidate_ids,
                    "lookup_permitted": False,
                    "lookup_request": deepcopy(lookup),
                    "lookup_result": {
                        "total_count": total_count,
                        "may_be_incomplete": incomplete,
                    },
                }
            )
            native_history = list(context.history_steps)
            first_input, first_steps = _interaction_chain((initial_plan,))
            if first_input is not None:
                native_history.append(first_input)
            native_history.extend(first_steps)
            try:
                resolved_plan = await self.gemini.plan_calendar_actions(
                    transcript,
                    reference_time=sent_time,
                    account=account,
                    application_state=application_state,
                    recent_conversation=context.recent_conversation,
                    history_steps=native_history,
                )
            except GeminiError:
                LOGGER.warning(
                    "Gemini candidate matching failed for update %s; calendar unchanged",
                    update_id,
                )
                job["final_html"] = format_error_card(
                    "Gemini не смогла выбрать точное событие. Уточните название "
                    "или время новым сообщением.",
                    transcript=transcript,
                    elapsed_seconds=self._elapsed(job),
                )
                job["final_reply_markup"] = None
                job["status"] = "final_ready"
            else:
                if resolved_plan.get("action") in {"read", "lookup"}:
                    resolved_plan = {
                        "action": "clarify",
                        "operations": [],
                        "lookup": None,
                        "clarification_question": (
                            "Не удалось однозначно выбрать событие. Уточните "
                            "его название или время."
                        ),
                        "confidence": float(resolved_plan.get("confidence", 0)),
                        "_interaction_input": resolved_plan.get(
                            "_interaction_input"
                        ),
                        "_interaction_steps": resolved_plan.get(
                            "_interaction_steps"
                        ),
                    }
                job["resolved_plan"] = resolved_plan
                job["status"] = "calendar_lookup_planned"
            self.state.save_job(update_id, job)

        if job.get("status") in {"planned", "calendar_lookup_planned"}:
            transcript = str(job["transcript"])
            initial_plan = job.get("plan")
            plan = (
                job.get("resolved_plan")
                if job.get("status") == "calendar_lookup_planned"
                else initial_plan
            )
            if not isinstance(initial_plan, dict) or not isinstance(plan, dict):
                raise RuntimeError("Persisted calendar operation plan is invalid")
            if plan.get("action") in {"read", "lookup"}:
                raise RuntimeError("Calendar discovery plan reached the mutation stage")
            operations = plan.get("operations")
            operation_items = operations if isinstance(operations, list) else []
            operation_types = {
                str(item.get("type"))
                for item in operation_items
                if isinstance(item, dict)
            }
            will_apply = plan.get("action") == "execute"
            if will_apply:
                single_action = next(iter(operation_types), None)
                progress_action = (
                    single_action
                    if len(operation_types) == 1
                    and single_action in {"create", "update", "delete"}
                    else None
                )
                await self._edit_progress_best_effort(
                    chat_id=chat_id,
                    job=job,
                    html_text=format_progress_card(
                        "calendar",
                        action=progress_action,  # type: ignore[arg-type]
                        input_kind=input_kind,
                    ),
                )
                self.state.save_job(update_id, job)
            sent_time = datetime.fromtimestamp(
                int(job["sent_at"]), tz=timezone.utc
            ).astimezone(ZoneInfo(self.config.calendar_timezone))
            interaction_input, interaction_steps = _interaction_chain(
                (initial_plan, plan) if plan is not initial_plan else (plan,)
            )
            lookup_result = job.get("calendar_query_result")
            trusted_event_ids = None
            displayed_candidates = None
            if job.get("status") == "calendar_lookup_planned":
                persisted_candidates = job.get("calendar_lookup_candidates")
                candidates = (
                    persisted_candidates
                    if isinstance(persisted_candidates, list)
                    else (
                        lookup_result.get("events", [])[:_LOOKUP_DISPLAY_LIMIT]
                        if isinstance(lookup_result, dict)
                        and isinstance(lookup_result.get("events"), list)
                        else []
                    )
                )
                displayed_candidates = candidates
                trusted_event_ids = [
                    str(candidate["event_id"])
                    for candidate in candidates
                    if isinstance(candidate, dict) and candidate.get("event_id")
                ]
            try:
                execution = await pipeline.apply_plan(
                    source_update_id=update_id,
                    account=account,
                    owner_user_id=chat_id,
                    chat_id=chat_id,
                    transcript=transcript,
                    reference_time=sent_time,
                    plan=plan,
                    interaction_input=interaction_input,
                    interaction_steps=interaction_steps,
                    allowed_event_ids=trusted_event_ids,
                    displayed_candidates=displayed_candidates,
                )
            except CalendarOperationError as exc:
                if exc.retryable:
                    retry_count = int(job.get("calendar_write_retry_count", 0)) + 1
                    job["calendar_write_retry_count"] = retry_count
                    if retry_count < _CALENDAR_WRITE_RETRY_LIMIT:
                        # Leave the durable job at its mutation-ready stage.  The
                        # webhook runtime will replay the same Telegram update,
                        # which in turn reuses the operation journal and its
                        # stable per-item idempotency keys.
                        self.state.save_job(update_id, job)
                        LOGGER.warning(
                            "Calendar write outcome is uncertain for update %s; "
                            "retry %s/%s scheduled",
                            update_id,
                            retry_count,
                            _CALENDAR_WRITE_RETRY_LIMIT,
                        )
                        raise
                    LOGGER.error(
                        "Calendar write retry budget exhausted for update %s",
                        update_id,
                    )
                    job["final_html"] = format_error_card(
                        (
                            "Google Calendar не подтвердил итог операции после "
                            f"{_CALENDAR_WRITE_RETRY_LIMIT} попыток. Изменение "
                            "могло примениться полностью или частично. Не "
                            "отправляйте ту же команду повторно: сначала "
                            "проверьте событие в Google Calendar."
                        ),
                        transcript=transcript,
                        elapsed_seconds=self._elapsed(job),
                        calendar_unchanged=False,
                    )
                    job["final_reply_markup"] = None
                else:
                    LOGGER.warning(
                        "Calendar operation failed for update %s (%s)",
                        update_id,
                        "partial" if exc.partially_applied else "unchanged",
                    )
                    job["final_html"] = format_error_card(
                        str(exc),
                        transcript=transcript,
                        elapsed_seconds=self._elapsed(job),
                        calendar_unchanged=not (
                            exc.partially_applied or exc.outcome_uncertain
                        ),
                    )
                    job["final_reply_markup"] = None
            else:
                job.pop("calendar_write_retry_count", None)
                job["operation_id"] = execution.operation_id
                if execution.stage == "applied":
                    html_text, reply_markup = _success_card(
                        execution.record,
                        transcript=transcript,
                        elapsed_seconds=self._elapsed(job),
                    )
                    job["final_html"] = html_text
                    job["final_reply_markup"] = reply_markup
                elif execution.stage == "clarify":
                    question = str(
                        execution.record.get("clarification_question")
                        or "Уточните календарную команду."
                    )
                    candidates = displayed_candidates or []
                    job["final_html"] = (
                        format_lookup_clarify_card(
                            question,
                            candidates,
                            transcript=transcript,
                            elapsed_seconds=self._elapsed(job),
                        )
                        if candidates
                        else format_clarify_card(
                            question,
                            transcript=transcript,
                            elapsed_seconds=self._elapsed(job),
                        )
                    )
                    job["final_reply_markup"] = None
                else:
                    job["final_html"] = format_ignore_card(
                        transcript=transcript,
                        elapsed_seconds=self._elapsed(job),
                    )
                    job["final_reply_markup"] = None
            job["status"] = "final_ready"
            self.state.save_job(update_id, job)

        if job.get("status") == "final_ready":
            await self._finish_v2_job(
                update_id=update_id,
                chat_id=chat_id,
                bot_message_id=bot_message_id,
                job=job,
            )

    async def _process_text(
        self,
        *,
        update_id: int,
        account: str,
        chat_id: int,
        bot_message_id: int,
        sent_at: int,
        text: str,
    ) -> None:
        """Enter the legacy confirmation pipeline after transcription."""

        job = self.state.job(update_id)
        if job is None:
            job = {
                "account": account,
                "user_message_id": 0,
                "sent_at": sent_at,
                "input_kind": "text",
                "transcript": text,
                "status": "transcribed",
            }
            self.state.save_job(update_id, job)
        elif (
            job.get("account") != account
            or job.get("input_kind") != "text"
            or job.get("transcript") != text
        ):
            raise RuntimeError("Persisted text job contradicts its Telegram update")

        await self._process_voice(
            update_id=update_id,
            account=account,
            chat_id=chat_id,
            bot_message_id=bot_message_id,
            sent_at=sent_at,
            duration=0,
            file_size=None,
        )

    async def _process_voice(
        self,
        *,
        update_id: int,
        account: str,
        chat_id: int,
        bot_message_id: int,
        sent_at: int,
        duration: int,
        file_size: int | None,
    ) -> None:
        await self.bot.send_chat_action(chat_id)
        job = self.state.job(update_id)

        if job is None:
            match: dict[str, Any] | None = None
            for delay in (0, 0.5, 1, 2, 3):
                if delay:
                    await asyncio.sleep(delay)
                result = await self.gateway.read(
                    account,
                    "find_recent_outgoing_voice",
                    {
                        "chat_id": self.config.bot_chat_id,
                        "sent_at": sent_at,
                        "duration": duration,
                        "file_size": file_size,
                        "after_message_id": self.state.after_message_id(account),
                        "search_limit": 100,
                        "tolerance_seconds": 5,
                    },
                )
                if result.get("ok") and isinstance(result.get("match"), dict):
                    match = result["match"]
                    break
            if match is None:
                await self.bot.send_text(
                    chat_id,
                    "Не удалось сопоставить голосовое с пользовательской сессией. "
                    "Отправьте его ещё раз.",
                    reply_to_message_id=bot_message_id,
                )
                return

            job = {
                "account": account,
                "user_message_id": int(match["message_id"]),
                "sent_at": sent_at,
                "status": "found",
            }
            self.state.save_job(update_id, job)
        elif "sent_at" not in job:
            # Backward-compatible recovery for jobs created by an older version.
            job["sent_at"] = sent_at
            self.state.save_job(update_id, job)

        if job.get("status") == "found":
            result = await self.gateway.write(
                account,
                "transcribe_voice_message",
                {
                    "chat_id": self.config.bot_chat_id,
                    "message_id": int(job["user_message_id"]),
                    "wait_timeout": self.config.transcription_timeout_seconds,
                },
                request_id=f"stt-{account}-{update_id}",
                timeout=self.config.transcription_timeout_seconds + 30,
            )
            transcript = (
                str(result.get("text", ""))
                if result.get("ok") and result.get("status") == "completed"
                else ""
            )
            if transcript:
                # Persist the Telegram result before invoking Gemini. A restart
                # must not consume the transcription quota a second time.
                job["transcript"] = transcript
                job["status"] = "transcribed"
            else:
                job["reply"] = transcription_reply(result)
                job["next_chunk"] = 0
                job["status"] = "ready"
            self.state.save_job(update_id, job)

        if job.get("status") == "transcribed":
            transcript = str(job["transcript"])
            if self.gemini_available:
                sent_time = datetime.fromtimestamp(
                    int(job["sent_at"]), tz=timezone.utc
                ).astimezone(ZoneInfo(self.config.calendar_timezone))
                try:
                    intent = await self.gemini.extract_event(
                        transcript,
                        reference_time=sent_time,
                        account=account,
                    )
                except GeminiError:
                    LOGGER.warning(
                        "Gemini extraction failed for update %s; transcript fallback used",
                        update_id,
                    )
                    job["reply"] = (
                        f"Расшифровка:\n{transcript}\n\n"
                        "⚠️ Gemini сейчас не смогла разобрать событие."
                    )
                else:
                    job["intent"] = intent
                    # Persist model output before preparing a confirmation.  On
                    # restart we must not ask the model again and risk a changed
                    # intent contradicting an already-issued callback ID.
                    job["status"] = "intent_extracted"
                    self.state.save_job(update_id, job)
            else:
                job["reply"] = (
                    f"Расшифровка:\n{transcript}\n\n"
                    "⚠️ Gemini сейчас недоступна."
                )
            if job.get("status") != "intent_extracted":
                job["next_chunk"] = 0
                job["status"] = "ready"
                self.state.save_job(update_id, job)

        if job.get("status") == "intent_extracted":
            transcript = str(job["transcript"])
            intent = job.get("intent")
            if not isinstance(intent, dict):
                raise RuntimeError("Persisted calendar intent is invalid")

            create_footer: str | None = None
            job.pop("reply_markup", None)
            if (
                intent.get("action") == "create"
                and self.calendar_confirmation is not None
            ):
                prepared = self.calendar_confirmation.prepare(
                    source_update_id=update_id,
                    account=account,
                    owner_user_id=chat_id,
                    chat_id=chat_id,
                    intent=intent,
                )
                if prepared is not None:
                    job["confirmation_id"] = prepared.confirmation_id
                    job["reply_markup"] = prepared.reply_markup
                    create_footer = (
                        "Проверьте данные и нажмите «Добавить» либо «Отмена»."
                    )
                else:
                    threshold = self.calendar_confirmation.confidence_threshold
                    create_footer = (
                        f"⚠️ Уверенность модели ниже {threshold:.0%}. "
                        "Кнопка добавления не показана — уточните дату или время."
                    )

            job["reply"] = format_calendar_preview(
                transcript, intent, create_footer=create_footer
            )
            job["next_chunk"] = 0
            job["status"] = "ready"
            self.state.save_job(update_id, job)

        if job.get("status") == "ready":
            chunks = telegram_text_chunks(str(job["reply"]))
            next_chunk = int(job.get("next_chunk", 0))
            for index in range(next_chunk, len(chunks)):
                send_options: dict[str, Any] = {
                    "reply_to_message_id": (
                        bot_message_id if index == 0 else None
                    )
                }
                reply_markup = job.get("reply_markup")
                if index == 0 and isinstance(reply_markup, dict):
                    send_options["reply_markup"] = reply_markup
                await self.bot.send_text(chat_id, chunks[index], **send_options)
                job["next_chunk"] = index + 1
                self.state.save_job(update_id, job)
            job["status"] = "sent"
            self.state.save_job(update_id, job)


async def async_main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    # HTTPX's INFO message includes the complete Bot API URL, whose path embeds
    # the bot token. Keep transport logging below INFO in every run mode.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    load_dotenv(PROJECT_ROOT / ".env", override=False)
    config = Config()
    token = read_secret(
        environment=config.bot_token_environment,
        account=config.bot_keychain_account,
        service=config.bot_keychain_service,
    )
    webhook_secret: str | None = None
    if config.bot_update_mode == "webhook":
        webhook_secret = read_secret(environment=config.webhook_secret_environment)
    launcher = config.discover_gateway_launcher()
    state = StateStore(
        config.state_path,
        completed_update_limit=config.webhook_completed_ids_limit,
    )
    gemini_cli = GeminiCli(
        config.gemini_cli_path,
        model=config.gemini_cli_model,
        timeout_seconds=config.gemini_timeout_seconds,
        timezone=config.calendar_timezone,
    )
    gemini: GeminiProvider = gemini_cli
    gemini_api: GeminiApi | None = None
    try:
        gemini_api_key = read_secret(
            environment=config.gemini_api_key_environment,
            account=config.gemini_keychain_account,
            service=config.gemini_keychain_service,
        )
    except RuntimeError:
        LOGGER.warning("Gemini API key unavailable; Antigravity CLI fallback enabled")
    else:
        gemini_api = GeminiApi(
            gemini_api_key,
            model=config.gemini_model,
            timeout_seconds=config.gemini_timeout_seconds,
            timezone=config.calendar_timezone,
        )
        del gemini_api_key
        gemini = GeminiFallback(gemini_api, gemini_cli)

    try:
        async with BotApi(token) as bot:
            async with open_gateway(
                launcher, default_timeout=config.gateway_call_timeout_seconds
            ) as gateway:
                async with open_calendar_mcp(
                    config.calendar_mcp_binary_path,
                    account_mapping=config.calendar_mcp_account_mapping,
                    calendar_id=config.calendar_mcp_calendar_id,
                    default_timeout_seconds=config.calendar_mcp_timeout_seconds,
                    working_directory=config.calendar_mcp_working_directory,
                    env=config.calendar_mcp_env,
                ) as calendar:
                    await calendar.validate()
                    calendar_confirmation = build_calendar_confirmation(
                        config, calendar
                    )
                    calendar_operations = build_calendar_operations(
                        config, calendar
                    )
                    service = VoiceBotService(
                        config,
                        bot,
                        gateway,
                        state,
                        gemini,
                        calendar_confirmation,
                    )
                    # Assign after the legacy-compatible constructor call so
                    # deployment wrappers that decorate VoiceBotService do not
                    # need a synchronized signature change.
                    service.calendar_operations = calendar_operations
                    await service.initialize()
                    if config.bot_update_mode == "polling":
                        await service.run()
                    else:
                        if webhook_secret is None:  # pragma: no cover - guarded above
                            raise RuntimeError("Telegram webhook secret is unavailable")
                        webhook = WebhookRuntime(
                            service,
                            state,
                            secret_token=webhook_secret,
                            path=config.webhook_path,
                            host=config.webhook_listen_host,
                            port=config.webhook_listen_port,
                            health_path=config.webhook_health_path,
                            max_body_bytes=config.webhook_max_body_bytes,
                            retry_seconds=config.webhook_retry_seconds,
                        )
                        await webhook.start()
                        try:
                            if config.webhook_register_with_telegram:
                                # Telegram can deliver immediately after this call,
                                # so the durable listener and worker must exist first.
                                await bot.set_webhook(
                                    config.webhook_public_url, webhook_secret
                                )
                                LOGGER.info("Authenticated Telegram webhook is active")
                            else:
                                LOGGER.info(
                                    "Authenticated Telegram webhook listener is active; "
                                    "registration is managed externally"
                                )
                            await webhook.run_forever()
                        finally:
                            await webhook.close()
    finally:
        if gemini_api is not None:
            await gemini_api.aclose()


def main() -> None:
    asyncio.run(async_main())


if __name__ == "__main__":
    main()

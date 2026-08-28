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

from .bot_api import BotApi, BotApiError, BotApiFileError, read_secret
from .calendar import (
    CalendarClient,
    CalendarConnectionError,
    CalendarEventQueryResult,
)
from .calendar_mcp import open_calendar_mcp
from .config import PROJECT_ROOT, Config
from .confirmation import CalendarConfirmationPipeline, ConfirmationStore
from .codex_cli import (
    CodexCliAuthenticationError,
    CodexCliConfigurationError,
    CodexCliError,
    CodexCliQuotaError,
    CodexCliRunnerApi,
)
from .fast_read import plan_fast_calendar_read
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
    GeminiProviderChain,
    GeminiProviderStage,
    GeminiRateLimitError,
    PLANNER_MODEL_FIELD,
    planner_diagnostic_context,
)
from .gigachat import (
    GigaChatApi,
    GigaChatApiError,
    GigaChatAuthenticationError,
    GigaChatConfigurationError,
    GigaChatQuotaError,
    GigaChatRateLimitError,
    GigaChatRequestRejectedError,
)
from .intent import format_calendar_preview
from .operations import (
    CalendarOperationError,
    CalendarOperationPipeline,
    OperationStore,
)
from .openrouter import (
    OpenRouterApi,
    OpenRouterAuthenticationError,
    OpenRouterCreditError,
    OpenRouterRateLimitError,
    OpenRouterRequestRejectedError,
)
from .state import StateStore
from .text import telegram_text_chunks
from .ui import (
    CalendarAction,
    FieldChange,
    ProgressPhase,
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
from .vision import (
    GeminiVisionProvider,
    OpenAICompatibleVisionProvider,
    RapidOcrProvider,
    VisionProviderChain,
    VisionStage,
)
from .webhook import WebhookRuntime

LOGGER = logging.getLogger("tg_voice_transcriber_bot")
_PLANNER_STAGE_PRIORITIES = {
    "Codex Luna": 0,
    "Nemotron 3 Super": 1,
    "GLM 5.2 Free": 2,
    "Gemini 3.7 Flash": 3,
    "Gemini CLI": 3,
    "GigaChat 2 Max": 4,
}

START_TEXT = (
    "Пришлите голосовое сообщение или напишите календарную команду текстом. "
    "Для голосового я найду исходное сообщение через вашу пользовательскую "
    "Telegram-сессию и запрошу серверную расшифровку Telegram. ИИ-планировщик "
    "выделит событие и покажет предпросмотр. После проверки нажмите "
    "«Добавить», и событие попадёт в основной Google Calendar. Аудиофайл не "
    "скачивается. Без этого подтверждения календарь не изменяется."
)

START_TEXT_V2 = (
    "Пришлите голосовое, напишите календарную команду текстом или "
    "отправьте один скриншот. "
    "Голосовое Telegram расшифрует на своих серверах, а ИИ-планировщик "
    "с учётом последних команд сразу добавит, изменит или удалит событие в "
    "основном Google Calendar. В одном обновляемом сообщении будут видны ход "
    "обработки и извлечённые данные: расшифровка голоса либо описание и текст "
    "изображения. Затем оно станет итоговой карточкой со временем выполнения. "
    "Если результат не "
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
_MODEL_TITLE_LIMIT = 300
_MODEL_LOCATION_LIMIT = 300
_MODEL_DESCRIPTION_LIMIT = 500
_MODEL_RECURRENCE_LIMIT = 500
_FAST_READ_MODEL_LABEL = "Без LLM · быстрый разбор"
_IMAGE_INPUT_KINDS = frozenset({"image", "text_and_image"})
_SUPPORTED_TELEGRAM_IMAGE_MIME_TYPES = frozenset(
    {"image/jpeg", "image/png", "image/webp"}
)
_PLANNER_IMAGE_DESCRIPTION_BYTES = 1_536
_PLANNER_IMAGE_VISIBLE_TEXT_BYTES = 6_656


class _UnknownEventReference(ValueError):
    """The model selected an event reference outside the server allowlist."""


def _nonnegative_telegram_int(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value


def _telegram_image(message: Mapping[str, Any]) -> dict[str, Any] | None:
    """Return the largest Telegram photo or one image document."""

    raw_photos = message.get("photo")
    if isinstance(raw_photos, Sequence) and not isinstance(
        raw_photos, (str, bytes, bytearray)
    ):
        photos = [
            item
            for item in raw_photos
            if isinstance(item, Mapping)
            and isinstance(item.get("file_id"), str)
            and str(item["file_id"]).strip()
        ]
        if photos:

            def photo_rank(item: Mapping[str, Any]) -> tuple[int, int]:
                width = _nonnegative_telegram_int(item.get("width")) or 0
                height = _nonnegative_telegram_int(item.get("height")) or 0
                file_size = _nonnegative_telegram_int(item.get("file_size")) or 0
                return width * height, file_size

            largest = max(
                photos,
                key=photo_rank,
            )
            file_size = _nonnegative_telegram_int(largest.get("file_size"))
            return {
                "file_id": str(largest["file_id"]),
                "mime_type": "image/jpeg",
                "file_size": file_size,
            }

    document = message.get("document")
    if not isinstance(document, Mapping):
        return None
    mime_type = document.get("mime_type")
    file_id = document.get("file_id")
    if (
        not isinstance(mime_type, str)
        or mime_type.casefold() not in _SUPPORTED_TELEGRAM_IMAGE_MIME_TYPES
        or not isinstance(file_id, str)
        or not file_id.strip()
    ):
        return None
    file_size = _nonnegative_telegram_int(document.get("file_size"))
    return {
        "file_id": file_id,
        "mime_type": mime_type.casefold(),
        "file_size": file_size,
    }


def _vision_observation(result: Any) -> tuple[dict[str, str], str | None]:
    """Convert a VisionResult-like value into bounded durable planner input."""

    def value(name: str, limit: int) -> str:
        raw = (
            result.get(name)
            if isinstance(result, Mapping)
            else getattr(result, name, "")
        )
        return str(raw or "").strip()[:limit]

    description = value("description", 4_000)
    visible_text = value("visible_text", 16_000)
    provider = value("provider", 128) or "unknown"
    model = value("model", 200) or None
    used_local_ocr = (
        result.get("used_local_ocr")
        if isinstance(result, Mapping)
        else getattr(result, "used_local_ocr", False)
    )
    return (
        {
            "description": description,
            "visible_text": visible_text,
            "source": provider,
            "mode": "local_ocr" if used_local_ocr is True else "vision",
        },
        model,
    )


def _job_image_observations(job: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
    raw = job.get("image_observations", [])
    if not isinstance(raw, list) or any(not isinstance(item, Mapping) for item in raw):
        raise RuntimeError("Persisted image observations are invalid")
    bounded: list[dict[str, Any]] = []
    for item in raw:
        copied = deepcopy(dict(item))
        description = copied.get("description")
        visible_text = copied.get("visible_text")
        if isinstance(description, str):
            copied["description"] = _truncate_utf8(
                description, _PLANNER_IMAGE_DESCRIPTION_BYTES
            )
        if isinstance(visible_text, str):
            copied["visible_text"] = _truncate_utf8(
                visible_text, _PLANNER_IMAGE_VISIBLE_TEXT_BYTES
            )
        bounded.append(copied)
    return tuple(bounded)


def _job_progress_card(
    job: Mapping[str, Any],
    phase: ProgressPhase,
    *,
    action: CalendarAction | None = None,
) -> str:
    """Render a phase with the exact bounded evidence sent to the planner."""

    input_kind = job.get("input_kind")
    if input_kind not in {"voice", "text", "image", "text_and_image"}:
        raise RuntimeError("Persisted calendar job has an invalid input kind")
    observations = _job_image_observations(job)
    observation = observations[0] if observations else {}
    return format_progress_card(
        phase,
        action=action,
        input_kind=input_kind,
        transcript=job.get("transcript"),
        image_description=observation.get("description"),
        image_visible_text=observation.get("visible_text"),
    )


def _truncate_utf8(value: str, max_bytes: int) -> str:
    encoded = value.encode("utf-8")
    if len(encoded) <= max_bytes:
        return value
    marker = "\n[… сокращено …]"
    marker_bytes = marker.encode("utf-8")
    prefix = encoded[: max(0, max_bytes - len(marker_bytes))]
    return prefix.decode("utf-8", errors="ignore").rstrip() + marker


def _job_memory_text(job: Mapping[str, Any]) -> str:
    """Keep bounded image evidence in durable follow-up conversation memory."""

    transcript = str(job.get("transcript") or "").strip()
    source_kind = job.get("source_input_kind", job.get("input_kind"))
    if source_kind not in _IMAGE_INPUT_KINDS:
        return transcript

    evidence: list[str] = []
    for observation in _job_image_observations(job):
        description = str(observation.get("description") or "").strip()[:150]
        visible_text = str(observation.get("visible_text") or "").strip()[:650]
        if visible_text:
            evidence.append(f"Видимый текст: {visible_text}")
        if description:
            evidence.append(f"Описание: {description}")

    parts: list[str] = []
    if transcript:
        # Preserve room for OCR facts needed by follow-ups even when the caption
        # itself is long. The full current-turn evidence remains in the durable
        # planner interaction; this is only compact cross-turn memory.
        parts.append(transcript[:300] if evidence else transcript)
    if evidence:
        parts.append("Данные изображения:\n" + "\n".join(evidence))
    memory = "\n\n".join(parts)
    return memory if len(memory) <= 1_000 else memory[:999].rstrip() + "…"


def _planner_model_name(value: Mapping[str, Any] | None) -> str | None:
    if not isinstance(value, Mapping):
        return None
    model_name = value.get(PLANNER_MODEL_FIELD)
    if not isinstance(model_name, str) or not model_name.strip():
        return None
    return model_name.strip()


def _job_planner_model(job: Mapping[str, Any]) -> str | None:
    model_name = job.get("planner_model")
    if isinstance(model_name, str) and model_name.strip():
        return model_name.strip()
    for key in ("resolved_plan", "plan"):
        candidate = job.get(key)
        if isinstance(candidate, Mapping):
            model_name = _planner_model_name(candidate)
            if model_name is not None:
                return model_name
    return None


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


def build_vision_pipeline(
    config: Config,
    *,
    openrouter_api_key: str | None,
    gemini_api_key: str | None,
) -> VisionProviderChain:
    """Build cloud narration fallbacks followed by mandatory local OCR."""

    stages: list[VisionStage] = []
    if openrouter_api_key is not None:
        endpoint_url = "https://openrouter.ai/api/v1/chat/completions"
        stages.extend(
            (
                VisionStage(
                    OpenAICompatibleVisionProvider(
                        openrouter_api_key,
                        endpoint_url=endpoint_url,
                        provider_name="OpenRouter Vision",
                        model=config.openrouter_vision_model,
                        timeout_seconds=config.openrouter_vision_timeout_seconds,
                        # The current free Gemma endpoints expose JSON mode but
                        # do not guarantee strict-schema enforcement. The local
                        # parser still rejects every field outside our neutral
                        # description/visible_text contract.
                        strict_json_schema=False,
                    ),
                    config.openrouter_vision_timeout_seconds,
                ),
                VisionStage(
                    OpenAICompatibleVisionProvider(
                        openrouter_api_key,
                        endpoint_url=endpoint_url,
                        provider_name="OpenRouter Vision",
                        model=config.openrouter_vision_fallback_model,
                        timeout_seconds=(
                            config.openrouter_vision_fallback_timeout_seconds
                        ),
                        strict_json_schema=False,
                    ),
                    config.openrouter_vision_fallback_timeout_seconds,
                ),
            )
        )
    if gemini_api_key is not None:
        stages.append(
            VisionStage(
                GeminiVisionProvider(
                    gemini_api_key,
                    model=config.gemini_vision_model,
                    timeout_seconds=config.gemini_vision_timeout_seconds,
                ),
                config.gemini_vision_timeout_seconds,
            )
        )

    pipeline = VisionProviderChain(
        stages,
        local_ocr=RapidOcrProvider(
            model_root_dir=config.vision_ocr_model_dir,
        ),
        local_timeout_seconds=config.vision_local_ocr_timeout_seconds,
        max_image_bytes=config.vision_max_image_bytes,
        max_pixels=config.vision_max_image_pixels,
        max_description_chars=config.vision_max_description_chars,
        max_visible_text_chars=config.vision_max_visible_text_chars,
    )
    LOGGER.info(
        "Image understanding configured; cloud_stages=%d local_ocr=enabled",
        len(stages),
    )
    return pipeline


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
    record: dict[str, Any],
    *,
    transcript: str,
    elapsed_seconds: float,
    model_name: str | None = None,
) -> tuple[str, dict[str, Any]]:
    action = _record_action(record)
    events = _record_events(record)
    operation_id = str(record["operation_id"])
    best_effort_undo = _record_undo_is_best_effort(record)
    if action == "create":
        html = format_create_card(
            events,
            transcript=transcript,
            elapsed_seconds=elapsed_seconds,
            model_name=model_name,
        )
    elif action == "update":
        html = format_update_card(
            events,
            transcript=transcript,
            elapsed_seconds=elapsed_seconds,
            changes=_record_changes(record),
            model_name=model_name,
        )
    elif action == "delete":
        html = format_delete_card(
            events,
            transcript=transcript,
            elapsed_seconds=elapsed_seconds,
            best_effort_undo=best_effort_undo,
            model_name=model_name,
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
            model_name=model_name,
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


def _compact_lookup_candidates(
    candidates: Sequence[Mapping[str, Any]], *, timezone_name: str
) -> tuple[list[dict[str, Any]], dict[str, str], dict[str, str]]:
    """Expose a minimal, per-request alias set to the lookup second pass.

    Provider IDs and metadata stay server-side.  The model sees only short
    references (``c1``, ``c2``, ...) and the fields needed to distinguish the
    rows rendered to the owner.
    """

    compact: list[dict[str, Any]] = []
    event_id_by_ref: dict[str, str] = {}
    series_event_id_by_ref: dict[str, str] = {}
    zone = ZoneInfo(timezone_name)

    def bounded(value: Any, limit: int) -> str | None:
        if not isinstance(value, str):
            return None
        return value if len(value) <= limit else value[: limit - 1] + "…"

    def normalized_series_context(
        value: Any,
    ) -> dict[str, Any] | None:
        if not isinstance(value, Mapping):
            return None
        all_day = bool(value.get("all_day"))
        start_at = value.get("start_at")
        end_at = value.get("end_at")
        if not isinstance(start_at, str) or not isinstance(end_at, str):
            raise RuntimeError("Persisted Calendar series has no time range")
        if not all_day:
            normalized: list[str] = []
            for field, raw_value in (("start_at", start_at), ("end_at", end_at)):
                try:
                    parsed = datetime.fromisoformat(raw_value)
                except ValueError:
                    raise RuntimeError(
                        f"Persisted Calendar series has invalid {field} timestamp"
                    ) from None
                if parsed.tzinfo is None or parsed.utcoffset() is None:
                    raise RuntimeError(
                        f"Persisted Calendar series has naive {field} timestamp"
                    )
                normalized.append(parsed.astimezone(zone).isoformat())
            start_at, end_at = normalized
        rules = value.get("recurrence_rrules")
        recurrence = None
        if (
            isinstance(rules, Sequence)
            and not isinstance(rules, (str, bytes, bytearray))
            and rules
        ):
            recurrence = next(
                (
                    str(rule)
                    for rule in rules
                    if str(rule).startswith("RRULE:")
                ),
                None,
            )
        elif isinstance(value.get("recurrence_rrule"), str):
            recurrence = str(value["recurrence_rrule"])
        return {
            "start_at": start_at,
            "end_at": end_at,
            "all_day": all_day,
            "timezone": timezone_name,
            "recurrence_rrule": bounded(recurrence, _MODEL_RECURRENCE_LIMIT),
        }

    for display_index, candidate in enumerate(candidates, start=1):
        event_id = str(candidate.get("event_id") or "")
        if not event_id:
            raise RuntimeError("Persisted Calendar candidate has no event ID")
        event_ref = f"c{display_index}"
        event_id_by_ref[event_ref] = event_id
        series_event_id_by_ref[event_ref] = str(
            candidate.get("recurring_event_id") or event_id
        )
        all_day = bool(candidate.get("all_day"))
        start_at = candidate.get("start_at")
        end_at = candidate.get("end_at")
        if not all_day:
            normalized_times: list[str] = []
            for field, value in (("start_at", start_at), ("end_at", end_at)):
                if not isinstance(value, str):
                    raise RuntimeError(
                        f"Persisted Calendar candidate has no {field} timestamp"
                    )
                try:
                    parsed = datetime.fromisoformat(value)
                except ValueError:
                    raise RuntimeError(
                        f"Persisted Calendar candidate has invalid {field} timestamp"
                    ) from None
                if parsed.tzinfo is None or parsed.utcoffset() is None:
                    raise RuntimeError(
                        f"Persisted Calendar candidate has naive {field} timestamp"
                    )
                normalized_times.append(parsed.astimezone(zone).isoformat())
            start_at, end_at = normalized_times
        series_context = normalized_series_context(candidate.get("series_context"))
        recurrence = candidate.get("recurrence_rrule")
        if recurrence is None:
            recurrence_rules = candidate.get("recurrence_rrules")
            if (
                isinstance(recurrence_rules, Sequence)
                and not isinstance(recurrence_rules, (str, bytes, bytearray))
                and recurrence_rules
            ):
                recurrence = next(
                    (
                        str(rule)
                        for rule in recurrence_rules
                        if str(rule).startswith("RRULE:")
                    ),
                    None,
                )
        if recurrence is None and series_context is not None:
            recurrence = series_context.get("recurrence_rrule")
        item: dict[str, Any] = {
            "event_id": event_ref,
            "display_index": display_index,
            "title": bounded(
                candidate.get("title") or "Без названия", _MODEL_TITLE_LIMIT
            ),
            "start_at": start_at,
            "end_at": end_at,
            "all_day": all_day,
            "timezone": timezone_name,
            "location": bounded(
                candidate.get("location"), _MODEL_LOCATION_LIMIT
            ),
            "description": bounded(
                candidate.get("description"), _MODEL_DESCRIPTION_LIMIT
            ),
            "recurrence_rrule": bounded(
                recurrence, _MODEL_RECURRENCE_LIMIT
            ),
            "recurring": bool(recurrence or candidate.get("recurring_event_id")),
            "recurring_instance": bool(candidate.get("recurring_event_id")),
            "status": str(candidate.get("status") or "confirmed"),
        }
        if series_context is not None:
            item["series_context"] = series_context
        compact.append(item)
    return compact, event_id_by_ref, series_event_id_by_ref


async def _hydrate_lookup_series_candidates(
    pipeline: CalendarOperationPipeline,
    *,
    account: str,
    candidates: Sequence[Mapping[str, Any]],
    limit: int,
) -> list[dict[str, Any]]:
    """Attach provider-fresh recurrence data without exposing master IDs.

    Google returns expanded occurrences from list/search.  Those rows carry a
    master ID but normally omit the master's RRULE.  A relative command such
    as “убери пятницу” is therefore unsafe to plan until the exact master has
    been read.  Only the bounded rows that can become mutation candidates are
    hydrated, and duplicate occurrences of one series share a single read.
    """

    hydrated = [deepcopy(dict(candidate)) for candidate in candidates]
    master_contexts: dict[str, dict[str, Any]] = {}
    visible = 0
    for candidate in hydrated:
        if candidate.get("status") not in {None, "confirmed", "tentative"}:
            continue
        if visible >= limit:
            break
        visible += 1
        master_id = str(candidate.get("recurring_event_id") or "")
        event_id = str(candidate.get("event_id") or "")
        if not master_id or master_id == event_id:
            continue
        series_context = candidate.get("series_context")
        if isinstance(series_context, Mapping):
            continue
        if master_id not in master_contexts:
            master = await pipeline.read_event_snapshot(
                account=account, event_id=master_id
            )
            rules = master.get("recurrence_rrules")
            if not (
                isinstance(rules, Sequence)
                and not isinstance(rules, (str, bytes, bytearray))
                and rules
            ):
                # Missing recurrence metadata is provider state, not a reason
                # for the bot to veto the model's next operation.  The exact
                # visible occurrence remains available as a mutation target.
                continue
            master_contexts[master_id] = {
                "start_at": master.get("start_at"),
                "end_at": master.get("end_at"),
                "all_day": bool(master.get("all_day")),
                "timezone": master.get("timezone"),
                "recurrence_rrules": list(rules),
            }
        candidate["series_context"] = deepcopy(master_contexts[master_id])
    return hydrated


def _resolve_plan_event_references(
    plan: Mapping[str, Any],
    event_id_by_ref: Mapping[str, str],
    series_event_id_by_ref: Mapping[str, str] | None = None,
    recurring_event_refs: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Translate model-visible aliases to trusted provider event IDs."""

    resolved = deepcopy(dict(plan))
    operations = resolved.get("operations")
    if not isinstance(operations, list):
        return resolved
    # Kept for API compatibility with callers that already computed this
    # metadata.  Scope is now trusted as model output instead of being turned
    # into a bot-authored clarification guard.
    _ = recurring_event_refs

    for operation in operations:
        if not isinstance(operation, dict):
            continue
        operation_type = operation.get("type")
        if operation_type not in {"update", "delete"}:
            continue
        event_ref = operation.get("target_event_id")
        if not isinstance(event_ref, str) or event_ref not in event_id_by_ref:
            raise _UnknownEventReference("unknown model event reference")
        recurrence_scope = operation.get("recurrence_scope")
        target_map = event_id_by_ref
        if recurrence_scope == "series" and series_event_id_by_ref is not None:
            target_map = series_event_id_by_ref
        if event_ref not in target_map:
            raise _UnknownEventReference("unknown model event reference")
        operation["target_event_id"] = str(target_map[event_ref])
    return resolved


def _planner_timed_out(exc: GeminiError) -> bool:
    diagnostic = f"{type(exc).__name__}: {exc}".casefold()
    return any(
        marker in diagnostic
        for marker in ("timeout", "timed out", "deadline", "readtimeout")
    )


def _planner_failure_copy(exc: GeminiError, *, matching: bool = False) -> str:
    if isinstance(exc, CodexCliAuthenticationError):
        return (
            "Codex отклонил ChatGPT-сессию, а резервные модели тоже не "
            "ответили. Обновите вход Codex на сервере и повторите команду."
        )
    if isinstance(exc, CodexCliConfigurationError):
        return (
            "Codex runner отклонил настройки, а резервные модели тоже не "
            "ответили. Проверьте серверную конфигурацию и повторите команду."
        )
    if isinstance(exc, CodexCliQuotaError):
        return (
            "Лимит Codex по ChatGPT-подписке исчерпан, а резервные модели "
            "тоже не ответили. Дождитесь сброса лимита и повторите команду."
        )
    if isinstance(exc, GigaChatAuthenticationError):
        return (
            "GigaChat отклонил авторизацию, а резервные модели тоже не "
            "ответили. Проверьте credentials и scope, затем повторите команду."
        )
    if isinstance(exc, GigaChatConfigurationError):
        return (
            "GigaChat отклонил настройки модели или подключения, а резервные "
            "модели тоже не ответили. Проверьте конфигурацию и повторите команду."
        )
    if isinstance(exc, GigaChatRequestRejectedError):
        return (
            "GigaChat отклонил запрос, а резервные модели тоже не ответили. "
            "Google Calendar не изменён; уточните формулировку и повторите."
        )
    if isinstance(exc, GigaChatQuotaError):
        return (
            "GigaChat отклонил запрос из-за квоты или тарифа, а резервные "
            "модели тоже не ответили. Проверьте доступ и повторите команду."
        )
    if isinstance(exc, GigaChatRateLimitError):
        return (
            "Провайдеры ИИ-планировщика временно ограничили запросы. "
            "Подождите немного и повторите команду."
        )
    if isinstance(exc, OpenRouterAuthenticationError):
        return (
            "OpenRouter отклонил API-ключ, а резервные модели тоже не ответили. "
            "Проверьте ключ и его ограничения, затем повторите команду."
        )
    if isinstance(exc, OpenRouterRequestRejectedError):
        return (
            "OpenRouter отклонил запрос, а резервные модели тоже не ответили. "
            "Google Calendar не изменён; уточните формулировку и повторите."
        )
    if isinstance(exc, OpenRouterCreditError):
        return (
            "OpenRouter отклонил запрос из-за лимита ключа или баланса, "
            "а резервные модели тоже не ответили. Проверьте аккаунт и повторите."
        )
    if isinstance(exc, OpenRouterRateLimitError):
        return (
            "Провайдеры ИИ-планировщика временно ограничили запросы. "
            "Подождите немного и повторите команду."
        )
    if isinstance(exc, GeminiRateLimitError):
        return (
            "Провайдер модели отклонил запрос из-за временного лимита или "
            "исчерпанной квоты API. Подождите немного и повторите команду."
        )
    if _planner_timed_out(exc):
        return (
            "ИИ-планировщик не успел обработать команду за отведённое время. "
            "Попробуйте повторить её через несколько минут."
        )
    if matching:
        return (
            "ИИ-планировщик не смог выбрать точное событие. Уточните название "
            "или время новым сообщением."
        )
    return (
        "ИИ-планировщик не смог надёжно разобрать календарную команду. "
        "Попробуйте уточнить её новым сообщением."
    )


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
    """Flatten one or two stateless planner exchanges for durable replay."""

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
        vision: Any | None = None,
    ) -> None:
        self.config = config
        self.bot = bot
        self.gateway = gateway
        self.state = state
        self.gemini = gemini
        self.calendar_confirmation = calendar_confirmation
        self.calendar_operations = calendar_operations
        self.vision = vision
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
            with planner_diagnostic_context("startup-validation"):
                await self.gemini.validate()
            if (
                self.config.bot_update_mode == "webhook"
                and isinstance(self.gemini, GeminiProviderChain)
                and not self.gemini.terminal_available
            ):
                raise self.gemini.terminal_validation_error or GeminiError(
                    "Direct Gemini terminal provider is unavailable"
                )
            if (
                isinstance(self.gemini, GeminiFallback)
                and not self.gemini.primary_available
                and self.calendar_operations is not None
            ):
                raise self.gemini.primary_validation_error or GeminiError(
                    "Primary calendar planner is unavailable"
                )
        except GeminiError as exc:
            self.gemini_available = False
            if self.calendar_operations is not None:
                LOGGER.error(
                    "AI planner validation failed; error_type=%s error=%s; "
                    "calendar bot startup aborted",
                    type(exc).__name__,
                    str(exc),
                )
                raise
            LOGGER.warning(
                "AI planner validation failed; error_type=%s error=%s; "
                "transcript fallback enabled",
                type(exc).__name__,
                str(exc),
            )
        if self.vision is not None:
            validate_vision = getattr(self.vision, "validate", None)
            if validate_vision is not None:
                await validate_vision()
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
            available_provider_names = getattr(
                self.gemini, "available_provider_names", ()
            )
            fallback_active = isinstance(
                self.gemini, GeminiFallback
            ) and not self.gemini.primary_available
            if self.gemini_available and available_provider_names:
                gemini_status = (
                    "ИИ-планировщик доступен: "
                    + " → ".join(available_provider_names)
                )
            elif self.gemini_available and fallback_active:
                gemini_status = (
                    "Основной ИИ-провайдер недоступен; активен резервный Gemini"
                )
            elif self.gemini_available:
                gemini_status = "ИИ-планировщик доступен"
            else:
                gemini_status = (
                    "ИИ-планировщик сейчас недоступен; останется обычная расшифровка"
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
                    " Пришлите голосовое, текстовую команду или один скриншот."
                    if self.enabled_accounts is None
                    or account in self.enabled_accounts
                    else " Текстовые команды и скриншоты доступны без Telegram-сессии."
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

        image = _telegram_image(message)
        if image is not None:
            if self.calendar_operations is None:
                await self.bot.send_text(
                    chat_id,
                    "Обработка скриншотов доступна только в режиме Google Calendar.",
                    reply_to_message_id=bot_message_id,
                )
                return
            if message.get("media_group_id") is not None:
                await self.bot.send_text(
                    chat_id,
                    "Пока отправьте один скриншот отдельным сообщением, не альбомом.",
                    reply_to_message_id=bot_message_id,
                )
                return
            raw_caption = message.get("caption")
            caption = raw_caption if isinstance(raw_caption, str) else ""
            await self._process_image_v2(
                update_id=update_id,
                account=account,
                chat_id=chat_id,
                bot_message_id=bot_message_id,
                sent_at=int(message.get("date", 0) or 0),
                file_id=str(image["file_id"]),
                mime_type=str(image["mime_type"]),
                file_size=(
                    int(image["file_size"])
                    if image.get("file_size") is not None
                    else None
                ),
                caption=caption,
            )
            return

        voice = message.get("voice")
        if not isinstance(voice, dict):
            await self.bot.send_text(
                chat_id,
                "Пришлите голосовое сообщение, записанное в Telegram, или "
                "напишите календарную команду текстом или отправьте один скриншот.",
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

    async def _process_image_v2(
        self,
        *,
        update_id: int,
        account: str,
        chat_id: int,
        bot_message_id: int,
        sent_at: int,
        file_id: str,
        mime_type: str,
        file_size: int | None,
        caption: str,
    ) -> None:
        """Download and understand one Telegram image before shared planning."""

        input_kind = "text_and_image" if caption.strip() else "image"
        job = self.state.job(update_id)
        if job is None:
            job = {
                "account": account,
                # Images do not advance the MTProto voice matching cursor.
                "user_message_id": 0,
                "sent_at": sent_at,
                "started_at": time.time(),
                "started_monotonic": time.monotonic(),
                "input_kind": input_kind,
                "source_input_kind": input_kind,
                "transcript": caption,
                "image_file_id": file_id,
                "image_mime_type": mime_type,
                "image_file_size": file_size,
                "status": "image_pending",
            }
            self.state.save_job(update_id, job)
        elif (
            job.get("account") != account
            or job.get("source_input_kind", job.get("input_kind")) != input_kind
            or job.get("transcript") != caption
            or job.get("image_file_id") != file_id
            or job.get("image_mime_type") != mime_type
            or job.get("image_file_size") != file_size
        ):
            raise RuntimeError("Persisted image job contradicts its Telegram update")

        await self._process_voice_v2(
            update_id=update_id,
            account=account,
            chat_id=chat_id,
            bot_message_id=bot_message_id,
            sent_at=sent_at,
            duration=0,
            file_size=file_size,
        )

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
        raw_input_kind = job.get("input_kind")
        if raw_input_kind not in {"voice", "text", "image", "text_and_image"}:
            raise RuntimeError("Persisted calendar job has an invalid input kind")
        input_kind = str(raw_input_kind)
        if int(job.get("status_message_id", 0) or 0) <= 0:
            initial_stage = "matching"
            initial_fast_read = False
            if input_kind == "text":
                initial_stage = "gemini"
                initial_transcript = str(job.get("transcript") or "")
                initial_time = datetime.fromtimestamp(
                    int(job["sent_at"]), tz=timezone.utc
                ).astimezone(ZoneInfo(self.config.calendar_timezone))
                if plan_fast_calendar_read(
                    initial_transcript,
                    reference_time=initial_time,
                    timezone=self.config.calendar_timezone,
                ) is not None:
                    initial_fast_read = True
            elif input_kind in _IMAGE_INPUT_KINDS:
                initial_stage = "image_downloading"
            matching_html = (
                _job_progress_card(job, "fast_read")
                if initial_fast_read
                else _job_progress_card(
                    job,
                    initial_stage,  # type: ignore[arg-type]
                )
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

        if job.get("status") == "image_pending":
            transcript = str(job.get("transcript") or "")
            if self.vision is None:
                LOGGER.warning(
                    "Image understanding unavailable for update %s; "
                    "caption fallback=%s",
                    update_id,
                    bool(transcript.strip()),
                )
                if transcript.strip():
                    job["image_observations"] = []
                    job["vision_model"] = None
                    job["input_kind"] = "text"
                    input_kind = "text"
                    job["status"] = "transcribed"
                else:
                    job["final_html"] = format_error_card(
                        "Сейчас не удалось распознать изображение. "
                        "Добавьте к нему подпись с датой, временем "
                        "и названием события.",
                        elapsed_seconds=self._elapsed(job),
                    )
                    job["final_reply_markup"] = None
                    job["status"] = "final_ready"
                self.state.save_job(update_id, job)
            else:
                image_limit = self.config.vision_max_image_bytes
                try:
                    declared_size = job.get("image_file_size")
                    if (
                        isinstance(declared_size, int)
                        and not isinstance(declared_size, bool)
                        and declared_size > image_limit
                    ):
                        raise BotApiFileError(
                            "Telegram file exceeds the configured size limit"
                        )
                    file_info = await self.bot.get_file(
                        str(job["image_file_id"]),
                        max_file_size=image_limit,
                    )
                    file_path = file_info.get("file_path")
                    if not isinstance(file_path, str) or not file_path.strip():
                        raise BotApiFileError(
                            "Telegram returned invalid image metadata"
                        )
                    image_data = await self.bot.download_file(
                        file_path,
                        max_bytes=image_limit,
                    )
                except BotApiFileError:
                    LOGGER.warning(
                        "Telegram image rejected for update %s; "
                        "reason=file_validation caption_fallback=%s",
                        update_id,
                        bool(transcript.strip()),
                    )
                    if transcript.strip():
                        job["image_observations"] = []
                        job["vision_model"] = None
                        job["input_kind"] = "text"
                        input_kind = "text"
                        job["status"] = "transcribed"
                        self.state.save_job(update_id, job)
                    else:
                        job["final_html"] = format_error_card(
                            "Изображение не удалось загрузить или оно превышает "
                            "допустимый размер. Отправьте скриншот размером до "
                            f"{image_limit // (1024 * 1024)} МБ.",
                            elapsed_seconds=self._elapsed(job),
                        )
                        job["final_reply_markup"] = None
                        job["status"] = "final_ready"
                        self.state.save_job(update_id, job)
                        await self._finish_v2_job(
                            update_id=update_id,
                            chat_id=chat_id,
                            bot_message_id=bot_message_id,
                            job=job,
                        )
                        return
                else:
                    await self._edit_progress_best_effort(
                        chat_id=chat_id,
                        job=job,
                        html_text=_job_progress_card(
                            job,
                            "vision",
                        ),
                    )
                    self.state.save_job(update_id, job)
                    # Keep the dependency optional for transcript-only deployments;
                    # image-enabled runtimes provide the concrete module and chain.
                    from .vision import VisionError, VisionImage

                    try:
                        vision_result = await self.vision.analyze(
                            VisionImage(
                                data=image_data,
                                mime_type=str(job["image_mime_type"]),
                            )
                        )
                        observation, vision_model = _vision_observation(vision_result)
                        if (
                            not observation["description"]
                            and not observation["visible_text"]
                        ):
                            raise VisionError(
                                "Vision providers returned no image evidence"
                            )
                    except VisionError as exc:
                        LOGGER.warning(
                            "Image understanding failed for update %s; error_type=%s; "
                            "caption fallback=%s",
                            update_id,
                            type(exc).__name__,
                            bool(transcript.strip()),
                        )
                        if transcript.strip():
                            job["image_observations"] = []
                            job["vision_model"] = None
                            job["input_kind"] = "text"
                            input_kind = "text"
                            job["status"] = "transcribed"
                        else:
                            job["final_html"] = format_error_card(
                                "Не удалось прочитать содержимое изображения. "
                                "Попробуйте отправить его файлом или "
                                "добавьте текстовую подпись.",
                                elapsed_seconds=self._elapsed(job),
                            )
                            job["final_reply_markup"] = None
                            job["status"] = "final_ready"
                    else:
                        job["image_observations"] = [observation]
                        job["vision_model"] = vision_model
                        job["status"] = "transcribed"
                        LOGGER.info(
                            "Image understanding succeeded for update %s; provider=%s "
                            "mode=%s description_chars=%d visible_text_chars=%d",
                            update_id,
                            observation["source"],
                            observation["mode"],
                            len(observation["description"]),
                            len(observation["visible_text"]),
                        )
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
                html_text=_job_progress_card(job, "transcribing"),
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
            sent_time = datetime.fromtimestamp(
                int(job["sent_at"]), tz=timezone.utc
            ).astimezone(ZoneInfo(self.config.calendar_timezone))
            fast_plan = (
                None
                if input_kind in _IMAGE_INPUT_KINDS
                else plan_fast_calendar_read(
                    transcript,
                    reference_time=sent_time,
                    timezone=self.config.calendar_timezone,
                )
            )
            if fast_plan is not None:
                # Exact, bounded read phrases do not need an LLM round trip.
                fast_plan[PLANNER_MODEL_FIELD] = _FAST_READ_MODEL_LABEL
                job["plan"] = fast_plan
                job["planner_model"] = _FAST_READ_MODEL_LABEL
                job["fast_read"] = True
                job["status"] = "planned"
                self.state.save_job(update_id, job)
            elif not self.gemini_available:
                job["final_html"] = format_error_card(
                    "ИИ-планировщик сейчас недоступен. Команда сохранена в этой карточке.",
                    transcript=_job_memory_text(job),
                    elapsed_seconds=self._elapsed(job),
                )
                job["final_reply_markup"] = None
                job["status"] = "final_ready"
                self.state.save_job(update_id, job)
            else:
                await self._edit_progress_best_effort(
                    chat_id=chat_id,
                    job=job,
                    html_text=_job_progress_card(job, "gemini"),
                )
                self.state.save_job(update_id, job)
                context = pipeline.context(
                    account=account, chat_id=chat_id, now=sent_time
                )
                planning_started = time.monotonic()
                try:
                    with planner_diagnostic_context(
                        f"telegram-update-{update_id}:initial"
                    ):
                        plan = await self.gemini.plan_calendar_actions(
                            transcript,
                            reference_time=sent_time,
                            account=account,
                            application_state=context.application_state,
                            recent_conversation=context.recent_conversation,
                            history_steps=context.history_steps,
                            input_kind=input_kind,
                            image_observations=_job_image_observations(job),
                        )
                    selected_model = _planner_model_name(plan)
                    if selected_model is not None:
                        job["planner_model"] = selected_model
                    plan = _resolve_plan_event_references(
                        plan,
                        context.event_id_by_ref,
                        context.series_event_id_by_ref,
                        tuple(
                            str(candidate.get("event_id"))
                            for candidate in context.application_state.get(
                                "candidate_events", []
                            )
                            if isinstance(candidate, Mapping)
                            and candidate.get("recurring") is True
                            and candidate.get("event_id")
                        ),
                    )
                except _UnknownEventReference:
                    LOGGER.warning(
                        "AI planner returned an unknown event reference "
                        "for update %s; calendar unchanged",
                        update_id,
                    )
                    job["final_html"] = format_error_card(
                        "ИИ-планировщик выбрал событие вне доступного контекста. "
                        "Уточните его название или время новым сообщением.",
                        transcript=transcript,
                        elapsed_seconds=self._elapsed(job),
                        model_name=_job_planner_model(job),
                    )
                    job["final_reply_markup"] = None
                    job["status"] = "final_ready"
                except GeminiError as exc:
                    planning_elapsed = time.monotonic() - planning_started
                    LOGGER.warning(
                        "AI planner failed for update %s; error_type=%s "
                        "error=%s elapsed=%.3fs; calendar unchanged",
                        update_id,
                        type(exc).__name__,
                        str(exc),
                        planning_elapsed,
                    )
                    job["final_html"] = format_error_card(
                        _planner_failure_copy(exc),
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
                    html_text=(
                        _job_progress_card(job, "fast_read")
                        if job.get("fast_read") is True
                        else _job_progress_card(
                            job,
                            "calendar_lookup",
                        )
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
                        source_update_id=update_id,
                    )
                    durable_result = _calendar_query_payload(query_result)
                    if plan.get("action") == "lookup":
                        durable_result["events"] = (
                            await _hydrate_lookup_series_candidates(
                                pipeline,
                                account=account,
                                candidates=durable_result["events"],
                                limit=_LOOKUP_DISPLAY_LIMIT,
                            )
                        )
                except CalendarOperationError as exc:
                    LOGGER.warning(
                        "Calendar lookup failed for update %s; calendar unchanged",
                        update_id,
                    )
                    job["final_html"] = format_error_card(
                        str(exc),
                        transcript=str(job["transcript"]),
                        elapsed_seconds=self._elapsed(job),
                        model_name=_job_planner_model(job),
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
                transcript=_job_memory_text(job),
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
                model_name=_job_planner_model(job),
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
            # owner or invoking the planner for a second pass. Only rows that can be
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
                series_context_by_event_id = {
                    str(event.get("event_id")): deepcopy(event["series_context"])
                    for event in events
                    if isinstance(event, Mapping)
                    and event.get("event_id")
                    and isinstance(event.get("series_context"), Mapping)
                }
                observed = pipeline.observe_lookup_events(account, events)
                trusted_candidates = [deepcopy(dict(event)) for event in observed]
                for candidate in trusted_candidates:
                    series_context = series_context_by_event_id.get(
                        str(candidate.get("event_id") or "")
                    )
                    if series_context is not None:
                        candidate["series_context"] = deepcopy(series_context)
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
                html_text=_job_progress_card(job, "gemini_match"),
            )
            self.state.save_job(update_id, job)
            sent_time = datetime.fromtimestamp(
                int(job["sent_at"]), tz=timezone.utc
            ).astimezone(ZoneInfo(self.config.calendar_timezone))
            context = pipeline.context(
                account=account, chat_id=chat_id, now=sent_time
            )
            application_state = deepcopy(context.application_state)
            (
                compact_candidates,
                lookup_event_id_by_ref,
                lookup_series_event_id_by_ref,
            ) = (
                _compact_lookup_candidates(
                    visible_candidates,
                    timezone_name=self.config.calendar_timezone,
                )
            )
            application_state.update(
                {
                    "candidate_events": compact_candidates,
                    "allowed_event_ids": list(lookup_event_id_by_ref),
                    "lookup_permitted": False,
                    "lookup_request": deepcopy(lookup),
                    "lookup_result": {
                        "total_count": total_count,
                        "may_be_incomplete": incomplete,
                    },
                }
            )
            # Preserve native thought signatures only while continuing this
            # same command from discovery to candidate selection.  Previous
            # user commands are represented by compact application memory.
            native_history: list[dict[str, Any]] = []
            first_input, first_steps = _interaction_chain((initial_plan,))
            if first_input is not None:
                native_history.append(first_input)
            native_history.extend(first_steps)
            planning_started = time.monotonic()
            try:
                with planner_diagnostic_context(
                    f"telegram-update-{update_id}:matching"
                ):
                    resolved_plan = await self.gemini.plan_calendar_actions(
                        transcript,
                        reference_time=sent_time,
                        account=account,
                        application_state=application_state,
                        recent_conversation=context.recent_conversation,
                        history_steps=native_history,
                        input_kind=input_kind,
                        image_observations=_job_image_observations(job),
                    )
                selected_model = _planner_model_name(resolved_plan)
                if selected_model is not None:
                    job["planner_model"] = selected_model
                resolved_plan = _resolve_plan_event_references(
                    resolved_plan,
                    lookup_event_id_by_ref,
                    lookup_series_event_id_by_ref,
                    tuple(
                        str(candidate["event_id"])
                        for candidate in compact_candidates
                        if candidate.get("recurring") is True
                    ),
                )
            except _UnknownEventReference:
                LOGGER.warning(
                    "AI planner candidate matching returned an unknown event "
                    "reference for update %s; calendar unchanged",
                    update_id,
                )
                job["final_html"] = format_error_card(
                    "ИИ-планировщик выбрал событие вне показанного списка. Уточните "
                    "его название или время новым сообщением.",
                    transcript=transcript,
                    elapsed_seconds=self._elapsed(job),
                    model_name=_job_planner_model(job),
                )
                job["final_reply_markup"] = None
                job["status"] = "final_ready"
            except GeminiError as exc:
                planning_elapsed = time.monotonic() - planning_started
                LOGGER.warning(
                    "AI planner candidate matching failed for update %s; "
                    "error_type=%s error=%s elapsed=%.3fs; calendar unchanged",
                    update_id,
                    type(exc).__name__,
                    str(exc),
                    planning_elapsed,
                )
                job["final_html"] = format_error_card(
                    _planner_failure_copy(exc, matching=True),
                    transcript=transcript,
                    elapsed_seconds=self._elapsed(job),
                )
                job["final_reply_markup"] = None
                job["status"] = "final_ready"
            else:
                if resolved_plan.get("action") in {"read", "lookup"}:
                    selected_model = _planner_model_name(resolved_plan)
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
                        PLANNER_MODEL_FIELD: selected_model,
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
                    html_text=_job_progress_card(
                        job,
                        "calendar",
                        action=progress_action,  # type: ignore[arg-type]
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
                _compact, visible_refs, series_refs = _compact_lookup_candidates(
                    candidates,
                    timezone_name=self.config.calendar_timezone,
                )
                trusted_event_ids = list(
                    dict.fromkeys((*visible_refs.values(), *series_refs.values()))
                )
            try:
                execution = await pipeline.apply_plan(
                    source_update_id=update_id,
                    account=account,
                    owner_user_id=chat_id,
                    chat_id=chat_id,
                    transcript=_job_memory_text(job),
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
                        model_name=_job_planner_model(job),
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
                        model_name=_job_planner_model(job),
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
                        model_name=_job_planner_model(job),
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
                            model_name=_job_planner_model(job),
                        )
                        if candidates
                        else format_clarify_card(
                            question,
                            transcript=transcript,
                            elapsed_seconds=self._elapsed(job),
                            model_name=_job_planner_model(job),
                        )
                    )
                    job["final_reply_markup"] = None
                else:
                    job["final_html"] = format_ignore_card(
                        transcript=transcript,
                        elapsed_seconds=self._elapsed(job),
                        model_name=_job_planner_model(job),
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
                # Persist the Telegram result before invoking the planner. A restart
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
                    with planner_diagnostic_context(
                        f"telegram-update-{update_id}:extract"
                    ):
                        intent = await self.gemini.extract_event(
                            transcript,
                            reference_time=sent_time,
                            account=account,
                        )
                except GeminiError as exc:
                    LOGGER.warning(
                        "AI planner extraction failed for update %s; "
                        "error_type=%s error=%s; transcript fallback used",
                        update_id,
                        type(exc).__name__,
                        str(exc),
                    )
                    job["reply"] = (
                        f"Расшифровка:\n{transcript}\n\n"
                        "⚠️ ИИ-планировщик сейчас не смог разобрать событие."
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
                    "⚠️ ИИ-планировщик сейчас недоступен."
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
    planner_stages: list[GeminiProviderStage] = []
    codex_runner_token: str | None = None
    try:
        codex_runner_token = read_secret(
            environment=config.codex_runner_token_environment,
        )
        codex_provider = CodexCliRunnerApi(
            base_url=config.codex_runner_url,
            bearer_token=codex_runner_token,
            model=config.codex_model,
            reasoning_effort=config.codex_reasoning_effort,
            timeout_seconds=config.codex_timeout_seconds,
            timezone=config.calendar_timezone,
        )
    except (CodexCliError, RuntimeError) as exc:
        LOGGER.warning(
            "Codex CLI provider configuration unavailable; "
            "error_type=%s error=%s; skipping Codex CLI",
            type(exc).__name__,
            str(exc),
        )
    else:
        planner_stages.append(
            GeminiProviderStage(
                "Codex Luna",
                codex_provider,
                config.codex_timeout_seconds,
            )
        )
    finally:
        codex_runner_token = None
    gigachat_credentials: str | None = None
    try:
        gigachat_credentials = read_secret(
            environment=config.gigachat_credentials_environment,
            account=config.gigachat_keychain_account,
            service=config.gigachat_keychain_service,
        )
    except RuntimeError:
        LOGGER.warning("GigaChat credentials unavailable; skipping GigaChat")
    else:
        try:
            gigachat_provider = GigaChatApi(
                gigachat_credentials,
                ca_bundle_path=config.gigachat_ca_bundle_file,
                timeout_seconds=config.gigachat_timeout_seconds,
                timezone=config.calendar_timezone,
                scope=config.gigachat_scope,
                model=config.gigachat_model,
                base_url=config.gigachat_base_url,
                auth_url=config.gigachat_auth_url,
                max_retries=1,
            )
        except GigaChatApiError as exc:
            LOGGER.warning(
                "GigaChat provider configuration unavailable; "
                "error_type=%s error=%s; skipping GigaChat",
                type(exc).__name__,
                str(exc),
            )
        else:
            planner_stages.append(
                GeminiProviderStage(
                    "GigaChat 2 Max",
                    gigachat_provider,
                    config.gigachat_timeout_seconds,
                )
            )
        finally:
            gigachat_credentials = None
    openrouter_api_key: str | None = None
    try:
        openrouter_api_key = read_secret(
            environment=config.openrouter_api_key_environment,
            account=config.openrouter_keychain_account,
            service=config.openrouter_keychain_service,
        )
    except RuntimeError:
        LOGGER.warning("OpenRouter API key unavailable; skipping free models")
    else:
        planner_stages.extend(
            (
                GeminiProviderStage(
                    "Nemotron 3 Super",
                    OpenRouterApi(
                        openrouter_api_key,
                        model=config.openrouter_model,
                        timeout_seconds=config.openrouter_timeout_seconds,
                        timezone=config.calendar_timezone,
                        reasoning_effort=config.openrouter_reasoning_effort,
                        max_tokens=config.openrouter_max_tokens,
                        max_retries=0,
                    ),
                    config.openrouter_timeout_seconds,
                ),
                GeminiProviderStage(
                    "GLM 5.2 Free",
                    OpenRouterApi(
                        openrouter_api_key,
                        model=config.openrouter_fallback_model,
                        timeout_seconds=config.openrouter_fallback_timeout_seconds,
                        timezone=config.calendar_timezone,
                        reasoning_effort=(
                            config.openrouter_fallback_reasoning_effort
                        ),
                        max_tokens=config.openrouter_max_tokens,
                        max_retries=0,
                    ),
                    config.openrouter_fallback_timeout_seconds,
                ),
            )
        )

    gemini_api_key: str | None = None
    try:
        gemini_api_key = read_secret(
            environment=config.gemini_api_key_environment,
            account=config.gemini_keychain_account,
            service=config.gemini_keychain_service,
        )
    except RuntimeError:
        if config.bot_update_mode == "webhook":
            if planner_stages:
                cleanup_chain = GeminiProviderChain(
                    planner_stages,
                    timeout_seconds=config.calendar_planner_timeout_seconds,
                )
                try:
                    await cleanup_chain.aclose()
                except GeminiError:
                    LOGGER.warning(
                        "Planner provider cleanup failed during startup abort"
                    )
            raise RuntimeError(
                "Gemini API key is required in webhook mode"
            ) from None
        LOGGER.warning(
            "Gemini API key unavailable; local Antigravity CLI is the last fallback"
        )
        terminal_provider: GeminiProvider = GeminiCli(
            config.gemini_cli_path,
            model=config.gemini_cli_model,
            timeout_seconds=config.gemini_timeout_seconds,
            timezone=config.calendar_timezone,
        )
        terminal_name = "Gemini CLI"
    else:
        terminal_provider = GeminiApi(
            gemini_api_key,
            model=config.gemini_model,
            timeout_seconds=config.gemini_timeout_seconds,
            timezone=config.calendar_timezone,
            max_retries=1,
        )
        terminal_name = "Gemini 3.7 Flash"

    planner_stages.append(
        GeminiProviderStage(
            terminal_name,
            terminal_provider,
            config.gemini_timeout_seconds,
        )
    )
    planner_stages.sort(
        key=lambda stage: _PLANNER_STAGE_PRIORITIES[stage.name]
    )
    gemini = GeminiProviderChain(
        planner_stages,
        timeout_seconds=config.calendar_planner_timeout_seconds,
    )
    try:
        vision = build_vision_pipeline(
            config,
            openrouter_api_key=openrouter_api_key,
            gemini_api_key=gemini_api_key,
        )
    except Exception:
        await gemini.aclose()
        raise
    # Provider instances retain the credentials they need. Drop the temporary
    # builder references before entering the long-running service.
    openrouter_api_key = None
    gemini_api_key = None

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
                    service.vision = vision
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
        try:
            await vision.aclose()
        finally:
            await gemini.aclose()


def main() -> None:
    asyncio.run(async_main())


if __name__ == "__main__":
    main()

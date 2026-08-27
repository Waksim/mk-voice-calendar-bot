"""Safe, bounded Telegram HTML cards for calendar operations.

The model and Telegram transcript are untrusted text.  Every dynamic value is
escaped here; callers supply data, never markup.  Cards are deliberately kept
to one Telegram message so an HTML tag or entity is never split across chunks.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from html import escape, unescape
import math
import re
from typing import Any, Literal
from urllib.parse import urlsplit
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from .text import utf16_units


CalendarAction = Literal["create", "read", "update", "delete"]
MutationAction = Literal["create", "update", "delete"]
UndoAction = Literal["create", "update", "delete", "mixed"]
InputKind = Literal["voice", "text"]
ProgressPhase = Literal[
    "matching",
    "transcribing",
    "gemini",
    "calendar_lookup",
    "gemini_match",
    "calendar",
]

TELEGRAM_TEXT_LIMIT = 4096
MAX_EVENTS_PER_CARD = 5

_CONTROL_CHARACTERS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_HTML_TAGS = re.compile(
    r'</?(?:b|i|s|a|code|blockquote)(?: expandable| href="[^"]*")?>'
)
_UNDO_CALLBACK = re.compile(r"cal:undo:([A-Za-z0-9_-]{16,32})\Z")
_OPERATION_ID = re.compile(r"[A-Za-z0-9_-]{16,32}\Z")

_MONTHS = (
    "января",
    "февраля",
    "марта",
    "апреля",
    "мая",
    "июня",
    "июля",
    "августа",
    "сентября",
    "октября",
    "ноября",
    "декабря",
)
_WEEKDAYS = (
    "понедельник",
    "вторник",
    "среда",
    "четверг",
    "пятница",
    "суббота",
    "воскресенье",
)

_ACTION_LABELS: dict[str, str] = {
    "create": "добавить событие",
    "read": "показать события",
    "update": "изменить событие",
    "delete": "удалить событие",
}
_OPERATION_HEADERS: dict[str, str] = {
    "create": "✅ <b>Добавлено в календарь</b>",
    "update": "✏️ <b>Событие обновлено</b>",
    "delete": "🗑️ <b>Событие удалено</b>",
}
_UNDO_BUTTON_LABELS: dict[str, str] = {
    "create": "↩️ Отменить добавление",
    "update": "↩️ Отменить изменение",
    "delete": "↩️ Восстановить событие",
    "mixed": "↩️ Отменить операцию",
}
_CHANGE_LABELS: dict[str, str] = {
    "title": "Название",
    "start_at": "Начало",
    "end_at": "Окончание",
    "all_day": "Весь день",
    "location": "Место",
    "description": "Описание",
    "recurrence_rrule": "Повтор",
}


@dataclass(frozen=True)
class FieldChange:
    """One user-facing before/after field in an update result."""

    field: str
    before: str | None
    after: str | None


def _clean_text(value: object, *, multiline: bool = False) -> str:
    text = str(value).encode("utf-8", "replace").decode("utf-8")
    text = _CONTROL_CHARACTERS.sub("", text.replace("\r\n", "\n").replace("\r", "\n"))
    if multiline:
        lines = [line.rstrip() for line in text.split("\n")]
        text = "\n".join(lines).strip()
        return re.sub(r"\n{3,}", "\n\n", text)
    return " ".join(text.split())


def _truncate_utf16(value: str, limit: int, *, suffix: str = "…") -> str:
    if limit <= 0:
        return ""
    if utf16_units(value) <= limit:
        return value
    suffix_units = utf16_units(suffix)
    if suffix_units >= limit:
        return suffix if suffix_units == limit else ""
    target = limit - suffix_units
    result: list[str] = []
    units = 0
    for character in value:
        character_units = 2 if ord(character) > 0xFFFF else 1
        if units + character_units > target:
            break
        result.append(character)
        units += character_units
    return "".join(result).rstrip() + suffix


def _dynamic(
    value: object,
    *,
    limit: int,
    multiline: bool = False,
) -> str:
    cleaned = _clean_text(value, multiline=multiline)
    return escape(_truncate_utf16(cleaned, limit), quote=True)


def _visible_text(html_text: str) -> str:
    # Dynamic text is escaped before this point, so only our literal template
    # tags are removed.  Unescape afterwards so "&lt;" counts as one visible
    # character for Telegram's post-entity limit.
    return unescape(_HTML_TAGS.sub("", html_text))


def _bounded(html_text: str) -> str:
    if not html_text:
        raise ValueError("Telegram card must not be empty")
    if utf16_units(_visible_text(html_text)) > TELEGRAM_TEXT_LIMIT:
        raise ValueError("Telegram card exceeds the 4096-character limit")
    return html_text


def format_duration(elapsed_seconds: float) -> str:
    """Format monotonic elapsed time using compact Russian units."""
    if (
        isinstance(elapsed_seconds, bool)
        or not isinstance(elapsed_seconds, (int, float))
        or not math.isfinite(float(elapsed_seconds))
        or elapsed_seconds < 0
    ):
        raise ValueError("elapsed_seconds must be a finite non-negative number")
    seconds = float(elapsed_seconds)
    if seconds < 60:
        return f"{seconds:.1f}".replace(".", ",") + " с"
    rounded = int(round(seconds))
    hours, remainder = divmod(rounded, 3600)
    minutes, whole_seconds = divmod(remainder, 60)
    if hours:
        return f"{hours} ч {minutes} мин {whole_seconds} с"
    return f"{minutes} мин {whole_seconds} с"


def safe_google_calendar_link(value: object) -> str | None:
    """Return only an HTTPS link hosted by Google, suitable for an HTML href."""
    if not isinstance(value, str) or not value or len(value) > 2048:
        return None
    if any(ord(character) <= 0x20 for character in value):
        return None
    try:
        parsed = urlsplit(value)
        hostname = parsed.hostname
        port = parsed.port
    except ValueError:
        return None
    if (
        parsed.scheme != "https"
        or hostname is None
        or parsed.username is not None
        or parsed.password is not None
        or port not in (None, 443)
        or not (hostname == "google.com" or hostname.endswith(".google.com"))
    ):
        return None
    return value


def format_progress_card(
    phase: ProgressPhase,
    *,
    action: CalendarAction | None = None,
    input_kind: InputKind = "voice",
) -> str:
    """Render the current processing phase as a small editable status card."""
    if action is not None and action not in _ACTION_LABELS:
        raise ValueError("invalid calendar action")
    if input_kind not in {"voice", "text"}:
        raise ValueError("invalid input kind")
    processing_header = (
        "🎙️ <b>Обрабатываю голосовое</b>"
        if input_kind == "voice"
        else "💬 <b>Обрабатываю текстовую команду</b>"
    )
    command_received = (
        "✅ Расшифровка Telegram получена"
        if input_kind == "voice"
        else "✅ Текстовая команда получена"
    )
    cards = {
        "matching": (
            "🎙️ <b>Обрабатываю голосовое</b>\n\n"
            "⏳ Ищу сообщение в Telegram…\n"
            "▫️ Расшифровка\n"
            "▫️ ИИ-планировщик\n"
            "▫️ Google Calendar"
        ),
        "transcribing": (
            "🎙️ <b>Обрабатываю голосовое</b>\n\n"
            "✅ Голосовое найдено\n"
            "⏳ Получаю расшифровку от Telegram…\n"
            "▫️ ИИ-планировщик\n"
            "▫️ Google Calendar"
        ),
        "gemini": (
            f"{processing_header}\n\n"
            f"{command_received}\n"
            "⏳ ИИ-планировщик разбирает команду и контекст…\n"
            "▫️ Google Calendar"
        ),
    }
    if phase == "calendar":
        operation = _ACTION_LABELS.get(action or "", "обработать команду")
        verb = {
            "create": "Добавляю событие в Google Calendar…",
            "read": "Читаю события из Google Calendar…",
            "update": "Обновляю событие в Google Calendar…",
            "delete": "Удаляю событие из Google Calendar…",
        }.get(action, "Синхронизирую данные с Google Calendar…")
        return _bounded(
            f"{processing_header}\n\n"
            f"{command_received}\n"
            f"✅ ИИ-планировщик: {operation}\n"
            f"⏳ {verb}"
        )
    if phase == "calendar_lookup":
        return _bounded(
            f"{processing_header}\n\n"
            f"{command_received}\n"
            "✅ ИИ-планировщик определил период поиска\n"
            "⏳ Ищу события в Google Calendar…"
        )
    if phase == "gemini_match":
        return _bounded(
            f"{processing_header}\n\n"
            f"{command_received}\n"
            "✅ Подходящие события найдены\n"
            "⏳ ИИ-планировщик выбирает точную запись…"
        )
    try:
        return _bounded(cards[phase])
    except KeyError:
        raise ValueError("invalid progress phase") from None


def _parse_datetime(value: object) -> datetime | None:
    if not isinstance(value, str):
        return None
    normalized = value[:-1] + "+00:00" if value.endswith("Z") else value
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError:
        return None
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        return None
    return parsed


def _format_date(value: date) -> str:
    return f"{_WEEKDAYS[value.weekday()]}, {value.day} {_MONTHS[value.month - 1]} {value.year}"


def _when_lines(event: Mapping[str, Any]) -> list[str]:
    start_raw = event.get("start_at")
    end_raw = event.get("end_at")
    if event.get("all_day") is True:
        try:
            start = date.fromisoformat(str(start_raw))
            end_exclusive = date.fromisoformat(str(end_raw))
        except ValueError:
            start = end_exclusive = None
        if start is not None and end_exclusive is not None and end_exclusive > start:
            inclusive_end = end_exclusive - timedelta(days=1)
            if inclusive_end == start:
                return [f"📅 {_format_date(start)} · весь день"]
            return [
                f"📅 {_format_date(start)} — {_format_date(inclusive_end)} · весь день"
            ]
    else:
        start = _parse_datetime(start_raw)
        end = _parse_datetime(end_raw)
        if start is not None and end is not None and end > start:
            timezone_name = event.get("timezone")
            if isinstance(timezone_name, str) and timezone_name:
                try:
                    display_zone = ZoneInfo(timezone_name)
                except (ValueError, ZoneInfoNotFoundError):
                    # Provider snapshots can omit or expose an unknown zone.
                    # Their RFC3339 offsets are still safe to display as-is.
                    pass
                else:
                    start = start.astimezone(display_zone)
                    end = end.astimezone(display_zone)
            if start.date() == end.date():
                date_line = f"📅 {_format_date(start.date())}"
            else:
                date_line = (
                    f"📅 {_format_date(start.date())} — {_format_date(end.date())}"
                )
            return [date_line, f"🕒 {start:%H:%M}–{end:%H:%M}"]

    start_text = _dynamic(start_raw or "не указано", limit=64)
    end_text = _dynamic(end_raw or "не указано", limit=64)
    return [f"📅 {start_text} — {end_text}"]


def _event_link(
    event: Mapping[str, Any],
    html_links: Sequence[str | None] | None,
    index: int,
) -> str | None:
    candidate: object = event.get("html_link")
    if html_links is not None and index < len(html_links):
        candidate = html_links[index]
    return safe_google_calendar_link(candidate)


def _event_block(
    event: Mapping[str, Any],
    *,
    index: int,
    multiple: bool,
    html_link: str | None,
) -> str:
    title_limit = 80 if multiple else 120
    optional_limit = 80 if multiple else 120
    description_limit = 100 if multiple else 160
    title = _dynamic(
        event.get("title") or event.get("summary") or "Без названия",
        limit=title_limit,
    )
    prefix = f"{index + 1}. " if multiple else ""
    lines = [f"{prefix}🗓 <b>{title}</b>", *_when_lines(event)]
    location = _clean_text(event.get("location") or "")
    if location:
        lines.append(f"📍 {_dynamic(location, limit=optional_limit)}")
    recurrence = _clean_text(event.get("recurrence_rrule") or "")
    if recurrence:
        lines.append(f"🔁 {_dynamic(recurrence, limit=optional_limit)}")
    description = _clean_text(event.get("description") or "")
    if description:
        lines.append(f"📝 {_dynamic(description, limit=description_limit)}")
    if html_link is not None:
        lines.append(
            f'<a href="{escape(html_link, quote=True)}">Открыть в Google Calendar</a>'
        )
    return "\n".join(lines)


def _coerce_change(value: FieldChange | Mapping[str, Any]) -> FieldChange:
    if isinstance(value, FieldChange):
        return value
    if not isinstance(value, Mapping):
        raise ValueError("calendar changes must be FieldChange objects or mappings")
    field = value.get("field")
    if not isinstance(field, str) or not field:
        raise ValueError("calendar change field is required")
    before = value.get("before")
    after = value.get("after")
    return FieldChange(
        field=field,
        before=None if before is None else str(before),
        after=None if after is None else str(after),
    )


def _changes_block(
    changes: Sequence[FieldChange | Mapping[str, Any]],
) -> str | None:
    rendered: list[str] = []
    for raw_change in changes[:5]:
        change = _coerce_change(raw_change)
        if change.before == change.after:
            continue
        label = _CHANGE_LABELS.get(change.field, _clean_text(change.field))
        before = change.before if change.before not in (None, "") else "не указано"
        after = change.after if change.after not in (None, "") else "не указано"
        rendered.append(
            f"{_dynamic(label, limit=40)}: "
            f"<s>{_dynamic(before, limit=60)}</s> → "
            f"<b>{_dynamic(after, limit=60)}</b>"
        )
    if not rendered:
        return None
    return "<b>Что изменилось</b>\n" + "\n".join(rendered)


def _transcript_block(transcript: object, *, limit: int) -> str | None:
    cleaned = _clean_text(transcript, multiline=True)
    if not cleaned:
        return None
    return (
        "<blockquote expandable><b>💬 Команда</b>\n"
        f"{_dynamic(cleaned, limit=limit, multiline=True)}</blockquote>"
    )


def _model_block(model_name: object | None) -> str | None:
    cleaned = _clean_text(model_name) if model_name is not None else ""
    if not cleaned:
        return None
    return f"🤖 <code>{_dynamic(cleaned, limit=100)}</code>"


def format_operation_card(
    action: CalendarAction,
    events: Sequence[Mapping[str, Any]],
    *,
    transcript: str,
    elapsed_seconds: float,
    changes: Sequence[FieldChange | Mapping[str, Any]] = (),
    html_links: Sequence[str | None] | None = None,
    best_effort_undo: bool = False,
    model_name: str | None = None,
) -> str:
    """Render a successful create, update, or delete operation."""
    if action not in _OPERATION_HEADERS:
        raise ValueError("invalid calendar action")
    if isinstance(events, (str, bytes)) or not 1 <= len(events) <= MAX_EVENTS_PER_CARD:
        raise ValueError("events must contain between one and five items")
    if any(not isinstance(event, Mapping) for event in events):
        raise ValueError("events must contain mappings")
    if isinstance(changes, (str, bytes)):
        raise ValueError("changes must be a sequence")
    if html_links is not None and isinstance(html_links, (str, bytes)):
        raise ValueError("html_links must be a sequence")

    parts = [_OPERATION_HEADERS[action]]
    if action == "update":
        changes_block = _changes_block(changes)
        if changes_block:
            parts.append(changes_block)
            parts.append("<b>Теперь</b>")

    multiple = len(events) > 1
    event_blocks: list[str] = []
    for index, event in enumerate(events):
        link = None if action == "delete" else _event_link(event, html_links, index)
        event_blocks.append(
            _event_block(
                event,
                index=index,
                multiple=multiple,
                html_link=link,
            )
        )
    parts.append("\n\n".join(event_blocks))
    if best_effort_undo:
        parts.append(
            "⚠️ <i>Кнопка отмены восстановит основные поля, но расширенные "
            "данные удалённого события могут не вернуться.</i>"
        )

    transcript_block = _transcript_block(
        transcript,
        limit=650 if len(events) == 1 else 300,
    )
    if transcript_block:
        parts.append(transcript_block)
    model_block = _model_block(model_name)
    if model_block:
        parts.append(model_block)
    parts.append(f"⏱ Готово за <b>{format_duration(elapsed_seconds)}</b>")
    return _bounded("\n\n".join(parts))


def format_create_card(
    events: Sequence[Mapping[str, Any]],
    *,
    transcript: str,
    elapsed_seconds: float,
    html_links: Sequence[str | None] | None = None,
    model_name: str | None = None,
) -> str:
    return format_operation_card(
        "create",
        events,
        transcript=transcript,
        elapsed_seconds=elapsed_seconds,
        html_links=html_links,
        model_name=model_name,
    )


def format_update_card(
    events: Sequence[Mapping[str, Any]],
    *,
    transcript: str,
    elapsed_seconds: float,
    changes: Sequence[FieldChange | Mapping[str, Any]] = (),
    html_links: Sequence[str | None] | None = None,
    model_name: str | None = None,
) -> str:
    return format_operation_card(
        "update",
        events,
        transcript=transcript,
        elapsed_seconds=elapsed_seconds,
        changes=changes,
        html_links=html_links,
        model_name=model_name,
    )


def format_delete_card(
    events: Sequence[Mapping[str, Any]],
    *,
    transcript: str,
    elapsed_seconds: float,
    best_effort_undo: bool = False,
    model_name: str | None = None,
) -> str:
    return format_operation_card(
        "delete",
        events,
        transcript=transcript,
        elapsed_seconds=elapsed_seconds,
        best_effort_undo=best_effort_undo,
        model_name=model_name,
    )


def _read_event_block(event: Mapping[str, Any], index: int) -> str:
    title = _dynamic(
        event.get("title") or event.get("summary") or "Без названия",
        limit=90,
    )
    lines = [f"{index + 1}. <b>{title}</b>", *_when_lines(event)]
    location = _clean_text(event.get("location") or "")
    if location:
        lines.append(f"📍 {_dynamic(location, limit=80)}")
    link = safe_google_calendar_link(event.get("html_link"))
    if link is not None:
        lines.append(
            f'<a href="{escape(link, quote=True)}">Открыть событие</a>'
        )
    return "\n".join(lines)


def format_read_card(
    events: Sequence[Mapping[str, Any]],
    *,
    transcript: str,
    elapsed_seconds: float,
    total_count: int | None = None,
    may_be_incomplete: bool = False,
    model_name: str | None = None,
) -> str:
    """Render a bounded read-only Calendar result without an undo control."""

    if isinstance(events, (str, bytes)) or any(
        not isinstance(event, Mapping) for event in events
    ):
        raise ValueError("events must be a sequence of mappings")
    if total_count is not None and (
        isinstance(total_count, bool) or not isinstance(total_count, int) or total_count < 0
    ):
        raise ValueError("total_count must be a non-negative integer")

    shown = list(events[:8])
    if shown:
        parts = [
            "📆 <b>События в календаре</b>",
            "\n\n".join(
                _read_event_block(event, index) for index, event in enumerate(shown)
            ),
        ]
        known_total = total_count if total_count is not None else len(events)
        omitted = max(0, known_total - len(shown))
        if omitted:
            parts.append(f"Ещё событий в этом периоде: <b>{omitted}</b>.")
    else:
        parts = [
            "📭 <b>Событий не найдено</b>",
            "В указанном периоде подходящих записей нет.",
        ]
    if may_be_incomplete:
        parts.append(
            "⚠️ <i>Результат мог быть сокращён. Уточните период или название события.</i>"
        )
    transcript_block = _transcript_block(transcript, limit=450)
    if transcript_block:
        parts.append(transcript_block)
    model_block = _model_block(model_name)
    if model_block:
        parts.append(model_block)
    parts.append(f"⏱ Готово за <b>{format_duration(elapsed_seconds)}</b>")
    return _bounded("\n\n".join(parts))


def format_lookup_clarify_card(
    question: str,
    candidates: Sequence[Mapping[str, Any]],
    *,
    transcript: str,
    elapsed_seconds: float,
    model_name: str | None = None,
) -> str:
    """Show safe candidate summaries when a mutation target is ambiguous."""

    if isinstance(candidates, (str, bytes)) or any(
        not isinstance(event, Mapping) for event in candidates
    ):
        raise ValueError("candidates must be a sequence of mappings")
    cleaned_question = _clean_text(question, multiline=True)
    if not cleaned_question:
        raise ValueError("clarification question is required")
    parts = [
        "🤔 <b>Нужно выбрать событие</b>",
        _dynamic(cleaned_question, limit=600, multiline=True),
    ]
    if candidates:
        parts.append(
            "\n\n".join(
                _read_event_block(event, index)
                for index, event in enumerate(candidates[:5])
            )
        )
    parts.append(
        "<i>Уточните название или время следующим сообщением.</i>"
    )
    transcript_block = _transcript_block(transcript, limit=400)
    if transcript_block:
        parts.append(transcript_block)
    model_block = _model_block(model_name)
    if model_block:
        parts.append(model_block)
    parts.append(f"⏱ Ответ за <b>{format_duration(elapsed_seconds)}</b>")
    return _bounded("\n\n".join(parts))


def format_mixed_operation_card(
    items: Sequence[Mapping[str, Any]],
    *,
    transcript: str,
    elapsed_seconds: float,
    best_effort_undo: bool = False,
    model_name: str | None = None,
) -> str:
    """Render a bounded result for a batch containing different action types."""
    if isinstance(items, (str, bytes)) or not 1 <= len(items) <= MAX_EVENTS_PER_CARD:
        raise ValueError("items must contain between one and five operations")
    blocks: list[str] = []
    labels = {
        "create": "➕ <b>Добавлено</b>",
        "update": "✏️ <b>Изменено</b>",
        "delete": "🗑 <b>Удалено</b>",
    }
    for index, item in enumerate(items):
        if not isinstance(item, Mapping) or item.get("type") not in labels:
            raise ValueError("mixed operation item is invalid")
        event = item.get("event")
        if not isinstance(event, Mapping):
            raise ValueError("mixed operation event is invalid")
        action = str(item["type"])
        link = None if action == "delete" else safe_google_calendar_link(
            event.get("html_link")
        )
        blocks.append(
            labels[action]
            + "\n"
            + _event_block(
                event,
                index=index,
                multiple=True,
                html_link=link,
            )
        )
    parts = ["✅ <b>Календарь обновлён</b>", "\n\n".join(blocks)]
    if best_effort_undo:
        parts.append(
            "⚠️ <i>Кнопка отмены восстановит основные поля удалённых событий, "
            "но их расширенные данные могут не вернуться.</i>"
        )
    transcript_block = _transcript_block(transcript, limit=300)
    if transcript_block:
        parts.append(transcript_block)
    model_block = _model_block(model_name)
    if model_block:
        parts.append(model_block)
    parts.append(f"⏱ Готово за <b>{format_duration(elapsed_seconds)}</b>")
    return _bounded("\n\n".join(parts))


def format_clarify_card(
    question: str,
    *,
    transcript: str,
    elapsed_seconds: float,
    model_name: str | None = None,
) -> str:
    cleaned_question = _clean_text(question, multiline=True)
    if not cleaned_question:
        raise ValueError("clarification question is required")
    parts = [
        "🤔 <b>Нужно уточнение</b>",
        _dynamic(cleaned_question, limit=900, multiline=True),
        "<i>Ответьте следующим сообщением — я продолжу эту операцию.</i>",
    ]
    transcript_block = _transcript_block(transcript, limit=650)
    if transcript_block:
        parts.append(transcript_block)
    model_block = _model_block(model_name)
    if model_block:
        parts.append(model_block)
    parts.append(f"⏱ Ответ за <b>{format_duration(elapsed_seconds)}</b>")
    return _bounded("\n\n".join(parts))


def format_ignore_card(
    *,
    transcript: str,
    elapsed_seconds: float,
    model_name: str | None = None,
) -> str:
    """Render a non-error response when no calendar command was present."""
    parts = [
        "ℹ️ <b>Календарь не изменён</b>",
        "ИИ-планировщик не нашёл в сообщении команды для календаря.",
    ]
    transcript_block = _transcript_block(transcript, limit=650)
    if transcript_block:
        parts.append(transcript_block)
    model_block = _model_block(model_name)
    if model_block:
        parts.append(model_block)
    parts.append(f"⏱ Готово за <b>{format_duration(elapsed_seconds)}</b>")
    return _bounded("\n\n".join(parts))


def format_error_card(
    message: str,
    *,
    transcript: str | None = None,
    elapsed_seconds: float | None = None,
    calendar_unchanged: bool = True,
    model_name: str | None = None,
) -> str:
    cleaned_message = _clean_text(message, multiline=True)
    if not cleaned_message:
        raise ValueError("error message is required")
    parts = [
        "❌ <b>Не удалось выполнить команду</b>",
        _dynamic(cleaned_message, limit=900, multiline=True),
    ]
    if calendar_unchanged:
        parts.append("<i>Google Calendar не изменён.</i>")
    else:
        parts.append("<i>Проверьте итог в Google Calendar.</i>")
    if transcript:
        transcript_block = _transcript_block(transcript, limit=650)
        if transcript_block:
            parts.append(transcript_block)
    model_block = _model_block(model_name)
    if model_block:
        parts.append(model_block)
    if elapsed_seconds is not None:
        parts.append(f"⏱ Остановлено через <b>{format_duration(elapsed_seconds)}</b>")
    return _bounded("\n\n".join(parts))


def format_undo_card(
    original_action: UndoAction,
    event_titles: Sequence[str],
    *,
    elapsed_seconds: float,
    best_effort: bool = False,
) -> str:
    if original_action not in _UNDO_BUTTON_LABELS:
        raise ValueError("invalid original calendar action")
    if isinstance(event_titles, (str, bytes)) or not 1 <= len(event_titles) <= 5:
        raise ValueError("event_titles must contain between one and five titles")
    titles = [f"• <b>{_dynamic(title or 'Без названия', limit=120)}</b>" for title in event_titles]
    plural = len(titles) > 1
    effect = {
        "create": "События удалены из календаря." if plural else "Событие удалено из календаря.",
        "update": "Для событий возвращены прежние данные." if plural else "Для события возвращены прежние данные.",
        "delete": "События восстановлены." if plural else "Событие восстановлено.",
        "mixed": "Изменения в календаре отменены.",
    }[original_action]
    header = "↩️ <b>Действие отменено</b>"
    if best_effort:
        header = "↩️ <b>Основные изменения отменены</b>"
        effect = (
            "Основные поля удалённых событий восстановлены, но их расширенные "
            "данные могли быть потеряны."
        )
    return _bounded(
        header
        + "\n\n"
        + "\n".join(titles)
        + f"\n\n{effect}\n\n"
        + f"⏱ Отменено за <b>{format_duration(elapsed_seconds)}</b>"
    )


def undo_reply_markup(
    operation_id: str,
    action: UndoAction,
    *,
    best_effort: bool = False,
) -> dict[str, Any]:
    """Build one owner-bound undo button; ownership is checked by the service."""
    if not isinstance(operation_id, str) or not _OPERATION_ID.fullmatch(operation_id):
        raise ValueError("invalid operation ID")
    try:
        label = _UNDO_BUTTON_LABELS[action]
    except KeyError:
        raise ValueError("invalid calendar action") from None
    if best_effort:
        label = (
            "↩️ Восстановить основные поля"
            if action == "delete"
            else "↩️ Отменить основные изменения"
        )
    return {
        "inline_keyboard": [
            [
                {
                    "text": label,
                    "callback_data": f"cal:undo:{operation_id}",
                }
            ]
        ]
    }


def parse_undo_callback(data: Any) -> str | None:
    """Return the opaque operation ID for an exact, size-bounded undo callback."""
    if not isinstance(data, str):
        return None
    try:
        if not 1 <= len(data.encode("utf-8")) <= 64:
            return None
    except UnicodeEncodeError:
        return None
    match = _UNDO_CALLBACK.fullmatch(data)
    return match.group(1) if match is not None else None

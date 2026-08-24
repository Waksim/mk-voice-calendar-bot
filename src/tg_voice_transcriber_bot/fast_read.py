"""Deterministic planner for a small set of safe calendar read requests.

The fast path is deliberately narrower than the LLM planner.  It recognizes
only unambiguous requests to list calendar events in a bounded time window and
never emits a Calendar mutation or a title query.
"""

from __future__ import annotations

from datetime import datetime, timedelta
import re
import unicodedata
from typing import Any
from zoneinfo import ZoneInfo

from .intent import validate_calendar_operation_plan


_MAX_INPUT_LENGTH = 1_000
_MAX_LOOKAHEAD_HOURS = 31 * 24
_MAX_LOOKAHEAD_DAYS = 31

# A mutation word makes the whole message ineligible even if it also contains
# a phrase that otherwise resembles a read request.  False negatives are much
# safer here than routing a mutation through a read-only shortcut.
_MUTATION_WORD = re.compile(
    r"\b(?:"
    r"добав\w*|созда\w*|постав\w*|запиш\w*|занес\w*|внес\w*|"
    r"удал\w*|убер\w*|отмен\w*|измен\w*|исправ\w*|редакт\w*|"
    r"перенес\w*|перенос\w*|сдвин\w*|помен\w*|замен\w*|"
    r"обнов\w*|допол\w*|назнач\w*|напомн\w*"
    r")\b"
)

_EVENTS = r"(?:события|встречи|мероприятия)"
_READ_PREFIX = re.compile(
    rf"(?:"
    rf"какие(?:\s+у\s+меня)?\s+{_EVENTS}|"
    rf"какие\s+{_EVENTS}\s+у\s+меня|"
    rf"покажи(?:\s+мне)?(?:\s+мои)?\s+{_EVENTS}|"
    rf"перечисли(?:\s+мне)?(?:\s+мои)?\s+{_EVENTS}|"
    rf"есть\s+ли(?:\s+у\s+меня)?\s+{_EVENTS}|"
    rf"что(?:\s+у\s+меня)?\s+(?:запланировано|в\s+календаре)|"
    rf"что\s+запланировано\s+у\s+меня|"
    rf"покажи(?:\s+мне)?(?:\s+мой)?\s+(?:календарь|расписание)|"
    rf"какое\s+у\s+меня\s+расписание|"
    rf"мое\s+расписание|"
    rf"(?:мои\s+)?{_EVENTS}"
    rf")(?:\s+в\s+(?:моем\s+)?календаре)?"
)

_HOUR_COUNT_SUFFIX = re.compile(
    r"(?:в|на)\s+(?:ближайшие|следующие)\s+"
    r"(?P<count>[1-9]\d{0,2})\s+(?:час|часа|часов)$"
)
_HOUR_DURATION_SUFFIX = re.compile(
    r"в\s+течение\s+(?P<count>[1-9]\d{0,2})\s+"
    r"(?:часа|часов|час)$"
)
_DAY_COUNT_SUFFIX = re.compile(
    r"(?:в|на)\s+(?:ближайшие|следующие)\s+"
    r"(?P<count>[1-9]\d?)\s+(?:день|дня|дней)$"
)
_DAY_DURATION_SUFFIX = re.compile(
    r"в\s+течение\s+(?P<count>[1-9]\d?)\s+"
    r"(?:дня|дней|день)$"
)


def _normalize_text(text: str) -> str:
    """Remove Telegram isolates and punctuation without interpreting markup."""

    normalized = unicodedata.normalize("NFKC", text).casefold().replace("ё", "е")
    characters: list[str] = []
    for character in normalized:
        category = unicodedata.category(character)
        if category == "Cf":
            # Includes LRM/RLM and the LRI/RLI/FSI/PDI isolate characters that
            # Telegram may wrap around quoted text.
            continue
        if category.startswith("P") or category.startswith("Z") or character.isspace():
            characters.append(" ")
        else:
            characters.append(character)
    return " ".join("".join(characters).split())


def _midnight(value: datetime) -> datetime:
    return value.replace(hour=0, minute=0, second=0, microsecond=0)


def _read_prefix_before(text: str, match: re.Match[str]) -> bool:
    return _READ_PREFIX.fullmatch(text[: match.start()].strip()) is not None


def _calendar_window(
    text: str, reference_time: datetime
) -> tuple[datetime, datetime] | None:
    """Return a bounded local time window when the complete phrase is safe."""

    fixed_windows: tuple[tuple[re.Pattern[str], str], ...] = (
        (re.compile(r"(?:на\s+)?послезавтра$"), "day_after_tomorrow"),
        (re.compile(r"(?:на\s+)?завтра$"), "tomorrow"),
        (re.compile(r"(?:на\s+)?сегодня$"), "today"),
        (
            re.compile(r"(?:на\s+сегодня\s+)?до\s+конца\s+(?:сегодняшнего\s+)?дня$"),
            "end_of_day",
        ),
        (
            re.compile(r"(?:в|на)\s+(?:ближайший|следующий)\s+час$"),
            "one_hour",
        ),
    )
    for suffix, kind in fixed_windows:
        match = suffix.search(text)
        if match is None or not _read_prefix_before(text, match):
            continue
        today = _midnight(reference_time)
        if kind == "today":
            return today, today + timedelta(days=1)
        if kind == "tomorrow":
            return today + timedelta(days=1), today + timedelta(days=2)
        if kind == "day_after_tomorrow":
            return today + timedelta(days=2), today + timedelta(days=3)
        if kind == "end_of_day":
            return reference_time, today + timedelta(days=1)
        return reference_time, reference_time + timedelta(hours=1)

    counted_windows: tuple[
        tuple[re.Pattern[str], timedelta, int], ...
    ] = (
        (_HOUR_COUNT_SUFFIX, timedelta(hours=1), _MAX_LOOKAHEAD_HOURS),
        (_HOUR_DURATION_SUFFIX, timedelta(hours=1), _MAX_LOOKAHEAD_HOURS),
        (_DAY_COUNT_SUFFIX, timedelta(days=1), _MAX_LOOKAHEAD_DAYS),
        (_DAY_DURATION_SUFFIX, timedelta(days=1), _MAX_LOOKAHEAD_DAYS),
    )
    for suffix, unit, maximum in counted_windows:
        match = suffix.search(text)
        if match is None or not _read_prefix_before(text, match):
            continue
        count = int(match.group("count"))
        if count > maximum:
            return None
        return reference_time, reference_time + unit * count
    return None


def plan_fast_calendar_read(
    text: str,
    *,
    reference_time: datetime,
    timezone: str = "Europe/Moscow",
) -> dict[str, Any] | None:
    """Build a validated read plan, or ``None`` for anything not clearly safe.

    The returned lookup always has ``query=None``.  Requests mentioning a
    particular title/person therefore fall back to the regular planner rather
    than becoming an accidental broad Calendar read.
    """

    if not isinstance(text, str) or not text.strip() or len(text) > _MAX_INPUT_LENGTH:
        return None
    if reference_time.tzinfo is None or reference_time.utcoffset() is None:
        raise ValueError("reference_time must be timezone-aware")

    normalized = _normalize_text(text)
    if not normalized or _MUTATION_WORD.search(normalized):
        return None

    zone = ZoneInfo(timezone)
    local_reference = reference_time.astimezone(zone)
    window = _calendar_window(normalized, local_reference)
    if window is None:
        return None
    time_min, time_max = window

    return validate_calendar_operation_plan(
        {
            "action": "read",
            "operations": [],
            "lookup": {
                "query": None,
                "time_min": time_min.isoformat(),
                "time_max": time_max.isoformat(),
            },
            "clarification_question": None,
            "confidence": 1.0,
        },
        set(),
        expected_timezone=timezone,
    )


__all__ = ["plan_fast_calendar_read"]

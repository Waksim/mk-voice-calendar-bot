"""Strict calendar intent schema and user-facing previews."""

from __future__ import annotations

from collections.abc import Collection
from datetime import date, datetime, timedelta
import re
from typing import Any
from zoneinfo import ZoneInfo


CALENDAR_INTENT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "action": {"type": "string", "enum": ["create", "clarify", "ignore"]},
        "events": {
            "type": "array",
            "maxItems": 5,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "title": {"type": "string", "minLength": 1, "maxLength": 300},
                    "start_at": {
                        "anyOf": [{"type": "string"}, {"type": "null"}]
                    },
                    "end_at": {
                        "anyOf": [{"type": "string"}, {"type": "null"}]
                    },
                    "all_day": {"type": "boolean"},
                    "timezone": {"type": "string"},
                    "location": {
                        "anyOf": [{"type": "string"}, {"type": "null"}]
                    },
                    "description": {
                        "anyOf": [{"type": "string"}, {"type": "null"}]
                    },
                    "recurrence_rrule": {
                        "anyOf": [{"type": "string"}, {"type": "null"}]
                    },
                },
                "required": [
                    "title",
                    "start_at",
                    "end_at",
                    "all_day",
                    "timezone",
                    "location",
                    "description",
                    "recurrence_rrule",
                ],
            },
        },
        "clarification_question": {
            "anyOf": [{"type": "string"}, {"type": "null"}]
        },
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
    },
    "required": ["action", "events", "clarification_question", "confidence"],
}


# The operation schema intentionally keeps creation data and update data in
# separate properties.  In particular, a missing patch field means "preserve
# the current value"; clearing one of the three nullable Calendar fields is an
# explicit operation expressed through ``clear_fields``.  This prevents a model
# that eagerly emits nulls from erasing existing event data by accident.
_CALENDAR_EVENT_PROPERTIES: dict[str, Any] = {
    "title": {"type": "string", "minLength": 1, "maxLength": 300},
    "start_at": {"type": "string"},
    "end_at": {"type": "string"},
    "all_day": {"type": "boolean"},
    "timezone": {"type": "string"},
    "location": {"anyOf": [{"type": "string"}, {"type": "null"}]},
    "description": {"anyOf": [{"type": "string"}, {"type": "null"}]},
    "recurrence_rrule": {
        "anyOf": [{"type": "string"}, {"type": "null"}]
    },
}

_CALENDAR_PATCH_PROPERTIES: dict[str, Any] = {
    "title": {"type": "string", "minLength": 1, "maxLength": 300},
    "start_at": {"type": "string"},
    "end_at": {"type": "string"},
    "all_day": {"type": "boolean"},
    "timezone": {"type": "string"},
    "location": {"type": "string", "minLength": 1, "maxLength": 500},
    "description": {"type": "string", "minLength": 1, "maxLength": 5000},
    "recurrence_rrule": {"type": "string", "minLength": 1, "maxLength": 500},
}

_CALENDAR_LOOKUP_PROPERTIES: dict[str, Any] = {
    "query": {"anyOf": [{"type": "string", "maxLength": 300}, {"type": "null"}]},
    "time_min": {"type": "string"},
    "time_max": {"type": "string"},
}

CALENDAR_OPERATION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "action": {
            "type": "string",
            "enum": ["execute", "read", "lookup", "clarify", "ignore"],
        },
        "operations": {
            "type": "array",
            "maxItems": 5,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "type": {
                        "type": "string",
                        "enum": ["create", "update", "delete"],
                    },
                    "target_event_id": {
                        "anyOf": [{"type": "string"}, {"type": "null"}]
                    },
                    "recurrence_scope": {
                        "anyOf": [
                            {
                                "type": "string",
                                "enum": ["series", "occurrence"],
                            },
                            {"type": "null"},
                        ]
                    },
                    "event": {
                        "anyOf": [
                            {
                                "type": "object",
                                "additionalProperties": False,
                                "properties": _CALENDAR_EVENT_PROPERTIES,
                                "required": list(_CALENDAR_EVENT_PROPERTIES),
                            },
                            {"type": "null"},
                        ]
                    },
                    "patch": {
                        "anyOf": [
                            {
                                "type": "object",
                                "additionalProperties": False,
                                "properties": _CALENDAR_PATCH_PROPERTIES,
                            },
                            {"type": "null"},
                        ]
                    },
                    "clear_fields": {
                        "type": "array",
                        "maxItems": 3,
                        "items": {
                            "type": "string",
                            "enum": [
                                "location",
                                "description",
                                "recurrence_rrule",
                            ],
                        },
                    },
                },
                "required": [
                    "type",
                    "target_event_id",
                    "recurrence_scope",
                    "event",
                    "patch",
                    "clear_fields",
                ],
            },
        },
        "clarification_question": {
            "anyOf": [{"type": "string"}, {"type": "null"}]
        },
        "lookup": {
            "anyOf": [
                {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": _CALENDAR_LOOKUP_PROPERTIES,
                    "required": list(_CALENDAR_LOOKUP_PROPERTIES),
                },
                {"type": "null"},
            ]
        },
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
    },
    "required": [
        "action",
        "operations",
        "lookup",
        "clarification_question",
        "confidence",
    ],
}


_ROOT_KEYS = frozenset(CALENDAR_INTENT_SCHEMA["properties"])
_EVENT_KEYS = frozenset(
    CALENDAR_INTENT_SCHEMA["properties"]["events"]["items"]["properties"]
)


def _optional_text(value: Any, field: str, *, max_length: int) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"{field} must be a string or null")
    stripped = value.strip()
    if len(stripped) > max_length:
        raise ValueError(f"{field} is too long")
    return stripped or None


def _parse_timed(value: str, field: str) -> datetime:
    normalized = value[:-1] + "+00:00" if value.endswith("Z") else value
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError as exc:
        raise ValueError(f"{field} must be RFC3339") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{field} must contain an explicit UTC offset")
    return parsed


def _parse_all_day(value: str, field: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"{field} must be YYYY-MM-DD") from exc


def _validate_rrule(value: str | None) -> str | None:
    if value is None:
        return None
    if "\r" in value or "\n" in value or len(value) > 500:
        raise ValueError("recurrence_rrule contains invalid characters")
    if not value.startswith("RRULE:"):
        raise ValueError("recurrence_rrule must begin with RRULE:")
    fields: dict[str, str] = {}
    for component in value[6:].split(";"):
        if component.count("=") != 1:
            raise ValueError("recurrence_rrule has invalid syntax")
        key, field_value = component.split("=", maxsplit=1)
        if not key or not field_value or key in fields:
            raise ValueError("recurrence_rrule has invalid syntax")
        fields[key] = field_value
    allowed = {
        "FREQ",
        "INTERVAL",
        "COUNT",
        "UNTIL",
        "BYDAY",
        "BYMONTHDAY",
        "BYMONTH",
        "BYSETPOS",
        "WKST",
    }
    if set(fields) - allowed:
        raise ValueError("recurrence_rrule contains unsupported fields")
    if fields.get("FREQ") not in {"DAILY", "WEEKLY", "MONTHLY", "YEARLY"}:
        raise ValueError("recurrence_rrule has invalid FREQ")
    if "COUNT" in fields and "UNTIL" in fields:
        raise ValueError("recurrence_rrule cannot use COUNT and UNTIL together")
    for key, maximum in (("INTERVAL", 100), ("COUNT", 1000)):
        if key in fields and (
            not fields[key].isdigit() or not 1 <= int(fields[key]) <= maximum
        ):
            raise ValueError(f"recurrence_rrule has invalid {key}")
    if "UNTIL" in fields and not re.fullmatch(
        r"(?:\d{8}|\d{8}T\d{6}Z)", fields["UNTIL"]
    ):
        raise ValueError("recurrence_rrule has invalid UNTIL")
    if "WKST" in fields and fields["WKST"] not in {
        "MO",
        "TU",
        "WE",
        "TH",
        "FR",
        "SA",
        "SU",
    }:
        raise ValueError("recurrence_rrule has invalid WKST")
    if "BYDAY" in fields:
        day_pattern = re.compile(r"(?:[+-]?(?:[1-9]|[1-4]\d|5[0-3]))?(?:MO|TU|WE|TH|FR|SA|SU)")
        if not all(day_pattern.fullmatch(item) for item in fields["BYDAY"].split(",")):
            raise ValueError("recurrence_rrule has invalid BYDAY")
    for key, lower, upper in (
        ("BYMONTHDAY", -31, 31),
        ("BYMONTH", 1, 12),
        ("BYSETPOS", -366, 366),
    ):
        if key in fields:
            try:
                numbers = [int(item) for item in fields[key].split(",")]
            except ValueError:
                raise ValueError(f"recurrence_rrule has invalid {key}") from None
            if not numbers or any(number == 0 or not lower <= number <= upper for number in numbers):
                raise ValueError(f"recurrence_rrule has invalid {key}")
    return value


def validate_calendar_intent(
    payload: Any, *, expected_timezone: str = "Europe/Moscow"
) -> dict[str, Any]:
    """Validate schema shape and the temporal semantics used by the backend."""
    if not isinstance(payload, dict) or set(payload) != _ROOT_KEYS:
        raise ValueError("calendar intent has unexpected or missing fields")

    action = payload.get("action")
    if action not in {"create", "clarify", "ignore"}:
        raise ValueError("invalid calendar action")
    confidence = payload.get("confidence")
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
        raise ValueError("confidence must be a number")
    if not 0 <= float(confidence) <= 1:
        raise ValueError("confidence must be between 0 and 1")

    events = payload.get("events")
    if not isinstance(events, list) or len(events) > 5:
        raise ValueError("events must be an array of at most five items")
    question = _optional_text(
        payload.get("clarification_question"),
        "clarification_question",
        max_length=500,
    )
    if action == "create" and not events:
        raise ValueError("create action must contain an event")
    if action == "create" and question is not None:
        raise ValueError("create action cannot contain a clarification question")
    if action == "clarify" and (not question or events):
        raise ValueError("clarify action must contain only a question")
    if action == "ignore" and (events or question is not None):
        raise ValueError("ignore action cannot contain events or a question")

    normalized_events: list[dict[str, Any]] = []
    for raw_event in events:
        if not isinstance(raw_event, dict) or set(raw_event) != _EVENT_KEYS:
            raise ValueError("calendar event has unexpected or missing fields")
        title = _optional_text(raw_event.get("title"), "title", max_length=300)
        if not title or len(title) > 300:
            raise ValueError("event title is required")
        if not isinstance(raw_event.get("all_day"), bool):
            raise ValueError("all_day must be boolean")
        all_day = raw_event["all_day"]
        timezone = raw_event.get("timezone")
        if timezone != expected_timezone:
            raise ValueError("event timezone is not the configured timezone")
        start_at = _optional_text(raw_event.get("start_at"), "start_at", max_length=64)
        end_at = _optional_text(raw_event.get("end_at"), "end_at", max_length=64)
        if not start_at or not end_at:
            raise ValueError("complete events require both start_at and end_at")
        if all_day:
            start_value = _parse_all_day(start_at, "start_at")
            end_value = _parse_all_day(end_at, "end_at")
        else:
            start_value = _parse_timed(start_at, "start_at")
            end_value = _parse_timed(end_at, "end_at")
            expected_zone = ZoneInfo(expected_timezone)
            if (
                start_value.utcoffset()
                != start_value.astimezone(expected_zone).utcoffset()
                or end_value.utcoffset()
                != end_value.astimezone(expected_zone).utcoffset()
            ):
                raise ValueError("event UTC offset does not match its timezone")
        if end_value <= start_value:
            raise ValueError("event end must be after its start")
        if end_value - start_value > timedelta(days=366):
            raise ValueError("event duration is unreasonably long")

        recurrence = _validate_rrule(
            _optional_text(
                raw_event.get("recurrence_rrule"),
                "recurrence_rrule",
                max_length=500,
            )
        )
        normalized_events.append(
            {
                "title": title,
                "start_at": start_at,
                "end_at": end_at,
                "all_day": all_day,
                "timezone": timezone,
                "location": _optional_text(
                    raw_event.get("location"), "location", max_length=500
                ),
                "description": _optional_text(
                    raw_event.get("description"), "description", max_length=5000
                ),
                "recurrence_rrule": recurrence,
            }
        )

    return {
        "action": action,
        "events": normalized_events,
        "clarification_question": question,
        "confidence": float(confidence),
    }


_OPERATION_ROOT_KEYS = frozenset(CALENDAR_OPERATION_SCHEMA["properties"])
_OPERATION_KEYS = frozenset(
    CALENDAR_OPERATION_SCHEMA["properties"]["operations"]["items"]["properties"]
)
_PATCH_KEYS = frozenset(_CALENDAR_PATCH_PROPERTIES)
_CLEARABLE_EVENT_FIELDS = frozenset(
    {"location", "description", "recurrence_rrule"}
)
_LOOKUP_KEYS = frozenset(_CALENDAR_LOOKUP_PROPERTIES)
_OPERATION_REQUIRED_KEYS = frozenset(
    {"action", "operations", "lookup", "clarification_question", "confidence"}
)


def _normalize_complete_event(
    raw_event: Any, *, expected_timezone: str
) -> dict[str, Any]:
    """Reuse the established full-event validation without changing v1."""
    wrapped = {
        "action": "create",
        "events": [raw_event],
        "clarification_question": None,
        "confidence": 1.0,
    }
    return validate_calendar_intent(
        wrapped, expected_timezone=expected_timezone
    )["events"][0]


def _patch_text(value: Any, field: str, *, max_length: int) -> str:
    if not isinstance(value, str):
        raise ValueError(f"patch {field} must be a non-null string")
    stripped = value.strip()
    if not stripped:
        raise ValueError(f"patch {field} cannot be empty; use clear_fields")
    if len(stripped) > max_length:
        raise ValueError(f"patch {field} is too long")
    return stripped


def _timed_patch_value(
    value: str, field: str, *, expected_timezone: str
) -> datetime:
    parsed = _parse_timed(value, field)
    expected_zone = ZoneInfo(expected_timezone)
    if parsed.utcoffset() != parsed.astimezone(expected_zone).utcoffset():
        raise ValueError("event UTC offset does not match its timezone")
    return parsed


def _normalize_event_patch(
    raw_patch: Any, *, expected_timezone: str
) -> dict[str, Any]:
    if not isinstance(raw_patch, dict) or not set(raw_patch) <= _PATCH_KEYS:
        raise ValueError("calendar patch has unexpected fields or is not an object")

    normalized: dict[str, Any] = {}
    if "title" in raw_patch:
        normalized["title"] = _patch_text(
            raw_patch["title"], "title", max_length=300
        )
    if "location" in raw_patch:
        normalized["location"] = _patch_text(
            raw_patch["location"], "location", max_length=500
        )
    if "description" in raw_patch:
        normalized["description"] = _patch_text(
            raw_patch["description"], "description", max_length=5000
        )
    if "recurrence_rrule" in raw_patch:
        recurrence = _validate_rrule(raw_patch["recurrence_rrule"])
        if recurrence is None:  # pragma: no cover - guarded by the schema path
            raise ValueError("patch recurrence_rrule must be a non-null string")
        normalized["recurrence_rrule"] = recurrence
    if "all_day" in raw_patch:
        if not isinstance(raw_patch["all_day"], bool):
            raise ValueError("patch all_day must be boolean")
        if not {"start_at", "end_at"} <= set(raw_patch):
            raise ValueError(
                "patch all_day requires both start_at and end_at"
            )
        normalized["all_day"] = raw_patch["all_day"]
    if "timezone" in raw_patch:
        if raw_patch["timezone"] != expected_timezone:
            raise ValueError("event timezone is not the configured timezone")
        normalized["timezone"] = expected_timezone

    parsed_temporal: dict[str, tuple[date | datetime, bool]] = {}
    all_day_hint = raw_patch.get("all_day")
    for field in ("start_at", "end_at"):
        if field not in raw_patch:
            continue
        value = _patch_text(raw_patch[field], field, max_length=64)
        normalized[field] = value
        is_all_day = (
            all_day_hint
            if isinstance(all_day_hint, bool)
            else re.fullmatch(r"\d{4}-\d{2}-\d{2}", value) is not None
        )
        if is_all_day:
            parsed_temporal[field] = (_parse_all_day(value, field), True)
        else:
            parsed_temporal[field] = (
                _timed_patch_value(
                    value, field, expected_timezone=expected_timezone
                ),
                False,
            )

    if len(parsed_temporal) == 2:
        start_value, start_all_day = parsed_temporal["start_at"]
        end_value, end_all_day = parsed_temporal["end_at"]
        if start_all_day != end_all_day:
            raise ValueError("patch start_at and end_at use different date formats")
        if end_value <= start_value:
            raise ValueError("event end must be after its start")
        if end_value - start_value > timedelta(days=366):
            raise ValueError("event duration is unreasonably long")

    return normalized


def _normalize_calendar_lookup(
    raw_lookup: Any, *, expected_timezone: str
) -> dict[str, Any]:
    if not isinstance(raw_lookup, dict) or set(raw_lookup) != _LOOKUP_KEYS:
        raise ValueError("calendar lookup has unexpected or missing fields")

    query = _optional_text(raw_lookup.get("query"), "lookup query", max_length=300)
    time_min_text = _optional_text(
        raw_lookup.get("time_min"), "lookup time_min", max_length=64
    )
    time_max_text = _optional_text(
        raw_lookup.get("time_max"), "lookup time_max", max_length=64
    )
    if not time_min_text or not time_max_text:
        raise ValueError("calendar lookup requires a complete time range")

    time_min = _timed_patch_value(
        time_min_text, "lookup time_min", expected_timezone=expected_timezone
    )
    time_max = _timed_patch_value(
        time_max_text, "lookup time_max", expected_timezone=expected_timezone
    )
    if time_max <= time_min:
        raise ValueError("calendar lookup time_max must be after time_min")
    if time_max - time_min > timedelta(days=31):
        raise ValueError("calendar lookup range cannot exceed 31 days")
    return {
        "query": query,
        "time_min": time_min.isoformat(),
        "time_max": time_max.isoformat(),
    }


def _allowed_id_set(allowed_event_ids: Collection[str]) -> frozenset[str]:
    if isinstance(allowed_event_ids, (str, bytes)) or not isinstance(
        allowed_event_ids, Collection
    ):
        raise ValueError("allowed_event_ids must be a collection of event IDs")
    if any(
        not isinstance(event_id, str)
        or not event_id
        or event_id != event_id.strip()
        or len(event_id) > 1024
        for event_id in allowed_event_ids
    ):
        raise ValueError("allowed_event_ids contains an invalid event ID")
    return frozenset(allowed_event_ids)


def validate_calendar_operation_plan(
    payload: Any,
    allowed_event_ids: Collection[str],
    expected_timezone: str = "Europe/Moscow",
) -> dict[str, Any]:
    """Validate a create/update/delete plan before it can mutate Calendar.

    Update and delete targets must be exact IDs supplied by application state.
    Update patches preserve omitted fields and can clear nullable fields only by
    naming them in ``clear_fields``.
    """
    if (
        not isinstance(payload, dict)
        or not _OPERATION_REQUIRED_KEYS.issubset(payload)
        or set(payload) - _OPERATION_ROOT_KEYS
    ):
        raise ValueError("calendar operation plan has unexpected or missing fields")

    action = payload.get("action")
    if action not in {"execute", "read", "lookup", "clarify", "ignore"}:
        raise ValueError("invalid calendar operation action")
    confidence = payload.get("confidence")
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
        raise ValueError("confidence must be a number")
    if not 0 <= float(confidence) <= 1:
        raise ValueError("confidence must be between 0 and 1")

    operations = payload.get("operations")
    if not isinstance(operations, list) or len(operations) > 5:
        raise ValueError("operations must be an array of at most five items")
    question = _optional_text(
        payload.get("clarification_question"),
        "clarification_question",
        max_length=500,
    )
    raw_lookup = payload.get("lookup")
    lookup = (
        _normalize_calendar_lookup(
            raw_lookup, expected_timezone=expected_timezone
        )
        if raw_lookup is not None
        else None
    )
    if action == "execute" and not operations:
        raise ValueError("execute action must contain an operation")
    if action == "execute" and (question is not None or lookup is not None):
        raise ValueError("execute action cannot contain clarification or lookup data")
    if action in {"read", "lookup"} and (
        operations or question is not None or lookup is None
    ):
        raise ValueError(f"{action} action must contain only a calendar lookup")
    if action == "clarify" and (not question or operations or lookup is not None):
        raise ValueError("clarify action must contain only a question")
    if action == "ignore" and (operations or question is not None or lookup is not None):
        raise ValueError("ignore action cannot contain operations, lookup, or a question")

    allowed_ids = _allowed_id_set(allowed_event_ids)
    normalized_operations: list[dict[str, Any]] = []
    targeted_ids: set[str] = set()
    for raw_operation in operations:
        if (
            not isinstance(raw_operation, dict)
            or set(raw_operation) != _OPERATION_KEYS
        ):
            raise ValueError("calendar operation has unexpected or missing fields")
        operation_type = raw_operation.get("type")
        if operation_type not in {"create", "update", "delete"}:
            raise ValueError("invalid calendar operation type")

        target_event_id = raw_operation.get("target_event_id")
        recurrence_scope = raw_operation.get("recurrence_scope")
        if recurrence_scope is not None and (
            not isinstance(recurrence_scope, str)
            or recurrence_scope not in {"series", "occurrence"}
        ):
            raise ValueError(
                "recurrence_scope must be series, occurrence, or null"
            )
        event = raw_operation.get("event")
        patch = raw_operation.get("patch")
        clear_fields = raw_operation.get("clear_fields")
        if not isinstance(clear_fields, list) or len(clear_fields) > 3:
            raise ValueError("clear_fields must be an array of at most three items")
        if (
            any(
                not isinstance(field, str)
                or field not in _CLEARABLE_EVENT_FIELDS
                for field in clear_fields
            )
            or len(set(clear_fields)) != len(clear_fields)
        ):
            raise ValueError("clear_fields contains invalid or duplicate fields")

        if operation_type == "create":
            if (
                target_event_id is not None
                or recurrence_scope is not None
                or patch is not None
                or clear_fields
            ):
                raise ValueError("create operation must contain only a complete event")
            normalized_event = _normalize_complete_event(
                event, expected_timezone=expected_timezone
            )
            normalized_patch = None
        else:
            if (
                not isinstance(target_event_id, str)
                or not target_event_id
                or target_event_id != target_event_id.strip()
                or len(target_event_id) > 1024
            ):
                raise ValueError(f"{operation_type} requires an exact target_event_id")
            if target_event_id not in allowed_ids:
                raise ValueError("target_event_id is not a known calendar event")
            if target_event_id in targeted_ids:
                raise ValueError("an event may be targeted only once per plan")
            targeted_ids.add(target_event_id)
            if event is not None:
                raise ValueError(f"{operation_type} operation cannot contain an event")
            normalized_event = None

            if operation_type == "update":
                if patch is None:
                    normalized_patch = None
                else:
                    normalized_patch = _normalize_event_patch(
                        patch, expected_timezone=expected_timezone
                    )
                if not normalized_patch and not clear_fields:
                    raise ValueError("update operation must change or clear a field")
                if normalized_patch and set(normalized_patch) & set(clear_fields):
                    raise ValueError("a field cannot be patched and cleared together")
                recurrence_changed = (
                    normalized_patch is not None
                    and "recurrence_rrule" in normalized_patch
                ) or "recurrence_rrule" in clear_fields
                if recurrence_changed and recurrence_scope != "series":
                    raise ValueError(
                        "recurrence rule changes require recurrence_scope=series"
                    )
            else:
                if patch is not None or clear_fields:
                    raise ValueError("delete operation cannot contain a patch")
                normalized_patch = None

        normalized_operations.append(
            {
                "type": operation_type,
                "target_event_id": target_event_id,
                "recurrence_scope": recurrence_scope,
                "event": normalized_event,
                "patch": normalized_patch,
                "clear_fields": list(clear_fields),
            }
        )

    return {
        "action": action,
        "operations": normalized_operations,
        "lookup": lookup,
        "clarification_question": question,
        "confidence": float(confidence),
    }


def _format_when(event: dict[str, Any]) -> str:
    if event["all_day"]:
        return f'{event["start_at"]} (весь день; конец {event["end_at"]})'
    return f'{event["start_at"]} — {event["end_at"]}'


def format_calendar_preview(
    transcript: str,
    intent: dict[str, Any],
    *,
    create_footer: str | None = None,
) -> str:
    """Build a plain-text preview; this function never performs a calendar write."""
    parts = [f"Команда:\n{transcript}"]
    action = intent["action"]
    if action == "create":
        parts.append("ИИ-планировщик распознал событие:")
        lines: list[str] = []
        multiple = len(intent["events"]) > 1
        for index, event in enumerate(intent["events"], start=1):
            prefix = f"{index}. " if multiple else ""
            item = [f'{prefix}{event["title"]}', f'Когда: {_format_when(event)}']
            if event["location"]:
                item.append(f'Где: {event["location"]}')
            if event["recurrence_rrule"]:
                item.append(f'Повтор: {event["recurrence_rrule"]}')
            if event["description"]:
                item.append(f'Описание: {event["description"]}')
            lines.append("\n".join(item))
        parts.append("\n\n".join(lines))
        parts.append(
            create_footer
            or "Пока это предпросмотр: запись в календарь ещё не подключена."
        )
    elif action == "clarify":
        parts.append(f'Нужно уточнить: {intent["clarification_question"]}')
    else:
        parts.append(
            "ИИ-планировщик не нашёл в сообщении события для календаря."
        )
    return "\n\n".join(parts)

"""Google Calendar adapter backed by the pinned Calendar MCP subprocess.

The adapter deliberately uses one ``create-event`` call per event.  The bulk
tool does not accept caller supplied event IDs, while a deterministic Google
event ID is what makes a retry safe even after the original response was lost.
"""

from __future__ import annotations

import base64
from collections.abc import Mapping, Sequence
from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import fields
from datetime import date, datetime, timedelta, timezone
import hashlib
from http import HTTPStatus
import json
import logging
import math
import os
from pathlib import Path
import re
import time
from typing import Any, AsyncIterator
from urllib.parse import urlsplit
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import anyio
from mcp import ClientSession, McpError, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.types import CONNECTION_CLOSED

from .calendar import (
    CalendarConnectionError,
    CalendarEventQueryResult,
    CalendarEventSnapshot,
    CalendarStateConflictError,
    CalendarWriteRejectedError,
    CreatedCalendarEvent,
    DeletedCalendarEvent,
    UpdatedCalendarEvent,
)


LOGGER = logging.getLogger("tg_voice_transcriber_bot.calendar_mcp")
CALENDAR_MCP_TOOLS = (
    "create-event",
    "delete-event",
    "get-event",
    "list-calendars",
    "list-events",
    "search-events",
    "update-event",
)
_GOOGLE_EVENT_ID_RE = re.compile(r"^[a-v0-9]{5,1024}$")
_MCP_ACCOUNT_RE = re.compile(r"^[a-z0-9_-]{1,64}$")
_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_AWARE_DATETIME_RE = re.compile(
    r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}"
    r"(?:\.\d{1,6})?(?:Z|[+-]\d{2}:\d{2})$"
)
_UPDATE_PATCH_FIELDS = frozenset(
    {
        "title",
        "description",
        "location",
        "start_at",
        "end_at",
        "timezone",
        "recurrence_rrules",
    }
)
_MAX_TEXT_RESPONSE_CHARS = 2_000_000
_MAX_EVENT_ID_CHARS = 2048
_MAX_QUERY_TEXT_CHARS = 300
_MAX_QUERY_LIMIT = 250
# The pinned MCP does not expose Google's nextPageToken.  It also omits
# maxResults, for which Calendar API uses a 250-item default page.  Reaching
# that boundary is therefore the only safe signal that another page may exist.
_UPSTREAM_EVENT_PAGE_SIZE = 250
_SNAPSHOT_EVENT_FIELDS = (
    "description",
    "attendees",
    "recurrence",
    "reminders",
    "colorId",
    "transparency",
    "updated",
    "creator",
    "organizer",
    "recurringEventId",
    "originalStartTime",
    "visibility",
    "eventType",
    "conferenceData",
    "hangoutLink",
    "attachments",
    "extendedProperties",
    "source",
    "anyoneCanAddSelf",
    "guestsCanInviteOthers",
    "guestsCanModify",
    "guestsCanSeeOtherGuests",
    "privateCopy",
    "locked",
)
_SAFETY_METADATA_FIELDS = (
    "attendees",
    "recurrence",
    "reminders",
    "colorId",
    "transparency",
    "creator",
    "organizer",
    "recurringEventId",
    "originalStartTime",
    "visibility",
    "eventType",
    "conferenceData",
    "hangoutLink",
    "attachments",
    "extendedProperties",
    "source",
    "anyoneCanAddSelf",
    "guestsCanInviteOthers",
    "guestsCanModify",
    "guestsCanSeeOtherGuests",
    "privateCopy",
    "locked",
)
_MAX_PROVIDER_TEXT_CHARS = 100_000
_MAX_PROVIDER_COLLECTION_ITEMS = 1_000
_MAX_PROVIDER_METADATA_DEPTH = 8
_MAX_PROVIDER_METADATA_NODES = 10_000
_MAX_TOOL_ERROR_CLASSIFICATION_CHARS = 8_192


class CalendarMcpError(RuntimeError):
    """A sanitized Calendar MCP failure safe to surface in service logs."""


class CalendarMcpConnectionError(CalendarMcpError, CalendarConnectionError):
    """The Calendar MCP stdio session is no longer usable in this process."""


class CalendarMcpCollisionError(CalendarMcpError):
    """The deterministic ID exists but belongs to a different event."""


class CalendarMcpWriteRejectedError(CalendarMcpError, CalendarWriteRejectedError):
    """A write rejection followed by an authoritative missing-ID probe."""


class _ToolFailure(Exception):
    pass


class _ToolRejected(_ToolFailure):
    pass


class _ToolNotFound(_ToolFailure):
    pass


class _ResponseMismatch(Exception):
    pass


# Values used by the pinned Python MCP SDK for a closed session and a timed
# out request.  A timed-out stdio request is fatal here: the client cannot
# prove that the subprocess will ever consume another request, while retrying
# forever would pin the durable webhook queue behind the same broken session.
_MCP_FATAL_LABELS = frozenset({"CONNECTION_CLOSED", "REQUEST_TIMEOUT"})
_BROKEN_STDIO_ERRORS = (
    anyio.BrokenResourceError,
    anyio.ClosedResourceError,
    anyio.EndOfStream,
    EOFError,
    OSError,
)


def _is_connection_failure(exception: BaseException) -> bool:
    """Recognize dead MCP transports, including nested task-group failures."""

    pending: list[BaseException] = [exception]
    seen: set[int] = set()
    while pending:
        current = pending.pop()
        identity = id(current)
        if identity in seen:
            continue
        seen.add(identity)

        if isinstance(current, CalendarConnectionError):
            return True
        if isinstance(current, _BROKEN_STDIO_ERRORS):
            return True
        if isinstance(current, McpError):
            error = getattr(current, "error", None)
            code = getattr(error, "code", None)
            if code in {CONNECTION_CLOSED, HTTPStatus.REQUEST_TIMEOUT}:
                return True
            message = getattr(error, "message", None)
            if isinstance(message, str):
                normalized = re.sub(r"[^A-Z0-9]+", "_", message.upper()).strip("_")
                padded = f"_{normalized}_"
                if any(f"_{label}_" in padded for label in _MCP_FATAL_LABELS):
                    return True

        nested = getattr(current, "exceptions", ())
        if isinstance(nested, Sequence) and not isinstance(nested, (str, bytes)):
            pending.extend(item for item in nested if isinstance(item, BaseException))
        cause = current.__cause__
        if cause is not None:
            pending.append(cause)
        context = current.__context__
        if context is not None and not current.__suppress_context__:
            pending.append(context)
        pending.extend(
            item for item in current.args if isinstance(item, BaseException)
        )
    return False


def _connection_error() -> CalendarMcpConnectionError:
    return CalendarMcpConnectionError("Calendar MCP connection is unavailable")


def google_event_id(idempotency_key: str, index: int) -> str:
    """Return a stable Google-compliant base32hex ID for one batch item.

    A NUL separator and a fixed-width index avoid ambiguous concatenations.
    SHA-256 makes the resulting 52-character identifier both bounded and safe
    for arbitrary Unicode idempotency keys.
    """
    if not isinstance(idempotency_key, str) or not idempotency_key:
        raise ValueError("idempotency_key must be a non-empty string")
    if (
        isinstance(index, bool)
        or not isinstance(index, int)
        or not 0 <= index < 2**64
    ):
        raise ValueError("index must be a non-negative integer")

    try:
        encoded_key = idempotency_key.encode("utf-8")
    except UnicodeEncodeError:
        raise ValueError("idempotency_key must be valid Unicode") from None
    digest = hashlib.sha256(
        encoded_key + b"\x00" + index.to_bytes(8, "big")
    ).digest()
    event_id = base64.b32hexencode(digest).decode("ascii").rstrip("=").lower()
    # This is an invariant check, not validation of provider-controlled data.
    if not _GOOGLE_EVENT_ID_RE.fullmatch(event_id):  # pragma: no cover
        raise AssertionError("generated an invalid Google event ID")
    return event_id


def _result_attribute(result: Any, name: str, default: Any = None) -> Any:
    if isinstance(result, Mapping):
        return result.get(name, default)
    return getattr(result, name, default)


def _content_attribute(content: Any, name: str, default: Any = None) -> Any:
    if isinstance(content, Mapping):
        return content.get(name, default)
    return getattr(content, name, default)


def _parse_tool_payload(result: Any) -> dict[str, Any]:
    """Decode one object without ever echoing provider-controlled text."""
    structured = _result_attribute(result, "structuredContent")
    if structured is None:
        # Accommodate clients which expose aliases in snake_case.
        structured = _result_attribute(result, "structured_content")
    if structured is not None:
        if not isinstance(structured, dict):
            raise _ResponseMismatch
        try:
            serialized = json.dumps(
                structured,
                ensure_ascii=False,
                separators=(",", ":"),
            )
        except (TypeError, ValueError):
            raise _ResponseMismatch from None
        if len(serialized) > _MAX_TEXT_RESPONSE_CHARS:
            raise _ResponseMismatch
        return structured

    candidates: list[dict[str, Any]] = []
    content = _result_attribute(result, "content", [])
    if not isinstance(content, Sequence) or isinstance(content, (str, bytes)):
        raise _ResponseMismatch
    for block in content:
        if _content_attribute(block, "type") != "text":
            continue
        text = _content_attribute(block, "text")
        if not isinstance(text, str) or len(text) > _MAX_TEXT_RESPONSE_CHARS:
            continue
        try:
            decoded = json.loads(text)
        except (TypeError, ValueError):
            continue
        if isinstance(decoded, dict):
            candidates.append(decoded)
    if len(candidates) != 1:
        raise _ResponseMismatch
    return candidates[0]


def _bounded_tool_error_text(result: Any) -> str | None:
    """Return one bounded diagnostic only for internal error classification.

    The caller must never log or surface this provider-controlled value. The
    pinned MCP exposes HTTP status classes only as stable text prefixes, so we
    retain just enough information to distinguish a definite 400 rejection
    and an authoritative exact-ID 404 from an ambiguous provider failure.
    """

    content = _result_attribute(result, "content", [])
    if not isinstance(content, Sequence) or isinstance(content, (str, bytes)):
        return None
    texts = [
        _content_attribute(block, "text")
        for block in content
        if _content_attribute(block, "type") == "text"
        and isinstance(_content_attribute(block, "text"), str)
        and len(_content_attribute(block, "text"))
        <= _MAX_TOOL_ERROR_CLASSIFICATION_CHARS
    ]
    if len(texts) != 1:
        return None
    return str(texts[0]).strip()


def _classify_tool_error(
    name: str,
    arguments: Mapping[str, Any],
    result: Any,
) -> type[_ToolFailure]:
    text = _bounded_tool_error_text(result)
    if text is not None:
        # The Node MCP SDK prefixes the package's stable status text with its
        # JSON-RPC error code. A create 400 is a semantic rejection, while an
        # exact get-event miss is emitted by the pinned handler as an Internal
        # error containing the caller-supplied ID. Require that exact ID so no
        # unrelated provider diagnostic can be mistaken for reconciliation.
        if name == "create-event" and re.match(
            r"^(?:MCP error -?\d+: )?Bad Request:", text
        ):
            return _ToolRejected
        event_id = arguments.get("eventId")
        if name == "get-event" and isinstance(event_id, str):
            missing_pattern = re.compile(
                r"^(?:MCP error -?\d+: )?Internal error: Event with ID '"
                + re.escape(event_id)
                + r"' not found in calendar '[^'\r\n]{1,1024}'\.$"
            )
            if missing_pattern.fullmatch(text):
                return _ToolNotFound
    return _ToolFailure


def _parse_datetime(value: Any) -> datetime | None:
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


def _times_match(actual: Any, expected: str, *, all_day: bool) -> bool:
    if not isinstance(actual, dict):
        return False
    if all_day:
        return actual.get("date") == expected and not actual.get("dateTime")
    actual_datetime = _parse_datetime(actual.get("dateTime"))
    expected_datetime = _parse_datetime(expected)
    return (
        actual_datetime is not None
        and expected_datetime is not None
        and actual_datetime == expected_datetime
        and not actual.get("date")
    )


def _mcp_recurring_wall_time(value: str, timezone_name: str) -> str:
    """Format one recurring instant for the pinned MCP's time parser.

    ``@cocal/google-calendar-mcp`` 2.6.2 discards its ``timeZone`` argument
    when a datetime already contains an RFC3339 offset. Google, however,
    requires an IANA time zone on timed recurring events so it can expand the
    series across offset changes. A local wall-clock value makes the MCP emit
    both ``dateTime`` and ``timeZone`` while the original aware value remains
    available separately for response verification.
    """

    parsed = _parse_datetime(value)
    if parsed is None:
        raise CalendarMcpError("Calendar event payload is invalid")
    try:
        zone = ZoneInfo(timezone_name)
    except (ValueError, ZoneInfoNotFoundError):
        raise CalendarMcpError("Calendar event payload is invalid") from None
    return parsed.astimezone(zone).replace(tzinfo=None).isoformat()


def _optional_text_matches(actual: Any, expected: str | None) -> bool:
    if expected is None:
        return actual in (None, "")
    return actual == expected


def _safe_html_link(value: Any) -> str | None:
    if not isinstance(value, str) or len(value) > 4096:
        return None
    try:
        parsed = urlsplit(value)
        hostname = parsed.hostname
    except ValueError:
        return None
    if (
        parsed.scheme != "https"
        or hostname is None
        or not (hostname == "google.com" or hostname.endswith(".google.com"))
    ):
        return None
    return value


def _validate_target_event_id(event_id: Any) -> str:
    if (
        not isinstance(event_id, str)
        or not event_id
        or event_id != event_id.strip()
        or len(event_id) > _MAX_EVENT_ID_CHARS
        or any(ord(character) < 32 for character in event_id)
    ):
        raise CalendarMcpError("Calendar event ID is invalid")
    return event_id


def _validate_idempotency_key(idempotency_key: Any) -> str:
    if (
        not isinstance(idempotency_key, str)
        or not idempotency_key
        or len(idempotency_key) > 4096
    ):
        raise CalendarMcpError("Calendar idempotency key is invalid")
    return idempotency_key


def _optional_provider_text(event: Mapping[str, Any], field: str) -> str | None:
    value = event.get(field)
    if value is None:
        return None
    if (
        not isinstance(value, str)
        or len(value) > _MAX_PROVIDER_TEXT_CHARS
        or any(ord(character) < 9 for character in value)
    ):
        raise _ResponseMismatch
    return value


def _optional_provider_bool(
    event: Mapping[str, Any], field: str
) -> bool | None:
    value = event.get(field)
    if value is None:
        return None
    if not isinstance(value, bool):
        raise _ResponseMismatch
    return value


def _validate_metadata_json(
    value: Any,
    *,
    depth: int = 0,
    node_count: list[int] | None = None,
) -> None:
    """Bound provider metadata before hashing it into a non-secret marker."""

    if node_count is None:
        node_count = [0]
    node_count[0] += 1
    if (
        node_count[0] > _MAX_PROVIDER_METADATA_NODES
        or depth > _MAX_PROVIDER_METADATA_DEPTH
    ):
        raise _ResponseMismatch
    if value is None or isinstance(value, (bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise _ResponseMismatch
        return
    if isinstance(value, str):
        if len(value) > _MAX_PROVIDER_TEXT_CHARS:
            raise _ResponseMismatch
        return
    if isinstance(value, Mapping):
        if len(value) > _MAX_PROVIDER_COLLECTION_ITEMS:
            raise _ResponseMismatch
        for key, item in value.items():
            if not isinstance(key, str) or len(key) > 256:
                raise _ResponseMismatch
            _validate_metadata_json(
                item,
                depth=depth + 1,
                node_count=node_count,
            )
        return
    if isinstance(value, list):
        if len(value) > _MAX_PROVIDER_COLLECTION_ITEMS:
            raise _ResponseMismatch
        for item in value:
            _validate_metadata_json(
                item,
                depth=depth + 1,
                node_count=node_count,
            )
        return
    raise _ResponseMismatch


def _metadata_fingerprint(event: Mapping[str, Any]) -> str:
    metadata = {
        field: event[field]
        for field in _SAFETY_METADATA_FIELDS
        if field in event
    }
    _validate_metadata_json(metadata)
    try:
        encoded = json.dumps(
            metadata,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError, UnicodeEncodeError):
        raise _ResponseMismatch from None
    if len(encoded) > _MAX_TEXT_RESPONSE_CHARS:
        raise _ResponseMismatch
    return hashlib.sha256(encoded).hexdigest()


def _provider_reminders(
    event: Mapping[str, Any],
) -> tuple[bool, bool | None, tuple[tuple[str, int], ...]]:
    value = event.get("reminders")
    if value is None:
        return False, None, ()
    if not isinstance(value, Mapping) or set(value) - {"useDefault", "overrides"}:
        raise _ResponseMismatch
    use_default = value.get("useDefault")
    if use_default is not None and not isinstance(use_default, bool):
        raise _ResponseMismatch
    raw_overrides = value.get("overrides")
    if raw_overrides is None:
        return True, use_default, ()
    if (
        not isinstance(raw_overrides, list)
        or len(raw_overrides) > _MAX_PROVIDER_COLLECTION_ITEMS
    ):
        raise _ResponseMismatch
    overrides: list[tuple[str, int]] = []
    for raw_override in raw_overrides:
        if (
            not isinstance(raw_override, Mapping)
            or set(raw_override) != {"method", "minutes"}
        ):
            raise _ResponseMismatch
        method = raw_override.get("method")
        minutes = raw_override.get("minutes")
        if (
            method not in {"email", "popup"}
            or isinstance(minutes, bool)
            or not isinstance(minutes, int)
            or not 0 <= minutes <= 40320
        ):
            raise _ResponseMismatch
        overrides.append((method, minutes))
    return True, use_default, tuple(overrides)


def _provider_has_mapping_data(event: Mapping[str, Any], field: str) -> bool:
    value = event.get(field)
    if value is None:
        return False
    if not isinstance(value, Mapping):
        raise _ResponseMismatch
    _validate_metadata_json(value)
    return bool(value)


def _provider_has_list_data(event: Mapping[str, Any], field: str) -> bool:
    value = event.get(field)
    if value is None:
        return False
    if not isinstance(value, list):
        raise _ResponseMismatch
    _validate_metadata_json(value)
    return bool(value)


def _provider_time(value: Any) -> tuple[str, bool, str | None]:
    if not isinstance(value, Mapping):
        raise _ResponseMismatch
    date_value = value.get("date")
    datetime_value = value.get("dateTime")
    timezone = value.get("timeZone")
    if timezone is not None and (
        not isinstance(timezone, str)
        or not timezone.strip()
        or len(timezone) > 128
        or any(ord(character) < 32 for character in timezone)
    ):
        raise _ResponseMismatch
    if date_value is not None:
        if datetime_value is not None or not isinstance(date_value, str):
            raise _ResponseMismatch
        if not _DATE_RE.fullmatch(date_value):
            raise _ResponseMismatch
        try:
            date.fromisoformat(date_value)
        except ValueError:
            raise _ResponseMismatch from None
        return date_value, True, timezone
    if not isinstance(datetime_value, str) or _parse_datetime(datetime_value) is None:
        raise _ResponseMismatch
    return datetime_value, False, timezone


def _provider_original_start(value: Any) -> str | None:
    if value is None:
        return None
    start, _all_day, _timezone = _provider_time(value)
    return start


def _provider_identity(
    event: Mapping[str, Any], field: str
) -> tuple[str | None, bool | None]:
    """Parse the ownership subset emitted by the pinned MCP converter."""
    value = event.get(field)
    if value is None:
        return None, None
    if not isinstance(value, Mapping) or set(value) - {
        "email",
        "displayName",
        "self",
    }:
        raise _ResponseMismatch
    email = value.get("email")
    display_name = value.get("displayName")
    is_self = value.get("self")
    if (
        not isinstance(email, str)
        or len(email) > 320
        or any(ord(character) < 32 for character in email)
        or (
            display_name is not None
            and (
                not isinstance(display_name, str)
                or len(display_name) > _MAX_PROVIDER_TEXT_CHARS
            )
        )
        or (is_self is not None and not isinstance(is_self, bool))
    ):
        raise _ResponseMismatch
    return email or None, is_self


def _snapshot_from_event(
    event: Any,
    *,
    logical_account: str,
    mcp_account: str,
    requested_calendar_id: str,
    expected_event_id: str,
) -> CalendarEventSnapshot:
    if not isinstance(event, Mapping):
        raise _ResponseMismatch
    if event.get("id") != expected_event_id:
        raise _ResponseMismatch
    provider_account = event.get("accountId")
    if provider_account is not None and provider_account != mcp_account:
        raise _ResponseMismatch
    provider_calendar = event.get("calendarId", requested_calendar_id)
    if not isinstance(provider_calendar, str) or not provider_calendar:
        raise _ResponseMismatch

    start_at, start_all_day, start_timezone = _provider_time(event.get("start"))
    end_at, end_all_day, end_timezone = _provider_time(event.get("end"))
    if start_all_day != end_all_day:
        raise _ResponseMismatch
    if start_all_day:
        if date.fromisoformat(end_at) <= date.fromisoformat(start_at):
            raise _ResponseMismatch
    else:
        parsed_start = _parse_datetime(start_at)
        parsed_end = _parse_datetime(end_at)
        if parsed_start is None or parsed_end is None or parsed_end <= parsed_start:
            raise _ResponseMismatch

    status = event.get("status")
    if status not in {"confirmed", "tentative", "cancelled"}:
        raise _ResponseMismatch
    recurrence = event.get("recurrence")
    if recurrence is None:
        recurrence_rules: tuple[str, ...] = ()
    elif (
        isinstance(recurrence, list)
        and len(recurrence) <= _MAX_PROVIDER_COLLECTION_ITEMS
        and all(
            isinstance(rule, str)
            and len(rule) <= _MAX_PROVIDER_TEXT_CHARS
            for rule in recurrence
        )
    ):
        recurrence_rules = tuple(recurrence)
    else:
        raise _ResponseMismatch

    attendees = event.get("attendees")
    attendee_emails: list[str] = []
    if attendees is not None:
        if (
            not isinstance(attendees, list)
            or len(attendees) > _MAX_PROVIDER_COLLECTION_ITEMS
        ):
            raise _ResponseMismatch
        for attendee in attendees:
            if not isinstance(attendee, Mapping):
                raise _ResponseMismatch
            email = attendee.get("email")
            if (
                not isinstance(email, str)
                or not email
                or len(email) > 320
                or any(ord(character) < 32 for character in email)
            ):
                raise _ResponseMismatch
            attendee_emails.append(email)

    creator_email, creator_is_self = _provider_identity(event, "creator")
    organizer_email, organizer_is_self = _provider_identity(event, "organizer")
    reminders_present, reminders_use_default, reminder_overrides = (
        _provider_reminders(event)
    )
    color_id = _optional_provider_text(event, "colorId")
    transparency = _optional_provider_text(event, "transparency")
    visibility = _optional_provider_text(event, "visibility")
    event_type = _optional_provider_text(event, "eventType")
    if transparency not in {None, "opaque", "transparent"}:
        raise _ResponseMismatch
    if visibility not in {None, "default", "public", "private", "confidential"}:
        raise _ResponseMismatch
    if event_type not in {
        None,
        "default",
        "birthday",
        "fromGmail",
        "outOfOffice",
        "focusTime",
        "workingLocation",
    }:
        raise _ResponseMismatch
    hangout_link = _optional_provider_text(event, "hangoutLink")

    return CalendarEventSnapshot(
        account=logical_account,
        calendar_id=provider_calendar,
        event_id=expected_event_id,
        title=_optional_provider_text(event, "summary"),
        description=_optional_provider_text(event, "description"),
        location=_optional_provider_text(event, "location"),
        start_at=start_at,
        end_at=end_at,
        all_day=start_all_day,
        timezone=start_timezone or end_timezone,
        status=status,
        html_link=_safe_html_link(event.get("htmlLink")),
        recurrence_rrules=recurrence_rules,
        attendee_emails=tuple(attendee_emails),
        updated_at=_optional_provider_text(event, "updated"),
        recurring_event_id=_optional_provider_text(event, "recurringEventId"),
        original_start_at=_provider_original_start(event.get("originalStartTime")),
        color_id=color_id,
        transparency=transparency,
        visibility=visibility,
        event_type=event_type,
        creator_email=creator_email,
        creator_is_self=creator_is_self,
        organizer_email=organizer_email,
        organizer_is_self=organizer_is_self,
        reminders_present=reminders_present,
        reminders_use_default=reminders_use_default,
        reminder_overrides=reminder_overrides,
        has_conference_data=_provider_has_mapping_data(event, "conferenceData"),
        has_hangout_link=bool(hangout_link),
        has_attachments=_provider_has_list_data(event, "attachments"),
        has_extended_properties=_provider_has_mapping_data(
            event, "extendedProperties"
        ),
        has_source=_provider_has_mapping_data(event, "source"),
        anyone_can_add_self=_optional_provider_bool(event, "anyoneCanAddSelf"),
        guests_can_invite_others=_optional_provider_bool(
            event, "guestsCanInviteOthers"
        ),
        guests_can_modify=_optional_provider_bool(event, "guestsCanModify"),
        guests_can_see_other_guests=_optional_provider_bool(
            event, "guestsCanSeeOtherGuests"
        ),
        private_copy=_optional_provider_bool(event, "privateCopy"),
        locked=_optional_provider_bool(event, "locked"),
        safety_metadata_complete=True,
        safety_metadata_fingerprint=_metadata_fingerprint(event),
    )


def _normalize_update_patch(patch: Any) -> dict[str, Any]:
    if not isinstance(patch, Mapping):
        raise CalendarMcpError("Calendar event patch is invalid")
    normalized = dict(patch)
    if not normalized or not all(isinstance(key, str) for key in normalized):
        raise CalendarMcpError("Calendar event patch is invalid")
    if set(normalized) - _UPDATE_PATCH_FIELDS:
        raise CalendarMcpError("Calendar event patch contains unsupported fields")

    title = normalized.get("title")
    if "title" in normalized and (
        not isinstance(title, str) or not title.strip()
    ):
        raise CalendarMcpError("Calendar event patch is invalid")
    for field in ("description", "location"):
        if field in normalized and not isinstance(normalized[field], str):
            raise CalendarMcpError("Calendar event patch is invalid")
    if "recurrence_rrules" in normalized:
        recurrence_rules = normalized["recurrence_rrules"]
        if (
            not isinstance(recurrence_rules, Sequence)
            or isinstance(recurrence_rules, (str, bytes, bytearray))
            or len(recurrence_rules) > _MAX_PROVIDER_COLLECTION_ITEMS
            or any(
                not isinstance(rule, str)
                or not rule
                or len(rule) > _MAX_PROVIDER_TEXT_CHARS
                or "\r" in rule
                or "\n" in rule
                for rule in recurrence_rules
            )
        ):
            raise CalendarMcpError("Calendar event recurrence patch is invalid")
        normalized["recurrence_rrules"] = tuple(recurrence_rules)

    has_start = "start_at" in normalized
    has_end = "end_at" in normalized
    if has_start != has_end:
        raise CalendarMcpError("Calendar event patch must update start and end together")
    if "timezone" in normalized:
        timezone = normalized["timezone"]
        if not isinstance(timezone, str) or not timezone.strip() or not has_start:
            raise CalendarMcpError("Calendar event patch is invalid")
    if not has_start and set(normalized) == {"timezone"}:
        raise CalendarMcpError("Calendar event patch is invalid")

    if has_start:
        start = normalized["start_at"]
        end = normalized["end_at"]
        if not isinstance(start, str) or not isinstance(end, str):
            raise CalendarMcpError("Calendar event patch is invalid")
        start_all_day = _DATE_RE.fullmatch(start) is not None
        end_all_day = _DATE_RE.fullmatch(end) is not None
        if start_all_day != end_all_day:
            raise CalendarMcpError("Calendar event patch mixes date and datetime")
        if start_all_day:
            try:
                if date.fromisoformat(end) <= date.fromisoformat(start):
                    raise CalendarMcpError("Calendar event patch has an invalid range")
            except ValueError:
                raise CalendarMcpError("Calendar event patch is invalid") from None
        else:
            if not _AWARE_DATETIME_RE.fullmatch(start) or not _AWARE_DATETIME_RE.fullmatch(
                end
            ):
                raise CalendarMcpError("Calendar event patch is invalid")
            parsed_start = _parse_datetime(start)
            parsed_end = _parse_datetime(end)
            if parsed_start is None or parsed_end is None or parsed_end <= parsed_start:
                raise CalendarMcpError("Calendar event patch has an invalid range")

    if not (set(normalized) - {"timezone"}):
        raise CalendarMcpError("Calendar event patch is invalid")
    return normalized


def _snapshot_matches_patch(
    snapshot: CalendarEventSnapshot, patch: Mapping[str, Any]
) -> bool:
    if snapshot.status == "cancelled":
        return False
    if "title" in patch and snapshot.title != patch["title"]:
        return False
    for field in ("description", "location"):
        if field not in patch:
            continue
        expected = patch[field]
        actual = getattr(snapshot, field)
        if expected == "":
            if actual not in (None, ""):
                return False
        elif actual != expected:
            return False
    if "recurrence_rrules" in patch:
        expected_recurrence = _canonical_recurrence_rules(
            patch["recurrence_rrules"]
        )
        actual_recurrence = _canonical_recurrence_rules(
            snapshot.recurrence_rrules
        )
        if (
            expected_recurrence is None
            or actual_recurrence is None
            or actual_recurrence != expected_recurrence
        ):
            return False
    if "start_at" in patch:
        expected_start = patch["start_at"]
        expected_end = patch["end_at"]
        expected_all_day = _DATE_RE.fullmatch(expected_start) is not None
        if snapshot.all_day != expected_all_day:
            return False
        if (
            not expected_all_day
            and "timezone" in patch
            and snapshot.timezone != patch["timezone"]
        ):
            return False
        if expected_all_day:
            if snapshot.start_at != expected_start or snapshot.end_at != expected_end:
                return False
        else:
            actual_start = _parse_datetime(snapshot.start_at)
            actual_end = _parse_datetime(snapshot.end_at)
            requested_start = _parse_datetime(expected_start)
            requested_end = _parse_datetime(expected_end)
            if (
                actual_start is None
                or actual_end is None
                or actual_start != requested_start
                or actual_end != requested_end
            ):
                return False
    return True


def _canonical_recurrence_rules(value: Any) -> tuple[str, ...] | None:
    """Canonicalize provider RRULE ordering without weakening comparison.

    Google may return an equivalent RRULE with its ``KEY=VALUE`` components in
    a different order.  Recurrence lines and RRULE components are unordered for
    equality purposes, while every key and value must still match exactly.
    Invalid or duplicate RRULE components remain a mismatch.
    """

    if value in (None, [], ()):
        return ()
    if not isinstance(value, Sequence) or isinstance(
        value, (str, bytes, bytearray)
    ):
        return None

    normalized: list[str] = []
    for rule in value:
        if not isinstance(rule, str) or not rule:
            return None
        if not rule.startswith("RRULE:"):
            normalized.append(rule)
            continue
        components = rule[len("RRULE:") :].split(";")
        parsed: list[tuple[str, str]] = []
        seen_keys: set[str] = set()
        for component in components:
            key, separator, component_value = component.partition("=")
            if (
                separator != "="
                or not key
                or not component_value
                or key in seen_keys
            ):
                return None
            seen_keys.add(key)
            parsed.append((key, component_value))
        normalized.append(
            "RRULE:"
            + ";".join(
                f"{key}={component_value}"
                for key, component_value in sorted(parsed)
            )
        )
    return tuple(sorted(normalized))


def _snapshot_matches_precondition(
    current: CalendarEventSnapshot,
    expected: CalendarEventSnapshot,
) -> bool:
    """Compare mutation-relevant state while ignoring provider bookkeeping."""

    return all(
        getattr(current, field.name) == getattr(expected, field.name)
        for field in fields(CalendarEventSnapshot)
        if field.name not in {"html_link", "updated_at"}
    )


def _validate_query_range(time_min: Any, time_max: Any) -> tuple[str, str]:
    values = (time_min, time_max)
    if any(
        not isinstance(value, str)
        or len(value) > 64
        or value != value.strip()
        or _AWARE_DATETIME_RE.fullmatch(value) is None
        for value in values
    ):
        raise CalendarMcpError("Calendar event query range is invalid")
    parsed_min = _parse_datetime(time_min)
    parsed_max = _parse_datetime(time_max)
    if (
        parsed_min is None
        or parsed_max is None
        or parsed_max <= parsed_min
        or parsed_max - parsed_min > timedelta(days=31)
    ):
        raise CalendarMcpError("Calendar event query range is invalid")
    return time_min, time_max


def _validate_query_limit(limit: Any) -> int:
    if (
        isinstance(limit, bool)
        or not isinstance(limit, int)
        or not 1 <= limit <= _MAX_QUERY_LIMIT
    ):
        raise CalendarMcpError("Calendar event query limit is invalid")
    return limit


def _validate_search_query(query: Any) -> str:
    if not isinstance(query, str) or len(query) > _MAX_QUERY_TEXT_CHARS:
        raise CalendarMcpError("Calendar event search query is invalid")
    normalized = query.strip()
    if (
        not normalized
        or len(normalized) > _MAX_QUERY_TEXT_CHARS
        or any(
            ord(character) < 32 or ord(character) == 127
            for character in normalized
        )
    ):
        raise CalendarMcpError("Calendar event search query is invalid")
    return normalized


def _query_snapshot_sort_key(
    snapshot: CalendarEventSnapshot,
) -> tuple[datetime, int, str, str, str]:
    if snapshot.all_day:
        start = datetime.combine(
            date.fromisoformat(snapshot.start_at),
            datetime.min.time(),
            tzinfo=timezone.utc,
        )
        kind = 0
    else:
        parsed = _parse_datetime(snapshot.start_at)
        if parsed is None:  # Parsed and checked by _snapshot_from_event.
            raise _ResponseMismatch
        start = parsed.astimezone(timezone.utc)
        kind = 1
    return (
        start,
        kind,
        snapshot.end_at,
        snapshot.calendar_id,
        snapshot.event_id,
    )


def _validate_search_response_metadata(
    payload: Mapping[str, Any],
    *,
    query: str,
    time_min: str,
    time_max: str,
) -> str:
    if payload.get("query") != query:
        raise _ResponseMismatch
    calendar_id = payload.get("calendarId")
    if (
        not isinstance(calendar_id, str)
        or not calendar_id
        or len(calendar_id) > 1024
        or any(ord(character) < 32 for character in calendar_id)
    ):
        raise _ResponseMismatch
    time_range = payload.get("timeRange")
    if not isinstance(time_range, Mapping) or set(time_range) != {"start", "end"}:
        raise _ResponseMismatch
    response_min = _parse_datetime(time_range.get("start"))
    response_max = _parse_datetime(time_range.get("end"))
    requested_min = _parse_datetime(time_min)
    requested_max = _parse_datetime(time_max)
    if (
        response_min is None
        or response_max is None
        or response_min != requested_min
        or response_max != requested_max
    ):
        raise _ResponseMismatch
    return calendar_id


def _query_result_from_payload(
    payload: Mapping[str, Any],
    *,
    tool_name: str,
    logical_account: str,
    mcp_account: str,
    requested_calendar_id: str,
    time_min: str,
    time_max: str,
    limit: int,
    query: str | None = None,
) -> CalendarEventQueryResult:
    # The pinned MCP omits these keys when empty.  Their presence means the
    # provider returned only a partial view, which must never be treated as a
    # safe target set for later mutation.
    if "warnings" in payload or "partialFailures" in payload:
        raise _ResponseMismatch

    if tool_name == "list-events":
        if set(payload) != {"events", "totalCount"} or query is not None:
            raise _ResponseMismatch
        response_calendar_id: str | None = None
    elif tool_name == "search-events":
        if set(payload) != {
            "events",
            "totalCount",
            "query",
            "calendarId",
            "timeRange",
        } or query is None:
            raise _ResponseMismatch
        response_calendar_id = _validate_search_response_metadata(
            payload,
            query=query,
            time_min=time_min,
            time_max=time_max,
        )
    else:  # Internal invariant.
        raise _ResponseMismatch

    raw_events = payload.get("events")
    total_count = payload.get("totalCount")
    if (
        not isinstance(raw_events, list)
        or isinstance(total_count, bool)
        or not isinstance(total_count, int)
        or total_count != len(raw_events)
        or not 0 <= total_count <= _UPSTREAM_EVENT_PAGE_SIZE
    ):
        raise _ResponseMismatch

    snapshots: list[CalendarEventSnapshot] = []
    seen_event_ids: set[str] = set()
    observed_calendar_id: str | None = None
    for event in raw_events:
        if not isinstance(event, Mapping):
            raise _ResponseMismatch
        event_id = event.get("id")
        try:
            event_id = _validate_target_event_id(event_id)
        except CalendarMcpError:
            raise _ResponseMismatch from None
        if event_id in seen_event_ids:
            raise _ResponseMismatch
        seen_event_ids.add(event_id)

        if event.get("accountId") != mcp_account:
            raise _ResponseMismatch
        provider_calendar_id = event.get("calendarId")
        if (
            not isinstance(provider_calendar_id, str)
            or not provider_calendar_id
            or len(provider_calendar_id) > 1024
            or any(ord(character) < 32 for character in provider_calendar_id)
        ):
            raise _ResponseMismatch
        if (
            response_calendar_id is not None
            and provider_calendar_id != response_calendar_id
        ):
            raise _ResponseMismatch
        if observed_calendar_id is None:
            observed_calendar_id = provider_calendar_id
        elif provider_calendar_id != observed_calendar_id:
            raise _ResponseMismatch

        snapshot = _snapshot_from_event(
            event,
            logical_account=logical_account,
            mcp_account=mcp_account,
            requested_calendar_id=requested_calendar_id,
            expected_event_id=event_id,
        )
        if snapshot.status in {"confirmed", "tentative"}:
            snapshots.append(snapshot)

    snapshots.sort(key=_query_snapshot_sort_key)
    return CalendarEventQueryResult(
        events=tuple(snapshots[:limit]),
        total_count=total_count,
        may_be_incomplete=(
            total_count > limit or total_count >= _UPSTREAM_EVENT_PAGE_SIZE
        ),
    )


class GoogleCalendarMcpClient:
    """Idempotent implementation of :class:`~.calendar.CalendarClient`."""

    def __init__(
        self,
        session: ClientSession,
        *,
        account_mapping: Mapping[str, str],
        calendar_id: str = "primary",
        calendar_id_by_account: Mapping[str, str] | None = None,
        default_timeout_seconds: int = 30,
    ) -> None:
        if not isinstance(default_timeout_seconds, int) or default_timeout_seconds <= 0:
            raise ValueError("default_timeout_seconds must be positive")
        if not isinstance(calendar_id, str) or not calendar_id.strip():
            raise ValueError("calendar_id must be a non-empty string")
        if not account_mapping:
            raise ValueError("account_mapping must not be empty")

        copied_mapping: dict[str, str] = {}
        for logical_account, mcp_account in account_mapping.items():
            if not isinstance(logical_account, str) or not logical_account:
                raise ValueError("logical account names must be non-empty strings")
            if not isinstance(mcp_account, str) or not _MCP_ACCOUNT_RE.fullmatch(
                mcp_account
            ):
                raise ValueError("MCP account names must use the supported format")
            copied_mapping[logical_account] = mcp_account

        copied_calendars: dict[str, str] = {}
        for logical_account, mapped_calendar_id in (
            calendar_id_by_account or {}
        ).items():
            if logical_account not in copied_mapping:
                raise ValueError("calendar mapping refers to an unknown account")
            if (
                not isinstance(mapped_calendar_id, str)
                or not mapped_calendar_id.strip()
            ):
                raise ValueError("mapped calendar IDs must be non-empty strings")
            copied_calendars[logical_account] = mapped_calendar_id

        self._session = session
        self._account_mapping = copied_mapping
        self._default_calendar_id = calendar_id
        self._calendar_id_by_account = copied_calendars
        self._timeout = default_timeout_seconds

    def _target(self, logical_account: str) -> tuple[str, str]:
        try:
            mcp_account = self._account_mapping[logical_account]
        except (KeyError, TypeError):
            raise CalendarMcpError("Calendar account is not configured") from None
        calendar_id = self._calendar_id_by_account.get(
            logical_account, self._default_calendar_id
        )
        return mcp_account, calendar_id

    async def _tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        started = time.monotonic()
        safe_tool = name if name in CALENDAR_MCP_TOOLS else "unknown"
        raw_account = arguments.get("account")
        account = (
            raw_account
            if isinstance(raw_account, str) and _MCP_ACCOUNT_RE.fullmatch(raw_account)
            else "none"
        )
        LOGGER.info(
            "Calendar MCP tool started; tool=%s account=%s status=started "
            "timeout=%ss",
            safe_tool,
            account,
            self._timeout,
        )
        try:
            result = await self._session.call_tool(
                name,
                arguments,
                read_timeout_seconds=timedelta(seconds=self._timeout),
            )
        except Exception as exc:
            # Do not chain an MCP exception: provider errors may include event
            # text, account metadata, or credential-related diagnostics.
            if _is_connection_failure(exc):
                LOGGER.error(
                    "Calendar MCP tool finished; tool=%s account=%s "
                    "status=connection_error elapsed=%.3fs error_type=%s",
                    safe_tool,
                    account,
                    time.monotonic() - started,
                    type(exc).__name__,
                )
                raise _connection_error() from None
            LOGGER.warning(
                "Calendar MCP tool finished; tool=%s account=%s "
                "status=transport_error elapsed=%.3fs error_type=%s",
                safe_tool,
                account,
                time.monotonic() - started,
                type(exc).__name__,
            )
            raise _ToolFailure from None
        if bool(_result_attribute(result, "isError", False)):
            error = _classify_tool_error(name, arguments, result)
            LOGGER.warning(
                "Calendar MCP tool finished; tool=%s account=%s "
                "status=tool_error elapsed=%.3fs error_type=%s",
                safe_tool,
                account,
                time.monotonic() - started,
                error.__name__,
            )
            raise error
        try:
            payload = _parse_tool_payload(result)
        except _ResponseMismatch:
            LOGGER.warning(
                "Calendar MCP tool finished; tool=%s account=%s "
                "status=invalid_response elapsed=%.3fs error_type=%s",
                safe_tool,
                account,
                time.monotonic() - started,
                _ResponseMismatch.__name__,
            )
            raise _ToolFailure from None
        LOGGER.info(
            "Calendar MCP tool finished; tool=%s account=%s status=success "
            "elapsed=%.3fs",
            safe_tool,
            account,
            time.monotonic() - started,
        )
        return payload

    async def list_calendars(self, account: str) -> tuple[dict[str, Any], ...]:
        """List calendars for a configured logical account."""
        mcp_account, _ = self._target(account)
        try:
            payload = await self._tool("list-calendars", {"account": mcp_account})
        except _ToolFailure:
            raise CalendarMcpError("Calendar MCP validation failed") from None
        calendars = payload.get("calendars")
        if not isinstance(calendars, list) or not all(
            isinstance(item, dict) for item in calendars
        ):
            raise CalendarMcpError("Calendar MCP returned an invalid response")
        return tuple(dict(item) for item in calendars)

    async def validate(self) -> None:
        """Verify every supplied account mapping and target calendar."""
        checked: set[tuple[str, str]] = set()
        for logical_account in self._account_mapping:
            mcp_account, calendar_id = self._target(logical_account)
            target = (mcp_account, calendar_id)
            if target in checked:
                continue
            checked.add(target)
            calendars = await self.list_calendars(logical_account)
            if calendar_id == "primary":
                found = any(item.get("primary") is True for item in calendars)
            else:
                found = any(
                    calendar_id
                    in {
                        item.get("id"),
                        item.get("summary"),
                        item.get("summaryOverride"),
                    }
                    for item in calendars
                )
            if not found:
                raise CalendarMcpError("Configured calendar is unavailable")

    async def _discover_events(
        self,
        *,
        tool_name: str,
        account: str,
        time_min: str,
        time_max: str,
        limit: int,
        query: str | None = None,
    ) -> CalendarEventQueryResult:
        mcp_account, calendar_id = self._target(account)
        arguments: dict[str, Any] = {
            "account": mcp_account,
            "calendarId": calendar_id,
            "timeMin": time_min,
            "timeMax": time_max,
            "fields": list(_SNAPSHOT_EVENT_FIELDS),
        }
        if query is not None:
            arguments["query"] = query
        try:
            payload = await self._tool(tool_name, arguments)
            return _query_result_from_payload(
                payload,
                tool_name=tool_name,
                logical_account=account,
                mcp_account=mcp_account,
                requested_calendar_id=calendar_id,
                time_min=time_min,
                time_max=time_max,
                limit=limit,
                query=query,
            )
        except (_ToolFailure, _ResponseMismatch):
            raise CalendarMcpError("Calendar event discovery failed") from None

    async def list_events(
        self,
        *,
        account: str,
        time_min: str,
        time_max: str,
        limit: int = 50,
    ) -> CalendarEventQueryResult:
        """List a bounded, chronological event window for one account."""
        time_min, time_max = _validate_query_range(time_min, time_max)
        limit = _validate_query_limit(limit)
        return await self._discover_events(
            tool_name="list-events",
            account=account,
            time_min=time_min,
            time_max=time_max,
            limit=limit,
        )

    async def search_events(
        self,
        *,
        account: str,
        query: str,
        time_min: str,
        time_max: str,
        limit: int = 50,
    ) -> CalendarEventQueryResult:
        """Search a bounded event window without trusting partial MCP data."""
        time_min, time_max = _validate_query_range(time_min, time_max)
        limit = _validate_query_limit(limit)
        query = _validate_search_query(query)
        return await self._discover_events(
            tool_name="search-events",
            account=account,
            query=query,
            time_min=time_min,
            time_max=time_max,
            limit=limit,
        )

    async def _read_snapshot(
        self, *, account: str, event_id: str
    ) -> CalendarEventSnapshot:
        event_id = _validate_target_event_id(event_id)
        mcp_account, calendar_id = self._target(account)
        payload = await self._tool(
            "get-event",
            {
                "account": mcp_account,
                "calendarId": calendar_id,
                "eventId": event_id,
                "fields": list(_SNAPSHOT_EVENT_FIELDS),
            },
        )
        return _snapshot_from_event(
            payload.get("event"),
            logical_account=account,
            mcp_account=mcp_account,
            requested_calendar_id=calendar_id,
            expected_event_id=event_id,
        )

    async def _probe_snapshot(
        self, *, account: str, event_id: str
    ) -> CalendarEventSnapshot | None:
        try:
            return await self._read_snapshot(account=account, event_id=event_id)
        except CalendarConnectionError:
            raise
        except (_ToolFailure, _ResponseMismatch, CalendarMcpError):
            return None

    async def get_event(
        self, *, account: str, event_id: str
    ) -> CalendarEventSnapshot:
        """Return one fully normalized event from the configured calendar."""
        try:
            return await self._read_snapshot(account=account, event_id=event_id)
        except CalendarMcpError:
            raise
        except (_ToolFailure, _ResponseMismatch):
            raise CalendarMcpError("Calendar event read failed") from None

    async def update_event(
        self,
        *,
        account: str,
        event_id: str,
        patch: Mapping[str, Any],
        idempotency_key: str,
        expected_current: CalendarEventSnapshot | None = None,
    ) -> UpdatedCalendarEvent:
        """Apply a retry-safe patch and verify the provider's resulting state.

        The idempotency key belongs to the caller's durable operation journal.
        Calendar MCP cannot persist it, so this boundary obtains the current
        event before writing, skips an already-applied patch, and probes after
        any ambiguous provider response.
        """
        _validate_idempotency_key(idempotency_key)
        event_id = _validate_target_event_id(event_id)
        normalized_patch = _normalize_update_patch(patch)
        if expected_current is not None and not isinstance(
            expected_current, CalendarEventSnapshot
        ):
            raise CalendarMcpError("Calendar state precondition is invalid")
        try:
            before = await self._read_snapshot(account=account, event_id=event_id)
        except CalendarMcpError:
            raise
        except (_ToolFailure, _ResponseMismatch):
            raise CalendarMcpError("Calendar event update failed") from None
        # Reconciliation takes precedence over the precondition: if a previous
        # attempt already applied this exact patch, this call performs no write.
        if _snapshot_matches_patch(before, normalized_patch):
            return UpdatedCalendarEvent(
                previous=before,
                current=before,
                already_applied=True,
            )
        if expected_current is not None and not _snapshot_matches_precondition(
            before, expected_current
        ):
            raise CalendarStateConflictError(
                "Calendar event changed after it was observed"
            ) from None
        if before.status == "cancelled":
            raise CalendarMcpError("Calendar event is deleted")

        mcp_account, calendar_id = self._target(account)
        arguments: dict[str, Any] = {
            "account": mcp_account,
            "calendarId": calendar_id,
            "eventId": event_id,
            # Upstream 2.6.2 currently ignores this update argument, but keep
            # the explicit safe policy for compatibility with fixed releases.
            "sendUpdates": "none",
        }
        field_mapping = {
            "title": "summary",
            "description": "description",
            "location": "location",
            "start_at": "start",
            "end_at": "end",
            "timezone": "timeZone",
        }
        for field, value in normalized_patch.items():
            if field == "recurrence_rrules":
                arguments["recurrence"] = list(value)
                arguments["modificationScope"] = "all"
            else:
                arguments[field_mapping[field]] = value

        try:
            payload = await self._tool("update-event", arguments)
            updated = _snapshot_from_event(
                payload.get("event"),
                logical_account=account,
                mcp_account=mcp_account,
                requested_calendar_id=calendar_id,
                expected_event_id=event_id,
            )
            if _snapshot_matches_patch(updated, normalized_patch):
                return UpdatedCalendarEvent(previous=before, current=updated)
        except (_ToolFailure, _ResponseMismatch):
            pass

        recovered = await self._probe_snapshot(account=account, event_id=event_id)
        if recovered is not None and _snapshot_matches_patch(
            recovered, normalized_patch
        ):
            return UpdatedCalendarEvent(previous=before, current=recovered)
        raise CalendarMcpError("Calendar event update failed") from None

    async def delete_event(
        self,
        *,
        account: str,
        event_id: str,
        idempotency_key: str,
        expected_current: CalendarEventSnapshot | None = None,
    ) -> DeletedCalendarEvent:
        """Delete one target without guest notifications, with retry probing."""
        _validate_idempotency_key(idempotency_key)
        event_id = _validate_target_event_id(event_id)
        if expected_current is not None and not isinstance(
            expected_current, CalendarEventSnapshot
        ):
            raise CalendarMcpError("Calendar state precondition is invalid")
        try:
            previous = await self._read_snapshot(account=account, event_id=event_id)
        except CalendarMcpError:
            raise
        except (_ToolFailure, _ResponseMismatch):
            raise CalendarMcpError("Calendar event deletion failed") from None
        if previous.status == "cancelled":
            return DeletedCalendarEvent(
                previous=previous,
                current=previous,
                already_deleted=True,
                verified_cancelled=True,
            )
        if expected_current is not None and not _snapshot_matches_precondition(
            previous, expected_current
        ):
            raise CalendarStateConflictError(
                "Calendar event changed after it was observed"
            ) from None

        mcp_account, calendar_id = self._target(account)
        try:
            payload = await self._tool(
                "delete-event",
                {
                    "account": mcp_account,
                    "calendarId": calendar_id,
                    "eventId": event_id,
                    "sendUpdates": "none",
                },
            )
            if (
                payload.get("success") is not True
                or payload.get("eventId") != event_id
                or not isinstance(payload.get("calendarId"), str)
                or not payload["calendarId"]
            ):
                raise _ResponseMismatch
        except (_ToolFailure, _ResponseMismatch):
            recovered = await self._probe_snapshot(
                account=account, event_id=event_id
            )
            if recovered is not None and recovered.status == "cancelled":
                return DeletedCalendarEvent(
                    previous=previous,
                    current=recovered,
                    verified_cancelled=True,
                )
            raise CalendarMcpError("Calendar event deletion failed") from None

        current = await self._probe_snapshot(account=account, event_id=event_id)
        return DeletedCalendarEvent(
            previous=previous,
            current=current,
            verified_cancelled=current is not None and current.status == "cancelled",
        )

    @staticmethod
    def _event_arguments(
        event: Mapping[str, Any],
        *,
        mcp_account: str,
        calendar_id: str,
        event_id: str,
    ) -> dict[str, Any]:
        title = event.get("title")
        start = event.get("start_at")
        end = event.get("end_at")
        timezone = event.get("timezone")
        all_day = event.get("all_day")
        location = event.get("location")
        description = event.get("description")
        recurrence = event.get("recurrence_rrule")
        if (
            not isinstance(title, str)
            or not title
            or not isinstance(start, str)
            or not start
            or not isinstance(end, str)
            or not end
            or not isinstance(timezone, str)
            or not timezone
            or not isinstance(all_day, bool)
            or (location is not None and not isinstance(location, str))
            or (description is not None and not isinstance(description, str))
            or (recurrence is not None and not isinstance(recurrence, str))
        ):
            raise CalendarMcpError("Calendar event payload is invalid")
        if all_day:
            if not _DATE_RE.fullmatch(start) or not _DATE_RE.fullmatch(end):
                raise CalendarMcpError("Calendar event payload is invalid")
        elif _parse_datetime(start) is None or _parse_datetime(end) is None:
            raise CalendarMcpError("Calendar event payload is invalid")

        mcp_start = start
        mcp_end = end
        if recurrence and not all_day:
            mcp_start = _mcp_recurring_wall_time(start, timezone)
            mcp_end = _mcp_recurring_wall_time(end, timezone)

        arguments: dict[str, Any] = {
            "account": mcp_account,
            "calendarId": calendar_id,
            "eventId": event_id,
            "summary": title,
            "start": mcp_start,
            "end": mcp_end,
            "timeZone": timezone,
            # Deterministic IDs, rather than fuzzy title/time matching, define
            # this adapter's duplicate semantics.
            "allowDuplicates": True,
        }
        if description:
            arguments["description"] = description
        if location:
            arguments["location"] = location
        if recurrence:
            arguments["recurrence"] = [recurrence]
        return arguments

    @staticmethod
    def _response_matches(
        event: Any,
        *,
        arguments: Mapping[str, Any],
        requested_event: Mapping[str, Any],
    ) -> bool:
        if not isinstance(event, dict):
            return False
        if event.get("id") != arguments["eventId"]:
            return False
        if event.get("status") == "cancelled":
            return False
        if event.get("summary") != arguments["summary"]:
            return False
        expected_start = requested_event.get("start_at")
        expected_end = requested_event.get("end_at")
        all_day = requested_event.get("all_day")
        if (
            not isinstance(expected_start, str)
            or not isinstance(expected_end, str)
            or not isinstance(all_day, bool)
        ):
            return False
        if not _times_match(event.get("start"), expected_start, all_day=all_day):
            return False
        if not _times_match(event.get("end"), expected_end, all_day=all_day):
            return False
        if not _optional_text_matches(
            event.get("description"), arguments.get("description")
        ):
            return False
        if not _optional_text_matches(event.get("location"), arguments.get("location")):
            return False
        actual_recurrence = _canonical_recurrence_rules(
            event.get("recurrence")
        )
        expected_recurrence = _canonical_recurrence_rules(
            arguments.get("recurrence")
        )
        if (
            actual_recurrence is None
            or expected_recurrence is None
            or actual_recurrence != expected_recurrence
        ):
            return False
        if actual_recurrence and not all_day:
            expected_timezone = requested_event.get("timezone")
            if not isinstance(expected_timezone, str) or any(
                not isinstance(event.get(boundary), Mapping)
                or event[boundary].get("timeZone") != expected_timezone
                for boundary in ("start", "end")
            ):
                return False
        if event.get("accountId") not in (None, arguments["account"]):
            return False
        if (
            arguments["calendarId"] != "primary"
            and event.get("calendarId") not in (None, arguments["calendarId"])
        ):
            return False
        return True

    @classmethod
    def _created_event(
        cls,
        payload: Mapping[str, Any],
        *,
        arguments: Mapping[str, Any],
        requested_event: Mapping[str, Any],
    ) -> CreatedCalendarEvent:
        event = payload.get("event")
        if not cls._response_matches(
            event,
            arguments=arguments,
            requested_event=requested_event,
        ):
            raise _ResponseMismatch
        return CreatedCalendarEvent(
            event_id=arguments["eventId"],
            html_link=_safe_html_link(event.get("htmlLink")),
        )

    async def _recover_existing(
        self,
        arguments: Mapping[str, Any],
        requested_event: Mapping[str, Any],
        *,
        rejected_create: bool = False,
    ) -> CreatedCalendarEvent:
        get_arguments = {
            "account": arguments["account"],
            "calendarId": arguments["calendarId"],
            "eventId": arguments["eventId"],
            "fields": list(_SNAPSHOT_EVENT_FIELDS),
        }
        try:
            payload = await self._tool("get-event", get_arguments)
        except _ToolNotFound:
            if rejected_create:
                raise CalendarMcpWriteRejectedError(
                    "Calendar event creation was rejected"
                ) from None
            raise CalendarMcpError("Calendar event creation failed") from None
        except _ToolFailure:
            raise CalendarMcpError("Calendar event creation failed") from None
        try:
            return self._created_event(
                payload,
                arguments=arguments,
                requested_event=requested_event,
            )
        except _ResponseMismatch:
            raise CalendarMcpCollisionError(
                "Calendar event ID collision detected"
            ) from None

    async def create_events(
        self,
        *,
        account: str,
        events: Sequence[dict[str, Any]],
        idempotency_key: str,
    ) -> tuple[CreatedCalendarEvent, ...]:
        """Create a batch one-by-one with deterministic retry-safe IDs."""
        if not isinstance(idempotency_key, str) or not idempotency_key:
            raise CalendarMcpError("Calendar idempotency key is invalid")
        if not isinstance(events, Sequence) or isinstance(events, (str, bytes)):
            raise CalendarMcpError("Calendar event batch is invalid")
        mcp_account, calendar_id = self._target(account)

        created: list[CreatedCalendarEvent] = []
        for index, event in enumerate(events):
            if not isinstance(event, Mapping):
                raise CalendarMcpError("Calendar event payload is invalid")
            try:
                event_id = google_event_id(idempotency_key, index)
            except ValueError:
                raise CalendarMcpError("Calendar idempotency key is invalid") from None
            arguments = self._event_arguments(
                event,
                mcp_account=mcp_account,
                calendar_id=calendar_id,
                event_id=event_id,
            )
            try:
                payload = await self._tool("create-event", arguments)
                result = self._created_event(
                    payload,
                    arguments=arguments,
                    requested_event=event,
                )
            except _ToolRejected:
                # A Google 400 is definite only after an exact deterministic-ID
                # read proves that no event was committed. Any failed or
                # malformed probe remains ambiguous and follows the durable
                # retry path instead.
                result = await self._recover_existing(
                    arguments,
                    event,
                    rejected_create=True,
                )
            except (_ToolFailure, _ResponseMismatch):
                # The package performs fuzzy duplicate detection before insert,
                # so a replay can be reported as either a 409 or a generic
                # duplicate error.  Probe the deterministic ID after *any*
                # create failure, and accept it only when the stored payload
                # matches the requested event.
                result = await self._recover_existing(arguments, event)
            created.append(result)
        return tuple(created)


@asynccontextmanager
async def open_calendar_mcp(
    binary_path: Path,
    *,
    account_mapping: Mapping[str, str],
    calendar_id: str = "primary",
    calendar_id_by_account: Mapping[str, str] | None = None,
    default_timeout_seconds: int = 30,
    working_directory: Path | None = None,
    env: Mapping[str, str] | None = None,
) -> AsyncIterator[GoogleCalendarMcpClient]:
    """Start the installed ``@cocal/google-calendar-mcp`` stdio binary.

    Server stderr is intentionally discarded because upstream diagnostics may
    contain event or authentication context.  No OAuth flow is started here.
    """
    started = time.monotonic()
    LOGGER.info("Calendar MCP lifecycle; status=starting")
    path = Path(binary_path).expanduser().resolve()
    if not path.is_file() or not os.access(path, os.X_OK):
        LOGGER.error(
            "Calendar MCP lifecycle; status=binary_unavailable elapsed=%.3fs",
            time.monotonic() - started,
        )
        raise CalendarMcpError("Calendar MCP binary is unavailable")
    cwd = (
        Path(working_directory).expanduser().resolve()
        if working_directory is not None
        else path.parent
    )
    if not cwd.is_dir():
        LOGGER.error(
            "Calendar MCP lifecycle; status=working_directory_unavailable "
            "elapsed=%.3fs",
            time.monotonic() - started,
        )
        raise CalendarMcpError("Calendar MCP working directory is unavailable")
    child_env: dict[str, str] = {}
    for key, value in (env or {}).items():
        if not isinstance(key, str) or not key or not isinstance(value, str):
            raise CalendarMcpError("Calendar MCP environment is invalid")
        child_env[key] = value
    parameters = StdioServerParameters(
        command=str(path),
        args=["start", "--enable-tools", ",".join(CALENDAR_MCP_TOOLS)],
        cwd=cwd,
        env=child_env or None,
    )
    with open(os.devnull, "w", encoding="utf-8") as stderr_sink:
        stack = AsyncExitStack()
        try:
            reader, writer = await stack.enter_async_context(
                stdio_client(parameters, errlog=stderr_sink)
            )
            session = await stack.enter_async_context(ClientSession(reader, writer))
            await session.initialize()
            client = GoogleCalendarMcpClient(
                session,
                account_mapping=account_mapping,
                calendar_id=calendar_id,
                calendar_id_by_account=calendar_id_by_account,
                default_timeout_seconds=default_timeout_seconds,
            )
        except Exception as exc:
            connection_failed = _is_connection_failure(exc)
            try:
                await stack.aclose()
            except Exception as close_exc:
                connection_failed = connection_failed or _is_connection_failure(
                    close_exc
                )
            if connection_failed:
                LOGGER.error(
                    "Calendar MCP lifecycle; status=connection_error "
                    "elapsed=%.3fs error_type=%s",
                    time.monotonic() - started,
                    type(exc).__name__,
                )
                raise _connection_error() from None
            LOGGER.error(
                "Calendar MCP lifecycle; status=startup_error elapsed=%.3fs "
                "error_type=%s",
                time.monotonic() - started,
                type(exc).__name__,
            )
            raise CalendarMcpError("Calendar MCP subprocess failed") from None

        LOGGER.info(
            "Calendar MCP lifecycle; status=ready elapsed=%.3fs",
            time.monotonic() - started,
        )
        try:
            yield client
        except BaseException:
            # Preserve exceptions from the caller's service body; they are not
            # Calendar MCP startup failures.  Teardown diagnostics stay muted.
            close_started = time.monotonic()
            try:
                await stack.aclose()
            except Exception:
                pass
            LOGGER.info(
                "Calendar MCP lifecycle; status=closed elapsed=%.3fs",
                time.monotonic() - close_started,
            )
            raise
        else:
            close_started = time.monotonic()
            try:
                await stack.aclose()
            except Exception as exc:
                if _is_connection_failure(exc):
                    LOGGER.error(
                        "Calendar MCP lifecycle; status=close_connection_error "
                        "elapsed=%.3fs error_type=%s",
                        time.monotonic() - close_started,
                        type(exc).__name__,
                    )
                    raise _connection_error() from None
                LOGGER.error(
                    "Calendar MCP lifecycle; status=close_error elapsed=%.3fs "
                    "error_type=%s",
                    time.monotonic() - close_started,
                    type(exc).__name__,
                )
                raise CalendarMcpError("Calendar MCP subprocess failed") from None
            LOGGER.info(
                "Calendar MCP lifecycle; status=closed elapsed=%.3fs",
                time.monotonic() - close_started,
            )


# A descriptive alias for callers that prefer the provider in the name.
open_google_calendar_mcp = open_calendar_mcp

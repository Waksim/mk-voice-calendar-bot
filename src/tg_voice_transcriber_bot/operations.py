"""Durable calendar actions, short conversation memory, and deterministic undo."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import asdict, dataclass, is_dataclass
from datetime import date, datetime, timedelta, timezone
import base64
import hashlib
import json
import os
from pathlib import Path
import secrets
from typing import Any, Literal
from zoneinfo import ZoneInfo

from .calendar import (
    CalendarClient,
    CalendarConnectionError,
    CalendarEventQueryResult,
    CalendarWriteRejectedError,
)
from .intent import validate_calendar_intent, validate_calendar_operation_plan


_MAX_TURNS = 2
_MAX_CONTEXT_EVENTS = 50
_MAX_CONTEXT_ACTIONS = 10
_NON_MATERIAL_SNAPSHOT_FIELDS = {
    "account",
    "html_link",
    "provider_updated_at",
}


class OperationStateError(RuntimeError):
    """The local operation journal is missing or inconsistent."""


class CalendarOperationError(RuntimeError):
    """Sanitized mutation failure safe to show without provider diagnostics."""

    def __init__(
        self,
        message: str,
        *,
        partially_applied: bool = False,
        retryable: bool = False,
        outcome_uncertain: bool = False,
    ) -> None:
        super().__init__(message)
        self.partially_applied = partially_applied
        self.retryable = retryable
        self.outcome_uncertain = outcome_uncertain


def _write_rejected_error(*, partially_applied: bool) -> CalendarOperationError:
    return CalendarOperationError(
        (
            "Google Calendar отклонил часть операции. Ранее подтверждённые "
            "изменения сохранены."
            if partially_applied
            else "Google Calendar отклонил операцию. Календарь не изменён."
        ),
        partially_applied=partially_applied,
        retryable=False,
        outcome_uncertain=False,
    )


@dataclass(frozen=True)
class OperationContext:
    application_state: dict[str, Any]
    recent_conversation: tuple[dict[str, Any], ...]
    history_steps: tuple[dict[str, Any], ...]
    allowed_event_ids: tuple[str, ...]


@dataclass(frozen=True)
class ActionExecutionResult:
    operation_id: str
    stage: str
    record: dict[str, Any]
    replayed: bool = False


@dataclass(frozen=True)
class UndoResult:
    handled: bool
    outcome: Literal[
        "undone",
        "already_undone",
        "blocked",
        "retryable_error",
        "rejected",
    ]
    record: dict[str, Any] | None = None
    event_titles: tuple[str, ...] = ()


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _iso_now() -> str:
    return _utc_now().isoformat()


def _opaque_id(seed: str | None = None) -> str:
    if seed is None:
        return secrets.token_urlsafe(12)
    digest = hashlib.sha256(seed.encode("utf-8")).digest()[:12]
    return base64.urlsafe_b64encode(digest).decode("ascii").rstrip("=")


def _source_key(source_update_id: int) -> str:
    if isinstance(source_update_id, bool) or not isinstance(source_update_id, int):
        raise ValueError("source_update_id must be an integer")
    if source_update_id < 0:
        raise ValueError("source_update_id must be non-negative")
    return f"telegram-update:{source_update_id}"


def _conversation_key(account: str, chat_id: int) -> str:
    return f"{account}:{chat_id}"


def _mapping(value: Any) -> dict[str, Any]:
    if is_dataclass(value) and not isinstance(value, type):
        return asdict(value)
    if isinstance(value, Mapping):
        return dict(value)
    raise OperationStateError("Calendar adapter returned an invalid snapshot")


def _snapshot(
    value: Any,
    *,
    account: str,
    fallback: Mapping[str, Any] | None = None,
    event_id: str | None = None,
    html_link: str | None = None,
    timezone_name: str | None = None,
) -> dict[str, Any]:
    raw = _mapping(value) if value is not None else {}
    fallback = dict(fallback or {})
    snapshot_timezone = (
        raw.get("timezone")
        if value is not None
        else fallback.get("timezone") or "Europe/Moscow"
    )
    raw_reminder_overrides = (
        raw["reminder_overrides"]
        if "reminder_overrides" in raw
        else fallback.get("reminder_overrides", ())
    )
    recurrence = raw.get("recurrence_rrule")
    recurrence_list: list[str] = []
    if recurrence is None:
        recurrence_values = raw.get("recurrence_rrules")
        if (
            isinstance(recurrence_values, Sequence)
            and not isinstance(recurrence_values, (str, bytes))
            and recurrence_values
        ):
            recurrence_list = [str(value) for value in recurrence_values]
            recurrence = recurrence_list[0]
    else:
        recurrence_list = [str(recurrence)]
    if not recurrence_list:
        fallback_recurrences = fallback.get("recurrence_rrules")
        if isinstance(fallback_recurrences, Sequence) and not isinstance(
            fallback_recurrences, (str, bytes)
        ):
            recurrence_list = [str(value) for value in fallback_recurrences]
        elif fallback.get("recurrence_rrule"):
            recurrence_list = [str(fallback["recurrence_rrule"])]
    normalized = {
        "account": str(raw.get("account") or account),
        "calendar_id": str(raw.get("calendar_id") or fallback.get("calendar_id") or "primary"),
        "event_id": str(raw.get("event_id") or event_id or fallback.get("event_id") or ""),
        "title": raw.get("title", fallback.get("title")),
        "start_at": raw.get("start_at", fallback.get("start_at")),
        "end_at": raw.get("end_at", fallback.get("end_at")),
        "all_day": raw.get("all_day", fallback.get("all_day")),
        # An absent named zone is material provider state: do not invent
        # Europe/Moscow for an offset-only event.  The default is reserved for
        # snapshots synthesized solely from a bot-authored fallback payload.
        "timezone": snapshot_timezone,
        "location": raw.get("location", fallback.get("location")),
        "description": raw.get("description", fallback.get("description")),
        "recurrence_rrule": (
            recurrence
            if recurrence is not None
            else fallback.get("recurrence_rrule")
        ),
        "recurrence_rrules": recurrence_list,
        "status": str(raw.get("status") or fallback.get("status") or "confirmed"),
        "html_link": raw.get("html_link") or html_link or fallback.get("html_link"),
        "provider_updated_at": raw.get("updated_at") or fallback.get("provider_updated_at"),
        "recurring_event_id": raw.get("recurring_event_id")
        or fallback.get("recurring_event_id"),
        "original_start_at": raw.get("original_start_at")
        or fallback.get("original_start_at"),
        "attendee_emails": list(
            raw.get("attendee_emails") or fallback.get("attendee_emails") or ()
        ),
        "organizer_self": raw.get(
            "organizer_self",
            raw.get(
                "organizer_is_self",
                fallback.get("organizer_self", fallback.get("organizer_is_self")),
            ),
        ),
        "organizer_email": raw.get(
            "organizer_email", fallback.get("organizer_email")
        ),
        "creator_email": raw.get("creator_email", fallback.get("creator_email")),
        "creator_self": raw.get(
            "creator_self",
            raw.get(
                "creator_is_self",
                fallback.get("creator_self", fallback.get("creator_is_self")),
            ),
        ),
        "event_type": raw.get("event_type") or fallback.get("event_type"),
        "color_id": raw.get("color_id", fallback.get("color_id")),
        "transparency": raw.get("transparency", fallback.get("transparency")),
        "visibility": raw.get("visibility", fallback.get("visibility")),
        "reminders_present": raw.get(
            "reminders_present", fallback.get("reminders_present", False)
        ),
        "reminders_use_default": raw.get(
            "reminders_use_default", fallback.get("reminders_use_default")
        ),
        "reminder_overrides": [
            list(item)
            for item in (raw_reminder_overrides or ())
        ],
        "has_conference_data": raw.get(
            "has_conference_data", fallback.get("has_conference_data", False)
        ),
        "has_hangout_link": raw.get(
            "has_hangout_link", fallback.get("has_hangout_link", False)
        ),
        "has_attachments": raw.get(
            "has_attachments", fallback.get("has_attachments", False)
        ),
        "has_extended_properties": raw.get(
            "has_extended_properties",
            fallback.get("has_extended_properties", False),
        ),
        "has_source": raw.get("has_source", fallback.get("has_source", False)),
        "anyone_can_add_self": raw.get(
            "anyone_can_add_self", fallback.get("anyone_can_add_self")
        ),
        "guests_can_invite_others": raw.get(
            "guests_can_invite_others",
            fallback.get("guests_can_invite_others"),
        ),
        "guests_can_modify": raw.get(
            "guests_can_modify", fallback.get("guests_can_modify")
        ),
        "guests_can_see_other_guests": raw.get(
            "guests_can_see_other_guests",
            fallback.get("guests_can_see_other_guests"),
        ),
        "private_copy": raw.get("private_copy", fallback.get("private_copy")),
        "locked": raw.get("locked", fallback.get("locked")),
        "safety_metadata_complete": raw.get(
            "safety_metadata_complete",
            fallback.get("safety_metadata_complete", False),
        ),
        "safety_metadata_fingerprint": raw.get(
            "safety_metadata_fingerprint",
            fallback.get("safety_metadata_fingerprint"),
        ),
    }
    if not normalized["event_id"]:
        raise OperationStateError("Calendar snapshot has no event ID")
    if normalized["title"] is None:
        normalized["title"] = "Без названия"
    if timezone_name is not None and normalized["timezone"] is not None:
        normalized["timezone"] = timezone_name
        if normalized["all_day"] is False:
            zone = ZoneInfo(timezone_name)
            for field in ("start_at", "end_at"):
                value = normalized.get(field)
                if not isinstance(value, str):
                    continue
                parsed = _parse_temporal(value, all_day=False)
                assert isinstance(parsed, datetime)
                normalized[field] = parsed.astimezone(zone).isoformat()
    return normalized


def _event_payload(snapshot: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "title": snapshot.get("title") or "Без названия",
        "start_at": snapshot.get("start_at"),
        "end_at": snapshot.get("end_at"),
        "all_day": bool(snapshot.get("all_day")),
        "timezone": snapshot.get("timezone"),
        "location": snapshot.get("location"),
        "description": snapshot.get("description"),
        "recurrence_rrule": snapshot.get("recurrence_rrule"),
    }


def _delete_undo_is_core_only(snapshot: Mapping[str, Any]) -> bool:
    """Whether recreating a deleted event cannot restore all known metadata."""

    recurrence_rules = snapshot.get("recurrence_rrules") or ()
    return bool(
        snapshot.get("status") == "tentative"
        or snapshot.get("attendee_emails")
        or snapshot.get("creator_self") is not True
        or snapshot.get("organizer_self") is not True
        or snapshot.get("event_type") not in (None, "default")
        or (
            snapshot.get("reminders_present")
            and (
                snapshot.get("reminders_use_default") is not True
                or snapshot.get("reminder_overrides")
            )
        )
        or snapshot.get("color_id") is not None
        or snapshot.get("transparency") not in (None, "opaque")
        or snapshot.get("visibility") not in (None, "default")
        or snapshot.get("has_conference_data")
        or snapshot.get("has_hangout_link")
        or snapshot.get("has_attachments")
        or snapshot.get("has_extended_properties")
        or snapshot.get("has_source")
        or snapshot.get("anyone_can_add_self") not in (None, False)
        or snapshot.get("guests_can_invite_others") not in (None, True)
        or snapshot.get("guests_can_modify") not in (None, False)
        or snapshot.get("guests_can_see_other_guests") not in (None, True)
        or snapshot.get("private_copy") is True
        or snapshot.get("locked") is True
        or snapshot.get("recurring_event_id")
        or len(recurrence_rules) > 1
        or (
            snapshot.get("all_day") is False
            and not isinstance(snapshot.get("timezone"), str)
        )
        or snapshot.get("safety_metadata_complete") is not True
    )


def _material_event_snapshot(snapshot: Mapping[str, Any]) -> dict[str, Any]:
    """Return the provider state whose changes must invalidate a local undo.

    Logical account names, links, and provider observation timestamps can vary
    while the calendar event itself stays identical. Everything else in our
    normalized snapshot is mutable event data or mutation-safety metadata and
    therefore participates in the comparison.
    """

    material = {
        key: deepcopy(value)
        for key, value in snapshot.items()
        if key not in _NON_MATERIAL_SNAPSHOT_FIELDS
    }
    attendee_emails = material.get("attendee_emails")
    if isinstance(attendee_emails, Sequence) and not isinstance(
        attendee_emails, (str, bytes)
    ):
        material["attendee_emails"] = sorted(str(value) for value in attendee_emails)
    return material


def _materially_equivalent(
    left: Mapping[str, Any], right: Mapping[str, Any]
) -> bool:
    return _material_event_snapshot(left) == _material_event_snapshot(right)


def _parse_temporal(value: str, *, all_day: bool) -> date | datetime:
    if all_day:
        return date.fromisoformat(value)
    normalized = value[:-1] + "+00:00" if value.endswith("Z") else value
    parsed = datetime.fromisoformat(normalized)
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("timed event is missing an offset")
    return parsed


def _format_temporal(value: date | datetime, *, all_day: bool) -> str:
    if all_day:
        assert isinstance(value, date) and not isinstance(value, datetime)
        return value.isoformat()
    assert isinstance(value, datetime)
    return value.isoformat()


class OperationStore:
    """Atomic JSON journal and two-turn memory shared by the calendar flow."""

    def __init__(
        self,
        path: Path,
        legacy_confirmation_path: Path | None = None,
        calendar_scope_by_account: Mapping[str, str] | None = None,
        default_calendar_scope: str = "owner:primary",
    ) -> None:
        self.path = Path(path)
        self.legacy_confirmation_path = (
            Path(legacy_confirmation_path)
            if legacy_confirmation_path is not None
            else None
        )
        self._scope_by_account = dict(calendar_scope_by_account or {})
        self._default_scope = default_calendar_scope
        self._data = self._load()
        if not self.path.exists() and self._import_legacy():
            self._save()

    @staticmethod
    def _default() -> dict[str, Any]:
        return {
            "version": 2,
            "operations": {},
            "source_index": {},
            "conversations": {},
            "events": {},
        }

    def _load(self) -> dict[str, Any]:
        if not self.path.exists():
            return self._default()
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise OperationStateError(
                f"Cannot read operation state: {type(exc).__name__}"
            ) from exc
        if not isinstance(raw, dict) or raw.get("version") != 2:
            raise OperationStateError("Unsupported operation state")
        expected = {"version", "operations", "source_index", "conversations", "events"}
        if set(raw) != expected or any(
            not isinstance(raw[key], dict)
            for key in ("operations", "source_index", "conversations", "events")
        ):
            raise OperationStateError("Invalid operation state root")
        return raw

    def _scope(self, account: str) -> str:
        return self._scope_by_account.get(account, self._default_scope)

    def _import_legacy(self) -> bool:
        path = self.legacy_confirmation_path
        if path is None or not path.exists():
            return False
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return False
        records = raw.get("records") if isinstance(raw, dict) else None
        if not isinstance(records, dict):
            return False
        imported = False
        ordered = sorted(
            (record for record in records.values() if isinstance(record, dict)),
            key=lambda record: str(record.get("created_at", "")),
        )
        for legacy in ordered:
            if legacy.get("stage") != "created":
                continue
            intent = legacy.get("intent")
            references = legacy.get("calendar_events")
            events = intent.get("events") if isinstance(intent, dict) else None
            if not isinstance(events, list) or not isinstance(references, list):
                continue
            if len(events) != len(references) or not events:
                continue
            source = str(legacy.get("source_key") or "")
            account = str(legacy.get("account") or "personal")
            op_id = _opaque_id(f"legacy:{legacy.get('confirmation_id')}:{source}")
            if op_id in self._data["operations"]:
                continue
            items: list[dict[str, Any]] = []
            for index, (event, reference) in enumerate(zip(events, references, strict=True)):
                if not isinstance(event, dict) or not isinstance(reference, dict):
                    items = []
                    break
                event_id = reference.get("event_id")
                if not isinstance(event_id, str) or not event_id:
                    items = []
                    break
                after = _snapshot(
                    None,
                    account=account,
                    fallback={**event, "status": "confirmed"},
                    event_id=event_id,
                    html_link=reference.get("html_link"),
                )
                items.append(
                    {
                        "index": index,
                        "type": "create",
                        "stage": "applied",
                        "target_event_id": None,
                        "request": {"event": deepcopy(event), "patch": None, "clear_fields": []},
                        "before": None,
                        "after": after,
                        "undo_stage": "pending",
                    }
                )
                self._put_event(account, after, active=True, operation_id=op_id)
            if not items:
                continue
            record = {
                "operation_id": op_id,
                "source_key": source or f"legacy:{op_id}",
                "conversation_key": _conversation_key(
                    account, int(legacy.get("chat_id") or 0)
                ),
                "account": account,
                "owner_user_id": int(legacy.get("owner_user_id") or 0),
                "chat_id": int(legacy.get("chat_id") or 0),
                "transcript": None,
                "reference_time": legacy.get("created_at"),
                "stage": "applied",
                "confidence": intent.get("confidence"),
                "items": items,
                "created_at": legacy.get("created_at") or _iso_now(),
                "updated_at": legacy.get("updated_at") or _iso_now(),
                "undo": {"stage": "available"},
                "interaction_input": None,
                "interaction_steps": [],
                "assistant_text": "Импортировано ранее выполненное создание событий.",
                "legacy": True,
            }
            self._data["operations"][op_id] = record
            self._data["source_index"][record["source_key"]] = op_id
            imported = True
        return imported

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_name(f".{self.path.name}.{os.getpid()}.tmp")
        serialized = json.dumps(
            self._data, ensure_ascii=False, sort_keys=True, indent=2
        ) + "\n"
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                descriptor = -1
                handle.write(serialized)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, self.path)
            os.chmod(self.path, 0o600)
        finally:
            if descriptor >= 0:
                os.close(descriptor)
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass

    def get(self, operation_id: str) -> dict[str, Any] | None:
        record = self._data["operations"].get(operation_id)
        return deepcopy(record) if isinstance(record, dict) else None

    def find_by_source(self, source_key: str) -> dict[str, Any] | None:
        operation_id = self._data["source_index"].get(source_key)
        return self.get(operation_id) if isinstance(operation_id, str) else None

    def put(self, record: Mapping[str, Any]) -> None:
        operation_id = str(record["operation_id"])
        source_key = str(record["source_key"])
        self._data["operations"][operation_id] = deepcopy(dict(record))
        self._data["source_index"][source_key] = operation_id
        self._save()

    def _put_event(
        self,
        account: str,
        snapshot: Mapping[str, Any],
        *,
        active: bool,
        operation_id: str,
    ) -> None:
        scope = self._scope(account)
        events = self._data["events"].setdefault(scope, {})
        event_id = str(snapshot["event_id"])
        events[event_id] = {
            "snapshot": deepcopy(dict(snapshot)),
            "active": active,
            "last_operation_id": operation_id,
            "updated_at": _iso_now(),
        }

    def put_event(
        self,
        account: str,
        snapshot: Mapping[str, Any],
        *,
        active: bool,
        operation_id: str,
    ) -> None:
        self._put_event(account, snapshot, active=active, operation_id=operation_id)
        self._save()

    def observe_events(
        self, account: str, snapshots: Sequence[Any]
    ) -> tuple[dict[str, Any], ...]:
        """Cache trusted provider observations without weakening undo freshness."""

        scope = self._scope(account)
        events = self._data["events"].setdefault(scope, {})
        observed: list[dict[str, Any]] = []
        seen: set[str] = set()
        for raw_snapshot in snapshots:
            snapshot = _snapshot(
                raw_snapshot,
                account=account,
                timezone_name=None,
            )
            event_id = str(snapshot["event_id"])
            if event_id in seen:
                raise OperationStateError("Calendar lookup returned duplicate event IDs")
            seen.add(event_id)
            previous = events.get(event_id)
            previous = previous if isinstance(previous, dict) else {}
            previous_snapshot = previous.get("snapshot")
            preserve_operation_id = (
                previous.get("last_operation_id")
                if isinstance(previous_snapshot, Mapping)
                and _materially_equivalent(previous_snapshot, snapshot)
                else None
            )
            events[event_id] = {
                "snapshot": deepcopy(snapshot),
                "active": snapshot.get("status") in {"confirmed", "tentative"},
                # Preserve undo ownership only when the provider confirms that
                # the event is still materially identical to our last write.
                "last_operation_id": preserve_operation_id,
                "origin": previous.get("origin") or "lookup",
                "updated_at": _iso_now(),
            }
            observed.append(snapshot)
        self._save()
        return tuple(deepcopy(observed))

    def event_entry(self, account: str, event_id: str) -> dict[str, Any] | None:
        entry = self._data["events"].get(self._scope(account), {}).get(event_id)
        return deepcopy(entry) if isinstance(entry, dict) else None

    def append_turn(self, record: Mapping[str, Any]) -> None:
        conversation_key = str(record["conversation_key"])
        conversation = self._data["conversations"].setdefault(
            conversation_key, {"turns": []}
        )
        turns = conversation.setdefault("turns", [])
        source_key = str(record["source_key"])
        turn = {
            "source_key": source_key,
            "operation_id": record["operation_id"],
            "user_message": record.get("transcript"),
            "assistant_message": record.get("assistant_text"),
            "stage": record.get("stage"),
            "actions": [
                {
                    "type": item.get("type"),
                    "event_id": (
                        (item.get("after") or item.get("before") or {}).get("event_id")
                        if isinstance(item, dict)
                        else None
                    ),
                    "before": item.get("before") if isinstance(item, dict) else None,
                    "after": item.get("after") if isinstance(item, dict) else None,
                }
                for item in record.get("items", [])
            ],
            "displayed_candidates": (
                deepcopy(record.get("displayed_candidates"))
                if isinstance(record.get("displayed_candidates"), list)
                else None
            ),
            "interaction_input": record.get("interaction_input"),
            "interaction_steps": record.get("interaction_steps", []),
        }
        turns[:] = [item for item in turns if item.get("source_key") != source_key]
        turns.append(turn)
        del turns[:-_MAX_TURNS]
        self._save()

    def context(
        self,
        account: str,
        chat_id: int,
        now: datetime,
        timezone_name: str,
    ) -> OperationContext:
        scope_events = self._data["events"].get(self._scope(account), {})
        turns = deepcopy(
            self._data["conversations"]
            .get(_conversation_key(account, chat_id), {})
            .get("turns", [])[-_MAX_TURNS:]
        )
        active_entries = sorted(
            (
                entry
                for entry in scope_events.values()
                if isinstance(entry, dict) and entry.get("active") is True
            ),
            key=lambda entry: str(entry.get("updated_at", "")),
            reverse=True,
        )
        latest_displayed = (
            turns[-1].get("displayed_candidates")
            if turns and isinstance(turns[-1], dict)
            else None
        )
        if isinstance(latest_displayed, list):
            # Relative references such as "второй" must resolve against the
            # exact list and order the user just saw, never against a recency
            # sort of the broader calendar cache.
            candidate_events = []
            seen_ids: set[str] = set()
            seen_display_indexes: set[int] = set()
            for displayed in latest_displayed[:_MAX_CONTEXT_EVENTS]:
                if not isinstance(displayed, Mapping):
                    continue
                event_id = str(displayed.get("event_id") or "")
                if not event_id or event_id in seen_ids:
                    continue
                entry = scope_events.get(event_id)
                if not isinstance(entry, dict) or entry.get("active") is not True:
                    continue
                snapshot = entry.get("snapshot")
                if not isinstance(snapshot, dict):
                    continue
                seen_ids.add(event_id)
                candidate = deepcopy(snapshot)
                stored_index = displayed.get("display_index")
                if (
                    isinstance(stored_index, int)
                    and not isinstance(stored_index, bool)
                    and stored_index > 0
                ):
                    display_index = stored_index
                else:
                    display_index = 1
                    while display_index in seen_display_indexes:
                        display_index += 1
                if display_index in seen_display_indexes:
                    continue
                seen_display_indexes.add(display_index)
                candidate["display_index"] = display_index
                candidate_events.append(candidate)
        else:
            candidate_events = [
                deepcopy(entry["snapshot"])
                for entry in active_entries[:_MAX_CONTEXT_EVENTS]
            ]
        allowed_ids = tuple(str(event["event_id"]) for event in candidate_events)

        operations = sorted(
            (
                record
                for record in self._data["operations"].values()
                if isinstance(record, dict)
                and record.get("conversation_key")
                == _conversation_key(account, chat_id)
                and record.get("stage")
                in {"applied", "undone", "read", "clarify", "ignored"}
            ),
            key=lambda record: str(record.get("updated_at", "")),
            reverse=True,
        )[:_MAX_CONTEXT_ACTIONS]
        recent_actions = [
            {
                "operation_id": record.get("operation_id"),
                "status": record.get("stage"),
                "items": [
                    {
                        "type": item.get("type"),
                        "event_id": (
                            (item.get("after") or item.get("before") or {}).get("event_id")
                            if isinstance(item, dict)
                            else None
                        ),
                        "before": item.get("before"),
                        "after": item.get("after"),
                    }
                    for item in record.get("items", [])
                    if isinstance(item, dict)
                ],
            }
            for record in operations
        ]
        recent_conversation = tuple(
            {
                "user_message": turn.get("user_message"),
                "assistant_message": turn.get("assistant_message"),
                "status": turn.get("stage"),
                "actions": turn.get("actions", []),
            }
            for turn in turns
        )
        history_steps: list[dict[str, Any]] = []
        for turn in turns:
            interaction_input = turn.get("interaction_input")
            interaction_steps = turn.get("interaction_steps")
            if isinstance(interaction_input, dict) and isinstance(interaction_steps, list):
                history_steps.append(deepcopy(interaction_input))
                history_steps.extend(
                    deepcopy(step) for step in interaction_steps if isinstance(step, dict)
                )
        application_state = {
            "reference_time": now.isoformat(),
            "timezone": timezone_name,
            "calendar_profile": account,
            "allowed_event_ids": list(allowed_ids),
            "candidate_events": candidate_events,
            "recent_actions": recent_actions,
            "lookup_permitted": True,
        }
        return OperationContext(
            application_state=application_state,
            recent_conversation=recent_conversation,
            history_steps=tuple(history_steps),
            allowed_event_ids=allowed_ids,
        )


class CalendarOperationPipeline:
    """Apply model plans immediately and persist deterministic undo state.

    Undo of a rich deleted provider event is intentionally best-effort: the
    Calendar create boundary can restore the core event fields but not every
    provider-specific metadata field.
    """

    def __init__(
        self,
        store: OperationStore,
        calendar: CalendarClient,
        *,
        timezone_name: str = "Europe/Moscow",
    ) -> None:
        self.store = store
        self.calendar = calendar
        self.timezone_name = timezone_name
        self._lock = asyncio.Lock()

    def context(
        self, *, account: str, chat_id: int, now: datetime
    ) -> OperationContext:
        return self.store.context(account, chat_id, now, self.timezone_name)

    def mark_undo_notified(self, operation_id: str) -> None:
        """Persist that the separate Telegram undo result reached the owner."""
        record = self.store.get(operation_id)
        if record is None or record.get("stage") != "undone":
            return
        undo = dict(record.get("undo") or {})
        undo["chat_notified"] = True
        undo["updated_at"] = _iso_now()
        record["undo"] = undo
        self.store.put(record)

    async def query_events(
        self,
        *,
        account: str,
        query: str | None,
        time_min: str,
        time_max: str,
        limit: int = 20,
    ) -> CalendarEventQueryResult:
        """Run a bounded provider read and hide provider diagnostics from callers."""

        try:
            if query is None:
                result = await self.calendar.list_events(
                    account=account,
                    time_min=time_min,
                    time_max=time_max,
                    limit=limit,
                )
            else:
                result = await self.calendar.search_events(
                    account=account,
                    query=query,
                    time_min=time_min,
                    time_max=time_max,
                    limit=limit,
                )
        except CalendarConnectionError:
            raise
        except Exception:
            raise CalendarOperationError(
                "Не удалось прочитать события из Google Calendar. Попробуйте ещё раз."
            ) from None
        if not isinstance(result, CalendarEventQueryResult):
            raise CalendarOperationError(
                "Google Calendar вернул некорректный результат. Попробуйте ещё раз."
            )
        return result

    async def apply_plan(
        self,
        *,
        source_update_id: int,
        account: str,
        owner_user_id: int,
        chat_id: int,
        transcript: str,
        reference_time: datetime,
        plan: Mapping[str, Any],
        interaction_input: Mapping[str, Any] | None = None,
        interaction_steps: Sequence[Mapping[str, Any]] | None = None,
        assistant_text: str | None = None,
        allowed_event_ids: Sequence[str] | None = None,
        displayed_candidates: Sequence[Any] | None = None,
    ) -> ActionExecutionResult:
        source_key = _source_key(source_update_id)
        async with self._lock:
            existing = self.store.find_by_source(source_key)
            if existing is not None:
                self._verify_source(existing, account, owner_user_id, chat_id)
                if existing.get("stage") in {"applied", "read", "clarify", "ignored"}:
                    return ActionExecutionResult(
                        str(existing["operation_id"]),
                        str(existing["stage"]),
                        existing,
                        True,
                    )
                if existing.get("stage") in {"rejected", "partially_rejected"}:
                    raise _write_rejected_error(
                        partially_applied=(
                            existing.get("stage") == "partially_rejected"
                        )
                    )

            context = self.context(account=account, chat_id=chat_id, now=reference_time)
            trusted_event_ids = (
                context.allowed_event_ids
                if allowed_event_ids is None
                else tuple(allowed_event_ids)
            )
            if existing is not None:
                # These targets were already allowlisted and validated before
                # the first provider write.  A lost delete response may make a
                # later observation mark the target inactive; that must not
                # prevent replaying the existing journal item with its stable
                # idempotency key.
                journal_target_ids = tuple(
                    str(item.get("target_event_id"))
                    for item in existing.get("items", [])
                    if isinstance(item, Mapping)
                    and item.get("type") in {"update", "delete"}
                    and item.get("target_event_id")
                )
                trusted_event_ids = tuple(
                    dict.fromkeys((*trusted_event_ids, *journal_target_ids))
                )
            normalized = validate_calendar_operation_plan(
                {
                    key: deepcopy(value)
                    for key, value in plan.items()
                    if not str(key).startswith("_")
                },
                trusted_event_ids,
                expected_timezone=self.timezone_name,
            )
            normalized_displayed = self._normalize_displayed_candidates(
                account, displayed_candidates
            )
            if normalized_displayed is not None and any(
                candidate["event_id"] not in trusted_event_ids
                for candidate in normalized_displayed
            ):
                raise OperationStateError(
                    "Displayed candidates contain an untrusted event ID"
                )
            if normalized["action"] in {"read", "lookup"}:
                raise OperationStateError(
                    "Read and discovery plans must be handled before mutation"
                )
            preflight_snapshots: dict[str, dict[str, Any]] = {}
            if normalized["action"] == "execute" and existing is None:
                preflight_snapshots = await self._preflight_mutations(
                    account, normalized["operations"]
                )

            if existing is None:
                record = self._new_record(
                    source_key=source_key,
                    account=account,
                    owner_user_id=owner_user_id,
                    chat_id=chat_id,
                    transcript=transcript,
                    reference_time=reference_time,
                    plan=normalized,
                    interaction_input=interaction_input,
                    interaction_steps=interaction_steps,
                    assistant_text=assistant_text,
                    displayed_candidates=(
                        normalized_displayed
                        if normalized["action"] in {"execute", "clarify"}
                        else None
                    ),
                    before_snapshots=preflight_snapshots,
                )
                self.store.put(record)
            else:
                record = existing

            if normalized["action"] != "execute":
                record["stage"] = "clarify" if normalized["action"] == "clarify" else "ignored"
                record["clarification_question"] = normalized["clarification_question"]
                record["assistant_text"] = assistant_text or normalized["clarification_question"]
                record["updated_at"] = _iso_now()
                self.store.put(record)
                self.store.append_turn(record)
                return ActionExecutionResult(
                    str(record["operation_id"]), str(record["stage"]), deepcopy(record)
                )

            record["stage"] = "applying"
            record["updated_at"] = _iso_now()
            self.store.put(record)
            applied = sum(item.get("stage") == "applied" for item in record["items"])
            fresh_preflight_ids = set(preflight_snapshots)
            try:
                for item in record["items"]:
                    if item.get("stage") == "applied":
                        continue
                    target_event_id = str(item.get("target_event_id"))
                    await self._apply_item(
                        record,
                        item,
                        before_is_fresh=target_event_id in fresh_preflight_ids,
                    )
                    fresh_preflight_ids.discard(target_event_id)
                    applied += 1
                    record["updated_at"] = _iso_now()
                    self.store.put(record)
            except CalendarConnectionError:
                record["stage"] = "applying"
                record["updated_at"] = _iso_now()
                self.store.put(record)
                raise
            except CalendarWriteRejectedError as exc:
                item["stage"] = "failed"
                item["last_error_class"] = type(exc).__name__
                record["stage"] = (
                    "partially_rejected" if applied else "rejected"
                )
                record["updated_at"] = _iso_now()
                self.store.put(record)
                raise _write_rejected_error(partially_applied=bool(applied)) from None
            except Exception as exc:
                write_started = bool(item.get("provider_write_started_at"))
                item["stage"] = "applying" if write_started else "failed"
                item["last_error_class"] = type(exc).__name__
                record["stage"] = (
                    "partially_applied"
                    if applied
                    else "applying"
                    if write_started
                    else "failed"
                )
                record["updated_at"] = _iso_now()
                self.store.put(record)
                raise CalendarOperationError(
                    "Не удалось применить операцию к Google Calendar.",
                    partially_applied=bool(applied),
                    # Confirmed earlier items make terminalizing this source
                    # unsafe even when the current item failed before its own
                    # write marker. Replay skips those applied items and can
                    # safely retry the remainder with the same journal keys.
                    retryable=write_started or bool(applied),
                    outcome_uncertain=write_started,
                ) from None

            record["stage"] = "applied"
            undo_fidelity = (
                "core_only"
                if any(
                    item.get("undo_fidelity") == "core_only"
                    or (
                        item.get("type") == "delete"
                        and isinstance(item.get("before"), Mapping)
                        and _delete_undo_is_core_only(item["before"])
                    )
                    for item in record["items"]
                )
                else "full"
            )
            record["undo"] = {
                "stage": "available",
                "fidelity": undo_fidelity,
                "updated_at": _iso_now(),
            }
            record["displayed_candidates"] = self._successful_displayed_candidates(
                record
            )
            record["assistant_text"] = assistant_text or self._result_text(record)
            record["updated_at"] = _iso_now()
            self.store.put(record)
            self.store.append_turn(record)
            return ActionExecutionResult(
                str(record["operation_id"]), "applied", deepcopy(record)
            )

    @staticmethod
    def _successful_displayed_candidates(
        record: Mapping[str, Any],
    ) -> list[dict[str, Any]]:
        """Persist the exact active rows shown by a successful batch card.

        Deletes are visible in a mixed result card but cannot be targeted on
        the next turn.  Keeping the original one-based item index on the
        remaining rows deliberately leaves a hole, so “the deleted second
        one” cannot silently resolve to the third row.
        """

        candidates: list[dict[str, Any]] = []
        for position, item in enumerate(record.get("items", [])):
            if not isinstance(item, Mapping) or item.get("type") not in {
                "create",
                "update",
            }:
                continue
            after = item.get("after")
            if not isinstance(after, Mapping) or after.get("status") not in {
                "confirmed",
                "tentative",
            }:
                continue
            item_index = item.get("index")
            display_index = (
                item_index + 1
                if isinstance(item_index, int)
                and not isinstance(item_index, bool)
                and item_index >= 0
                else position + 1
            )
            candidate = deepcopy(dict(after))
            candidate["display_index"] = display_index
            candidates.append(candidate)
        return candidates

    async def record_read(
        self,
        *,
        source_update_id: int,
        account: str,
        owner_user_id: int,
        chat_id: int,
        transcript: str,
        reference_time: datetime,
        lookup: Mapping[str, Any],
        events: Sequence[Any],
        total_count: int,
        may_be_incomplete: bool,
        interaction_input: Mapping[str, Any] | None = None,
        interaction_steps: Sequence[Mapping[str, Any]] | None = None,
        assistant_text: str | None = None,
        displayed_candidates: Sequence[Any] | None = None,
    ) -> ActionExecutionResult:
        """Persist a read result and conversation turn without creating undo state."""

        source_key = _source_key(source_update_id)
        async with self._lock:
            existing = self.store.find_by_source(source_key)
            if existing is not None:
                self._verify_source(existing, account, owner_user_id, chat_id)
                if existing.get("stage") == "read":
                    return ActionExecutionResult(
                        str(existing["operation_id"]), "read", existing, True
                    )
                raise OperationStateError("Source update already has another operation")

            observed = self.store.observe_events(account, events)
            visible = self._normalize_displayed_candidates(
                account,
                observed if displayed_candidates is None else displayed_candidates,
            )
            observed_ids = {event["event_id"] for event in observed}
            if visible is not None and any(
                candidate["event_id"] not in observed_ids for candidate in visible
            ):
                raise OperationStateError(
                    "Displayed read candidates were not returned by the provider"
                )
            operation_id = _opaque_id()
            now = _iso_now()
            record = {
                "operation_id": operation_id,
                "source_key": source_key,
                "conversation_key": _conversation_key(account, chat_id),
                "account": account,
                "owner_user_id": owner_user_id,
                "chat_id": chat_id,
                "transcript": transcript,
                "reference_time": reference_time.isoformat(),
                "stage": "read",
                "confidence": None,
                "clarification_question": None,
                "lookup": deepcopy(dict(lookup)),
                "items": [
                    {
                        "index": index,
                        "type": "read",
                        "stage": "observed",
                        "target_event_id": event["event_id"],
                        "request": {"lookup": deepcopy(dict(lookup))},
                        "before": None,
                        "after": deepcopy(dict(event)),
                        "undo_stage": "unavailable",
                    }
                    for index, event in enumerate(observed)
                ],
                "total_count": int(total_count),
                "may_be_incomplete": bool(may_be_incomplete),
                "created_at": now,
                "updated_at": now,
                "undo": {"stage": "unavailable"},
                "displayed_candidates": list(visible or ()),
                "interaction_input": (
                    deepcopy(dict(interaction_input)) if interaction_input else None
                ),
                "interaction_steps": [
                    deepcopy(dict(step)) for step in (interaction_steps or ())
                ],
                "assistant_text": assistant_text
                or f"Показано событий: {len(observed)}.",
                "legacy": False,
            }
            self.store.put(record)
            self.store.append_turn(record)
            return ActionExecutionResult(operation_id, "read", deepcopy(record))

    def observe_lookup_events(
        self, account: str, events: Sequence[Any]
    ) -> tuple[dict[str, Any], ...]:
        return self.store.observe_events(account, events)

    def _normalize_displayed_candidates(
        self, account: str, candidates: Sequence[Any] | None
    ) -> tuple[dict[str, Any], ...] | None:
        if candidates is None:
            return None
        normalized: list[dict[str, Any]] = []
        seen_ids: set[str] = set()
        for candidate in candidates[:_MAX_CONTEXT_EVENTS]:
            snapshot = _snapshot(
                candidate,
                account=account,
                timezone_name=self.timezone_name,
            )
            event_id = str(snapshot["event_id"])
            if event_id in seen_ids:
                raise OperationStateError("Displayed candidates contain duplicate IDs")
            seen_ids.add(event_id)
            normalized.append(snapshot)
        return tuple(normalized)

    async def _preflight_mutations(
        self, account: str, operations: Sequence[Mapping[str, Any]]
    ) -> dict[str, dict[str, Any]]:
        snapshots: dict[str, dict[str, Any]] = {}
        for operation in operations:
            if operation.get("type") not in {"update", "delete"}:
                continue
            event_id = str(operation.get("target_event_id") or "")
            if event_id in snapshots:
                continue
            entry = self.store.event_entry(account, event_id)
            if entry is None or entry.get("active") is not True:
                raise OperationStateError("Mutation target is not active")
            if not isinstance(entry.get("snapshot"), Mapping):
                raise OperationStateError("Mutation target snapshot is invalid")
            try:
                provider = await self.calendar.get_event(
                    account=account, event_id=event_id
                )
            except CalendarConnectionError:
                raise
            except Exception:
                raise CalendarOperationError(
                    "Не удалось проверить актуальное состояние события. "
                    "Попробуйте ещё раз."
                ) from None
            snapshot = _snapshot(
                provider,
                account=account,
                timezone_name=None,
            )
            if snapshot["event_id"] != event_id:
                raise CalendarOperationError(
                    "Google Calendar вернул другое событие. Попробуйте ещё раз."
                )
            snapshots[event_id] = snapshot
        if snapshots:
            self.store.observe_events(account, tuple(snapshots.values()))
        return snapshots

    def _new_record(
        self,
        *,
        source_key: str,
        account: str,
        owner_user_id: int,
        chat_id: int,
        transcript: str,
        reference_time: datetime,
        plan: Mapping[str, Any],
        interaction_input: Mapping[str, Any] | None,
        interaction_steps: Sequence[Mapping[str, Any]] | None,
        assistant_text: str | None,
        displayed_candidates: Sequence[Mapping[str, Any]] | None,
        before_snapshots: Mapping[str, Mapping[str, Any]],
    ) -> dict[str, Any]:
        if not account or owner_user_id <= 0 or chat_id <= 0:
            raise ValueError("invalid operation ownership")
        operation_id = _opaque_id()
        items = [
            {
                "index": index,
                "type": operation["type"],
                "stage": "planned",
                "target_event_id": operation["target_event_id"],
                "request": {
                    "event": deepcopy(operation["event"]),
                    "patch": deepcopy(operation["patch"]),
                    "clear_fields": deepcopy(operation["clear_fields"]),
                },
                "before": (
                    deepcopy(dict(before_snapshots[operation["target_event_id"]]))
                    if operation["type"] in {"update", "delete"}
                    and operation["target_event_id"] in before_snapshots
                    else None
                ),
                "after": None,
                "undo_stage": "pending",
                "undo_fidelity": (
                    "core_only"
                    if operation["type"] == "delete"
                    and operation["target_event_id"] in before_snapshots
                    and _delete_undo_is_core_only(
                        before_snapshots[operation["target_event_id"]]
                    )
                    else "full"
                ),
            }
            for index, operation in enumerate(plan["operations"])
        ]
        now = _iso_now()
        return {
            "operation_id": operation_id,
            "source_key": source_key,
            "conversation_key": _conversation_key(account, chat_id),
            "account": account,
            "owner_user_id": owner_user_id,
            "chat_id": chat_id,
            "transcript": transcript,
            "reference_time": reference_time.isoformat(),
            "stage": "planned",
            "confidence": plan["confidence"],
            "clarification_question": plan.get("clarification_question"),
            "items": items,
            "created_at": now,
            "updated_at": now,
            "undo": {"stage": "unavailable"},
            "displayed_candidates": (
                [deepcopy(dict(candidate)) for candidate in displayed_candidates]
                if displayed_candidates is not None
                else None
            ),
            "interaction_input": deepcopy(dict(interaction_input)) if interaction_input else None,
            "interaction_steps": [deepcopy(dict(step)) for step in (interaction_steps or ())],
            "assistant_text": assistant_text,
            "legacy": False,
        }

    @staticmethod
    def _verify_source(
        record: Mapping[str, Any], account: str, owner_user_id: int, chat_id: int
    ) -> None:
        if (
            record.get("account") != account
            or record.get("owner_user_id") != owner_user_id
            or record.get("chat_id") != chat_id
        ):
            raise OperationStateError("Source update contradicts its journal record")

    async def _apply_item(
        self,
        record: dict[str, Any],
        item: dict[str, Any],
        *,
        before_is_fresh: bool = False,
    ) -> None:
        account = str(record["account"])
        operation_id = str(record["operation_id"])
        index = int(item["index"])
        action = str(item["type"])
        item["stage"] = "applying"
        self.store.put(record)
        key = f"calendar-operation:{operation_id}:{index}:{action}"
        if action == "create":
            event = item["request"]["event"]
            item["provider_write_started_at"] = _iso_now()
            self.store.put(record)
            created = await self.calendar.create_events(
                account=account, events=[event], idempotency_key=key
            )
            if len(created) != 1:
                raise OperationStateError("Calendar create returned the wrong count")
            reference = created[0]
            event_id = str(getattr(reference, "event_id", ""))
            html_link = getattr(reference, "html_link", None)
            try:
                provider = await self.calendar.get_event(account=account, event_id=event_id)
            except CalendarConnectionError:
                raise
            except Exception:
                provider = None
            after = _snapshot(
                provider,
                account=account,
                fallback=event,
                event_id=event_id,
                html_link=html_link,
                timezone_name=None,
            )
            item["after"] = after
            self.store._put_event(account, after, active=True, operation_id=operation_id)
        elif action == "update":
            event_id = str(item["target_event_id"])
            entry = self.store.event_entry(account, event_id)
            if (
                (entry is None or entry.get("active") is not True)
                and item.get("provider_write_started_at") is None
            ):
                raise OperationStateError("Update target is not active")
            if (
                item.get("provider_write_started_at") is None
                and not before_is_fresh
            ):
                fresh_before = _snapshot(
                    await self.calendar.get_event(account=account, event_id=event_id),
                    account=account,
                    timezone_name=None,
                )
                if fresh_before["event_id"] != event_id:
                    raise OperationStateError("Fresh update target has another ID")
                before = fresh_before
                item["before"] = fresh_before
                self.store.observe_events(account, [fresh_before])
                self.store.put(record)
            else:
                before = item.get("before")
                if not isinstance(before, dict):
                    raise OperationStateError("Update retry has no before snapshot")
            provider_patch, desired = self._merged_update(
                before,
                item["request"].get("patch") or {},
                item["request"].get("clear_fields") or [],
            )
            item["desired_after"] = desired
            if not provider_patch:
                item["write_skipped"] = True
                item["after"] = deepcopy(before)
                item["stage"] = "applied"
                item.pop("last_error_class", None)
                return
            item["provider_write_started_at"] = _iso_now()
            self.store.put(record)
            provider = await self.calendar.update_event(
                account=account,
                event_id=event_id,
                patch=provider_patch,
                idempotency_key=key,
            )
            after = _snapshot(
                provider,
                account=account,
                fallback=desired,
                timezone_name=None,
            )
            item["after"] = after
            self.store._put_event(account, after, active=True, operation_id=operation_id)
        elif action == "delete":
            event_id = str(item["target_event_id"])
            entry = self.store.event_entry(account, event_id)
            if (
                (entry is None or entry.get("active") is not True)
                and item.get("provider_write_started_at") is None
            ):
                raise OperationStateError("Delete target is not active")
            if (
                item.get("provider_write_started_at") is None
                and not before_is_fresh
            ):
                fresh_before = _snapshot(
                    await self.calendar.get_event(account=account, event_id=event_id),
                    account=account,
                    timezone_name=None,
                )
                if fresh_before["event_id"] != event_id:
                    raise OperationStateError("Fresh delete target has another ID")
                before = fresh_before
                item["before"] = fresh_before
                self.store.observe_events(account, [fresh_before])
                self.store.put(record)
            before = item.get("before")
            if not isinstance(before, dict):
                raise OperationStateError("Delete retry has no before snapshot")
            item["provider_write_started_at"] = _iso_now()
            self.store.put(record)
            deletion = await self.calendar.delete_event(
                account=account, event_id=event_id, idempotency_key=key
            )
            provider_current = getattr(deletion, "current", None)
            deleted_snapshot = _snapshot(
                provider_current,
                account=account,
                fallback={**before, "status": "cancelled"},
                event_id=event_id,
                timezone_name=None,
            )
            if deleted_snapshot.get("status") != "cancelled":
                raise OperationStateError("Calendar delete was not confirmed")
            self.store._put_event(
                account, deleted_snapshot, active=False, operation_id=operation_id
            )
        else:  # pragma: no cover - protected by plan validation
            raise OperationStateError("Unknown calendar operation")
        item["stage"] = "applied"
        item.pop("last_error_class", None)

    def _merged_update(
        self,
        before: Mapping[str, Any],
        patch: Mapping[str, Any],
        clear_fields: Sequence[str],
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        if "recurrence_rrule" in patch or "recurrence_rrule" in clear_fields:
            raise OperationStateError("Recurring event updates require explicit scope")
        desired = {**dict(before), **dict(patch)}
        for field in clear_fields:
            desired[field] = None
        old_all_day = bool(before.get("all_day"))
        new_all_day = bool(desired.get("all_day"))
        if "start_at" in patch and "end_at" not in patch:
            old_start = _parse_temporal(str(before["start_at"]), all_day=old_all_day)
            old_end = _parse_temporal(str(before["end_at"]), all_day=old_all_day)
            new_start = _parse_temporal(str(desired["start_at"]), all_day=new_all_day)
            desired["end_at"] = _format_temporal(
                new_start + (old_end - old_start), all_day=new_all_day
            )
        temporal_patch = bool(
            {"start_at", "end_at", "all_day", "timezone"} & set(patch)
        )
        if temporal_patch:
            # Model-authored temporal values are validated in the configured
            # calendar zone.  Convert any untouched timed endpoint into that
            # zone as well, so an end-only change cannot leave a mixed-offset
            # event after moving a UTC/London provider event.
            desired["timezone"] = self.timezone_name
            if not new_all_day:
                configured_zone = ZoneInfo(self.timezone_name)
                for field in ("start_at", "end_at"):
                    parsed = _parse_temporal(
                        str(desired[field]), all_day=False
                    )
                    assert isinstance(parsed, datetime)
                    desired[field] = parsed.astimezone(configured_zone).isoformat()
        complete = _event_payload(desired)
        validation_timezone = (
            self.timezone_name
            if temporal_patch
            else desired.get("timezone")
        )
        if validation_timezone is None:
            # Google can return a valid offset-aware timed event without an
            # IANA timezone.  A title/location-only patch must not relabel or
            # rewrite its temporal fields merely to satisfy model-side schema.
            start_value = _parse_temporal(
                str(complete["start_at"]), all_day=bool(complete["all_day"])
            )
            end_value = _parse_temporal(
                str(complete["end_at"]), all_day=bool(complete["all_day"])
            )
            if end_value <= start_value or end_value - start_value > timedelta(
                days=366
            ):
                raise OperationStateError("Provider event has invalid temporal bounds")
            validated = complete
        else:
            validated = validate_calendar_intent(
                {
                    "action": "create",
                    "events": [complete],
                    "clarification_question": None,
                    "confidence": 1,
                },
                expected_timezone=str(validation_timezone),
            )["events"][0]
        desired.update(validated)
        provider_patch: dict[str, Any] = {}
        for field in ("title", "description", "location", "timezone"):
            before_value = before.get(field)
            after_value = desired.get(field)
            if before_value != after_value:
                provider_patch[field] = "" if after_value is None else after_value
        if (
            before.get("start_at") != desired.get("start_at")
            or before.get("end_at") != desired.get("end_at")
            or old_all_day != new_all_day
        ):
            provider_patch["start_at"] = desired["start_at"]
            provider_patch["end_at"] = desired["end_at"]
            provider_patch["timezone"] = desired["timezone"]
        return provider_patch, desired

    @staticmethod
    def _result_text(record: Mapping[str, Any]) -> str:
        actions = [str(item.get("type")) for item in record.get("items", [])]
        return "Google Calendar обновлён: " + ", ".join(actions)

    async def undo(
        self,
        *,
        operation_id: str,
        owner_user_id: int,
        chat_id: int,
        source_update_id: int | None = None,
        assistant_text: str | None = None,
    ) -> UndoResult:
        async with self._lock:
            record = self.store.get(operation_id)
            if record is None or (
                record.get("owner_user_id") != owner_user_id
                or record.get("chat_id") != chat_id
            ):
                return UndoResult(True, "rejected")
            titles = self._titles(record)
            undo = record.get("undo") or {}
            undo_fidelity = (
                "core_only" if undo.get("fidelity") == "core_only" else "full"
            )
            if record.get("stage") == "undone" or undo.get("stage") == "undone":
                return UndoResult(True, "already_undone", record, titles)
            if record.get("stage") != "applied":
                return UndoResult(True, "blocked", record, titles)
            if not self._undo_is_fresh(record):
                record["undo"] = {
                    "stage": "blocked",
                    "fidelity": undo_fidelity,
                    "updated_at": _iso_now(),
                }
                self.store.put(record)
                return UndoResult(True, "blocked", record, titles)
            try:
                provider_fresh = await self._provider_undo_is_fresh(record)
            except CalendarConnectionError:
                record["undo"] = {
                    "stage": "undoing",
                    "fidelity": undo_fidelity,
                    "updated_at": _iso_now(),
                }
                record["updated_at"] = _iso_now()
                self.store.put(record)
                raise
            except Exception:
                # An ambiguous provider read must not erase the local freshness
                # marker or turn uncertainty into permission to overwrite.
                return UndoResult(True, "retryable_error", record, titles)
            if not provider_fresh:
                record["undo"] = {
                    "stage": "blocked",
                    "fidelity": undo_fidelity,
                    "updated_at": _iso_now(),
                }
                self.store.put(record)
                return UndoResult(True, "blocked", record, titles)

            record["undo"] = {
                "stage": "undoing",
                "fidelity": undo_fidelity,
                "updated_at": _iso_now(),
            }
            self.store.put(record)
            try:
                for item in reversed(record["items"]):
                    if item.get("undo_stage") == "undone":
                        continue
                    await self._undo_item(record, item)
                    item["undo_stage"] = "undone"
                    record["updated_at"] = _iso_now()
                    self.store.put(record)
            except CalendarConnectionError:
                record["undo"] = {
                    "stage": "undoing",
                    "fidelity": undo_fidelity,
                    "updated_at": _iso_now(),
                }
                record["updated_at"] = _iso_now()
                self.store.put(record)
                raise
            except Exception as exc:
                record["undo"] = {
                    "stage": "failed",
                    "fidelity": undo_fidelity,
                    "last_error_class": type(exc).__name__,
                    "updated_at": _iso_now(),
                }
                self.store.put(record)
                return UndoResult(True, "retryable_error", record, titles)

            record["stage"] = "undone"
            record["undo"] = {
                "stage": "undone",
                "fidelity": undo_fidelity,
                "updated_at": _iso_now(),
            }
            record["updated_at"] = _iso_now()
            self.store.put(record)
            undo_record = deepcopy(record)
            undo_record["source_key"] = (
                _source_key(source_update_id)
                if source_update_id is not None
                else f"undo:{operation_id}:{record['updated_at']}"
            )
            undo_record["transcript"] = "Пользователь нажал кнопку отмены операции."
            undo_record["assistant_text"] = assistant_text or "Операция отменена."
            undo_record["interaction_input"] = None
            undo_record["interaction_steps"] = []
            self.store.append_turn(undo_record)
            return UndoResult(True, "undone", deepcopy(record), titles)

    def _undo_is_fresh(self, record: Mapping[str, Any]) -> bool:
        operation_id = record.get("operation_id")
        account = str(record.get("account"))
        for item in record.get("items", []):
            if not isinstance(item, dict):
                return False
            if item.get("write_skipped") is True:
                continue
            snapshot = item.get("after") or item.get("before")
            if not isinstance(snapshot, dict):
                return False
            entry = self.store.event_entry(account, str(snapshot.get("event_id")))
            if entry is None or entry.get("last_operation_id") != operation_id:
                return False
        return True

    async def _provider_undo_is_fresh(self, record: Mapping[str, Any]) -> bool:
        """Verify pending compensations against fresh provider state.

        Items whose undo write already started deliberately skip this probe:
        after an ambiguous response, replaying their idempotency key is the only
        safe way to finish without duplicating or abandoning the compensation.
        """

        account = str(record.get("account"))
        for item in record.get("items", []):
            if not isinstance(item, Mapping):
                return False
            if item.get("write_skipped") is True:
                continue
            if item.get("undo_stage") in {"undoing", "undone"}:
                continue
            action = str(item.get("type"))
            expected = item.get("after") or item.get("before")
            if not isinstance(expected, Mapping):
                return False
            provider = await self.calendar.get_event(
                account=account,
                event_id=str(expected.get("event_id") or ""),
            )
            current = _snapshot(
                provider,
                account=account,
                timezone_name=None,
            )
            if action == "delete":
                if current.get("status") != "cancelled":
                    return False
            elif action in {"create", "update"}:
                if not _materially_equivalent(expected, current):
                    return False
            else:
                return False
        return True

    async def _undo_item(
        self, record: dict[str, Any], item: dict[str, Any]
    ) -> None:
        account = str(record["account"])
        operation_id = str(record["operation_id"])
        index = int(item["index"])
        action = str(item["type"])
        key = f"calendar-operation:{operation_id}:{index}:undo"
        item["undo_stage"] = "undoing"
        record["updated_at"] = _iso_now()
        self.store.put(record)
        if action == "create":
            event_id = str(item["after"]["event_id"])
            await self.calendar.delete_event(
                account=account, event_id=event_id, idempotency_key=key
            )
            self.store._put_event(
                account, item["after"], active=False, operation_id=operation_id
            )
        elif action == "update":
            before = item["before"]
            after = item["after"]
            if item.get("write_skipped") is True:
                item["undo_after"] = deepcopy(before)
                return
            patch: dict[str, Any] = {}
            for field in ("title", "description", "location", "timezone"):
                if before.get(field) != after.get(field):
                    patch[field] = "" if before.get(field) is None else before.get(field)
            if (
                before.get("start_at") != after.get("start_at")
                or before.get("end_at") != after.get("end_at")
                or before.get("all_day") != after.get("all_day")
            ):
                patch.update(
                    start_at=before["start_at"],
                    end_at=before["end_at"],
                    timezone=before["timezone"],
                )
            restored = await self.calendar.update_event(
                account=account,
                event_id=str(after["event_id"]),
                patch=patch,
                idempotency_key=key,
            )
            restored_snapshot = _snapshot(
                restored,
                account=account,
                fallback=before,
                timezone_name=None,
            )
            item["undo_after"] = restored_snapshot
            self.store._put_event(
                account, restored_snapshot, active=True, operation_id=operation_id
            )
        elif action == "delete":
            before = item["before"]
            restore_event = _event_payload(before)
            if restore_event.get("timezone") is None:
                if restore_event.get("all_day") is not True:
                    raise OperationStateError(
                        "Timed calendar restore is missing its timezone"
                    )
                # Google normally omits a timezone for all-day events, while the
                # Calendar MCP create schema still requires one.  The timezone is
                # semantically irrelevant to date-only boundaries, so use the
                # bot's configured zone for the compensating create only.
                restore_event["timezone"] = self.timezone_name
            created = await self.calendar.create_events(
                account=account,
                events=[restore_event],
                idempotency_key=key,
            )
            if len(created) != 1:
                raise OperationStateError("Calendar restore returned the wrong count")
            reference = created[0]
            restored_id = str(getattr(reference, "event_id", ""))
            try:
                provider = await self.calendar.get_event(
                    account=account, event_id=restored_id
                )
            except CalendarConnectionError:
                raise
            except Exception:
                provider = None
            restored = _snapshot(
                provider,
                account=account,
                fallback=before,
                event_id=restored_id,
                html_link=getattr(reference, "html_link", None),
                timezone_name=None,
            )
            item["undo_after"] = restored
            self.store._put_event(
                account, restored, active=True, operation_id=operation_id
            )
        else:  # pragma: no cover
            raise OperationStateError("Unknown undo operation")

    @staticmethod
    def _titles(record: Mapping[str, Any]) -> tuple[str, ...]:
        return tuple(
            str((item.get("after") or item.get("before") or {}).get("title") or "Без названия")
            for item in record.get("items", [])
            if isinstance(item, dict)
        )

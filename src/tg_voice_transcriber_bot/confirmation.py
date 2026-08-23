"""Persistent, owner-bound confirmation flow for future calendar writes."""

from __future__ import annotations

import asyncio
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import re
import secrets
from typing import Any, Callable, Literal, Mapping

from .calendar import CalendarClient, CalendarConnectionError, CreatedCalendarEvent
from .intent import validate_calendar_intent


CONFIDENCE_THRESHOLD = 0.85
_CALLBACK_PATTERN = re.compile(r"cal:(add|cancel):([A-Za-z0-9_-]{16,32})\Z")
_TERMINAL_STAGES = frozenset({"created", "cancelled", "expired"})
_VALID_STAGES = frozenset({"pending", "creating", *_TERMINAL_STAGES})


class ConfirmationStateError(RuntimeError):
    """The confirmation journal is corrupt or contradicts an existing record."""


@dataclass(frozen=True)
class PreparedConfirmation:
    confirmation_id: str
    reply_markup: dict[str, Any]


@dataclass(frozen=True)
class CallbackResult:
    handled: bool
    outcome: Literal[
        "not_handled",
        "rejected",
        "expired",
        "cancelled",
        "already_cancelled",
        "creating",
        "created",
        "already_created",
        "retryable_error",
    ]
    answer_text: str
    remove_keyboard: bool = False
    created_events: tuple[CreatedCalendarEvent, ...] = ()


def confirmation_reply_markup(confirmation_id: str) -> dict[str, Any]:
    """Build a Telegram inline keyboard containing only allowlisted actions."""
    if not re.fullmatch(r"[A-Za-z0-9_-]{16,32}", confirmation_id):
        raise ValueError("invalid confirmation ID")
    return {
        "inline_keyboard": [
            [
                {
                    "text": "Добавить",
                    "callback_data": f"cal:add:{confirmation_id}",
                },
                {
                    "text": "Отмена",
                    "callback_data": f"cal:cancel:{confirmation_id}",
                },
            ]
        ]
    }


def parse_calendar_callback(data: Any) -> tuple[Literal["add", "cancel"], str] | None:
    """Return only exact, size-bounded callback commands owned by this flow."""
    if not isinstance(data, str):
        return None
    try:
        if len(data.encode("utf-8")) > 64:
            return None
    except UnicodeEncodeError:
        return None
    match = _CALLBACK_PATTERN.fullmatch(data)
    if match is None:
        return None
    action = match.group(1)
    if action not in {"add", "cancel"}:  # Defensive if the regex changes later.
        return None
    return action, match.group(2)  # type: ignore[return-value]


class ConfirmationStore:
    """Small atomic JSON journal, intentionally separate from polling offsets."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._data = self._load()

    @staticmethod
    def _default() -> dict[str, Any]:
        return {"version": 1, "records": {}, "source_index": {}}

    def _load(self) -> dict[str, Any]:
        if not self.path.exists():
            return self._default()
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ConfirmationStateError(
                f"Cannot read confirmation state: {type(exc).__name__}"
            ) from exc
        if not isinstance(raw, dict) or set(raw) != {
            "version",
            "records",
            "source_index",
        }:
            raise ConfirmationStateError("Invalid confirmation state root")
        if raw["version"] != 1:
            raise ConfirmationStateError("Unsupported confirmation state version")
        records = raw["records"]
        source_index = raw["source_index"]
        if not isinstance(records, dict) or not isinstance(source_index, dict):
            raise ConfirmationStateError("Invalid confirmation state indexes")
        for confirmation_id, record in records.items():
            if (
                not isinstance(confirmation_id, str)
                or not isinstance(record, dict)
                or record.get("confirmation_id") != confirmation_id
                or record.get("stage") not in _VALID_STAGES
            ):
                raise ConfirmationStateError("Invalid confirmation record")
        if any(
            not isinstance(source, str)
            or not isinstance(confirmation_id, str)
            or confirmation_id not in records
            for source, confirmation_id in source_index.items()
        ):
            raise ConfirmationStateError("Invalid confirmation source index")
        return raw

    def get(self, confirmation_id: str) -> dict[str, Any] | None:
        record = self._data["records"].get(confirmation_id)
        return deepcopy(record) if isinstance(record, dict) else None

    def find_by_source(self, source_key: str) -> dict[str, Any] | None:
        confirmation_id = self._data["source_index"].get(source_key)
        return self.get(confirmation_id) if isinstance(confirmation_id, str) else None

    def put_new(self, record: Mapping[str, Any]) -> None:
        confirmation_id = str(record["confirmation_id"])
        source_key = str(record["source_key"])
        if confirmation_id in self._data["records"]:
            raise ConfirmationStateError("Confirmation ID collision")
        if source_key in self._data["source_index"]:
            raise ConfirmationStateError("Confirmation source already exists")
        self._data["records"][confirmation_id] = deepcopy(dict(record))
        self._data["source_index"][source_key] = confirmation_id
        self._save()

    def update(self, record: Mapping[str, Any]) -> None:
        confirmation_id = str(record["confirmation_id"])
        existing = self._data["records"].get(confirmation_id)
        if not isinstance(existing, dict):
            raise ConfirmationStateError("Unknown confirmation record")
        if existing.get("source_key") != record.get("source_key"):
            raise ConfirmationStateError("Confirmation source cannot change")
        self._data["records"][confirmation_id] = deepcopy(dict(record))
        self._save()

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_name(f".{self.path.name}.{os.getpid()}.tmp")
        serialized = json.dumps(
            self._data, ensure_ascii=False, sort_keys=True, indent=2
        ) + "\n"
        descriptor = os.open(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o600,
        )
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


class CalendarConfirmationPipeline:
    """Prepare previews and execute only an authenticated explicit confirmation."""

    def __init__(
        self,
        store: ConfirmationStore,
        calendar: CalendarClient,
        *,
        timezone_name: str = "Europe/Moscow",
        confidence_threshold: float = CONFIDENCE_THRESHOLD,
        confirmation_ttl: timedelta = timedelta(hours=24),
        now: Callable[[], datetime] | None = None,
    ) -> None:
        if not CONFIDENCE_THRESHOLD <= confidence_threshold <= 1:
            raise ValueError(
                f"confidence threshold must be between {CONFIDENCE_THRESHOLD} and 1"
            )
        if confirmation_ttl <= timedelta(0):
            raise ValueError("confirmation TTL must be positive")
        self.store = store
        self.calendar = calendar
        self.timezone_name = timezone_name
        self.confidence_threshold = confidence_threshold
        self.confirmation_ttl = confirmation_ttl
        self._now = now or (lambda: datetime.now(timezone.utc))
        self._lock = asyncio.Lock()
        self._inflight: set[str] = set()

    def prepare(
        self,
        *,
        source_update_id: int,
        account: str,
        owner_user_id: int,
        chat_id: int,
        intent: Mapping[str, Any],
    ) -> PreparedConfirmation | None:
        """Persist one pending confirmation if confidence passes the hard gate."""
        normalized = validate_calendar_intent(
            dict(intent), expected_timezone=self.timezone_name
        )
        if (
            normalized["action"] != "create"
            or normalized["confidence"] < self.confidence_threshold
        ):
            return None
        if source_update_id < 0 or owner_user_id <= 0 or chat_id <= 0 or not account:
            raise ValueError("invalid confirmation ownership metadata")

        source_key = f"telegram-update:{source_update_id}"
        existing = self.store.find_by_source(source_key)
        expected = {
            "account": account,
            "owner_user_id": owner_user_id,
            "chat_id": chat_id,
            "intent": normalized,
        }
        if existing is not None:
            if any(existing.get(key) != value for key, value in expected.items()):
                raise ConfirmationStateError(
                    "Source update contradicts its persisted confirmation"
                )
            return PreparedConfirmation(
                confirmation_id=str(existing["confirmation_id"]),
                reply_markup=confirmation_reply_markup(
                    str(existing["confirmation_id"])
                ),
            )

        confirmation_id = self._new_id()
        now = self._utc_now()
        record = {
            "confirmation_id": confirmation_id,
            "source_key": source_key,
            "account": account,
            "owner_user_id": owner_user_id,
            "chat_id": chat_id,
            "intent": normalized,
            "stage": "pending",
            "created_at": now.isoformat(),
            "expires_at": (now + self.confirmation_ttl).isoformat(),
            "updated_at": now.isoformat(),
            "attempts": 0,
            "calendar_events": [],
        }
        self.store.put_new(record)
        return PreparedConfirmation(
            confirmation_id=confirmation_id,
            reply_markup=confirmation_reply_markup(confirmation_id),
        )

    async def handle_callback(
        self,
        *,
        data: Any,
        owner_user_id: int,
        chat_id: int,
    ) -> CallbackResult:
        """Apply an allowlisted callback after binding it to its owner and chat."""
        parsed = parse_calendar_callback(data)
        if parsed is None:
            return CallbackResult(False, "not_handled", "")
        action, confirmation_id = parsed

        async with self._lock:
            record = self.store.get(confirmation_id)
            if record is None or (
                record.get("owner_user_id") != owner_user_id
                or record.get("chat_id") != chat_id
            ):
                # Do not disclose whether a guessed opaque ID exists.
                return CallbackResult(True, "rejected", "Недоступно.", True)

            stage = str(record["stage"])
            if stage == "pending" and self._is_expired(record):
                self._transition(record, "expired")
                return CallbackResult(
                    True, "expired", "Подтверждение устарело.", True
                )

            if action == "cancel":
                if stage == "pending":
                    self._transition(record, "cancelled")
                    return CallbackResult(True, "cancelled", "Отменено.", True)
                if stage in {"cancelled", "expired"}:
                    return CallbackResult(
                        True, "already_cancelled", "Уже отменено.", True
                    )
                if stage == "created":
                    return CallbackResult(
                        True, "already_created", "Событие уже добавлено.", True
                    )
                # Never claim cancellation while an idempotent write may have
                # reached the provider but its response is still unknown.
                return CallbackResult(
                    True, "creating", "Добавление уже выполняется."
                )

            if stage in {"cancelled", "expired"}:
                return CallbackResult(
                    True, "already_cancelled", "Подтверждение уже отменено.", True
                )
            if stage == "created":
                return CallbackResult(
                    True,
                    "already_created",
                    "Событие уже добавлено.",
                    True,
                    self._created_events(record),
                )
            if confirmation_id in self._inflight:
                return CallbackResult(
                    True, "creating", "Добавление уже выполняется."
                )

            # ``creating`` can be resumed after a process crash.  The same
            # durable idempotency key makes that retry safe at the adapter.
            record["stage"] = "creating"
            record["attempts"] = int(record.get("attempts", 0)) + 1
            record["updated_at"] = self._utc_now().isoformat()
            record.pop("last_error", None)
            self.store.update(record)
            self._inflight.add(confirmation_id)

        try:
            created = tuple(
                await self.calendar.create_events(
                    account=str(record["account"]),
                    events=deepcopy(record["intent"]["events"]),
                    idempotency_key=f"tg-calendar:{confirmation_id}",
                )
            )
            if len(created) != len(record["intent"]["events"]) or any(
                not isinstance(item, CreatedCalendarEvent) or not item.event_id
                for item in created
            ):
                raise RuntimeError("calendar adapter returned an invalid result")
        except CalendarConnectionError:
            # The stdio session cannot recover in-process. Keep the durable
            # record in ``creating`` so the same idempotency key is resumed
            # after the supervisor rebuilds Calendar MCP, but release the
            # process-local guard before propagating the fatal marker.
            async with self._lock:
                self._inflight.discard(confirmation_id)
            raise
        except Exception as exc:
            async with self._lock:
                self._inflight.discard(confirmation_id)
                current = self.store.get(confirmation_id)
                if current is not None and current.get("stage") == "creating":
                    # Store only the exception class; provider errors may embed
                    # tokens, event content, or URLs.
                    current["last_error"] = type(exc).__name__
                    current["updated_at"] = self._utc_now().isoformat()
                    self.store.update(current)
            return CallbackResult(
                True,
                "retryable_error",
                "Не удалось добавить событие. Нажмите «Добавить» ещё раз.",
            )

        async with self._lock:
            self._inflight.discard(confirmation_id)
            current = self.store.get(confirmation_id)
            if current is None:
                raise ConfirmationStateError("Confirmation disappeared during write")
            current["calendar_events"] = [
                {"event_id": item.event_id, "html_link": item.html_link}
                for item in created
            ]
            self._transition(current, "created")
        return CallbackResult(
            True,
            "created",
            "Событие добавлено.",
            True,
            created,
        )

    def _transition(self, record: dict[str, Any], stage: str) -> None:
        if stage not in _VALID_STAGES:
            raise ValueError("invalid confirmation stage")
        record["stage"] = stage
        record["updated_at"] = self._utc_now().isoformat()
        self.store.update(record)

    def _new_id(self) -> str:
        for _ in range(10):
            value = secrets.token_urlsafe(12)
            if self.store.get(value) is None:
                return value
        raise ConfirmationStateError("Cannot allocate a unique confirmation ID")

    def _utc_now(self) -> datetime:
        value = self._now()
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("confirmation clock must return an aware datetime")
        return value.astimezone(timezone.utc)

    def _is_expired(self, record: Mapping[str, Any]) -> bool:
        try:
            expires_at = datetime.fromisoformat(str(record["expires_at"]))
        except (KeyError, ValueError) as exc:
            raise ConfirmationStateError("Invalid confirmation expiry") from exc
        if expires_at.tzinfo is None or expires_at.utcoffset() is None:
            raise ConfirmationStateError("Invalid confirmation expiry timezone")
        return self._utc_now() >= expires_at.astimezone(timezone.utc)

    @staticmethod
    def _created_events(record: Mapping[str, Any]) -> tuple[CreatedCalendarEvent, ...]:
        raw = record.get("calendar_events", [])
        if not isinstance(raw, list) or any(not isinstance(item, dict) for item in raw):
            raise ConfirmationStateError("Invalid stored calendar events")
        try:
            return tuple(
                CreatedCalendarEvent(
                    event_id=str(item["event_id"]),
                    html_link=(
                        str(item["html_link"])
                        if item.get("html_link") is not None
                        else None
                    ),
                )
                for item in raw
            )
        except KeyError as exc:
            raise ConfirmationStateError("Invalid stored calendar event") from exc

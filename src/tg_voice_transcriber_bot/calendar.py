"""Provider-neutral calendar read and write boundary.

Implementations must make retries safe.  Creation uses durable, deterministic
provider IDs; updates and deletes use the target event's observed state to
recover when a provider response is lost.
"""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Mapping
from typing import Any, Protocol, Sequence


class CalendarConnectionError(RuntimeError):
    """The calendar transport is dead and must be rebuilt out of process.

    This provider-neutral marker deliberately carries no upstream diagnostic
    text.  Webhook runtimes use it to keep the current update durable and
    terminate the worker so a fresh subprocess is created on restart.
    """


class CalendarWriteRejectedError(RuntimeError):
    """The provider conclusively rejected a write before it took effect.

    Adapters may raise this only after they can prove that the requested
    mutation is absent.  Unlike a lost response or dead transport, replaying
    the same payload cannot reconcile this outcome and should not consume the
    service's ambiguity retry budget.
    """


@dataclass(frozen=True)
class CreatedCalendarEvent:
    """Stable, non-secret reference returned by a calendar adapter."""

    event_id: str
    html_link: str | None = None


@dataclass(frozen=True)
class CalendarEventSnapshot:
    """Normalized event state returned by a calendar adapter.

    The shape deliberately contains provider-neutral values that are useful
    for conversation history, optimistic verification, and compensating
    updates.  ``start_at`` and ``end_at`` are either ISO dates for all-day
    events or timezone-aware RFC3339 datetimes for timed events.
    """

    account: str
    calendar_id: str
    event_id: str
    title: str | None
    description: str | None
    location: str | None
    start_at: str
    end_at: str
    all_day: bool
    timezone: str | None
    status: str
    html_link: str | None = None
    recurrence_rrules: tuple[str, ...] = ()
    attendee_emails: tuple[str, ...] = ()
    updated_at: str | None = None
    recurring_event_id: str | None = None
    original_start_at: str | None = None
    color_id: str | None = None
    transparency: str | None = None
    visibility: str | None = None
    event_type: str | None = None
    creator_email: str | None = None
    creator_is_self: bool | None = None
    organizer_email: str | None = None
    organizer_is_self: bool | None = None
    reminders_present: bool = False
    reminders_use_default: bool | None = None
    reminder_overrides: tuple[tuple[str, int], ...] = ()
    has_conference_data: bool = False
    has_hangout_link: bool = False
    has_attachments: bool = False
    has_extended_properties: bool = False
    has_source: bool = False
    anyone_can_add_self: bool | None = None
    guests_can_invite_others: bool | None = None
    guests_can_modify: bool | None = None
    guests_can_see_other_guests: bool | None = None
    private_copy: bool | None = None
    locked: bool | None = None
    safety_metadata_complete: bool = False
    safety_metadata_fingerprint: str | None = None


@dataclass(frozen=True)
class CalendarEventQueryResult:
    """Bounded, normalized result from calendar event discovery.

    ``total_count`` is the provider's exact count for the page received by the
    adapter, before local status filtering or truncation.  It is therefore a
    useful lower bound when ``may_be_incomplete`` is true.
    """

    events: tuple[CalendarEventSnapshot, ...]
    total_count: int
    may_be_incomplete: bool = False


@dataclass(frozen=True)
class DeletedCalendarEvent:
    """Result of an idempotent delete boundary.

    ``previous`` retains the data needed for history or a compensating
    recreate.  Google may keep a cancelled tombstone which is exposed as
    ``current``.  A missing ``current`` does not negate a successful delete;
    it only means the provider could not be probed after the write.
    """

    previous: CalendarEventSnapshot
    current: CalendarEventSnapshot | None = None
    already_deleted: bool = False
    verified_cancelled: bool = False


class CalendarClient(Protocol):
    """Boundary for retry-safe calendar creation and mutation.

    Implementations must treat ``idempotency_key`` as durable.  If the caller
    repeats the operation after losing a response, the implementation must not
    create a second copy of any event in the batch.
    """

    async def create_events(
        self,
        *,
        account: str,
        events: Sequence[dict[str, Any]],
        idempotency_key: str,
    ) -> Sequence[CreatedCalendarEvent]: ...

    async def get_event(
        self,
        *,
        account: str,
        event_id: str,
    ) -> CalendarEventSnapshot: ...

    async def list_events(
        self,
        *,
        account: str,
        time_min: str,
        time_max: str,
        limit: int = 50,
    ) -> CalendarEventQueryResult: ...

    async def search_events(
        self,
        *,
        account: str,
        query: str,
        time_min: str,
        time_max: str,
        limit: int = 50,
    ) -> CalendarEventQueryResult: ...

    async def update_event(
        self,
        *,
        account: str,
        event_id: str,
        patch: Mapping[str, Any],
        idempotency_key: str,
    ) -> CalendarEventSnapshot: ...

    async def delete_event(
        self,
        *,
        account: str,
        event_id: str,
        idempotency_key: str,
    ) -> DeletedCalendarEvent: ...

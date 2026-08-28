import asyncio
from contextlib import asynccontextmanager
import json
from types import SimpleNamespace

import anyio
from mcp.shared.exceptions import McpError
from mcp.types import ErrorData
import pytest

import tg_voice_transcriber_bot.calendar_mcp as calendar_mcp_module
from tg_voice_transcriber_bot.calendar import (
    CalendarConnectionError,
    CalendarEventQueryResult,
    CalendarEventSnapshot,
    CalendarStateConflictError,
    DeletedCalendarEvent,
)
from tg_voice_transcriber_bot.calendar_mcp import (
    CALENDAR_MCP_TOOLS,
    CalendarMcpCollisionError,
    CalendarMcpConnectionError,
    CalendarMcpError,
    CalendarMcpWriteRejectedError,
    GoogleCalendarMcpClient,
    google_event_id,
)


def timed_event(**updates):
    event = {
        "title": "Встреча с Анной",
        "start_at": "2026-08-24T15:00:00+03:00",
        "end_at": "2026-08-24T16:00:00+03:00",
        "all_day": False,
        "timezone": "Europe/Moscow",
        "location": None,
        "description": None,
        "recurrence_rrule": None,
    }
    event.update(updates)
    return event


def stored_event(event_id, source, *, html_link="https://calendar.google.com/event/1"):
    all_day = source["all_day"]
    return {
        "id": event_id,
        "summary": source["title"],
        "description": source["description"],
        "location": source["location"],
        "start": {
            "date" if all_day else "dateTime": source["start_at"],
        },
        "end": {
            "date" if all_day else "dateTime": source["end_at"],
        },
        "recurrence": (
            [source["recurrence_rrule"]] if source["recurrence_rrule"] else None
        ),
        "status": "confirmed",
        "htmlLink": html_link,
        "updated": "2026-08-22T10:15:00Z",
        "accountId": "google_personal",
        "calendarId": "primary@example.com",
    }


def normalized_snapshot(event_id, source, **provider_updates):
    provider_event = stored_event(event_id, source)
    provider_event.update(provider_updates)
    return calendar_mcp_module._snapshot_from_event(
        provider_event,
        logical_account="personal",
        mcp_account="google_personal",
        requested_calendar_id="primary",
        expected_event_id=event_id,
    )


def result(
    payload=None,
    *,
    error=False,
    text_fallback=False,
    error_text="provider error details",
):
    if payload is None:
        content = [SimpleNamespace(type="text", text=error_text)]
        structured = None
    elif text_fallback:
        content = [SimpleNamespace(type="text", text=json.dumps(payload))]
        structured = None
    else:
        content = []
        structured = payload
    return SimpleNamespace(
        structuredContent=structured,
        content=content,
        isError=error,
    )


class FakeSession:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    async def call_tool(self, name, arguments, *, read_timeout_seconds):
        self.calls.append((name, arguments, read_timeout_seconds.total_seconds()))
        if not self.responses:
            raise AssertionError("unexpected MCP call")
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


def client(session):
    return GoogleCalendarMcpClient(
        session,
        account_mapping={"personal": "google_personal", "work": "google_work"},
        default_timeout_seconds=17,
    )


def test_calendar_mcp_tool_lifecycle_logs_are_secret_safe(caplog):
    session = FakeSession(
        [
            result(
                {
                    "calendars": [
                        {
                            "id": "private-calendar-id",
                            "summary": "PRIVATE_RESULT_TEXT",
                        }
                    ]
                }
            ),
            RuntimeError("PRIVATE_PROVIDER_DIAGNOSTIC"),
        ]
    )
    calendar = client(session)

    async def scenario():
        calendars = await calendar.list_calendars("personal")
        assert calendars[0]["summary"] == "PRIVATE_RESULT_TEXT"
        with pytest.raises(CalendarMcpError, match="validation failed"):
            await calendar.list_calendars("personal")

    with caplog.at_level(
        "INFO", logger="tg_voice_transcriber_bot.calendar_mcp"
    ):
        asyncio.run(scenario())

    messages = "\n".join(
        record.getMessage()
        for record in caplog.records
        if record.name == "tg_voice_transcriber_bot.calendar_mcp"
    )
    assert "tool=list-calendars" in messages
    assert "account=google_personal" in messages
    assert "status=started" in messages
    assert "status=success" in messages
    assert "status=transport_error" in messages
    assert "error_type=RuntimeError" in messages
    assert "elapsed=" in messages
    assert "PRIVATE_RESULT_TEXT" not in messages
    assert "private-calendar-id" not in messages
    assert "PRIVATE_PROVIDER_DIAGNOSTIC" not in messages


SNAPSHOT_FIELDS = [
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
]


def test_timed_event_exact_create_payload_and_base32hex_id():
    event = timed_event()
    event_id = google_event_id("confirmation-77", 0)
    session = FakeSession([result({"event": stored_event(event_id, event)})])

    created = asyncio.run(
        client(session).create_events(
            account="personal", events=[event], idempotency_key="confirmation-77"
        )
    )

    assert len(event_id) == 52
    assert set(event_id) <= set("0123456789abcdefghijklmnopqrstuv")
    assert google_event_id("confirmation-77", 0) == event_id
    assert google_event_id("confirmation-77", 1) != event_id
    assert session.calls == [
        (
            "create-event",
            {
                "account": "google_personal",
                "calendarId": "primary",
                "eventId": event_id,
                "summary": "Встреча с Анной",
                "start": "2026-08-24T15:00:00+03:00",
                "end": "2026-08-24T16:00:00+03:00",
                "timeZone": "Europe/Moscow",
                "allowDuplicates": True,
            },
            17.0,
        )
    ]
    assert created[0].event_id == event_id
    assert created[0].html_link == "https://calendar.google.com/event/1"


def test_timed_weekday_recurrence_uses_wall_clock_plus_iana_timezone():
    event = timed_event(
        title="Командный стендап",
        start_at="2027-02-01T09:15:00+03:00",
        end_at="2027-02-01T09:45:00+03:00",
        recurrence_rrule="RRULE:FREQ=WEEKLY;BYDAY=MO,TU,WE,TH,FR",
    )
    event_id = google_event_id("weekday-series", 0)
    provider_event = stored_event(event_id, event)
    provider_event["start"]["timeZone"] = "Europe/Moscow"
    provider_event["end"]["timeZone"] = "Europe/Moscow"
    session = FakeSession([result({"event": provider_event})])

    created = asyncio.run(
        client(session).create_events(
            account="personal", events=[event], idempotency_key="weekday-series"
        )
    )

    assert session.calls[0][1] == {
        "account": "google_personal",
        "calendarId": "primary",
        "eventId": event_id,
        "summary": "Командный стендап",
        "start": "2027-02-01T09:15:00",
        "end": "2027-02-01T09:45:00",
        "timeZone": "Europe/Moscow",
        "allowDuplicates": True,
        "recurrence": ["RRULE:FREQ=WEEKLY;BYDAY=MO,TU,WE,TH,FR"],
    }
    assert created[0].event_id == event_id


def test_create_accepts_google_reordered_rrule_components():
    event = timed_event(
        recurrence_rrule="RRULE:FREQ=WEEKLY;INTERVAL=2;COUNT=7"
    )
    event_id = google_event_id("reordered-create", 0)
    provider_event = stored_event(event_id, event)
    provider_event["recurrence"] = [
        "RRULE:FREQ=WEEKLY;COUNT=7;INTERVAL=2"
    ]
    provider_event["start"]["timeZone"] = "Europe/Moscow"
    provider_event["end"]["timeZone"] = "Europe/Moscow"
    session = FakeSession([result({"event": provider_event})])

    created = asyncio.run(
        client(session).create_events(
            account="personal",
            events=[event],
            idempotency_key="reordered-create",
        )
    )

    assert created[0].event_id == event_id
    assert [call[0] for call in session.calls] == ["create-event"]


def test_create_recovery_accepts_google_reordered_rrule_components():
    event = timed_event(
        recurrence_rrule="RRULE:FREQ=WEEKLY;INTERVAL=2;COUNT=7"
    )
    event_id = google_event_id("reordered-recovery", 0)
    provider_event = stored_event(event_id, event)
    provider_event["recurrence"] = [
        "RRULE:FREQ=WEEKLY;COUNT=7;INTERVAL=2"
    ]
    provider_event["start"]["timeZone"] = "Europe/Moscow"
    provider_event["end"]["timeZone"] = "Europe/Moscow"
    session = FakeSession(
        [result(error=True), result({"event": provider_event})]
    )

    created = asyncio.run(
        client(session).create_events(
            account="personal",
            events=[event],
            idempotency_key="reordered-recovery",
        )
    )

    assert created[0].event_id == event_id
    assert [call[0] for call in session.calls] == ["create-event", "get-event"]


def test_timed_recurrence_recovery_verifies_original_instants():
    event = timed_event(
        start_at="2027-02-01T09:15:00+03:00",
        end_at="2027-02-01T09:45:00+03:00",
        recurrence_rrule="RRULE:FREQ=WEEKLY;BYDAY=MO,TU,WE,TH,FR",
    )
    event_id = google_event_id("weekday-retry", 0)
    provider_event = stored_event(event_id, event)
    provider_event["start"] = {
        "dateTime": "2027-02-01T06:15:00Z",
        "timeZone": "Europe/Moscow",
    }
    provider_event["end"] = {
        "dateTime": "2027-02-01T06:45:00Z",
        "timeZone": "Europe/Moscow",
    }
    session = FakeSession(
        [result(error=True), result({"event": provider_event}, text_fallback=True)]
    )

    created = asyncio.run(
        client(session).create_events(
            account="personal", events=[event], idempotency_key="weekday-retry"
        )
    )

    assert created[0].event_id == event_id
    assert [call[0] for call in session.calls] == ["create-event", "get-event"]
    assert session.calls[0][1]["start"] == "2027-02-01T09:15:00"
    assert session.calls[0][1]["end"] == "2027-02-01T09:45:00"
    assert session.calls[1][1]["eventId"] == event_id


def test_timed_recurrence_converts_instant_to_dst_zone_wall_clock():
    event = timed_event(
        start_at="2026-07-01T08:20:00Z",
        end_at="2026-07-01T08:50:00Z",
        timezone="Europe/Berlin",
        recurrence_rrule="RRULE:FREQ=WEEKLY;BYDAY=WE",
    )
    event_id = google_event_id("dst-series", 0)
    provider_event = stored_event(event_id, event)
    provider_event["start"] = {
        "dateTime": "2026-07-01T10:20:00+02:00",
        "timeZone": "Europe/Berlin",
    }
    provider_event["end"] = {
        "dateTime": "2026-07-01T10:50:00+02:00",
        "timeZone": "Europe/Berlin",
    }
    session = FakeSession([result({"event": provider_event})])

    created = asyncio.run(
        client(session).create_events(
            account="personal", events=[event], idempotency_key="dst-series"
        )
    )

    assert created[0].event_id == event_id
    assert session.calls[0][1]["start"] == "2026-07-01T10:20:00"
    assert session.calls[0][1]["end"] == "2026-07-01T10:50:00"
    assert session.calls[0][1]["timeZone"] == "Europe/Berlin"


def test_all_day_recurrence_maps_exclusive_dates_and_parses_text_content():
    event = timed_event(
        title="Дежурство",
        start_at="2026-08-24",
        end_at="2026-08-25",
        all_day=True,
        description="Командное дежурство",
        location="Удалённо",
        recurrence_rrule="RRULE:FREQ=WEEKLY;BYDAY=MO;COUNT=4",
    )
    event_id = google_event_id("all-day", 0)
    session = FakeSession(
        [result({"event": stored_event(event_id, event)}, text_fallback=True)]
    )

    created = asyncio.run(
        client(session).create_events(
            account="personal", events=[event], idempotency_key="all-day"
        )
    )

    assert session.calls[0][1] == {
        "account": "google_personal",
        "calendarId": "primary",
        "eventId": event_id,
        "summary": "Дежурство",
        "start": "2026-08-24",
        "end": "2026-08-25",
        "timeZone": "Europe/Moscow",
        "allowDuplicates": True,
        "description": "Командное дежурство",
        "location": "Удалённо",
        "recurrence": ["RRULE:FREQ=WEEKLY;BYDAY=MO;COUNT=4"],
    }
    assert created[0].event_id == event_id


def test_duplicate_or_generic_create_error_reads_and_verifies_existing_event():
    event = timed_event(description="Повторный запрос")
    event_id = google_event_id("retry-key", 0)
    session = FakeSession(
        [
            result(error=True),
            result({"event": stored_event(event_id, event)}, text_fallback=True),
        ]
    )

    created = asyncio.run(
        client(session).create_events(
            account="personal", events=[event], idempotency_key="retry-key"
        )
    )

    assert created[0].event_id == event_id
    assert [call[0] for call in session.calls] == ["create-event", "get-event"]
    assert session.calls[1][1] == {
        "account": "google_personal",
        "calendarId": "primary",
        "eventId": event_id,
        "fields": SNAPSHOT_FIELDS,
    }


def test_definite_bad_request_and_exact_missing_id_is_terminal_rejection():
    event = timed_event(
        recurrence_rrule="RRULE:FREQ=WEEKLY;BYDAY=MO,TU,WE,TH,FR"
    )
    session = FakeSession(
        [
            result(
                error=True,
                error_text=(
                    "MCP error -32602: Bad Request: Invalid recurrence timezone"
                ),
            ),
            result(
                error=True,
                error_text=(
                    "MCP error -32603: Internal error: Event with ID '"
                    + google_event_id("definite-rejection", 0)
                    + "' not found in calendar 'primary'."
                ),
            ),
        ]
    )

    with pytest.raises(CalendarMcpWriteRejectedError) as raised:
        asyncio.run(
            client(session).create_events(
                account="personal",
                events=[event],
                idempotency_key="definite-rejection",
            )
        )

    assert str(raised.value) == "Calendar event creation was rejected"
    assert "recurrence" not in str(raised.value)
    assert [call[0] for call in session.calls] == ["create-event", "get-event"]
    assert session.calls[0][1]["eventId"] == session.calls[1][1]["eventId"]


def test_bad_request_with_unknown_probe_remains_ambiguous():
    session = FakeSession(
        [
            result(error=True, error_text="MCP error -32602: Bad Request: Invalid request"),
            result(error=True, error_text="Rate limit exceeded. Try again later."),
        ]
    )

    with pytest.raises(CalendarMcpError) as raised:
        asyncio.run(
            client(session).create_events(
                account="personal",
                events=[timed_event()],
                idempotency_key="ambiguous-probe",
            )
        )

    assert not isinstance(raised.value, CalendarMcpWriteRejectedError)
    assert raised.value.__cause__ is None


def test_unknown_create_failure_plus_missing_id_remains_ambiguous():
    session = FakeSession(
        [
            result(error=True, error_text="Internal error: response unavailable"),
            result(
                error=True,
                error_text=(
                    "MCP error -32603: Internal error: Event with ID '"
                    + google_event_id("ambiguous-create", 0)
                    + "' not found in calendar 'primary'."
                ),
            ),
        ]
    )

    with pytest.raises(CalendarMcpError) as raised:
        asyncio.run(
            client(session).create_events(
                account="personal",
                events=[timed_event()],
                idempotency_key="ambiguous-create",
            )
        )

    assert not isinstance(raised.value, CalendarMcpWriteRejectedError)
    assert raised.value.__cause__ is None


def test_bad_request_recovers_when_deterministic_event_already_exists():
    event = timed_event(description="Уже создано прошлой попыткой")
    event_id = google_event_id("rejected-but-existing", 0)
    session = FakeSession(
        [
            result(error=True, error_text="MCP error -32602: Bad Request: retry rejected"),
            result({"event": stored_event(event_id, event)}),
        ]
    )

    created = asyncio.run(
        client(session).create_events(
            account="personal",
            events=[event],
            idempotency_key="rejected-but-existing",
        )
    )

    assert created[0].event_id == event_id


def test_retry_after_lost_response_uses_same_ids_for_every_batch_item():
    events = [timed_event(), timed_event(title="Вторая встреча")]
    ids = [google_event_id("batch-key", index) for index in range(2)]
    session = FakeSession(
        [
            RuntimeError("response lost after provider write"),
            result({"event": stored_event(ids[0], events[0])}),
            result({"event": stored_event(ids[1], events[1])}),
        ]
    )

    created = asyncio.run(
        client(session).create_events(
            account="personal", events=events, idempotency_key="batch-key"
        )
    )

    assert [item.event_id for item in created] == ids
    assert [call[0] for call in session.calls] == [
        "create-event",
        "get-event",
        "create-event",
    ]
    assert session.calls[0][1]["eventId"] == session.calls[1][1]["eventId"]


def test_existing_event_with_same_id_but_other_payload_is_a_collision():
    requested = timed_event()
    existing = timed_event(title="Другое событие")
    event_id = google_event_id("collision-key", 0)
    session = FakeSession(
        [
            result(error=True),
            result({"event": stored_event(event_id, existing)}),
        ]
    )

    with pytest.raises(CalendarMcpCollisionError, match="collision"):
        asyncio.run(
            client(session).create_events(
                account="personal",
                events=[requested],
                idempotency_key="collision-key",
            )
        )


def test_create_and_recovery_failure_is_sanitized():
    session = FakeSession(
        [
            RuntimeError("secret provider diagnostic"),
            RuntimeError("credential and event details"),
        ]
    )

    with pytest.raises(CalendarMcpError) as raised:
        asyncio.run(
            client(session).create_events(
                account="personal",
                events=[timed_event()],
                idempotency_key="failed-key",
            )
        )

    assert "secret" not in str(raised.value)
    assert "credential" not in str(raised.value)
    assert raised.value.__cause__ is None


def test_unknown_account_fails_without_calling_mcp():
    session = FakeSession([])

    with pytest.raises(CalendarMcpError, match="not configured"):
        asyncio.run(
            client(session).create_events(
                account="unknown",
                events=[timed_event()],
                idempotency_key="unknown-account",
            )
        )
    assert session.calls == []


def test_validate_uses_list_calendars_for_supplied_account_mapping():
    session = FakeSession(
        [
            result({"calendars": [{"id": "a@example.com", "primary": True}]}),
            result({"calendars": [{"id": "b@example.com", "primary": True}]}),
        ]
    )

    asyncio.run(client(session).validate())

    assert [(name, args) for name, args, _ in session.calls] == [
        ("list-calendars", {"account": "google_personal"}),
        ("list-calendars", {"account": "google_work"}),
    ]


def test_get_event_returns_normalized_full_snapshot():
    source = timed_event(
        description="Первая встреча",
        location="Клиника",
        recurrence_rrule="RRULE:FREQ=WEEKLY;COUNT=3",
    )
    provider_event = stored_event("event-42", source)
    provider_event.update(
        {
            "attendees": [{"email": "attendee@example.com"}],
            "recurringEventId": "series-7",
            "originalStartTime": {"dateTime": source["start_at"]},
            "colorId": "5",
            "transparency": "opaque",
            "visibility": "private",
            "eventType": "default",
            "creator": {
                "email": "owner@example.com",
                "displayName": "Owner",
                "self": True,
            },
            "organizer": {
                "email": "organizer@example.com",
                "self": False,
            },
        }
    )
    session = FakeSession([result({"event": provider_event}, text_fallback=True)])

    snapshot = asyncio.run(
        client(session).get_event(account="personal", event_id="event-42")
    )

    assert isinstance(snapshot, CalendarEventSnapshot)
    assert snapshot == CalendarEventSnapshot(
        account="personal",
        calendar_id="primary@example.com",
        event_id="event-42",
        title="Встреча с Анной",
        description="Первая встреча",
        location="Клиника",
        start_at="2026-08-24T15:00:00+03:00",
        end_at="2026-08-24T16:00:00+03:00",
        all_day=False,
        timezone=None,
        status="confirmed",
        html_link="https://calendar.google.com/event/1",
        recurrence_rrules=("RRULE:FREQ=WEEKLY;COUNT=3",),
        attendee_emails=("attendee@example.com",),
        updated_at="2026-08-22T10:15:00Z",
        recurring_event_id="series-7",
        original_start_at="2026-08-24T15:00:00+03:00",
        color_id="5",
        transparency="opaque",
        visibility="private",
        event_type="default",
        creator_email="owner@example.com",
        creator_is_self=True,
        organizer_email="organizer@example.com",
        organizer_is_self=False,
        safety_metadata_complete=True,
        safety_metadata_fingerprint=snapshot.safety_metadata_fingerprint,
    )
    assert session.calls == [
        (
            "get-event",
            {
                "account": "google_personal",
                "calendarId": "primary",
                "eventId": "event-42",
                "fields": SNAPSHOT_FIELDS,
            },
            17.0,
        )
    ]


def test_get_event_normalizes_rich_delete_safety_metadata_without_raw_data():
    provider_event = stored_event("rich-event", timed_event())
    provider_event.update(
        {
            "reminders": {
                "useDefault": False,
                "overrides": [{"method": "popup", "minutes": 10}],
            },
            "conferenceData": {"conferenceId": "private-meeting-id"},
            "hangoutLink": "https://meet.google.com/private-code",
            "attachments": [{"fileId": "private-file-id"}],
            "extendedProperties": {"private": {"secret": "value"}},
            "source": {"url": "https://example.invalid/private"},
            "anyoneCanAddSelf": False,
            "guestsCanInviteOthers": True,
            "guestsCanModify": False,
            "guestsCanSeeOtherGuests": True,
            "privateCopy": False,
            "locked": False,
        }
    )
    session = FakeSession([result({"event": provider_event})])

    snapshot = asyncio.run(
        client(session).get_event(account="personal", event_id="rich-event")
    )

    assert snapshot.reminders_present is True
    assert snapshot.reminders_use_default is False
    assert snapshot.reminder_overrides == (("popup", 10),)
    assert snapshot.has_conference_data is True
    assert snapshot.has_hangout_link is True
    assert snapshot.has_attachments is True
    assert snapshot.has_extended_properties is True
    assert snapshot.has_source is True
    assert snapshot.anyone_can_add_self is False
    assert snapshot.guests_can_invite_others is True
    assert snapshot.guests_can_modify is False
    assert snapshot.guests_can_see_other_guests is True
    assert snapshot.private_copy is False
    assert snapshot.locked is False
    serialized = json.dumps(snapshot.__dict__, ensure_ascii=False)
    assert "private-meeting-id" not in serialized
    assert "private-file-id" not in serialized
    assert "secret" not in serialized
    assert session.calls[0][1]["fields"] == SNAPSHOT_FIELDS


def test_list_events_returns_unique_sorted_bounded_snapshots_with_ownership():
    time_min = "2026-08-24T00:00:00+03:00"
    time_max = "2026-08-27T00:00:00+03:00"
    late = stored_event(
        "late-event",
        timed_event(
            start_at="2026-08-25T14:00:00+03:00",
            end_at="2026-08-25T15:00:00+03:00",
        ),
    )
    early = stored_event(
        "early-event",
        timed_event(
            start_at="2026-08-24T10:00:00+03:00",
            end_at="2026-08-24T11:00:00+03:00",
        ),
    )
    early["creator"] = {"email": "owner@example.com", "self": True}
    early["organizer"] = {"email": "owner@example.com", "self": True}
    tentative = stored_event(
        "tentative-event",
        timed_event(
            start_at="2026-08-24T12:00:00+03:00",
            end_at="2026-08-24T13:00:00+03:00",
        ),
    )
    tentative["status"] = "tentative"
    cancelled = stored_event(
        "cancelled-event",
        timed_event(
            start_at="2026-08-24T11:00:00+03:00",
            end_at="2026-08-24T12:00:00+03:00",
        ),
    )
    cancelled["status"] = "cancelled"
    session = FakeSession(
        [
            result(
                {
                    "events": [late, cancelled, tentative, early],
                    "totalCount": 4,
                },
                text_fallback=True,
            )
        ]
    )

    queried = asyncio.run(
        client(session).list_events(
            account="personal",
            time_min=time_min,
            time_max=time_max,
            limit=2,
        )
    )

    assert isinstance(queried, CalendarEventQueryResult)
    assert [event.event_id for event in queried.events] == [
        "early-event",
        "tentative-event",
    ]
    assert queried.events[0].creator_email == "owner@example.com"
    assert queried.events[0].creator_is_self is True
    assert queried.events[0].organizer_is_self is True
    assert queried.total_count == 4
    assert queried.may_be_incomplete is True
    assert session.calls == [
        (
            "list-events",
            {
                "account": "google_personal",
                "calendarId": "primary",
                "timeMin": time_min,
                "timeMax": time_max,
                "fields": SNAPSHOT_FIELDS,
            },
            17.0,
        )
    ]


@pytest.mark.parametrize("event_type", ["birthday", "fromGmail"])
def test_list_events_accepts_google_read_only_event_types(event_type):
    time_min = "2026-08-24T00:00:00+03:00"
    time_max = "2026-08-27T00:00:00+03:00"
    provider_event = stored_event("special-event", timed_event())
    provider_event["eventType"] = event_type
    session = FakeSession(
        [result({"events": [provider_event], "totalCount": 1})]
    )

    queried = asyncio.run(
        client(session).list_events(
            account="personal",
            time_min=time_min,
            time_max=time_max,
        )
    )

    assert [event.event_type for event in queried.events] == [event_type]


def test_search_events_normalizes_query_and_validates_response_metadata():
    time_min = "2026-08-24T00:00:00+03:00"
    time_max = "2026-08-31T00:00:00+03:00"
    provider_event = stored_event("planning-event", timed_event())
    session = FakeSession(
        [
            result(
                {
                    "events": [provider_event],
                    "totalCount": 1,
                    "query": "планёрка",
                    "calendarId": "primary@example.com",
                    "timeRange": {"start": time_min, "end": time_max},
                }
            )
        ]
    )

    queried = asyncio.run(
        client(session).search_events(
            account="personal",
            query="  планёрка  ",
            time_min=time_min,
            time_max=time_max,
        )
    )

    assert [event.event_id for event in queried.events] == ["planning-event"]
    assert queried.total_count == 1
    assert queried.may_be_incomplete is False
    assert session.calls == [
        (
            "search-events",
            {
                "account": "google_personal",
                "calendarId": "primary",
                "timeMin": time_min,
                "timeMax": time_max,
                "fields": SNAPSHOT_FIELDS,
                "query": "планёрка",
            },
            17.0,
        )
    ]


@pytest.mark.parametrize(
    ("method_name", "arguments"),
    [
        (
            "list_events",
            {
                "time_min": "2026-08-24T00:00:00",
                "time_max": "2026-08-25T00:00:00+03:00",
            },
        ),
        (
            "list_events",
            {
                "time_min": "2026-08-25T00:00:00+03:00",
                "time_max": "2026-08-24T00:00:00+03:00",
            },
        ),
        (
            "list_events",
            {
                "time_min": "2026-08-01T00:00:00+03:00",
                "time_max": "2026-09-02T00:00:00+03:00",
            },
        ),
        (
            "list_events",
            {
                "time_min": "2026-08-24T00:00:00+03:00",
                "time_max": "2026-08-25T00:00:00+03:00",
                "limit": 0,
            },
        ),
        (
            "list_events",
            {
                "time_min": "2026-08-24T00:00:00+03:00",
                "time_max": "2026-08-25T00:00:00+03:00",
                "limit": 251,
            },
        ),
        (
            "search_events",
            {
                "query": "   ",
                "time_min": "2026-08-24T00:00:00+03:00",
                "time_max": "2026-08-25T00:00:00+03:00",
            },
        ),
        (
            "search_events",
            {
                "query": "unsafe\nquery",
                "time_min": "2026-08-24T00:00:00+03:00",
                "time_max": "2026-08-25T00:00:00+03:00",
            },
        ),
    ],
)
def test_discovery_rejects_invalid_ranges_limits_and_queries_without_mcp(
    method_name, arguments
):
    session = FakeSession([])

    with pytest.raises(CalendarMcpError, match="query"):
        asyncio.run(
            getattr(client(session), method_name)(account="personal", **arguments)
        )

    assert session.calls == []


@pytest.mark.parametrize(
    "payload",
    [
        {"events": [], "totalCount": 0, "warnings": ["secret event title"]},
        {
            "events": [],
            "totalCount": 0,
            "partialFailures": [
                {"accountId": "owner", "reason": "secret credential detail"}
            ],
        },
        {"events": [], "totalCount": 1},
        {
            "events": [
                stored_event("duplicate-event", timed_event()),
                stored_event("duplicate-event", timed_event()),
            ],
            "totalCount": 2,
        },
    ],
)
def test_discovery_rejects_untrusted_partial_malformed_or_duplicate_payloads(
    payload,
):
    session = FakeSession([result(payload)])

    with pytest.raises(CalendarMcpError, match="discovery failed") as raised:
        asyncio.run(
            client(session).list_events(
                account="personal",
                time_min="2026-08-24T00:00:00+03:00",
                time_max="2026-08-25T00:00:00+03:00",
            )
        )

    assert "secret" not in str(raised.value)
    assert raised.value.__cause__ is None


def test_discovery_marks_a_saturated_provider_page_incomplete():
    events = [
        stored_event(f"event-{index}", timed_event())
        for index in range(250)
    ]
    session = FakeSession([result({"events": events, "totalCount": 250})])

    queried = asyncio.run(
        client(session).list_events(
            account="personal",
            time_min="2026-08-24T00:00:00+03:00",
            time_max="2026-08-25T00:00:00+03:00",
            limit=250,
        )
    )

    assert len(queried.events) == 250
    assert queried.total_count == 250
    assert queried.may_be_incomplete is True


@pytest.mark.parametrize(
    "failure",
    [
        anyio.ClosedResourceError(),
        EOFError("private EOF diagnostic"),
        OSError("private stdio diagnostic"),
        McpError(ErrorData(code=-32000, message="Connection closed")),
        McpError(ErrorData(code=408, message="Timed out waiting for response")),
        ExceptionGroup(
            "private task group detail",
            [RuntimeError("wrapper"), anyio.BrokenResourceError()],
        ),
    ],
)
def test_discovery_classifies_dead_or_timed_out_stdio_as_fatal(failure):
    session = FakeSession([failure])

    with pytest.raises(CalendarMcpConnectionError) as raised:
        asyncio.run(
            client(session).list_events(
                account="personal",
                time_min="2026-08-24T00:00:00+03:00",
                time_max="2026-08-25T00:00:00+03:00",
            )
        )

    assert isinstance(raised.value, CalendarConnectionError)
    assert str(raised.value) == "Calendar MCP connection is unavailable"
    assert raised.value.__cause__ is None


def test_normal_tool_failure_remains_a_sanitized_retryable_calendar_error():
    session = FakeSession([RuntimeError("private provider diagnostic")])

    with pytest.raises(CalendarMcpError, match="discovery failed") as raised:
        asyncio.run(
            client(session).list_events(
                account="personal",
                time_min="2026-08-24T00:00:00+03:00",
                time_max="2026-08-25T00:00:00+03:00",
            )
        )

    assert not isinstance(raised.value, CalendarConnectionError)
    assert "private" not in str(raised.value)
    assert raised.value.__cause__ is None


def test_update_event_sends_safe_patch_and_parses_provider_state():
    before = timed_event(location=None)
    after = timed_event(location="переговорная А")
    session = FakeSession(
        [
            result({"event": stored_event("event-42", before)}),
            result({"event": stored_event("event-42", after)}),
        ]
    )

    updated = asyncio.run(
        client(session).update_event(
            account="personal",
            event_id="event-42",
            patch={"location": "переговорная А"},
            idempotency_key="operation-update-1",
        )
    )

    assert updated.previous.location is None
    assert updated.current.location == "переговорная А"
    assert updated.already_applied is False
    assert [call[:2] for call in session.calls] == [
        (
            "get-event",
            {
                "account": "google_personal",
                "calendarId": "primary",
                "eventId": "event-42",
                "fields": SNAPSHOT_FIELDS,
            },
        ),
        (
            "update-event",
            {
                "account": "google_personal",
                "calendarId": "primary",
                "eventId": "event-42",
                "sendUpdates": "none",
                "location": "переговорная А",
            },
        ),
    ]


def test_update_event_is_noop_when_patch_is_already_applied():
    after = timed_event(location="переговорная А")
    session = FakeSession([result({"event": stored_event("event-42", after)})])

    updated = asyncio.run(
        client(session).update_event(
            account="personal",
            event_id="event-42",
            patch={"location": "переговорная А"},
            idempotency_key="operation-update-retry",
        )
    )

    assert updated.previous.location == "переговорная А"
    assert updated.current is updated.previous
    assert updated.already_applied is True
    assert [call[0] for call in session.calls] == ["get-event"]


def test_update_event_does_not_treat_equivalent_instants_in_another_timezone_as_noop():
    before = timed_event(
        start_at="2026-08-24T07:00:00+00:00",
        end_at="2026-08-24T08:00:00+00:00",
        timezone="UTC",
    )
    after = timed_event(
        start_at="2026-08-24T10:00:00+03:00",
        end_at="2026-08-24T11:00:00+03:00",
        timezone="Europe/Moscow",
    )
    before_provider = stored_event("event-42", before)
    before_provider["start"]["timeZone"] = "UTC"
    before_provider["end"]["timeZone"] = "UTC"
    after_provider = stored_event("event-42", after)
    after_provider["start"]["timeZone"] = "Europe/Moscow"
    after_provider["end"]["timeZone"] = "Europe/Moscow"
    session = FakeSession(
        [
            result({"event": before_provider}),
            result({"event": after_provider}),
        ]
    )

    updated = asyncio.run(
        client(session).update_event(
            account="personal",
            event_id="event-42",
            patch={
                "start_at": after["start_at"],
                "end_at": after["end_at"],
                "timezone": after["timezone"],
            },
            idempotency_key="operation-update-timezone",
        )
    )

    assert updated.already_applied is False
    assert updated.previous.timezone == "UTC"
    assert updated.current.timezone == "Europe/Moscow"
    assert [call[0] for call in session.calls] == ["get-event", "update-event"]
    assert session.calls[1][1]["start"] == "2026-08-24T10:00:00+03:00"
    assert session.calls[1][1]["end"] == "2026-08-24T11:00:00+03:00"
    assert session.calls[1][1]["timeZone"] == "Europe/Moscow"


def test_update_event_timezone_verification_recovers_only_matching_provider_zone():
    before = timed_event(
        start_at="2026-08-24T07:00:00+00:00",
        end_at="2026-08-24T08:00:00+00:00",
        timezone="UTC",
    )
    requested = timed_event(
        start_at="2026-08-24T10:00:00+03:00",
        end_at="2026-08-24T11:00:00+03:00",
        timezone="Europe/Moscow",
    )
    before_provider = stored_event("event-42", before)
    before_provider["start"]["timeZone"] = "UTC"
    before_provider["end"]["timeZone"] = "UTC"
    wrong_update_response = stored_event("event-42", requested)
    wrong_update_response["start"]["timeZone"] = "UTC"
    wrong_update_response["end"]["timeZone"] = "UTC"
    recovered_provider = stored_event("event-42", requested)
    recovered_provider["start"]["timeZone"] = "Europe/Moscow"
    recovered_provider["end"]["timeZone"] = "Europe/Moscow"
    session = FakeSession(
        [
            result({"event": before_provider}),
            result({"event": wrong_update_response}),
            result({"event": recovered_provider}),
        ]
    )

    updated = asyncio.run(
        client(session).update_event(
            account="personal",
            event_id="event-42",
            patch={
                "start_at": requested["start_at"],
                "end_at": requested["end_at"],
                "timezone": requested["timezone"],
            },
            idempotency_key="operation-update-timezone-recovery",
        )
    )

    assert updated.already_applied is False
    assert updated.current.timezone == "Europe/Moscow"
    assert [call[0] for call in session.calls] == [
        "get-event",
        "update-event",
        "get-event",
    ]


def test_update_event_reconciles_applied_patch_before_stale_precondition():
    expected = normalized_snapshot(
        "event-42", timed_event(location="Состояние до попытки")
    )
    applied = timed_event(location="Уже применено")
    session = FakeSession([result({"event": stored_event("event-42", applied)})])

    updated = asyncio.run(
        client(session).update_event(
            account="personal",
            event_id="event-42",
            patch={"location": "Уже применено"},
            idempotency_key="operation-update-lost-response-retry",
            expected_current=expected,
        )
    )

    assert updated.already_applied is True
    assert updated.current.location == "Уже применено"
    assert [call[0] for call in session.calls] == ["get-event"]


def test_update_event_precondition_conflict_stops_before_provider_write():
    expected = normalized_snapshot(
        "event-42", timed_event(location="Изменено ботом")
    )
    provider_current = timed_event(location="Ручная правка после проверки")
    session = FakeSession(
        [result({"event": stored_event("event-42", provider_current)})]
    )

    with pytest.raises(CalendarStateConflictError) as raised:
        asyncio.run(
            client(session).update_event(
                account="personal",
                event_id="event-42",
                patch={"location": "До изменения ботом"},
                idempotency_key="operation-update-conflict",
                expected_current=expected,
            )
        )

    assert str(raised.value) == "Calendar event changed after it was observed"
    assert raised.value.__cause__ is None
    assert [call[0] for call in session.calls] == ["get-event"]


def test_update_event_precondition_ignores_link_and_provider_update_time():
    source = timed_event(location="Изменено ботом")
    expected = normalized_snapshot("event-42", source)
    provider_current = stored_event(
        "event-42",
        source,
        html_link="https://calendar.google.com/event/new-link",
    )
    provider_current["updated"] = "2026-08-24T12:00:00Z"
    provider_after = stored_event(
        "event-42",
        timed_event(location="До изменения ботом"),
        html_link="https://calendar.google.com/event/new-link",
    )
    provider_after["updated"] = "2026-08-24T12:00:01Z"
    session = FakeSession(
        [
            result({"event": provider_current}),
            result({"event": provider_after}),
        ]
    )

    updated = asyncio.run(
        client(session).update_event(
            account="personal",
            event_id="event-42",
            patch={"location": "До изменения ботом"},
            idempotency_key="operation-update-bookkeeping-only",
            expected_current=expected,
        )
    )

    assert updated.current.location == "До изменения ботом"
    assert [call[0] for call in session.calls] == ["get-event", "update-event"]


def test_update_event_moves_both_bounds_with_explicit_timezone():
    before = timed_event()
    after = timed_event(
        start_at="2026-08-26T15:00:00+03:00",
        end_at="2026-08-26T16:00:00+03:00",
    )
    after_provider = stored_event("event-42", after)
    after_provider["start"]["timeZone"] = "Europe/Moscow"
    after_provider["end"]["timeZone"] = "Europe/Moscow"
    session = FakeSession(
        [
            result({"event": stored_event("event-42", before)}),
            result({"event": after_provider}),
        ]
    )

    updated = asyncio.run(
        client(session).update_event(
            account="personal",
            event_id="event-42",
            patch={
                "start_at": "2026-08-26T15:00:00+03:00",
                "end_at": "2026-08-26T16:00:00+03:00",
                "timezone": "Europe/Moscow",
            },
            idempotency_key="operation-move-1",
        )
    )

    assert updated.previous.start_at == "2026-08-24T15:00:00+03:00"
    assert updated.current.start_at == "2026-08-26T15:00:00+03:00"
    assert updated.already_applied is False
    assert session.calls[1][1] == {
        "account": "google_personal",
        "calendarId": "primary",
        "eventId": "event-42",
        "sendUpdates": "none",
        "start": "2026-08-26T15:00:00+03:00",
        "end": "2026-08-26T16:00:00+03:00",
        "timeZone": "Europe/Moscow",
    }


def test_update_event_replaces_series_recurrence_with_explicit_all_scope():
    before = timed_event(recurrence_rrule="RRULE:FREQ=WEEKLY;BYDAY=MO,TU,WE,TH,FR")
    after = timed_event(recurrence_rrule="RRULE:FREQ=WEEKLY;BYDAY=MO,WE,FR")
    session = FakeSession(
        [
            result({"event": stored_event("event-42", before)}),
            result({"event": stored_event("event-42", after)}),
        ]
    )

    updated = asyncio.run(
        client(session).update_event(
            account="personal",
            event_id="event-42",
            patch={
                "recurrence_rrules": [
                    "RRULE:FREQ=WEEKLY;BYDAY=MO,WE,FR"
                ]
            },
            idempotency_key="operation-recurrence-1",
        )
    )

    assert updated.current.recurrence_rrules == (
        "RRULE:FREQ=WEEKLY;BYDAY=MO,WE,FR",
    )
    assert updated.already_applied is False
    assert session.calls[1][1] == {
        "account": "google_personal",
        "calendarId": "primary",
        "eventId": "event-42",
        "sendUpdates": "none",
        "recurrence": ["RRULE:FREQ=WEEKLY;BYDAY=MO,WE,FR"],
        "modificationScope": "all",
    }


def test_update_accepts_google_reordered_rrule_components():
    before = timed_event(recurrence_rrule="RRULE:FREQ=WEEKLY;COUNT=4")
    requested_rule = "RRULE:FREQ=WEEKLY;INTERVAL=2;COUNT=7"
    after = timed_event(recurrence_rrule=requested_rule)
    provider_after = stored_event("event-42", after)
    provider_after["recurrence"] = [
        "RRULE:FREQ=WEEKLY;COUNT=7;INTERVAL=2"
    ]
    session = FakeSession(
        [
            result({"event": stored_event("event-42", before)}),
            result({"event": provider_after}),
        ]
    )

    updated = asyncio.run(
        client(session).update_event(
            account="personal",
            event_id="event-42",
            patch={"recurrence_rrules": [requested_rule]},
            idempotency_key="operation-reordered-recurrence",
        )
    )

    assert updated.current.recurrence_rrules == (
        "RRULE:FREQ=WEEKLY;COUNT=7;INTERVAL=2",
    )
    assert updated.already_applied is False


def test_rrule_comparison_still_rejects_a_different_count():
    event = timed_event(
        recurrence_rrule="RRULE:FREQ=WEEKLY;INTERVAL=2;COUNT=7"
    )
    event_id = google_event_id("different-count", 0)
    provider_event = stored_event(event_id, event)
    provider_event["recurrence"] = [
        "RRULE:FREQ=WEEKLY;COUNT=8;INTERVAL=2"
    ]
    provider_event["start"]["timeZone"] = "Europe/Moscow"
    provider_event["end"]["timeZone"] = "Europe/Moscow"
    session = FakeSession(
        [result(error=True), result({"event": provider_event})]
    )

    with pytest.raises(CalendarMcpCollisionError):
        asyncio.run(
            client(session).create_events(
                account="personal",
                events=[event],
                idempotency_key="different-count",
            )
        )


def test_update_event_clears_series_recurrence_with_an_empty_array():
    before = timed_event(recurrence_rrule="RRULE:FREQ=DAILY")
    after = timed_event(recurrence_rrule=None)
    session = FakeSession(
        [
            result({"event": stored_event("event-42", before)}),
            result({"event": stored_event("event-42", after)}),
        ]
    )

    updated = asyncio.run(
        client(session).update_event(
            account="personal",
            event_id="event-42",
            patch={"recurrence_rrules": []},
            idempotency_key="operation-recurrence-clear",
        )
    )

    assert updated.previous.recurrence_rrules == ("RRULE:FREQ=DAILY",)
    assert updated.current.recurrence_rrules == ()
    assert updated.already_applied is False
    assert session.calls[1][1]["recurrence"] == []
    assert session.calls[1][1]["modificationScope"] == "all"


def test_update_event_recovers_after_lost_response_by_probing_target_state():
    before = timed_event(location="Клиника")
    after = timed_event(location="переговорная А")
    session = FakeSession(
        [
            result({"event": stored_event("event-42", before)}),
            RuntimeError("provider response lost"),
            result({"event": stored_event("event-42", after)}),
        ]
    )

    updated = asyncio.run(
        client(session).update_event(
            account="personal",
            event_id="event-42",
            patch={"location": "переговорная А"},
            idempotency_key="operation-update-lost",
        )
    )

    assert updated.previous.location == "Клиника"
    assert updated.current.location == "переговорная А"
    assert updated.already_applied is False
    assert [call[0] for call in session.calls] == [
        "get-event",
        "update-event",
        "get-event",
    ]


def test_write_recovery_does_not_swallow_a_fatal_probe_connection_error():
    before = timed_event(location="Клиника")
    session = FakeSession(
        [
            result({"event": stored_event("event-42", before)}),
            RuntimeError("ambiguous provider response"),
            anyio.EndOfStream(),
        ]
    )

    with pytest.raises(CalendarMcpConnectionError):
        asyncio.run(
            client(session).update_event(
                account="personal",
                event_id="event-42",
                patch={"location": "переговорная А"},
                idempotency_key="operation-update-dead-probe",
            )
        )

    assert [call[0] for call in session.calls] == [
        "get-event",
        "update-event",
        "get-event",
    ]


@pytest.mark.parametrize(
    "patch",
    [
        {},
        {"unknown": "value"},
        {"start_at": "2026-08-25T15:00:00+03:00"},
        {
            "start_at": "2026-08-25T15:00:00+03:00",
            "end_at": "2026-08-25",
        },
        {"timezone": "Europe/Moscow"},
    ],
)
def test_update_event_rejects_unsafe_patch_without_provider_call(patch):
    session = FakeSession([])

    with pytest.raises(CalendarMcpError, match="patch"):
        asyncio.run(
            client(session).update_event(
                account="personal",
                event_id="event-42",
                patch=patch,
                idempotency_key="operation-invalid",
            )
        )

    assert session.calls == []


def test_update_event_rejects_unverified_provider_result_safely():
    before = timed_event(location="Клиника")
    wrong = timed_event(location="Другой адрес")
    session = FakeSession(
        [
            result({"event": stored_event("event-42", before)}),
            result({"event": stored_event("event-42", wrong)}),
            result({"event": stored_event("event-42", wrong)}),
        ]
    )

    with pytest.raises(CalendarMcpError, match="update failed") as raised:
        asyncio.run(
            client(session).update_event(
                account="personal",
                event_id="event-42",
                patch={"location": "переговорная А"},
                idempotency_key="operation-mismatch",
            )
        )

    assert raised.value.__cause__ is None


def test_delete_event_sends_no_notifications_and_verifies_cancelled_state():
    source = timed_event(location="переговорная А")
    before = stored_event("event-42", source)
    cancelled = {**before, "status": "cancelled"}
    session = FakeSession(
        [
            result({"event": before}),
            result(
                {
                    "success": True,
                    "eventId": "event-42",
                    "calendarId": "primary@example.com",
                    "message": "Event deleted successfully",
                }
            ),
            result({"event": cancelled}),
        ]
    )

    deleted = asyncio.run(
        client(session).delete_event(
            account="personal",
            event_id="event-42",
            idempotency_key="operation-delete-1",
        )
    )

    assert isinstance(deleted, DeletedCalendarEvent)
    assert deleted.previous.location == "переговорная А"
    assert deleted.current is not None
    assert deleted.current.status == "cancelled"
    assert deleted.already_deleted is False
    assert deleted.verified_cancelled is True
    assert [call[:2] for call in session.calls] == [
        (
            "get-event",
            {
                "account": "google_personal",
                "calendarId": "primary",
                "eventId": "event-42",
                "fields": SNAPSHOT_FIELDS,
            },
        ),
        (
            "delete-event",
            {
                "account": "google_personal",
                "calendarId": "primary",
                "eventId": "event-42",
                "sendUpdates": "none",
            },
        ),
        (
            "get-event",
            {
                "account": "google_personal",
                "calendarId": "primary",
                "eventId": "event-42",
                "fields": SNAPSHOT_FIELDS,
            },
        ),
    ]


def test_delete_event_retry_skips_second_delete_for_cancelled_event():
    provider_event = stored_event("event-42", timed_event())
    provider_event["status"] = "cancelled"
    session = FakeSession([result({"event": provider_event})])
    expected = normalized_snapshot("event-42", timed_event())

    deleted = asyncio.run(
        client(session).delete_event(
            account="personal",
            event_id="event-42",
            idempotency_key="operation-delete-retry",
            expected_current=expected,
        )
    )

    assert deleted.already_deleted is True
    assert deleted.verified_cancelled is True
    assert [call[0] for call in session.calls] == ["get-event"]


def test_delete_event_precondition_conflict_stops_before_provider_write():
    expected = normalized_snapshot(
        "event-42", timed_event(description="Состояние после бота")
    )
    provider_current = timed_event(description="Ручная правка после проверки")
    session = FakeSession(
        [result({"event": stored_event("event-42", provider_current)})]
    )

    with pytest.raises(CalendarStateConflictError) as raised:
        asyncio.run(
            client(session).delete_event(
                account="personal",
                event_id="event-42",
                idempotency_key="operation-delete-conflict",
                expected_current=expected,
            )
        )

    assert str(raised.value) == "Calendar event changed after it was observed"
    assert raised.value.__cause__ is None
    assert [call[0] for call in session.calls] == ["get-event"]


def test_delete_event_allows_tentative_target():
    before = stored_event("event-42", timed_event())
    before["status"] = "tentative"
    cancelled = {**before, "status": "cancelled"}
    session = FakeSession(
        [
            result({"event": before}),
            result(
                {
                    "success": True,
                    "eventId": "event-42",
                    "calendarId": "primary@example.com",
                    "message": "Event deleted successfully",
                }
            ),
            result({"event": cancelled}),
        ]
    )

    deleted = asyncio.run(
        client(session).delete_event(
            account="personal",
            event_id="event-42",
            idempotency_key="operation-delete-tentative",
        )
    )

    assert deleted.previous.status == "tentative"
    assert deleted.verified_cancelled is True
    assert [call[0] for call in session.calls] == [
        "get-event",
        "delete-event",
        "get-event",
    ]


def test_delete_event_recovers_when_delete_response_is_lost():
    before = stored_event("event-42", timed_event())
    cancelled = {**before, "status": "cancelled"}
    session = FakeSession(
        [
            result({"event": before}),
            RuntimeError("provider response lost"),
            result({"event": cancelled}),
        ]
    )

    deleted = asyncio.run(
        client(session).delete_event(
            account="personal",
            event_id="event-42",
            idempotency_key="operation-delete-lost",
        )
    )

    assert deleted.verified_cancelled is True
    assert [call[0] for call in session.calls] == [
        "get-event",
        "delete-event",
        "get-event",
    ]


def test_enabled_tools_include_read_update_and_delete_boundaries():
    assert set(CALENDAR_MCP_TOOLS) == {
        "create-event",
        "delete-event",
        "get-event",
        "list-calendars",
        "list-events",
        "search-events",
        "update-event",
    }


def test_open_calendar_mcp_passes_exact_stdio_command_cwd_and_environment(
    tmp_path, monkeypatch
):
    binary = tmp_path / "node_modules" / ".bin" / "google-calendar-mcp"
    binary.parent.mkdir(parents=True)
    binary.write_text("#!/bin/sh\n", encoding="utf-8")
    binary.chmod(0o700)
    captured = {}

    @asynccontextmanager
    async def fake_stdio_client(parameters, *, errlog):
        captured["parameters"] = parameters
        captured["stderr_name"] = errlog.name
        yield object(), object()

    class FakeClientSessionContext:
        def __init__(self, reader, writer):
            self.initialized = False

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, traceback):
            return False

        async def initialize(self):
            self.initialized = True
            captured["initialized"] = True

    monkeypatch.setattr(calendar_mcp_module, "stdio_client", fake_stdio_client)
    monkeypatch.setattr(
        calendar_mcp_module, "ClientSession", FakeClientSessionContext
    )
    environment = {
        "GOOGLE_OAUTH_CREDENTIALS": "/private/oauth.json",
        "GOOGLE_CALENDAR_MCP_TOKEN_PATH": "/private/tokens.json",
        "GOOGLE_ACCOUNT_MODE": "owner",
        "PATH": "/opt/homebrew/bin:/usr/bin:/bin:/usr/sbin:/sbin",
    }

    async def scenario():
        async with calendar_mcp_module.open_calendar_mcp(
            binary,
            account_mapping={"personal": "owner", "work": "owner"},
            working_directory=tmp_path,
            env=environment,
        ) as opened:
            assert isinstance(opened, GoogleCalendarMcpClient)

    asyncio.run(scenario())

    parameters = captured["parameters"]
    assert parameters.command == str(binary.resolve())
    assert parameters.args == [
        "start",
        "--enable-tools",
        ",".join(CALENDAR_MCP_TOOLS),
    ]
    assert parameters.cwd == tmp_path.resolve()
    assert parameters.env == environment
    assert captured["initialized"] is True


def test_open_calendar_mcp_classifies_nested_startup_transport_failure(
    tmp_path, monkeypatch
):
    binary = tmp_path / "google-calendar-mcp"
    binary.write_text("#!/bin/sh\n", encoding="utf-8")
    binary.chmod(0o700)

    @asynccontextmanager
    async def fake_stdio_client(parameters, *, errlog):
        del parameters, errlog
        yield object(), object()

    class DeadClientSessionContext:
        def __init__(self, reader, writer):
            del reader, writer

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, traceback):
            del exc_type, exc, traceback
            return False

        async def initialize(self):
            raise ExceptionGroup(
                "private task group detail", [anyio.ClosedResourceError()]
            )

    monkeypatch.setattr(calendar_mcp_module, "stdio_client", fake_stdio_client)
    monkeypatch.setattr(
        calendar_mcp_module, "ClientSession", DeadClientSessionContext
    )

    async def scenario():
        async with calendar_mcp_module.open_calendar_mcp(
            binary,
            account_mapping={"personal": "owner"},
            working_directory=tmp_path,
        ):
            raise AssertionError("dead MCP must not yield a client")

    with pytest.raises(CalendarMcpConnectionError) as raised:
        asyncio.run(scenario())

    assert str(raised.value) == "Calendar MCP connection is unavailable"
    assert raised.value.__cause__ is None

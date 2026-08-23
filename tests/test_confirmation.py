import asyncio
from datetime import datetime, timedelta, timezone
import stat

import pytest

from tg_voice_transcriber_bot.calendar import (
    CalendarConnectionError,
    CreatedCalendarEvent,
)
from tg_voice_transcriber_bot.confirmation import (
    CalendarConfirmationPipeline,
    ConfirmationStore,
    confirmation_reply_markup,
    parse_calendar_callback,
)


def calendar_intent(*, confidence=0.91, action="create"):
    event = {
        "title": "Созвон",
        "start_at": "2026-08-24T10:00:00+03:00",
        "end_at": "2026-08-24T11:00:00+03:00",
        "all_day": False,
        "timezone": "Europe/Moscow",
        "location": None,
        "description": None,
        "recurrence_rrule": None,
    }
    return {
        "action": action,
        "events": [event] if action == "create" else [],
        "clarification_question": "Когда?" if action == "clarify" else None,
        "confidence": confidence,
    }


class FakeCalendarClient:
    """In-memory adapter that implements the protocol's idempotency contract."""

    def __init__(self):
        self.calls = 0
        self.created_batches = {}
        self.fail_next = False
        self.fatal_next = False

    async def create_events(self, *, account, events, idempotency_key):
        self.calls += 1
        if self.fatal_next:
            self.fatal_next = False
            raise CalendarConnectionError("sanitized fatal transport")
        if self.fail_next:
            self.fail_next = False
            raise RuntimeError("provider details that must not reach state")
        if idempotency_key not in self.created_batches:
            self.created_batches[idempotency_key] = tuple(
                CreatedCalendarEvent(f"fake-{index}")
                for index, _event in enumerate(events, start=1)
            )
        return self.created_batches[idempotency_key]


def prepare(pipeline, *, source_update_id=77, confidence=0.91):
    return pipeline.prepare(
        source_update_id=source_update_id,
        account="personal",
        owner_user_id=100000001,
        chat_id=100000001,
        intent=calendar_intent(confidence=confidence),
    )


def test_confidence_gate_and_persistent_source_idempotency(tmp_path):
    path = tmp_path / "calendar-confirmations.json"
    calendar = FakeCalendarClient()
    pipeline = CalendarConfirmationPipeline(ConfirmationStore(path), calendar)

    assert prepare(pipeline, source_update_id=1, confidence=0.8499) is None
    assert not path.exists()

    prepared = prepare(pipeline, source_update_id=2, confidence=0.85)
    assert prepared is not None
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    buttons = prepared.reply_markup["inline_keyboard"][0]
    assert [button["text"] for button in buttons] == ["Добавить", "Отмена"]
    assert all(len(button["callback_data"].encode()) <= 64 for button in buttons)

    # A restart or repeated delivery of the Telegram update reuses the same
    # durable record and callback ID instead of generating another operation.
    restored = CalendarConfirmationPipeline(ConfirmationStore(path), calendar)
    repeated = prepare(restored, source_update_id=2, confidence=0.85)
    assert repeated == prepared


def test_only_exact_callback_allowlist_is_accepted():
    confirmation_id = "abcdefghijklmnop"
    markup = confirmation_reply_markup(confirmation_id)
    assert parse_calendar_callback(
        markup["inline_keyboard"][0][0]["callback_data"]
    ) == ("add", confirmation_id)
    assert parse_calendar_callback(f"cal:cancel:{confirmation_id}") == (
        "cancel",
        confirmation_id,
    )
    assert parse_calendar_callback(f"cal:delete:{confirmation_id}") is None
    assert parse_calendar_callback(f"cal:add:{confirmation_id}:admin") is None
    assert parse_calendar_callback(f"other:add:{confirmation_id}") is None
    assert parse_calendar_callback("cal:add:short") is None
    assert parse_calendar_callback("x" * 65) is None
    assert parse_calendar_callback("\ud800") is None
    assert parse_calendar_callback(None) is None


def test_owner_bound_add_is_exactly_once_for_duplicate_callbacks(tmp_path):
    async def scenario():
        calendar = FakeCalendarClient()
        store = ConfirmationStore(tmp_path / "confirmations.json")
        pipeline = CalendarConfirmationPipeline(store, calendar)
        prepared = prepare(pipeline)
        data = prepared.reply_markup["inline_keyboard"][0][0]["callback_data"]

        rejected = await pipeline.handle_callback(
            data=data,
            owner_user_id=100000002,
            chat_id=100000002,
        )
        first = await pipeline.handle_callback(
            data=data,
            owner_user_id=100000001,
            chat_id=100000001,
        )
        duplicate = await pipeline.handle_callback(
            data=data,
            owner_user_id=100000001,
            chat_id=100000001,
        )
        return calendar, store, rejected, first, duplicate, prepared

    calendar, store, rejected, first, duplicate, prepared = asyncio.run(scenario())
    assert rejected.outcome == "rejected"
    assert first.outcome == "created"
    assert first.remove_keyboard is True
    assert duplicate.outcome == "already_created"
    assert calendar.calls == 1
    assert len(calendar.created_batches) == 1
    record = store.get(prepared.confirmation_id)
    assert record["stage"] == "created"
    assert record["attempts"] == 1
    assert record["calendar_events"] == [
        {"event_id": "fake-1", "html_link": None}
    ]


def test_cancel_is_terminal_and_never_calls_calendar(tmp_path):
    async def scenario():
        calendar = FakeCalendarClient()
        pipeline = CalendarConfirmationPipeline(
            ConfirmationStore(tmp_path / "confirmations.json"), calendar
        )
        prepared = prepare(pipeline)
        cancel_data = prepared.reply_markup["inline_keyboard"][0][1]["callback_data"]
        add_data = prepared.reply_markup["inline_keyboard"][0][0]["callback_data"]
        cancelled = await pipeline.handle_callback(
            data=cancel_data,
            owner_user_id=100000001,
            chat_id=100000001,
        )
        replay = await pipeline.handle_callback(
            data=add_data,
            owner_user_id=100000001,
            chat_id=100000001,
        )
        return calendar, cancelled, replay

    calendar, cancelled, replay = asyncio.run(scenario())
    assert cancelled.outcome == "cancelled"
    assert cancelled.remove_keyboard is True
    assert replay.outcome == "already_cancelled"
    assert calendar.calls == 0


def test_expired_confirmation_is_rejected_before_write(tmp_path):
    async def scenario():
        current = datetime(2026, 8, 22, 12, tzinfo=timezone.utc)

        def now():
            return current

        calendar = FakeCalendarClient()
        pipeline = CalendarConfirmationPipeline(
            ConfirmationStore(tmp_path / "confirmations.json"),
            calendar,
            confirmation_ttl=timedelta(minutes=5),
            now=now,
        )
        prepared = prepare(pipeline)
        current += timedelta(minutes=6)
        result = await pipeline.handle_callback(
            data=prepared.reply_markup["inline_keyboard"][0][0]["callback_data"],
            owner_user_id=100000001,
            chat_id=100000001,
        )
        return calendar, result

    calendar, result = asyncio.run(scenario())
    assert result.outcome == "expired"
    assert result.remove_keyboard is True
    assert calendar.calls == 0


def test_retry_after_unknown_provider_result_reuses_idempotency_key(tmp_path):
    async def scenario():
        calendar = FakeCalendarClient()
        calendar.fail_next = True
        path = tmp_path / "confirmations.json"
        first_pipeline = CalendarConfirmationPipeline(
            ConfirmationStore(path), calendar
        )
        prepared = prepare(first_pipeline)
        data = prepared.reply_markup["inline_keyboard"][0][0]["callback_data"]
        failed = await first_pipeline.handle_callback(
            data=data,
            owner_user_id=100000001,
            chat_id=100000001,
        )

        # Simulate a restart while the durable stage is ``creating``.  Retrying
        # is permitted only with the same key, which the fake deduplicates.
        restored = CalendarConfirmationPipeline(ConfirmationStore(path), calendar)
        retried = await restored.handle_callback(
            data=data,
            owner_user_id=100000001,
            chat_id=100000001,
        )
        return calendar, ConfirmationStore(path), failed, retried, prepared

    calendar, store, failed, retried, prepared = asyncio.run(scenario())
    assert failed.outcome == "retryable_error"
    assert retried.outcome == "created"
    assert calendar.calls == 2
    assert len(calendar.created_batches) == 1
    record = store.get(prepared.confirmation_id)
    assert record["stage"] == "created"
    assert record["attempts"] == 2
    assert "last_error" not in record


def test_fatal_calendar_connection_escapes_and_keeps_creating_record(tmp_path):
    async def scenario():
        calendar = FakeCalendarClient()
        calendar.fatal_next = True
        path = tmp_path / "confirmations.json"
        pipeline = CalendarConfirmationPipeline(ConfirmationStore(path), calendar)
        prepared = prepare(pipeline)
        data = prepared.reply_markup["inline_keyboard"][0][0]["callback_data"]

        with pytest.raises(CalendarConnectionError):
            await pipeline.handle_callback(
                data=data,
                owner_user_id=100000001,
                chat_id=100000001,
            )

        # The process-local in-flight guard was released, while the durable
        # stage remains resumable with the original idempotency key.
        resumed = await pipeline.handle_callback(
            data=data,
            owner_user_id=100000001,
            chat_id=100000001,
        )
        return ConfirmationStore(path), prepared, resumed

    store, prepared, resumed = asyncio.run(scenario())
    assert resumed.outcome == "created"
    record = store.get(prepared.confirmation_id)
    assert record["stage"] == "created"
    assert record["attempts"] == 2
    assert "last_error" not in record


def test_non_create_intent_never_gets_confirmation(tmp_path):
    calendar = FakeCalendarClient()
    pipeline = CalendarConfirmationPipeline(
        ConfirmationStore(tmp_path / "confirmations.json"), calendar
    )
    result = pipeline.prepare(
        source_update_id=99,
        account="personal",
        owner_user_id=100000001,
        chat_id=100000001,
        intent=calendar_intent(confidence=0.99, action="clarify"),
    )
    assert result is None
    assert calendar.calls == 0


def test_confidence_threshold_cannot_be_configured_below_safety_floor(tmp_path):
    with pytest.raises(ValueError, match="0.85"):
        CalendarConfirmationPipeline(
            ConfirmationStore(tmp_path / "confirmations.json"),
            FakeCalendarClient(),
            confidence_threshold=0.84,
        )

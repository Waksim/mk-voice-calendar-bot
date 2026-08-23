import asyncio
from dataclasses import replace
from datetime import datetime
import hashlib
import json
import stat
from zoneinfo import ZoneInfo

import pytest

from tg_voice_transcriber_bot.calendar import (
    CalendarConnectionError,
    CalendarEventSnapshot,
    CalendarWriteRejectedError,
    CreatedCalendarEvent,
    DeletedCalendarEvent,
)
from tg_voice_transcriber_bot.operations import (
    CalendarOperationError,
    CalendarOperationPipeline,
    OperationStore,
    _snapshot,
)


OWNER = 100000001
NOW = datetime(2026, 8, 22, 18, tzinfo=ZoneInfo("Europe/Moscow"))


def event(**changes):
    value = {
        "title": "Планёрка",
        "start_at": "2026-08-24T17:30:00+03:00",
        "end_at": "2026-08-24T18:00:00+03:00",
        "all_day": False,
        "timezone": "Europe/Moscow",
        "location": None,
        "description": "Повторная встреча",
        "recurrence_rrule": None,
    }
    value.update(changes)
    return value


def plan(*operations, confidence=0.95):
    return {
        "action": "execute",
        "operations": list(operations),
        "clarification_question": None,
        "confidence": confidence,
    }


def create_op(payload=None):
    return {
        "type": "create",
        "target_event_id": None,
        "event": payload or event(),
        "patch": None,
        "clear_fields": [],
    }


def update_op(event_id, patch, clear_fields=None):
    return {
        "type": "update",
        "target_event_id": event_id,
        "event": None,
        "patch": patch,
        "clear_fields": list(clear_fields or []),
    }


def delete_op(event_id):
    return {
        "type": "delete",
        "target_event_id": event_id,
        "event": None,
        "patch": None,
        "clear_fields": [],
    }


def provider_event(event_id="provider-event", **changes):
    values = {
        "account": "personal",
        "calendar_id": "primary",
        "event_id": event_id,
        "title": "Планёрка",
        "description": "Повторная встреча",
        "location": None,
        "start_at": "2026-08-24T17:30:00+03:00",
        "end_at": "2026-08-24T18:00:00+03:00",
        "all_day": False,
        "timezone": "Europe/Moscow",
        "status": "confirmed",
        "html_link": f"https://calendar.google.com/event?eid={event_id}",
        "recurrence_rrules": (),
        "attendee_emails": (),
        "updated_at": "2026-08-22T12:00:00Z",
        "recurring_event_id": None,
        "original_start_at": None,
        "creator_is_self": True,
        "organizer_is_self": True,
        "safety_metadata_complete": True,
        "safety_metadata_fingerprint": "basic-event-v1",
    }
    values.update(changes)
    return CalendarEventSnapshot(**values)


class FakeCalendar:
    def __init__(self):
        self.events = {}
        self.calls = []
        self.created_by_key = {}

    @staticmethod
    def _snapshot(event_id, source, *, status="confirmed"):
        return CalendarEventSnapshot(
            account="personal",
            calendar_id="primary",
            event_id=event_id,
            title=source["title"],
            description=source.get("description"),
            location=source.get("location"),
            start_at=source["start_at"],
            end_at=source["end_at"],
            all_day=source["all_day"],
            timezone=source["timezone"],
            status=status,
            html_link=f"https://calendar.google.com/event?eid={event_id}",
            recurrence_rrules=(
                (source["recurrence_rrule"],)
                if source.get("recurrence_rrule")
                else ()
            ),
            creator_is_self=True,
            organizer_is_self=True,
            safety_metadata_complete=True,
            safety_metadata_fingerprint="basic-event-v1",
        )

    async def create_events(self, *, account, events, idempotency_key):
        self.calls.append(("create", idempotency_key))
        if idempotency_key in self.created_by_key:
            return self.created_by_key[idempotency_key]
        created = []
        for index, source in enumerate(events):
            event_id = hashlib.sha256(
                f"{idempotency_key}:{index}".encode()
            ).hexdigest()[:24]
            snapshot = self._snapshot(event_id, source)
            self.events[event_id] = snapshot
            created.append(CreatedCalendarEvent(event_id, snapshot.html_link))
        result = tuple(created)
        self.created_by_key[idempotency_key] = result
        return result

    async def get_event(self, *, account, event_id):
        self.calls.append(("get", event_id))
        return self.events[event_id]

    async def update_event(self, *, account, event_id, patch, idempotency_key):
        self.calls.append(("update", event_id, dict(patch), idempotency_key))
        before = self.events[event_id]
        updates = {}
        for key, value in patch.items():
            if key in {"description", "location"} and value == "":
                value = None
            updates[key] = value
        if "start_at" in updates:
            updates["all_day"] = len(str(updates["start_at"])) == 10
        result = replace(before, **updates)
        self.events[event_id] = result
        return result

    async def delete_event(self, *, account, event_id, idempotency_key):
        self.calls.append(("delete", event_id, idempotency_key))
        previous = self.events[event_id]
        cancelled = replace(previous, status="cancelled")
        self.events[event_id] = cancelled
        return DeletedCalendarEvent(
            previous=previous,
            current=cancelled,
            verified_cancelled=True,
        )


def apply(
    pipeline,
    source_id,
    operation_plan,
    *,
    account="personal",
    chat_id=OWNER,
    allowed_event_ids=None,
    displayed_candidates=None,
):
    return pipeline.apply_plan(
        source_update_id=source_id,
        account=account,
        owner_user_id=chat_id,
        chat_id=chat_id,
        transcript="Голосовая команда целиком",
        reference_time=NOW,
        plan=operation_plan,
        interaction_input={"type": "user_input", "content": [{"type": "text", "text": "x"}]},
        interaction_steps=[{"type": "model_output", "content": []}],
        allowed_event_ids=allowed_event_ids,
        displayed_candidates=displayed_candidates,
    )


def test_legacy_confirmations_seed_exact_known_ids_without_inventing_transcript(tmp_path):
    legacy = tmp_path / "calendar-confirmations.json"
    legacy.write_text(
        json.dumps(
            {
                "version": 1,
                "records": {
                    "abcdefghijklmnop": {
                        "confirmation_id": "abcdefghijklmnop",
                        "source_key": "telegram-update:1",
                        "account": "personal",
                        "owner_user_id": OWNER,
                        "chat_id": OWNER,
                        "stage": "created",
                        "created_at": "2026-08-22T10:00:00+00:00",
                        "updated_at": "2026-08-22T10:01:00+00:00",
                        "intent": {
                            "action": "create",
                            "events": [event()],
                            "clarification_question": None,
                            "confidence": 0.95,
                        },
                        "calendar_events": [
                            {"event_id": "known-google-id", "html_link": None}
                        ],
                    }
                },
                "source_index": {"telegram-update:1": "abcdefghijklmnop"},
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    path = tmp_path / "operations.json"
    store = OperationStore(path, legacy)
    context = store.context("work", OWNER + 1, NOW, "Europe/Moscow")

    assert context.allowed_event_ids == ("known-google-id",)
    assert context.application_state["candidate_events"][0]["start_at"].endswith(
        "17:30:00+03:00"
    )
    assert context.recent_conversation == ()
    assert stat.S_IMODE(path.stat().st_mode) == 0o600


def test_provider_timezone_is_canonicalized_for_prompt_and_updates():
    provider = CalendarEventSnapshot(
        account="personal",
        calendar_id="primary",
        event_id="known",
        title="Планёрка",
        description=None,
        location=None,
        start_at="2026-08-24T16:30:00+02:00",
        end_at="2026-08-24T17:00:00+02:00",
        all_day=False,
        timezone="Europe/Amsterdam",
        status="confirmed",
    )
    normalized = _snapshot(
        provider,
        account="personal",
        timezone_name="Europe/Moscow",
    )
    assert normalized["start_at"] == "2026-08-24T17:30:00+03:00"
    assert normalized["end_at"] == "2026-08-24T18:00:00+03:00"
    assert normalized["timezone"] == "Europe/Moscow"


def test_offset_only_provider_snapshot_preserves_missing_named_timezone():
    provider = provider_event(
        "offset-only",
        start_at="2026-08-24T14:30:00+00:00",
        end_at="2026-08-24T15:00:00+00:00",
        timezone=None,
    )

    normalized = _snapshot(
        provider,
        account="personal",
        timezone_name="Europe/Moscow",
    )

    assert normalized["timezone"] is None
    assert normalized["start_at"] == "2026-08-24T14:30:00+00:00"
    assert normalized["end_at"] == "2026-08-24T15:00:00+00:00"


def test_bot_authored_fallback_without_timezone_uses_default():
    payload = event()
    payload.pop("timezone")

    normalized = _snapshot(
        None,
        account="personal",
        fallback=payload,
        event_id="bot-fallback",
    )

    assert normalized["timezone"] == "Europe/Moscow"


def test_create_is_immediate_replay_safe_and_undo_is_exactly_once(tmp_path):
    async def scenario():
        calendar = FakeCalendar()
        pipeline = CalendarOperationPipeline(OperationStore(tmp_path / "ops.json"), calendar)
        first = await apply(pipeline, 10, plan(create_op()))
        replay = await apply(pipeline, 10, plan(create_op()))
        undone = await pipeline.undo(
            operation_id=first.operation_id, owner_user_id=OWNER, chat_id=OWNER
        )
        duplicate = await pipeline.undo(
            operation_id=first.operation_id, owner_user_id=OWNER, chat_id=OWNER
        )
        return calendar, pipeline, first, replay, undone, duplicate

    calendar, pipeline, first, replay, undone, duplicate = asyncio.run(scenario())
    assert first.stage == "applied"
    assert replay.replayed is True
    assert [call[0] for call in calendar.calls].count("create") == 1
    assert [call[0] for call in calendar.calls].count("delete") == 1
    assert undone.outcome == "undone"
    assert duplicate.outcome == "already_undone"
    assert pipeline.context(account="personal", chat_id=OWNER, now=NOW).allowed_event_ids == ()


def test_recent_actions_are_isolated_per_telegram_conversation(tmp_path):
    async def scenario():
        calendar = FakeCalendar()
        store = OperationStore(tmp_path / "ops.json")
        pipeline = CalendarOperationPipeline(store, calendar)
        created = await apply(pipeline, 11, plan(create_op()))
        event_id = created.record["items"][0]["after"]["event_id"]
        personal = pipeline.context(account="personal", chat_id=OWNER, now=NOW)
        work = pipeline.context(account="work", chat_id=OWNER + 1, now=NOW)
        return event_id, personal, work

    event_id, personal, work = asyncio.run(scenario())

    assert personal.application_state["recent_actions"][0]["operation_id"]
    assert work.application_state["recent_actions"] == []
    # Provider events remain shared so the other profile can still discover
    # and target the event explicitly instead of inheriting invisible chat history.
    assert event_id in work.allowed_event_ids


def test_update_location_keeps_time_and_undo_restores_before_snapshot(tmp_path):
    async def scenario():
        calendar = FakeCalendar()
        pipeline = CalendarOperationPipeline(OperationStore(tmp_path / "ops.json"), calendar)
        created = await apply(pipeline, 20, plan(create_op()))
        event_id = created.record["items"][0]["after"]["event_id"]
        updated = await apply(
            pipeline,
            21,
            plan(update_op(event_id, {"location": "переговорная А"})),
        )
        before = updated.record["items"][0]["before"]
        after = updated.record["items"][0]["after"]
        undone = await pipeline.undo(
            operation_id=updated.operation_id, owner_user_id=OWNER, chat_id=OWNER
        )
        return calendar, before, after, undone, event_id

    calendar, before, after, undone, event_id = asyncio.run(scenario())
    assert after["location"] == "переговорная А"
    assert after["start_at"] == before["start_at"]
    assert after["end_at"] == before["end_at"]
    assert undone.outcome == "undone"
    assert calendar.events[event_id].location is None


def test_low_confidence_execute_is_not_policy_blocked(tmp_path):
    async def scenario():
        calendar = FakeCalendar()
        pipeline = CalendarOperationPipeline(
            OperationStore(tmp_path / "ops.json"), calendar
        )
        provider = provider_event("low-confidence-update")
        calendar.events[provider.event_id] = provider
        pipeline.observe_lookup_events("personal", [provider])

        result = await apply(
            pipeline,
            200,
            plan(
                update_op(provider.event_id, {"location": "переговорная А"}),
                confidence=0.01,
            ),
        )
        return calendar, result

    calendar, result = asyncio.run(scenario())

    assert result.stage == "applied"
    assert result.record["confidence"] == 0.01
    assert [call[0] for call in calendar.calls] == ["get", "update"]
    assert calendar.events["low-confidence-update"].location == "переговорная А"


def test_noop_update_is_applied_without_provider_update_call(tmp_path):
    async def scenario():
        calendar = FakeCalendar()
        pipeline = CalendarOperationPipeline(
            OperationStore(tmp_path / "ops.json"), calendar
        )
        provider = provider_event("noop-update", location="переговорная А")
        calendar.events[provider.event_id] = provider
        pipeline.observe_lookup_events("personal", [provider])

        result = await apply(
            pipeline,
            201,
            plan(update_op(provider.event_id, {"location": "переговорная А"})),
        )
        return calendar, result

    calendar, result = asyncio.run(scenario())

    assert result.stage == "applied"
    assert result.record["items"][0]["stage"] == "applied"
    assert result.record["items"][0]["after"] == result.record["items"][0]["before"]
    assert result.record["items"][0].get("provider_write_started_at") is None
    assert [call[0] for call in calendar.calls] == ["get"]


def test_noop_after_real_update_keeps_real_undo_ownership(tmp_path):
    async def scenario():
        calendar = FakeCalendar()
        pipeline = CalendarOperationPipeline(
            OperationStore(tmp_path / "ops.json"), calendar
        )
        provider = provider_event("real-then-noop")
        calendar.events[provider.event_id] = provider
        pipeline.observe_lookup_events("personal", [provider])

        real = await apply(
            pipeline,
            202,
            plan(update_op(provider.event_id, {"location": "переговорная А"})),
        )
        noop = await apply(
            pipeline,
            203,
            plan(update_op(provider.event_id, {"location": "переговорная А"})),
        )
        entry_after_noop = pipeline.store.event_entry(
            "personal", provider.event_id
        )
        undone = await pipeline.undo(
            operation_id=real.operation_id,
            owner_user_id=OWNER,
            chat_id=OWNER,
        )
        return calendar, real, noop, entry_after_noop, undone

    calendar, real, noop, entry_after_noop, undone = asyncio.run(scenario())

    assert noop.stage == "applied"
    assert noop.record["items"][0]["write_skipped"] is True
    assert entry_after_noop["last_operation_id"] == real.operation_id
    assert undone.outcome == "undone"
    assert calendar.events["real-then-noop"].location is None
    assert [call[0] for call in calendar.calls].count("update") == 2


def test_write_skipped_item_does_not_participate_in_mixed_undo_freshness(
    tmp_path,
):
    async def scenario():
        calendar = FakeCalendar()
        pipeline = CalendarOperationPipeline(
            OperationStore(tmp_path / "ops.json"), calendar
        )
        changed = provider_event("mixed-real-update")
        unchanged = provider_event(
            "mixed-skipped-update", location="уже задано"
        )
        calendar.events.update(
            {event.event_id: event for event in (changed, unchanged)}
        )
        pipeline.observe_lookup_events("personal", [changed, unchanged])

        result = await apply(
            pipeline,
            204,
            plan(
                update_op(changed.event_id, {"location": "новое место"}),
                update_op(unchanged.event_id, {"location": "уже задано"}),
            ),
        )
        changed_entry = pipeline.store.event_entry("personal", changed.event_id)
        skipped_entry = pipeline.store.event_entry("personal", unchanged.event_id)
        calendar.calls.clear()
        undone = await pipeline.undo(
            operation_id=result.operation_id,
            owner_user_id=OWNER,
            chat_id=OWNER,
        )
        return calendar, result, changed_entry, skipped_entry, undone

    calendar, result, changed_entry, skipped_entry, undone = asyncio.run(scenario())

    assert [item.get("write_skipped", False) for item in result.record["items"]] == [
        False,
        True,
    ]
    assert changed_entry["last_operation_id"] == result.operation_id
    assert skipped_entry["last_operation_id"] is None
    assert undone.outcome == "undone"
    assert [call[0] for call in calendar.calls] == ["get", "update"]
    assert calendar.events["mixed-real-update"].location is None
    assert calendar.events["mixed-skipped-update"].location == "уже задано"


def test_non_temporal_update_and_undo_preserve_utc_provider_time_exactly(tmp_path):
    async def scenario():
        calendar = FakeCalendar()
        pipeline = CalendarOperationPipeline(
            OperationStore(tmp_path / "ops.json"), calendar
        )
        provider = provider_event(
            "utc-event",
            start_at="2026-08-24T14:30:00+00:00",
            end_at="2026-08-24T15:00:00+00:00",
            timezone="UTC",
        )
        calendar.events[provider.event_id] = provider
        pipeline.observe_lookup_events("personal", [provider])
        updated = await apply(
            pipeline,
            22,
            plan(update_op(provider.event_id, {"location": "Переговорная А"})),
        )
        undone = await pipeline.undo(
            operation_id=updated.operation_id,
            owner_user_id=OWNER,
            chat_id=OWNER,
        )
        return calendar, updated, undone

    calendar, updated, undone = asyncio.run(scenario())

    before = updated.record["items"][0]["before"]
    after = updated.record["items"][0]["after"]
    assert (before["start_at"], before["end_at"], before["timezone"]) == (
        "2026-08-24T14:30:00+00:00",
        "2026-08-24T15:00:00+00:00",
        "UTC",
    )
    assert (after["start_at"], after["end_at"], after["timezone"]) == (
        "2026-08-24T14:30:00+00:00",
        "2026-08-24T15:00:00+00:00",
        "UTC",
    )
    update_calls = [call for call in calendar.calls if call[0] == "update"]
    assert update_calls[0][2] == {"location": "Переговорная А"}
    assert update_calls[1][2] == {"location": ""}
    restored = calendar.events["utc-event"]
    assert undone.outcome == "undone"
    assert (restored.start_at, restored.end_at, restored.timezone) == (
        "2026-08-24T14:30:00+00:00",
        "2026-08-24T15:00:00+00:00",
        "UTC",
    )


def test_non_temporal_update_preserves_offset_only_event_without_timezone_write(
    tmp_path,
):
    async def scenario():
        calendar = FakeCalendar()
        pipeline = CalendarOperationPipeline(
            OperationStore(tmp_path / "ops.json"), calendar
        )
        provider = provider_event(
            "offset-only-update",
            start_at="2026-08-24T14:30:00+00:00",
            end_at="2026-08-24T15:00:00+00:00",
            timezone=None,
        )
        calendar.events[provider.event_id] = provider
        pipeline.observe_lookup_events("personal", [provider])
        updated = await apply(
            pipeline,
            23,
            plan(update_op(provider.event_id, {"location": "Переговорная А"})),
        )
        undone = await pipeline.undo(
            operation_id=updated.operation_id,
            owner_user_id=OWNER,
            chat_id=OWNER,
        )
        return calendar, updated, undone

    calendar, updated, undone = asyncio.run(scenario())

    after = updated.record["items"][0]["after"]
    assert (after["start_at"], after["end_at"], after["timezone"]) == (
        "2026-08-24T14:30:00+00:00",
        "2026-08-24T15:00:00+00:00",
        None,
    )
    update_calls = [call for call in calendar.calls if call[0] == "update"]
    assert update_calls[0][2] == {"location": "Переговорная А"}
    assert update_calls[1][2] == {"location": ""}
    assert undone.outcome == "undone"
    restored = calendar.events["offset-only-update"]
    assert (restored.start_at, restored.end_at, restored.timezone) == (
        "2026-08-24T14:30:00+00:00",
        "2026-08-24T15:00:00+00:00",
        None,
    )


def test_timed_delete_without_named_timezone_is_not_policy_blocked(tmp_path):
    async def scenario():
        calendar = FakeCalendar()
        pipeline = CalendarOperationPipeline(
            OperationStore(tmp_path / "ops.json"), calendar
        )
        provider = provider_event(
            "offset-only-delete",
            start_at="2026-08-24T14:30:00+00:00",
            end_at="2026-08-24T15:00:00+00:00",
            timezone=None,
        )
        calendar.events[provider.event_id] = provider
        pipeline.observe_lookup_events("personal", [provider])
        result = await apply(
            pipeline,
            24,
            plan(delete_op(provider.event_id)),
        )
        return calendar, result

    calendar, result = asyncio.run(scenario())

    assert result.stage == "applied"
    assert [item["stage"] for item in result.record["items"]] == ["applied"]
    assert [call[0] for call in calendar.calls] == ["get", "delete"]
    assert calendar.events["offset-only-delete"].status == "cancelled"


def test_moving_only_start_preserves_original_duration(tmp_path):
    async def scenario():
        calendar = FakeCalendar()
        pipeline = CalendarOperationPipeline(OperationStore(tmp_path / "ops.json"), calendar)
        created = await apply(pipeline, 30, plan(create_op()))
        event_id = created.record["items"][0]["after"]["event_id"]
        return await apply(
            pipeline,
            31,
            plan(update_op(event_id, {"start_at": "2026-08-25T12:00:00+03:00"})),
        )

    result = asyncio.run(scenario())
    after = result.record["items"][0]["after"]
    assert after["start_at"] == "2026-08-25T12:00:00+03:00"
    assert after["end_at"] == "2026-08-25T12:30:00+03:00"


def test_delete_undo_recreates_event_with_a_new_known_id(tmp_path):
    async def scenario():
        calendar = FakeCalendar()
        pipeline = CalendarOperationPipeline(OperationStore(tmp_path / "ops.json"), calendar)
        created = await apply(pipeline, 40, plan(create_op()))
        old_id = created.record["items"][0]["after"]["event_id"]
        deleted = await apply(pipeline, 41, plan(delete_op(old_id)))
        pipeline.observe_lookup_events("personal", [calendar.events[old_id]])
        deleted_entry = pipeline.store.event_entry("personal", old_id)
        undone = await pipeline.undo(
            operation_id=deleted.operation_id, owner_user_id=OWNER, chat_id=OWNER
        )
        context = pipeline.context(account="work", chat_id=OWNER + 1, now=NOW)
        return old_id, deleted, deleted_entry, undone, context

    old_id, deleted, deleted_entry, undone, context = asyncio.run(scenario())
    assert deleted_entry["last_operation_id"] == deleted.operation_id
    assert undone.outcome == "undone"
    assert old_id not in context.allowed_event_ids
    assert len(context.allowed_event_ids) == 1


def test_unknown_target_is_rejected_before_any_calendar_write(tmp_path):
    calendar = FakeCalendar()
    pipeline = CalendarOperationPipeline(OperationStore(tmp_path / "ops.json"), calendar)
    with pytest.raises(ValueError, match="known"):
        asyncio.run(
            apply(
                pipeline,
                50,
                plan(update_op("hallucinated", {"location": "Переговорная А"})),
            )
        )
    assert calendar.calls == []


def test_clarification_is_kept_in_two_turn_history_without_calendar_write(tmp_path):
    async def scenario():
        calendar = FakeCalendar()
        pipeline = CalendarOperationPipeline(OperationStore(tmp_path / "ops.json"), calendar)
        clarification = {
            "action": "clarify",
            "operations": [],
            "clarification_question": "Какую именно планёрку перенести?",
            "confidence": 0.7,
        }
        result = await apply(pipeline, 60, clarification)
        context = pipeline.context(account="personal", chat_id=OWNER, now=NOW)
        return calendar, result, context

    calendar, result, context = asyncio.run(scenario())
    assert result.stage == "clarify"
    assert calendar.calls == []
    assert context.recent_conversation[-1]["assistant_message"].startswith("Какую")
    assert [step["type"] for step in context.history_steps] == [
        "user_input",
        "model_output",
    ]


def test_undo_is_blocked_after_a_later_operation_touches_same_event(tmp_path):
    async def scenario():
        calendar = FakeCalendar()
        pipeline = CalendarOperationPipeline(OperationStore(tmp_path / "ops.json"), calendar)
        created = await apply(pipeline, 70, plan(create_op()))
        event_id = created.record["items"][0]["after"]["event_id"]
        first_update = await apply(
            pipeline, 71, plan(update_op(event_id, {"location": "Переговорная А"}))
        )
        await apply(
            pipeline, 72, plan(update_op(event_id, {"description": "Новый текст"}))
        )
        return await pipeline.undo(
            operation_id=first_update.operation_id,
            owner_user_id=OWNER,
            chat_id=OWNER,
        )

    assert asyncio.run(scenario()).outcome == "blocked"


@pytest.mark.parametrize("action", ["create", "update", "delete"])
def test_undo_is_blocked_by_unobserved_provider_edit(tmp_path, action):
    async def scenario():
        calendar = FakeCalendar()
        pipeline = CalendarOperationPipeline(OperationStore(tmp_path / "ops.json"), calendar)
        created = await apply(pipeline, 73, plan(create_op()))
        event_id = created.record["items"][0]["after"]["event_id"]
        target = created
        if action == "update":
            target = await apply(
                pipeline,
                74,
                plan(update_op(event_id, {"location": "Переговорная А"})),
            )
        elif action == "delete":
            target = await apply(pipeline, 74, plan(delete_op(event_id)))

        calendar.events[event_id] = replace(
            calendar.events[event_id],
            description="Ручная правка вне бота",
            status="confirmed",
        )
        writes_before_undo = [
            call for call in calendar.calls if call[0] in {"create", "update", "delete"}
        ]
        result = await pipeline.undo(
            operation_id=target.operation_id,
            owner_user_id=OWNER,
            chat_id=OWNER,
        )
        writes_after_undo = [
            call for call in calendar.calls if call[0] in {"create", "update", "delete"}
        ]
        return result, writes_before_undo, writes_after_undo

    result, writes_before_undo, writes_after_undo = asyncio.run(scenario())

    assert result.outcome == "blocked"
    assert writes_after_undo == writes_before_undo


def test_undo_provider_read_error_is_retryable_without_clearing_marker(tmp_path):
    async def scenario():
        calendar = FakeCalendar()
        pipeline = CalendarOperationPipeline(OperationStore(tmp_path / "ops.json"), calendar)
        created = await apply(pipeline, 75, plan(create_op()))
        event_id = created.record["items"][0]["after"]["event_id"]

        async def unavailable_get_event(*, account, event_id):
            raise RuntimeError("temporary provider read failure")

        calendar.get_event = unavailable_get_event
        result = await pipeline.undo(
            operation_id=created.operation_id,
            owner_user_id=OWNER,
            chat_id=OWNER,
        )
        entry = pipeline.store.event_entry("personal", event_id)
        return created, result, entry

    created, result, entry = asyncio.run(scenario())

    assert result.outcome == "retryable_error"
    assert entry["last_operation_id"] == created.operation_id


def test_record_read_caches_events_and_conversation_without_undo(tmp_path):
    async def scenario():
        calendar = FakeCalendar()
        pipeline = CalendarOperationPipeline(OperationStore(tmp_path / "ops.json"), calendar)
        result = await pipeline.record_read(
            source_update_id=80,
            account="personal",
            owner_user_id=OWNER,
            chat_id=OWNER,
            transcript="Что у меня в календаре завтра?",
            reference_time=NOW,
            lookup={
                "query": None,
                "time_min": "2026-08-23T00:00:00+03:00",
                "time_max": "2026-08-24T00:00:00+03:00",
            },
            events=[provider_event()],
            total_count=1,
            may_be_incomplete=False,
            interaction_input={
                "type": "user_input",
                "content": [{"type": "text", "text": "calendar read"}],
            },
            interaction_steps=[{"type": "model_output", "content": []}],
            assistant_text="Нашёл одно событие.",
        )
        context = pipeline.context(account="personal", chat_id=OWNER, now=NOW)
        undo = await pipeline.undo(
            operation_id=result.operation_id,
            owner_user_id=OWNER,
            chat_id=OWNER,
        )
        return calendar, result, context, undo

    calendar, result, context, undo = asyncio.run(scenario())

    assert result.stage == "read"
    assert result.record["undo"] == {"stage": "unavailable"}
    assert result.record["items"][0]["undo_stage"] == "unavailable"
    assert context.allowed_event_ids == ("provider-event",)
    assert context.application_state["candidate_events"][0]["title"] == "Планёрка"
    assert context.recent_conversation[-1] == {
        "user_message": "Что у меня в календаре завтра?",
        "assistant_message": "Нашёл одно событие.",
        "status": "read",
        "actions": [
            {
                "type": "read",
                "event_id": "provider-event",
                "before": None,
                "after": result.record["items"][0]["after"],
            }
        ],
    }
    assert undo.outcome == "blocked"
    assert calendar.calls == []


def test_record_read_replay_is_idempotent(tmp_path):
    async def scenario():
        calendar = FakeCalendar()
        pipeline = CalendarOperationPipeline(OperationStore(tmp_path / "ops.json"), calendar)
        arguments = {
            "source_update_id": 81,
            "account": "personal",
            "owner_user_id": OWNER,
            "chat_id": OWNER,
            "transcript": "Покажи встречи",
            "reference_time": NOW,
            "lookup": {
                "query": "встреча",
                "time_min": "2026-08-22T00:00:00+03:00",
                "time_max": "2026-08-29T00:00:00+03:00",
            },
            "events": [provider_event("read-replay")],
            "total_count": 1,
            "may_be_incomplete": False,
        }
        first = await pipeline.record_read(**arguments)
        replay = await pipeline.record_read(
            **{
                **arguments,
                "events": [provider_event("must-not-be-observed")],
                "total_count": 2,
            }
        )
        context = pipeline.context(account="personal", chat_id=OWNER, now=NOW)
        return first, replay, context

    first, replay, context = asyncio.run(scenario())

    assert replay.replayed is True
    assert replay.operation_id == first.operation_id
    assert replay.record == first.record
    assert context.allowed_event_ids == ("read-replay",)
    assert len(context.recent_conversation) == 1


def test_lookup_observation_of_external_edit_clears_undo_freshness(tmp_path):
    async def scenario():
        calendar = FakeCalendar()
        pipeline = CalendarOperationPipeline(OperationStore(tmp_path / "ops.json"), calendar)
        created = await apply(pipeline, 82, plan(create_op()))
        event_id = created.record["items"][0]["after"]["event_id"]
        operation_id = created.operation_id

        observed = replace(
            calendar.events[event_id],
            account="work",
            location="Переговорная А",
        )
        pipeline.observe_lookup_events("work", [observed])
        personal_entry = pipeline.store.event_entry("personal", event_id)
        work_entry = pipeline.store.event_entry("work", event_id)
        personal_context = pipeline.context(account="personal", chat_id=OWNER, now=NOW)
        work_context = pipeline.context(account="work", chat_id=OWNER + 1, now=NOW)
        undone = await pipeline.undo(
            operation_id=operation_id,
            owner_user_id=OWNER,
            chat_id=OWNER,
        )
        return (
            event_id,
            operation_id,
            personal_entry,
            work_entry,
            personal_context,
            work_context,
            undone,
        )

    (
        event_id,
        operation_id,
        personal_entry,
        work_entry,
        personal_context,
        work_context,
        undone,
    ) = asyncio.run(scenario())

    assert personal_entry == work_entry
    assert personal_entry["last_operation_id"] is None
    assert personal_entry["snapshot"]["location"] == "Переговорная А"
    assert event_id in personal_context.allowed_event_ids
    assert event_id in work_context.allowed_event_ids
    assert undone.outcome == "blocked"


def test_equivalent_lookup_observation_preserves_last_mutation_and_undo(tmp_path):
    async def scenario():
        calendar = FakeCalendar()
        pipeline = CalendarOperationPipeline(OperationStore(tmp_path / "ops.json"), calendar)
        created = await apply(pipeline, 83, plan(create_op()))
        event_id = created.record["items"][0]["after"]["event_id"]
        observed = replace(
            calendar.events[event_id],
            account="work",
            html_link="https://calendar.google.com/a-different-observation-link",
            updated_at="2026-08-23T09:15:00Z",
        )

        pipeline.observe_lookup_events("work", [observed])
        entry = pipeline.store.event_entry("personal", event_id)
        undone = await pipeline.undo(
            operation_id=created.operation_id,
            owner_user_id=OWNER,
            chat_id=OWNER,
        )
        return created, entry, undone

    created, entry, undone = asyncio.run(scenario())

    assert entry["last_operation_id"] == created.operation_id
    assert undone.outcome == "undone"


def test_clarification_preserves_displayed_candidate_order_for_next_turn(tmp_path):
    async def scenario():
        calendar = FakeCalendar()
        pipeline = CalendarOperationPipeline(OperationStore(tmp_path / "ops.json"), calendar)
        first = provider_event("first", title="Первое событие")
        second = provider_event("second", title="Второе событие")
        pipeline.observe_lookup_events("personal", [first, second])
        clarification = {
            "action": "clarify",
            "operations": [],
            "clarification_question": "Какое из этих событий изменить?",
            "confidence": 0.8,
        }
        result = await apply(
            pipeline,
            84,
            clarification,
            allowed_event_ids=("first", "second"),
            displayed_candidates=(second, first),
        )
        context = pipeline.context(account="personal", chat_id=OWNER, now=NOW)
        return result, context

    result, context = asyncio.run(scenario())

    assert [event["event_id"] for event in result.record["displayed_candidates"]] == [
        "second",
        "first",
    ]
    assert context.allowed_event_ids == ("second", "first")
    assert [
        (event["display_index"], event["event_id"])
        for event in context.application_state["candidate_events"]
    ] == [(1, "second"), (2, "first")]


@pytest.mark.parametrize("action", ["update", "delete"])
def test_fresh_provider_recurrence_change_does_not_block_exact_id_mutation(
    tmp_path, action
):
    async def scenario():
        calendar = FakeCalendar()
        pipeline = CalendarOperationPipeline(OperationStore(tmp_path / "ops.json"), calendar)
        event_id = f"changed-{action}"
        cached = provider_event(event_id)
        pipeline.observe_lookup_events("personal", [cached])
        calendar.events[event_id] = provider_event(
            event_id, recurrence_rrules=("RRULE:FREQ=WEEKLY",)
        )
        operation = (
            update_op(event_id, {"location": "Переговорная А"})
            if action == "update"
            else delete_op(event_id)
        )

        result = await apply(pipeline, 85, plan(operation))
        entry = pipeline.store.event_entry("personal", event_id)
        return calendar, result, entry

    calendar, result, entry = asyncio.run(scenario())

    assert result.stage == "applied"
    assert result.record["items"][0]["before"]["recurrence_rrules"] == [
        "RRULE:FREQ=WEEKLY"
    ]
    assert [call[0] for call in calendar.calls] == ["get", action]
    assert entry["snapshot"]["event_id"] == f"changed-{action}"
    if action == "update":
        assert calendar.events[f"changed-{action}"].location == "Переговорная А"
        assert calendar.events[f"changed-{action}"].recurrence_rrules == (
            "RRULE:FREQ=WEEKLY",
        )
    else:
        assert calendar.events[f"changed-{action}"].status == "cancelled"


@pytest.mark.parametrize("action", ["update", "delete"])
@pytest.mark.parametrize(
    "fresh_changes",
    [
        {
            "attendee_emails": ("guest@example.com",),
            "organizer_is_self": False,
        },
        {"event_type": "birthday"},
    ],
)
def test_fresh_provider_policy_metadata_does_not_block_mutation(
    tmp_path, action, fresh_changes
):
    async def scenario():
        calendar = FakeCalendar()
        pipeline = CalendarOperationPipeline(
            OperationStore(tmp_path / "ops.json"), calendar
        )
        event_id = f"fresh-policy-metadata-{action}"
        cached = provider_event(event_id)
        pipeline.observe_lookup_events("personal", [cached])
        calendar.events[event_id] = provider_event(event_id, **fresh_changes)
        operation = (
            update_op(event_id, {"location": "Переговорная А"})
            if action == "update"
            else delete_op(event_id)
        )

        result = await apply(pipeline, 850, plan(operation))
        return calendar, result

    calendar, result = asyncio.run(scenario())

    assert result.stage == "applied"
    assert [item["stage"] for item in result.record["items"]] == ["applied"]
    assert [call[0] for call in calendar.calls] == ["get", action]


@pytest.mark.parametrize("action", ["update", "delete"])
@pytest.mark.parametrize(
    "fresh_changes",
    [
        {"title": "Переименовано вручную"},
        {
            "start_at": "2026-08-25T17:30:00+03:00",
            "end_at": "2026-08-25T18:00:00+03:00",
        },
    ],
)
def test_semantically_changed_target_uses_fresh_snapshot_for_exact_id_mutation(
    tmp_path, action, fresh_changes
):
    async def scenario():
        calendar = FakeCalendar()
        pipeline = CalendarOperationPipeline(
            OperationStore(tmp_path / "ops.json"), calendar
        )
        event_id = f"semantic-change-{action}"
        cached = provider_event(event_id)
        pipeline.observe_lookup_events("personal", [cached])
        calendar.events[event_id] = provider_event(event_id, **fresh_changes)
        operation = (
            update_op(event_id, {"location": "Переговорная А"})
            if action == "update"
            else delete_op(event_id)
        )
        result = await apply(pipeline, 185, plan(operation))
        entry = pipeline.store.event_entry("personal", event_id)
        return calendar, result, entry

    calendar, result, entry = asyncio.run(scenario())

    assert result.stage == "applied"
    assert [call[0] for call in calendar.calls] == ["get", action]
    before = result.record["items"][0]["before"]
    assert before["title"] == fresh_changes.get(
        "title", "Планёрка"
    )
    assert before["start_at"] == fresh_changes.get(
        "start_at", "2026-08-24T17:30:00+03:00"
    )
    assert entry["snapshot"]["title"] == before["title"]
    assert entry["snapshot"]["start_at"] == before["start_at"]
    if action == "update":
        assert entry["snapshot"]["location"] == "Переговорная А"
    else:
        assert entry["snapshot"]["status"] == "cancelled"


@pytest.mark.parametrize("action", ["update", "delete"])
def test_richer_fresh_metadata_does_not_look_like_semantic_target_change(
    tmp_path, action
):
    async def scenario():
        calendar = FakeCalendar()
        pipeline = CalendarOperationPipeline(
            OperationStore(tmp_path / "ops.json"), calendar
        )
        event_id = f"legacy-rich-{action}"
        cached = provider_event(
            event_id,
            creator_is_self=None,
            organizer_is_self=None,
            safety_metadata_complete=False,
            safety_metadata_fingerprint=None,
        )
        fresh = provider_event(event_id)
        pipeline.observe_lookup_events("personal", [cached])
        calendar.events[event_id] = fresh
        operation = (
            update_op(event_id, {"location": "Переговорная А"})
            if action == "update"
            else delete_op(event_id)
        )
        result = await apply(pipeline, 186, plan(operation))
        return calendar, result

    calendar, result = asyncio.run(scenario())

    assert result.stage == "applied"
    assert [call[0] for call in calendar.calls] == ["get", action]


def test_resume_refreshes_before_and_applies_when_provider_write_never_started(
    tmp_path,
):
    async def scenario():
        calendar = FakeCalendar()
        pipeline = CalendarOperationPipeline(OperationStore(tmp_path / "ops.json"), calendar)
        provider = provider_event("resume-target")
        calendar.events[provider.event_id] = provider
        pipeline.observe_lookup_events("personal", [provider])
        original_apply_item = pipeline._apply_item

        async def crash_before_item(record, item, *, before_is_fresh=False):
            raise RuntimeError("crash before provider write")

        pipeline._apply_item = crash_before_item
        with pytest.raises(CalendarOperationError):
            await apply(
                pipeline,
                86,
                plan(update_op(provider.event_id, {"location": "Переговорная А"})),
            )
        failed = pipeline.store.find_by_source("telegram-update:86")
        assert failed["items"][0].get("provider_write_started_at") is None

        calendar.events[provider.event_id] = provider_event(
            provider.event_id,
            recurrence_rrules=("RRULE:FREQ=WEEKLY",),
        )
        pipeline._apply_item = original_apply_item
        resumed = await apply(
            pipeline,
            86,
            plan(update_op(provider.event_id, {"location": "Переговорная А"})),
        )
        return calendar, resumed

    calendar, resumed = asyncio.run(scenario())

    assert resumed.stage == "applied"
    assert resumed.record["items"][0]["before"]["recurrence_rrules"] == [
        "RRULE:FREQ=WEEKLY"
    ]
    assert [call[0] for call in calendar.calls] == ["get", "get", "update"]
    assert calendar.events["resume-target"].location == "Переговорная А"
    assert calendar.events["resume-target"].recurrence_rrules == (
        "RRULE:FREQ=WEEKLY",
    )


@pytest.mark.parametrize("action", ["update", "delete"])
@pytest.mark.parametrize(
    "fresh_changes",
    [
        {"title": "Переименовано во время перезапуска"},
        {
            "start_at": "2026-08-26T17:30:00+03:00",
            "end_at": "2026-08-26T18:00:00+03:00",
        },
    ],
)
def test_resume_uses_fresh_snapshot_for_exact_id_before_write(
    tmp_path, action, fresh_changes
):
    async def scenario():
        calendar = FakeCalendar()
        pipeline = CalendarOperationPipeline(
            OperationStore(tmp_path / "ops.json"), calendar
        )
        event_id = f"resume-semantic-{action}"
        provider = provider_event(event_id)
        calendar.events[event_id] = provider
        pipeline.observe_lookup_events("personal", [provider])
        operation = (
            update_op(event_id, {"location": "Переговорная А"})
            if action == "update"
            else delete_op(event_id)
        )
        operation_plan = plan(operation)
        original_apply_item = pipeline._apply_item

        async def transport_dies_before_item(
            record, item, *, before_is_fresh=False
        ):
            raise CalendarConnectionError("dead before provider write")

        pipeline._apply_item = transport_dies_before_item
        with pytest.raises(CalendarConnectionError):
            await apply(pipeline, 190, operation_plan)
        persisted = pipeline.store.find_by_source("telegram-update:190")
        assert isinstance(persisted["items"][0]["before"], dict)
        assert persisted["items"][0].get("provider_write_started_at") is None

        calendar.events[event_id] = provider_event(event_id, **fresh_changes)
        pipeline._apply_item = original_apply_item
        result = await apply(pipeline, 190, operation_plan)
        entry = pipeline.store.event_entry("personal", event_id)
        return calendar, result, entry

    calendar, result, entry = asyncio.run(scenario())

    assert result.stage == "applied"
    assert [call[0] for call in calendar.calls] == ["get", "get", action]
    before = result.record["items"][0]["before"]
    assert before["title"] == fresh_changes.get(
        "title", "Планёрка"
    )
    assert before["start_at"] == fresh_changes.get(
        "start_at", "2026-08-24T17:30:00+03:00"
    )
    assert entry["snapshot"]["title"] == before["title"]
    assert entry["snapshot"]["start_at"] == before["start_at"]
    if action == "update":
        assert entry["snapshot"]["location"] == "Переговорная А"
    else:
        assert entry["snapshot"]["status"] == "cancelled"


def test_uncertain_delete_replays_after_target_was_observed_cancelled(tmp_path):
    class LostDeleteResponseCalendar(FakeCalendar):
        def __init__(self):
            super().__init__()
            self.response_lost = False

        async def delete_event(self, *, account, event_id, idempotency_key):
            result = await super().delete_event(
                account=account,
                event_id=event_id,
                idempotency_key=idempotency_key,
            )
            if not self.response_lost:
                self.response_lost = True
                raise RuntimeError("response lost after delete")
            return result

    async def scenario():
        calendar = LostDeleteResponseCalendar()
        pipeline = CalendarOperationPipeline(
            OperationStore(tmp_path / "ops.json"), calendar
        )
        created = await apply(pipeline, 187, plan(create_op()))
        event_id = created.record["items"][0]["after"]["event_id"]
        deletion_plan = plan(delete_op(event_id))
        with pytest.raises(CalendarOperationError) as raised:
            await apply(pipeline, 188, deletion_plan)
        assert raised.value.retryable is True
        assert raised.value.outcome_uncertain is True

        # A read between webhook attempts may already see the tombstone.  The
        # original journal target remains trusted for idempotent reconciliation.
        pipeline.observe_lookup_events("personal", [calendar.events[event_id]])
        reconciled = await apply(pipeline, 188, deletion_plan)
        return calendar, reconciled, event_id

    calendar, reconciled, event_id = asyncio.run(scenario())

    delete_calls = [call for call in calendar.calls if call[0] == "delete"]
    assert len(delete_calls) == 2
    assert delete_calls[0][2] == delete_calls[1][2]
    assert reconciled.stage == "applied"
    assert reconciled.record["displayed_candidates"] == []
    assert calendar.events[event_id].status == "cancelled"


def test_confirmed_partial_batch_retries_transient_prewrite_failure(tmp_path):
    async def scenario():
        calendar = FakeCalendar()
        pipeline = CalendarOperationPipeline(
            OperationStore(tmp_path / "ops.json"), calendar
        )
        target = provider_event("partial-prewrite-target")
        calendar.events[target.event_id] = target
        pipeline.observe_lookup_events("personal", [target])
        operation_plan = plan(
            create_op(event(title="Новая встреча")),
            update_op(target.event_id, {"location": "Переговорная А"}),
        )
        original_merge = pipeline._merged_update
        failed_once = False

        def flaky_merge(before, patch, clear_fields):
            nonlocal failed_once
            if not failed_once:
                failed_once = True
                raise RuntimeError("transient local prewrite failure")
            return original_merge(before, patch, clear_fields)

        pipeline._merged_update = flaky_merge
        with pytest.raises(CalendarOperationError) as raised:
            await apply(pipeline, 189, operation_plan)
        failed = pipeline.store.find_by_source("telegram-update:189")
        replayed = await apply(pipeline, 189, operation_plan)
        return calendar, raised.value, failed, replayed

    calendar, error, failed, replayed = asyncio.run(scenario())

    assert error.partially_applied is True
    assert error.retryable is True
    assert error.outcome_uncertain is False
    assert [item["stage"] for item in failed["items"]] == ["applied", "failed"]
    assert failed["items"][1].get("provider_write_started_at") is None
    assert replayed.stage == "applied"
    assert len([call for call in calendar.calls if call[0] == "create"]) == 1
    assert len([call for call in calendar.calls if call[0] == "update"]) == 1
    assert calendar.events["partial-prewrite-target"].location == "Переговорная А"


def test_definitive_create_rejection_is_terminal_and_calendar_is_unchanged(tmp_path):
    class RejectedCreateCalendar(FakeCalendar):
        async def create_events(self, *, account, events, idempotency_key):
            self.calls.append(("create", idempotency_key))
            raise CalendarWriteRejectedError("sanitized provider rejection")

    async def scenario():
        calendar = RejectedCreateCalendar()
        pipeline = CalendarOperationPipeline(
            OperationStore(tmp_path / "ops.json"), calendar
        )
        with pytest.raises(CalendarOperationError) as raised:
            await apply(pipeline, 191, plan(create_op()))
        return calendar, raised.value, pipeline.store.find_by_source(
            "telegram-update:191"
        )

    calendar, error, record = asyncio.run(scenario())

    assert error.partially_applied is False
    assert error.retryable is False
    assert error.outcome_uncertain is False
    assert "Календарь не изменён" in str(error)
    assert calendar.events == {}
    assert [call[0] for call in calendar.calls] == ["create"]
    assert record["stage"] == "rejected"
    assert record["items"][0]["stage"] == "failed"
    assert record["items"][0]["last_error_class"] == "CalendarWriteRejectedError"


def test_definitive_rejection_after_applied_item_is_terminal_partial_batch(tmp_path):
    class RejectSecondCreateCalendar(FakeCalendar):
        async def create_events(self, *, account, events, idempotency_key):
            if idempotency_key.endswith(":1:create"):
                self.calls.append(("create", idempotency_key))
                raise CalendarWriteRejectedError("sanitized provider rejection")
            return await super().create_events(
                account=account,
                events=events,
                idempotency_key=idempotency_key,
            )

    async def scenario():
        calendar = RejectSecondCreateCalendar()
        pipeline = CalendarOperationPipeline(
            OperationStore(tmp_path / "ops.json"), calendar
        )
        operation_plan = plan(
            create_op(event(title="Первое")),
            create_op(event(title="Второе")),
        )
        with pytest.raises(CalendarOperationError) as raised:
            await apply(pipeline, 192, operation_plan)
        return calendar, raised.value, pipeline.store.find_by_source(
            "telegram-update:192"
        )

    calendar, error, record = asyncio.run(scenario())

    assert error.partially_applied is True
    assert error.retryable is False
    assert error.outcome_uncertain is False
    assert record["stage"] == "partially_rejected"
    assert [item["stage"] for item in record["items"]] == ["applied", "failed"]
    assert len(calendar.events) == 1
    assert [snapshot.title for snapshot in calendar.events.values()] == ["Первое"]


def test_terminal_rejection_replay_never_reissues_provider_write(tmp_path):
    class RejectedCreateCalendar(FakeCalendar):
        async def create_events(self, *, account, events, idempotency_key):
            self.calls.append(("create", idempotency_key))
            raise CalendarWriteRejectedError("sanitized provider rejection")

    async def scenario():
        calendar = RejectedCreateCalendar()
        path = tmp_path / "ops.json"
        first_pipeline = CalendarOperationPipeline(OperationStore(path), calendar)
        operation_plan = plan(create_op())
        with pytest.raises(CalendarOperationError):
            await apply(first_pipeline, 193, operation_plan)

        # Rebuild the pipeline from disk to model a crash after journaling the
        # rejection but before the Telegram error card was persisted.
        second_pipeline = CalendarOperationPipeline(OperationStore(path), calendar)
        with pytest.raises(CalendarOperationError) as replayed:
            await apply(second_pipeline, 193, operation_plan)
        return calendar, replayed.value, second_pipeline.store.find_by_source(
            "telegram-update:193"
        )

    calendar, error, record = asyncio.run(scenario())

    assert [call[0] for call in calendar.calls] == ["create"]
    assert error.retryable is False
    assert error.outcome_uncertain is False
    assert error.partially_applied is False
    assert record["stage"] == "rejected"


@pytest.mark.parametrize("action", ["update", "delete"])
@pytest.mark.parametrize(
    "event_changes",
    [
        {"recurrence_rrules": ("RRULE:FREQ=WEEKLY",)},
        {
            "attendee_emails": ("guest@example.com",),
            "organizer_is_self": False,
        },
    ],
)
def test_policy_metadata_on_allowlisted_event_does_not_block_mutation(
    tmp_path, action, event_changes
):
    async def scenario():
        calendar = FakeCalendar()
        pipeline = CalendarOperationPipeline(OperationStore(tmp_path / "ops.json"), calendar)
        event_id = f"unsafe-{action}"
        provider = provider_event(event_id, **event_changes)
        calendar.events[event_id] = provider
        pipeline.observe_lookup_events("personal", [provider])
        operation = (
            update_op(event_id, {"location": "Переговорная А"})
            if action == "update"
            else delete_op(event_id)
        )
        result = await apply(pipeline, 90, plan(operation))
        return calendar, result

    calendar, result = asyncio.run(scenario())

    assert result.stage == "applied"
    assert [item["stage"] for item in result.record["items"]] == ["applied"]
    assert [call[0] for call in calendar.calls] == ["get", action]


def test_self_organized_event_with_attendees_can_be_updated(tmp_path):
    async def scenario():
        calendar = FakeCalendar()
        pipeline = CalendarOperationPipeline(OperationStore(tmp_path / "ops.json"), calendar)
        provider = provider_event(
            "self-organized",
            attendee_emails=("guest@example.com",),
            organizer_is_self=True,
        )
        calendar.events[provider.event_id] = provider
        pipeline.observe_lookup_events("personal", [provider])
        result = await apply(
            pipeline,
            91,
            plan(update_op(provider.event_id, {"location": "Переговорная А"})),
            displayed_candidates=(provider,),
        )
        context = pipeline.context(account="personal", chat_id=OWNER, now=NOW)
        return calendar, result, context

    calendar, result, context = asyncio.run(scenario())

    assert result.stage == "applied"
    assert [
        (event["display_index"], event["event_id"])
        for event in result.record["displayed_candidates"]
    ] == [(1, "self-organized")]
    assert [
        (event["display_index"], event["event_id"])
        for event in context.application_state["candidate_events"]
    ] == [(1, "self-organized")]
    assert any(call[0] == "update" for call in calendar.calls)


def test_batch_create_persists_success_card_order_for_relative_followup(tmp_path):
    async def scenario():
        calendar = FakeCalendar()
        pipeline = CalendarOperationPipeline(
            OperationStore(tmp_path / "ops.json"), calendar
        )
        result = await apply(
            pipeline,
            191,
            plan(
                create_op(event(title="Первое событие")),
                create_op(
                    event(
                        title="Второе событие",
                        start_at="2026-08-25T17:30:00+03:00",
                        end_at="2026-08-25T18:00:00+03:00",
                    )
                ),
            ),
        )
        context = pipeline.context(account="personal", chat_id=OWNER, now=NOW)
        return result, context

    result, context = asyncio.run(scenario())

    result_rows = [
        (candidate["display_index"], candidate["event_id"])
        for candidate in result.record["displayed_candidates"]
    ]
    context_rows = [
        (candidate["display_index"], candidate["event_id"])
        for candidate in context.application_state["candidate_events"]
    ]
    assert result_rows == [
        (1, result.record["items"][0]["after"]["event_id"]),
        (2, result.record["items"][1]["after"]["event_id"]),
    ]
    assert context_rows == result_rows
    assert context.allowed_event_ids == tuple(event_id for _, event_id in result_rows)


def test_batch_update_persists_operation_order_not_cache_recency(tmp_path):
    async def scenario():
        calendar = FakeCalendar()
        pipeline = CalendarOperationPipeline(
            OperationStore(tmp_path / "ops.json"), calendar
        )
        first = provider_event("first-update", title="Первое событие")
        second = provider_event("second-update", title="Второе событие")
        calendar.events.update({first.event_id: first, second.event_id: second})
        pipeline.observe_lookup_events("personal", [first, second])
        result = await apply(
            pipeline,
            192,
            plan(
                update_op(second.event_id, {"location": "Второй кабинет"}),
                update_op(first.event_id, {"location": "Первый кабинет"}),
            ),
        )
        context = pipeline.context(account="personal", chat_id=OWNER, now=NOW)
        return result, context

    result, context = asyncio.run(scenario())

    expected = [(1, "second-update"), (2, "first-update")]
    assert [
        (candidate["display_index"], candidate["event_id"])
        for candidate in result.record["displayed_candidates"]
    ] == expected
    assert [
        (candidate["display_index"], candidate["event_id"])
        for candidate in context.application_state["candidate_events"]
    ] == expected
    assert context.allowed_event_ids == ("second-update", "first-update")


def test_mixed_success_omits_deleted_candidate_without_renumbering(tmp_path):
    async def scenario():
        calendar = FakeCalendar()
        pipeline = CalendarOperationPipeline(
            OperationStore(tmp_path / "ops.json"), calendar
        )
        first = provider_event("mixed-first", title="Первое событие")
        second = provider_event("mixed-second", title="Второе событие")
        third = provider_event("mixed-third", title="Третье событие")
        calendar.events.update(
            {event.event_id: event for event in (first, second, third)}
        )
        pipeline.observe_lookup_events("personal", [first, second, third])
        result = await apply(
            pipeline,
            193,
            plan(
                update_op(first.event_id, {"location": "Первый кабинет"}),
                delete_op(second.event_id),
                update_op(third.event_id, {"location": "Третий кабинет"}),
            ),
        )
        context = pipeline.context(account="personal", chat_id=OWNER, now=NOW)
        return result, context

    result, context = asyncio.run(scenario())

    expected = [(1, "mixed-first"), (3, "mixed-third")]
    assert [
        (candidate["display_index"], candidate["event_id"])
        for candidate in result.record["displayed_candidates"]
    ] == expected
    assert [
        (candidate["display_index"], candidate["event_id"])
        for candidate in context.application_state["candidate_events"]
    ] == expected
    assert context.allowed_event_ids == ("mixed-first", "mixed-third")


def test_mixed_batch_deletes_custom_reminders_and_updates_locations(
    tmp_path,
):
    async def scenario():
        calendar = FakeCalendar()
        pipeline = CalendarOperationPipeline(
            OperationStore(tmp_path / "ops.json"), calendar
        )
        all_day_planning = provider_event(
            "all-day-planning",
            title="Командная планёрка",
            start_at="2027-02-01",
            end_at="2027-02-02",
            all_day=True,
            timezone=None,
            location="Переговорная Б",
            reminders_present=True,
            reminders_use_default=False,
            reminder_overrides=(("popup", 30),),
            safety_metadata_fingerprint="planning-custom-reminder",
        )
        all_day_sync = provider_event(
            "all-day-sync",
            title="Проектный созвон",
            start_at="2027-02-01",
            end_at="2027-02-02",
            all_day=True,
            timezone=None,
            location="Переговорная Б",
            reminders_present=True,
            reminders_use_default=False,
            reminder_overrides=(("popup", 30),),
            safety_metadata_fingerprint="sync-custom-reminder",
        )
        timed_sync = provider_event(
            "timed-sync",
            title="Созвон без времени",
            start_at="2027-02-01T13:00:00+01:00",
            end_at="2027-02-01T13:30:00+01:00",
            timezone="Europe/Amsterdam",
        )
        timed_planning = provider_event(
            "timed-planning",
            title="Повторная планёрка",
            start_at="2027-02-01T14:00:00+01:00",
            end_at="2027-02-01T14:30:00+01:00",
            timezone="Europe/Amsterdam",
        )
        candidates = (
            all_day_planning,
            all_day_sync,
            timed_sync,
            timed_planning,
        )
        calendar.events.update(
            {candidate.event_id: candidate for candidate in candidates}
        )
        pipeline.observe_lookup_events("personal", candidates)

        result = await apply(
            pipeline,
            195,
            plan(
                delete_op(all_day_planning.event_id),
                delete_op(all_day_sync.event_id),
                update_op(timed_sync.event_id, {"location": "переговорная А"}),
                update_op(
                    timed_planning.event_id,
                    {"location": "переговорная А"},
                ),
            ),
            displayed_candidates=candidates,
        )
        apply_call_names = [call[0] for call in calendar.calls]
        applied_state = {
            event_id: (
                calendar.events[event_id].status,
                calendar.events[event_id].location,
            )
            for event_id in (
                "all-day-planning",
                "all-day-sync",
                "timed-sync",
                "timed-planning",
            )
        }
        undone = await pipeline.undo(
            operation_id=result.operation_id,
            owner_user_id=OWNER,
            chat_id=OWNER,
        )
        persisted = pipeline.store.get(result.operation_id)
        return result, apply_call_names, applied_state, undone, persisted

    result, apply_call_names, applied_state, undone, persisted = asyncio.run(
        scenario()
    )

    assert result.stage == "applied"
    assert [item["type"] for item in result.record["items"]] == [
        "delete",
        "delete",
        "update",
        "update",
    ]
    assert [item["target_event_id"] for item in result.record["items"]] == [
        "all-day-planning",
        "all-day-sync",
        "timed-sync",
        "timed-planning",
    ]
    assert [item["stage"] for item in result.record["items"]] == [
        "applied",
        "applied",
        "applied",
        "applied",
    ]
    assert [
        item["before"]["reminder_overrides"]
        for item in result.record["items"][:2]
    ] == [[["popup", 30]], [["popup", 30]]]
    assert [item["undo_fidelity"] for item in result.record["items"]] == [
        "core_only",
        "core_only",
        "full",
        "full",
    ]
    assert result.record["undo"]["fidelity"] == "core_only"
    assert apply_call_names == [
        "get",
        "get",
        "get",
        "get",
        "delete",
        "delete",
        "update",
        "update",
    ]
    assert applied_state["all-day-planning"][0] == "cancelled"
    assert applied_state["all-day-sync"][0] == "cancelled"
    assert applied_state["timed-sync"][1] == "переговорная А"
    assert applied_state["timed-planning"][1] == "переговорная А"
    assert undone.outcome == "undone"
    assert undone.record["undo"]["fidelity"] == "core_only"
    assert persisted["undo"]["fidelity"] == "core_only"


def test_pure_delete_pins_an_exact_empty_candidate_set(tmp_path):
    async def scenario():
        calendar = FakeCalendar()
        pipeline = CalendarOperationPipeline(
            OperationStore(tmp_path / "ops.json"), calendar
        )
        deleted = provider_event("deleted-row", title="Удаляемое событие")
        unrelated = provider_event("unrelated-row", title="Другое событие")
        calendar.events.update(
            {deleted.event_id: deleted, unrelated.event_id: unrelated}
        )
        pipeline.observe_lookup_events("personal", [deleted, unrelated])
        result = await apply(
            pipeline,
            194,
            plan(delete_op(deleted.event_id)),
        )
        context = pipeline.context(account="personal", chat_id=OWNER, now=NOW)
        return result, context

    result, context = asyncio.run(scenario())

    assert result.record["displayed_candidates"] == []
    assert context.application_state["candidate_events"] == []
    assert context.allowed_event_ids == ()


@pytest.mark.parametrize(
    "fresh_changes",
    [
        {
            "reminders_present": True,
            "reminders_use_default": False,
            "reminder_overrides": (("popup", 10),),
            "safety_metadata_fingerprint": "custom-reminder",
        },
        {
            "color_id": "5",
            "safety_metadata_fingerprint": "custom-color",
        },
        {
            "has_conference_data": True,
            "safety_metadata_fingerprint": "conference",
        },
        {
            "attendee_emails": ("guest@example.com",),
            "safety_metadata_fingerprint": "attendee",
        },
        {"creator_is_self": None},
    ],
)
def test_fresh_delete_policy_metadata_does_not_block_provider_write(
    tmp_path, fresh_changes
):
    async def scenario():
        calendar = FakeCalendar()
        pipeline = CalendarOperationPipeline(
            OperationStore(tmp_path / "ops.json"), calendar
        )
        cached = provider_event("delete-target")
        pipeline.observe_lookup_events("personal", [cached])
        calendar.events[cached.event_id] = provider_event(
            cached.event_id, **fresh_changes
        )

        result = await apply(pipeline, 92, plan(delete_op(cached.event_id)))
        return calendar, result

    calendar, result = asyncio.run(scenario())

    assert result.stage == "applied"
    assert [item["stage"] for item in result.record["items"]] == ["applied"]
    assert [call[0] for call in calendar.calls] == ["get", "delete"]
    assert calendar.events["delete-target"].status == "cancelled"


def test_fresh_delete_uses_changed_status_snapshot_for_exact_id(tmp_path):
    async def scenario():
        calendar = FakeCalendar()
        pipeline = CalendarOperationPipeline(
            OperationStore(tmp_path / "ops.json"), calendar
        )
        cached = provider_event("delete-status-changed")
        pipeline.observe_lookup_events("personal", [cached])
        calendar.events[cached.event_id] = provider_event(
            cached.event_id, status="tentative"
        )

        result = await apply(pipeline, 920, plan(delete_op(cached.event_id)))
        return calendar, result

    calendar, result = asyncio.run(scenario())

    assert result.stage == "applied"
    assert result.record["items"][0]["before"]["status"] == "tentative"
    assert [call[0] for call in calendar.calls] == ["get", "delete"]
    assert calendar.events["delete-status-changed"].status == "cancelled"


def test_delete_undo_preserves_original_provider_timezone(tmp_path):
    async def scenario():
        calendar = FakeCalendar()
        pipeline = CalendarOperationPipeline(
            OperationStore(tmp_path / "ops.json"), calendar
        )
        original = provider_event(
            "amsterdam-event",
            start_at="2026-08-24T16:30:00+02:00",
            end_at="2026-08-24T17:00:00+02:00",
            timezone="Europe/Amsterdam",
        )
        calendar.events[original.event_id] = original
        pipeline.observe_lookup_events("personal", [original])
        deleted = await apply(
            pipeline, 93, plan(delete_op(original.event_id))
        )
        undone = await pipeline.undo(
            operation_id=deleted.operation_id,
            owner_user_id=OWNER,
            chat_id=OWNER,
        )
        restored = [
            event
            for event in calendar.events.values()
            if event.status == "confirmed" and event.event_id != original.event_id
        ]
        return deleted, undone, restored

    deleted, undone, restored = asyncio.run(scenario())

    assert deleted.record["items"][0]["before"]["timezone"] == "Europe/Amsterdam"
    assert undone.outcome == "undone"
    assert len(restored) == 1
    assert restored[0].timezone == "Europe/Amsterdam"
    assert restored[0].start_at == "2026-08-24T16:30:00+02:00"


def test_delete_undo_restores_all_day_event_without_provider_timezone(tmp_path):
    async def scenario():
        calendar = FakeCalendar()
        pipeline = CalendarOperationPipeline(
            OperationStore(tmp_path / "ops.json"), calendar
        )
        original = provider_event(
            "all-day-event",
            start_at="2026-08-24",
            end_at="2026-08-25",
            all_day=True,
            timezone=None,
        )
        calendar.events[original.event_id] = original
        pipeline.observe_lookup_events("personal", [original])
        deleted = await apply(
            pipeline, 940, plan(delete_op(original.event_id))
        )
        undone = await pipeline.undo(
            operation_id=deleted.operation_id,
            owner_user_id=OWNER,
            chat_id=OWNER,
        )
        restored = [
            event
            for event in calendar.events.values()
            if event.status == "confirmed" and event.event_id != original.event_id
        ]
        return deleted, undone, restored

    deleted, undone, restored = asyncio.run(scenario())

    assert deleted.record["items"][0]["before"]["timezone"] is None
    assert undone.outcome == "undone"
    assert len(restored) == 1
    assert restored[0].all_day is True
    assert restored[0].start_at == "2026-08-24"
    assert restored[0].end_at == "2026-08-25"
    assert restored[0].timezone == "Europe/Moscow"


def test_manual_rich_metadata_change_blocks_old_undo(tmp_path):
    async def scenario():
        calendar = FakeCalendar()
        pipeline = CalendarOperationPipeline(
            OperationStore(tmp_path / "ops.json"), calendar
        )
        created = await apply(pipeline, 94, plan(create_op()))
        event_id = created.record["items"][0]["after"]["event_id"]
        calendar.events[event_id] = replace(
            calendar.events[event_id],
            has_conference_data=True,
            safety_metadata_fingerprint="manually-added-conference",
        )
        writes_before = list(calendar.calls)
        undone = await pipeline.undo(
            operation_id=created.operation_id,
            owner_user_id=OWNER,
            chat_id=OWNER,
        )
        return undone, writes_before, calendar.calls

    undone, writes_before, writes_after = asyncio.run(scenario())

    assert undone.outcome == "blocked"
    assert writes_after[:-1] == writes_before
    assert writes_after[-1][0] == "get"
    assert not any(call[0] == "delete" for call in writes_after)

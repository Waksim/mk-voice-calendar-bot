import pytest

from tg_voice_transcriber_bot.intent import (
    CALENDAR_OPERATION_SCHEMA,
    format_calendar_preview,
    validate_calendar_intent,
    validate_calendar_operation_plan,
)


def valid_event():
    return {
        "action": "create",
        "events": [
            {
                "title": "Встреча с Анной",
                "start_at": "2026-08-24T15:00:00+03:00",
                "end_at": "2026-08-24T16:00:00+03:00",
                "all_day": False,
                "timezone": "Europe/Moscow",
                "location": "Офис",
                "description": None,
                "recurrence_rrule": None,
            }
        ],
        "clarification_question": None,
        "confidence": 0.96,
    }


def test_valid_event_is_normalized_and_previewed():
    intent = validate_calendar_intent(valid_event())
    preview = format_calendar_preview("В понедельник встреча", intent)

    assert intent["confidence"] == 0.96
    assert "Встреча с Анной" in preview
    assert "2026-08-24T15:00:00+03:00" in preview
    assert "предпросмотр" in preview


def test_end_must_be_after_start():
    payload = valid_event()
    payload["events"][0]["end_at"] = payload["events"][0]["start_at"]

    with pytest.raises(ValueError, match="after"):
        validate_calendar_intent(payload)


def test_clarification_cannot_sneak_in_an_event():
    payload = valid_event()
    payload["action"] = "clarify"
    payload["clarification_question"] = "Во сколько?"

    with pytest.raises(ValueError, match="clarify"):
        validate_calendar_intent(payload)


def test_all_day_requires_exclusive_end_date():
    payload = valid_event()
    payload["events"][0].update(
        {
            "start_at": "2026-08-24",
            "end_at": "2026-08-25",
            "all_day": True,
        }
    )

    assert validate_calendar_intent(payload)["events"][0]["all_day"] is True


def test_rrule_prefix_is_enforced():
    payload = valid_event()
    payload["events"][0]["recurrence_rrule"] = "FREQ=WEEKLY"

    with pytest.raises(ValueError, match="RRULE"):
        validate_calendar_intent(payload)


def test_rrule_rejects_injected_ics_line():
    payload = valid_event()
    payload["events"][0]["recurrence_rrule"] = (
        "RRULE:FREQ=WEEKLY\nATTENDEE:someone@example.com"
    )

    with pytest.raises(ValueError, match="characters"):
        validate_calendar_intent(payload)


def test_common_rrule_is_allowed():
    payload = valid_event()
    payload["events"][0]["recurrence_rrule"] = (
        "RRULE:FREQ=WEEKLY;INTERVAL=2;BYDAY=TU,TH;COUNT=10"
    )

    assert validate_calendar_intent(payload)["events"][0][
        "recurrence_rrule"
    ].endswith("COUNT=10")


def test_explicit_offset_must_match_moscow_timezone():
    payload = valid_event()
    payload["events"][0]["start_at"] = "2026-08-24T15:00:00+05:00"
    payload["events"][0]["end_at"] = "2026-08-24T16:00:00+05:00"

    with pytest.raises(ValueError, match="offset"):
        validate_calendar_intent(payload)


def create_operation(event=None):
    return {
        "type": "create",
        "target_event_id": None,
        "event": event or valid_event()["events"][0],
        "patch": None,
        "clear_fields": [],
    }


def update_operation(event_id="known-event", *, patch=None, clear_fields=None):
    return {
        "type": "update",
        "target_event_id": event_id,
        "event": None,
        "patch": patch,
        "clear_fields": clear_fields or [],
    }


def delete_operation(event_id="known-event"):
    return {
        "type": "delete",
        "target_event_id": event_id,
        "event": None,
        "patch": None,
        "clear_fields": [],
    }


def operation_plan(*operations):
    return {
        "action": "execute",
        "operations": list(operations),
        "clarification_question": None,
        "confidence": 0.94,
    }


def test_operation_schema_exposes_all_mutation_types_and_five_item_limit():
    operations = CALENDAR_OPERATION_SCHEMA["properties"]["operations"]
    operation_type = operations["items"]["properties"]["type"]

    assert operations["maxItems"] == 5
    assert operation_type["enum"] == ["create", "update", "delete"]


def test_create_operation_requires_and_normalizes_a_complete_event():
    payload = operation_plan(create_operation())

    plan = validate_calendar_operation_plan(payload, set())

    assert plan["action"] == "execute"
    assert plan["confidence"] == 0.94
    assert plan["operations"][0]["event"]["title"] == "Встреча с Анной"
    assert plan["operations"][0]["patch"] is None


def test_create_operation_rejects_partial_event_and_target_id():
    partial = {"title": "Недостаточно полей"}
    payload = operation_plan(create_operation(partial))

    with pytest.raises(ValueError, match="event"):
        validate_calendar_operation_plan(payload, set())

    payload = operation_plan(create_operation())
    payload["operations"][0]["target_event_id"] = "known-event"
    with pytest.raises(ValueError, match="create"):
        validate_calendar_operation_plan(payload, {"known-event"})


def test_update_preserves_omitted_fields_and_uses_explicit_clear_fields():
    payload = operation_plan(
        update_operation(
            patch={"location": "  переговорная А  "},
            clear_fields=["description"],
        )
    )

    plan = validate_calendar_operation_plan(payload, {"known-event"})
    operation = plan["operations"][0]

    assert operation["patch"] == {"location": "переговорная А"}
    assert "title" not in operation["patch"]
    assert operation["clear_fields"] == ["description"]


def test_update_rejects_null_or_blank_patch_instead_of_accidentally_clearing():
    for value in (None, "   "):
        payload = operation_plan(update_operation(patch={"location": value}))

        with pytest.raises(ValueError, match="patch location"):
            validate_calendar_operation_plan(payload, {"known-event"})


def test_update_can_clear_only_supported_nullable_calendar_fields():
    valid = operation_plan(
        update_operation(patch=None, clear_fields=["description"])
    )
    assert validate_calendar_operation_plan(valid, {"known-event"})[
        "operations"
    ][0]["clear_fields"] == ["description"]

    for field in ("title", "recurrence_rrule"):
        invalid = operation_plan(
            update_operation(patch=None, clear_fields=[field])
        )
        with pytest.raises(ValueError, match="clear_fields"):
            validate_calendar_operation_plan(invalid, {"known-event"})


def test_update_cannot_patch_and_clear_the_same_field():
    payload = operation_plan(
        update_operation(
            patch={"description": "Новое описание"},
            clear_fields=["description"],
        )
    )

    with pytest.raises(ValueError, match="patched and cleared"):
        validate_calendar_operation_plan(payload, {"known-event"})


def test_update_and_delete_require_an_exact_allowlisted_event_id():
    for operation in (update_operation(patch={"title": "Новое"}), delete_operation()):
        payload = operation_plan(operation)
        with pytest.raises(ValueError, match="known calendar event"):
            validate_calendar_operation_plan(payload, {"some-other-event"})


def test_delete_rejects_event_data_patch_or_clear_fields():
    payload = operation_plan(delete_operation())
    payload["operations"][0]["patch"] = {"title": "Не должно быть"}
    with pytest.raises(ValueError, match="delete"):
        validate_calendar_operation_plan(payload, {"known-event"})

    payload = operation_plan(delete_operation())
    payload["operations"][0]["event"] = valid_event()["events"][0]
    with pytest.raises(ValueError, match="delete"):
        validate_calendar_operation_plan(payload, {"known-event"})


def test_multiple_different_operations_are_allowed():
    payload = operation_plan(
        create_operation(),
        update_operation("first", patch={"title": "Перенесённая встреча"}),
        delete_operation("second"),
    )

    plan = validate_calendar_operation_plan(payload, {"first", "second"})

    assert [item["type"] for item in plan["operations"]] == [
        "create",
        "update",
        "delete",
    ]


def test_same_known_event_cannot_be_targeted_twice_in_one_plan():
    payload = operation_plan(
        update_operation(patch={"title": "Новое"}),
        delete_operation(),
    )

    with pytest.raises(ValueError, match="only once"):
        validate_calendar_operation_plan(payload, {"known-event"})


def test_temporal_patch_enforces_iso_order_and_configured_timezone_offset():
    reversed_times = operation_plan(
        update_operation(
            patch={
                "start_at": "2026-08-24T17:00:00+03:00",
                "end_at": "2026-08-24T16:00:00+03:00",
            }
        )
    )
    with pytest.raises(ValueError, match="after"):
        validate_calendar_operation_plan(reversed_times, {"known-event"})

    wrong_offset = operation_plan(
        update_operation(patch={"start_at": "2026-08-24T15:00:00+05:00"})
    )
    with pytest.raises(ValueError, match="offset"):
        validate_calendar_operation_plan(wrong_offset, {"known-event"})


@pytest.mark.parametrize(
    ("all_day", "start_at", "end_at"),
    [
        (True, "2026-08-25", "2026-08-26"),
        (
            False,
            "2026-08-25T15:00:00+03:00",
            "2026-08-25T16:00:00+03:00",
        ),
    ],
)
def test_all_day_patch_requires_complete_bounds_in_matching_format(
    all_day, start_at, end_at
):
    payload = operation_plan(
        update_operation(
            patch={
                "start_at": start_at,
                "end_at": end_at,
                "all_day": all_day,
            }
        )
    )

    patch = validate_calendar_operation_plan(payload, {"known-event"})[
        "operations"
    ][0]["patch"]

    assert patch["all_day"] is all_day
    assert patch["start_at"] == start_at
    assert patch["end_at"] == end_at


@pytest.mark.parametrize(
    "patch",
    [
        {"all_day": True},
        {"all_day": False},
        {"all_day": True, "start_at": "2026-08-25"},
        {"all_day": False, "end_at": "2026-08-25T16:00:00+03:00"},
    ],
)
def test_all_day_patch_rejects_missing_temporal_bound(patch):
    payload = operation_plan(update_operation(patch=patch))

    with pytest.raises(ValueError, match="requires both start_at and end_at"):
        validate_calendar_operation_plan(payload, {"known-event"})


@pytest.mark.parametrize(
    "patch",
    [
        {
            "all_day": True,
            "start_at": "2026-08-25T15:00:00+03:00",
            "end_at": "2026-08-25T16:00:00+03:00",
        },
        {
            "all_day": False,
            "start_at": "2026-08-25",
            "end_at": "2026-08-26",
        },
        {
            "all_day": True,
            "start_at": "2026-08-25",
            "end_at": "2026-08-26T16:00:00+03:00",
        },
        {
            "all_day": False,
            "start_at": "2026-08-25T15:00:00+03:00",
            "end_at": "2026-08-26",
        },
    ],
)
def test_all_day_patch_rejects_wrong_or_mixed_temporal_formats(patch):
    payload = operation_plan(update_operation(patch=patch))

    with pytest.raises(ValueError, match="YYYY-MM-DD|RFC3339|UTC offset"):
        validate_calendar_operation_plan(payload, {"known-event"})


@pytest.mark.parametrize(
    "patch",
    [
        {"start_at": "2026-08-25"},
        {"start_at": "2026-08-25T15:00:00+03:00"},
    ],
)
def test_partial_start_patch_remains_valid_without_all_day(patch):
    payload = operation_plan(update_operation(patch=patch))

    assert validate_calendar_operation_plan(payload, {"known-event"})[
        "operations"
    ][0]["patch"] == patch


def test_execute_clarify_and_ignore_have_disjoint_payloads():
    clarify = {
        "action": "clarify",
        "operations": [],
        "clarification_question": "Какое событие изменить?",
        "confidence": 0.4,
    }
    assert validate_calendar_operation_plan(clarify, set())[
        "clarification_question"
    ] == "Какое событие изменить?"

    ignore = {
        "action": "ignore",
        "operations": [],
        "clarification_question": None,
        "confidence": 0.99,
    }
    assert validate_calendar_operation_plan(ignore, set())["action"] == "ignore"

    clarify["operations"] = [create_operation()]
    with pytest.raises(ValueError, match="clarify"):
        validate_calendar_operation_plan(clarify, set())


def test_read_and_lookup_require_a_bounded_timezone_aware_range():
    for action in ("read", "lookup"):
        payload = {
            "action": action,
            "operations": [],
            "lookup": {
                "query": "  планёрка  ",
                "time_min": "2026-08-23T00:00:00+03:00",
                "time_max": "2026-08-30T00:00:00+03:00",
            },
            "clarification_question": None,
            "confidence": 0.96,
        }
        normalized = validate_calendar_operation_plan(payload, set())
        assert normalized["action"] == action
        assert normalized["lookup"] == {
            "query": "планёрка",
            "time_min": "2026-08-23T00:00:00+03:00",
            "time_max": "2026-08-30T00:00:00+03:00",
        }


@pytest.mark.parametrize(
    ("time_min", "time_max", "error"),
    [
        (
            "2026-08-23T00:00:00",
            "2026-08-30T00:00:00+03:00",
            "offset",
        ),
        (
            "2026-08-30T00:00:00+03:00",
            "2026-08-23T00:00:00+03:00",
            "after",
        ),
        (
            "2026-08-23T00:00:00+03:00",
            "2026-09-24T00:00:00+03:00",
            "31 days",
        ),
    ],
)
def test_lookup_rejects_invalid_time_ranges(time_min, time_max, error):
    payload = {
        "action": "read",
        "operations": [],
        "lookup": {"query": None, "time_min": time_min, "time_max": time_max},
        "clarification_question": None,
        "confidence": 0.9,
    }
    with pytest.raises(ValueError, match=error):
        validate_calendar_operation_plan(payload, set())


def test_lookup_data_is_disjoint_from_mutation_and_clarification():
    lookup = {
        "query": None,
        "time_min": "2026-08-23T00:00:00+03:00",
        "time_max": "2026-08-24T00:00:00+03:00",
    }
    execute = operation_plan(create_operation())
    execute["lookup"] = lookup
    with pytest.raises(ValueError, match="execute"):
        validate_calendar_operation_plan(execute, set())

    clarify = {
        "action": "clarify",
        "operations": [],
        "lookup": lookup,
        "clarification_question": "Какое событие?",
        "confidence": 0.5,
    }
    with pytest.raises(ValueError, match="clarify"):
        validate_calendar_operation_plan(clarify, set())


def test_operation_plan_rejects_more_than_five_operations_and_extra_keys():
    payload = operation_plan(*(create_operation() for _ in range(6)))
    with pytest.raises(ValueError, match="five"):
        validate_calendar_operation_plan(payload, set())

    payload = operation_plan(create_operation())
    payload["unexpected"] = True
    with pytest.raises(ValueError, match="unexpected"):
        validate_calendar_operation_plan(payload, set())

from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import pytest

from tg_voice_transcriber_bot.fast_read import plan_fast_calendar_read


MOSCOW = ZoneInfo("Europe/Moscow")
REFERENCE_TIME = datetime(2026, 8, 24, 15, 45, 49, tzinfo=MOSCOW)


def test_nearest_hour_question_is_a_validated_read_without_a_query():
    plan = plan_fast_calendar_read(
        "Какие у меня события в ближайший час?",
        reference_time=REFERENCE_TIME,
    )

    assert plan == {
        "action": "read",
        "operations": [],
        "lookup": {
            "query": None,
            "time_min": "2026-08-24T15:45:49+03:00",
            "time_max": "2026-08-24T16:45:49+03:00",
        },
        "clarification_question": None,
        "confidence": 1.0,
    }


def test_telegram_unicode_isolates_and_punctuation_are_ignored():
    plan = plan_fast_calendar_read(
        "\u200e\u2068Какие у меня события в ближайший час?!\u2069",
        reference_time=REFERENCE_TIME,
    )

    assert plan is not None
    assert plan["lookup"]["time_max"] == "2026-08-24T16:45:49+03:00"


@pytest.mark.parametrize(
    ("text", "time_min", "time_max"),
    [
        (
            "Покажи мои встречи на сегодня",
            "2026-08-24T00:00:00+03:00",
            "2026-08-25T00:00:00+03:00",
        ),
        (
            "Что у меня запланировано завтра?",
            "2026-08-25T00:00:00+03:00",
            "2026-08-26T00:00:00+03:00",
        ),
        (
            "Какие мероприятия у меня послезавтра",
            "2026-08-26T00:00:00+03:00",
            "2026-08-27T00:00:00+03:00",
        ),
        (
            "Покажи календарь до конца дня",
            "2026-08-24T15:45:49+03:00",
            "2026-08-25T00:00:00+03:00",
        ),
        (
            "Перечисли мои события в следующие 3 часа",
            "2026-08-24T15:45:49+03:00",
            "2026-08-24T18:45:49+03:00",
        ),
        (
            "Есть ли у меня встречи в течение 2 дней",
            "2026-08-24T15:45:49+03:00",
            "2026-08-26T15:45:49+03:00",
        ),
    ],
)
def test_supported_bounded_windows(text, time_min, time_max):
    plan = plan_fast_calendar_read(text, reference_time=REFERENCE_TIME)

    assert plan is not None
    assert plan["lookup"] == {
        "query": None,
        "time_min": time_min,
        "time_max": time_max,
    }


def test_reference_time_is_converted_to_the_configured_calendar_timezone():
    utc_reference = REFERENCE_TIME.astimezone(timezone.utc)

    plan = plan_fast_calendar_read(
        "Какие события в следующий час",
        reference_time=utc_reference,
    )

    assert plan is not None
    assert plan["lookup"]["time_min"] == "2026-08-24T15:45:49+03:00"


@pytest.mark.parametrize(
    "text",
    [
        "Добавь событие на завтра",
        "Покажи события на сегодня и удали их",
        "Перенеси мои встречи на завтра",
        "Отмени все мероприятия сегодня",
        "Покажи встречи с Иваном на завтра",
        "Какие у меня события?",
        "Что у меня сегодня?",
        "Какая погода сегодня?",
        "Найди эндокринолога завтра",
        "Покажи календарь",
        "События врача в ближайший час",
    ],
)
def test_mutations_specific_searches_and_ambiguous_text_fall_back(text):
    assert (
        plan_fast_calendar_read(text, reference_time=REFERENCE_TIME) is None
    )


@pytest.mark.parametrize(
    "text",
    [
        "Какие события в ближайшие 745 часов?",
        "Покажи встречи на следующие 32 дня",
        "Какие мероприятия в течение 0 часов",
    ],
)
def test_ranges_are_bounded_and_positive(text):
    assert (
        plan_fast_calendar_read(text, reference_time=REFERENCE_TIME) is None
    )


def test_maximum_31_day_window_is_accepted():
    plan = plan_fast_calendar_read(
        "Покажи мероприятия на ближайшие 31 день",
        reference_time=REFERENCE_TIME,
    )

    assert plan is not None
    assert datetime.fromisoformat(plan["lookup"]["time_max"]) - datetime.fromisoformat(
        plan["lookup"]["time_min"]
    ) == timedelta(days=31)


def test_naive_reference_time_is_rejected():
    with pytest.raises(ValueError, match="timezone-aware"):
        plan_fast_calendar_read(
            "Какие события в ближайший час",
            reference_time=REFERENCE_TIME.replace(tzinfo=None),
        )

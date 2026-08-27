import pytest

from tg_voice_transcriber_bot import ui
from tg_voice_transcriber_bot.text import utf16_units
from tg_voice_transcriber_bot.ui import (
    FieldChange,
    format_clarify_card,
    format_create_card,
    format_delete_card,
    format_duration,
    format_error_card,
    format_ignore_card,
    format_lookup_clarify_card,
    format_mixed_operation_card,
    format_progress_card,
    format_read_card,
    format_undo_card,
    format_update_card,
    parse_undo_callback,
    safe_google_calendar_link,
    undo_reply_markup,
)


def event(**overrides):
    value = {
        "title": "Планёрка",
        "start_at": "2026-08-24T15:00:00+03:00",
        "end_at": "2026-08-24T16:00:00+03:00",
        "all_day": False,
        "timezone": "Europe/Moscow",
        "location": "переговорная А",
        "description": "Первая встреча",
        "recurrence_rrule": None,
    }
    value.update(overrides)
    return value


def test_progress_cards_are_single_safe_html_messages():
    matching = format_progress_card("matching")
    transcribing = format_progress_card("transcribing")
    gemini = format_progress_card("gemini")
    text_gemini = format_progress_card("gemini", input_kind="text")
    image_downloading = format_progress_card("image_downloading", input_kind="image")
    image_vision = format_progress_card("vision", input_kind="image")
    image_gemini = format_progress_card("gemini", input_kind="image")
    captioned_image = format_progress_card("gemini", input_kind="text_and_image")
    lookup = format_progress_card("calendar_lookup")
    matching_candidates = format_progress_card("gemini_match")
    calendar = format_progress_card("calendar", action="update")
    reading = format_progress_card("calendar", action="read")

    assert "Ищу сообщение в Telegram" in matching
    assert "Получаю расшифровку" in transcribing
    assert "ИИ-планировщик разбирает" in gemini
    assert "OpenRouter" not in gemini
    assert "Расшифровка Telegram получена" in gemini
    assert "Текстовая команда получена" in text_gemini
    assert "Обрабатываю текстовую команду" in text_gemini
    assert "Загружаю изображение из Telegram" in image_downloading
    assert "Извлекаю текст и детали" in image_vision
    assert "Данные изображения распознаны" in image_gemini
    assert "изображение с подписью" in captioned_image
    assert "Ищу события в Google Calendar" in lookup
    assert "ИИ-планировщик выбирает точную запись" in matching_candidates
    assert "ИИ-планировщик: изменить событие" in calendar
    assert "Обновляю событие" in calendar
    assert "Читаю события" in reading
    assert all(
        utf16_units(ui._visible_text(card)) <= 4096
        for card in (
            matching,
            transcribing,
            gemini,
            text_gemini,
            image_downloading,
            image_vision,
            image_gemini,
            captioned_image,
            lookup,
            matching_candidates,
            calendar,
            reading,
        )
    )

    with pytest.raises(ValueError, match="phase"):
        format_progress_card("unknown")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="input kind"):
        format_progress_card("gemini", input_kind="unknown")  # type: ignore[arg-type]


def test_progress_card_shows_bounded_escaped_extraction_preview():
    voice = format_progress_card(
        "gemini",
        input_kind="voice",
        transcript='Позвонить <b>Бабе Тане</b> & записать\nна завтра',
    )
    image = format_progress_card(
        "calendar",
        action="create",
        input_kind="text_and_image",
        transcript="Добавь эту бронь",
        image_description="Экран <оплаты>",
        image_visible_text="29 августа\n8:00–10:00\nLunda & Padel",
    )
    huge = format_progress_card(
        "gemini",
        input_kind="text_and_image",
        transcript="💬" * 5_000,
        image_description="🖼" * 5_000,
        image_visible_text="🔤" * 5_000,
    )

    assert "🔎 <b>Извлечённые данные</b>" in voice
    assert "🎙️ <b>Расшифровка Telegram</b>" in voice
    assert "&lt;b&gt;Бабе Тане&lt;/b&gt; &amp; записать\nна завтра" in voice
    assert "💬 <b>Подпись</b>" in image
    assert "🖼️ <b>Описание изображения</b>" in image
    assert "🔤 <b>Текст на изображении</b>" in image
    assert "Экран &lt;оплаты&gt;" in image
    assert "29 августа\n8:00–10:00\nLunda &amp; Padel" in image
    assert "…" in huge
    assert utf16_units(ui._visible_text(huge)) <= 4096


def test_create_card_is_localized_linked_and_escaped():
    malicious = '</b><a href="https://evil.example">click</a>&'
    card = format_create_card(
        [
            event(
                title=malicious,
                location="<script>metro</script>",
                description='A & B "quoted"',
            )
        ],
        transcript="Поставь <b>встречу</b> & позвони",
        elapsed_seconds=4.75,
        html_links=["https://calendar.google.com/calendar/event?eid=a&x=1"],
    )

    assert "✅ <b>Добавлено в календарь</b>" in card
    assert "понедельник, 24 августа 2026" in card
    assert "🕒 15:00–16:00" in card
    assert "<blockquote expandable>" in card
    assert "Готово за <b>4,8 с</b>" in card
    assert malicious not in card
    assert "&lt;/b&gt;&lt;a href=&quot;" in card
    assert "&lt;script&gt;metro&lt;/script&gt;" in card
    assert "Поставь &lt;b&gt;встречу&lt;/b&gt; &amp; позвони" in card
    assert "💬 Команда" in card
    assert (
        'href="https://calendar.google.com/calendar/event?eid=a&amp;x=1"'
        in card
    )


def test_event_time_is_rendered_in_its_named_timezone():
    card = format_create_card(
        [
            event(
                start_at="2026-08-24T10:30:00+02:00",
                end_at="2026-08-24T11:00:00+02:00",
                timezone="Europe/Moscow",
            )
        ],
        transcript="Поставь встречу с 11:30 до 12:00",
        elapsed_seconds=1,
    )

    assert "🕒 11:30–12:00" in card
    assert "10:30–11:00" not in card


def test_event_timezone_conversion_updates_date_and_unknown_zone_falls_back():
    converted = format_create_card(
        [
            event(
                start_at="2026-08-24T23:30:00+00:00",
                end_at="2026-08-25T00:30:00+00:00",
                timezone="Asia/Tokyo",
            )
        ],
        transcript="Поставь встречу",
        elapsed_seconds=1,
    )
    fallback = format_create_card(
        [
            event(
                start_at="2026-08-24T10:30:00+02:00",
                end_at="2026-08-24T11:00:00+02:00",
                timezone="Unknown/Calendar-Zone",
            )
        ],
        transcript="Поставь встречу",
        elapsed_seconds=1,
    )

    assert "вторник, 25 августа 2026" in converted
    assert "🕒 08:30–09:30" in converted
    assert "🕒 10:30–11:00" in fallback


def test_update_delete_clarify_error_and_undo_cards_have_distinct_outcomes():
    update = format_update_card(
        [event()],
        transcript="Добавь переговорная А",
        elapsed_seconds=3.6,
        changes=[FieldChange("location", None, "переговорная А")],
    )
    deleted = format_delete_card(
        [event(html_link="https://calendar.google.com/event")],
        transcript="Удали планёрку",
        elapsed_seconds=3.1,
    )
    clarify = format_clarify_card(
        "Во сколько поставить <планёрку>?",
        transcript="Перенеси его на послезавтра",
        elapsed_seconds=2.7,
    )
    error = format_error_card(
        "Gemini ответила <невалидно> & повторите",
        transcript="Моя команда",
        elapsed_seconds=1.25,
    )
    undone = format_undo_card(
        "update", ["Планёрка <важно>"], elapsed_seconds=1.2
    )

    assert "✏️ <b>Событие обновлено</b>" in update
    assert "Место: <s>не указано</s> → <b>переговорная А</b>" in update
    assert "🗑️ <b>Событие удалено</b>" in deleted
    assert "href=" not in deleted
    assert "🤔 <b>Нужно уточнение</b>" in clarify
    assert "&lt;планёрку&gt;" in clarify
    assert "❌ <b>Не удалось выполнить команду</b>" in error
    assert "Google Calendar не изменён" in error
    assert "&lt;невалидно&gt; &amp;" in error
    assert "↩️ <b>Действие отменено</b>" in undone
    assert "возвращены прежние данные" in undone
    assert "&lt;важно&gt;" in undone


def test_ignore_card_uses_provider_neutral_copy():
    card = format_ignore_card(transcript="Привет", elapsed_seconds=1)

    assert "ИИ-планировщик не нашёл" in card
    assert "Muse" not in card
    assert "OpenRouter" not in card
    assert "Gemini" not in card


def test_model_badge_is_escaped_and_precedes_timing_in_every_result_card():
    model_name = 'Gemini </b><script>alert("x")</script> & Co'
    escaped_badge = (
        "🤖 <code>Gemini &lt;/b&gt;&lt;script&gt;alert(&quot;x&quot;)"
        "&lt;/script&gt; &amp; Co</code>"
    )
    cards = [
        format_create_card(
            [event()],
            transcript="Создай встречу",
            elapsed_seconds=1,
            model_name=model_name,
        ),
        format_update_card(
            [event()],
            transcript="Измени встречу",
            elapsed_seconds=1,
            model_name=model_name,
        ),
        format_delete_card(
            [event()],
            transcript="Удали встречу",
            elapsed_seconds=1,
            model_name=model_name,
        ),
        format_mixed_operation_card(
            [{"type": "create", "event": event()}],
            transcript="Обнови календарь",
            elapsed_seconds=1,
            model_name=model_name,
        ),
        format_read_card(
            [event()],
            transcript="Что в календаре?",
            elapsed_seconds=1,
            model_name=model_name,
        ),
        format_lookup_clarify_card(
            "Какое событие?",
            [event()],
            transcript="Перенеси встречу",
            elapsed_seconds=1,
            model_name=model_name,
        ),
        format_clarify_card(
            "Во сколько?",
            transcript="Создай встречу",
            elapsed_seconds=1,
            model_name=model_name,
        ),
        format_ignore_card(
            transcript="Привет",
            elapsed_seconds=1,
            model_name=model_name,
        ),
        format_error_card(
            "Google Calendar отклонил операцию",
            transcript="Создай встречу",
            elapsed_seconds=1,
            model_name=model_name,
        ),
    ]

    for card in cards:
        assert escaped_badge in card
        assert model_name not in card
        assert card.index("💬 Команда") < card.index(escaped_badge)
        assert card.index(escaped_badge) < card.index("⏱")
        assert utf16_units(ui._visible_text(card)) <= 4096


def test_empty_model_name_does_not_render_a_model_badge():
    card = format_create_card(
        [event()],
        transcript="Создай встречу",
        elapsed_seconds=1,
        model_name="   ",
    )

    assert "🤖 <code>" not in card


def test_mixed_batch_has_one_bounded_card_and_generic_undo():
    card = format_mixed_operation_card(
        [
            {"type": "create", "event": event(title="Новая встреча")},
            {"type": "delete", "event": event(title="Старая встреча")},
        ],
        transcript="Замени старую встречу новой",
        elapsed_seconds=2.5,
    )
    assert "Календарь обновлён" in card
    assert "Добавлено" in card
    assert "Удалено" in card
    assert utf16_units(ui._visible_text(card)) <= 4096
    assert undo_reply_markup("abcdefghijklmnop", "mixed")["inline_keyboard"][0][0][
        "text"
    ] == "↩️ Отменить операцию"
    assert "Изменения в календаре отменены" in format_undo_card(
        "mixed", ["Новая встреча", "Старая встреча"], elapsed_seconds=1
    )


def test_best_effort_undo_is_explicit_in_cards_markup_and_result_wording():
    deleted = format_delete_card(
        [event(title="Событие с расширенными данными")],
        transcript="Удали событие",
        elapsed_seconds=1,
        best_effort_undo=True,
    )
    mixed = format_mixed_operation_card(
        [
            {"type": "delete", "event": event(title="Старое событие")},
            {"type": "update", "event": event(title="Другое событие")},
        ],
        transcript="Обнови календарь",
        elapsed_seconds=1,
        best_effort_undo=True,
    )
    delete_button = undo_reply_markup(
        "abcdefghijklmnop", "delete", best_effort=True
    )["inline_keyboard"][0][0]
    mixed_button = undo_reply_markup(
        "abcdefghijklmnop", "mixed", best_effort=True
    )["inline_keyboard"][0][0]
    undone = format_undo_card(
        "mixed",
        ["Старое событие", "Другое событие"],
        elapsed_seconds=1,
        best_effort=True,
    )

    assert "восстановит основные поля" in deleted
    assert "основные поля удалённых событий" in mixed
    assert "их расширенные данные могут не вернуться" in mixed
    assert delete_button["text"] == "↩️ Восстановить основные поля"
    assert mixed_button["text"] == "↩️ Отменить основные изменения"
    assert delete_button["callback_data"] == "cal:undo:abcdefghijklmnop"
    assert mixed_button["callback_data"] == "cal:undo:abcdefghijklmnop"
    assert "↩️ <b>Основные изменения отменены</b>" in undone
    assert "расширенные данные могли быть потеряны" in undone


def test_read_card_is_bounded_escaped_and_has_no_undo_control():
    events = [
        event(
            title=f"Встреча {index} <важно>",
            html_link=f"https://calendar.google.com/event?eid={index}",
        )
        for index in range(10)
    ]
    card = format_read_card(
        events,
        transcript="Что у меня <завтра>?",
        elapsed_seconds=2.4,
        total_count=12,
        may_be_incomplete=True,
    )

    assert "📆 <b>События в календаре</b>" in card
    assert "Встреча 0 &lt;важно&gt;" in card
    assert "Встреча 7" in card
    assert "Встреча 8" not in card
    assert "Ещё событий в этом периоде: <b>4</b>" in card
    assert "Результат мог быть сокращён" in card
    assert "Что у меня &lt;завтра&gt;?" in card
    assert "Отмен" not in card
    assert utf16_units(ui._visible_text(card)) <= 4096

    empty = format_read_card(
        [], transcript="Что сегодня?", elapsed_seconds=0.5, total_count=0
    )
    assert "Событий не найдено" in empty


def test_lookup_clarification_shows_only_five_safe_candidates():
    candidates = [event(title=f"Кандидат {index}") for index in range(7)]
    card = format_lookup_clarify_card(
        "Какой именно <планёрка>?",
        candidates,
        transcript="Перенеси планёрку",
        elapsed_seconds=1.8,
    )

    assert "Нужно выбрать событие" in card
    assert "Какой именно &lt;планёрка&gt;?" in card
    assert "Кандидат 4" in card
    assert "Кандидат 5" not in card
    assert "event_id" not in card
    assert "Отмен" not in card


def test_all_day_uses_inclusive_user_facing_end_date():
    card = format_create_card(
        [
            event(
                start_at="2026-08-24",
                end_at="2026-08-27",
                all_day=True,
            )
        ],
        transcript="Отпуск с понедельника по среду",
        elapsed_seconds=1,
    )
    assert "24 августа 2026 — среда, 26 августа 2026 · весь день" in card
    assert "27 августа" not in card


def test_maximal_card_is_bounded_without_broken_template_tags():
    huge = "<&😀" * 5000
    events = [
        event(
            title=huge,
            location=huge,
            description=huge,
            recurrence_rrule="RRULE:" + huge,
        )
        for _ in range(5)
    ]
    changes = [FieldChange(f"custom-{index}", huge, huge + str(index)) for index in range(5)]
    card = format_update_card(
        events,
        transcript=huge,
        elapsed_seconds=59.96,
        changes=changes,
    )

    assert utf16_units(ui._visible_text(card)) <= 4096
    assert card.count("<b>") == card.count("</b>")
    assert card.count("<s>") == card.count("</s>")
    assert card.count("<blockquote expandable>") == card.count("</blockquote>")
    assert "…" in card


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (4.75, "4,8 с"),
        (60, "1 мин 0 с"),
        (3672, "1 ч 1 мин 12 с"),
    ],
)
def test_duration_format(value, expected):
    assert format_duration(value) == expected


def test_duration_rejects_invalid_numbers():
    for value in (-1, float("nan"), float("inf"), True):
        with pytest.raises(ValueError, match="finite"):
            format_duration(value)


def test_google_links_are_strictly_allowlisted():
    assert (
        safe_google_calendar_link("https://calendar.google.com/calendar/event?eid=x")
        == "https://calendar.google.com/calendar/event?eid=x"
    )
    assert safe_google_calendar_link("http://calendar.google.com/event") is None
    assert safe_google_calendar_link("https://calendar.google.com.evil.test/event") is None
    assert safe_google_calendar_link("https://user@calendar.google.com/event") is None
    assert safe_google_calendar_link("https://calendar.google.com:444/event") is None
    assert safe_google_calendar_link("https://calendar.google.com/event\nnext") is None


def test_undo_markup_and_parser_use_an_opaque_bounded_callback():
    operation_id = "abcdefghijklmnop"
    markup = undo_reply_markup(operation_id, "delete")
    button = markup["inline_keyboard"][0][0]

    assert button["text"] == "↩️ Восстановить событие"
    assert button["callback_data"] == f"cal:undo:{operation_id}"
    assert len(button["callback_data"].encode("utf-8")) <= 64
    assert parse_undo_callback(button["callback_data"]) == operation_id

    for invalid in (
        None,
        "cal:add:abcdefghijklmnop",
        "cal:undo:short",
        "cal:undo:abcdefghijklmnop:extra",
        "x" * 65,
    ):
        assert parse_undo_callback(invalid) is None
    with pytest.raises(ValueError, match="operation ID"):
        undo_reply_markup("short", "create")

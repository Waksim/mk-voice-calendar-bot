import asyncio

from tg_voice_transcriber_bot.bot_api import BotApiError
from tg_voice_transcriber_bot.calendar import CreatedCalendarEvent
from tg_voice_transcriber_bot.confirmation import (
    CalendarConfirmationPipeline,
    ConfirmationStore,
)
from tg_voice_transcriber_bot.config import Config
from tg_voice_transcriber_bot.service import (
    START_TEXT,
    VoiceBotService,
    build_calendar_confirmation,
)
from tg_voice_transcriber_bot.state import StateStore


OWNER_ID = 100000001
OTHER_OWNER_ID = 100000002


def create_intent(confidence=0.91):
    return {
        "action": "create",
        "events": [
            {
                "title": "Созвон",
                "start_at": "2026-08-24T10:00:00+03:00",
                "end_at": "2026-08-24T11:00:00+03:00",
                "all_day": False,
                "timezone": "Europe/Moscow",
                "location": None,
                "description": None,
                "recurrence_rrule": None,
            }
        ],
        "clarification_question": None,
        "confidence": confidence,
    }


def clarify_intent():
    return {
        "action": "clarify",
        "events": [],
        "clarification_question": "Во сколько начать встречу?",
        "confidence": 0.72,
    }


class FakeCalendarClient:
    def __init__(self):
        self.calls = 0
        self.created = {}

    async def create_events(self, *, account, events, idempotency_key):
        self.calls += 1
        if idempotency_key not in self.created:
            self.created[idempotency_key] = tuple(
                CreatedCalendarEvent(f"event-{index}")
                for index, _event in enumerate(events, start=1)
            )
        return self.created[idempotency_key]


class FakeBot:
    def __init__(self, updates=None, *, fail_callback_ui=False):
        self.messages = []
        self.callback_answers = []
        self.keyboard_removals = []
        self.updates = list(updates or [])
        self.poll_offsets = []
        self.fail_callback_ui = fail_callback_ui

    async def get_updates(self, offset):
        self.poll_offsets.append(offset)
        if self.updates:
            updates, self.updates = self.updates, []
            return updates
        raise asyncio.CancelledError

    async def send_chat_action(self, chat_id):
        return None

    async def send_text(
        self,
        chat_id,
        text,
        *,
        reply_to_message_id=None,
        reply_markup=None,
    ):
        self.messages.append(
            (chat_id, text, reply_to_message_id, reply_markup)
        )

    async def answer_callback_query(self, callback_query_id, text, *, show_alert=False):
        self.callback_answers.append((callback_query_id, text, show_alert))
        if self.fail_callback_ui:
            raise BotApiError("stale callback")

    async def remove_inline_keyboard(self, chat_id, message_id):
        self.keyboard_removals.append((chat_id, message_id))
        if self.fail_callback_ui:
            raise BotApiError("message no longer editable")


class FakeGateway:
    def __init__(self, transcript="Завтра созвон"):
        self.transcript = transcript
        self.calls = 0

    async def write(self, account, operation, arguments, **kwargs):
        self.calls += 1
        return {"ok": True, "status": "completed", "text": self.transcript}


class FakeGemini:
    def __init__(self, intent):
        self.intent = intent
        self.calls = 0

    async def extract_event(self, transcript, **kwargs):
        self.calls += 1
        return self.intent


def make_pipeline(tmp_path, calendar):
    return CalendarConfirmationPipeline(
        ConfirmationStore(tmp_path / "calendar-confirmations.json"), calendar
    )


def prepare_confirmation(pipeline, source_update_id=77):
    return pipeline.prepare(
        source_update_id=source_update_id,
        account="personal",
        owner_user_id=OWNER_ID,
        chat_id=OWNER_ID,
        intent=create_intent(),
    )


def callback_update(update_id, data, *, user_id=OWNER_ID, callback_id="cb-1"):
    return {
        "update_id": update_id,
        "callback_query": {
            "id": callback_id,
            "from": {"id": user_id},
            "data": data,
            "message": {
                "message_id": 900,
                "chat": {"id": user_id, "type": "private"},
            },
        },
    }


def test_start_copy_requires_confirmation_before_google_calendar_write():
    assert "основной Google Calendar" in START_TEXT
    assert "Muse Spark 1.2" in START_TEXT
    assert "OpenRouter" in START_TEXT
    assert "нажмите «Добавить»" in START_TEXT
    assert "Без этого подтверждения календарь не изменяется" in START_TEXT


def test_status_reports_connected_calendar_confirmation_pipeline(tmp_path):
    async def scenario():
        config = Config(state_path=tmp_path / "state.json")
        bot = FakeBot()
        service = VoiceBotService(
            config,
            bot,
            FakeGateway(),
            StateStore(config.state_path),
            FakeGemini(create_intent()),
            make_pipeline(tmp_path, FakeCalendarClient()),
        )
        await service.handle_update(
            {
                "update_id": 1,
                "message": {
                    "message_id": 2,
                    "from": {"id": OWNER_ID},
                    "chat": {"id": OWNER_ID, "type": "private"},
                    "text": "/status",
                },
            }
        )
        return bot.messages

    messages = asyncio.run(scenario())
    assert len(messages) == 1
    assert "Muse Spark 1.2 через OpenRouter доступна" in messages[0][1]
    assert "Google Calendar подключён" in messages[0][1]
    assert "только после подтверждения" in messages[0][1]


def test_high_confidence_preview_persists_confirmation_and_adds_buttons(tmp_path):
    async def scenario():
        config = Config(
            state_path=tmp_path / "state.json",
            confirmation_state_path=tmp_path / "calendar-confirmations.json",
        )
        state = StateStore(config.state_path)
        state.save_job(
            77,
            {
                "account": "personal",
                "user_message_id": 123,
                "sent_at": 1787400000,
                "status": "found",
            },
        )
        bot = FakeBot()
        gateway = FakeGateway()
        gemini = FakeGemini(create_intent(0.85))
        calendar = FakeCalendarClient()
        pipeline = make_pipeline(tmp_path, calendar)
        service = VoiceBotService(
            config, bot, gateway, state, gemini, pipeline
        )
        await service._process_voice(
            update_id=77,
            account="personal",
            chat_id=OWNER_ID,
            bot_message_id=456,
            sent_at=1787400000,
            duration=3,
            file_size=42,
        )
        return state, bot, gateway, gemini, calendar, pipeline

    state, bot, gateway, gemini, calendar, pipeline = asyncio.run(scenario())
    assert gateway.calls == 1
    assert gemini.calls == 1
    assert calendar.calls == 0
    assert len(bot.messages) == 1
    markup = bot.messages[0][3]
    assert [button["text"] for button in markup["inline_keyboard"][0]] == [
        "Добавить",
        "Отмена",
    ]
    job = state.job(77)
    assert job["status"] == "sent"
    record = pipeline.store.get(job["confirmation_id"])
    assert record["stage"] == "pending"
    assert "нажмите «Добавить»" in bot.messages[0][1]


def test_optional_pipeline_uses_configured_journal_only_with_a_client(tmp_path):
    config = Config(
        confirmation_state_path=tmp_path / "configured-confirmations.json"
    )
    assert build_calendar_confirmation(config, None) is None

    pipeline = build_calendar_confirmation(config, FakeCalendarClient())
    assert pipeline is not None
    assert pipeline.store.path == config.confirmation_state_path
    assert not config.confirmation_state_path.exists()


def test_low_confidence_preview_has_no_buttons(tmp_path):
    async def scenario():
        config = Config(state_path=tmp_path / "state.json")
        state = StateStore(config.state_path)
        state.save_job(
            78,
            {
                "account": "personal",
                "user_message_id": 124,
                "sent_at": 1787400000,
                "status": "found",
            },
        )
        bot = FakeBot()
        calendar = FakeCalendarClient()
        service = VoiceBotService(
            config,
            bot,
            FakeGateway(),
            state,
            FakeGemini(create_intent(0.8499)),
            make_pipeline(tmp_path, calendar),
        )
        await service._process_voice(
            update_id=78,
            account="personal",
            chat_id=OWNER_ID,
            bot_message_id=457,
            sent_at=1787400000,
            duration=3,
            file_size=42,
        )
        return bot, calendar

    bot, calendar = asyncio.run(scenario())
    assert bot.messages[0][3] is None
    assert "ниже 85%" in bot.messages[0][1]
    assert calendar.calls == 0


def test_clarification_is_a_normal_reply_without_buttons(tmp_path):
    async def scenario():
        config = Config(state_path=tmp_path / "state.json")
        state = StateStore(config.state_path)
        state.save_job(
            79,
            {
                "account": "personal",
                "user_message_id": 125,
                "sent_at": 1787400000,
                "status": "found",
            },
        )
        bot = FakeBot()
        calendar = FakeCalendarClient()
        service = VoiceBotService(
            config,
            bot,
            FakeGateway(),
            state,
            FakeGemini(clarify_intent()),
            make_pipeline(tmp_path, calendar),
        )
        await service._process_voice(
            update_id=79,
            account="personal",
            chat_id=OWNER_ID,
            bot_message_id=458,
            sent_at=1787400000,
            duration=3,
            file_size=42,
        )
        return bot, calendar

    bot, calendar = asyncio.run(scenario())
    assert bot.messages[0][3] is None
    assert "Нужно уточнить: Во сколько начать встречу?" in bot.messages[0][1]
    assert calendar.calls == 0


def test_other_allowed_account_cannot_use_owner_confirmation(tmp_path):
    async def scenario():
        calendar = FakeCalendarClient()
        pipeline = make_pipeline(tmp_path, calendar)
        prepared = prepare_confirmation(pipeline)
        add_data = prepared.reply_markup["inline_keyboard"][0][0]["callback_data"]
        bot = FakeBot()
        service = VoiceBotService(
            Config(state_path=tmp_path / "state.json"),
            bot,
            FakeGateway(),
            StateStore(tmp_path / "state.json"),
            FakeGemini(create_intent()),
            pipeline,
        )
        await service.handle_update(
            callback_update(100, add_data, user_id=OTHER_OWNER_ID)
        )
        return bot, calendar, pipeline, prepared

    bot, calendar, pipeline, prepared = asyncio.run(scenario())
    assert calendar.calls == 0
    assert bot.callback_answers == [("cb-1", "Недоступно.", True)]
    assert bot.keyboard_removals == []
    assert pipeline.store.get(prepared.confirmation_id)["stage"] == "pending"


def test_duplicate_callback_after_restart_does_not_duplicate_event(tmp_path):
    async def scenario():
        path = tmp_path / "calendar-confirmations.json"
        calendar = FakeCalendarClient()
        first_pipeline = CalendarConfirmationPipeline(
            ConfirmationStore(path), calendar
        )
        prepared = prepare_confirmation(first_pipeline)
        add_data = prepared.reply_markup["inline_keyboard"][0][0]["callback_data"]
        first_bot = FakeBot()
        first_service = VoiceBotService(
            Config(state_path=tmp_path / "state-1.json"),
            first_bot,
            FakeGateway(),
            StateStore(tmp_path / "state-1.json"),
            FakeGemini(create_intent()),
            first_pipeline,
        )
        await first_service.handle_update(callback_update(101, add_data))

        restored_pipeline = CalendarConfirmationPipeline(
            ConfirmationStore(path), calendar
        )
        restored_bot = FakeBot()
        restored_service = VoiceBotService(
            Config(state_path=tmp_path / "state-2.json"),
            restored_bot,
            FakeGateway(),
            StateStore(tmp_path / "state-2.json"),
            FakeGemini(create_intent()),
            restored_pipeline,
        )
        await restored_service.handle_update(
            callback_update(102, add_data, callback_id="cb-2")
        )
        return calendar, first_bot, restored_bot, restored_pipeline, prepared

    calendar, first_bot, restored_bot, pipeline, prepared = asyncio.run(scenario())
    assert calendar.calls == 1
    assert len(calendar.created) == 1
    assert first_bot.callback_answers[0][1] == "Событие добавлено."
    assert restored_bot.callback_answers[0][1] == "Событие уже добавлено."
    assert pipeline.store.get(prepared.confirmation_id)["stage"] == "created"


def test_cancel_callback_is_persisted_without_calendar_write(tmp_path):
    async def scenario():
        calendar = FakeCalendarClient()
        pipeline = make_pipeline(tmp_path, calendar)
        prepared = prepare_confirmation(pipeline)
        cancel_data = prepared.reply_markup["inline_keyboard"][0][1]["callback_data"]
        bot = FakeBot()
        service = VoiceBotService(
            Config(state_path=tmp_path / "state.json"),
            bot,
            FakeGateway(),
            StateStore(tmp_path / "state.json"),
            FakeGemini(create_intent()),
            pipeline,
        )
        await service.handle_update(callback_update(103, cancel_data))
        return calendar, pipeline, prepared, bot

    calendar, pipeline, prepared, bot = asyncio.run(scenario())
    assert calendar.calls == 0
    assert pipeline.store.get(prepared.confirmation_id)["stage"] == "cancelled"
    assert bot.keyboard_removals == [(OWNER_ID, 900)]
    assert bot.callback_answers[0][1] == "Отменено."


def test_callback_batch_advances_offset_even_if_ui_answer_is_stale(tmp_path):
    async def scenario():
        calendar = FakeCalendarClient()
        pipeline = make_pipeline(tmp_path, calendar)
        prepared = prepare_confirmation(pipeline)
        add_data = prepared.reply_markup["inline_keyboard"][0][0]["callback_data"]
        update = callback_update(110, add_data)
        bot = FakeBot([update], fail_callback_ui=True)
        state = StateStore(tmp_path / "state.json")
        service = VoiceBotService(
            Config(state_path=tmp_path / "state.json"),
            bot,
            FakeGateway(),
            state,
            FakeGemini(create_intent()),
            pipeline,
        )
        try:
            await service.run()
        except asyncio.CancelledError:
            pass
        return state, pipeline, prepared, calendar, bot

    state, pipeline, prepared, calendar, bot = asyncio.run(scenario())
    assert state.offset == 111
    assert bot.poll_offsets == [0, 111]
    assert state.job(110) is None
    assert pipeline.store.get(prepared.confirmation_id)["stage"] == "created"
    assert calendar.calls == 1

import asyncio

from tg_voice_transcriber_bot.config import Config
from tg_voice_transcriber_bot.service import (
    VoiceBotService,
    message_command,
    transcription_reply,
)
from tg_voice_transcriber_bot.state import StateStore


def test_empty_caption_is_not_a_command():
    assert message_command("") == ""
    assert message_command("   ") == ""


def test_completed_transcript_is_returned_verbatim():
    assert transcription_reply(
        {"ok": True, "status": "completed", "text": "Привет, мир"}
    ) == "Привет, мир"


def test_quota_error_is_actionable():
    response = transcription_reply(
        {
            "ok": False,
            "status": "telegram_error",
            "error": "PREMIUM_ACCOUNT_REQUIRED",
        }
    )
    assert "Premium" in response
    assert "квота" in response


def test_initialize_accepts_one_configured_production_account(tmp_path):
    class FakeBot:
        def __init__(self):
            self.configured = False

        async def call(self, method):
            assert method == "getMe"
            return {"username": "mk_voice_text_bot"}

        async def configure(self):
            self.configured = True

    class FakeGateway:
        def __init__(self):
            self.read_accounts = []

        async def validate_operations(self):
            return frozenset({"personal"})

        async def read(self, account, operation, arguments):
            self.read_accounts.append(account)
            assert operation == "get_me"
            assert arguments == {}
            return {"id": 100000001}

    class FakeGemini:
        async def validate(self):
            return None

    async def scenario():
        bot = FakeBot()
        gateway = FakeGateway()
        service = VoiceBotService(
            Config(state_path=tmp_path / "state.json"),
            bot,
            gateway,
            StateStore(tmp_path / "state.json"),
            FakeGemini(),
        )
        await service.initialize()
        return service, bot, gateway

    service, bot, gateway = asyncio.run(scenario())

    assert service.enabled_accounts == frozenset({"personal"})
    assert gateway.read_accounts == ["personal"]
    assert bot.configured is True


def test_voice_from_temporarily_disabled_owner_does_not_call_gateway(tmp_path):
    class FakeBot:
        def __init__(self):
            self.messages = []

        async def send_text(self, chat_id, text, *, reply_to_message_id=None):
            self.messages.append((chat_id, text, reply_to_message_id))

    class FakeGateway:
        async def write(self, *_args, **_kwargs):
            raise AssertionError("disabled account reached Telegram gateway")

    class FakeGemini:
        pass

    async def scenario():
        bot = FakeBot()
        service = VoiceBotService(
            Config(state_path=tmp_path / "state.json"),
            bot,
            FakeGateway(),
            StateStore(tmp_path / "state.json"),
            FakeGemini(),
        )
        service.enabled_accounts = frozenset({"personal"})
        await service.handle_update(
            {
                "update_id": 900,
                "message": {
                    "message_id": 44,
                    "date": 1787400000,
                    "from": {"id": 100000002},
                    "chat": {"id": 100000002, "type": "private"},
                    "voice": {"duration": 3, "file_size": 42},
                },
            }
        )
        return bot

    bot = asyncio.run(scenario())

    assert len(bot.messages) == 1
    assert "временно не подключена" in bot.messages[0][1]


def test_status_does_not_invite_disabled_owner_to_send_voice(tmp_path):
    class FakeBot:
        def __init__(self):
            self.messages = []

        async def send_text(self, chat_id, text, *, reply_to_message_id=None):
            self.messages.append((chat_id, text, reply_to_message_id))

    async def scenario():
        bot = FakeBot()
        service = VoiceBotService(
            Config(state_path=tmp_path / "state.json"),
            bot,
            object(),
            StateStore(tmp_path / "state.json"),
            object(),
        )
        service.enabled_accounts = frozenset({"personal"})
        await service.handle_update(
            {
                "update_id": 901,
                "message": {
                    "message_id": 45,
                    "date": 1787400000,
                    "from": {"id": 100000002},
                    "chat": {"id": 100000002, "type": "private"},
                    "text": "/status",
                },
            }
        )
        return bot

    bot = asyncio.run(scenario())

    assert "временно не подключена" in bot.messages[0][1]
    assert "Пришлите голосовое" not in bot.messages[0][1]


def test_transcript_is_persisted_before_gemini_and_reply_is_not_duplicated(tmp_path):
    class FakeBot:
        def __init__(self):
            self.messages = []

        async def send_chat_action(self, chat_id):
            return None

        async def send_text(self, chat_id, text, *, reply_to_message_id=None):
            self.messages.append((chat_id, text, reply_to_message_id))

    class FakeGateway:
        def __init__(self):
            self.calls = 0

        async def write(self, account, operation, arguments, **kwargs):
            self.calls += 1
            return {"ok": True, "status": "completed", "text": "Завтра созвон"}

    class FakeGemini:
        def __init__(self):
            self.calls = 0

        async def extract_event(self, transcript, **kwargs):
            self.calls += 1
            return {
                "action": "create",
                "events": [
                    {
                        "title": "Созвон",
                        "start_at": "2026-08-23T10:00:00+03:00",
                        "end_at": "2026-08-23T11:00:00+03:00",
                        "all_day": False,
                        "timezone": "Europe/Moscow",
                        "location": None,
                        "description": None,
                        "recurrence_rrule": None,
                    }
                ],
                "clarification_question": None,
                "confidence": 0.9,
            }

    async def scenario():
        config = Config(state_path=tmp_path / "state.json")
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
        gemini = FakeGemini()
        service = VoiceBotService(config, bot, gateway, state, gemini)

        arguments = {
            "update_id": 77,
            "account": "personal",
            "chat_id": 100000001,
            "bot_message_id": 456,
            "sent_at": 1787400000,
            "duration": 3,
            "file_size": 42,
        }
        await service._process_voice(**arguments)
        await service._process_voice(**arguments)
        return state, bot, gateway, gemini

    state, bot, gateway, gemini = asyncio.run(scenario())

    assert state.job(77)["status"] == "sent"
    assert state.job(77)["transcript"] == "Завтра созвон"
    assert gateway.calls == 1
    assert gemini.calls == 1
    assert len(bot.messages) == 1
    assert "Созвон" in bot.messages[0][1]

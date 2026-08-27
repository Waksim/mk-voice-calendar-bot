import asyncio

import pytest

from tg_voice_transcriber_bot.config import Config
from tg_voice_transcriber_bot.gemini import (
    GeminiApiError,
    GeminiAuthenticationError,
    GeminiFallback,
    GeminiProviderChain,
    GeminiProviderStage,
)
from tg_voice_transcriber_bot.gigachat import (
    GigaChatAuthenticationError,
    GigaChatConfigurationError,
    GigaChatQuotaError,
    GigaChatRateLimitError,
    GigaChatRequestRejectedError,
)
from tg_voice_transcriber_bot.openrouter import OpenRouterCreditError
from tg_voice_transcriber_bot.service import (
    VoiceBotService,
    _planner_failure_copy,
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


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (GigaChatAuthenticationError("sensitive"), "авторизацию"),
        (GigaChatConfigurationError("sensitive"), "настройки"),
        (GigaChatRequestRejectedError("sensitive"), "отклонил запрос"),
        (GigaChatQuotaError("sensitive"), "квоты"),
        (GigaChatRateLimitError("sensitive"), "временно ограничили"),
    ],
)
def test_gigachat_failure_copy_is_actionable_and_sanitized(error, expected):
    response = _planner_failure_copy(error)

    assert expected in response
    assert "sensitive" not in response


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


def test_calendar_bot_startup_fails_when_planner_validation_fails(tmp_path):
    class FakeBot:
        async def call(self, method):
            assert method == "getMe"
            return {"username": "mk_voice_text_bot"}

        async def configure(self):
            return None

    class FakeGateway:
        async def validate_operations(self):
            return frozenset({"personal"})

        async def read(self, account, operation, arguments):
            return {"id": 100000001}

    class OutOfCreditPlanner:
        async def validate(self):
            raise OpenRouterCreditError("OpenRouter API credits are exhausted")

    async def scenario():
        service = VoiceBotService(
            Config(state_path=tmp_path / "state.json"),
            FakeBot(),
            FakeGateway(),
            StateStore(tmp_path / "state.json"),
            OutOfCreditPlanner(),
        )
        service.calendar_operations = object()  # type: ignore[assignment]
        await service.initialize()

    with pytest.raises(OpenRouterCreditError):
        asyncio.run(scenario())


def test_webhook_startup_requires_usable_direct_gemini_terminal(tmp_path):
    class FakeBot:
        async def call(self, method):
            assert method == "getMe"
            return {"username": "mk_voice_text_bot"}

        async def configure_profile(self):
            return None

    class FakeGateway:
        async def validate_operations(self):
            return frozenset({"personal"})

        async def read(self, account, operation, arguments):
            return {"id": 100000001}

    class WorkingNemotron:
        async def validate(self):
            return None

    terminal_error = GeminiAuthenticationError("Gemini credential rejected")

    class RejectedGemini:
        async def validate(self):
            raise terminal_error

    async def scenario():
        planner = GeminiProviderChain(
            [
                GeminiProviderStage(
                    "Nemotron 3 Super", WorkingNemotron(), 0.1
                ),
                GeminiProviderStage(
                    "Gemini 3.7 Flash", RejectedGemini(), 0.1
                ),
            ],
            timeout_seconds=0.5,
        )
        service = VoiceBotService(
            Config(
                    state_path=tmp_path / "state.json",
                    bot_update_mode="webhook",
                    webhook_register_with_telegram=False,
                    webhook_path_override="/telegram/test/webhook",
                ),
            FakeBot(),
            FakeGateway(),
            StateStore(tmp_path / "state.json"),
            planner,
        )
        service.calendar_operations = object()  # type: ignore[assignment]
        await service.initialize()

    with pytest.raises(GeminiAuthenticationError) as captured:
        asyncio.run(scenario())
    assert captured.value is terminal_error


def test_calendar_bot_rejects_silent_cli_fallback_after_primary_outage(tmp_path):
    class FakeBot:
        async def call(self, method):
            return {"username": "mk_voice_text_bot"}

        async def configure(self):
            return None

    class FakeGateway:
        async def validate_operations(self):
            return frozenset({"personal"})

        async def read(self, account, operation, arguments):
            return {"id": 100000001}

    primary_error = GeminiApiError("primary startup outage")

    class FailedPrimary:
        async def validate(self):
            raise primary_error

    class WorkingFallback:
        @staticmethod
        def is_available():
            return True

        async def validate(self):
            return None

    async def scenario():
        planner = GeminiFallback(
            FailedPrimary(),  # type: ignore[arg-type]
            WorkingFallback(),  # type: ignore[arg-type]
            timeout_seconds=1,
        )
        service = VoiceBotService(
            Config(state_path=tmp_path / "state.json"),
            FakeBot(),
            FakeGateway(),
            StateStore(tmp_path / "state.json"),
            planner,
        )
        service.calendar_operations = object()  # type: ignore[assignment]
        await service.initialize()

    with pytest.raises(GeminiApiError) as raised:
        asyncio.run(scenario())
    assert raised.value is primary_error


def test_legacy_status_names_active_fallback_without_claiming_primary(tmp_path):
    class FakeBot:
        def __init__(self):
            self.messages = []

        async def call(self, method):
            return {"username": "mk_voice_text_bot"}

        async def configure(self):
            return None

        async def send_text(self, chat_id, text, *, reply_to_message_id=None):
            self.messages.append(text)

    class FakeGateway:
        async def validate_operations(self):
            return frozenset({"personal"})

        async def read(self, account, operation, arguments):
            return {"id": 100000001}

    class FailedPrimary:
        async def validate(self):
            raise GeminiApiError("primary startup outage")

    class WorkingFallback:
        @staticmethod
        def is_available():
            return True

        async def validate(self):
            return None

    async def scenario():
        bot = FakeBot()
        planner = GeminiFallback(
            FailedPrimary(),  # type: ignore[arg-type]
            WorkingFallback(),  # type: ignore[arg-type]
            timeout_seconds=1,
        )
        service = VoiceBotService(
            Config(state_path=tmp_path / "state.json"),
            bot,
            FakeGateway(),
            StateStore(tmp_path / "state.json"),
            planner,
        )
        await service.initialize()
        await service.handle_update(
            {
                "update_id": 902,
                "message": {
                    "message_id": 46,
                    "date": 1787400000,
                    "from": {"id": 100000001},
                    "chat": {"id": 100000001, "type": "private"},
                    "text": "/status",
                },
            }
        )
        return bot

    bot = asyncio.run(scenario())
    assert "Основной ИИ-провайдер недоступен" in bot.messages[-1]
    assert "активен резервный Gemini" in bot.messages[-1]
    assert "ИИ-планировщик доступен" not in bot.messages[-1]


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
    assert "ИИ-планировщик доступен" in bot.messages[0][1]
    assert "Пришлите голосовое" not in bot.messages[0][1]


def test_text_uses_legacy_gemini_path_without_telegram_gateway(tmp_path):
    class FakeBot:
        def __init__(self):
            self.messages = []

        async def send_chat_action(self, chat_id):
            return None

        async def send_text(self, chat_id, text, *, reply_to_message_id=None):
            self.messages.append((chat_id, text, reply_to_message_id))

    class FakeGateway:
        async def read(self, *_args, **_kwargs):
            raise AssertionError("text reached Telegram gateway")

        async def write(self, *_args, **_kwargs):
            raise AssertionError("text reached Telegram gateway")

    class FakeGemini:
        def __init__(self):
            self.transcripts = []

        async def extract_event(self, transcript, **kwargs):
            self.transcripts.append(transcript)
            return {
                "action": "ignore",
                "events": [],
                "clarification_question": None,
                "confidence": 1,
            }

    async def scenario():
        config = Config(state_path=tmp_path / "state.json")
        bot = FakeBot()
        gemini = FakeGemini()
        service = VoiceBotService(
            config,
            bot,
            FakeGateway(),
            StateStore(config.state_path),
            gemini,
        )
        service.enabled_accounts = frozenset()
        await service.handle_update(
            {
                "update_id": 902,
                "message": {
                    "message_id": 46,
                    "date": 1787400000,
                    "from": {"id": 100000001},
                    "chat": {"id": 100000001, "type": "private"},
                    "text": "Что у меня завтра?",
                },
            }
        )
        return bot, gemini

    bot, gemini = asyncio.run(scenario())

    assert gemini.transcripts == ["Что у меня завтра?"]
    assert len(bot.messages) == 1
    assert "Команда:" in bot.messages[0][1]
    assert bot.messages[0][2] == 46


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

import asyncio
from dataclasses import replace
from datetime import datetime, timezone

import pytest

from tg_voice_transcriber_bot.bot_api import BotApiError
from tg_voice_transcriber_bot.calendar import (
    CalendarConnectionError,
    CalendarEventQueryResult,
    CalendarEventSnapshot,
    CalendarStateConflictError,
    CalendarWriteRejectedError,
    CreatedCalendarEvent,
    DeletedCalendarEvent,
    UpdatedCalendarEvent,
)
from tg_voice_transcriber_bot.config import Config
from tg_voice_transcriber_bot.gemini import PLANNER_MODEL_FIELD
from tg_voice_transcriber_bot.openrouter import (
    OpenRouterApiError,
    OpenRouterAuthenticationError,
    OpenRouterCreditError,
    OpenRouterRateLimitError,
    OpenRouterRequestRejectedError,
)
from tg_voice_transcriber_bot.operations import (
    CalendarOperationError,
    CalendarOperationPipeline,
    OperationStore,
)
from tg_voice_transcriber_bot.service import (
    VoiceBotService,
    _compact_lookup_candidates,
    _job_image_observations,
    _job_memory_text,
    _resolve_plan_event_references,
    build_vision_pipeline,
)
from tg_voice_transcriber_bot.state import StateStore
from tg_voice_transcriber_bot.vision import VisionError, VisionResult


OWNER = 100000001
SENT_AT = 1787400000
_EDIT_UNSET = object()


def test_runtime_vision_pipeline_wires_cloud_fallbacks_then_local_ocr():
    config = Config()
    pipeline = build_vision_pipeline(
        config,
        openrouter_api_key="router-secret",
        gemini_api_key="gemini-secret",
    )
    try:
        assert [stage.provider.model for stage in pipeline.stages] == [
            "google/gemma-4-31b-it:free",
            "google/gemma-4-26b-a4b-it:free",
            "gemini-3.7-flash",
        ]
        assert [stage.timeout_seconds for stage in pipeline.stages] == [
            15,
            12,
            20,
        ]
        assert pipeline.local_ocr.model == "rapidocr/pp-ocrv5-cyrillic"
        assert pipeline.local_timeout_seconds == 15
    finally:
        asyncio.run(pipeline.aclose())


def test_image_followup_memory_reserves_space_for_visible_text():
    memory = _job_memory_text(
        {
            "input_kind": "text_and_image",
            "source_input_kind": "text_and_image",
            "transcript": "Подпись " * 300,
            "image_observations": [
                {
                    "description": "Описание " * 600,
                    "visible_text": "29 августа 8:00–10:00, метро Киевская",
                    "source": "Vision",
                    "mode": "vision",
                }
            ],
        }
    )

    assert len(memory) <= 1_000
    assert "29 августа 8:00–10:00" in memory
    assert "метро Киевская" in memory


def test_planner_image_evidence_has_a_strict_utf8_byte_budget():
    observations = _job_image_observations(
        {
            "image_observations": [
                {
                    "description": "описание" * 1_000,
                    "visible_text": "текст со скриншота" * 2_000,
                    "source": "Vision",
                    "mode": "vision",
                }
            ]
        }
    )

    observation = observations[0]
    assert len(observation["description"].encode("utf-8")) <= 1_536
    assert len(observation["visible_text"].encode("utf-8")) <= 6_656
    assert observation["description"].endswith("[… сокращено …]")
    assert observation["visible_text"].endswith("[… сокращено …]")


def calendar_event(**changes):
    event = {
        "title": "Планёрка",
        "start_at": "2026-08-24T17:30:00+03:00",
        "end_at": "2026-08-24T18:00:00+03:00",
        "all_day": False,
        "timezone": "Europe/Moscow",
        "location": None,
        "description": "Повторная встреча",
        "recurrence_rrule": None,
    }
    event.update(changes)
    return event


def create_plan(*, metadata=True):
    plan = {
        "action": "execute",
        "operations": [
            {
                "type": "create",
                "target_event_id": None,
                "recurrence_scope": None,
                "event": calendar_event(),
                "patch": None,
                "clear_fields": [],
            }
        ],
        "lookup": None,
        "clarification_question": None,
        "confidence": 0.96,
    }
    if metadata:
        plan["_interaction_input"] = {
            "type": "user_input",
            "content": [{"type": "text", "text": "first-input"}],
        }
        plan["_interaction_steps"] = [
            {"type": "model_output", "content": []}
        ]
    return plan


def multi_create_plan():
    plan = create_plan()
    plan["operations"].append(
        {
            "type": "create",
            "target_event_id": None,
            "recurrence_scope": None,
            "event": calendar_event(
                title="Стоматолог",
                start_at="2026-08-25T12:00:00+03:00",
                end_at="2026-08-25T12:30:00+03:00",
                description=None,
            ),
            "patch": None,
            "clear_fields": [],
        }
    )
    return plan


def create_then_update_plan(event_id):
    plan = create_plan()
    plan["operations"].append(
        {
            "type": "update",
            "target_event_id": event_id,
            "recurrence_scope": None,
            "event": None,
            "patch": {"location": "переговорная А"},
            "clear_fields": [],
        }
    )
    return plan


def clarify_plan():
    return {
        "action": "clarify",
        "operations": [],
        "lookup": None,
        "clarification_question": "На какое время записать встречу?",
        "confidence": 0.61,
        "_interaction_input": {
            "type": "user_input",
            "content": [{"type": "text", "text": "clarify-input"}],
        },
        "_interaction_steps": [{"type": "model_output", "content": []}],
    }


def discovery_plan(action="lookup", *, query="планёрка", metadata=True):
    plan = {
        "action": action,
        "operations": [],
        "lookup": {
            "query": query,
            "time_min": "2026-08-22T00:00:00+03:00",
            "time_max": "2026-08-31T00:00:00+03:00",
        },
        "clarification_question": None,
        "confidence": 0.95,
    }
    if metadata:
        plan["_interaction_input"] = {
            "type": "user_input",
            "content": [{"type": "text", "text": "lookup-input"}],
        }
        plan["_interaction_steps"] = [
            {
                "type": "model_output",
                "content": [{"type": "text", "text": "lookup-output"}],
            }
        ]
    return plan


def update_plan(
    event_id, *, location="переговорная А", recurrence_scope=None
):
    return {
        "action": "execute",
        "operations": [
            {
                "type": "update",
                "target_event_id": event_id,
                "recurrence_scope": recurrence_scope,
                "event": None,
                "patch": {"location": location},
                "clear_fields": [],
            }
        ],
        "lookup": None,
        "clarification_question": None,
        "confidence": 0.98,
        "_interaction_input": {
            "type": "user_input",
            "content": [{"type": "text", "text": "resolved-input"}],
        },
        "_interaction_steps": [
            {
                "type": "model_output",
                "content": [{"type": "text", "text": "resolved-output"}],
            }
        ],
    }


def delete_plan(event_id, *, recurrence_scope=None):
    return {
        "action": "execute",
        "operations": [
            {
                "type": "delete",
                "target_event_id": event_id,
                "recurrence_scope": recurrence_scope,
                "event": None,
                "patch": None,
                "clear_fields": [],
            }
        ],
        "lookup": None,
        "clarification_question": None,
        "confidence": 0.98,
        "_interaction_input": {
            "type": "user_input",
            "content": [{"type": "text", "text": "delete-input"}],
        },
        "_interaction_steps": [{"type": "model_output", "content": []}],
    }


class FakeBot:
    def __init__(
        self,
        *,
        fail_intermediate=False,
        fail_final=False,
        downloaded_image=b"image-bytes",
    ):
        self.fail_intermediate = fail_intermediate
        self.fail_final = fail_final
        self.sent_html = []
        self.edited_html = []
        self.chat_actions = []
        self.callback_answers = []
        self.keyboard_removals = []
        self.get_file_calls = []
        self.download_file_calls = []
        self.downloaded_image = downloaded_image
        self._next_message_id = 700

    async def send_chat_action(self, chat_id):
        self.chat_actions.append(chat_id)

    async def send_html(
        self,
        chat_id,
        html_text,
        *,
        reply_to_message_id=None,
        reply_markup=None,
    ):
        message_id = self._next_message_id
        self._next_message_id += 1
        self.sent_html.append(
            {
                "chat_id": chat_id,
                "message_id": message_id,
                "html": html_text,
                "reply_to_message_id": reply_to_message_id,
                "reply_markup": reply_markup,
            }
        )
        return message_id

    async def edit_html(
        self,
        chat_id,
        message_id,
        html_text,
        *,
        reply_markup=_EDIT_UNSET,
    ):
        self.edited_html.append(
            {
                "chat_id": chat_id,
                "message_id": message_id,
                "html": html_text,
                "reply_markup": reply_markup,
            }
        )
        if self.fail_intermediate and reply_markup is _EDIT_UNSET:
            raise BotApiError("intermediate edit failed")
        if self.fail_final and reply_markup is not _EDIT_UNSET:
            raise BotApiError("final edit failed")

    async def answer_callback_query(self, callback_query_id, text, *, show_alert=False):
        self.callback_answers.append((callback_query_id, text, show_alert))

    async def remove_inline_keyboard(self, chat_id, message_id):
        self.keyboard_removals.append((chat_id, message_id))

    async def get_file(self, file_id, *, max_file_size):
        self.get_file_calls.append((file_id, max_file_size))
        return {"file_path": f"photos/{file_id}.jpg"}

    async def download_file(self, file_path, *, max_bytes):
        self.download_file_calls.append((file_path, max_bytes))
        return self.downloaded_image


class FakeGateway:
    def __init__(self, transcripts=None):
        self.transcripts = list(transcripts or ["Запиши планёрку"])
        self.read_calls = []
        self.write_calls = []
        self._next_message_id = 100

    async def read(self, account, operation, arguments):
        self.read_calls.append((account, operation, dict(arguments)))
        self._next_message_id += 1
        return {"ok": True, "match": {"message_id": self._next_message_id}}

    async def write(self, account, operation, arguments, **kwargs):
        self.write_calls.append((account, operation, dict(arguments), dict(kwargs)))
        return {
            "ok": True,
            "status": "completed",
            "text": self.transcripts.pop(0),
        }


class FakeGemini:
    def __init__(self, plans=None, *, dynamic=None):
        self.plans = list(plans or [])
        self.dynamic = dynamic
        self.calls = []

    async def plan_calendar_actions(self, transcript, **kwargs):
        self.calls.append((transcript, kwargs))
        if self.dynamic is not None:
            return self.dynamic(len(self.calls), transcript, kwargs)
        return self.plans.pop(0)


class FakeVision:
    def __init__(self, result=None, *, error=None):
        self.result = result or VisionResult(
            description="Экран бронирования корта",
            visible_text="Сб 29 августа, 8:00–10:00, Lunda Padel",
            provider="Gemini",
            model="gemini-3.7-flash",
            used_local_ocr=False,
        )
        self.error = error
        self.calls = []

    async def analyze(self, image):
        self.calls.append(image)
        if self.error is not None:
            raise self.error
        return self.result


class FakeCalendar:
    def __init__(self, *, on_create=None, query_result=None, query_error=None):
        self.events = {}
        self.calls = []
        self.created_by_key = {}
        self.on_create = on_create
        self.query_result = query_result or CalendarEventQueryResult((), 0)
        self.query_error = query_error

    @staticmethod
    def snapshot(event_id, event, *, status="confirmed"):
        return CalendarEventSnapshot(
            account="personal",
            calendar_id="primary",
            event_id=event_id,
            title=event["title"],
            description=event.get("description"),
            location=event.get("location"),
            start_at=event["start_at"],
            end_at=event["end_at"],
            all_day=event["all_day"],
            timezone=event["timezone"],
            status=status,
            html_link=f"https://calendar.google.com/event?eid={event_id}",
            recurrence_rrules=(
                (event["recurrence_rrule"],)
                if event.get("recurrence_rrule")
                else ()
            ),
            creator_is_self=True,
            organizer_is_self=True,
            safety_metadata_complete=True,
            safety_metadata_fingerprint="basic-event-v1",
        )

    async def create_events(self, *, account, events, idempotency_key):
        self.calls.append(("create", account, idempotency_key))
        if idempotency_key in self.created_by_key:
            return self.created_by_key[idempotency_key]
        references = []
        for index, event in enumerate(events, start=1):
            event_id = f"event-{len(self.events) + index:04d}"
            snapshot = self.snapshot(event_id, event)
            self.events[event_id] = snapshot
            references.append(CreatedCalendarEvent(event_id, snapshot.html_link))
        result = tuple(references)
        self.created_by_key[idempotency_key] = result
        if self.on_create is not None:
            self.on_create()
        return result

    async def get_event(self, *, account, event_id):
        self.calls.append(("get", account, event_id))
        return self.events[event_id]

    async def list_events(self, *, account, time_min, time_max, limit=50):
        self.calls.append(("list", account, time_min, time_max, limit))
        if self.query_error is not None:
            raise self.query_error
        return self.query_result

    async def search_events(
        self, *, account, query, time_min, time_max, limit=50
    ):
        self.calls.append(
            ("search", account, query, time_min, time_max, limit)
        )
        if self.query_error is not None:
            raise self.query_error
        return self.query_result

    async def update_event(
        self,
        *,
        account,
        event_id,
        patch,
        idempotency_key,
        expected_current=None,
    ):
        self.calls.append(("update", account, event_id, dict(patch), idempotency_key))
        before = self.events[event_id]
        if expected_current is not None and replace(
            before, updated_at=None, html_link=None
        ) != replace(expected_current, updated_at=None, html_link=None):
            raise CalendarStateConflictError
        changes = {}
        for field, value in patch.items():
            if field in {"location", "description"} and value == "":
                value = None
            if field == "recurrence_rrules":
                value = tuple(value)
            changes[field] = value
        if "start_at" in changes:
            changes["all_day"] = len(str(changes["start_at"])) == 10
        after = replace(before, **changes)
        self.events[event_id] = after
        return UpdatedCalendarEvent(
            previous=before,
            current=after,
            already_applied=after == before,
        )

    async def delete_event(
        self,
        *,
        account,
        event_id,
        idempotency_key,
        expected_current=None,
    ):
        self.calls.append(("delete", account, event_id, idempotency_key))
        previous = self.events[event_id]
        if expected_current is not None and replace(
            previous, updated_at=None, html_link=None
        ) != replace(expected_current, updated_at=None, html_link=None):
            raise CalendarStateConflictError
        current = replace(previous, status="cancelled")
        self.events[event_id] = current
        return DeletedCalendarEvent(
            previous=previous,
            current=current,
            already_deleted=previous.status == "cancelled",
            verified_cancelled=True,
        )


def make_service(tmp_path, *, bot, gateway, gemini, calendar, vision=None):
    config = Config(
        state_path=tmp_path / "state.json",
        operation_state_path=tmp_path / "calendar-operations.json",
        confirmation_state_path=tmp_path / "calendar-confirmations.json",
    )
    state = StateStore(config.state_path)
    pipeline = CalendarOperationPipeline(
        OperationStore(config.operation_state_path), calendar
    )
    service = VoiceBotService(
        config,
        bot,
        gateway,
        state,
        gemini,
        calendar_operations=pipeline,
        vision=vision,
    )
    return service, state, pipeline


async def process_voice(service, *, update_id=77, bot_message_id=456, sent_at=SENT_AT):
    await service._process_voice_v2(
        update_id=update_id,
        account="personal",
        chat_id=OWNER,
        bot_message_id=bot_message_id,
        sent_at=sent_at,
        duration=12,
        file_size=321,
    )


async def process_text(
    service,
    text,
    *,
    update_id=78,
    bot_message_id=457,
    sent_at=SENT_AT,
    owner=OWNER,
):
    await service.handle_update(
        {
            "update_id": update_id,
            "message": {
                "message_id": bot_message_id,
                "date": sent_at,
                "from": {"id": owner},
                "chat": {"id": owner, "type": "private"},
                "text": text,
            },
        }
    )


async def expose_candidate(
    pipeline,
    event,
    *,
    source_update_id=9000,
):
    """Persist a row the owner saw so the next plan receives a short alias."""

    await pipeline.record_read(
        source_update_id=source_update_id,
        account="personal",
        owner_user_id=OWNER,
        chat_id=OWNER,
        transcript="Покажи событие",
        reference_time=datetime.fromtimestamp(SENT_AT, tz=timezone.utc),
        lookup={
            "query": None,
            "time_min": "2026-08-22T00:00:00+03:00",
            "time_max": "2026-08-31T00:00:00+03:00",
        },
        events=[event],
        total_count=1,
        may_be_incomplete=False,
        displayed_candidates=[event],
    )


def test_text_enters_shared_calendar_pipeline_without_telegram_gateway(tmp_path):
    command = (
        "Добавь встречу завтра в 17:30\n"
        "Ссылка: https://meet.example.test/room"
    )

    async def scenario():
        bot = FakeBot()
        gateway = FakeGateway()
        calendar = FakeCalendar()
        gemini = FakeGemini([create_plan()])
        service, state, _pipeline = make_service(
            tmp_path,
            bot=bot,
            gateway=gateway,
            gemini=gemini,
            calendar=calendar,
        )
        # Text processing must not depend on an active MTProto session.
        service.enabled_accounts = frozenset()
        await process_text(service, command)
        await process_text(service, command)
        return bot, gateway, calendar, gemini, state

    bot, gateway, calendar, gemini, state = asyncio.run(scenario())

    assert gateway.read_calls == []
    assert gateway.write_calls == []
    assert [call[0] for call in calendar.calls].count("create") == 1
    assert [transcript for transcript, _kwargs in gemini.calls] == [command]
    assert len(bot.sent_html) == 1
    assert "ИИ-планировщик разбирает команду" in bot.sent_html[0]["html"]
    assert "OpenRouter" not in bot.sent_html[0]["html"]
    assert "Текстовая команда получена" in bot.sent_html[0]["html"]
    assert "💬 <b>Команда</b>" in bot.sent_html[0]["html"]
    assert command in bot.sent_html[0]["html"]
    assert "Ищу сообщение в Telegram" not in bot.sent_html[0]["html"]
    assert bot.sent_html[0]["reply_to_message_id"] == 457
    assert "Добавлено в календарь" in bot.edited_html[-1]["html"]
    assert "💬 Команда" in bot.edited_html[-1]["html"]
    assert command in bot.edited_html[-1]["html"]
    assert state.job(78)["status"] == "sent"
    assert state.job(78)["input_kind"] == "text"
    assert state.after_message_id("personal") == 0


def test_photo_uses_largest_size_and_passes_captioned_observation_to_planner(
    tmp_path,
):
    caption = "Добавь эту бронь в календарь"

    async def scenario():
        bot = FakeBot(downloaded_image=b"telegram-photo")
        gateway = FakeGateway()
        calendar = FakeCalendar()
        gemini = FakeGemini([create_plan()])
        vision = FakeVision()
        service, state, pipeline = make_service(
            tmp_path,
            bot=bot,
            gateway=gateway,
            gemini=gemini,
            calendar=calendar,
            vision=vision,
        )
        await service.handle_update(
            {
                "update_id": 801,
                "message": {
                    "message_id": 901,
                    "date": SENT_AT,
                    "from": {"id": OWNER},
                    "chat": {"id": OWNER, "type": "private"},
                    "caption": caption,
                    "photo": [
                        {
                            "file_id": "small-photo",
                            "width": 320,
                            "height": 200,
                            "file_size": 50_000,
                        },
                        {
                            "file_id": "large-photo",
                            "width": 1280,
                            "height": 800,
                            "file_size": 400_000,
                        },
                    ],
                },
            }
        )
        return bot, gateway, calendar, gemini, vision, state, pipeline

    bot, gateway, calendar, gemini, vision, state, pipeline = asyncio.run(
        scenario()
    )

    assert gateway.read_calls == []
    assert gateway.write_calls == []
    assert bot.get_file_calls == [("large-photo", 8 * 1024 * 1024)]
    assert bot.download_file_calls == [
        ("photos/large-photo.jpg", 8 * 1024 * 1024)
    ]
    assert len(vision.calls) == 1
    assert vision.calls[0].data == b"telegram-photo"
    assert vision.calls[0].mime_type == "image/jpeg"
    transcript, kwargs = gemini.calls[0]
    assert transcript == caption
    assert kwargs["input_kind"] == "text_and_image"
    assert kwargs["image_observations"] == (
        {
            "description": "Экран бронирования корта",
            "visible_text": "Сб 29 августа, 8:00–10:00, Lunda Padel",
            "source": "Gemini",
            "mode": "vision",
        },
    )
    assert [call[0] for call in calendar.calls].count("create") == 1
    assert "Загружаю изображение" in bot.sent_html[0]["html"]
    assert any("Извлекаю текст" in edit["html"] for edit in bot.edited_html)
    planner_progress = next(
        edit["html"]
        for edit in bot.edited_html
        if "ИИ-планировщик разбирает команду" in edit["html"]
    )
    assert "💬 <b>Подпись</b>" in planner_progress
    assert caption in planner_progress
    assert "🖼️ <b>Описание изображения</b>" in planner_progress
    assert "Экран бронирования корта" in planner_progress
    assert "🔤 <b>Текст на изображении</b>" in planner_progress
    assert "Lunda Padel" in planner_progress
    assert state.job(801)["input_kind"] == "text_and_image"
    assert state.job(801)["vision_model"] == "gemini-3.7-flash"
    memory = pipeline.store.find_by_source("telegram-update:801")["transcript"]
    assert caption in memory
    assert "Данные изображения" in memory
    assert "Lunda Padel" in memory
    assert len(memory) <= 1_000


def test_image_document_without_caption_reaches_planner_as_image_only(tmp_path):
    async def scenario():
        bot = FakeBot(downloaded_image=b"png-image")
        calendar = FakeCalendar()
        gemini = FakeGemini([create_plan()])
        vision = FakeVision()
        service, state, _pipeline = make_service(
            tmp_path,
            bot=bot,
            gateway=FakeGateway(),
            gemini=gemini,
            calendar=calendar,
            vision=vision,
        )
        await service.handle_update(
            {
                "update_id": 802,
                "message": {
                    "message_id": 902,
                    "date": SENT_AT,
                    "from": {"id": OWNER},
                    "chat": {"id": OWNER, "type": "private"},
                    "document": {
                        "file_id": "original-image",
                        "mime_type": "image/png",
                        "file_size": 900_000,
                    },
                },
            }
        )
        return bot, gemini, vision, state

    bot, gemini, vision, state = asyncio.run(scenario())

    assert vision.calls[0].data == b"png-image"
    assert vision.calls[0].mime_type == "image/png"
    transcript, kwargs = gemini.calls[0]
    assert transcript == ""
    assert kwargs["input_kind"] == "image"
    assert kwargs["image_observations"][0]["visible_text"].startswith("Сб 29")
    planner_progress = next(
        edit["html"]
        for edit in bot.edited_html
        if "ИИ-планировщик разбирает команду" in edit["html"]
    )
    assert "Описание изображения" in planner_progress
    assert "Текст на изображении" in planner_progress
    assert "Lunda Padel" in planner_progress
    assert "Добавлено в календарь" in bot.edited_html[-1]["html"]
    assert state.job(802)["status"] == "sent"


def test_image_evidence_remains_visible_when_planner_is_unavailable(tmp_path):
    async def scenario():
        bot = FakeBot(downloaded_image=b"png-image")
        calendar = FakeCalendar()
        gemini = FakeGemini([])
        service, state, _pipeline = make_service(
            tmp_path,
            bot=bot,
            gateway=FakeGateway(),
            gemini=gemini,
            calendar=calendar,
            vision=FakeVision(),
        )
        service.gemini_available = False
        await service.handle_update(
            {
                "update_id": 805,
                "message": {
                    "message_id": 905,
                    "date": SENT_AT,
                    "from": {"id": OWNER},
                    "chat": {"id": OWNER, "type": "private"},
                    "document": {
                        "file_id": "original-image",
                        "mime_type": "image/png",
                        "file_size": 900_000,
                    },
                },
            }
        )
        return bot, calendar, gemini, state

    bot, calendar, gemini, state = asyncio.run(scenario())

    final_html = bot.edited_html[-1]["html"]
    assert gemini.calls == []
    assert calendar.calls == []
    assert "ИИ-планировщик сейчас недоступен" in final_html
    assert "Данные изображения" in final_html
    assert "Экран бронирования корта" in final_html
    assert "Lunda Padel" in final_html
    assert state.job(805)["status"] == "sent"


def test_oversized_image_is_rejected_once_without_downloading_or_retrying(tmp_path):
    async def scenario():
        bot = FakeBot()
        calendar = FakeCalendar()
        gemini = FakeGemini([])
        vision = FakeVision()
        service, state, _pipeline = make_service(
            tmp_path,
            bot=bot,
            gateway=FakeGateway(),
            gemini=gemini,
            calendar=calendar,
            vision=vision,
        )
        await service.handle_update(
            {
                "update_id": 806,
                "message": {
                    "message_id": 906,
                    "date": SENT_AT,
                    "from": {"id": OWNER},
                    "chat": {"id": OWNER, "type": "private"},
                    "photo": [
                        {
                            "file_id": "oversized-image",
                            "width": 4096,
                            "height": 4096,
                            "file_size": service.config.vision_max_image_bytes + 1,
                        }
                    ],
                },
            }
        )
        return bot, calendar, gemini, vision, state

    bot, calendar, gemini, vision, state = asyncio.run(scenario())

    assert bot.get_file_calls == []
    assert bot.download_file_calls == []
    assert vision.calls == []
    assert gemini.calls == []
    assert calendar.calls == []
    assert "до 8 МБ" in bot.edited_html[-1]["html"]
    assert state.job(806)["status"] == "sent"


def test_oversized_image_with_caption_continues_as_text_only(tmp_path):
    caption = "Добавь встречу завтра в 17:30"

    async def scenario():
        bot = FakeBot()
        calendar = FakeCalendar()
        gemini = FakeGemini([create_plan()])
        vision = FakeVision()
        service, state, _pipeline = make_service(
            tmp_path,
            bot=bot,
            gateway=FakeGateway(),
            gemini=gemini,
            calendar=calendar,
            vision=vision,
        )
        await service.handle_update(
            {
                "update_id": 807,
                "message": {
                    "message_id": 907,
                    "date": SENT_AT,
                    "from": {"id": OWNER},
                    "chat": {"id": OWNER, "type": "private"},
                    "caption": caption,
                    "photo": [
                        {
                            "file_id": "oversized-captioned-image",
                            "width": 4096,
                            "height": 4096,
                            "file_size": service.config.vision_max_image_bytes + 1,
                        }
                    ],
                },
            }
        )
        return bot, calendar, gemini, vision, state

    bot, calendar, gemini, vision, state = asyncio.run(scenario())

    assert bot.get_file_calls == []
    assert bot.download_file_calls == []
    assert vision.calls == []
    transcript, kwargs = gemini.calls[0]
    assert transcript == caption
    assert kwargs["input_kind"] == "text"
    assert kwargs["image_observations"] == ()
    assert [call[0] for call in calendar.calls].count("create") == 1
    assert "Добавлено в календарь" in bot.edited_html[-1]["html"]
    assert state.job(807)["status"] == "sent"
    assert state.job(807)["source_input_kind"] == "text_and_image"


def test_image_observations_are_preserved_for_lookup_matching_pass(tmp_path):
    async def scenario():
        candidate = FakeCalendar.snapshot(
            "booking-event",
            calendar_event(title="Lunda Padel", location=None),
        )
        calendar = FakeCalendar(
            query_result=CalendarEventQueryResult((candidate,), 1)
        )
        calendar.events[candidate.event_id] = candidate
        gemini = FakeGemini(
            [
                discovery_plan(query="Lunda Padel"),
                update_plan("c1", location="ул. Большая Филёвская, 32"),
            ]
        )
        service, _state, _pipeline = make_service(
            tmp_path,
            bot=FakeBot(),
            gateway=FakeGateway(),
            gemini=gemini,
            calendar=calendar,
            vision=FakeVision(),
        )
        await service.handle_update(
            {
                "update_id": 805,
                "message": {
                    "message_id": 905,
                    "date": SENT_AT,
                    "from": {"id": OWNER},
                    "chat": {"id": OWNER, "type": "private"},
                    "caption": "Добавь адрес к этой брони",
                    "photo": [
                        {"file_id": "lookup-image", "width": 1280, "height": 800}
                    ],
                },
            }
        )
        return calendar, gemini

    calendar, gemini = asyncio.run(scenario())

    assert len(gemini.calls) == 2
    first = gemini.calls[0][1]
    second = gemini.calls[1][1]
    assert first["input_kind"] == second["input_kind"] == "text_and_image"
    assert first["image_observations"] == second["image_observations"]
    assert first["image_observations"][0]["source"] == "Gemini"
    update_calls = [call for call in calendar.calls if call[0] == "update"]
    assert update_calls[0][3]["location"] == "ул. Большая Филёвская, 32"


def test_captioned_image_never_bypasses_planner_via_text_fast_read(tmp_path):
    async def scenario():
        gemini = FakeGemini([create_plan()])
        service, _state, _pipeline = make_service(
            tmp_path,
            bot=FakeBot(),
            gateway=FakeGateway(),
            gemini=gemini,
            calendar=FakeCalendar(),
            vision=FakeVision(),
        )
        await service.handle_update(
            {
                "update_id": 808,
                "message": {
                    "message_id": 908,
                    "date": SENT_AT,
                    "from": {"id": OWNER},
                    "chat": {"id": OWNER, "type": "private"},
                    "caption": "Какие у меня события в ближайший час?",
                    "photo": [
                        {"file_id": "read-image", "width": 1280, "height": 800}
                    ],
                },
            }
        )
        return gemini

    gemini = asyncio.run(scenario())

    assert len(gemini.calls) == 1
    assert gemini.calls[0][1]["input_kind"] == "text_and_image"
    assert gemini.calls[0][1]["image_observations"]


def test_caption_falls_back_to_text_when_all_image_recognition_fails(tmp_path):
    caption = "Добавь встречу завтра в 17:30"

    async def scenario():
        calendar = FakeCalendar()
        gemini = FakeGemini([create_plan()])
        bot = FakeBot()
        service, state, _pipeline = make_service(
            tmp_path,
            bot=bot,
            gateway=FakeGateway(),
            gemini=gemini,
            calendar=calendar,
            vision=FakeVision(error=VisionError("all vision providers failed")),
        )
        await service.handle_update(
            {
                "update_id": 803,
                "message": {
                    "message_id": 903,
                    "date": SENT_AT,
                    "from": {"id": OWNER},
                    "chat": {"id": OWNER, "type": "private"},
                    "caption": caption,
                    "photo": [{"file_id": "captioned", "width": 800, "height": 600}],
                },
            }
        )
        return bot, calendar, gemini, state

    bot, calendar, gemini, state = asyncio.run(scenario())

    transcript, kwargs = gemini.calls[0]
    assert transcript == caption
    assert kwargs["input_kind"] == "text"
    assert kwargs["image_observations"] == ()
    assert [call[0] for call in calendar.calls].count("create") == 1
    assert "Добавлено в календарь" in bot.edited_html[-1]["html"]
    assert state.job(803)["input_kind"] == "text"
    assert state.job(803)["source_input_kind"] == "text_and_image"


def test_image_only_returns_bounded_error_when_all_recognition_fails(tmp_path):
    async def scenario():
        bot = FakeBot()
        calendar = FakeCalendar()
        gemini = FakeGemini([])
        service, state, _pipeline = make_service(
            tmp_path,
            bot=bot,
            gateway=FakeGateway(),
            gemini=gemini,
            calendar=calendar,
            vision=FakeVision(error=VisionError("all vision providers failed")),
        )
        await service.handle_update(
            {
                "update_id": 804,
                "message": {
                    "message_id": 904,
                    "date": SENT_AT,
                    "from": {"id": OWNER},
                    "chat": {"id": OWNER, "type": "private"},
                    "photo": [{"file_id": "unreadable", "width": 800, "height": 600}],
                },
            }
        )
        return bot, calendar, gemini, state

    bot, calendar, gemini, state = asyncio.run(scenario())

    assert gemini.calls == []
    assert calendar.calls == []
    assert "Не удалось прочитать содержимое" in bot.edited_html[-1]["html"]
    assert "Google Calendar не изменён" in bot.edited_html[-1]["html"]
    assert state.job(804)["status"] == "sent"


def test_nearest_hour_read_skips_gemini_even_when_provider_is_unavailable(tmp_path):
    async def scenario():
        candidate = FakeCalendar.snapshot(
            "nearest-event",
            calendar_event(
                title="Ближайшая встреча",
                start_at="2026-08-22T15:20:00+03:00",
                end_at="2026-08-22T15:40:00+03:00",
            ),
        )
        calendar = FakeCalendar(
            query_result=CalendarEventQueryResult((candidate,), 1)
        )
        gemini = FakeGemini([])
        bot = FakeBot()
        service, state, pipeline = make_service(
            tmp_path,
            bot=bot,
            gateway=FakeGateway(),
            gemini=gemini,
            calendar=calendar,
        )
        service.gemini_available = False
        await process_text(
            service,
            "Какие у меня события в ближайший час?",
            update_id=79,
            bot_message_id=479,
        )
        return bot, calendar, gemini, state, pipeline

    bot, calendar, gemini, state, pipeline = asyncio.run(scenario())

    assert gemini.calls == []
    assert "ИИ-планировщик" not in bot.sent_html[0]["html"]
    assert not any(
        "ИИ-планировщик" in edit["html"] for edit in bot.edited_html
    )
    assert calendar.calls == [
        (
            "list",
            "personal",
            "2026-08-22T15:00:00+03:00",
            "2026-08-22T16:00:00+03:00",
            20,
        )
    ]
    assert "События в календаре" in bot.edited_html[-1]["html"]
    assert "Ближайшая встреча" in bot.edited_html[-1]["html"]
    assert "🤖 <code>Без LLM · быстрый разбор</code>" in (
        bot.edited_html[-1]["html"]
    )
    assert state.job(79)["status"] == "sent"
    assert pipeline.store.find_by_source("telegram-update:79")["stage"] == "read"


def test_lookup_model_candidates_are_bounded_and_timezone_normalized():
    compact, refs, series_refs = _compact_lookup_candidates(
        [
            {
                "event_id": "provider-id-secret",
                "title": "T" * 700,
                "start_at": "2026-08-24T10:30:00+02:00",
                "end_at": "2026-08-24T11:00:00+02:00",
                "all_day": False,
                "timezone": "Europe/Amsterdam",
                "location": "L" * 1_200,
                "description": "D" * 2_500,
                "recurrence_rrules": [
                    "RRULE:FREQ=WEEKLY;BYDAY=" + "MO," * 500
                ],
                "status": "confirmed",
                "html_link": "https://calendar.example/secret",
                "creator_email": "private@example.test",
            }
        ],
        timezone_name="Europe/Moscow",
    )

    assert refs == {"c1": "provider-id-secret"}
    assert series_refs == {"c1": "provider-id-secret"}
    assert compact[0]["event_id"] == "c1"
    assert compact[0]["start_at"] == "2026-08-24T11:30:00+03:00"
    assert compact[0]["end_at"] == "2026-08-24T12:00:00+03:00"
    assert compact[0]["timezone"] == "Europe/Moscow"
    assert len(compact[0]["title"]) == 300
    assert len(compact[0]["location"]) == 300
    assert len(compact[0]["description"]) == 500
    assert len(compact[0]["recurrence_rrule"]) == 500
    assert compact[0]["title"].endswith("…")
    assert "html_link" not in compact[0]
    assert "creator_email" not in compact[0]


def test_recurrence_mutation_resolves_alias_to_trusted_series_master():
    compact, visible_refs, series_refs = _compact_lookup_candidates(
        [
            {
                "event_id": "provider-instance-secret",
                "recurring_event_id": "provider-master-secret",
                "title": "Дейлик",
                "start_at": "2026-08-24T10:50:00+03:00",
                "end_at": "2026-08-24T11:30:00+03:00",
                "all_day": False,
                "timezone": "Europe/Moscow",
                "location": None,
                "description": None,
                "recurrence_rrules": [],
                "status": "confirmed",
            }
        ],
        timezone_name="Europe/Moscow",
    )
    recurrence_plan = update_plan("c1")
    recurrence_plan["operations"][0]["patch"] = {
        "recurrence_rrule": "RRULE:FREQ=WEEKLY;BYDAY=MO,WE,FR"
    }
    recurrence_plan["operations"][0]["recurrence_scope"] = "series"
    location_plan = update_plan("c1")
    location_plan["operations"][0]["recurrence_scope"] = "occurrence"

    resolved_series = _resolve_plan_event_references(
        recurrence_plan, visible_refs, series_refs
    )
    resolved_occurrence = _resolve_plan_event_references(
        location_plan, visible_refs, series_refs
    )

    assert compact[0]["event_id"] == "c1"
    assert compact[0]["recurring"] is True
    assert compact[0]["recurring_instance"] is True
    assert "provider-instance-secret" not in str(compact)
    assert "provider-master-secret" not in str(compact)
    assert resolved_series["operations"][0]["target_event_id"] == (
        "provider-master-secret"
    )
    assert resolved_occurrence["operations"][0]["target_event_id"] == (
        "provider-instance-secret"
    )


def test_recurring_scope_controls_series_delete_and_ordinary_update():
    visible = {"c1": "provider-instance-secret"}
    series = {"c1": "provider-master-secret"}

    delete_series = _resolve_plan_event_references(
        delete_plan("c1", recurrence_scope="series"),
        visible,
        series,
        ("c1",),
    )
    update_series = _resolve_plan_event_references(
        update_plan("c1", recurrence_scope="series"),
        visible,
        series,
        ("c1",),
    )
    update_occurrence = _resolve_plan_event_references(
        update_plan("c1", recurrence_scope="occurrence"),
        visible,
        series,
        ("c1",),
    )
    ambiguous = _resolve_plan_event_references(
        delete_plan("c1"), visible, series, ("c1",)
    )

    assert delete_series["operations"][0]["target_event_id"] == (
        "provider-master-secret"
    )
    assert update_series["operations"][0]["target_event_id"] == (
        "provider-master-secret"
    )
    assert update_occurrence["operations"][0]["target_event_id"] == (
        "provider-instance-secret"
    )
    assert ambiguous["action"] == "execute"
    assert ambiguous["operations"][0]["target_event_id"] == (
        "provider-instance-secret"
    )

    master_occurrence = _resolve_plan_event_references(
        update_plan("e1", recurrence_scope="occurrence"),
        {"e1": "provider-master-secret"},
        {"e1": "provider-master-secret"},
        ("e1",),
    )
    assert master_occurrence["action"] == "execute"
    assert master_occurrence["operations"][0]["target_event_id"] == (
        "provider-master-secret"
    )


def test_unauthorized_text_is_silently_ignored(tmp_path):
    async def scenario():
        bot = FakeBot()
        gateway = FakeGateway()
        calendar = FakeCalendar()
        gemini = FakeGemini([create_plan()])
        service, _state, _pipeline = make_service(
            tmp_path,
            bot=bot,
            gateway=gateway,
            gemini=gemini,
            calendar=calendar,
        )
        await process_text(service, "Добавь встречу", owner=OWNER + 999)
        return bot, gateway, calendar, gemini

    bot, gateway, calendar, gemini = asyncio.run(scenario())

    assert bot.sent_html == []
    assert bot.edited_html == []
    assert bot.chat_actions == []
    assert gateway.read_calls == []
    assert gateway.write_calls == []
    assert calendar.calls == []
    assert gemini.calls == []


def test_v2_sends_one_status_then_edits_each_phase_and_applies_create_immediately(
    tmp_path, monkeypatch
):
    async def scenario():
        clock = {"now": 100.0}
        monkeypatch.setattr(
            "tg_voice_transcriber_bot.service.time.monotonic",
            lambda: clock["now"],
        )
        monkeypatch.setattr(
            "tg_voice_transcriber_bot.service.time.time",
            lambda: clock["now"],
        )
        bot = FakeBot()
        calendar = FakeCalendar(on_create=lambda: clock.update(now=104.8))
        plan = create_plan()
        plan[PLANNER_MODEL_FIELD] = "nvidia/nemotron-3-super-120b-a12b"
        service, state, pipeline = make_service(
            tmp_path,
            bot=bot,
            gateway=FakeGateway(),
            gemini=FakeGemini([plan]),
            calendar=calendar,
        )

        await process_voice(service)
        return bot, calendar, state, pipeline

    bot, calendar, state, pipeline = asyncio.run(scenario())

    assert len(bot.sent_html) == 1
    assert "Ищу сообщение в Telegram" in bot.sent_html[0]["html"]
    assert [
        expected in edit["html"]
        for expected, edit in zip(
            (
                "Получаю расшифровку от Telegram",
                "ИИ-планировщик разбирает команду и контекст",
                "Добавляю событие в Google Calendar",
                "Добавлено в календарь",
            ),
            bot.edited_html,
            strict=True,
        )
    ] == [True, True, True, True]
    assert "🎙️ <b>Расшифровка Telegram</b>" in bot.edited_html[1]["html"]
    assert "Запиши планёрку" in bot.edited_html[1]["html"]
    assert "Запиши планёрку" in bot.edited_html[2]["html"]
    assert {edit["message_id"] for edit in bot.edited_html} == {700}
    assert [call[0] for call in calendar.calls].count("create") == 1
    assert "4,8 с" in bot.edited_html[-1]["html"]
    assert (
        "🤖 <code>nvidia/nemotron-3-super-120b-a12b</code>"
        in bot.edited_html[-1]["html"]
    )
    markup = bot.edited_html[-1]["reply_markup"]
    assert markup["inline_keyboard"][0][0]["text"] == "↩️ Отменить добавление"
    assert state.job(77)["status"] == "sent"
    assert pipeline.store.get(state.job(77)["operation_id"])["stage"] == "applied"


def test_v2_passes_compact_known_event_without_cross_turn_native_steps(tmp_path):
    def dynamic_plan(call_number, transcript, kwargs):
        if call_number == 1:
            return create_plan()
        event_id = kwargs["application_state"]["candidate_events"][0]["event_id"]
        return {
            "action": "execute",
            "operations": [
                {
                    "type": "update",
                    "target_event_id": event_id,
                    "recurrence_scope": None,
                    "event": None,
                    "patch": {"location": "переговорная А"},
                    "clear_fields": [],
                }
            ],
            "lookup": None,
            "clarification_question": None,
            "confidence": 0.97,
            "_interaction_input": {
                "type": "user_input",
                "content": [{"type": "text", "text": "second-input"}],
            },
            "_interaction_steps": [{"type": "model_output", "content": []}],
        }

    async def scenario():
        bot = FakeBot()
        calendar = FakeCalendar()
        gemini = FakeGemini(dynamic=dynamic_plan)
        service, _state, _pipeline = make_service(
            tmp_path,
            bot=bot,
            gateway=FakeGateway(["Запиши планёрку"]),
            gemini=gemini,
            calendar=calendar,
        )
        await process_voice(service, update_id=80, bot_message_id=500)
        await process_text(
            service,
            "Добавь ему место переговорная А",
            update_id=81,
            bot_message_id=501,
        )
        return gemini, calendar, service.gateway

    gemini, calendar, gateway = asyncio.run(scenario())
    transcript, context = gemini.calls[1]

    assert transcript == "Добавь ему место переговорная А"
    assert len(context["application_state"]["candidate_events"]) == 1
    assert context["application_state"]["allowed_event_ids"] == ["e1"]
    assert context["application_state"]["candidate_events"][0]["event_id"] == "e1"
    assert context["recent_conversation"][0]["user_message"] == (
        "Запиши планёрку"
    )
    assert context["recent_conversation"][0]["actions"][0]["event_id"]
    assert context["history_steps"] == ()
    assert [call[0] for call in calendar.calls].count("create") == 1
    assert [call[0] for call in calendar.calls].count("update") == 1
    assert len(gateway.read_calls) == 1
    assert len(gateway.write_calls) == 1
    active = next(event for event in calendar.events.values() if event.status == "confirmed")
    assert active.location == "переговорная А"


def test_two_recent_creates_are_directly_editable_by_alias_without_calendar_read(
    tmp_path,
):
    first = create_plan()
    first["operations"][0]["event"] = calendar_event(title="Первый дейлик")
    second = create_plan()
    second["operations"][0]["event"] = calendar_event(
        title="Второй дейлик",
        start_at="2026-08-24T18:30:00+03:00",
        end_at="2026-08-24T19:00:00+03:00",
    )

    def dynamic_plan(call_number, _transcript, kwargs):
        if call_number == 1:
            return first
        if call_number == 2:
            return second
        candidates = kwargs["application_state"]["candidate_events"]
        refs_by_title = {
            candidate["title"]: candidate["event_id"] for candidate in candidates
        }
        return {
            "action": "execute",
            "operations": [
                {
                    "type": "update",
                    "target_event_id": refs_by_title["Первый дейлик"],
                    "recurrence_scope": None,
                    "event": None,
                    "patch": {"location": "Ссылка 1"},
                    "clear_fields": [],
                },
                {
                    "type": "update",
                    "target_event_id": refs_by_title["Второй дейлик"],
                    "recurrence_scope": None,
                    "event": None,
                    "patch": {"location": "Ссылка 2"},
                    "clear_fields": [],
                },
            ],
            "lookup": None,
            "clarification_question": None,
            "confidence": 0.99,
        }

    async def scenario():
        calendar = FakeCalendar()
        gemini = FakeGemini(dynamic=dynamic_plan)
        service, _state, _pipeline = make_service(
            tmp_path,
            bot=FakeBot(),
            gateway=FakeGateway(),
            gemini=gemini,
            calendar=calendar,
        )
        await process_text(
            service, "Создай первый дейлик", update_id=82, bot_message_id=502
        )
        await process_text(
            service, "Создай второй дейлик", update_id=83, bot_message_id=503
        )
        calendar.calls.clear()
        await process_text(
            service,
            "Добавь обоим этим дейликам ссылки",
            update_id=84,
            bot_message_id=504,
        )
        return calendar, gemini

    calendar, gemini = asyncio.run(scenario())

    assert len(gemini.calls[2][1]["application_state"]["candidate_events"]) == 2
    # Direct aliases skip discovery; provider-fresh baselines still protect
    # Undo from edits made outside the bot.
    assert [call[0] for call in calendar.calls] == [
        "get",
        "get",
        "update",
        "update",
    ]
    by_title = {event.title: event for event in calendar.events.values()}
    assert by_title["Первый дейлик"].location == "Ссылка 1"
    assert by_title["Второй дейлик"].location == "Ссылка 2"


def test_two_recurring_dailies_survive_ignored_greeting_and_receive_distinct_links(
    tmp_path,
):
    first_url = "https://meet.example/daily-a?token=one"
    second_url = "https://meet.example/daily-b?token=two"
    rrule = "RRULE:FREQ=WEEKLY;BYDAY=MO,TU,WE,TH,FR"
    first = create_plan(metadata=False)
    first["operations"][0]["event"] = calendar_event(
        title="Дейлик A",
        start_at="2026-08-24T10:50:00+03:00",
        end_at="2026-08-24T11:30:00+03:00",
        description=None,
        recurrence_rrule=rrule,
    )
    second = create_plan(metadata=False)
    second["operations"][0]["event"] = calendar_event(
        title="Дейлик B",
        start_at="2026-08-24T11:30:00+03:00",
        end_at="2026-08-24T12:00:00+03:00",
        description=None,
        recurrence_rrule=rrule,
    )
    ignored = {
        "action": "ignore",
        "operations": [],
        "lookup": None,
        "clarification_question": None,
        "confidence": 1.0,
    }

    def dynamic_plan(call_number, _transcript, kwargs):
        if call_number == 1:
            return first
        if call_number == 2:
            return second
        if call_number == 3:
            return ignored
        candidates = kwargs["application_state"]["candidate_events"]
        refs_by_title = {
            candidate["title"]: candidate["event_id"]
            for candidate in candidates
        }
        return {
            "action": "execute",
            "operations": [
                {
                    "type": "update",
                    "target_event_id": refs_by_title["Дейлик A"],
                    "recurrence_scope": "series",
                    "event": None,
                    "patch": {"description": first_url},
                    "clear_fields": [],
                },
                {
                    "type": "update",
                    "target_event_id": refs_by_title["Дейлик B"],
                    "recurrence_scope": "series",
                    "event": None,
                    "patch": {"description": second_url},
                    "clear_fields": [],
                },
            ],
            "lookup": None,
            "clarification_question": None,
            "confidence": 1.0,
        }

    async def scenario():
        bot = FakeBot()
        calendar = FakeCalendar()
        gemini = FakeGemini(dynamic=dynamic_plan)
        service, _state, _pipeline = make_service(
            tmp_path,
            bot=bot,
            gateway=FakeGateway(),
            gemini=gemini,
            calendar=calendar,
        )
        await process_text(
            service,
            "Создай дейлик A по будням с 10:50 до 11:30",
            update_id=840,
            bot_message_id=840,
        )
        await process_text(
            service,
            "Создай дейлик B по будням с 11:30 до 12:00",
            update_id=841,
            bot_message_id=841,
        )
        await process_text(
            service,
            "Привет",
            update_id=842,
            bot_message_id=842,
        )
        before_by_title = {
            event.title: event for event in calendar.events.values()
        }
        calendar.calls.clear()
        await process_text(
            service,
            (
                "Теперь к обоим этим дейликам добавь ссылки соответственно:\n\n"
                f"10:50 (Дейлик A) {first_url}\n\n"
                f"11:30 (Дейлик B) {second_url}"
            ),
            update_id=843,
            bot_message_id=843,
        )
        return bot, calendar, gemini, before_by_title

    bot, calendar, gemini, before_by_title = asyncio.run(scenario())

    context = gemini.calls[3][1]
    candidates = context["application_state"]["candidate_events"]
    assert [(candidate["event_id"], candidate["title"]) for candidate in candidates] == [
        ("e1", "Дейлик B"),
        ("e2", "Дейлик A"),
    ]
    assert all(candidate["recurring"] is True for candidate in candidates)
    assert all(candidate["recurrence_rrule"] == rrule for candidate in candidates)
    assert [turn["user_message"] for turn in context["recent_conversation"]] == [
        "Создай дейлик A по будням с 10:50 до 11:30",
        "Создай дейлик B по будням с 11:30 до 12:00",
    ]
    assert [call[0] for call in calendar.calls] == [
        "get",
        "get",
        "update",
        "update",
    ]
    update_calls = [call for call in calendar.calls if call[0] == "update"]
    assert [call[3] for call in update_calls] == [
        {"description": first_url},
        {"description": second_url},
    ]

    after_by_title = {event.title: event for event in calendar.events.values()}
    assert after_by_title["Дейлик A"].description == first_url
    assert after_by_title["Дейлик B"].description == second_url
    for title in ("Дейлик A", "Дейлик B"):
        before = before_by_title[title]
        after = after_by_title[title]
        assert (after.start_at, after.end_at, after.recurrence_rrules) == (
            before.start_at,
            before.end_at,
            before.recurrence_rrules,
        )
    assert "✏️ <b>Событие обновлено</b>" in bot.edited_html[-1]["html"]
    assert "Не удалось" not in bot.edited_html[-1]["html"]


def test_unknown_model_event_alias_is_rejected_before_calendar_access(tmp_path):
    async def scenario():
        calendar = FakeCalendar()
        bot = FakeBot()
        service, state, pipeline = make_service(
            tmp_path,
            bot=bot,
            gateway=FakeGateway(),
            gemini=FakeGemini([update_plan("e999")]),
            calendar=calendar,
        )
        await process_text(
            service,
            "Измени неизвестное событие",
            update_id=85,
            bot_message_id=505,
        )
        return bot, calendar, state, pipeline

    bot, calendar, state, pipeline = asyncio.run(scenario())

    assert calendar.calls == []
    assert "вне доступного контекста" in bot.edited_html[-1]["html"]
    assert "Google Calendar не изменён" in bot.edited_html[-1]["html"]
    assert pipeline.store.find_by_source("telegram-update:85") is None
    assert state.job(85)["status"] == "sent"


def test_openrouter_timeout_has_specific_copy_and_safe_diagnostic_log(
    tmp_path, caplog
):
    class TimedOutPlanner(FakeGemini):
        async def plan_calendar_actions(self, transcript, **kwargs):
            self.calls.append((transcript, kwargs))
            raise OpenRouterApiError("OpenRouter API transport error: ReadTimeout")

    async def scenario():
        calendar = FakeCalendar()
        bot = FakeBot()
        service, state, pipeline = make_service(
            tmp_path,
            bot=bot,
            gateway=FakeGateway(),
            gemini=TimedOutPlanner(),
            calendar=calendar,
        )
        await process_text(
            service,
            "Перенеси встречу на завтра",
            update_id=86,
            bot_message_id=506,
        )
        return bot, calendar, state, pipeline

    with caplog.at_level("WARNING", logger="tg_voice_transcriber_bot"):
        bot, calendar, state, pipeline = asyncio.run(scenario())

    final_html = bot.edited_html[-1]["html"]
    assert "не успел обработать команду за отведённое время" in final_html
    assert "не смог надёжно разобрать" not in final_html
    assert calendar.calls == []
    assert pipeline.store.find_by_source("telegram-update:86") is None
    assert state.job(86)["status"] == "sent"
    assert "error_type=OpenRouterApiError" in caplog.text
    assert "error=OpenRouter API transport error: ReadTimeout" in caplog.text
    assert "elapsed=" in caplog.text


def test_openrouter_rate_limit_has_honest_copy_and_does_not_mutate_calendar(
    tmp_path, caplog
):
    class RateLimitedPlanner(FakeGemini):
        async def plan_calendar_actions(self, transcript, **kwargs):
            self.calls.append((transcript, kwargs))
            raise OpenRouterRateLimitError("OpenRouter API rate limit exceeded")

    async def scenario():
        calendar = FakeCalendar()
        bot = FakeBot()
        service, state, pipeline = make_service(
            tmp_path,
            bot=bot,
            gateway=FakeGateway(),
            gemini=RateLimitedPlanner(),
            calendar=calendar,
        )
        await process_text(
            service,
            "Добавь ссылки к обоим дейликам",
            update_id=186,
            bot_message_id=606,
        )
        return bot, calendar, state, pipeline

    with caplog.at_level("WARNING", logger="tg_voice_transcriber_bot"):
        bot, calendar, state, pipeline = asyncio.run(scenario())

    final_html = bot.edited_html[-1]["html"]
    assert "Провайдеры ИИ-планировщика временно ограничили запросы" in final_html
    assert "не смог надёжно разобрать" not in final_html
    assert calendar.calls == []
    assert pipeline.store.find_by_source("telegram-update:186") is None
    assert state.job(186)["status"] == "sent"
    assert "error_type=OpenRouterRateLimitError" in caplog.text
    assert "OpenRouter API rate limit exceeded" in caplog.text


def test_openrouter_credit_error_requests_top_up_without_mutating_calendar(
    tmp_path, caplog
):
    class OutOfCreditPlanner(FakeGemini):
        async def plan_calendar_actions(self, transcript, **kwargs):
            self.calls.append((transcript, kwargs))
            raise OpenRouterCreditError("OpenRouter credits exhausted")

    async def scenario():
        calendar = FakeCalendar()
        bot = FakeBot()
        service, state, pipeline = make_service(
            tmp_path,
            bot=bot,
            gateway=FakeGateway(),
            gemini=OutOfCreditPlanner(),
            calendar=calendar,
        )
        await process_text(
            service,
            "Добавь встречу завтра",
            update_id=187,
            bot_message_id=607,
        )
        return bot, calendar, state, pipeline

    with caplog.at_level("WARNING", logger="tg_voice_transcriber_bot"):
        bot, calendar, state, pipeline = asyncio.run(scenario())

    final_html = bot.edited_html[-1]["html"]
    assert "OpenRouter отклонил запрос из-за лимита ключа или баланса" in final_html
    assert "Проверьте аккаунт" in final_html
    assert calendar.calls == []
    assert pipeline.store.find_by_source("telegram-update:187") is None
    assert state.job(187)["status"] == "sent"
    assert "error_type=OpenRouterCreditError" in caplog.text
    assert "OpenRouter credits exhausted" in caplog.text


def test_openrouter_auth_error_requests_key_check_without_mutating_calendar(
    tmp_path, caplog
):
    class RejectedPlanner(FakeGemini):
        async def plan_calendar_actions(self, transcript, **kwargs):
            self.calls.append((transcript, kwargs))
            raise OpenRouterAuthenticationError("OpenRouter access rejected")

    async def scenario():
        calendar = FakeCalendar()
        bot = FakeBot()
        service, state, pipeline = make_service(
            tmp_path,
            bot=bot,
            gateway=FakeGateway(),
            gemini=RejectedPlanner(),
            calendar=calendar,
        )
        await process_text(
            service,
            "Добавь встречу завтра",
            update_id=188,
            bot_message_id=608,
        )
        return bot, calendar, state, pipeline

    with caplog.at_level("WARNING", logger="tg_voice_transcriber_bot"):
        bot, calendar, state, pipeline = asyncio.run(scenario())

    final_html = bot.edited_html[-1]["html"]
    assert "OpenRouter отклонил API-ключ" in final_html
    assert "резервные модели тоже не ответили" in final_html
    assert "Проверьте ключ" in final_html
    assert calendar.calls == []
    assert pipeline.store.find_by_source("telegram-update:188") is None
    assert state.job(188)["status"] == "sent"
    assert "error_type=OpenRouterAuthenticationError" in caplog.text


def test_openrouter_request_rejection_is_not_misreported_as_bad_key(
    tmp_path, caplog
):
    class RejectedRequestPlanner(FakeGemini):
        async def plan_calendar_actions(self, transcript, **kwargs):
            self.calls.append((transcript, kwargs))
            raise OpenRouterRequestRejectedError("OpenRouter request rejected")

    async def scenario():
        calendar = FakeCalendar()
        bot = FakeBot()
        service, state, pipeline = make_service(
            tmp_path,
            bot=bot,
            gateway=FakeGateway(),
            gemini=RejectedRequestPlanner(),
            calendar=calendar,
        )
        await process_text(
            service,
            "Добавь встречу завтра",
            update_id=189,
            bot_message_id=609,
        )
        return bot, calendar, state, pipeline

    with caplog.at_level("WARNING", logger="tg_voice_transcriber_bot"):
        bot, calendar, state, pipeline = asyncio.run(scenario())

    final_html = bot.edited_html[-1]["html"]
    assert "OpenRouter отклонил запрос" in final_html
    assert "резервные модели тоже не ответили" in final_html
    assert "API-ключ" not in final_html
    assert calendar.calls == []
    assert pipeline.store.find_by_source("telegram-update:189") is None
    assert state.job(189)["status"] == "sent"


def test_intermediate_status_edit_failure_does_not_block_calendar_mutation(tmp_path):
    async def scenario():
        bot = FakeBot(fail_intermediate=True)
        calendar = FakeCalendar()
        service, state, _pipeline = make_service(
            tmp_path,
            bot=bot,
            gateway=FakeGateway(),
            gemini=FakeGemini([create_plan()]),
            calendar=calendar,
        )
        await process_voice(service)
        return bot, calendar, state

    bot, calendar, state = asyncio.run(scenario())

    assert len(bot.sent_html) == 1
    assert len(bot.edited_html) == 4
    assert [call[0] for call in calendar.calls].count("create") == 1
    assert "Добавлено в календарь" in bot.edited_html[-1]["html"]
    assert state.job(77)["status"] == "sent"


def test_final_edit_failure_sends_fallback_without_reapplying_calendar_plan(tmp_path):
    async def scenario():
        bot = FakeBot(fail_final=True)
        calendar = FakeCalendar()
        gemini = FakeGemini([create_plan()])
        service, state, _pipeline = make_service(
            tmp_path,
            bot=bot,
            gateway=FakeGateway(),
            gemini=gemini,
            calendar=calendar,
        )
        await process_voice(service)
        await process_voice(service)
        return bot, calendar, gemini, state

    bot, calendar, gemini, state = asyncio.run(scenario())

    assert len(bot.sent_html) == 2
    assert "Ищу сообщение в Telegram" in bot.sent_html[0]["html"]
    assert "Добавлено в календарь" in bot.sent_html[1]["html"]
    assert bot.sent_html[1]["reply_markup"] is not None
    assert [call[0] for call in calendar.calls].count("create") == 1
    assert len(gemini.calls) == 1
    assert state.job(77)["status"] == "sent"


def test_clarification_finishes_without_calendar_write_or_undo_button(tmp_path):
    async def scenario():
        bot = FakeBot()
        calendar = FakeCalendar()
        service, state, pipeline = make_service(
            tmp_path,
            bot=bot,
            gateway=FakeGateway(["Запиши встречу"]),
            gemini=FakeGemini([clarify_plan()]),
            calendar=calendar,
        )
        await process_voice(service)
        return bot, calendar, state, pipeline

    bot, calendar, state, pipeline = asyncio.run(scenario())

    assert calendar.calls == []
    assert len(bot.sent_html) == 1
    assert "Нужно уточнение" in bot.edited_html[-1]["html"]
    assert bot.edited_html[-1]["reply_markup"] is None
    record = pipeline.store.find_by_source("telegram-update:77")
    assert record["stage"] == "clarify"


def test_undo_callback_reverses_create_and_posts_a_separate_history_message(tmp_path):
    async def scenario():
        bot = FakeBot()
        calendar = FakeCalendar()
        service, state, pipeline = make_service(
            tmp_path,
            bot=bot,
            gateway=FakeGateway(),
            gemini=FakeGemini([create_plan()]),
            calendar=calendar,
        )
        await process_voice(service, update_id=90, bot_message_id=600)
        operation_id = state.job(90)["operation_id"]
        original_message_id = state.job(90)["final_message_id"]
        await service.handle_update(
            {
                "update_id": 91,
                "callback_query": {
                    "id": "callback-undo",
                    "from": {"id": OWNER},
                    "data": f"cal:undo:{operation_id}",
                    "message": {
                        "message_id": original_message_id,
                        "chat": {"id": OWNER, "type": "private"},
                    },
                },
            }
        )
        return bot, calendar, pipeline, operation_id

    bot, calendar, pipeline, operation_id = asyncio.run(scenario())

    assert bot.callback_answers == [("callback-undo", "Отменяю…", False)]
    assert bot.keyboard_removals == [(OWNER, 700)]
    assert len(bot.sent_html) == 2
    assert "Действие отменено" in bot.sent_html[-1]["html"]
    assert [call[0] for call in calendar.calls].count("delete") == 1
    record = pipeline.store.get(operation_id)
    assert record["stage"] == "undone"
    assert record["undo"]["chat_notified"] is True
    context = pipeline.context(
        account="personal",
        chat_id=OWNER,
        now=datetime.fromtimestamp(SENT_AT, tz=timezone.utc),
    )
    assert context.recent_conversation[-1]["assistant_message"] == "Операция отменена."


def test_read_lists_events_once_survives_replay_and_never_offers_undo(tmp_path):
    async def scenario():
        candidate = FakeCalendar.snapshot(
            "external-read-event",
            calendar_event(
                title="Созвон с командой",
                start_at="2026-08-25T10:00:00+03:00",
                end_at="2026-08-25T10:30:00+03:00",
            ),
        )
        calendar = FakeCalendar(
            query_result=CalendarEventQueryResult((candidate,), 1)
        )
        bot = FakeBot()
        gemini = FakeGemini([discovery_plan("read", query=None)])
        service, state, pipeline = make_service(
            tmp_path,
            bot=bot,
            gateway=FakeGateway(["Что у меня на следующей неделе?"]),
            gemini=gemini,
            calendar=calendar,
        )

        original_record_read = pipeline.record_read
        interrupted = False

        async def interrupt_after_query(**kwargs):
            nonlocal interrupted
            if not interrupted:
                interrupted = True
                raise RuntimeError("simulated process interruption")
            return await original_record_read(**kwargs)

        pipeline.record_read = interrupt_after_query  # type: ignore[method-assign]
        with pytest.raises(RuntimeError, match="simulated process interruption"):
            await process_voice(service, update_id=110, bot_message_id=610)

        durable_job = state.job(110)
        assert durable_job["status"] == "calendar_read_ready"
        assert durable_job["calendar_query_result"]["events"][0][
            "event_id"
        ] == "external-read-event"

        await process_voice(service, update_id=110, bot_message_id=610)
        # A fully completed duplicate delivery is also a no-op.
        await process_voice(service, update_id=110, bot_message_id=610)
        return bot, calendar, gemini, state, pipeline

    bot, calendar, gemini, state, pipeline = asyncio.run(scenario())

    assert len(gemini.calls) == 1
    assert [call[0] for call in calendar.calls] == ["list"]
    assert calendar.calls[0][2:] == (
        "2026-08-22T00:00:00+03:00",
        "2026-08-31T00:00:00+03:00",
        20,
    )
    assert "Ищу события в Google Calendar" in "\n".join(
        edit["html"] for edit in bot.edited_html
    )
    assert "События в календаре" in bot.edited_html[-1]["html"]
    assert "Созвон с командой" in bot.edited_html[-1]["html"]
    assert bot.edited_html[-1]["reply_markup"] is None
    assert "Отмен" not in bot.edited_html[-1]["html"]
    record = pipeline.store.find_by_source("telegram-update:110")
    assert record["stage"] == "read"
    assert record["undo"] == {"stage": "unavailable"}
    assert record["items"][0]["target_event_id"] == "external-read-event"
    assert state.job(110)["status"] == "sent"


def test_lookup_then_second_gemini_updates_exact_external_event(tmp_path):
    async def scenario():
        candidate = FakeCalendar.snapshot(
            "external-update-event", calendar_event()
        )
        calendar = FakeCalendar(
            query_result=CalendarEventQueryResult((candidate,), 1)
        )
        calendar.events[candidate.event_id] = candidate
        initial_plan = discovery_plan()
        initial_plan[PLANNER_MODEL_FIELD] = "nvidia/nemotron-3-super-120b-a12b"
        resolved_plan = update_plan("c1")
        resolved_plan[PLANNER_MODEL_FIELD] = "gemini-3.7-flash"
        gemini = FakeGemini([initial_plan, resolved_plan])
        bot = FakeBot()
        service, state, pipeline = make_service(
            tmp_path,
            bot=bot,
            gateway=FakeGateway(
                ["Добавь планёрке местоположение переговорная А"]
            ),
            gemini=gemini,
            calendar=calendar,
        )
        await process_voice(service, update_id=111, bot_message_id=611)
        return bot, calendar, gemini, state, pipeline

    bot, calendar, gemini, state, pipeline = asyncio.run(scenario())

    assert len(gemini.calls) == 2
    assert [transcript for transcript, _kwargs in gemini.calls] == [
        "Добавь планёрке местоположение переговорная А",
        "Добавь планёрке местоположение переговорная А",
    ]
    second_context = gemini.calls[1][1]
    persisted_candidates = state.job(111)["calendar_lookup_candidates"]
    model_candidates = second_context["application_state"]["candidate_events"]
    assert model_candidates == [
        {
            "event_id": "c1",
            "display_index": 1,
            "title": "Планёрка",
            "start_at": "2026-08-24T17:30:00+03:00",
            "end_at": "2026-08-24T18:00:00+03:00",
            "all_day": False,
            "timezone": "Europe/Moscow",
            "location": None,
            "description": "Повторная встреча",
            "recurrence_rrule": None,
            "recurring": False,
            "recurring_instance": False,
            "status": "confirmed",
        }
    ]
    assert persisted_candidates[0]["event_id"] == "external-update-event"
    assert "html_link" in persisted_candidates[0]
    assert "html_link" not in model_candidates[0]
    assert "creator_email" not in model_candidates[0]
    assert second_context["application_state"]["allowed_event_ids"] == ["c1"]
    assert second_context["application_state"]["lookup_permitted"] is False
    assert second_context["application_state"]["lookup_request"] == (
        discovery_plan(metadata=False)["lookup"]
    )
    assert [step["type"] for step in second_context["history_steps"]] == [
        "user_input",
        "model_output",
    ]
    assert second_context["history_steps"][0]["content"][0]["text"] == (
        "lookup-input"
    )
    assert [call[0] for call in calendar.calls] == ["search", "get", "update"]
    assert calendar.calls[0][2] == "планёрка"
    assert calendar.events["external-update-event"].location == "переговорная А"
    progress = "\n".join(edit["html"] for edit in bot.edited_html)
    assert "Ищу события в Google Calendar" in progress
    assert "ИИ-планировщик выбирает точную запись" in progress
    assert "Обновляю событие" in progress
    assert "Событие обновлено" in bot.edited_html[-1]["html"]
    assert (
        "🤖 <code>gemini-3.7-flash</code>"
        in bot.edited_html[-1]["html"]
    )
    assert "Nemotron 3 Super" not in bot.edited_html[-1]["html"]
    assert bot.edited_html[-1]["reply_markup"]["inline_keyboard"][0][0][
        "text"
    ] == "↩️ Отменить изменение"
    record = pipeline.store.find_by_source("telegram-update:111")
    assert record["stage"] == "applied"
    assert [
        (event["display_index"], event["event_id"])
        for event in record["displayed_candidates"]
    ] == [(1, "external-update-event")]
    assert record["items"][0]["target_event_id"] == "external-update-event"


def test_lookup_recurrence_update_targets_trusted_series_master(tmp_path):
    async def scenario():
        master = FakeCalendar.snapshot(
            "provider-master-secret",
            calendar_event(
                title="Дейлик",
                recurrence_rrule="RRULE:FREQ=WEEKLY;BYDAY=MO,TU,WE,TH,FR",
            ),
        )
        instance = replace(
            master,
            event_id="provider-instance-secret",
            recurrence_rrules=(),
            recurring_event_id=master.event_id,
            original_start_at=master.start_at,
        )
        calendar = FakeCalendar(
            query_result=CalendarEventQueryResult((instance,), 1)
        )
        calendar.events[master.event_id] = master
        calendar.events[instance.event_id] = instance
        recurrence_plan = update_plan("c1")
        recurrence_plan["operations"][0]["patch"] = {
            "recurrence_rrule": "RRULE:FREQ=WEEKLY;BYDAY=MO,WE,FR"
        }
        recurrence_plan["operations"][0]["recurrence_scope"] = "series"
        gemini = FakeGemini([discovery_plan(), recurrence_plan])
        service, state, pipeline = make_service(
            tmp_path,
            bot=FakeBot(),
            gateway=FakeGateway(["Оставь дейлик только в понедельник, среду и пятницу"]),
            gemini=gemini,
            calendar=calendar,
        )

        await process_voice(service, update_id=115, bot_message_id=615)
        return calendar, gemini, state, pipeline

    calendar, gemini, state, pipeline = asyncio.run(scenario())

    model_candidate = gemini.calls[1][1]["application_state"][
        "candidate_events"
    ][0]
    assert model_candidate["event_id"] == "c1"
    assert model_candidate["recurring"] is True
    assert model_candidate["recurring_instance"] is True
    assert model_candidate["recurrence_rrule"] == (
        "RRULE:FREQ=WEEKLY;BYDAY=MO,TU,WE,TH,FR"
    )
    assert model_candidate["series_context"] == {
        "start_at": "2026-08-24T17:30:00+03:00",
        "end_at": "2026-08-24T18:00:00+03:00",
        "all_day": False,
        "timezone": "Europe/Moscow",
        "recurrence_rrule": "RRULE:FREQ=WEEKLY;BYDAY=MO,TU,WE,TH,FR",
    }
    assert "provider-master-secret" not in str(model_candidate)
    assert "provider-instance-secret" not in str(model_candidate)
    assert [call[0] for call in calendar.calls] == [
        "search",
        "get",  # hydrate the master before Gemini computes a relative RRULE
        "get",  # provider-fresh mutation preflight for safe Undo
        "update",
    ]
    assert calendar.events["provider-master-secret"].recurrence_rrules == (
        "RRULE:FREQ=WEEKLY;BYDAY=MO,WE,FR",
    )
    assert calendar.events["provider-instance-secret"].recurrence_rrules == ()
    record = pipeline.store.find_by_source("telegram-update:115")
    assert record["items"][0]["target_event_id"] == "provider-master-secret"
    assert state.job(115)["status"] == "sent"
    assert [step["type"] for step in record["interaction_steps"]] == [
        "model_output",
        "user_input",
        "model_output",
    ]


def test_lookup_series_delete_targets_master_not_visible_occurrence(tmp_path):
    async def scenario():
        master = FakeCalendar.snapshot(
            "series-delete-master",
            calendar_event(
                title="Дейлик",
                recurrence_rrule="RRULE:FREQ=WEEKLY;BYDAY=MO,TU,WE,TH,FR",
            ),
        )
        instance = replace(
            master,
            event_id="series-delete-occurrence",
            recurrence_rrules=(),
            recurring_event_id=master.event_id,
            original_start_at=master.start_at,
        )
        calendar = FakeCalendar(
            query_result=CalendarEventQueryResult((instance,), 1)
        )
        calendar.events[master.event_id] = master
        calendar.events[instance.event_id] = instance
        gemini = FakeGemini(
            [
                discovery_plan(query="дейлик"),
                delete_plan("c1", recurrence_scope="series"),
            ]
        )
        service, state, pipeline = make_service(
            tmp_path,
            bot=FakeBot(),
            gateway=FakeGateway(["Удали весь дейлик"]),
            gemini=gemini,
            calendar=calendar,
        )

        await process_voice(service, update_id=116, bot_message_id=616)
        return calendar, state, pipeline

    calendar, state, pipeline = asyncio.run(scenario())

    assert calendar.events["series-delete-master"].status == "cancelled"
    assert calendar.events["series-delete-occurrence"].status == "confirmed"
    record = pipeline.store.find_by_source("telegram-update:116")
    assert record["items"][0]["target_event_id"] == "series-delete-master"
    assert record["items"][0]["request"]["recurrence_scope"] == "series"
    assert state.job(116)["status"] == "sent"


def test_lookup_then_second_gemini_deletes_external_event(tmp_path):
    async def scenario():
        candidate = FakeCalendar.snapshot(
            "external-delete-event",
            calendar_event(title="Отменяемая встреча"),
        )
        calendar = FakeCalendar(
            query_result=CalendarEventQueryResult((candidate,), 1)
        )
        calendar.events[candidate.event_id] = candidate
        gemini = FakeGemini(
            [
                discovery_plan(query="отменяемая встреча"),
                delete_plan("c1"),
            ]
        )
        bot = FakeBot()
        service, state, pipeline = make_service(
            tmp_path,
            bot=bot,
            gateway=FakeGateway(["Удали отменяемую встречу"]),
            gemini=gemini,
            calendar=calendar,
        )
        await process_voice(service, update_id=112, bot_message_id=612)
        return bot, calendar, gemini, state, pipeline

    bot, calendar, gemini, state, pipeline = asyncio.run(scenario())

    assert len(gemini.calls) == 2
    assert [call[0] for call in calendar.calls] == ["search", "get", "delete"]
    assert calendar.events["external-delete-event"].status == "cancelled"
    assert "Событие удалено" in bot.edited_html[-1]["html"]
    assert bot.edited_html[-1]["reply_markup"]["inline_keyboard"][0][0][
        "text"
    ] == "↩️ Восстановить событие"
    record = pipeline.store.find_by_source("telegram-update:112")
    assert record["stage"] == "applied"
    assert record["items"][0]["type"] == "delete"
    assert state.job(112)["status"] == "sent"


def test_empty_lookup_is_delegated_to_second_model_without_calendar_write(tmp_path):
    async def scenario():
        calendar = FakeCalendar(
            query_result=CalendarEventQueryResult((), 0, False)
        )
        gemini = FakeGemini([discovery_plan(), clarify_plan()])
        bot = FakeBot()
        service, _state, pipeline = make_service(
            tmp_path,
            bot=bot,
            gateway=FakeGateway(["Перенеси планёрку"]),
            gemini=gemini,
            calendar=calendar,
        )
        await process_voice(service, update_id=113, bot_message_id=613)
        return bot, calendar, gemini, pipeline

    bot, calendar, gemini, pipeline = asyncio.run(scenario())

    assert len(gemini.calls) == 2
    second_state = gemini.calls[1][1]["application_state"]
    assert second_state["candidate_events"] == []
    assert second_state["allowed_event_ids"] == []
    assert second_state["lookup_result"] == {
        "total_count": 0,
        "may_be_incomplete": False,
    }
    assert [call[0] for call in calendar.calls] == ["search"]
    assert "На какое время записать встречу?" in bot.edited_html[-1]["html"]
    assert bot.edited_html[-1]["reply_markup"] is None
    record = pipeline.store.find_by_source("telegram-update:113")
    assert record["stage"] == "clarify"
    assert not {"create", "update", "delete"} & {
        call[0] for call in calendar.calls
    }


def test_incomplete_lookup_is_delegated_to_gemini_and_second_pass_create_applies(
    tmp_path,
):
    async def scenario():
        candidate = FakeCalendar.snapshot(
            "one-of-many", calendar_event(title="Один из вариантов")
        )
        calendar = FakeCalendar(
            query_result=CalendarEventQueryResult((candidate,), 2, True)
        )
        gemini = FakeGemini([discovery_plan(), create_plan()])
        bot = FakeBot()
        service, state, pipeline = make_service(
            tmp_path,
            bot=bot,
            gateway=FakeGateway(["Перенеси планёрку"]),
            gemini=gemini,
            calendar=calendar,
        )
        await process_voice(service, update_id=113, bot_message_id=613)
        return bot, calendar, gemini, state, pipeline

    bot, calendar, gemini, state, pipeline = asyncio.run(scenario())

    assert len(gemini.calls) == 2
    second_state = gemini.calls[1][1]["application_state"]
    assert second_state["lookup_result"] == {
        "total_count": 2,
        "may_be_incomplete": True,
    }
    assert [call[0] for call in calendar.calls] == ["search", "create", "get"]
    assert pipeline.store.find_by_source("telegram-update:113")["stage"] == "applied"
    assert state.job(113)["resolved_plan"]["action"] == "execute"
    assert "Добавлено в календарь" in bot.edited_html[-1]["html"]


def test_second_discovery_request_is_blocked_into_clarification(tmp_path):
    async def scenario():
        candidate = FakeCalendar.snapshot(
            "ambiguous-event", calendar_event(title="Планёрка")
        )
        calendar = FakeCalendar(
            query_result=CalendarEventQueryResult((candidate,), 1)
        )
        gemini = FakeGemini(
            [
                discovery_plan(),
                discovery_plan(query="попробовать поиск ещё раз"),
            ]
        )
        bot = FakeBot()
        service, state, pipeline = make_service(
            tmp_path,
            bot=bot,
            gateway=FakeGateway(["Перенеси планёрку"]),
            gemini=gemini,
            calendar=calendar,
        )
        await process_voice(service, update_id=114, bot_message_id=614)
        return bot, calendar, gemini, state, pipeline

    bot, calendar, gemini, state, pipeline = asyncio.run(scenario())

    assert len(gemini.calls) == 2
    assert [call[0] for call in calendar.calls] == ["search"]
    assert gemini.calls[1][1]["application_state"]["lookup_permitted"] is False
    assert "Не удалось однозначно выбрать событие" in bot.edited_html[-1]["html"]
    assert "Планёрка" in bot.edited_html[-1]["html"]
    assert bot.edited_html[-1]["reply_markup"] is None
    assert pipeline.store.find_by_source("telegram-update:114")["stage"] == (
        "clarify"
    )
    assert state.job(114)["resolved_plan"]["action"] == "clarify"


def test_lookup_clarification_persists_exact_visible_provider_order(tmp_path):
    provider_ids = [
        "event-zeta",
        "event-alpha",
        "event-mike",
        "event-bravo",
        "event-yankee",
        "event-charlie",
        "event-xray",
    ]
    candidates = tuple(
        FakeCalendar.snapshot(
            event_id,
            calendar_event(title=f"Кандидат {index}"),
        )
        for index, event_id in enumerate(provider_ids, start=1)
    )

    async def scenario():
        calendar = FakeCalendar(
            query_result=CalendarEventQueryResult(candidates, len(candidates))
        )
        gemini = FakeGemini([discovery_plan(), clarify_plan()])
        bot = FakeBot()
        service, state, pipeline = make_service(
            tmp_path,
            bot=bot,
            gateway=FakeGateway(["Перенеси нужную планёрку"]),
            gemini=gemini,
            calendar=calendar,
        )
        await process_voice(service, update_id=118, bot_message_id=618)
        reloaded = CalendarOperationPipeline(
            OperationStore(tmp_path / "calendar-operations.json"), calendar
        )
        return bot, gemini, state, pipeline, reloaded

    bot, gemini, state, pipeline, reloaded = asyncio.run(scenario())

    visible_ids = provider_ids[:5]
    second_state = gemini.calls[1][1]["application_state"]
    assert [event["event_id"] for event in second_state["candidate_events"]] == (
        ["c1", "c2", "c3", "c4", "c5"]
    )
    assert [event["display_index"] for event in second_state["candidate_events"]] == (
        [1, 2, 3, 4, 5]
    )
    assert second_state["allowed_event_ids"] == ["c1", "c2", "c3", "c4", "c5"]
    assert [
        event["event_id"] for event in state.job(118)["calendar_lookup_candidates"]
    ] == visible_ids

    final_html = bot.edited_html[-1]["html"]
    for index in range(1, 6):
        assert f"Кандидат {index}" in final_html
    assert "Кандидат 6" not in final_html
    assert "Кандидат 7" not in final_html

    record = pipeline.store.find_by_source("telegram-update:118")
    assert [event["event_id"] for event in record["displayed_candidates"]] == (
        visible_ids
    )
    context = reloaded.context(
        account="personal",
        chat_id=OWNER,
        now=datetime.fromtimestamp(SENT_AT, tz=timezone.utc),
    )
    assert context.allowed_event_ids == tuple(visible_ids)
    assert [
        event["event_id"] for event in context.application_state["candidate_events"]
    ] == ["e1", "e2", "e3", "e4", "e5"]
    assert [
        event["display_index"]
        for event in context.application_state["candidate_events"]
    ] == [1, 2, 3, 4, 5]


def test_read_order_survives_restart_and_followup_targets_second_visible_event(
    tmp_path,
):
    provider_ids = [
        "read-zeta",
        "read-alpha",
        "read-mike",
        "read-bravo",
        "read-yankee",
        "read-charlie",
        "read-xray",
        "read-delta",
        "read-whiskey",
        "read-echo",
    ]
    candidates = tuple(
        FakeCalendar.snapshot(
            event_id,
            calendar_event(title=f"Встреча {index}"),
        )
        for index, event_id in enumerate(provider_ids, start=1)
    )
    captured_state = {}

    def followup_plan(_call_number, _transcript, kwargs):
        captured_state.update(kwargs["application_state"])
        return update_plan("e2", location="второй кабинет")

    async def scenario():
        calendar = FakeCalendar(
            query_result=CalendarEventQueryResult(candidates, len(candidates))
        )
        calendar.events.update({event.event_id: event for event in candidates})
        first_bot = FakeBot()
        first_service, _state, first_pipeline = make_service(
            tmp_path,
            bot=first_bot,
            gateway=FakeGateway(["Покажи встречи"]),
            gemini=FakeGemini([discovery_plan("read", query=None)]),
            calendar=calendar,
        )
        await process_voice(first_service, update_id=119, bot_message_id=619)

        # Rebuild state and operation objects to exercise the durable restart
        # path rather than relying on in-memory candidate ordering.
        second_bot = FakeBot()
        second_service, _state, second_pipeline = make_service(
            tmp_path,
            bot=second_bot,
            gateway=FakeGateway(["Добавь второму место: второй кабинет"]),
            gemini=FakeGemini(dynamic=followup_plan),
            calendar=calendar,
        )
        await process_voice(second_service, update_id=120, bot_message_id=620)
        return first_bot, calendar, first_pipeline, second_pipeline

    first_bot, calendar, first_pipeline, second_pipeline = asyncio.run(scenario())

    visible_ids = provider_ids[:8]
    read_record = first_pipeline.store.find_by_source("telegram-update:119")
    assert [
        event["event_id"] for event in read_record["displayed_candidates"]
    ] == visible_ids
    assert [event["event_id"] for event in captured_state["candidate_events"]] == (
        [f"e{index}" for index in range(1, 9)]
    )
    assert [event["display_index"] for event in captured_state["candidate_events"]] == (
        list(range(1, 9))
    )
    assert captured_state["allowed_event_ids"] == [
        f"e{index}" for index in range(1, 9)
    ]
    assert calendar.events[provider_ids[1]].location == "второй кабинет"
    assert calendar.events[provider_ids[8]].location is None
    assert "Встреча 8" in first_bot.edited_html[-1]["html"]
    assert "Встреча 9" not in first_bot.edited_html[-1]["html"]
    assert second_pipeline.store.find_by_source("telegram-update:120")["items"][0][
        "target_event_id"
    ] == provider_ids[1]


def test_calendar_query_failure_hides_provider_diagnostics_and_does_not_write(
    tmp_path,
):
    async def scenario():
        calendar = FakeCalendar(
            query_error=RuntimeError("secret provider token and stack trace")
        )
        gemini = FakeGemini([discovery_plan("read", query=None)])
        bot = FakeBot()
        service, state, pipeline = make_service(
            tmp_path,
            bot=bot,
            gateway=FakeGateway(["Что у меня завтра?"]),
            gemini=gemini,
            calendar=calendar,
        )
        await process_voice(service, update_id=115, bot_message_id=615)
        return bot, calendar, gemini, state, pipeline

    bot, calendar, gemini, state, pipeline = asyncio.run(scenario())

    assert len(gemini.calls) == 1
    assert [call[0] for call in calendar.calls] == ["list"]
    final_html = bot.edited_html[-1]["html"]
    assert "Не удалось прочитать события из Google Calendar" in final_html
    assert "secret provider token" not in final_html
    assert bot.edited_html[-1]["reply_markup"] is None
    assert pipeline.store.find_by_source("telegram-update:115") is None
    assert state.job(115)["status"] == "sent"


def test_fatal_calendar_query_connection_error_propagates_for_process_restart(
    tmp_path,
):
    async def scenario():
        calendar = FakeCalendar(
            query_error=CalendarConnectionError("sanitized dead connection")
        )
        bot = FakeBot()
        service, state, pipeline = make_service(
            tmp_path,
            bot=bot,
            gateway=FakeGateway(["Что у меня завтра?"]),
            gemini=FakeGemini([discovery_plan("read", query=None)]),
            calendar=calendar,
        )
        with pytest.raises(CalendarConnectionError):
            await process_voice(service, update_id=116, bot_message_id=616)
        return bot, calendar, state, pipeline

    bot, calendar, state, pipeline = asyncio.run(scenario())

    assert [call[0] for call in calendar.calls] == ["list"]
    assert pipeline.store.find_by_source("telegram-update:116") is None
    assert state.job(116)["status"] == "planned"
    assert "calendar_query_result" not in state.job(116)
    assert not any(
        "Не удалось прочитать события" in edit["html"]
        for edit in bot.edited_html
    )


def test_fatal_calendar_write_connection_error_propagates_without_finalizing_job(
    tmp_path,
):
    class DeadWriteCalendar(FakeCalendar):
        async def create_events(self, *, account, events, idempotency_key):
            self.calls.append(("create", account, idempotency_key))
            raise CalendarConnectionError("sanitized dead connection")

    async def scenario():
        calendar = DeadWriteCalendar()
        bot = FakeBot()
        service, state, pipeline = make_service(
            tmp_path,
            bot=bot,
            gateway=FakeGateway(),
            gemini=FakeGemini([create_plan()]),
            calendar=calendar,
        )
        with pytest.raises(CalendarConnectionError):
            await process_voice(service, update_id=117, bot_message_id=617)
        return bot, calendar, state, pipeline

    bot, calendar, state, pipeline = asyncio.run(scenario())

    assert [call[0] for call in calendar.calls] == ["create"]
    record = pipeline.store.find_by_source("telegram-update:117")
    assert record["stage"] == "applying"
    assert record["items"][0]["stage"] == "applying"
    assert record["items"][0]["provider_write_started_at"]
    assert state.job(117)["status"] == "planned"
    assert "operation_id" not in state.job(117)
    assert "calendar_write_retry_count" not in state.job(117)
    assert not any(
        "Не удалось применить операцию" in edit["html"]
        for edit in bot.edited_html
    )


def test_ambiguous_create_replays_after_durable_reload_with_same_idempotency_key(
    tmp_path,
):
    class LostFirstResponseCalendar(FakeCalendar):
        def __init__(self):
            super().__init__()
            self.response_lost = False

        async def create_events(self, *, account, events, idempotency_key):
            result = await super().create_events(
                account=account,
                events=events,
                idempotency_key=idempotency_key,
            )
            if not self.response_lost:
                self.response_lost = True
                raise RuntimeError("provider response lost after write")
            return result

    async def scenario():
        calendar = LostFirstResponseCalendar()
        first_bot = FakeBot()
        first_service, first_state, first_pipeline = make_service(
            tmp_path,
            bot=first_bot,
            gateway=FakeGateway(),
            gemini=FakeGemini([create_plan()]),
            calendar=calendar,
        )
        with pytest.raises(CalendarOperationError) as raised:
            await process_voice(first_service, update_id=121, bot_message_id=621)
        assert raised.value.retryable is True
        assert raised.value.outcome_uncertain is True
        assert first_state.job(121)["status"] == "planned"
        assert first_state.job(121)["calendar_write_retry_count"] == 1
        first_record = first_pipeline.store.find_by_source("telegram-update:121")
        assert first_record["stage"] == "applying"
        assert first_record["items"][0]["stage"] == "applying"
        assert first_record["items"][0]["provider_write_started_at"]

        # Rebuild both durable stores to model a webhook retry after restart.
        second_bot = FakeBot()
        second_service, second_state, second_pipeline = make_service(
            tmp_path,
            bot=second_bot,
            gateway=FakeGateway(),
            gemini=FakeGemini([]),
            calendar=calendar,
        )
        await process_voice(second_service, update_id=121, bot_message_id=621)
        return calendar, second_bot, second_state, second_pipeline

    calendar, bot, state, pipeline = asyncio.run(scenario())

    create_calls = [call for call in calendar.calls if call[0] == "create"]
    assert len(create_calls) == 2
    assert create_calls[0][2] == create_calls[1][2]
    assert len(calendar.events) == 1
    assert state.job(121)["status"] == "sent"
    assert "calendar_write_retry_count" not in state.job(121)
    record = pipeline.store.find_by_source("telegram-update:121")
    assert record["stage"] == "applied"
    assert record["items"][0]["stage"] == "applied"
    assert "Добавлено в календарь" in bot.edited_html[-1]["html"]


def test_ambiguous_multi_item_write_resumes_only_unfinished_item(tmp_path):
    class LostSecondResponseCalendar(FakeCalendar):
        def __init__(self):
            super().__init__()
            self.response_lost = False

        async def create_events(self, *, account, events, idempotency_key):
            result = await super().create_events(
                account=account,
                events=events,
                idempotency_key=idempotency_key,
            )
            if idempotency_key.endswith(":1:create") and not self.response_lost:
                self.response_lost = True
                raise RuntimeError("second response lost after write")
            return result

    async def scenario():
        calendar = LostSecondResponseCalendar()
        bot = FakeBot()
        service, state, pipeline = make_service(
            tmp_path,
            bot=bot,
            gateway=FakeGateway(),
            gemini=FakeGemini([multi_create_plan()]),
            calendar=calendar,
        )
        with pytest.raises(CalendarOperationError):
            await process_voice(service, update_id=122, bot_message_id=622)
        failed_record = pipeline.store.find_by_source("telegram-update:122")
        assert [item["stage"] for item in failed_record["items"]] == [
            "applied",
            "applying",
        ]

        await process_voice(service, update_id=122, bot_message_id=622)
        return calendar, bot, state, pipeline

    calendar, bot, state, pipeline = asyncio.run(scenario())

    create_calls = [call for call in calendar.calls if call[0] == "create"]
    keys = [call[2] for call in create_calls]
    assert keys.count(keys[0]) == 1
    assert keys.count(keys[1]) == 2
    assert keys[1] == keys[2]
    assert len(calendar.events) == 2
    assert state.job(122)["status"] == "sent"
    assert pipeline.store.find_by_source("telegram-update:122")["stage"] == "applied"
    assert "Добавлено в календарь" in bot.edited_html[-1]["html"]


def test_confirmed_partial_batch_prewrite_failure_stays_planned_and_replays(
    tmp_path,
):
    async def scenario():
        calendar = FakeCalendar()
        target = FakeCalendar.snapshot("partial-target", calendar_event())
        calendar.events[target.event_id] = target
        bot = FakeBot()
        service, state, pipeline = make_service(
            tmp_path,
            bot=bot,
            gateway=FakeGateway(),
            gemini=FakeGemini([create_then_update_plan("e1")]),
            calendar=calendar,
        )
        await expose_candidate(pipeline, target, source_update_id=9125)
        original_merge = pipeline._merged_update
        failed_once = False

        def flaky_merge(before, patch, clear_fields):
            nonlocal failed_once
            if not failed_once:
                failed_once = True
                raise RuntimeError("transient prewrite failure after first item")
            return original_merge(before, patch, clear_fields)

        pipeline._merged_update = flaky_merge
        with pytest.raises(CalendarOperationError) as raised:
            await process_voice(service, update_id=125, bot_message_id=625)
        assert raised.value.partially_applied is True
        assert raised.value.retryable is True
        assert raised.value.outcome_uncertain is False
        failed = pipeline.store.find_by_source("telegram-update:125")
        assert [item["stage"] for item in failed["items"]] == [
            "applied",
            "failed",
        ]
        assert state.job(125)["status"] == "planned"
        assert state.job(125)["calendar_write_retry_count"] == 1

        await process_voice(service, update_id=125, bot_message_id=625)
        return calendar, bot, state, pipeline

    calendar, bot, state, pipeline = asyncio.run(scenario())

    assert len([call for call in calendar.calls if call[0] == "create"]) == 1
    assert len([call for call in calendar.calls if call[0] == "update"]) == 1
    assert len(calendar.events) == 2
    assert calendar.events["partial-target"].location == "переговорная А"
    assert state.job(125)["status"] == "sent"
    assert "calendar_write_retry_count" not in state.job(125)
    assert pipeline.store.find_by_source("telegram-update:125")["stage"] == "applied"
    assert "Google Calendar не изменён" not in bot.edited_html[-1]["html"]


def test_ambiguous_write_budget_exhaustion_warns_not_to_resend_blindly(tmp_path):
    class NeverConfirmedCalendar(FakeCalendar):
        async def create_events(self, *, account, events, idempotency_key):
            await super().create_events(
                account=account,
                events=events,
                idempotency_key=idempotency_key,
            )
            raise RuntimeError("provider never returned a conclusive response")

    async def scenario():
        calendar = NeverConfirmedCalendar()
        bot = FakeBot()
        service, state, pipeline = make_service(
            tmp_path,
            bot=bot,
            gateway=FakeGateway(),
            gemini=FakeGemini([create_plan()]),
            calendar=calendar,
        )
        for attempt in range(1, 6):
            if attempt < 5:
                with pytest.raises(CalendarOperationError):
                    await process_voice(service, update_id=123, bot_message_id=623)
                assert state.job(123)["status"] == "planned"
                assert state.job(123)["calendar_write_retry_count"] == attempt
            else:
                await process_voice(service, update_id=123, bot_message_id=623)
        return calendar, bot, state, pipeline

    calendar, bot, state, pipeline = asyncio.run(scenario())

    assert len([call for call in calendar.calls if call[0] == "create"]) == 5
    assert len(calendar.events) == 1
    assert state.job(123)["status"] == "sent"
    assert state.job(123)["calendar_write_retry_count"] == 5
    final_html = bot.edited_html[-1]["html"]
    assert "могло примениться полностью или частично" in final_html
    assert "Не отправляйте ту же команду повторно" in final_html
    assert "Проверьте итог в Google Calendar" in final_html
    assert "Google Calendar не изменён" not in final_html
    record = pipeline.store.find_by_source("telegram-update:123")
    assert record["stage"] == "applying"
    assert record["items"][0]["stage"] == "applying"


def test_definitive_create_rejection_does_not_consume_retry_budget(tmp_path):
    class RejectedCreateCalendar(FakeCalendar):
        async def create_events(self, *, account, events, idempotency_key):
            self.calls.append(("create", account, idempotency_key))
            raise CalendarWriteRejectedError("sanitized provider rejection")

    async def scenario():
        calendar = RejectedCreateCalendar()
        bot = FakeBot()
        service, state, pipeline = make_service(
            tmp_path,
            bot=bot,
            gateway=FakeGateway(),
            gemini=FakeGemini([create_plan()]),
            calendar=calendar,
        )
        await process_voice(service, update_id=126, bot_message_id=626)
        return calendar, bot, state, pipeline

    calendar, bot, state, pipeline = asyncio.run(scenario())

    assert [call[0] for call in calendar.calls] == ["create"]
    assert state.job(126)["status"] == "sent"
    assert "calendar_write_retry_count" not in state.job(126)
    final_html = bot.edited_html[-1]["html"]
    assert "Google Calendar не изменён" in final_html
    assert "могло примениться" not in final_html
    record = pipeline.store.find_by_source("telegram-update:126")
    assert record["stage"] == "rejected"
    assert record["items"][0]["stage"] == "failed"


def test_permanent_prewrite_failure_terminalizes_as_calendar_unchanged(
    tmp_path, monkeypatch
):
    class UnreadableTargetCalendar(FakeCalendar):
        async def get_event(self, *, account, event_id):
            self.calls.append(("get", account, event_id))
            raise RuntimeError("provider validation rejected before write")

    async def scenario():
        calendar = UnreadableTargetCalendar()
        target = FakeCalendar.snapshot("event-existing", calendar_event())
        calendar.events[target.event_id] = target
        bot = FakeBot()
        service, state, pipeline = make_service(
            tmp_path,
            bot=bot,
            gateway=FakeGateway(["Добавь место переговорная А"]),
            gemini=FakeGemini([update_plan("e1")]),
            calendar=calendar,
        )
        await expose_candidate(pipeline, target, source_update_id=9124)
        # Force the trusted cache past its TTL so this test still exercises
        # the provider preflight failure path.
        monkeypatch.setattr(
            "tg_voice_transcriber_bot.operations._utc_now",
            lambda: datetime(2030, 1, 1, tzinfo=timezone.utc),
        )
        await process_voice(service, update_id=124, bot_message_id=624)
        return calendar, bot, state, pipeline

    calendar, bot, state, pipeline = asyncio.run(scenario())

    assert [call[0] for call in calendar.calls] == ["get"]
    assert state.job(124)["status"] == "sent"
    assert "calendar_write_retry_count" not in state.job(124)
    final_html = bot.edited_html[-1]["html"]
    assert "Google Calendar не изменён" in final_html
    assert "provider validation" not in final_html
    assert pipeline.store.find_by_source("telegram-update:124") is None

import asyncio
from copy import deepcopy
from datetime import datetime
import json
from pathlib import Path
import ssl
import uuid
from zoneinfo import ZoneInfo

import httpx
import pytest

import tg_voice_transcriber_bot.gigachat as gigachat_module
from tg_voice_transcriber_bot.gemini import (
    PLANNER_MODEL_FIELD,
    GeminiProviderChain,
    GeminiProviderStage,
    GeminiRateLimitError,
    ProviderAuthenticationError,
    ProviderCreditError,
    ProviderPermanentError,
)
from tg_voice_transcriber_bot.gigachat import (
    GigaChatApi,
    GigaChatApiError,
    GigaChatAuthenticationError,
    GigaChatConfigurationError,
    GigaChatQuotaError,
    GigaChatRateLimitError,
    GigaChatRequestRejectedError,
    _giga_function_schema,
)
from tg_voice_transcriber_bot.openrouter import (
    _OPENROUTER_CALENDAR_INTENT_SCHEMA,
    _OPENROUTER_CALENDAR_OPERATION_SCHEMA,
)


NOW = datetime(2026, 8, 28, 12, tzinfo=ZoneInfo("Europe/Moscow"))
NOW_EPOCH = NOW.timestamp()
TEST_CREDENTIALS = "Y2xpZW50LWlkOmNsaWVudC1zZWNyZXQ="
EXTRACT_FUNCTION = "extract_calendar_event"
PLAN_FUNCTION = "plan_calendar_actions"

CALENDAR_RESULT = {
    "action": "create",
    "events": [
        {
            "title": "Позвонить врачу",
            "start_at": "2026-08-29T10:00:00+03:00",
            "end_at": "2026-08-29T11:00:00+03:00",
            "all_day": False,
            "timezone": "Europe/Moscow",
            "location": None,
            "description": None,
            "recurrence_rrule": None,
        }
    ],
    "clarification_question": None,
    "confidence": 0.95,
}

CALENDAR_PLAN = {
    "action": "execute",
    "operations": [
        {
            "type": "update",
            "target_event_id": "e1",
            "patch": {
                "location": "метро Киевская",
            },
            "clear_fields": [],
        }
    ],
    "confidence": 0.98,
}


@pytest.fixture
def ca_bundle(tmp_path: Path) -> Path:
    path = tmp_path / "russian-trusted-root-ca.pem"
    path.write_text("test-only-ca", encoding="ascii")
    return path


def completion(result, *, function_name=None, actual_model="GigaChat-2-Max"):
    if function_name is None:
        function_name = (
            EXTRACT_FUNCTION if "events" in result else PLAN_FUNCTION
        )
    return {
        "choices": [
            {
                "finish_reason": "function_call",
                "index": 0,
                "message": {
                    "role": "assistant",
                    "function_call": {
                        "name": function_name,
                        "arguments": deepcopy(result),
                    },
                },
            }
        ],
        "model": actual_model,
        "usage": {
            "prompt_tokens": 101,
            "completion_tokens": 23,
            "total_tokens": 124,
        },
    }


def assert_no_any_of(value):
    if isinstance(value, dict):
        assert "anyOf" not in value
        for item in value.values():
            assert_no_any_of(item)
    elif isinstance(value, list):
        for item in value:
            assert_no_any_of(item)


def token_response(token="access-token-one", *, expires_at=None):
    if expires_at is None:
        expires_at = (NOW_EPOCH + 3600) * 1000
    return {
        "access_token": token,
        "expires_at": expires_at,
    }


def api(client, ca_bundle, **changes):
    arguments = {
        "ca_bundle_path": ca_bundle,
        "timeout_seconds": 30,
        "timezone": "Europe/Moscow",
        "client": client,
        "clock": lambda: NOW_EPOCH,
    }
    arguments.update(changes)
    return GigaChatApi(TEST_CREDENTIALS, **arguments)


def test_oauth_headers_form_and_forced_function_payload(ca_bundle, caplog):
    observed = []

    def handler(request: httpx.Request) -> httpx.Response:
        observed.append(request)
        if request.url.path == "/api/v2/oauth":
            return httpx.Response(200, json=token_response())
        return httpx.Response(200, json=completion(CALENDAR_RESULT))

    async def scenario():
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(handler), trust_env=False
        ) as http_client:
            return await api(http_client, ca_bundle).extract_event(
                "Завтра в десять позвонить врачу",
                reference_time=NOW,
                account="personal",
            )

    caplog.set_level("INFO", logger="tg_voice_transcriber_bot.planner.gigachat")
    result = asyncio.run(scenario())
    oauth, chat = observed
    oauth_form = dict(
        item.split("=", 1) for item in oauth.content.decode("ascii").split("&")
    )
    payload = json.loads(chat.content)

    assert result["events"][0]["title"] == "Позвонить врачу"
    assert oauth.method == "POST"
    assert str(oauth.url) == (
        "https://ngw.devices.sberbank.ru:9443/api/v2/oauth"
    )
    assert oauth.headers["authorization"] == f"Basic {TEST_CREDENTIALS}"
    assert uuid.UUID(oauth.headers["rquid"]).version == 4
    assert oauth.headers["content-type"] == "application/x-www-form-urlencoded"
    assert oauth.headers["accept"] == "application/json"
    assert oauth.headers["user-agent"]
    assert oauth_form == {"scope": "GIGACHAT_API_CORP"}

    assert chat.method == "POST"
    assert str(chat.url) == "https://api.giga.chat/v1/chat/completions"
    assert chat.headers["authorization"] == "Bearer access-token-one"
    assert chat.headers["user-agent"] == oauth.headers["user-agent"]
    assert payload["model"] == "GigaChat-2-Max"
    assert payload["temperature"] == 0.1
    assert payload["max_tokens"] == 8192
    assert payload["stream"] is False
    assert payload["function_call"] == {"name": EXTRACT_FUNCTION}
    assert len(payload["functions"]) == 1
    function = payload["functions"][0]
    assert function["name"] == EXTRACT_FUNCTION
    assert function["description"]
    assert function["parameters"]["type"] == "object"
    assert_no_any_of(function["parameters"])
    assert "response_format" not in payload
    assert '"actual_model":"GigaChat-2-Max"' in caplog.text
    assert '"finish_reason":"function_call"' in caplog.text
    assert '"total_tokens":124' in caplog.text


def test_access_token_is_cached_until_refresh_skew(ca_bundle):
    calls = {"oauth": 0, "chat": 0}
    clock = [NOW_EPOCH]

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/v2/oauth":
            calls["oauth"] += 1
            return httpx.Response(
                200,
                json=token_response(
                    f"access-token-{calls['oauth']}",
                    expires_at=(NOW_EPOCH + 120) * 1000,
                ),
            )
        calls["chat"] += 1
        return httpx.Response(200, json=completion(CALENDAR_RESULT))

    async def scenario():
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(handler)
        ) as http_client:
            provider = api(
                http_client,
                ca_bundle,
                clock=lambda: clock[0],
                token_refresh_skew_seconds=60,
            )
            await provider.extract_event(
                "Завтра позвонить врачу в десять",
                reference_time=NOW,
                account="personal",
            )
            clock[0] = NOW_EPOCH + 59
            await provider.extract_event(
                "Завтра позвонить врачу в десять",
                reference_time=NOW,
                account="personal",
            )
            clock[0] = NOW_EPOCH + 61
            await provider.extract_event(
                "Завтра позвонить врачу в десять",
                reference_time=NOW,
                account="personal",
            )

    asyncio.run(scenario())
    assert calls == {"oauth": 2, "chat": 3}


def test_one_401_forces_one_token_refresh(ca_bundle):
    oauth_tokens = []
    bearer_headers = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/v2/oauth":
            token = f"access-token-{len(oauth_tokens) + 1}"
            oauth_tokens.append(token)
            return httpx.Response(200, json=token_response(token))
        bearer_headers.append(request.headers["authorization"])
        if len(bearer_headers) == 1:
            return httpx.Response(401, json={"message": "expired"})
        return httpx.Response(200, json=completion(CALENDAR_RESULT))

    async def scenario():
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(handler)
        ) as http_client:
            await api(http_client, ca_bundle).extract_event(
                "Завтра позвонить врачу в десять",
                reference_time=NOW,
                account="personal",
            )

    asyncio.run(scenario())
    assert len(oauth_tokens) == 2
    assert bearer_headers == [
        "Bearer access-token-1",
        "Bearer access-token-2",
    ]


def test_retries_429_and_5xx_with_bounded_delays(ca_bundle):
    chat_attempts = 0
    sleeps = []

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal chat_attempts
        if request.url.path == "/api/v2/oauth":
            return httpx.Response(200, json=token_response())
        chat_attempts += 1
        if chat_attempts == 1:
            return httpx.Response(429, headers={"Retry-After": "999"})
        if chat_attempts == 2:
            return httpx.Response(503)
        return httpx.Response(200, json=completion(CALENDAR_RESULT))

    async def fake_sleep(delay):
        sleeps.append(delay)

    async def scenario():
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(handler)
        ) as http_client:
            await api(
                http_client,
                ca_bundle,
                sleep=fake_sleep,
                max_retries=2,
                max_retry_delay_seconds=3,
            ).extract_event(
                "Завтра позвонить врачу в десять",
                reference_time=NOW,
                account="personal",
            )

    asyncio.run(scenario())
    assert chat_attempts == 3
    assert sleeps == [3.0, 2.0]


def test_stage_timeout_during_retry_after_preserves_rate_limit_cause(ca_bundle):
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/v2/oauth":
            return httpx.Response(200, json=token_response())
        return httpx.Response(429, headers={"Retry-After": "30"})

    async def slow_sleep(_delay):
        await asyncio.sleep(60)

    async def scenario():
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(handler)
        ) as http_client:
            provider = api(
                http_client,
                ca_bundle,
                sleep=slow_sleep,
                max_retries=1,
            )
            chain = GeminiProviderChain(
                [GeminiProviderStage("GigaChat 2 Max", provider, 0.01)],
                timeout_seconds=0.1,
            )
            with pytest.raises(GigaChatRateLimitError):
                await chain.extract_event(
                    "Завтра позвонить врачу в десять",
                    reference_time=NOW,
                    account="personal",
                )

    asyncio.run(scenario())


def test_malformed_retry_after_overflow_is_ignored(monkeypatch):
    def overflow(_value):
        raise OverflowError

    monkeypatch.setattr(gigachat_module, "parsedate_to_datetime", overflow)
    response = httpx.Response(429, headers={"Retry-After": "not-a-date"})

    assert GigaChatApi._retry_after(response) is None


@pytest.mark.parametrize(
    ("status", "expected_type"),
    [
        (400, GigaChatRequestRejectedError),
        (402, GigaChatQuotaError),
        (429, GigaChatRateLimitError),
    ],
)
def test_chat_statuses_map_to_provider_error_classes(
    ca_bundle, status, expected_type
):
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/v2/oauth":
            return httpx.Response(200, json=token_response())
        return httpx.Response(status, json={"message": "provider detail"})

    async def scenario():
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(handler)
        ) as http_client:
            with pytest.raises(expected_type) as caught:
                await api(
                    http_client, ca_bundle, max_retries=0
                ).extract_event(
                    "Завтра позвонить врачу в десять",
                    reference_time=NOW,
                    account="personal",
                )
            return caught.value

    error = asyncio.run(scenario())
    if status == 400:
        assert isinstance(error, ProviderPermanentError)
    elif status == 402:
        assert isinstance(error, ProviderCreditError)
    else:
        assert isinstance(error, GeminiRateLimitError)


def test_oauth_rejection_maps_to_authentication_error(ca_bundle):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"message": "invalid secret"})

    async def scenario():
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(handler)
        ) as http_client:
            with pytest.raises(GigaChatAuthenticationError) as caught:
                await api(http_client, ca_bundle, max_retries=0).validate()
            return caught.value

    error = asyncio.run(scenario())
    assert isinstance(error, ProviderAuthenticationError)
    assert isinstance(error, ProviderPermanentError)


def test_recursive_provider_json_is_sanitized():
    class RecursiveResponse:
        content = b"{}"

        @staticmethod
        def json():
            raise RecursionError

    with pytest.raises(GigaChatApiError, match="invalid JSON"):
        GigaChatApi._response_json(RecursiveResponse())  # type: ignore[arg-type]


def test_recursive_function_arguments_are_sanitized():
    arguments = {}
    arguments["self"] = arguments
    body = {
        "choices": [
            {
                "finish_reason": "function_call",
                "message": {
                    "function_call": {
                        "name": "plan_calendar_actions",
                        "arguments": arguments,
                    }
                },
            }
        ]
    }

    with pytest.raises(GigaChatApiError, match="invalid function arguments"):
        GigaChatApi._function_arguments(
            body,
            expected_function_name="plan_calendar_actions",
        )


def test_validate_authorizes_and_requires_exact_stable_model_alias(ca_bundle):
    requested_paths = []

    def handler(request: httpx.Request) -> httpx.Response:
        requested_paths.append(request.url.path)
        if request.url.path == "/api/v2/oauth":
            return httpx.Response(200, json=token_response())
        return httpx.Response(
            200,
            json={
                "data": [
                    {"id": "GigaChat-2"},
                    {"id": "GigaChat-2-Max"},
                    {"id": "GigaChat-2-Max-preview"},
                ]
            },
        )

    async def scenario():
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(handler)
        ) as http_client:
            await api(http_client, ca_bundle).validate()

    asyncio.run(scenario())
    assert requested_paths == ["/api/v2/oauth", "/v1/models"]


def test_validate_rejects_catalog_without_exact_alias(ca_bundle):
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/v2/oauth":
            return httpx.Response(200, json=token_response())
        return httpx.Response(
            200,
            json={"data": [{"id": "GigaChat-2-Max-preview"}]},
        )

    async def scenario():
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(handler)
        ) as http_client:
            with pytest.raises(GigaChatConfigurationError):
                await api(http_client, ca_bundle).validate()

    asyncio.run(scenario())


def test_constructor_requires_stable_model_ca_and_safe_user_agent(ca_bundle):
    async def scenario():
        async with httpx.AsyncClient() as http_client:
            with pytest.raises(GigaChatConfigurationError):
                api(
                    http_client,
                    ca_bundle,
                    model="GigaChat-2-Max-preview",
                )
            with pytest.raises(GigaChatConfigurationError):
                api(http_client, ca_bundle, user_agent="")
            with pytest.raises(GigaChatConfigurationError):
                api(http_client, ca_bundle.parent / "missing.pem")

    asyncio.run(scenario())


def test_schema_transform_unwraps_nullable_unions_and_makes_them_optional():
    intent_schema = _giga_function_schema(
        _OPENROUTER_CALENDAR_INTENT_SCHEMA
    )
    operation_schema = _giga_function_schema(
        _OPENROUTER_CALENDAR_OPERATION_SCHEMA
    )

    assert_no_any_of(intent_schema)
    assert_no_any_of(operation_schema)

    intent_required = intent_schema["required"]
    event_schema = intent_schema["properties"]["events"]["items"]
    assert "clarification_question" not in intent_required
    assert set(event_schema["required"]) == {
        "title",
        "all_day",
        "timezone",
    }
    for optional_name in (
        "start_at",
        "end_at",
        "location",
        "description",
        "recurrence_rrule",
    ):
        assert optional_name not in event_schema["required"]

    assert set(operation_schema["required"]) == {
        "action",
        "operations",
        "confidence",
    }
    operation = operation_schema["properties"]["operations"]["items"]
    assert set(operation["required"]) == {"type", "clear_fields"}
    patch = operation["properties"]["patch"]
    assert patch["type"] == "object"
    assert patch["required"] == []
    lookup = operation_schema["properties"]["lookup"]
    assert set(lookup["required"]) == {"time_min", "time_max"}


def test_planner_preserves_history_image_continuation_and_model_marker(
    ca_bundle,
):
    payloads = []
    observation = {
        "description": "Скриншот подтверждения бронирования",
        "visible_text": "Сб 29 августа | 8:00–10:00\nLunda Padel",
        "source": "telegram_photo",
        "mode": "vision_description_and_ocr",
    }
    lookup_plan = {
        "action": "lookup",
        "operations": [],
        "lookup": {
            "query": "Lunda Padel",
            "time_min": "2026-08-28T00:00:00+03:00",
            "time_max": "2026-09-01T00:00:00+03:00",
        },
        "clarification_question": None,
        "confidence": 0.9,
    }

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/v2/oauth":
            return httpx.Response(200, json=token_response())
        payloads.append(json.loads(request.content))
        result = lookup_plan if len(payloads) == 1 else CALENDAR_PLAN
        return httpx.Response(200, json=completion(result))

    async def scenario():
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(handler)
        ) as http_client:
            provider = api(http_client, ca_bundle)
            first = await provider.plan_calendar_actions(
                "",
                reference_time=NOW,
                account="personal",
                application_state={"allowed_event_ids": []},
                recent_conversation=[],
                input_kind="image",
                image_observations=[observation],
            )
            history = [
                first["_interaction_input"],
                {
                    "type": "thought",
                    "signature": "DO-NOT-SEND-NATIVE-SIGNATURE",
                    "content": [
                        {
                            "type": "text",
                            "text": "DO-NOT-SEND-NATIVE-SIGNATURE",
                        }
                    ],
                },
                *first["_interaction_steps"],
            ]
            second = await provider.plan_calendar_actions(
                "",
                reference_time=NOW,
                account="personal",
                application_state={
                    "allowed_event_ids": ["e1"],
                    "candidate_events": [{"event_id": "e1"}],
                    "lookup_permitted": False,
                },
                recent_conversation=[],
                history_steps=history,
                input_kind="image",
                image_observations=[observation],
            )
            return first, second

    first, second = asyncio.run(scenario())
    messages = payloads[1]["messages"]
    raw = json.dumps(payloads[1], ensure_ascii=False)
    historical_text = messages[1]["content"]
    current_text = messages[-1]["content"]
    historical_observations = historical_text.partition(
        '<image_observations format="application/json" trust="untrusted" '
        'role="evidence_only">\n'
    )[2].partition("\n</image_observations>")[0]
    current_observations = current_text.partition(
        '<image_observations format="application/json" trust="untrusted" '
        'role="evidence_only">\n'
    )[2].partition("\n</image_observations>")[0]

    assert first[PLANNER_MODEL_FIELD] == "GigaChat-2-Max"
    assert second[PLANNER_MODEL_FIELD] == "GigaChat-2-Max"
    assert first["_interaction_steps"][0]["content"][0]["text"] == json.dumps(
        lookup_plan,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    assert [message["role"] for message in messages] == [
        "system",
        "user",
        "assistant",
        "user",
    ]
    assert "<image_observations>" in messages[0]["content"]
    assert json.loads(historical_observations) == [observation]
    assert json.loads(current_observations) == []
    assert '"image_evidence_in_history":true' in current_text
    assert "DO-NOT-SEND-NATIVE-SIGNATURE" not in raw
    assert second["operations"][0]["patch"] == {
        "location": "метро Киевская"
    }
    assert payloads[1]["function_call"] == {"name": PLAN_FUNCTION}
    assert len(payloads[1]["functions"]) == 1
    function = payloads[1]["functions"][0]
    assert function["name"] == PLAN_FUNCTION
    assert function["parameters"]["type"] == "object"
    assert_no_any_of(function["parameters"])
    assert "response_format" not in payloads[1]
    assert "<openrouter_patch_encoding>" not in messages[0]["content"]


def test_concurrent_calls_are_serialized_by_one_slot_semaphore(ca_bundle):
    active_chat_calls = 0
    max_active_chat_calls = 0
    oauth_calls = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal active_chat_calls, max_active_chat_calls, oauth_calls
        if request.url.path == "/api/v2/oauth":
            oauth_calls += 1
            return httpx.Response(200, json=token_response())
        active_chat_calls += 1
        max_active_chat_calls = max(max_active_chat_calls, active_chat_calls)
        await asyncio.sleep(0)
        active_chat_calls -= 1
        return httpx.Response(200, json=completion(CALENDAR_RESULT))

    async def scenario():
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(handler)
        ) as http_client:
            provider = api(http_client, ca_bundle)
            await asyncio.gather(
                provider.extract_event(
                    "Завтра позвонить врачу в десять",
                    reference_time=NOW,
                    account="personal",
                ),
                provider.extract_event(
                    "Завтра позвонить врачу в одиннадцать",
                    reference_time=NOW,
                    account="personal",
                ),
            )

    asyncio.run(scenario())
    assert oauth_calls == 1
    assert max_active_chat_calls == 1


def test_error_messages_and_logs_never_expose_credentials_or_tokens(
    ca_bundle, caplog
):
    credential_secret = "U0VDUkVUX0NSRURFTlRJQUw="
    token_secret = "VERY-SECRET-OAUTH-TOKEN"
    provider_body_secret = "VERY-SECRET-PROVIDER-DETAIL"

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/v2/oauth":
            return httpx.Response(
                200,
                json=token_response(token_secret),
            )
        return httpx.Response(
            500,
            json={"message": provider_body_secret},
        )

    async def scenario():
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(handler)
        ) as http_client:
            provider = GigaChatApi(
                credential_secret,
                ca_bundle_path=ca_bundle,
                timeout_seconds=30,
                timezone="Europe/Moscow",
                client=http_client,
                clock=lambda: NOW_EPOCH,
                max_retries=0,
            )
            with pytest.raises(GigaChatApiError) as caught:
                await provider.extract_event(
                    "Завтра позвонить врачу в десять",
                    reference_time=NOW,
                    account="personal",
                )
            return str(caught.value)

    caplog.set_level("INFO", logger="tg_voice_transcriber_bot.planner.gigachat")
    error_message = asyncio.run(scenario())
    exposed = caplog.text + error_message

    assert credential_secret not in exposed
    assert token_secret not in exposed
    assert provider_body_secret not in exposed
    assert "status=500" in caplog.text


@pytest.mark.parametrize(
    ("malformation", "error_fragment"),
    [
        ("finish", "incomplete"),
        ("name", "unexpected function"),
        ("arguments", "invalid function arguments"),
        ("missing_call", "no function call"),
    ],
)
def test_malformed_function_envelope_is_rejected_without_body_leak(
    ca_bundle, caplog, malformation, error_fragment
):
    body_secret = "DO-NOT-LOG-MALFORMED-FUNCTION-BODY"

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/v2/oauth":
            return httpx.Response(200, json=token_response())
        body = completion(CALENDAR_RESULT)
        call = body["choices"][0]["message"]["function_call"]
        if malformation == "finish":
            body["choices"][0]["finish_reason"] = "stop"
        elif malformation == "name":
            call["name"] = "wrong_function"
        elif malformation == "arguments":
            call["arguments"] = json.dumps(
                {"secret": body_secret}, ensure_ascii=False
            )
        else:
            body["choices"][0]["message"] = {"content": body_secret}
        body["provider_debug"] = body_secret
        return httpx.Response(200, json=body)

    async def scenario():
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(handler)
        ) as http_client:
            with pytest.raises(GigaChatApiError, match=error_fragment) as caught:
                await api(http_client, ca_bundle).extract_event(
                    "Завтра позвонить врачу в десять",
                    reference_time=NOW,
                    account="personal",
                )
            return str(caught.value)

    caplog.set_level("INFO", logger="tg_voice_transcriber_bot.planner.gigachat")
    error_message = asyncio.run(scenario())
    assert body_secret not in caplog.text + error_message


def test_owned_http_client_uses_supplied_ca_without_environment_proxy(
    ca_bundle, monkeypatch
):
    ssl_context = object()
    observed = {}

    class FakeAsyncClient:
        def __init__(self, **kwargs):
            observed["client_kwargs"] = kwargs

        async def aclose(self):
            observed["closed"] = True

    def fake_create_default_context(*, cafile):
        observed["cafile"] = cafile
        return ssl_context

    monkeypatch.setattr(
        gigachat_module.ssl,
        "create_default_context",
        fake_create_default_context,
    )
    monkeypatch.setattr(gigachat_module.httpx, "AsyncClient", FakeAsyncClient)

    provider = GigaChatApi(
        TEST_CREDENTIALS,
        ca_bundle_path=ca_bundle,
        timeout_seconds=30,
        timezone="Europe/Moscow",
    )
    asyncio.run(provider.aclose())

    assert observed["cafile"] == str(ca_bundle.resolve())
    assert observed["client_kwargs"]["verify"] is ssl_context
    assert observed["client_kwargs"]["trust_env"] is False
    limits = observed["client_kwargs"]["limits"]
    assert limits.max_connections == 1
    assert limits.max_keepalive_connections == 1
    assert observed["closed"] is True


def test_bad_ca_contents_are_reported_without_ssl_details(
    ca_bundle, monkeypatch
):
    def fail_create_default_context(*, cafile):
        assert cafile == str(ca_bundle.resolve())
        raise ssl.SSLError("sensitive absolute provider detail")

    monkeypatch.setattr(
        gigachat_module.ssl,
        "create_default_context",
        fail_create_default_context,
    )
    with pytest.raises(GigaChatConfigurationError) as caught:
        GigaChatApi(
            TEST_CREDENTIALS,
            ca_bundle_path=ca_bundle,
            timeout_seconds=30,
            timezone="Europe/Moscow",
        )

    assert "sensitive absolute provider detail" not in str(caught.value)

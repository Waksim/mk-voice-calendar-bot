import asyncio
from copy import deepcopy
from datetime import datetime
import json
from zoneinfo import ZoneInfo

import httpx
import pytest

from tg_voice_transcriber_bot.gemini import (
    GeminiCliError,
    GeminiFallback,
    GeminiRateLimitError,
)
from tg_voice_transcriber_bot.intent import (
    CALENDAR_INTENT_SCHEMA,
)
from tg_voice_transcriber_bot.openrouter import (
    OpenRouterApi,
    OpenRouterApiError,
    OpenRouterAuthenticationError,
    OpenRouterCreditError,
    OpenRouterRateLimitError,
    OpenRouterRequestRejectedError,
)


NOW = datetime(2026, 8, 25, 12, tzinfo=ZoneInfo("Europe/Moscow"))

CALENDAR_RESULT = {
    "action": "create",
    "events": [
        {
            "title": "Позвонить врачу",
            "start_at": "2026-08-26T10:00:00+03:00",
            "end_at": "2026-08-26T11:00:00+03:00",
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
            "recurrence_scope": None,
            "event": None,
            "patch": {
                "title": None,
                "start_at": None,
                "end_at": None,
                "all_day": None,
                "timezone": None,
                "location": "метро Киевская",
                "description": None,
                "recurrence_rrule": None,
            },
            "clear_fields": [],
        }
    ],
    "lookup": None,
    "clarification_question": None,
    "confidence": 0.98,
}


def assert_strict_object_schemas(node):
    if isinstance(node, dict):
        if node.get("type") == "object":
            properties = node.get("properties")
            assert isinstance(properties, dict)
            assert node.get("additionalProperties") is False
            assert set(node.get("required", [])) == set(properties)
        for value in node.values():
            assert_strict_object_schemas(value)
    elif isinstance(node, list):
        for value in node:
            assert_strict_object_schemas(value)


def completion(result, **message_fields):
    return {
        "id": "generation-id",
        "choices": [
            {
                "finish_reason": "stop",
                "message": {
                    "role": "assistant",
                    "content": json.dumps(result, ensure_ascii=False),
                    **message_fields,
                },
            }
        ],
    }


def api(client, **changes):
    arguments = {
        "timeout_seconds": 30,
        "timezone": "Europe/Moscow",
        "client": client,
    }
    arguments.update(changes)
    return OpenRouterApi("openrouter-unit-test-secret", **arguments)


def test_extract_uses_nemotron_free_strict_schema_and_required_provider_flags():
    observed = {}

    def handler(request: httpx.Request) -> httpx.Response:
        observed["request"] = request
        observed["payload"] = json.loads(request.content)
        return httpx.Response(200, json=completion(CALENDAR_RESULT))

    async def scenario():
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(handler)
        ) as http_client:
            return await api(http_client).extract_event(
                "Завтра в десять позвонить врачу",
                reference_time=NOW,
                account="personal",
            )

    result = asyncio.run(scenario())
    request = observed["request"]
    payload = observed["payload"]

    assert result["events"][0]["title"] == "Позвонить врачу"
    assert request.method == "POST"
    assert str(request.url) == "https://openrouter.ai/api/v1/chat/completions"
    assert request.headers["authorization"] == (
        "Bearer openrouter-unit-test-secret"
    )
    assert "openrouter-unit-test-secret" not in str(request.url)
    assert payload["model"] == "nvidia/nemotron-3-super-120b-a12b:free"
    assert payload["reasoning"] == {"effort": "medium"}
    assert payload["max_tokens"] == 8192
    assert payload["provider"] == {
        "require_parameters": True,
        "data_collection": "allow",
    }
    response_format = payload["response_format"]
    assert response_format["type"] == "json_schema"
    assert response_format["json_schema"]["name"] == "calendar_intent"
    assert response_format["json_schema"]["strict"] is True
    wire_schema = response_format["json_schema"]["schema"]
    assert wire_schema["properties"]["action"] == (
        CALENDAR_INTENT_SCHEMA["properties"]["action"]
    )
    assert "maxItems" not in wire_schema["properties"]["events"]
    assert "maxLength" not in wire_schema["properties"]["events"]["items"][
        "properties"
    ]["title"]
    assert [message["role"] for message in payload["messages"]] == ["user"]
    assert "tools" not in payload
    assert "store" not in payload
    assert_strict_object_schemas(wire_schema)


def test_reasoning_effort_and_max_tokens_are_configurable():
    observed = {}

    def handler(request: httpx.Request) -> httpx.Response:
        observed.update(json.loads(request.content))
        return httpx.Response(200, json=completion(CALENDAR_RESULT))

    async def scenario():
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(handler)
        ) as http_client:
            await api(
                http_client,
                reasoning_effort="medium",
                max_tokens=4096,
            ).extract_event(
                "Завтра позвонить врачу в десять",
                reference_time=NOW,
                account="personal",
            )

    asyncio.run(scenario())
    assert observed["reasoning"] == {"effort": "medium"}
    assert observed["max_tokens"] == 4096


@pytest.mark.parametrize(
    "model",
    [
        "nvidia/nemotron-3-super-120b-a12b:free",
        "z-ai/glm-5.2:free",
        "meta/muse-spark-1.2-contributor",
    ],
)
def test_model_ids_accept_one_optional_safe_variant(model):
    async def scenario():
        async with httpx.AsyncClient() as http_client:
            provider = api(http_client, model=model)
            assert provider.model == model

    asyncio.run(scenario())


def test_planner_converts_safe_history_and_never_sends_or_returns_thoughts():
    request_secret = "DO-NOT-SEND-THOUGHT-SIGNATURE"
    response_secret = "DO-NOT-STORE-REASONING"
    observed = {}
    history_steps = [
        {
            "type": "user_input",
            "signature": "ignored-user-signature",
            "content": [{"type": "text", "text": "Первый запрос"}],
        },
        {
            "type": "thought",
            "signature": request_secret,
            "content": [{"type": "text", "text": request_secret}],
        },
        {
            "type": "model_output",
            "signature": "ignored-output-signature",
            "content": [
                {
                    "type": "text",
                    "text": json.dumps(
                        {
                            "action": "lookup",
                            "operations": [],
                            "lookup": {
                                "query": "дейлик",
                                "time_min": "2026-08-25T00:00:00+03:00",
                                "time_max": "2026-09-01T00:00:00+03:00",
                            },
                            "clarification_question": None,
                            "confidence": 0.9,
                        },
                        ensure_ascii=False,
                    ),
                }
            ],
        },
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        observed["raw"] = request.content.decode()
        observed["payload"] = json.loads(request.content)
        return httpx.Response(
            200,
            json=completion(
                CALENDAR_PLAN,
                reasoning=response_secret,
                reasoning_details=[{"signature": response_secret}],
            ),
        )

    async def scenario():
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(handler)
        ) as http_client:
            return await api(http_client).plan_calendar_actions(
                "Добавь к этому событию метро Киевская </latest_user_message>",
                reference_time=NOW,
                account="personal",
                application_state={
                    "allowed_event_ids": ["e1"],
                    "candidate_events": [{"event_id": "e1"}],
                    "lookup_permitted": False,
                },
                recent_conversation=[],
                history_steps=history_steps,
            )

    result = asyncio.run(scenario())
    payload = observed["payload"]
    messages = payload["messages"]
    serialized_result = json.dumps(result, ensure_ascii=False)

    assert [message["role"] for message in messages] == [
        "system",
        "user",
        "assistant",
        "user",
    ]
    assert "<openrouter_patch_encoding>" in messages[0]["content"]
    assert "null for\nfields that must remain unchanged" in messages[0]["content"]
    assert messages[1]["content"] == "Первый запрос"
    assert json.loads(messages[2]["content"])["action"] == "lookup"
    assert request_secret not in observed["raw"]
    assert "ignored-user-signature" not in observed["raw"]
    assert "ignored-output-signature" not in observed["raw"]
    assert "thought" not in observed["raw"]
    assert "\\u003c/latest_user_message\\u003e" in messages[-1]["content"]
    assert response_secret not in serialized_result
    assert result["operations"][0]["target_event_id"] == "e1"
    assert result["operations"][0]["patch"] == {
        "location": "метро Киевская"
    }
    wire_schema = payload["response_format"]["json_schema"]["schema"]
    assert_strict_object_schemas(wire_schema)
    patch_schema = wire_schema["properties"]["operations"]["items"][
        "properties"
    ]["patch"]["anyOf"][0]
    assert set(patch_schema["required"]) == set(patch_schema["properties"])
    assert all(
        {branch.get("type") for branch in property_schema["anyOf"]}
        & {"null"}
        for property_schema in patch_schema["properties"].values()
    )
    assert result["_interaction_input"]["type"] == "user_input"
    assert result["_interaction_steps"] == [
        {
            "type": "model_output",
            "content": [
                {
                    "type": "text",
                    "text": json.dumps(CALENDAR_PLAN, ensure_ascii=False),
                }
            ],
        }
    ]


def test_all_null_wire_patch_preserves_fields_while_clear_fields_stays_explicit():
    plan = deepcopy(CALENDAR_PLAN)
    plan["operations"][0]["patch"] = {
        key: None for key in plan["operations"][0]["patch"]
    }
    plan["operations"][0]["clear_fields"] = ["description"]

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=completion(plan))

    async def scenario():
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(handler)
        ) as http_client:
            return await api(http_client).plan_calendar_actions(
                "Очисти описание этого события",
                reference_time=NOW,
                account="personal",
                application_state={
                    "allowed_event_ids": ["e1"],
                    "candidate_events": [{"event_id": "e1"}],
                    "lookup_permitted": False,
                },
                recent_conversation=[],
            )

    result = asyncio.run(scenario())
    assert result["operations"][0]["patch"] is None
    assert result["operations"][0]["clear_fields"] == ["description"]


def test_all_null_wire_patch_without_clear_is_rejected_as_noop():
    plan = deepcopy(CALENDAR_PLAN)
    plan["operations"][0]["patch"] = {
        key: None for key in plan["operations"][0]["patch"]
    }

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=completion(plan))

    async def scenario():
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(handler)
        ) as http_client:
            await api(http_client).plan_calendar_actions(
                "Измени это событие",
                reference_time=NOW,
                account="personal",
                application_state={
                    "allowed_event_ids": ["e1"],
                    "candidate_events": [{"event_id": "e1"}],
                    "lookup_permitted": False,
                },
                recent_conversation=[],
            )

    with pytest.raises(OpenRouterApiError, match="invalid calendar plan"):
        asyncio.run(scenario())


def test_validate_checks_authenticated_key_then_public_model_metadata():
    requests = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.path == "/api/v1/key":
            return httpx.Response(
                200,
                json={"data": {"label": "owner", "limit_remaining": 1}},
            )
        if request.url.path == "/api/v1/credits":
            return httpx.Response(
                200,
                json={"data": {"total_credits": 5, "total_usage": 0}},
            )
        if request.url.path == (
            "/api/v1/model/nvidia/nemotron-3-super-120b-a12b:free"
        ):
            return httpx.Response(
                200,
                json={
                    "data": {
                        "id": "nvidia/nemotron-3-super-120b-a12b:free",
                        "supported_parameters": [
                            "max_tokens",
                            "reasoning",
                            "response_format",
                            "structured_outputs",
                        ],
                        "reasoning": {"supported_efforts": ["medium"]},
                    }
                },
            )
        return httpx.Response(404)

    async def scenario():
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(handler)
        ) as http_client:
            await api(http_client).validate()

    asyncio.run(scenario())

    assert [request.method for request in requests] == ["GET", "GET", "GET"]
    assert requests[0].headers["authorization"] == (
        "Bearer openrouter-unit-test-secret"
    )
    assert requests[1].headers["authorization"] == (
        "Bearer openrouter-unit-test-secret"
    )
    assert requests[2].headers["authorization"] == (
        "Bearer openrouter-unit-test-secret"
    )
    assert all("openrouter-unit-test-secret" not in str(item.url) for item in requests)
    assert requests[2].url.raw_path.endswith(b"%3Afree")


@pytest.mark.parametrize(
    ("key_data", "credit_data", "changes"),
    [
        ({"limit_remaining": 0}, None, {}),
        (
            {"limit_remaining": 1},
            {"total_credits": 0, "total_usage": 0},
            {"model": "paid/model"},
        ),
        (
            {"limit_remaining": 1},
            {"total_credits": 5, "total_usage": 5},
            {"model": "paid/model"},
        ),
        (
            {"limit_remaining": 1},
            {"total_credits": 0, "total_usage": 1},
            {},
        ),
    ],
)
def test_validate_rejects_exhausted_key_limit_or_account_balance(
    key_data, credit_data, changes
):
    requests = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.path == "/api/v1/key":
            return httpx.Response(200, json={"data": key_data})
        if request.url.path == "/api/v1/credits" and credit_data is not None:
            return httpx.Response(200, json={"data": credit_data})
        raise AssertionError(f"unexpected request: {request.url.path}")

    async def scenario():
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(handler)
        ) as http_client:
            await api(http_client, **changes).validate()

    with pytest.raises(OpenRouterCreditError):
        asyncio.run(scenario())
    assert all("/api/v1/model/" not in request.url.path for request in requests)


def test_validate_allows_zero_account_balance_for_free_model():
    requests = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.path == "/api/v1/key":
            return httpx.Response(
                200,
                json={"data": {"label": "owner", "limit_remaining": None}},
            )
        if request.url.path == "/api/v1/credits":
            return httpx.Response(
                200,
                json={"data": {"total_credits": 5, "total_usage": 5}},
            )
        if request.url.path == (
            "/api/v1/model/nvidia/nemotron-3-super-120b-a12b:free"
        ):
            return httpx.Response(
                200,
                json={
                    "data": {
                        "id": "nvidia/nemotron-3-super-120b-a12b:free",
                        "supported_parameters": [
                            "max_tokens",
                            "reasoning",
                            "response_format",
                            "structured_outputs",
                        ],
                        "reasoning": {"supported_efforts": ["medium"]},
                    }
                },
            )
        raise AssertionError(f"unexpected request: {request.url.path}")

    async def scenario():
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(handler)
        ) as http_client:
            await api(http_client).validate()

    asyncio.run(scenario())
    assert [request.url.path for request in requests] == [
        "/api/v1/key",
        "/api/v1/credits",
        "/api/v1/model/nvidia/nemotron-3-super-120b-a12b:free",
    ]


@pytest.mark.parametrize("status", [408, 429, 500, 503, 524, 529])
def test_retryable_http_statuses_retry_then_succeed(status):
    calls = 0
    delays = []

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            return httpx.Response(status, headers={"Retry-After": "0.25"})
        return httpx.Response(200, json=completion(CALENDAR_RESULT))

    async def fake_sleep(delay):
        delays.append(delay)

    async def scenario():
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(handler)
        ) as http_client:
            return await api(
                http_client,
                max_retries=1,
                sleep=fake_sleep,
            ).extract_event(
                "Завтра позвонить врачу в десять",
                reference_time=NOW,
                account="personal",
            )

    result = asyncio.run(scenario())
    assert result["action"] == "create"
    assert calls == 2
    assert delays == [0.25]


def test_http_402_is_distinct_non_retryable_and_secret_safe():
    calls = 0
    secret = "private-provider-error-and-prompt"

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(402, json={"error": {"message": secret}})

    async def scenario():
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(handler)
        ) as http_client:
            await api(http_client, max_retries=5).extract_event(
                secret,
                reference_time=NOW,
                account="personal",
            )

    with pytest.raises(OpenRouterCreditError) as raised:
        asyncio.run(scenario())
    assert calls == 1
    assert secret not in str(raised.value)
    assert "openrouter-unit-test-secret" not in str(raised.value)


def test_authentication_rejection_is_distinct_non_retryable_and_secret_safe(
):
    calls = 0
    secret = "private-provider-error-and-prompt"

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(401, json={"error": {"message": secret}})

    async def scenario():
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(handler)
        ) as http_client:
            await api(http_client, max_retries=5).extract_event(
                secret,
                reference_time=NOW,
                account="personal",
            )

    with pytest.raises(OpenRouterAuthenticationError) as raised:
        asyncio.run(scenario())
    assert calls == 1
    assert secret not in str(raised.value)
    assert "openrouter-unit-test-secret" not in str(raised.value)


def test_http_403_is_request_rejection_not_misreported_as_bad_key():
    calls = 0
    secret = "private-moderation-or-policy-detail"

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(403, json={"error": {"message": secret}})

    async def scenario():
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(handler)
        ) as http_client:
            await api(http_client, max_retries=5).extract_event(
                "Завтра встреча",
                reference_time=NOW,
                account="personal",
            )

    with pytest.raises(OpenRouterRequestRejectedError) as raised:
        asyncio.run(scenario())
    assert not isinstance(raised.value, OpenRouterAuthenticationError)
    assert calls == 1
    assert secret not in str(raised.value)


def test_cli_fallback_failure_does_not_hide_openrouter_credit_error():
    class NoCreditPrimary:
        async def plan_calendar_actions(self, transcript, **kwargs):
            raise OpenRouterCreditError("OpenRouter API credits are exhausted")

    class FailedFallback:
        calls = 0

        @staticmethod
        def is_available():
            return True

        async def plan_calendar_actions(self, transcript, **kwargs):
            type(self).calls += 1
            raise GeminiCliError("private fallback response")

    async def scenario():
        provider = GeminiFallback(
            NoCreditPrimary(),  # type: ignore[arg-type]
            FailedFallback(),  # type: ignore[arg-type]
            timeout_seconds=1,
        )
        await provider.plan_calendar_actions(
            "Добавь встречу",
            reference_time=NOW,
            account="personal",
            application_state={"allowed_event_ids": []},
            recent_conversation=[],
        )

    with pytest.raises(OpenRouterCreditError):
        asyncio.run(scenario())
    assert FailedFallback.calls == 0


def test_credit_validation_failure_does_not_report_cli_as_muse_readiness():
    class NoCreditPrimary:
        async def validate(self):
            raise OpenRouterCreditError("OpenRouter API credits are exhausted")

    class UnexpectedFallback:
        calls = 0

        @staticmethod
        def is_available():
            return True

        async def validate(self):
            type(self).calls += 1

    async def scenario():
        provider = GeminiFallback(
            NoCreditPrimary(),  # type: ignore[arg-type]
            UnexpectedFallback(),  # type: ignore[arg-type]
            timeout_seconds=1,
        )
        await provider.validate()

    with pytest.raises(OpenRouterCreditError):
        asyncio.run(scenario())
    assert UnexpectedFallback.calls == 0


def test_openrouter_authentication_error_does_not_invoke_cli_fallback():
    class RejectedPrimary:
        async def plan_calendar_actions(self, transcript, **kwargs):
            raise OpenRouterAuthenticationError(
                "OpenRouter API credential or access was rejected"
            )

    class UnexpectedFallback:
        calls = 0

        @staticmethod
        def is_available():
            return True

        async def plan_calendar_actions(self, transcript, **kwargs):
            type(self).calls += 1
            raise AssertionError("permanent errors must not enter fallback")

    async def scenario():
        provider = GeminiFallback(
            RejectedPrimary(),  # type: ignore[arg-type]
            UnexpectedFallback(),  # type: ignore[arg-type]
            timeout_seconds=1,
        )
        await provider.plan_calendar_actions(
            "Добавь встречу",
            reference_time=NOW,
            account="personal",
            application_state={"allowed_event_ids": []},
            recent_conversation=[],
        )

    with pytest.raises(OpenRouterAuthenticationError):
        asyncio.run(scenario())
    assert UnexpectedFallback.calls == 0


def test_cli_fallback_failure_does_not_hide_openrouter_rate_limit_type():
    class RateLimitedPrimary:
        async def plan_calendar_actions(self, transcript, **kwargs):
            raise OpenRouterRateLimitError(
                "OpenRouter API rate limit exceeded"
            )

    class FailedFallback:
        @staticmethod
        def is_available():
            return True

        async def plan_calendar_actions(self, transcript, **kwargs):
            raise GeminiCliError("private fallback response")

    async def scenario():
        provider = GeminiFallback(
            RateLimitedPrimary(),  # type: ignore[arg-type]
            FailedFallback(),  # type: ignore[arg-type]
            timeout_seconds=1,
        )
        await provider.plan_calendar_actions(
            "Добавь встречу",
            reference_time=NOW,
            account="personal",
            application_state={"allowed_event_ids": []},
            recent_conversation=[],
        )

    with pytest.raises(OpenRouterRateLimitError) as raised:
        asyncio.run(scenario())
    assert str(raised.value) == "OpenRouter API rate limit exceeded"


def test_slow_cli_fallback_does_not_hide_openrouter_rate_limit_type():
    class RateLimitedPrimary:
        async def plan_calendar_actions(self, transcript, **kwargs):
            raise OpenRouterRateLimitError(
                "OpenRouter API rate limit exceeded"
            )

    class HangingFallback:
        @staticmethod
        def is_available():
            return True

        async def plan_calendar_actions(self, transcript, **kwargs):
            await asyncio.Event().wait()

    async def scenario():
        provider = GeminiFallback(
            RateLimitedPrimary(),  # type: ignore[arg-type]
            HangingFallback(),  # type: ignore[arg-type]
            timeout_seconds=0.01,
        )
        await provider.plan_calendar_actions(
            "Добавь встречу",
            reference_time=NOW,
            account="personal",
            application_state={"allowed_event_ids": []},
            recent_conversation=[],
        )

    with pytest.raises(OpenRouterRateLimitError) as raised:
        asyncio.run(scenario())
    assert str(raised.value) == "OpenRouter API rate limit exceeded"


def test_primary_budget_cancellation_during_429_backoff_preserves_subtype():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429)

    async def blocking_sleep(_delay):
        await asyncio.Event().wait()

    class FailedFallback:
        @staticmethod
        def is_available():
            return True

        async def extract_event(self, transcript, **kwargs):
            raise GeminiCliError("private fallback response")

    async def scenario():
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(handler)
        ) as http_client:
            primary = api(
                http_client,
                max_retries=1,
                sleep=blocking_sleep,
            )
            provider = GeminiFallback(
                primary,
                FailedFallback(),  # type: ignore[arg-type]
                timeout_seconds=0.02,
            )
            await provider.extract_event(
                "Завтра встреча",
                reference_time=NOW,
                account="personal",
            )

    with pytest.raises(OpenRouterRateLimitError) as raised:
        asyncio.run(scenario())
    assert str(raised.value) == "OpenRouter API rate limit exceeded"


def test_exhausted_http_429_raises_typed_rate_error():
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(429, json={"error": {"message": "private"}})

    async def fake_sleep(_delay):
        return None

    async def scenario():
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(handler)
        ) as http_client:
            await api(
                http_client,
                max_retries=1,
                sleep=fake_sleep,
            ).extract_event(
                "Завтра позвонить врачу в десять",
                reference_time=NOW,
                account="personal",
            )

    with pytest.raises(OpenRouterRateLimitError) as raised:
        asyncio.run(scenario())
    assert isinstance(raised.value, GeminiRateLimitError)
    assert calls == 2


def test_http_429_without_retry_after_uses_long_backoff():
    calls = 0
    delays = []

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls < 3:
            return httpx.Response(429)
        return httpx.Response(200, json=completion(CALENDAR_RESULT))

    async def fake_sleep(delay):
        delays.append(delay)

    async def scenario():
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(handler)
        ) as http_client:
            return await api(
                http_client,
                max_retries=2,
                sleep=fake_sleep,
            ).extract_event(
                "Завтра позвонить врачу в десять",
                reference_time=NOW,
                account="personal",
            )

    assert asyncio.run(scenario())["action"] == "create"
    assert calls == 3
    assert delays == [10.0, 20.0]


def test_timeout_is_not_retried_to_avoid_duplicate_generation_billing():
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        raise httpx.ReadTimeout("private", request=request)

    async def scenario():
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(handler)
        ) as http_client:
            await api(http_client, max_retries=5).extract_event(
                "Завтра позвонить врачу в десять",
                reference_time=NOW,
                account="personal",
            )

    with pytest.raises(OpenRouterApiError, match="request timed out"):
        asyncio.run(scenario())
    assert calls == 1


def test_oversized_planner_request_is_rejected_before_transport():
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(200, json=completion(CALENDAR_PLAN))

    async def scenario():
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(handler)
        ) as http_client:
            await api(http_client).plan_calendar_actions(
                "Измени событие",
                reference_time=NOW,
                account="personal",
                application_state={
                    "allowed_event_ids": [],
                    "oversized": "x" * 70_000,
                },
                recent_conversation=[],
            )

    with pytest.raises(OpenRouterApiError, match="request is too large"):
        asyncio.run(scenario())
    assert calls == 0


def test_oversized_response_is_rejected_without_parsing_body():
    secret = "response-private-secret"

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=(secret.encode() + b"x" * OpenRouterApi._MAX_RESPONSE_BYTES),
        )

    async def scenario():
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(handler)
        ) as http_client:
            await api(http_client).extract_event(
                "Завтра позвонить врачу в десять",
                reference_time=NOW,
                account="personal",
            )

    with pytest.raises(OpenRouterApiError, match="response was too large") as raised:
        asyncio.run(scenario())
    assert secret not in str(raised.value)


def test_invalid_semantic_plan_and_forged_target_are_rejected():
    forged = deepcopy(CALENDAR_PLAN)
    forged["operations"][0]["target_event_id"] = "provider-secret-id"

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=completion(forged))

    async def scenario():
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(handler)
        ) as http_client:
            await api(http_client).plan_calendar_actions(
                "Измени событие",
                reference_time=NOW,
                account="personal",
                application_state={"allowed_event_ids": ["e1"]},
                recent_conversation=[],
            )

    with pytest.raises(OpenRouterApiError, match="invalid calendar plan") as raised:
        asyncio.run(scenario())
    assert "provider-secret-id" not in str(raised.value)


def test_transport_errors_are_retried_and_do_not_expose_request_details():
    calls = 0
    secret = "transport-private-secret"

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        raise httpx.ConnectError(secret, request=request)

    async def fake_sleep(_delay):
        return None

    async def scenario():
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(handler)
        ) as http_client:
            await api(
                http_client,
                max_retries=1,
                sleep=fake_sleep,
            ).extract_event(
                "Завтра позвонить врачу в десять",
                reference_time=NOW,
                account="personal",
            )

    with pytest.raises(OpenRouterApiError, match="ConnectError") as raised:
        asyncio.run(scenario())
    assert calls == 2
    assert secret not in str(raised.value)
    assert "openrouter-unit-test-secret" not in str(raised.value)


@pytest.mark.parametrize(
    "changes",
    [
        {"reasoning_effort": "extreme"},
        {"max_tokens": 0},
        {"max_retries": -1},
        {"model": "https://attacker.invalid/model"},
        {"model": "vendor/model:free:extra"},
        {"model": "vendor/model:"},
        {"model": "vendor//model"},
        {"model": "vendor/model/extra"},
        {"model": "/model"},
        {"model": "vendor/"},
        {"model": "vendor/model?query=private"},
        {"timeout_seconds": 0},
        {"timeout_seconds": float("nan")},
    ],
)
def test_invalid_configuration_is_rejected(changes):
    async def scenario():
        async with httpx.AsyncClient() as http_client:
            api(http_client, **changes)

    with pytest.raises(OpenRouterApiError):
        asyncio.run(scenario())


def test_api_key_with_header_control_characters_is_rejected():
    with pytest.raises(OpenRouterApiError, match="key is invalid"):
        OpenRouterApi(
            "openrouter-key\nInjected: private",
            timeout_seconds=30,
            timezone="Europe/Moscow",
        )


def test_model_validation_rejects_another_public_model():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/v1/key":
            return httpx.Response(200, json={"data": {"label": "owner"}})
        if request.url.path == "/api/v1/credits":
            return httpx.Response(
                200,
                json={"data": {"total_credits": 5, "total_usage": 0}},
            )
        return httpx.Response(200, json={"data": {"id": "other/model"}})

    async def scenario():
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(handler)
        ) as http_client:
            await api(http_client).validate()

    with pytest.raises(OpenRouterApiError, match="model is unavailable"):
        asyncio.run(scenario())


def test_model_validation_requires_structured_output_and_reasoning_support():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/v1/key":
            return httpx.Response(200, json={"data": {"label": "owner"}})
        if request.url.path == "/api/v1/credits":
            return httpx.Response(
                200,
                json={"data": {"total_credits": 5, "total_usage": 0}},
            )
        return httpx.Response(
            200,
            json={
                "data": {
                    "id": "nvidia/nemotron-3-super-120b-a12b:free",
                    "supported_parameters": ["max_tokens"],
                    "reasoning": {"supported_efforts": ["medium"]},
                }
            },
        )

    async def scenario():
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(handler)
        ) as http_client:
            await api(http_client).validate()

    with pytest.raises(OpenRouterApiError, match="lacks required parameters"):
        asyncio.run(scenario())

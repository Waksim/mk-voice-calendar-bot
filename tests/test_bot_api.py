import asyncio
import json

import httpx
import pytest

from tg_voice_transcriber_bot import bot_api as bot_api_module
from tg_voice_transcriber_bot.bot_api import (
    BotApi,
    BotApiError,
    read_keychain_secret,
    read_secret,
)


def test_keychain_read_is_bounded_and_secret_agnostic(monkeypatch):
    captured = {}

    def fake_run(arguments, **kwargs):
        captured["arguments"] = arguments
        captured["kwargs"] = kwargs
        raise bot_api_module.subprocess.TimeoutExpired(arguments, 10)

    monkeypatch.setattr(bot_api_module.subprocess, "run", fake_run)

    with pytest.raises(RuntimeError, match="Cannot read secret"):
        read_keychain_secret(account="account", service="service")

    assert captured["arguments"] == [
        "/usr/bin/security",
        "find-generic-password",
        "-a",
        "account",
        "-s",
        "service",
        "-w",
    ]
    assert captured["kwargs"]["timeout"] == 10


def test_portable_secret_prefers_file_then_environment_and_falls_back(
    tmp_path, monkeypatch
):
    secret_file = tmp_path / "secret"
    secret_file.write_text(" file-value\n", encoding="utf-8")
    monkeypatch.setenv("TEST_SECRET_FILE", str(secret_file))
    assert read_secret(environment="TEST_SECRET") == "file-value"

    monkeypatch.delenv("TEST_SECRET_FILE")
    monkeypatch.setenv("TEST_SECRET", " env-value ")
    assert read_secret(environment="TEST_SECRET") == "env-value"

    monkeypatch.delenv("TEST_SECRET")
    monkeypatch.setattr(
        bot_api_module,
        "read_keychain_secret",
        lambda **_kwargs: "keychain-value",
    )
    assert (
        read_secret(environment="TEST_SECRET", account="account", service="service")
        == "keychain-value"
    )


def test_portable_secret_rejects_conflicting_or_empty_sources(tmp_path, monkeypatch):
    secret_file = tmp_path / "secret"
    secret_file.write_text("value", encoding="utf-8")
    monkeypatch.setenv("TEST_SECRET", "value")
    monkeypatch.setenv("TEST_SECRET_FILE", str(secret_file))
    with pytest.raises(RuntimeError, match="conflicting"):
        read_secret(environment="TEST_SECRET")

    monkeypatch.delenv("TEST_SECRET_FILE")
    monkeypatch.setenv("TEST_SECRET", "   ")
    with pytest.raises(RuntimeError, match="empty"):
        read_secret(environment="TEST_SECRET")


def test_retry_after_is_preserved_without_token_in_error():
    async def scenario():
        def handler(request):
            return httpx.Response(
                429,
                json={
                    "ok": False,
                    "error_code": 429,
                    "description": "Too Many Requests",
                    "parameters": {"retry_after": 17},
                },
            )

        api = BotApi("test-secret")
        await api._client.aclose()
        api._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        try:
            try:
                await api.call("getMe")
            except BotApiError as exc:
                assert exc.retry_after == 17
                assert "test-secret" not in str(exc)
            else:
                raise AssertionError("BotApiError was not raised")
        finally:
            await api._client.aclose()

    asyncio.run(scenario())


def test_callbacks_are_polled_and_keyboard_is_attached_to_first_chunk_only():
    async def scenario():
        requests = []

        def handler(request):
            requests.append(
                (request.url.path, json.loads(request.content.decode("utf-8")))
            )
            return httpx.Response(200, json={"ok": True, "result": []})

        api = BotApi("test-secret")
        await api._client.aclose()
        api._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        markup = {
            "inline_keyboard": [
                [{"text": "Добавить", "callback_data": "cal:add:abcdefghijklmnop"}]
            ]
        }
        try:
            await api.get_updates(12)
            await api.send_text(
                123,
                "а" * 4100,
                reply_to_message_id=77,
                reply_markup=markup,
            )
            await api.answer_callback_query("callback-1", "Готово")
            await api.remove_inline_keyboard(123, 88)
        finally:
            await api._client.aclose()
        return requests, markup

    requests, markup = asyncio.run(scenario())
    assert requests[0][0].endswith("/getUpdates")
    assert requests[0][1]["allowed_updates"] == ["message", "callback_query"]

    send_requests = [item for item in requests if item[0].endswith("/sendMessage")]
    assert len(send_requests) == 2
    assert send_requests[0][1]["reply_markup"] == markup
    assert "reply_markup" not in send_requests[1][1]
    assert "parse_mode" not in send_requests[0][1]
    assert "link_preview_options" not in send_requests[0][1]
    assert send_requests[0][1]["reply_parameters"]["message_id"] == 77
    assert "reply_parameters" not in send_requests[1][1]

    answer = next(item for item in requests if item[0].endswith("/answerCallbackQuery"))
    assert answer[1] == {
        "callback_query_id": "callback-1",
        "text": "Готово",
        "show_alert": False,
    }
    edit = next(
        item for item in requests if item[0].endswith("/editMessageReplyMarkup")
    )
    assert edit[1]["reply_markup"] == {"inline_keyboard": []}


def test_callback_data_and_answer_limits_are_checked_before_request():
    async def scenario():
        api = BotApi("test-secret")
        try:
            with pytest.raises(ValueError, match="1-64"):
                await api.send_text(
                    123,
                    "preview",
                    reply_markup={
                        "inline_keyboard": [
                            [{"text": "x", "callback_data": "x" * 65}]
                        ]
                    },
                )
            with pytest.raises(ValueError, match="200-character"):
                await api.answer_callback_query("callback-1", "x" * 201)
        finally:
            await api._client.aclose()

    asyncio.run(scenario())


def test_html_send_returns_message_id_and_edit_controls_keyboard():
    async def scenario():
        requests = []

        def handler(request):
            payload = json.loads(request.content.decode("utf-8"))
            requests.append((request.url.path, payload))
            result = {"message_id": 321} if request.url.path.endswith("/sendMessage") else True
            return httpx.Response(200, json={"ok": True, "result": result})

        api = BotApi("test-secret")
        await api._client.aclose()
        api._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        markup = {
            "inline_keyboard": [
                [{"text": "Отменить", "callback_data": "cal:undo:abcdefghijklmnop"}]
            ]
        }
        try:
            sent_id = await api.send_html(
                123,
                "<b>Готово</b>",
                reply_to_message_id=77,
                reply_markup=markup,
            )
            await api.edit_html(123, sent_id, "<b>Шаг 1</b>")
            await api.edit_html(123, sent_id, "<b>Шаг 2</b>", reply_markup=None)
            await api.edit_html(123, sent_id, "<b>Шаг 3</b>", reply_markup=markup)
        finally:
            await api._client.aclose()
        return sent_id, requests, markup

    sent_id, requests, markup = asyncio.run(scenario())
    assert sent_id == 321

    send_payload = requests[0][1]
    assert send_payload == {
        "chat_id": 123,
        "text": "<b>Готово</b>",
        "parse_mode": "HTML",
        "link_preview_options": {"is_disabled": True},
        "reply_parameters": {
            "message_id": 77,
            "allow_sending_without_reply": True,
        },
        "reply_markup": markup,
    }

    edits = [payload for path, payload in requests if path.endswith("/editMessageText")]
    assert len(edits) == 3
    assert "reply_markup" not in edits[0]
    assert edits[1]["reply_markup"] == {"inline_keyboard": []}
    assert edits[2]["reply_markup"] == markup
    assert all(edit["parse_mode"] == "HTML" for edit in edits)
    assert all(
        edit["link_preview_options"] == {"is_disabled": True} for edit in edits
    )


def test_html_send_rejects_invalid_message_result_without_leaking_token():
    async def scenario():
        def handler(request):
            return httpx.Response(200, json={"ok": True, "result": True})

        api = BotApi("test-secret")
        await api._client.aclose()
        api._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        try:
            with pytest.raises(BotApiError, match="invalid sent message") as caught:
                await api.send_html(123, "<b>Готово</b>")
            assert "test-secret" not in str(caught.value)
        finally:
            await api._client.aclose()

    asyncio.run(scenario())


def test_html_methods_validate_arguments_before_request():
    async def scenario():
        api = BotApi("test-secret")
        try:
            with pytest.raises(ValueError, match="non-empty"):
                await api.send_html(123, "")
            with pytest.raises(ValueError, match="positive"):
                await api.edit_html(123, 0, "valid")
            with pytest.raises(TypeError, match="UNSET"):
                await api.edit_html(123, 1, "valid", reply_markup="bad")  # type: ignore[arg-type]
        finally:
            await api._client.aclose()

    asyncio.run(scenario())


def test_webhook_configuration_is_authenticated_and_does_not_delete_webhook():
    async def scenario():
        requests = []

        def handler(request):
            requests.append(
                (request.url.path, json.loads(request.content.decode("utf-8")))
            )
            return httpx.Response(200, json={"ok": True, "result": True})

        api = BotApi("test-secret")
        await api._client.aclose()
        api._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        try:
            await api.configure_profile()
            await api.set_webhook(
                "https://calendar.example.test/telegram/bot/webhook", "webhook_secret-123"
            )
        finally:
            await api._client.aclose()
        return requests

    requests = asyncio.run(scenario())
    assert not any(path.endswith("/deleteWebhook") for path, _payload in requests)
    description = next(
        payload["description"]
        for path, payload in requests
        if path.endswith("/setMyDescription")
    )
    short_description = next(
        payload["short_description"]
        for path, payload in requests
        if path.endswith("/setMyShortDescription")
    )
    assert "Muse Spark 1.2 через OpenRouter" in description
    assert "Gemini" not in description
    assert "Muse Spark 1.2 (OpenRouter)" in short_description
    webhook_path, webhook_payload = requests[-1]
    assert webhook_path.endswith("/setWebhook")
    assert webhook_payload == {
        "url": "https://calendar.example.test/telegram/bot/webhook",
        "secret_token": "webhook_secret-123",
        "max_connections": 1,
        "allowed_updates": ["message", "callback_query"],
        "drop_pending_updates": False,
    }


def test_polling_configuration_remains_backward_compatible():
    async def scenario():
        requests = []

        def handler(request):
            requests.append(request.url.path)
            return httpx.Response(200, json={"ok": True, "result": True})

        api = BotApi("test-secret")
        await api._client.aclose()
        api._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        try:
            await api.configure()
        finally:
            await api._client.aclose()
        return requests

    requests = asyncio.run(scenario())
    assert requests[0].endswith("/deleteWebhook")
    assert requests[1].endswith("/setMyCommands")

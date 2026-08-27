import asyncio
import json

import httpx
import pytest

from tg_voice_transcriber_bot import bot_api as bot_api_module
from tg_voice_transcriber_bot.bot_api import (
    BotApi,
    BotApiError,
    BotApiFileError,
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


def test_bot_api_lifecycle_logs_are_secret_safe(caplog):
    async def scenario():
        calls = 0

        def handler(request):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise httpx.ConnectError(
                    "PRIVATE_TRANSPORT_DIAGNOSTIC", request=request
                )
            return httpx.Response(
                200,
                json={
                    "ok": True,
                    "result": {"text": "PRIVATE_BOT_API_RESULT"},
                },
            )

        api = BotApi("PRIVATE_BOT_TOKEN")
        await api._client.aclose()
        api._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        try:
            result = await api.call(
                "sendMessage", {"text": "PRIVATE_BOT_API_ARGUMENT"}
            )
            assert result["text"] == "PRIVATE_BOT_API_RESULT"
            with pytest.raises(BotApiError, match="ConnectError"):
                await api.call("getMe")
        finally:
            await api._client.aclose()

    with caplog.at_level("INFO", logger="tg_voice_transcriber_bot.bot_api"):
        asyncio.run(scenario())

    messages = "\n".join(
        record.getMessage()
        for record in caplog.records
        if record.name == "tg_voice_transcriber_bot.bot_api"
    )
    assert "method=sendMessage" in messages
    assert "method=getMe" in messages
    assert "status=started" in messages
    assert "status=success" in messages
    assert "status=transport_error" in messages
    assert "error_type=ConnectError" in messages
    assert "elapsed=" in messages
    assert "PRIVATE_BOT_TOKEN" not in messages
    assert "PRIVATE_BOT_API_ARGUMENT" not in messages
    assert "PRIVATE_BOT_API_RESULT" not in messages
    assert "PRIVATE_TRANSPORT_DIAGNOSTIC" not in messages
    assert "api.telegram.org" not in messages


def test_get_file_and_bounded_download_are_validated_and_secret_safe(caplog):
    token = "PRIVATE_BOT_TOKEN"
    file_id = "PRIVATE_FILE_ID"
    file_path = "photos/PRIVATE_FILE_PATH.jpg"
    requests = []

    async def scenario():
        def handler(request):
            requests.append((request.method, request.url.path, request.content))
            if request.url.path.endswith("/getFile"):
                return httpx.Response(
                    200,
                    json={
                        "ok": True,
                        "result": {
                            "file_id": file_id,
                            "file_unique_id": "UNIQUE_FILE_ID",
                            "file_size": 6,
                            "file_path": file_path,
                        },
                    },
                )
            return httpx.Response(200, content=b"image!")

        api = BotApi(token)
        await api._client.aclose()
        api._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        try:
            metadata = await api.get_file(file_id, max_file_size=10)
            content = await api.download_file(metadata["file_path"], max_bytes=10)
        finally:
            await api._client.aclose()
        return metadata, content

    with caplog.at_level("INFO", logger="tg_voice_transcriber_bot.bot_api"):
        metadata, content = asyncio.run(scenario())

    assert metadata == {
        "file_id": file_id,
        "file_unique_id": "UNIQUE_FILE_ID",
        "file_size": 6,
        "file_path": file_path,
    }
    assert content == b"image!"
    get_file_request = requests[0]
    assert get_file_request[0] == "POST"
    assert get_file_request[1].endswith("/getFile")
    assert json.loads(get_file_request[2].decode("utf-8")) == {"file_id": file_id}
    assert requests[1][0] == "GET"

    logs = "\n".join(record.getMessage() for record in caplog.records)
    assert "method=getFile" in logs
    assert "status=success" in logs
    assert "bytes=6" in logs
    assert token not in logs
    assert file_id not in logs
    assert file_path not in logs
    assert "api.telegram.org" not in logs


def test_get_file_rejects_invalid_or_oversized_metadata_without_identifiers():
    async def scenario(result, *, max_file_size=10):
        def handler(request):
            return httpx.Response(200, json={"ok": True, "result": result})

        api = BotApi("PRIVATE_BOT_TOKEN")
        await api._client.aclose()
        api._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        try:
            with pytest.raises(BotApiError) as caught:
                await api.get_file("PRIVATE_FILE_ID", max_file_size=max_file_size)
            return str(caught.value)
        finally:
            await api._client.aclose()

    oversized_error = asyncio.run(
        scenario(
            {
                "file_id": "PRIVATE_FILE_ID",
                "file_unique_id": "PRIVATE_UNIQUE_ID",
                "file_size": 11,
                "file_path": "photos/PRIVATE_FILE_PATH.jpg",
            }
        )
    )
    invalid_path_error = asyncio.run(
        scenario(
            {
                "file_id": "PRIVATE_FILE_ID",
                "file_size": 5,
                "file_path": "photos/../PRIVATE_FILE_PATH.jpg",
            }
        )
    )

    for message in (oversized_error, invalid_path_error):
        assert "PRIVATE_BOT_TOKEN" not in message
        assert "PRIVATE_FILE_ID" not in message
        assert "PRIVATE_FILE_PATH" not in message


def test_download_file_rejects_bad_paths_and_enforces_declared_and_streamed_size():
    async def scenario():
        calls = 0

        class NoLengthStream(httpx.AsyncByteStream):
            async def __aiter__(self):
                yield b"123"
                yield b"456"

        def handler(request):
            nonlocal calls
            calls += 1
            if calls == 1:
                return httpx.Response(
                    200,
                    headers={"content-length": "6"},
                    content=b"123456",
                )
            return httpx.Response(200, stream=NoLengthStream())

        api = BotApi("PRIVATE_BOT_TOKEN")
        await api._client.aclose()
        api._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        try:
            for bad_path in (
                "/photos/file.jpg",
                "photos/../file.jpg",
                "photos//file.jpg",
                "photos/file.jpg?token=secret",
            ):
                with pytest.raises(ValueError, match="path is invalid"):
                    await api.download_file(bad_path, max_bytes=5)
            with pytest.raises(BotApiError, match="size limit") as declared:
                await api.download_file("photos/file.jpg", max_bytes=5)
            with pytest.raises(BotApiError, match="size limit") as streamed:
                await api.download_file("photos/file.jpg", max_bytes=5)
            return str(declared.value), str(streamed.value), calls
        finally:
            await api._client.aclose()

    declared_error, streamed_error, calls = asyncio.run(scenario())
    assert calls == 2
    for message in (declared_error, streamed_error):
        assert "PRIVATE_BOT_TOKEN" not in message
        assert "photos/file.jpg" not in message


def test_file_download_transport_error_does_not_expose_token_or_path(caplog):
    async def scenario():
        def handler(request):
            raise httpx.ConnectError(
                "PRIVATE_TRANSPORT_DETAIL", request=request
            )

        api = BotApi("PRIVATE_BOT_TOKEN")
        await api._client.aclose()
        api._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        try:
            with pytest.raises(BotApiError, match="ConnectError") as caught:
                await api.download_file(
                    "photos/PRIVATE_FILE_PATH.jpg", max_bytes=100
                )
            return str(caught.value)
        finally:
            await api._client.aclose()

    with caplog.at_level("INFO", logger="tg_voice_transcriber_bot.bot_api"):
        message = asyncio.run(scenario())
    combined = message + "\n" + "\n".join(
        record.getMessage() for record in caplog.records
    )
    assert "PRIVATE_BOT_TOKEN" not in combined
    assert "PRIVATE_FILE_PATH" not in combined
    assert "PRIVATE_TRANSPORT_DETAIL" not in combined
    assert "api.telegram.org" not in combined


def test_permanent_file_api_errors_are_terminal_but_rate_limits_remain_retryable():
    async def scenario():
        responses = iter(
            (
                httpx.Response(
                    400,
                    json={
                        "ok": False,
                        "error_code": 400,
                        "description": "bad file",
                    },
                ),
                httpx.Response(404, content=b"not found"),
                httpx.Response(
                    429,
                    json={
                        "ok": False,
                        "error_code": 429,
                        "description": "rate limited",
                        "parameters": {"retry_after": 9},
                    },
                ),
            )
        )

        def handler(_request):
            return next(responses)

        api = BotApi("PRIVATE_BOT_TOKEN")
        await api._client.aclose()
        api._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        try:
            with pytest.raises(BotApiFileError):
                await api.get_file("PRIVATE_FILE_ID")
            with pytest.raises(BotApiFileError):
                await api.download_file("photos/file.jpg")
            with pytest.raises(BotApiError) as rate_limited:
                await api.get_file("PRIVATE_FILE_ID")
            assert not isinstance(rate_limited.value, BotApiFileError)
            assert rate_limited.value.retry_after == 9
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
    assert "ИИ-планировщик" in description
    assert "Muse" not in description
    assert "OpenRouter" not in description
    assert "Gemini" not in description
    assert "ИИ-планировщик" in short_description
    assert "Muse" not in short_description
    assert "OpenRouter" not in short_description
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

import asyncio

import pytest
from aiohttp import ClientSession

from tg_voice_transcriber_bot.bot_api import BotApiError
from tg_voice_transcriber_bot.calendar import CalendarConnectionError
from tg_voice_transcriber_bot.gateway import GatewayConnectionError, GatewayError
from tg_voice_transcriber_bot.state import StateStore
from tg_voice_transcriber_bot.webhook import WebhookRuntime

SECRET_HEADER = {"X-Telegram-Bot-Api-Secret-Token": "valid_secret-123"}


async def wait_until(predicate, *, timeout=2):
    async def poll():
        while not predicate():
            await asyncio.sleep(0.005)

    await asyncio.wait_for(poll(), timeout=timeout)


class BlockingHandler:
    def __init__(self):
        self.calls = []
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    async def handle_update(self, update):
        self.calls.append(update["update_id"])
        self.started.set()
        await self.release.wait()


def test_webhook_auth_validation_body_limit_and_durable_ack(tmp_path):
    async def scenario():
        state_path = tmp_path / "state.json"
        state = StateStore(state_path)
        handler = BlockingHandler()
        runtime = WebhookRuntime(
            handler,
            state,
            secret_token="valid_secret-123",
            path="/telegram/webhook",
            host="127.0.0.1",
            port=0,
            max_body_bytes=256,
            retry_seconds=0.01,
        )
        await runtime.start()
        try:
            base_url = f"http://127.0.0.1:{runtime.bound_port}"
            async with ClientSession() as client:
                response = await client.post(
                    f"{base_url}/telegram/webhook", json={"update_id": 7}
                )
                assert response.status == 403
                await response.read()

                response = await client.post(
                    f"{base_url}/telegram/webhook",
                    data="not json",
                    headers=SECRET_HEADER,
                )
                assert response.status == 415
                await response.read()

                response = await client.post(
                    f"{base_url}/telegram/webhook",
                    data="{broken",
                    headers={**SECRET_HEADER, "Content-Type": "application/json"},
                )
                assert response.status == 400
                await response.read()

                response = await client.post(
                    f"{base_url}/telegram/webhook",
                    json={"update_id": 8, "payload": "x" * 400},
                    headers=SECRET_HEADER,
                )
                assert response.status == 413
                await response.read()

                response = await client.post(
                    f"{base_url}/telegram/webhook",
                    json={"update_id": 7, "message": {"text": "hello"}},
                    headers=SECRET_HEADER,
                )
                assert response.status == 200
                await response.read()
                await asyncio.wait_for(handler.started.wait(), timeout=1)

                # The handler is still blocked, proving that HTTP 200 is based
                # on durable queueing rather than completed business work.
                restored = StateStore(state_path)
                assert restored.pending_update_count == 1
                assert restored.next_pending_update()["update_id"] == 7

                response = await client.post(
                    f"{base_url}/telegram/webhook",
                    json={"update_id": 7, "message": {"text": "duplicate"}},
                    headers=SECRET_HEADER,
                )
                assert response.status == 200
                await response.read()

                health = await client.get(f"{base_url}/healthz")
                assert health.status == 200
                assert (await health.json())["pending_updates"] == 1

            assert handler.calls == [7]
            handler.release.set()
            await wait_until(lambda: state.pending_update_count == 0)
            assert state.completed_update_ids == (7,)
        finally:
            handler.release.set()
            await runtime.close()

    asyncio.run(scenario())


def test_webhook_returns_non_2xx_if_atomic_persistence_fails(tmp_path):
    async def scenario():
        state = StateStore(tmp_path / "state.json")

        def fail_save():
            raise OSError("simulated disk failure")

        state.save = fail_save  # type: ignore[method-assign]
        handler = BlockingHandler()
        runtime = WebhookRuntime(
            handler,
            state,
            secret_token="valid_secret-123",
            path="/hook",
            host="127.0.0.1",
            port=0,
            retry_seconds=0.01,
        )
        await runtime.start()
        try:
            async with ClientSession() as client:
                response = await client.post(
                    f"http://127.0.0.1:{runtime.bound_port}/hook",
                    json={"update_id": 91},
                    headers=SECRET_HEADER,
                )
                assert response.status == 503
                await response.read()
            assert state.pending_update_count == 0
            assert handler.calls == []
        finally:
            handler.release.set()
            await runtime.close()

    asyncio.run(scenario())


def test_serial_worker_retries_in_order_without_using_polling_offset(tmp_path):
    class RetryHandler:
        def __init__(self):
            self.calls = []
            self.active = 0
            self.max_active = 0

        async def handle_update(self, update):
            update_id = update["update_id"]
            self.calls.append(update_id)
            self.active += 1
            self.max_active = max(self.max_active, self.active)
            try:
                await asyncio.sleep(0)
                if update_id == 50 and self.calls.count(50) == 1:
                    raise BotApiError("temporary failure")
            finally:
                self.active -= 1

    async def scenario():
        state = StateStore(tmp_path / "state.json")
        state.data["offset"] = 10_000
        state.save()
        assert state.enqueue_update({"update_id": 50})
        assert state.enqueue_update({"update_id": 2})
        handler = RetryHandler()
        runtime = WebhookRuntime(
            handler,
            state,
            secret_token="valid_secret-123",
            path="/hook",
            host="127.0.0.1",
            port=0,
            retry_seconds=0.01,
        )
        await runtime.start()
        try:
            await wait_until(lambda: state.pending_update_count == 0)
            return state, handler
        finally:
            await runtime.close()

    state, handler = asyncio.run(scenario())
    assert handler.calls == [50, 50, 2]
    assert handler.max_active == 1
    assert state.completed_update_ids == (50, 2)
    assert state.offset == 10_000


def test_webhook_lifecycle_logs_update_status_without_private_body(
    tmp_path, caplog
):
    class SecretRetryHandler:
        def __init__(self):
            self.calls = 0

        async def handle_update(self, _update):
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("PRIVATE_WEBHOOK_HANDLER_DIAGNOSTIC")

    async def scenario():
        state = StateStore(tmp_path / "state.json")
        handler = SecretRetryHandler()
        runtime = WebhookRuntime(
            handler,
            state,
            secret_token="valid_secret-123",
            path="/hook",
            host="127.0.0.1",
            port=0,
            retry_seconds=0.01,
        )
        await runtime.start()
        try:
            async with ClientSession() as client:
                response = await client.post(
                    f"http://127.0.0.1:{runtime.bound_port}/hook",
                    json={
                        "update_id": 81,
                        "message": {"text": "PRIVATE_WEBHOOK_USER_TEXT"},
                    },
                    headers=SECRET_HEADER,
                )
                assert response.status == 200
                await response.read()
            await wait_until(lambda: state.pending_update_count == 0)
            assert handler.calls == 2
        finally:
            await runtime.close()

    with caplog.at_level("INFO", logger="tg_voice_transcriber_bot.webhook"):
        asyncio.run(scenario())

    messages = "\n".join(
        record.getMessage()
        for record in caplog.records
        if record.name == "tg_voice_transcriber_bot.webhook"
    )
    assert "Telegram webhook receive" in messages
    assert "Telegram webhook update" in messages
    assert "update_id=81" in messages
    assert "status=enqueued" in messages
    assert "status=started" in messages
    assert "status=retry" in messages
    assert "status=completed" in messages
    assert "error_type=RuntimeError" in messages
    assert "elapsed=" in messages
    assert "PRIVATE_WEBHOOK_USER_TEXT" not in messages
    assert "PRIVATE_WEBHOOK_HANDLER_DIAGNOSTIC" not in messages
    assert "valid_secret-123" not in messages


def test_pending_update_is_drained_after_process_restart(tmp_path):
    class RecordingHandler:
        def __init__(self):
            self.calls = []

        async def handle_update(self, update):
            self.calls.append(update["update_id"])

    async def scenario():
        state_path = tmp_path / "state.json"
        first_process = StateStore(state_path)
        assert first_process.enqueue_update({"update_id": 44})

        restored = StateStore(state_path)
        handler = RecordingHandler()
        runtime = WebhookRuntime(
            handler,
            restored,
            secret_token="valid_secret-123",
            path="/hook",
            host="127.0.0.1",
            port=0,
            retry_seconds=0.01,
        )
        await runtime.start()
        try:
            await wait_until(lambda: restored.pending_update_count == 0)
            return restored, handler
        finally:
            await runtime.close()

    restored, handler = asyncio.run(scenario())
    assert handler.calls == [44]
    assert restored.completed_update_ids == (44,)


def test_retryable_gateway_error_keeps_worker_healthy(tmp_path):
    class RetryableGatewayHandler:
        def __init__(self):
            self.calls = []

        async def handle_update(self, update):
            self.calls.append(update["update_id"])
            if len(self.calls) == 1:
                raise GatewayError("temporary tool failure")

    async def scenario():
        state = StateStore(tmp_path / "state.json")
        assert state.enqueue_update({"update_id": 61})
        handler = RetryableGatewayHandler()
        runtime = WebhookRuntime(
            handler,
            state,
            secret_token="valid_secret-123",
            path="/hook",
            host="127.0.0.1",
            port=0,
            retry_seconds=0.01,
        )
        await runtime.start()
        try:
            await wait_until(lambda: state.pending_update_count == 0)
            async with ClientSession() as client:
                health = await client.get(
                    f"http://127.0.0.1:{runtime.bound_port}/healthz"
                )
                assert health.status == 200
                await health.read()
            return state, handler
        finally:
            await runtime.close()

    state, handler = asyncio.run(scenario())
    assert handler.calls == [61, 61]
    assert state.completed_update_ids == (61,)


@pytest.mark.parametrize(
    "fatal_error",
    [
        GatewayConnectionError("sanitized fatal transport failure"),
        CalendarConnectionError("sanitized fatal transport failure"),
    ],
)
def test_fatal_mcp_failure_stops_worker_and_preserves_pending_update(
    tmp_path, fatal_error
):
    class FatalMcpHandler:
        def __init__(self):
            self.calls = []

        async def handle_update(self, update):
            self.calls.append(update["update_id"])
            raise fatal_error

    async def scenario():
        state_path = tmp_path / "state.json"
        state = StateStore(state_path)
        assert state.enqueue_update({"update_id": 71})
        handler = FatalMcpHandler()
        runtime = WebhookRuntime(
            handler,
            state,
            secret_token="valid_secret-123",
            path="/hook",
            host="127.0.0.1",
            port=0,
            retry_seconds=0.01,
        )
        await runtime.start()
        try:
            await wait_until(
                lambda: runtime._worker is not None and runtime._worker.done()
            )
            async with ClientSession() as client:
                health = await client.get(
                    f"http://127.0.0.1:{runtime.bound_port}/healthz"
                )
                assert health.status == 503
                assert (await health.json())["pending_updates"] == 1

            with pytest.raises(type(fatal_error)):
                await runtime.run_forever()

            assert state.pending_update_count == 1
            assert state.completed_update_ids == ()
            assert state.next_pending_update()["update_id"] == 71
            return state_path, handler
        finally:
            await runtime.close()

    state_path, handler = asyncio.run(scenario())
    restored = StateStore(state_path)
    assert handler.calls == [71]
    assert restored.pending_update_count == 1
    assert restored.next_pending_update()["update_id"] == 71

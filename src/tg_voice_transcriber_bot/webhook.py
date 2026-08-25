"""Authenticated, crash-safe Telegram Bot API webhook runtime."""

from __future__ import annotations

import asyncio
import json
import logging
import re
import secrets
import time
from typing import Any, Protocol

from aiohttp import web

from .calendar import CalendarConnectionError
from .gateway import GatewayConnectionError
from .state import StateStore

LOGGER = logging.getLogger("tg_voice_transcriber_bot.webhook")
_SECRET_TOKEN_RE = re.compile(r"^[A-Za-z0-9_-]{1,256}$")


class UpdateHandler(Protocol):
    async def handle_update(self, update: dict[str, Any]) -> None: ...


class WebhookRuntime:
    """Persist webhook requests before ACK and process them one at a time."""

    def __init__(
        self,
        handler: UpdateHandler,
        state: StateStore,
        *,
        secret_token: str,
        path: str,
        host: str,
        port: int,
        health_path: str = "/healthz",
        max_body_bytes: int = 1_048_576,
        retry_seconds: float = 3,
    ) -> None:
        if not _SECRET_TOKEN_RE.fullmatch(secret_token):
            raise ValueError("webhook secret token has an invalid format")
        if not path.startswith("/") or "?" in path or "#" in path:
            raise ValueError("webhook path must be an absolute URL path")
        if not health_path.startswith("/") or health_path == path:
            raise ValueError("health path must be a distinct absolute URL path")
        if not host:
            raise ValueError("webhook host must not be empty")
        if not 0 <= port <= 65535:
            raise ValueError("webhook port must be between 0 and 65535")
        if max_body_bytes <= 0 or retry_seconds <= 0:
            raise ValueError("webhook limits must be positive")

        self.handler = handler
        self.state = state
        self.secret_token = secret_token
        self.path = path
        self.host = host
        self.port = port
        self.health_path = health_path
        self.max_body_bytes = max_body_bytes
        self.retry_seconds = retry_seconds
        self._wake = asyncio.Event()
        self._stop = asyncio.Event()
        self._runner: web.AppRunner | None = None
        self._site: web.TCPSite | None = None
        self._worker: asyncio.Task[None] | None = None

        self.application = web.Application(client_max_size=max_body_bytes)
        self.application.router.add_post(path, self._receive_update)
        self.application.router.add_get(health_path, self._health)

    async def _health(self, _request: web.Request) -> web.Response:
        worker_healthy = self._worker is not None and not self._worker.done()
        status = 200 if worker_healthy else 503
        return web.json_response(
            {
                "ok": worker_healthy,
                "pending_updates": self.state.pending_update_count,
            },
            status=status,
        )

    async def _receive_update(self, request: web.Request) -> web.Response:
        started = time.monotonic()
        LOGGER.info("Telegram webhook receive; status=started")
        provided_secret = request.headers.get(
            "X-Telegram-Bot-Api-Secret-Token", ""
        )
        if not secrets.compare_digest(provided_secret, self.secret_token):
            LOGGER.warning(
                "Telegram webhook receive; status=forbidden elapsed=%.3fs",
                time.monotonic() - started,
            )
            raise web.HTTPForbidden()
        if request.content_type != "application/json":
            LOGGER.warning(
                "Telegram webhook receive; status=unsupported_media_type "
                "elapsed=%.3fs",
                time.monotonic() - started,
            )
            raise web.HTTPUnsupportedMediaType()
        if (
            request.content_length is not None
            and request.content_length > self.max_body_bytes
        ):
            LOGGER.warning(
                "Telegram webhook receive; status=body_too_large elapsed=%.3fs",
                time.monotonic() - started,
            )
            raise web.HTTPRequestEntityTooLarge(
                max_size=self.max_body_bytes,
                actual_size=request.content_length,
            )
        try:
            update = await request.json(loads=json.loads)
        except web.HTTPRequestEntityTooLarge:
            LOGGER.warning(
                "Telegram webhook receive; status=body_too_large elapsed=%.3fs",
                time.monotonic() - started,
            )
            raise
        except (json.JSONDecodeError, UnicodeError, ValueError):
            LOGGER.warning(
                "Telegram webhook receive; status=invalid_json elapsed=%.3fs",
                time.monotonic() - started,
            )
            raise web.HTTPBadRequest() from None
        if not isinstance(update, dict):
            LOGGER.warning(
                "Telegram webhook receive; status=invalid_update elapsed=%.3fs",
                time.monotonic() - started,
            )
            raise web.HTTPBadRequest()

        raw_update_id = update.get("update_id")
        update_id = (
            raw_update_id
            if isinstance(raw_update_id, int) and not isinstance(raw_update_id, bool)
            else "unknown"
        )

        try:
            inserted = self.state.enqueue_update(update)
        except ValueError:
            LOGGER.warning(
                "Telegram webhook receive; update_id=%s status=invalid_update "
                "elapsed=%.3fs",
                update_id,
                time.monotonic() - started,
            )
            raise web.HTTPBadRequest() from None
        except Exception as exc:
            # A non-2xx response makes Telegram retry. Never acknowledge an
            # update whose atomic state-file write did not complete.
            LOGGER.error(
                "Telegram webhook receive; update_id=%s status=persistence_error "
                "elapsed=%.3fs error_type=%s",
                update_id,
                time.monotonic() - started,
                type(exc).__name__,
            )
            raise web.HTTPServiceUnavailable() from None
        if inserted:
            self._wake.set()
        LOGGER.info(
            "Telegram webhook receive; update_id=%s status=%s elapsed=%.3fs",
            update_id,
            "enqueued" if inserted else "duplicate",
            time.monotonic() - started,
        )
        return web.json_response({"ok": True})

    async def _wait_for_work(self) -> dict[str, Any]:
        while True:
            update = self.state.next_pending_update()
            if update is not None:
                return update
            self._wake.clear()
            # Close the small race between checking the state and clearing the
            # event while the request handler enqueues another update.
            update = self.state.next_pending_update()
            if update is not None:
                self._wake.set()
                return update
            await self._wake.wait()

    async def _retry_pause(self, exception: Exception) -> None:
        retry_after = getattr(exception, "retry_after", None)
        delay = (
            float(retry_after)
            if isinstance(retry_after, (int, float))
            and not isinstance(retry_after, bool)
            and retry_after > 0
            else self.retry_seconds
        )
        try:
            await asyncio.wait_for(self._stop.wait(), timeout=delay)
        except TimeoutError:
            pass

    async def _worker_loop(self) -> None:
        while not self._stop.is_set():
            update = await self._wait_for_work()
            update_id = int(update["update_id"])
            started = time.monotonic()
            LOGGER.info(
                "Telegram webhook update; update_id=%s status=started",
                update_id,
            )
            try:
                await self.handler.handle_update(update)
                self.state.complete_webhook_update(update_id)
                LOGGER.info(
                    "Telegram webhook update; update_id=%s status=completed "
                    "elapsed=%.3fs",
                    update_id,
                    time.monotonic() - started,
                )
            except asyncio.CancelledError:
                LOGGER.info(
                    "Telegram webhook update; update_id=%s status=cancelled "
                    "elapsed=%.3fs",
                    update_id,
                    time.monotonic() - started,
                )
                raise
            except (GatewayConnectionError, CalendarConnectionError) as exc:
                # A dead stdio MCP session cannot recover in-process. Keep the
                # durable update pending and let the container restart rebuild
                # every subprocess before attempting it again.
                LOGGER.critical(
                    "Telegram webhook update; update_id=%s "
                    "status=fatal_connection_error elapsed=%.3fs error_type=%s",
                    update_id,
                    time.monotonic() - started,
                    type(exc).__name__,
                )
                self._stop.set()
                raise
            except Exception as exc:  # noqa: BLE001 - retry boundary for update work
                # Log only the update ID and exception class: provider errors
                # can contain private calendar or transcription content.
                LOGGER.error(
                    "Telegram webhook update; update_id=%s status=retry "
                    "elapsed=%.3fs error_type=%s",
                    update_id,
                    time.monotonic() - started,
                    type(exc).__name__,
                )
                await self._retry_pause(exc)

    async def start(self) -> None:
        if self._runner is not None:
            raise RuntimeError("webhook runtime is already started")
        started = time.monotonic()
        LOGGER.info("Telegram webhook runtime; status=starting")
        self._stop.clear()
        runner = web.AppRunner(self.application, access_log=None)
        await runner.setup()
        site = web.TCPSite(runner, self.host, self.port)
        try:
            await site.start()
        except BaseException as exc:
            await runner.cleanup()
            LOGGER.error(
                "Telegram webhook runtime; status=startup_error elapsed=%.3fs "
                "error_type=%s",
                time.monotonic() - started,
                type(exc).__name__,
            )
            raise
        self._runner = runner
        self._site = site
        self._worker = asyncio.create_task(
            self._worker_loop(), name="telegram-webhook-worker"
        )
        if self.state.pending_update_count:
            self._wake.set()
        LOGGER.info(
            "Telegram webhook runtime; status=ready elapsed=%.3fs",
            time.monotonic() - started,
        )

    async def run_forever(self) -> None:
        if self._runner is None:
            raise RuntimeError("webhook runtime is not started")
        worker = self._worker
        if worker is None:
            raise RuntimeError("webhook worker is not started")
        stop_waiter = asyncio.create_task(
            self._stop.wait(), name="telegram-webhook-stop-waiter"
        )
        try:
            done, _pending = await asyncio.wait(
                (worker, stop_waiter), return_when=asyncio.FIRST_COMPLETED
            )
            if worker in done:
                # In particular, re-raise fatal MCP connection errors so the
                # top-level process exits and Docker can restart it.
                await worker
        finally:
            stop_waiter.cancel()
            try:
                await stop_waiter
            except asyncio.CancelledError:
                pass

    async def close(self) -> None:
        started = time.monotonic()
        LOGGER.info("Telegram webhook runtime; status=stopping")
        self._stop.set()
        self._wake.set()
        worker, self._worker = self._worker, None
        if worker is not None:
            if not worker.done():
                worker.cancel()
            try:
                await worker
            except asyncio.CancelledError:
                pass
            except (GatewayConnectionError, CalendarConnectionError):
                # run_forever is the propagation boundary; close only owns
                # resource cleanup and may observe the same completed task.
                pass
        runner, self._runner = self._runner, None
        self._site = None
        if runner is not None:
            await runner.cleanup()
        LOGGER.info(
            "Telegram webhook runtime; status=closed elapsed=%.3fs",
            time.monotonic() - started,
        )

    @property
    def bound_port(self) -> int | None:
        """Return the actual port, primarily for local integration tests."""

        if self._site is None or self._site._server is None:
            return None
        sockets = self._site._server.sockets
        return int(sockets[0].getsockname()[1]) if sockets else None

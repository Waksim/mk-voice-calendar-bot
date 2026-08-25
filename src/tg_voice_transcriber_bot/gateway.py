"""Typed client for the compact local Telegram MCP gateway."""

from __future__ import annotations

import logging
import os
from collections.abc import AsyncIterator
from contextlib import AsyncExitStack, asynccontextmanager
from datetime import timedelta
from http import HTTPStatus
from pathlib import Path
import re
import time
from typing import Any

import anyio
from mcp import ClientSession, McpError, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.types import CONNECTION_CLOSED


LOGGER = logging.getLogger("tg_voice_transcriber_bot.gateway")
_GATEWAY_TOOLS = frozenset(
    {"telegram_search_tools", "telegram_read", "telegram_write"}
)
_ACCOUNT_LABEL_RE = re.compile(r"[a-z0-9_-]{1,64}")
_GATEWAY_OPERATIONS = frozenset(
    {"find_recent_outgoing_voice", "get_me", "transcribe_voice_message"}
)


class GatewayError(RuntimeError):
    pass


class GatewayConnectionError(GatewayError):
    """Fatal loss of the long-lived local Telegram gateway transport."""


_CONNECTION_ERROR_MESSAGE = "Telegram gateway connection is unavailable"


def _is_connection_failure(exception: BaseException) -> bool:
    if isinstance(exception, McpError):
        return exception.error.code in {
            CONNECTION_CLOSED,
            HTTPStatus.REQUEST_TIMEOUT,
        }
    if isinstance(
        exception,
        (
            TimeoutError,
            EOFError,
            OSError,
            anyio.BrokenResourceError,
            anyio.ClosedResourceError,
            anyio.EndOfStream,
        ),
    ):
        return True
    if isinstance(exception, BaseExceptionGroup):
        return any(_is_connection_failure(item) for item in exception.exceptions)
    return False


class TelegramGateway:
    def __init__(self, session: ClientSession, default_timeout: int) -> None:
        self._session = session
        self._default_timeout = default_timeout

    async def _tool(
        self,
        name: str,
        arguments: dict[str, Any],
        *,
        timeout: int | None = None,
    ) -> dict[str, Any]:
        started = time.monotonic()
        safe_tool = name if name in _GATEWAY_TOOLS else "unknown"
        raw_account = arguments.get("account")
        account = (
            raw_account
            if isinstance(raw_account, str) and _ACCOUNT_LABEL_RE.fullmatch(raw_account)
            else "none"
        )
        raw_operation = arguments.get("operation")
        operation = (
            raw_operation
            if isinstance(raw_operation, str)
            and raw_operation in _GATEWAY_OPERATIONS
            else "catalog"
            if safe_tool == "telegram_search_tools"
            else "none"
        )
        effective_timeout = timeout or self._default_timeout
        LOGGER.info(
            "Telegram gateway tool started; tool=%s operation=%s account=%s "
            "status=started timeout=%ss",
            safe_tool,
            operation,
            account,
            effective_timeout,
        )
        try:
            result = await self._session.call_tool(
                name,
                arguments,
                read_timeout_seconds=timedelta(
                    seconds=effective_timeout
                ),
            )
        except Exception as exc:
            if _is_connection_failure(exc):
                LOGGER.error(
                    "Telegram gateway tool finished; tool=%s operation=%s "
                    "account=%s status=connection_error elapsed=%.3fs "
                    "error_type=%s",
                    safe_tool,
                    operation,
                    account,
                    time.monotonic() - started,
                    type(exc).__name__,
                )
                raise GatewayConnectionError(_CONNECTION_ERROR_MESSAGE) from None
            LOGGER.warning(
                "Telegram gateway tool finished; tool=%s operation=%s "
                "account=%s status=transport_error elapsed=%.3fs "
                "error_type=%s",
                safe_tool,
                operation,
                account,
                time.monotonic() - started,
                type(exc).__name__,
            )
            raise
        payload = result.structuredContent
        if result.isError or not isinstance(payload, dict):
            LOGGER.warning(
                "Telegram gateway tool finished; tool=%s operation=%s "
                "account=%s status=tool_error elapsed=%.3fs",
                safe_tool,
                operation,
                account,
                time.monotonic() - started,
            )
            raise GatewayError(f"Telegram gateway call failed: {name}")
        # The facade search tool returns its catalog directly; read/write calls
        # return a gateway envelope whose decoded backend value is in `result`.
        inner = payload.get("result") if "operation" in payload else payload
        if not isinstance(inner, dict):
            LOGGER.warning(
                "Telegram gateway tool finished; tool=%s operation=%s "
                "account=%s status=invalid_response elapsed=%.3fs",
                safe_tool,
                operation,
                account,
                time.monotonic() - started,
            )
            raise GatewayError(f"Telegram gateway returned an invalid result: {name}")
        LOGGER.info(
            "Telegram gateway tool finished; tool=%s operation=%s account=%s "
            "status=success elapsed=%.3fs",
            safe_tool,
            operation,
            account,
            time.monotonic() - started,
        )
        return inner

    async def validate_operations(self) -> frozenset[str]:
        result = await self._tool(
            "telegram_search_tools",
            {"query": "transcribe voice", "kind": "any", "limit": 8},
        )
        operations = {
            match.get("operation")
            for match in result.get("matches", [])
            if isinstance(match, dict)
        }
        required = {"find_recent_outgoing_voice", "transcribe_voice_message"}
        if not required.issubset(operations):
            raise GatewayError("Telegram gateway transcription extension is unavailable")
        accounts = result.get("accounts")
        if not isinstance(accounts, dict) or not accounts:
            raise GatewayError("Telegram gateway has no configured account")
        labels = frozenset(
            str(label).strip() for label in accounts if str(label).strip()
        )
        if len(labels) != len(accounts):
            raise GatewayError("Telegram gateway account catalog is invalid")
        return labels

    async def read(
        self, account: str, operation: str, arguments: dict[str, Any]
    ) -> dict[str, Any]:
        return await self._tool(
            "telegram_read",
            {"account": account, "operation": operation, "arguments": arguments},
        )

    async def write(
        self,
        account: str,
        operation: str,
        arguments: dict[str, Any],
        *,
        request_id: str,
        timeout: int,
    ) -> dict[str, Any]:
        return await self._tool(
            "telegram_write",
            {
                "account": account,
                "operation": operation,
                "arguments": arguments,
                "confirmed": True,
                "request_id": request_id,
            },
            timeout=timeout,
        )


@asynccontextmanager
async def open_gateway(
    launcher: Path, *, default_timeout: int
) -> AsyncIterator[TelegramGateway]:
    started = time.monotonic()
    LOGGER.info("Telegram gateway lifecycle; status=starting")
    launcher = Path(launcher).expanduser().resolve()
    if not launcher.is_file() or not os.access(launcher, os.X_OK):
        LOGGER.error(
            "Telegram gateway lifecycle; status=unavailable elapsed=%.3fs",
            time.monotonic() - started,
        )
        raise GatewayError("Telegram gateway launcher is unavailable")
    parameters = StdioServerParameters(
        # Execute the launcher directly so its shebang selects the appropriate
        # interpreter on macOS or Linux.
        command=str(launcher),
        args=[],
        cwd=launcher.parents[1],
    )
    stack = AsyncExitStack()
    try:
        reader, writer = await stack.enter_async_context(stdio_client(parameters))
        session = await stack.enter_async_context(ClientSession(reader, writer))
        try:
            await session.initialize()
        except Exception as exc:
            if _is_connection_failure(exc):
                raise GatewayConnectionError(_CONNECTION_ERROR_MESSAGE) from None
            raise
    except Exception as exc:
        if _is_connection_failure(exc):
            try:
                await stack.aclose()
            finally:
                LOGGER.error(
                    "Telegram gateway lifecycle; status=connection_error "
                    "elapsed=%.3fs error_type=%s",
                    time.monotonic() - started,
                    type(exc).__name__,
                )
                raise GatewayConnectionError(_CONNECTION_ERROR_MESSAGE) from None
        await stack.aclose()
        LOGGER.error(
            "Telegram gateway lifecycle; status=startup_error elapsed=%.3fs "
            "error_type=%s",
            time.monotonic() - started,
            type(exc).__name__,
        )
        raise

    LOGGER.info(
        "Telegram gateway lifecycle; status=ready elapsed=%.3fs",
        time.monotonic() - started,
    )
    try:
        yield TelegramGateway(session, default_timeout)
    finally:
        close_started = time.monotonic()
        try:
            await stack.aclose()
        except Exception as exc:
            LOGGER.error(
                "Telegram gateway lifecycle; status=close_error elapsed=%.3fs "
                "error_type=%s",
                time.monotonic() - close_started,
                type(exc).__name__,
            )
            raise
        else:
            LOGGER.info(
                "Telegram gateway lifecycle; status=closed elapsed=%.3fs",
                time.monotonic() - close_started,
            )

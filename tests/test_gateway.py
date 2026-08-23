import asyncio
from contextlib import asynccontextmanager
from http import HTTPStatus
from types import SimpleNamespace

import anyio
import pytest
from mcp import McpError
from mcp.types import CONNECTION_CLOSED, ErrorData

import tg_voice_transcriber_bot.gateway as gateway_module
from tg_voice_transcriber_bot.gateway import (
    GatewayConnectionError,
    GatewayError,
    TelegramGateway,
    open_gateway,
)


def test_gateway_launcher_is_executed_through_its_shebang(tmp_path, monkeypatch):
    launcher = tmp_path / "runtime" / "scripts" / "telegram-gateway"
    launcher.parent.mkdir(parents=True)
    launcher.write_text("#!/usr/bin/env python3\n", encoding="utf-8")
    launcher.chmod(0o755)
    captured = {}

    @asynccontextmanager
    async def fake_stdio_client(parameters):
        captured["parameters"] = parameters
        yield object(), object()

    class FakeSession:
        def __init__(self, reader, writer):
            assert reader is not None
            assert writer is not None

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, traceback):
            return False

        async def initialize(self):
            captured["initialized"] = True

    monkeypatch.setattr(gateway_module, "stdio_client", fake_stdio_client)
    monkeypatch.setattr(gateway_module, "ClientSession", FakeSession)

    async def scenario():
        async with open_gateway(launcher, default_timeout=30):
            pass

    asyncio.run(scenario())

    parameters = captured["parameters"]
    assert parameters.command == str(launcher.resolve())
    assert parameters.args == []
    assert parameters.cwd == launcher.resolve().parents[1]
    assert captured["initialized"] is True


def test_gateway_launcher_must_be_executable(tmp_path):
    launcher = tmp_path / "telegram-gateway"
    launcher.write_text("#!/bin/sh\n", encoding="utf-8")
    launcher.chmod(0o600)

    async def scenario():
        async with open_gateway(launcher, default_timeout=30):
            pass

    try:
        asyncio.run(scenario())
    except GatewayError as exc:
        assert "unavailable" in str(exc)
    else:
        raise AssertionError("non-executable gateway launcher was accepted")


@pytest.mark.parametrize(
    "transport_error",
    [
        McpError(
            ErrorData(
                code=HTTPStatus.REQUEST_TIMEOUT,
                message="private timeout details",
            )
        ),
        McpError(
            ErrorData(
                code=CONNECTION_CLOSED,
                message="private closed-transport details",
            )
        ),
        anyio.BrokenResourceError("private stdio details"),
    ],
)
def test_gateway_wraps_transport_failures_without_leaking_details(transport_error):
    class FailingSession:
        async def call_tool(self, *_args, **_kwargs):
            raise transport_error

    gateway = TelegramGateway(FailingSession(), default_timeout=30)

    async def scenario():
        await gateway.read("personal", "find_recent_outgoing_voice", {})

    with pytest.raises(GatewayConnectionError) as captured:
        asyncio.run(scenario())

    assert str(captured.value) == "Telegram gateway connection is unavailable"
    assert "private" not in str(captured.value)


def test_gateway_keeps_tool_errors_retryable():
    class ToolErrorSession:
        async def call_tool(self, *_args, **_kwargs):
            return SimpleNamespace(isError=True, structuredContent={})

    gateway = TelegramGateway(ToolErrorSession(), default_timeout=30)

    async def scenario():
        await gateway.read("personal", "find_recent_outgoing_voice", {})

    with pytest.raises(GatewayError) as captured:
        asyncio.run(scenario())

    assert not isinstance(captured.value, GatewayConnectionError)


def test_gateway_validation_returns_only_configured_account_labels():
    class CatalogSession:
        async def call_tool(self, name, arguments, **_kwargs):
            assert name == "telegram_search_tools"
            assert arguments["query"] == "transcribe voice"
            return SimpleNamespace(
                isError=False,
                structuredContent={
                    "accounts": {"personal": "Личный"},
                    "matches": [
                        {"operation": "find_recent_outgoing_voice"},
                        {"operation": "transcribe_voice_message"},
                    ],
                },
            )

    gateway = TelegramGateway(CatalogSession(), default_timeout=30)

    assert asyncio.run(gateway.validate_operations()) == frozenset({"personal"})


def test_gateway_validation_rejects_empty_account_catalog():
    class EmptyCatalogSession:
        async def call_tool(self, *_args, **_kwargs):
            return SimpleNamespace(
                isError=False,
                structuredContent={
                    "accounts": {},
                    "matches": [
                        {"operation": "find_recent_outgoing_voice"},
                        {"operation": "transcribe_voice_message"},
                    ],
                },
            )

    gateway = TelegramGateway(EmptyCatalogSession(), default_timeout=30)

    with pytest.raises(GatewayError, match="no configured account"):
        asyncio.run(gateway.validate_operations())


def test_gateway_wraps_stdio_start_failure(tmp_path, monkeypatch):
    launcher = tmp_path / "runtime" / "scripts" / "telegram-gateway"
    launcher.parent.mkdir(parents=True)
    launcher.write_text("#!/usr/bin/env python3\n", encoding="utf-8")
    launcher.chmod(0o755)

    @asynccontextmanager
    async def failing_stdio_client(_parameters):
        raise OSError("private executable details")
        yield  # pragma: no cover

    monkeypatch.setattr(gateway_module, "stdio_client", failing_stdio_client)

    async def scenario():
        async with open_gateway(launcher, default_timeout=30):
            pass

    with pytest.raises(GatewayConnectionError) as captured:
        asyncio.run(scenario())

    assert str(captured.value) == "Telegram gateway connection is unavailable"
    assert "private" not in str(captured.value)

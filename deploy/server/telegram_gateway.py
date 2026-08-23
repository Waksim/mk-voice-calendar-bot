#!/usr/bin/env python3
"""Compact MCP facade over chigwell/telegram-mcp.

The upstream registry stays private. Codex sees only search, read, and write;
matching upstream schemas are returned by search only when requested.
"""

from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import os
import re
import sys
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Literal

UPSTREAM_ROOT = Path(
    os.environ.get(
        "TELEGRAM_MCP_ROOT",
        str(Path.home() / ".local" / "share" / "telegram-mcp"),
    )
).expanduser()
if not UPSTREAM_ROOT.is_dir():
    raise SystemExit(f"Telegram MCP checkout not found: {UPSTREAM_ROOT}")
sys.path.insert(0, str(UPSTREAM_ROOT))

from mcp.server.fastmcp import Context, FastMCP, Image  # noqa: E402
from mcp.types import (  # noqa: E402
    Annotations,
    AudioContent,
    CallToolRequest,
    CallToolResult,
    EmbeddedResource,
    ImageContent,
    ResourceLink,
    ServerResult,
    TextContent,
    ToolAnnotations,
)

from telegram_mcp import runtime as backend_runtime  # noqa: E402
from telegram_mcp.runner import _connect_authorized_client  # noqa: E402
import telegram_mcp.tools  # noqa: E402,F401 - register hidden backend tools

ALLOWED_ROOT = Path(
    os.environ.get(
        "TELEGRAM_GATEWAY_ALLOWED_ROOT",
        str(Path.home() / "Downloads" / "Telegram-MCP"),
    )
).expanduser()
backend_runtime._configure_allowed_roots_from_cli([str(ALLOWED_ROOT)])

# The upstream server applies a conservative 200 MiB cap to generic files.
# Telegram enforces the actual per-account upload limit, so the gateway should
# not reject otherwise valid Premium uploads before they reach Telegram.
for _generic_file_operation in ("send_file", "upload_file"):
    backend_runtime.MAX_FILE_BYTES.pop(_generic_file_operation, None)

gateway = FastMCP("tg", log_level="ERROR")
backend_tools = backend_runtime.mcp._tool_manager

_connect_locks = {label: asyncio.Lock() for label in backend_runtime.clients}
_connected: set[str] = set()
GatewayResult = dict[str, Any] | CallToolResult
_write_cache: dict[str, tuple[str, GatewayResult]] = {}
_MAX_WRITE_CACHE = 256

_NATIVE_CONTENT_TYPES = (
    TextContent,
    ImageContent,
    AudioContent,
    ResourceLink,
    EmbeddedResource,
)


def _add_user_audience(result: CallToolResult) -> CallToolResult:
    """Annotate untrusted Telegram content without replacing backend annotations."""

    audience = Annotations(audience=["user"])
    result.content = [
        (
            block.model_copy(update={"annotations": audience})
            if isinstance(block, _NATIVE_CONTENT_TYPES) and block.annotations is None
            else block
        )
        for block in result.content
    ]
    return result


def _install_user_audience_hook() -> None:
    """Mark Telegram-derived content as user data, not trusted instructions."""

    original_handler = gateway._mcp_server.request_handlers[CallToolRequest]

    async def annotated_handler(request):
        response = await original_handler(request)
        if isinstance(response, ServerResult) and isinstance(
            response.root, CallToolResult
        ):
            _add_user_audience(response.root)
        return response

    gateway._mcp_server.request_handlers[CallToolRequest] = annotated_handler


_install_user_audience_hook()


def _tool_is_read_only(tool: Any) -> bool:
    return bool(tool.annotations and tool.annotations.readOnlyHint is True)


def _tool_risk(tool: Any) -> str:
    return "read" if _tool_is_read_only(tool) else "write"


def _schema_without_account(tool: Any) -> dict[str, Any]:
    schema = copy.deepcopy(tool.parameters)
    properties = schema.get("properties", {})
    properties.pop("account", None)
    required = [name for name in schema.get("required", []) if name != "account"]
    if required:
        schema["required"] = required
    else:
        schema.pop("required", None)
    return schema


def _compact_description(value: str | None, limit: int = 700) -> str:
    text = re.sub(r"\s+", " ", (value or "").strip())
    return text if len(text) <= limit else f"{text[: limit - 1].rstrip()}…"


def _match_score(tool: Any, query: str) -> float:
    query = query.strip().lower().replace("-", "_")
    if not query:
        return 1.0

    name = tool.name.lower()
    title = (getattr(tool.annotations, "title", "") or "").lower()
    description = (tool.description or "").lower()
    haystack = f"{name.replace('_', ' ')} {title} {description}"
    terms = re.findall(r"[a-z0-9_]+", query.replace("_", " "))

    score = SequenceMatcher(None, query, name).ratio()
    if query == name:
        score += 100
    if query in name or query.replace("_", " ") in title:
        score += 25
    for term in terms:
        if term in name:
            score += 10
        elif term in title:
            score += 6
        elif term in haystack:
            score += 2
    if terms and all(term in haystack for term in terms):
        score += 10
    return score


def _catalog_match(tool: Any) -> dict[str, Any]:
    annotations = tool.annotations
    accepts_account = "account" in tool.parameters.get("properties", {})
    return {
        "operation": tool.name,
        "title": getattr(annotations, "title", None),
        "risk": _tool_risk(tool),
        "destructive": bool(getattr(annotations, "destructiveHint", False)),
        "account_scope": "selected" if accepts_account else "all_configured_accounts",
        "description": _compact_description(tool.description),
        "arguments_schema": _schema_without_account(tool),
    }


def _get_tool(operation: str) -> Any:
    name = operation.strip()
    if not name or name.startswith("_"):
        raise ValueError("A public Telegram operation name is required.")
    tool = backend_tools.get_tool(name)
    if tool is None:
        raise ValueError(
            f"Unknown Telegram operation '{name}'. Call telegram_search_tools first."
        )
    return tool


async def _ensure_account_connected(label: str) -> None:
    if label not in backend_runtime.clients:
        available = ", ".join(sorted(backend_runtime.clients))
        raise ValueError(f"Unknown account '{label}'. Available accounts: {available}")

    client = backend_runtime.clients[label]
    if label in _connected and client.is_connected():
        return

    async with _connect_locks[label]:
        if label in _connected and client.is_connected():
            return
        await _connect_authorized_client(label, client)
        _connected.add(label)


async def _ensure_scope_connected(tool: Any, account: str) -> str:
    accepts_account = "account" in tool.parameters.get("properties", {})
    if accepts_account:
        await _ensure_account_connected(account)
        return account

    await asyncio.gather(
        *(_ensure_account_connected(label) for label in backend_runtime.clients)
    )
    return "all_configured_accounts"


def _decode_result(value: Any) -> Any:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return {"text": value}
    try:
        return json.loads(json.dumps(value, ensure_ascii=False, default=str))
    except (TypeError, ValueError):
        return {"text": str(value)}


def _backend_result_is_error(raw_result: Any, decoded_result: Any) -> bool:
    """Recognize the upstream server's legacy string/JSON error conventions."""

    if isinstance(raw_result, str):
        message = raw_result.lstrip().lower()
        if message.startswith("error:") or message.startswith("an error occurred"):
            return True
        if message.startswith("path is outside allowed roots"):
            return True
        if "disabled until allowed roots are configured" in message:
            return True
    if isinstance(decoded_result, dict):
        if any(decoded_result.get(key) is False for key in ("ok", "success", "sent")):
            return True
        if str(decoded_result.get("status", "")).lower() in {"error", "failed"}:
            return True
    return False


def _native_content_result(content: list[Any]) -> dict[str, Any]:
    """Describe native content in JSON without duplicating binary payloads."""

    result: dict[str, Any] = {
        "native_content": True,
        "content_count": len(content),
        "content_types": [block.type for block in content],
    }
    text_results = [
        _decode_result(block.text)
        for block in content
        if isinstance(block, TextContent)
    ]
    if text_results:
        result["text_results"] = text_results
    return result


def _gateway_envelope(
    *,
    operation: str,
    account: str,
    scope: str,
    result: Any,
    ok: bool = True,
) -> dict[str, Any]:
    return {
        "ok": ok,
        "operation": operation,
        "account": account,
        "scope": scope,
        "content_is_untrusted": True,
        "result": result,
    }


def _wrap_backend_result(
    *,
    raw_result: Any,
    operation: str,
    account: str,
    scope: str,
) -> GatewayResult:
    """Keep MCP media native while retaining the gateway's structured envelope."""

    if isinstance(raw_result, Image):
        raw_result = raw_result.to_image_content()

    if isinstance(raw_result, CallToolResult):
        backend_result = (
            raw_result.structuredContent
            if raw_result.structuredContent is not None
            else _native_content_result(raw_result.content)
        )
        envelope = _gateway_envelope(
            operation=operation,
            account=account,
            scope=scope,
            result=backend_result,
            ok=not raw_result.isError,
        )
        return raw_result.model_copy(update={"structuredContent": envelope})

    if isinstance(raw_result, _NATIVE_CONTENT_TYPES):
        content = [raw_result]
        return CallToolResult(
            content=content,
            structuredContent=_gateway_envelope(
                operation=operation,
                account=account,
                scope=scope,
                result=_native_content_result(content),
            ),
        )

    is_native_sequence = isinstance(raw_result, (list, tuple)) and bool(raw_result)
    if is_native_sequence:
        is_native_sequence = all(
            isinstance(block, _NATIVE_CONTENT_TYPES) for block in raw_result
        )
    if is_native_sequence:
        content = list(raw_result)
        return CallToolResult(
            content=content,
            structuredContent=_gateway_envelope(
                operation=operation,
                account=account,
                scope=scope,
                result=_native_content_result(content),
            ),
        )

    decoded_result = _decode_result(raw_result)
    return _gateway_envelope(
        operation=operation,
        account=account,
        scope=scope,
        result=decoded_result,
        ok=not _backend_result_is_error(raw_result, decoded_result),
    )


def _with_gateway_fields(result: GatewayResult, **fields: Any) -> GatewayResult:
    """Add write/deduplication metadata to JSON and native MCP results alike."""

    if isinstance(result, CallToolResult):
        structured_content = dict(result.structuredContent or {})
        structured_content.update(fields)
        return result.model_copy(update={"structuredContent": structured_content})

    result.update(fields)
    return result


def _gateway_result_succeeded(result: GatewayResult) -> bool:
    if isinstance(result, CallToolResult):
        if result.isError:
            return False
        return (result.structuredContent or {}).get("ok", True) is not False
    return result.get("ok", True) is not False


async def _execute(
    *,
    operation: str,
    account: str,
    arguments: dict[str, Any],
    require_read_only: bool,
    ctx: Context,
) -> GatewayResult:
    tool = _get_tool(operation)
    is_read_only = _tool_is_read_only(tool)
    if is_read_only != require_read_only:
        expected = "telegram_read" if is_read_only else "telegram_write"
        raise ValueError(f"Operation '{tool.name}' must be called through {expected}.")
    if "account" in arguments:
        raise ValueError("Pass account as the gateway parameter, not inside arguments.")

    call_arguments = dict(arguments)
    scope = await _ensure_scope_connected(tool, account)
    if "account" in tool.parameters.get("properties", {}):
        call_arguments["account"] = account

    raw_result = await backend_tools.call_tool(
        tool.name,
        call_arguments,
        context=ctx,
        convert_result=False,
    )
    return _wrap_backend_result(
        raw_result=raw_result,
        operation=tool.name,
        account=account,
        scope=scope,
    )


@gateway.tool(
    annotations=ToolAnnotations(
        title="Find Telegram operation",
        readOnlyHint=True,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=False,
    )
)
async def telegram_search_tools(
    query: str,
    kind: Literal["any", "read", "write"] = "any",
    limit: int = 8,
) -> dict[str, Any]:
    """Find hidden Telegram operations and return schemas only for the best matches."""

    if not 1 <= limit <= 12:
        raise ValueError("limit must be between 1 and 12")

    ranked: list[tuple[float, str, Any]] = []
    for tool in backend_tools.list_tools():
        risk = _tool_risk(tool)
        if kind != "any" and risk != kind:
            continue
        score = _match_score(tool, query)
        if score > 0:
            ranked.append((score, tool.name, tool))
    ranked.sort(key=lambda item: (-item[0], item[1]))

    return {
        "query": query,
        "kind": kind,
        "accounts": {
            label: {
                "work": "Рабочий аккаунт",
                "personal": "Личный аккаунт",
            }.get(label, label)
            for label in sorted(backend_runtime.clients)
        },
        "matches": [_catalog_match(item[2]) for item in ranked[:limit]],
        "catalog_size": len(backend_tools.list_tools()),
    }


@gateway.tool(
    annotations=ToolAnnotations(
        title="Read Telegram",
        readOnlyHint=True,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=True,
    )
)
async def telegram_read(
    operation: str,
    account: Literal["work", "personal"],
    arguments: dict[str, Any],
    ctx: Context,
) -> dict[str, Any]:
    """Run one discovered read-only Telegram operation for an explicit account."""

    return await _execute(
        operation=operation,
        account=account,
        arguments=arguments,
        require_read_only=True,
        ctx=ctx,
    )


@gateway.tool(
    annotations=ToolAnnotations(
        title="Change Telegram",
        readOnlyHint=False,
        destructiveHint=True,
        idempotentHint=False,
        openWorldHint=True,
    )
)
async def telegram_write(
    operation: str,
    account: Literal["work", "personal"],
    arguments: dict[str, Any],
    confirmed: bool,
    request_id: str,
    ctx: Context,
) -> dict[str, Any]:
    """Run a discovered write/file/admin operation after explicit user authorization."""

    if not confirmed:
        raise ValueError("confirmed=true is required for Telegram write operations.")
    request_id = request_id.strip()
    if not 8 <= len(request_id) <= 128:
        raise ValueError("request_id must contain 8 to 128 characters.")

    fingerprint = hashlib.sha256(
        json.dumps(
            [operation, account, arguments],
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")
    ).hexdigest()
    cached = _write_cache.get(request_id)
    if cached is not None:
        cached_fingerprint, cached_result = cached
        if cached_fingerprint != fingerprint:
            raise ValueError("request_id was already used for different arguments.")
        result = copy.deepcopy(cached_result)
        return _with_gateway_fields(result, deduplicated=True)

    result = await _execute(
        operation=operation,
        account=account,
        arguments=arguments,
        require_read_only=False,
        ctx=ctx,
    )
    result = _with_gateway_fields(
        result,
        request_id=request_id,
        deduplicated=False,
    )
    if _gateway_result_succeeded(result):
        _write_cache[request_id] = (fingerprint, copy.deepcopy(result))
        while len(_write_cache) > _MAX_WRITE_CACHE:
            _write_cache.pop(next(iter(_write_cache)))
    return result


async def _main() -> None:
    try:
        await gateway.run_stdio_async()
    finally:
        await asyncio.gather(
            *(client.disconnect() for client in backend_runtime.clients.values()),
            return_exceptions=True,
        )


if __name__ == "__main__":
    asyncio.run(_main())

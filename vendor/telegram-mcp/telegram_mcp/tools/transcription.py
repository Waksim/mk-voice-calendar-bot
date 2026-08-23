"""Server-side Telegram voice transcription helpers.

These operations intentionally work with messages that are already visible to
an authorized *user* session.  No audio bytes are downloaded: Telegram's
``messages.transcribeAudio`` RPC performs the speech recognition.
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from typing import Optional, Union

from mcp.types import ToolAnnotations
from telethon import events, functions, types, utils
from telethon.errors import RPCError

from telegram_mcp.runtime import get_client, mcp, resolve_entity, with_account


def _voice_duration(message) -> Optional[int]:
    document = getattr(message, "document", None)
    if document is None:
        return None
    for attribute in getattr(document, "attributes", ()) or ():
        if isinstance(attribute, types.DocumentAttributeAudio) and attribute.voice:
            return int(round(attribute.duration))
    return None


def _iso_or_none(value: Optional[datetime]) -> Optional[str]:
    return value.isoformat() if value is not None else None


@mcp.tool(
    annotations=ToolAnnotations(
        title="Find Recent Outgoing Voice",
        openWorldHint=True,
        readOnlyHint=True,
        destructiveHint=False,
        idempotentHint=True,
    )
)
@with_account(readonly=True)
async def find_recent_outgoing_voice(
    chat_id: Union[int, str],
    sent_at: int,
    duration: int,
    file_size: Optional[int] = None,
    after_message_id: int = 0,
    search_limit: int = 50,
    tolerance_seconds: int = 5,
    account: str = None,
) -> str:
    """Find the user-side ID of a recently sent voice note in a bot dialog.

    Args:
        chat_id: Bot username or Telegram peer ID visible to the user session.
        sent_at: Bot API message timestamp, as Unix seconds.
        duration: Bot API voice duration, in whole seconds.
        file_size: Optional Bot API voice file size; exact when supplied.
        after_message_id: Ignore already processed user-side message IDs.
        search_limit: Number of recent messages to inspect (1..200).
        tolerance_seconds: Allowed timestamp difference (1..30 seconds).
    """
    if not 1 <= search_limit <= 200:
        raise ValueError("search_limit must be between 1 and 200")
    if not 1 <= tolerance_seconds <= 30:
        raise ValueError("tolerance_seconds must be between 1 and 30")
    if sent_at <= 0:
        raise ValueError("sent_at must be a positive Unix timestamp")
    if duration < 0:
        raise ValueError("duration must be non-negative")
    if file_size is not None and file_size < 0:
        raise ValueError("file_size must be non-negative")

    client = get_client(account)
    entity = await resolve_entity(chat_id, client)
    if not getattr(entity, "bot", False):
        raise ValueError("chat_id must resolve to a Telegram bot")
    offset_date = datetime.fromtimestamp(
        sent_at + tolerance_seconds + 1, tz=timezone.utc
    )
    messages = await client.get_messages(
        entity,
        limit=search_limit,
        offset_date=offset_date,
        min_id=after_message_id,
        filter=types.InputMessagesFilterVoice(),
    )

    matches: list[tuple[tuple[int, int, int], dict]] = []
    for message in messages:
        message_id = int(getattr(message, "id", 0) or 0)
        if message_id <= after_message_id or not getattr(message, "out", False):
            continue
        if getattr(message, "voice", None) is None:
            continue

        message_date = getattr(message, "date", None)
        if message_date is None:
            continue
        timestamp = int(message_date.timestamp())
        time_delta = abs(timestamp - sent_at)
        if time_delta > tolerance_seconds:
            continue

        voice_duration = _voice_duration(message)
        if voice_duration is None:
            continue
        duration_delta = abs(voice_duration - duration)
        if duration_delta > 1:
            continue

        document = getattr(message, "document", None)
        voice_size = int(getattr(document, "size", 0) or 0)
        if file_size not in (None, 0) and voice_size != file_size:
            continue

        metadata = {
            "message_id": message_id,
            "sent_at": timestamp,
            "duration": voice_duration,
            "file_size": voice_size,
        }
        # Bot API updates are processed in order, so identical notes sent in the
        # same second must map to the earliest still-unprocessed user message.
        matches.append(((time_delta, duration_delta, message_id), metadata))

    matches.sort(key=lambda item: item[0])
    if not matches:
        return json.dumps(
            {
                "ok": False,
                "status": "not_found",
                "after_message_id": after_message_id,
            },
            ensure_ascii=False,
        )

    return json.dumps(
        {
            "ok": True,
            "status": "found",
            "match": matches[0][1],
            "candidate_count": len(matches),
        },
        ensure_ascii=False,
    )


@mcp.tool(
    annotations=ToolAnnotations(
        title="Transcribe Voice on Telegram Servers",
        openWorldHint=True,
        readOnlyHint=False,
        destructiveHint=False,
        idempotentHint=False,
    )
)
@with_account(readonly=False)
async def transcribe_voice_message(
    chat_id: Union[int, str],
    message_id: int,
    wait_timeout: int = 180,
    account: str = None,
) -> str:
    """Transcribe an existing voice note through Telegram's user-only RPC.

    The call consumes the selected user's Telegram transcription entitlement.
    It never downloads the voice file.  If Telegram initially returns a pending
    result, the operation waits for ``updateTranscribedAudio`` and periodically
    refreshes the RPC in case another connection consumed the raw update.
    """
    if message_id <= 0:
        raise ValueError("message_id must be positive")
    if not 5 <= wait_timeout <= 300:
        raise ValueError("wait_timeout must be between 5 and 300 seconds")

    client = get_client(account)
    entity = await resolve_entity(chat_id, client)
    if not getattr(entity, "bot", False):
        raise ValueError("chat_id must resolve to a Telegram bot")
    message = await client.get_messages(entity, ids=message_id)
    if (
        message is None
        or not getattr(message, "out", False)
        or getattr(message, "voice", None) is None
    ):
        return json.dumps(
            {
                "ok": False,
                "status": "invalid_message",
                "message_id": message_id,
                "error": "MSG_VOICE_MISSING",
            },
            ensure_ascii=False,
        )

    expected_peer_id = utils.get_peer_id(entity)
    update_queue: asyncio.Queue[types.UpdateTranscribedAudio] = asyncio.Queue()

    async def on_transcription_update(update: types.UpdateTranscribedAudio) -> None:
        update_peer_id = utils.get_peer_id(getattr(update, "peer", None))
        if (
            int(getattr(update, "msg_id", 0) or 0) == message_id
            and update_peer_id == expected_peer_id
        ):
            update_queue.put_nowait(update)

    event_builder = events.Raw(types=types.UpdateTranscribedAudio)
    client.add_event_handler(on_transcription_update, event_builder)
    request = functions.messages.TranscribeAudioRequest(peer=entity, msg_id=message_id)

    try:
        response = await client(request)
        transcription_id = int(response.transcription_id)
        latest_text = response.text or ""
        trial_remains_num = getattr(response, "trial_remains_num", None)
        trial_remains_until_date = getattr(response, "trial_remains_until_date", None)

        if not getattr(response, "pending", False):
            return json.dumps(
                {
                    "ok": True,
                    "status": "completed",
                    "message_id": message_id,
                    "transcription_id": str(transcription_id),
                    "text": latest_text,
                    "trial_remains_num": trial_remains_num,
                    "trial_remains_until_date": _iso_or_none(trial_remains_until_date),
                },
                ensure_ascii=False,
            )

        loop = asyncio.get_running_loop()
        deadline = loop.time() + wait_timeout
        while loop.time() < deadline:
            remaining = deadline - loop.time()
            try:
                update = await asyncio.wait_for(update_queue.get(), timeout=min(10, remaining))
            except asyncio.TimeoutError:
                # Raw updates can be consumed by another process sharing the same
                # authorization key. Repeating the same RPC reads Telegram's current
                # cached state without downloading the media.
                response = await client(request)
                if int(response.transcription_id) != transcription_id:
                    transcription_id = int(response.transcription_id)
                latest_text = response.text or latest_text
                trial_remains_num = getattr(response, "trial_remains_num", trial_remains_num)
                trial_remains_until_date = getattr(
                    response, "trial_remains_until_date", trial_remains_until_date
                )
                if not getattr(response, "pending", False):
                    return json.dumps(
                        {
                            "ok": True,
                            "status": "completed",
                            "message_id": message_id,
                            "transcription_id": str(transcription_id),
                            "text": latest_text,
                            "trial_remains_num": trial_remains_num,
                            "trial_remains_until_date": _iso_or_none(
                                trial_remains_until_date
                            ),
                        },
                        ensure_ascii=False,
                    )
                continue

            if int(update.transcription_id) != transcription_id:
                continue
            latest_text = update.text or latest_text
            if not getattr(update, "pending", False):
                return json.dumps(
                    {
                        "ok": True,
                        "status": "completed",
                        "message_id": message_id,
                        "transcription_id": str(transcription_id),
                        "text": latest_text,
                        "trial_remains_num": trial_remains_num,
                        "trial_remains_until_date": _iso_or_none(
                            trial_remains_until_date
                        ),
                    },
                    ensure_ascii=False,
                )

        return json.dumps(
            {
                "ok": False,
                "status": "timeout",
                "message_id": message_id,
                "transcription_id": str(transcription_id),
                "partial_text": latest_text,
            },
            ensure_ascii=False,
        )
    except RPCError as exc:
        rpc_message = str(getattr(exc, "message", "") or "").upper()
        known_errors = {
            "PREMIUM_ACCOUNT_REQUIRED",
            "MSG_VOICE_TOO_LONG",
            "MSG_VOICE_MISSING",
            "MSG_ID_INVALID",
            "PEER_ID_INVALID",
            "TRANSCRIPTION_FAILED",
        }
        error = rpc_message if rpc_message in known_errors else type(exc).__name__
        return json.dumps(
            {
                "ok": False,
                "status": "telegram_error",
                "message_id": message_id,
                "error": error,
                "error_code": getattr(exc, "code", None),
                "retry_after": getattr(exc, "seconds", None),
            },
            ensure_ascii=False,
        )
    finally:
        client.remove_event_handler(on_transcription_update, event_builder)

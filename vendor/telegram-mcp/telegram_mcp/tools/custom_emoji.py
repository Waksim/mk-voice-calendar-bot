"""Telegram custom (Premium) emoji discovery, inspection, and actions."""

from html import escape
import random

from mcp.server.fastmcp import Image
from telethon.errors import FileReferenceExpiredError

from telegram_mcp.runtime import *

_MAX_CUSTOM_EMOJI_IDS = 200
_CUSTOM_EMOJI_SET_KINDS = {"installed", "featured"}
_CUSTOM_EMOJI_REACTION_KINDS = {"recent", "top"}
_CUSTOM_EMOJI_GROUP_KINDS = {"general", "stickers", "status", "profile_photo"}
_EMOJI_STATUS_KINDS = {
    "recent",
    "default",
    "collectible",
    "channel_default",
    "channel_restricted",
}


def _coerce_int64(value, *, name: str, positive: bool) -> int:
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped or not stripped.lstrip("-").isdigit():
            raise ValueError(f"{name} must be a decimal integer or decimal string.")
        value = int(stripped)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer or decimal string.")
    if value < -(2**63) or value > 2**63 - 1:
        raise ValueError(f"{name} must fit in a signed 64-bit integer.")
    if positive and value <= 0:
        raise ValueError(f"{name} must be positive.")
    return value


def _validate_custom_emoji_ids(document_ids: List[Union[int, str]]) -> List[int]:
    if not isinstance(document_ids, list) or not document_ids:
        raise ValueError("document_ids must be a non-empty list of integers.")
    if len(document_ids) > _MAX_CUSTOM_EMOJI_IDS:
        raise ValueError(f"document_ids may contain at most {_MAX_CUSTOM_EMOJI_IDS} values.")

    unique = []
    for value in document_ids:
        value = _coerce_int64(value, name="document_id", positive=True)
        if value not in unique:
            unique.append(value)
    return unique


def _custom_emoji_attribute(document):
    return next(
        (
            attribute
            for attribute in (getattr(document, "attributes", None) or [])
            if isinstance(attribute, types.DocumentAttributeCustomEmoji)
        ),
        None,
    )


def _input_sticker_set_payload(stickerset) -> Optional[dict]:
    if isinstance(stickerset, types.InputStickerSetID):
        return {
            "id": stickerset.id,
            "id_str": str(stickerset.id),
            "access_hash": stickerset.access_hash,
            "access_hash_str": str(stickerset.access_hash),
        }
    if isinstance(stickerset, types.InputStickerSetShortName):
        return {"short_name": sanitize_name(stickerset.short_name)}
    if stickerset is None:
        return None
    return {"type": type(stickerset).__name__}


def _custom_emoji_document_payload(document) -> Optional[dict]:
    if not isinstance(document, types.Document):
        return None
    attribute = _custom_emoji_attribute(document)
    if attribute is None:
        return None

    fallback = getattr(attribute, "alt", "") or "🙂"
    if not isinstance(fallback, str):
        fallback = "🙂"
    payload = {
        "document_id": document.id,
        "document_id_str": str(document.id),
        "access_hash": document.access_hash,
        "access_hash_str": str(document.access_hash),
        "fallback": fallback,
        "html": (f'<tg-emoji emoji-id="{document.id}">{escape(fallback)}</tg-emoji>'),
        "mime_type": document.mime_type,
        "size": document.size,
        "dc_id": document.dc_id,
        "free": bool(getattr(attribute, "free", False)),
        "text_color": bool(getattr(attribute, "text_color", False)),
    }
    if document.date is not None:
        payload["date"] = document.date.isoformat()
    stickerset = _input_sticker_set_payload(getattr(attribute, "stickerset", None))
    if stickerset:
        payload["sticker_set"] = stickerset

    for item in document.attributes:
        if isinstance(item, types.DocumentAttributeFilename):
            payload["file_name"] = sanitize_name(item.file_name)
        elif isinstance(item, types.DocumentAttributeAnimated):
            payload["animated"] = True
        elif isinstance(item, types.DocumentAttributeVideo):
            payload["video"] = {
                "duration": item.duration,
                "width": item.w,
                "height": item.h,
            }

    thumbs = []
    for thumb in getattr(document, "thumbs", None) or []:
        record = {"type": type(thumb).__name__}
        for source, target in (
            ("type", "label"),
            ("w", "width"),
            ("h", "height"),
            ("size", "size"),
        ):
            value = getattr(thumb, source, None)
            if value is not None:
                record[target] = value
        thumbs.append(record)
    if thumbs:
        payload["thumbnails"] = thumbs
    if getattr(document, "video_thumbs", None):
        payload["has_video_thumbnail"] = True
    return payload


def _custom_emoji_documents_payload(documents) -> List[dict]:
    payload = []
    for document in documents or []:
        record = _custom_emoji_document_payload(document)
        if record is not None:
            payload.append(record)
    return payload


def _largest_static_thumbnail(document):
    candidates = [
        thumb
        for thumb in (getattr(document, "thumbs", None) or [])
        if isinstance(
            thumb,
            (types.PhotoSize, types.PhotoCachedSize, types.PhotoSizeProgressive),
        )
    ]
    if not candidates:
        raise ValueError("this custom emoji has no static preview thumbnail.")
    return max(
        candidates,
        key=lambda thumb: (getattr(thumb, "w", 0) * getattr(thumb, "h", 0)),
    )


def _preview_image_format(data: bytes) -> str:
    if data.startswith(b"\xff\xd8\xff"):
        return "jpeg"
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "png"
    if data.startswith(b"RIFF") and data[8:12] == b"WEBP":
        return "webp"
    raise ValueError("Telegram returned an unsupported preview image format.")


async def _download_custom_emoji_media(
    cl,
    *,
    document_id: int,
    document,
    file,
    preview: bool,
):
    """Download custom-emoji media once, refreshing an expired file reference if needed."""

    async def download(current_document):
        kwargs = {"thumb": _largest_static_thumbnail(current_document)} if preview else {}
        return await cl.download_media(current_document, file=file, **kwargs)

    try:
        return await download(document), document
    except FileReferenceExpiredError:
        _requested, refreshed = await _fetch_custom_emoji_documents(cl, [document_id])
        if not refreshed:
            raise ValueError("custom emoji disappeared while refreshing its file reference.")
        return await download(refreshed[0]), refreshed[0]


async def _fetch_custom_emoji_documents(cl, document_ids: List[Union[int, str]]):
    ids = _validate_custom_emoji_ids(document_ids)
    documents = await cl(functions.messages.GetCustomEmojiDocumentsRequest(document_id=ids))
    valid = [
        document for document in (documents or []) if _custom_emoji_attribute(document) is not None
    ]
    return ids, valid


async def _recent_custom_emoji_ids(cl, limit: int):
    """Return candidate IDs from actual recent reactions and emoji statuses."""
    reactions_result = await cl(functions.messages.GetRecentReactionsRequest(limit=limit, hash=0))
    statuses_result = await cl(functions.account.GetRecentEmojiStatusesRequest(hash=0))

    sources = {}
    for reaction in getattr(reactions_result, "reactions", None) or []:
        if isinstance(reaction, types.ReactionCustomEmoji):
            sources.setdefault(reaction.document_id, []).append("recent_reaction")
    for status in getattr(statuses_result, "statuses", None) or []:
        document_id = getattr(status, "document_id", None)
        if document_id:
            sources.setdefault(document_id, []).append("recent_status")

    return list(sources)[:limit], sources


def _normalise_lang_codes(lang_codes: Optional[List[str]]) -> List[str]:
    if lang_codes is None:
        return ["en"]
    if not isinstance(lang_codes, list) or not lang_codes:
        raise ValueError("lang_codes must be a non-empty list of language codes.")
    values = []
    for value in lang_codes:
        if not isinstance(value, str):
            raise ValueError("lang_codes must contain only strings.")
        code = value.strip().lower()
        if not code or len(code) > 16 or not all(char.isalnum() or char == "-" for char in code):
            raise ValueError(f"invalid language code: {value!r}.")
        if code not in values:
            values.append(code)
    return values


async def _search_custom_emoji_documents(
    cl,
    *,
    query: str,
    emoticon: str,
    lang_codes: Optional[List[str]],
    offset: int,
    limit: int,
):
    query = query.strip()
    emoticon = emoticon.strip()
    if not query and not emoticon:
        raise ValueError("provide query, emoticon, or both.")
    if not isinstance(offset, int) or isinstance(offset, bool) or offset < 0:
        raise ValueError("offset must be a non-negative integer.")
    if not isinstance(limit, int) or isinstance(limit, bool) or not 1 <= limit <= 100:
        raise ValueError("limit must be between 1 and 100.")

    if query:
        result = await cl(
            functions.messages.SearchStickersRequest(
                q=query,
                emoticon=emoticon,
                lang_code=_normalise_lang_codes(lang_codes),
                offset=offset,
                limit=limit,
                hash=0,
                emojis=True,
            )
        )
        return (
            [
                document
                for document in (getattr(result, "stickers", None) or [])
                if _custom_emoji_attribute(document) is not None
            ],
            getattr(result, "next_offset", None),
            "text_search",
        )

    result = await cl(functions.messages.SearchCustomEmojiRequest(emoticon=emoticon, hash=0))
    all_ids = list(getattr(result, "document_id", None) or [])
    page_ids = all_ids[offset : offset + limit]
    if not page_ids:
        return [], None, "emoticon_search"
    _ids, documents = await _fetch_custom_emoji_documents(cl, page_ids)
    next_offset = offset + limit if offset + limit < len(all_ids) else None
    return documents, next_offset, "emoticon_search"


def _sticker_set_payload(value) -> Optional[dict]:
    sticker_set = getattr(value, "set", value)
    if not isinstance(sticker_set, types.StickerSet):
        return None
    payload = {
        "id": sticker_set.id,
        "id_str": str(sticker_set.id),
        "access_hash": sticker_set.access_hash,
        "access_hash_str": str(sticker_set.access_hash),
        "title": sanitize_name(sticker_set.title),
        "short_name": sanitize_name(sticker_set.short_name),
        "count": sticker_set.count,
        "installed": getattr(sticker_set, "installed_date", None) is not None,
        "archived": bool(getattr(sticker_set, "archived", False)),
        "official": bool(getattr(sticker_set, "official", False)),
        "custom_emoji": bool(getattr(sticker_set, "emojis", False)),
        "text_color": bool(getattr(sticker_set, "text_color", False)),
    }
    if getattr(sticker_set, "installed_date", None) is not None:
        payload["installed_date"] = sticker_set.installed_date.isoformat()
    covers = []
    for document in [
        getattr(value, "cover", None),
        *(getattr(value, "covers", None) or []),
    ]:
        record = _custom_emoji_document_payload(document)
        if record is not None:
            covers.append(record)
    if covers:
        payload["covers"] = covers
    return payload


def _sticker_sets_payload(values) -> List[dict]:
    payload = []
    for value in values or []:
        record = _sticker_set_payload(value)
        if record is not None and record["custom_emoji"]:
            payload.append(record)
    return payload


def _resolve_sticker_set_input(
    *,
    short_name: Optional[str],
    set_id: Optional[Union[int, str]],
    access_hash: Optional[Union[int, str]],
):
    if short_name is not None:
        if set_id is not None or access_hash is not None:
            raise ValueError("pass short_name or set_id/access_hash, not both.")
        short_name = short_name.strip()
        if not short_name:
            raise ValueError("short_name must not be empty.")
        return types.InputStickerSetShortName(short_name=short_name)
    if set_id is None or access_hash is None:
        raise ValueError("pass short_name or both set_id and access_hash.")
    set_id = _coerce_int64(set_id, name="set_id", positive=True)
    access_hash = _coerce_int64(access_hash, name="access_hash", positive=False)
    return types.InputStickerSetID(id=set_id, access_hash=access_hash)


async def _get_custom_emoji_set_result(cl, stickerset):
    result = await cl(functions.messages.GetStickerSetRequest(stickerset=stickerset, hash=0))
    set_payload = _sticker_set_payload(getattr(result, "set", None))
    if set_payload is None or not set_payload["custom_emoji"]:
        raise ValueError("the specified sticker set is not a custom-emoji set.")
    return result, set_payload


def _utf16_slice(value: str, offset: int, length: int) -> str:
    data = value.encode("utf-16-le")
    start = max(offset, 0) * 2
    end = max(offset + length, 0) * 2
    return data[start:end].decode("utf-16-le", errors="replace")


def _custom_emoji_entity_occurrences(
    text: str,
    entities,
    *,
    source: str,
) -> List[dict]:
    occurrences = []
    for entity in entities or []:
        if not isinstance(entity, types.MessageEntityCustomEmoji):
            continue
        occurrences.append(
            {
                "source": source,
                "document_id": entity.document_id,
                "document_id_str": str(entity.document_id),
                "offset": entity.offset,
                "length": entity.length,
                "fallback": sanitize_user_content(
                    _utf16_slice(text or "", entity.offset, entity.length)
                ),
            }
        )
    return occurrences


def _emoji_status_payload(status) -> dict:
    if status is None or isinstance(status, types.EmojiStatusEmpty):
        return {"type": "empty"}
    if isinstance(status, types.EmojiStatusCollectible):
        payload = {
            "type": "collectible",
            "collectible_id": status.collectible_id,
            "collectible_id_str": str(status.collectible_id),
            "document_id": status.document_id,
            "document_id_str": str(status.document_id),
            "title": sanitize_name(status.title),
            "slug": sanitize_name(status.slug),
            "pattern_document_id": status.pattern_document_id,
            "pattern_document_id_str": str(status.pattern_document_id),
            "colors": {
                "center": status.center_color,
                "edge": status.edge_color,
                "pattern": status.pattern_color,
                "text": status.text_color,
            },
        }
    else:
        payload = {
            "type": "custom_emoji",
            "document_id": getattr(status, "document_id", None),
        }
        if payload["document_id"] is not None:
            payload["document_id_str"] = str(payload["document_id"])
    if getattr(status, "until", None) is not None:
        payload["until"] = status.until.isoformat()
    return payload


def _reaction_payload(reaction) -> dict:
    if isinstance(reaction, types.ReactionCustomEmoji):
        return {
            "type": "custom_emoji",
            "document_id": reaction.document_id,
            "document_id_str": str(reaction.document_id),
        }
    if isinstance(reaction, types.ReactionEmoji):
        return {
            "type": "emoji",
            "emoticon": sanitize_user_content(reaction.emoticon),
        }
    if isinstance(reaction, types.ReactionPaid):
        return {"type": "paid"}
    return {"type": type(reaction).__name__}


def _reaction_identity(reaction):
    if isinstance(reaction, types.ReactionCustomEmoji):
        return ("custom_emoji", reaction.document_id)
    if isinstance(reaction, types.ReactionEmoji):
        return ("emoji", reaction.emoticon)
    return (type(reaction).__name__, None)


def _telegram_json_value(value):
    if isinstance(value, types.JsonObject):
        return {item.key: _telegram_json_value(item.value) for item in value.value}
    if isinstance(value, types.JsonArray):
        return [_telegram_json_value(item) for item in value.value]
    if isinstance(value, (types.JsonBool, types.JsonNumber, types.JsonString)):
        return value.value
    if isinstance(value, types.JsonNull):
        return None
    return None


async def _reaction_limit(cl, *, premium: bool) -> int:
    """Read Telegram's current per-user reaction-vector limit with safe fallbacks."""
    fallback = 3 if premium else 1
    try:
        result = await cl(functions.help.GetAppConfigRequest(hash=0))
        config = _telegram_json_value(getattr(result, "config", None)) or {}
        key = "reactions_user_max_premium" if premium else "reactions_user_max_default"
        value = config.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool) and value >= 1:
            return int(value)
    except Exception:
        pass
    return fallback


def _emoji_group_payload(group) -> dict:
    icon_emoji_id = getattr(group, "icon_emoji_id", None)
    payload = {
        "type": type(group).__name__,
        "title": sanitize_name(getattr(group, "title", "") or ""),
        "emoticons": [
            sanitize_user_content(value) for value in (getattr(group, "emoticons", None) or [])
        ],
    }
    if icon_emoji_id is not None:
        payload["icon_emoji_id"] = icon_emoji_id
        payload["icon_emoji_id_str"] = str(icon_emoji_id)
    return payload


def _parse_until(value: Optional[Union[str, int]]) -> Optional[datetime]:
    if value is None:
        return None
    try:
        if isinstance(value, bool):
            raise ValueError
        if isinstance(value, int):
            parsed = datetime.fromtimestamp(value, tz=timezone.utc)
        elif isinstance(value, str):
            parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            else:
                parsed = parsed.astimezone(timezone.utc)
        else:
            raise ValueError
    except (ValueError, OverflowError, OSError):
        raise ValueError("until must be an ISO-8601 date or Unix timestamp.") from None
    if parsed <= datetime.now(timezone.utc):
        raise ValueError("until must be in the future.")
    return parsed


async def _require_premium(cl) -> None:
    me = await cl.get_me()
    if not bool(getattr(me, "premium", False)):
        raise ValueError("Telegram Premium is required to send custom emoji.")
    return me


async def _require_custom_emoji_entitlement(cl, document) -> None:
    """Allow Telegram's free custom emoji, otherwise require Premium."""
    attribute = _custom_emoji_attribute(document)
    if attribute is not None and bool(getattr(attribute, "free", False)):
        return
    await _require_premium(cl)


@mcp.tool(
    annotations=ToolAnnotations(title="Search Custom Emoji", openWorldHint=True, readOnlyHint=True)
)
@with_account(readonly=True)
async def search_custom_emojis(
    query: str = "",
    emoticon: str = "",
    lang_codes: Optional[List[str]] = None,
    offset: int = 0,
    limit: int = 50,
    account: str = None,
) -> str:
    """Search Telegram custom emoji by text, base emoji, or both."""
    try:
        cl = get_client(account)
        documents, next_offset, source = await _search_custom_emoji_documents(
            cl,
            query=query,
            emoticon=emoticon,
            lang_codes=lang_codes,
            offset=offset,
            limit=limit,
        )
        results = _custom_emoji_documents_payload(documents)
        return json.dumps(
            {
                "query": query,
                "emoticon": emoticon,
                "source": source,
                "offset": offset,
                "next_offset": next_offset,
                "count": len(results),
                "results": results,
            },
            ensure_ascii=False,
            indent=2,
        )
    except ValueError as e:
        return f"Error: {e}"
    except Exception as e:
        return log_and_format_error(
            "search_custom_emojis", e, query=query, emoticon=emoticon, offset=offset, limit=limit
        )


@mcp.tool(
    annotations=ToolAnnotations(
        title="Get Custom Emoji Documents", openWorldHint=True, readOnlyHint=True
    )
)
@with_account(readonly=True)
async def get_custom_emoji_documents(
    document_ids: List[Union[int, str]], account: str = None
) -> str:
    """Validate custom-emoji document IDs and return reusable metadata and HTML."""
    try:
        cl = get_client(account)
        requested, documents = await _fetch_custom_emoji_documents(cl, document_ids)
        results = _custom_emoji_documents_payload(documents)
        found_ids = {item["document_id"] for item in results}
        return json.dumps(
            {
                "requested_ids": requested,
                "requested_id_strings": [str(value) for value in requested],
                "found": results,
                "missing_or_not_custom_emoji": [
                    value for value in requested if value not in found_ids
                ],
                "missing_or_not_custom_emoji_strings": [
                    str(value) for value in requested if value not in found_ids
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
    except ValueError as e:
        return f"Error: {e}"
    except Exception as e:
        return log_and_format_error("get_custom_emoji_documents", e, document_ids=document_ids)


@mcp.tool(
    annotations=ToolAnnotations(
        title="List Recent Custom Emoji", openWorldHint=True, readOnlyHint=True
    )
)
@with_account(readonly=True)
async def list_recent_custom_emojis(limit: int = 50, account: str = None) -> str:
    """List custom emoji from the account's recent reactions and emoji statuses."""
    try:
        if not isinstance(limit, int) or isinstance(limit, bool) or not 1 <= limit <= 200:
            raise ValueError("limit must be between 1 and 200.")
        cl = get_client(account)
        document_ids, sources = await _recent_custom_emoji_ids(cl, limit)
        documents = []
        if document_ids:
            _requested, documents = await _fetch_custom_emoji_documents(cl, document_ids)
        payload = _custom_emoji_documents_payload(documents)
        for item in payload:
            item["recent_sources"] = sources.get(item["document_id"], [])
        return json.dumps(
            {
                "count": len(payload),
                "sources": ["recent_reactions", "recent_emoji_statuses"],
                "results": payload,
            },
            ensure_ascii=False,
            indent=2,
        )
    except ValueError as e:
        return f"Error: {e}"
    except Exception as e:
        return log_and_format_error("list_recent_custom_emojis", e, limit=limit)


@mcp.tool(
    annotations=ToolAnnotations(
        title="List Custom Emoji Sets", openWorldHint=True, readOnlyHint=True
    )
)
@with_account(readonly=True)
async def list_custom_emoji_sets(
    kind: str = "installed",
    offset: int = 0,
    limit: int = 100,
    account: str = None,
) -> str:
    """List installed or featured custom-emoji sets."""
    try:
        kind = kind.strip().lower()
        if kind not in _CUSTOM_EMOJI_SET_KINDS:
            raise ValueError("kind must be installed or featured.")
        if not isinstance(offset, int) or isinstance(offset, bool) or offset < 0:
            raise ValueError("offset must be a non-negative integer.")
        if not isinstance(limit, int) or isinstance(limit, bool) or not 1 <= limit <= 200:
            raise ValueError("limit must be between 1 and 200.")
        cl = get_client(account)
        if kind == "installed":
            result = await cl(functions.messages.GetEmojiStickersRequest(hash=0))
        else:
            result = await cl(functions.messages.GetFeaturedEmojiStickersRequest(hash=0))
        all_sets = _sticker_sets_payload(getattr(result, "sets", None) or [])
        page = all_sets[offset : offset + limit]
        next_offset = offset + limit if offset + limit < len(all_sets) else None
        return json.dumps(
            {
                "kind": kind,
                "total": getattr(result, "count", None) or len(all_sets),
                "offset": offset,
                "next_offset": next_offset,
                "count": len(page),
                "sets": page,
            },
            ensure_ascii=False,
            indent=2,
        )
    except ValueError as e:
        return f"Error: {e}"
    except Exception as e:
        return log_and_format_error(
            "list_custom_emoji_sets", e, kind=kind, offset=offset, limit=limit
        )


@mcp.tool(
    annotations=ToolAnnotations(
        title="List Archived Custom Emoji Sets", openWorldHint=True, readOnlyHint=True
    )
)
@with_account(readonly=True)
async def list_archived_custom_emoji_sets(
    offset_id: Union[int, str] = 0,
    limit: int = 100,
    account: str = None,
) -> str:
    """List archived custom-emoji sets using Telegram's server-side pagination."""
    try:
        offset_id = _coerce_int64(offset_id, name="offset_id", positive=False)
        if offset_id < 0:
            raise ValueError("offset_id must be a non-negative integer.")
        if not isinstance(limit, int) or isinstance(limit, bool) or not 1 <= limit <= 100:
            raise ValueError("limit must be between 1 and 100.")
        cl = get_client(account)
        result = await cl(
            functions.messages.GetArchivedStickersRequest(
                offset_id=offset_id,
                limit=limit,
                emojis=True,
            )
        )
        raw_sets = list(getattr(result, "sets", None) or [])
        sets = _sticker_sets_payload(raw_sets)
        raw_last_set = getattr(raw_sets[-1], "set", raw_sets[-1]) if raw_sets else None
        next_offset_id = getattr(raw_last_set, "id", None) if len(raw_sets) == limit else None
        return json.dumps(
            {
                "total": getattr(result, "count", None),
                "offset_id": offset_id,
                "next_offset_id": next_offset_id,
                "next_offset_id_str": (
                    str(next_offset_id) if next_offset_id is not None else None
                ),
                "count": len(sets),
                "sets": sets,
            },
            ensure_ascii=False,
            indent=2,
        )
    except ValueError as e:
        return f"Error: {e}"
    except Exception as e:
        return log_and_format_error(
            "list_archived_custom_emoji_sets", e, offset_id=offset_id, limit=limit
        )


@mcp.tool(
    annotations=ToolAnnotations(
        title="List Custom Emoji Groups", openWorldHint=True, readOnlyHint=True
    )
)
@with_account(readonly=True)
async def list_custom_emoji_groups(
    kind: str = "stickers",
    resolve_icons: bool = True,
    account: str = None,
) -> str:
    """Browse Telegram's custom-emoji categories for search, stickers, status, or profile."""
    try:
        kind = kind.strip().lower()
        if kind not in _CUSTOM_EMOJI_GROUP_KINDS:
            raise ValueError("kind must be general, stickers, status, or profile_photo.")
        requests = {
            "general": functions.messages.GetEmojiGroupsRequest(hash=0),
            "stickers": functions.messages.GetEmojiStickerGroupsRequest(hash=0),
            "status": functions.messages.GetEmojiStatusGroupsRequest(hash=0),
            "profile_photo": functions.messages.GetEmojiProfilePhotoGroupsRequest(hash=0),
        }
        cl = get_client(account)
        result = await cl(requests[kind])
        groups = [_emoji_group_payload(group) for group in (getattr(result, "groups", None) or [])]
        icon_ids = list(
            dict.fromkeys(group["icon_emoji_id"] for group in groups if group.get("icon_emoji_id"))
        )
        icon_documents = []
        if resolve_icons and icon_ids:
            _requested, documents = await _fetch_custom_emoji_documents(cl, icon_ids)
            icon_documents = _custom_emoji_documents_payload(documents)
        return json.dumps(
            {
                "kind": kind,
                "count": len(groups),
                "groups": groups,
                "icon_documents": icon_documents,
            },
            ensure_ascii=False,
            indent=2,
        )
    except ValueError as e:
        return f"Error: {e}"
    except Exception as e:
        return log_and_format_error("list_custom_emoji_groups", e, kind=kind)


@mcp.tool(
    annotations=ToolAnnotations(
        title="List Emoji Reactions", openWorldHint=True, readOnlyHint=True
    )
)
@with_account(readonly=True)
async def list_emoji_reactions(
    kind: str = "recent",
    limit: int = 50,
    resolve_documents: bool = True,
    account: str = None,
) -> str:
    """List the account's recent or top reactions and resolve custom-emoji entries."""
    try:
        kind = kind.strip().lower()
        if kind not in _CUSTOM_EMOJI_REACTION_KINDS:
            raise ValueError("kind must be recent or top.")
        if not isinstance(limit, int) or isinstance(limit, bool) or not 1 <= limit <= 100:
            raise ValueError("limit must be between 1 and 100.")
        request = (
            functions.messages.GetRecentReactionsRequest(limit=limit, hash=0)
            if kind == "recent"
            else functions.messages.GetTopReactionsRequest(limit=limit, hash=0)
        )
        cl = get_client(account)
        result = await cl(request)
        reactions = [
            _reaction_payload(reaction) for reaction in (getattr(result, "reactions", None) or [])
        ]
        document_ids = list(
            dict.fromkeys(
                reaction["document_id"] for reaction in reactions if reaction.get("document_id")
            )
        )
        documents = []
        if resolve_documents and document_ids:
            _requested, fetched = await _fetch_custom_emoji_documents(cl, document_ids)
            documents = _custom_emoji_documents_payload(fetched)
        return json.dumps(
            {
                "kind": kind,
                "count": len(reactions),
                "reactions": reactions,
                "custom_emoji_documents": documents,
            },
            ensure_ascii=False,
            indent=2,
        )
    except ValueError as e:
        return f"Error: {e}"
    except Exception as e:
        return log_and_format_error("list_emoji_reactions", e, kind=kind, limit=limit)


@mcp.tool(
    annotations=ToolAnnotations(
        title="Search Custom Emoji Sets", openWorldHint=True, readOnlyHint=True
    )
)
@with_account(readonly=True)
async def search_custom_emoji_sets(
    query: str,
    exclude_featured: bool = False,
    limit: int = 100,
    account: str = None,
) -> str:
    """Search custom-emoji packs by title or short name."""
    try:
        query = query.strip()
        if not query:
            raise ValueError("query must not be empty.")
        if not isinstance(limit, int) or isinstance(limit, bool) or not 1 <= limit <= 200:
            raise ValueError("limit must be between 1 and 200.")
        cl = get_client(account)
        result = await cl(
            functions.messages.SearchEmojiStickerSetsRequest(
                q=query,
                hash=0,
                exclude_featured=exclude_featured,
            )
        )
        sets = _sticker_sets_payload(getattr(result, "sets", None) or [])[:limit]
        return json.dumps(
            {"query": query, "count": len(sets), "sets": sets},
            ensure_ascii=False,
            indent=2,
        )
    except ValueError as e:
        return f"Error: {e}"
    except Exception as e:
        return log_and_format_error(
            "search_custom_emoji_sets", e, query=query, exclude_featured=exclude_featured
        )


@mcp.tool(
    annotations=ToolAnnotations(
        title="Get Custom Emoji Set", openWorldHint=True, readOnlyHint=True
    )
)
@with_account(readonly=True)
async def get_custom_emoji_set(
    short_name: Optional[str] = None,
    set_id: Optional[Union[int, str]] = None,
    access_hash: Optional[Union[int, str]] = None,
    offset: int = 0,
    limit: int = 200,
    account: str = None,
) -> str:
    """Get one custom-emoji set and its reusable document IDs."""
    try:
        if not isinstance(offset, int) or isinstance(offset, bool) or offset < 0:
            raise ValueError("offset must be a non-negative integer.")
        if not isinstance(limit, int) or isinstance(limit, bool) or not 1 <= limit <= 200:
            raise ValueError("limit must be between 1 and 200.")
        stickerset = _resolve_sticker_set_input(
            short_name=short_name, set_id=set_id, access_hash=access_hash
        )
        cl = get_client(account)
        result, set_payload = await _get_custom_emoji_set_result(cl, stickerset)
        all_documents = _custom_emoji_documents_payload(getattr(result, "documents", None) or [])
        page = all_documents[offset : offset + limit]
        packs = [
            {
                "emoticon": sanitize_user_content(pack.emoticon),
                "document_ids": list(pack.documents),
                "document_id_strings": [str(value) for value in pack.documents],
            }
            for pack in (getattr(result, "packs", None) or [])
        ]
        keywords = [
            {
                "document_id": keyword.document_id,
                "document_id_str": str(keyword.document_id),
                "keywords": [sanitize_user_content(value) for value in keyword.keyword],
            }
            for keyword in (getattr(result, "keywords", None) or [])
        ]
        return json.dumps(
            {
                "set": set_payload,
                "total_documents": len(all_documents),
                "offset": offset,
                "next_offset": (offset + limit if offset + limit < len(all_documents) else None),
                "documents": page,
                "packs": packs,
                "keywords": keywords,
            },
            ensure_ascii=False,
            indent=2,
        )
    except ValueError as e:
        return f"Error: {e}"
    except Exception as e:
        return log_and_format_error(
            "get_custom_emoji_set", e, short_name=short_name, set_id=set_id
        )


@mcp.tool(
    annotations=ToolAnnotations(
        title="Pick Random Custom Emoji", openWorldHint=True, readOnlyHint=True
    )
)
@with_account(readonly=True)
async def pick_random_custom_emoji(
    source: str = "recent",
    query: str = "",
    emoticon: str = "",
    lang_codes: Optional[List[str]] = None,
    limit: int = 100,
    seed: Optional[int] = None,
    account: str = None,
) -> str:
    """Pick a valid random custom emoji from search, recent, installed, or featured results."""
    try:
        source = source.strip().lower()
        if source not in {"search", "recent", "installed", "featured"}:
            raise ValueError("source must be search, recent, installed, or featured.")
        if not isinstance(limit, int) or isinstance(limit, bool) or not 1 <= limit <= 200:
            raise ValueError("limit must be between 1 and 200.")
        if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int)):
            raise ValueError("seed must be an integer or null.")
        rng = random.Random(seed) if seed is not None else random.SystemRandom()
        cl = get_client(account)

        if source == "search":
            documents, _next_offset, _search_source = await _search_custom_emoji_documents(
                cl,
                query=query,
                emoticon=emoticon,
                lang_codes=lang_codes,
                offset=0,
                limit=min(limit, 100),
            )
        elif source == "recent":
            document_ids, _sources = await _recent_custom_emoji_ids(cl, limit)
            if document_ids:
                _requested, documents = await _fetch_custom_emoji_documents(cl, document_ids)
            else:
                documents = []
        else:
            if source == "installed":
                result = await cl(functions.messages.GetEmojiStickersRequest(hash=0))
            else:
                result = await cl(functions.messages.GetFeaturedEmojiStickersRequest(hash=0))
            sets = [
                value
                for value in (getattr(result, "sets", None) or [])
                if (_sticker_set_payload(value) or {}).get("custom_emoji")
            ]
            if not sets:
                documents = []
            else:
                chosen_set_value = rng.choice(sets)
                chosen_set = getattr(chosen_set_value, "set", None) or chosen_set_value
                stickerset = types.InputStickerSetID(
                    id=chosen_set.id, access_hash=chosen_set.access_hash
                )
                set_result, _set_payload = await _get_custom_emoji_set_result(cl, stickerset)
                documents = [
                    document
                    for document in (getattr(set_result, "documents", None) or [])
                    if _custom_emoji_attribute(document) is not None
                ][:limit]

        if not documents:
            raise ValueError(f"no custom emoji were found for source={source}.")
        selected = _custom_emoji_document_payload(rng.choice(documents))
        return json.dumps(
            {
                "source": source,
                "pool_size": len(documents),
                "seed": seed,
                "selected": selected,
            },
            ensure_ascii=False,
            indent=2,
        )
    except ValueError as e:
        return f"Error: {e}"
    except Exception as e:
        return log_and_format_error(
            "pick_random_custom_emoji", e, source=source, query=query, emoticon=emoticon
        )


@mcp.tool(
    annotations=ToolAnnotations(
        title="Inspect Message Custom Emoji", openWorldHint=True, readOnlyHint=True
    )
)
@with_account(readonly=True)
@validate_id("chat_id")
async def inspect_message_custom_emojis(
    chat_id: Union[int, str],
    message_id: int,
    resolve_documents: bool = True,
    account: str = None,
) -> str:
    """Extract custom-emoji IDs from message text, captions, polls, and reactions."""
    try:
        cl = get_client(account)
        entity = await resolve_entity(chat_id, cl)
        message = await cl.get_messages(entity, ids=message_id)
        if not message:
            raise ValueError(f"message {message_id} was not found.")

        occurrences = _custom_emoji_entity_occurrences(
            getattr(message, "message", "") or "",
            getattr(message, "entities", None),
            source="message",
        )
        poll_media = getattr(message, "poll", None)
        poll = getattr(poll_media, "poll", None)
        if isinstance(poll_media, types.Poll):
            poll = poll_media
        if poll is not None:
            question = getattr(poll, "question", None)
            occurrences.extend(
                _custom_emoji_entity_occurrences(
                    getattr(question, "text", "") or "",
                    getattr(question, "entities", None),
                    source="poll.question",
                )
            )
            for index, answer in enumerate(getattr(poll, "answers", None) or []):
                answer_text = getattr(answer, "text", None)
                occurrences.extend(
                    _custom_emoji_entity_occurrences(
                        getattr(answer_text, "text", "") or "",
                        getattr(answer_text, "entities", None),
                        source=f"poll.answer[{index}]",
                    )
                )
            poll_results = getattr(poll_media, "results", None)
            occurrences.extend(
                _custom_emoji_entity_occurrences(
                    getattr(poll_results, "solution", "") or "",
                    getattr(poll_results, "solution_entities", None),
                    source="poll.solution",
                )
            )

        for reaction_result in getattr(getattr(message, "reactions", None), "results", None) or []:
            reaction = getattr(reaction_result, "reaction", None)
            if isinstance(reaction, types.ReactionCustomEmoji):
                occurrences.append(
                    {
                        "source": "reaction",
                        "document_id": reaction.document_id,
                        "document_id_str": str(reaction.document_id),
                        "count": getattr(reaction_result, "count", None),
                        "chosen": getattr(reaction_result, "chosen_order", None) is not None,
                        "chosen_order": getattr(reaction_result, "chosen_order", None),
                    }
                )

        document_ids = list(
            dict.fromkeys(
                occurrence["document_id"]
                for occurrence in occurrences
                if occurrence.get("document_id")
            )
        )
        documents = []
        if resolve_documents and document_ids:
            _requested, fetched = await _fetch_custom_emoji_documents(cl, document_ids)
            documents = _custom_emoji_documents_payload(fetched)
        resolved_ids = {document["document_id"] for document in documents}
        return json.dumps(
            {
                "message_id": message_id,
                "count": len(occurrences),
                "document_ids": document_ids,
                "document_id_strings": [str(value) for value in document_ids],
                "occurrences": occurrences,
                "documents": documents,
                "missing_documents": [
                    value for value in document_ids if value not in resolved_ids
                ],
                "missing_document_strings": [
                    str(value) for value in document_ids if value not in resolved_ids
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
    except ValueError as e:
        return f"Error: {e}"
    except Exception as e:
        return log_and_format_error(
            "inspect_message_custom_emojis", e, chat_id=chat_id, message_id=message_id
        )


@mcp.tool(
    annotations=ToolAnnotations(
        title="Preview Custom Emoji", openWorldHint=True, readOnlyHint=True
    )
)
@with_account(readonly=True)
async def preview_custom_emoji(
    document_id: Union[int, str],
    account: str = None,
):
    """Render a custom emoji's largest static Telegram thumbnail inline."""
    try:
        document_id = _coerce_int64(document_id, name="document_id", positive=True)
        cl = get_client(account)
        _requested, documents = await _fetch_custom_emoji_documents(cl, [document_id])
        if not documents:
            raise ValueError(f"document_id {document_id} is not a valid custom emoji.")
        data, _document = await _download_custom_emoji_media(
            cl,
            document_id=document_id,
            document=documents[0],
            file=bytes,
            preview=True,
        )
        if not isinstance(data, (bytes, bytearray)) or not data:
            raise ValueError("Telegram did not return preview image bytes.")
        data = bytes(data)
        return Image(data=data, format=_preview_image_format(data))
    except ValueError as e:
        return f"Error: {e}"
    except Exception as e:
        return log_and_format_error("preview_custom_emoji", e, document_id=document_id)


@mcp.tool(
    annotations=ToolAnnotations(
        title="Download Custom Emoji", openWorldHint=True, destructiveHint=True
    )
)
@with_account(readonly=False)
async def download_custom_emoji(
    document_id: Union[int, str],
    preview: bool = False,
    file_path: Optional[str] = None,
    ctx: Optional[Context] = None,
    account: str = None,
) -> str:
    """Download a custom emoji's native asset or its largest Telegram thumbnail."""
    try:
        document_id = _coerce_int64(document_id, name="document_id", positive=True)
        cl = get_client(account)
        _requested, documents = await _fetch_custom_emoji_documents(cl, [document_id])
        if not documents:
            raise ValueError(f"document_id {document_id} is not a valid custom emoji.")
        document = documents[0]
        default_name = f"custom_emoji_{'preview_' if preview else ''}{document_id}"
        out_path, path_error = await _resolve_writable_file_path(
            raw_path=file_path,
            default_filename=default_name,
            ctx=ctx,
            tool_name="download_custom_emoji",
        )
        if path_error:
            return (
                path_error
                if path_error.lstrip().lower().startswith("error:")
                else f"Error: {path_error}"
            )
        target = out_path.with_suffix("")
        downloaded, document = await _download_custom_emoji_media(
            cl,
            document_id=document_id,
            document=document,
            file=str(target),
            preview=preview,
        )
        if not downloaded:
            raise ValueError("Telegram did not return a downloadable asset.")
        final_path = Path(downloaded).resolve(strict=True)
        roots, roots_error = await _ensure_allowed_roots(ctx, "download_custom_emoji")
        if roots_error:
            return (
                roots_error
                if roots_error.lstrip().lower().startswith("error:")
                else f"Error: {roots_error}"
            )
        if not _path_is_within_any_root(final_path, roots):
            raise ValueError("downloaded path is outside allowed roots.")
        return json.dumps(
            {
                "path": str(final_path),
                "preview": preview,
                "document": _custom_emoji_document_payload(document),
            },
            ensure_ascii=False,
            indent=2,
        )
    except ValueError as e:
        return f"Error: {e}"
    except Exception as e:
        return log_and_format_error(
            "download_custom_emoji", e, document_id=document_id, preview=preview
        )


@mcp.tool(
    annotations=ToolAnnotations(
        title="Send Custom Emoji Message", openWorldHint=True, destructiveHint=True
    )
)
@with_account(readonly=False)
@validate_id("chat_id")
async def send_custom_emoji_message(
    chat_id: Union[int, str],
    document_id: Union[int, str],
    fallback: Optional[str] = None,
    text_before: str = "",
    text_after: str = "",
    silent: bool = False,
    reply_to_message_id: Optional[int] = None,
    account: str = None,
) -> str:
    """Send a validated custom emoji, optionally surrounded by normal text."""
    try:
        document_id = _coerce_int64(document_id, name="document_id", positive=True)
        cl = get_client(account)
        _requested, documents = await _fetch_custom_emoji_documents(cl, [document_id])
        if not documents:
            raise ValueError(f"document_id {document_id} is not a valid custom emoji.")
        document = documents[0]
        await _require_custom_emoji_entitlement(cl, document)
        document_payload = _custom_emoji_document_payload(document)
        expected_fallback = document_payload["fallback"]
        if fallback is not None and fallback != expected_fallback:
            raise ValueError(
                f"fallback must match the custom emoji alt exactly: {expected_fallback!r}."
            )
        fallback = expected_fallback
        entity = await resolve_entity(chat_id, cl)
        html = (
            f"{escape(text_before)}"
            f'<tg-emoji emoji-id="{document_id}">{escape(fallback)}</tg-emoji>'
            f"{escape(text_after)}"
        )
        sent = await cl.send_message(
            entity,
            html,
            parse_mode="html",
            silent=silent,
            reply_to=reply_to_message_id,
        )
        return json.dumps(
            {
                "message_id": getattr(sent, "id", None),
                "chat_id": chat_id,
                "document_id": document_id,
                "document_id_str": str(document_id),
                "fallback": fallback,
            },
            ensure_ascii=False,
            indent=2,
        )
    except ValueError as e:
        return f"Error: {e}"
    except Exception as e:
        return log_and_format_error(
            "send_custom_emoji_message", e, chat_id=chat_id, document_id=document_id
        )


@mcp.tool(
    annotations=ToolAnnotations(
        title="Send Custom Emoji Reaction",
        openWorldHint=True,
        destructiveHint=False,
        idempotentHint=True,
    )
)
@with_account(readonly=False)
@validate_id("chat_id")
async def send_custom_emoji_reaction(
    chat_id: Union[int, str],
    message_id: int,
    document_id: Union[int, str],
    big: bool = False,
    add_to_recent: bool = True,
    replace_existing: bool = False,
    account: str = None,
) -> str:
    """Add a validated custom reaction, preserving existing chosen reactions by default."""
    try:
        document_id = _coerce_int64(document_id, name="document_id", positive=True)
        cl = get_client(account)
        _requested, documents = await _fetch_custom_emoji_documents(cl, [document_id])
        if not documents:
            raise ValueError(f"document_id {document_id} is not a valid custom emoji.")
        me = await _require_premium(cl)
        reactions = []
        if not replace_existing:
            entity = await resolve_entity(chat_id, cl)
            message = await cl.get_messages(entity, ids=message_id)
            if not message:
                raise ValueError(f"message {message_id} was not found.")
            chosen_results = sorted(
                [
                    result
                    for result in (
                        getattr(getattr(message, "reactions", None), "results", None) or []
                    )
                    if getattr(result, "chosen_order", None) is not None
                ],
                key=lambda result: (
                    getattr(result, "chosen_order", None) is None,
                    getattr(result, "chosen_order", 0) or 0,
                ),
            )
            reactions = [
                result.reaction
                for result in chosen_results
                if isinstance(result.reaction, (types.ReactionEmoji, types.ReactionCustomEmoji))
            ]
        custom_reaction = types.ReactionCustomEmoji(document_id=document_id)
        already_present = _reaction_identity(custom_reaction) in {
            _reaction_identity(reaction) for reaction in reactions
        }
        if not already_present:
            reactions.append(custom_reaction)
        reaction_limit = await _reaction_limit(cl, premium=bool(getattr(me, "premium", False)))
        if len(reactions) > reaction_limit:
            reactions = reactions[-reaction_limit:]
        if not already_present or replace_existing:
            peer = await resolve_input_entity(chat_id, cl)
            await cl(
                functions.messages.SendReactionRequest(
                    peer=peer,
                    msg_id=message_id,
                    big=big,
                    add_to_recent=add_to_recent,
                    reaction=reactions,
                )
            )
        return json.dumps(
            {
                "message_id": message_id,
                "document_id": document_id,
                "document_id_str": str(document_id),
                "replace_existing": replace_existing,
                "already_present": already_present,
                "reaction_limit": reaction_limit,
                "reactions": [_reaction_payload(reaction) for reaction in reactions],
            },
            ensure_ascii=False,
            indent=2,
        )
    except ValueError as e:
        return f"Error: {e}"
    except Exception as e:
        return log_and_format_error(
            "send_custom_emoji_reaction",
            e,
            chat_id=chat_id,
            message_id=message_id,
            document_id=document_id,
        )


@mcp.tool(
    annotations=ToolAnnotations(title="Get My Emoji Status", openWorldHint=True, readOnlyHint=True)
)
@with_account(readonly=True)
async def get_my_emoji_status(account: str = None) -> str:
    """Get the account's current custom-emoji status and resolved document."""
    try:
        cl = get_client(account)
        me = await cl.get_me()
        status = _emoji_status_payload(getattr(me, "emoji_status", None))
        documents = []
        document_id = status.get("document_id")
        if document_id:
            _requested, fetched = await _fetch_custom_emoji_documents(cl, [document_id])
            documents = _custom_emoji_documents_payload(fetched)
        return json.dumps(
            {"status": status, "documents": documents},
            ensure_ascii=False,
            indent=2,
        )
    except ValueError as e:
        return f"Error: {e}"
    except Exception as e:
        return log_and_format_error("get_my_emoji_status", e)


@mcp.tool(
    annotations=ToolAnnotations(title="List Emoji Statuses", openWorldHint=True, readOnlyHint=True)
)
@with_account(readonly=True)
async def list_emoji_statuses(kind: str = "recent", limit: int = 100, account: str = None) -> str:
    """List recent, default, collectible, or channel custom-emoji statuses."""
    try:
        kind = kind.strip().lower()
        if kind not in _EMOJI_STATUS_KINDS:
            raise ValueError(
                "kind must be recent, default, collectible, channel_default, or channel_restricted."
            )
        if not isinstance(limit, int) or isinstance(limit, bool) or not 1 <= limit <= 200:
            raise ValueError("limit must be between 1 and 200.")
        requests = {
            "recent": functions.account.GetRecentEmojiStatusesRequest(hash=0),
            "default": functions.account.GetDefaultEmojiStatusesRequest(hash=0),
            "collectible": functions.account.GetCollectibleEmojiStatusesRequest(hash=0),
            "channel_default": functions.account.GetChannelDefaultEmojiStatusesRequest(hash=0),
            "channel_restricted": functions.account.GetChannelRestrictedStatusEmojisRequest(
                hash=0
            ),
        }
        cl = get_client(account)
        result = await cl(requests[kind])
        if kind == "channel_restricted":
            statuses = [
                {
                    "type": "custom_emoji",
                    "document_id": document_id,
                    "document_id_str": str(document_id),
                }
                for document_id in (getattr(result, "document_id", None) or [])[:limit]
            ]
        else:
            statuses = [
                _emoji_status_payload(status)
                for status in (getattr(result, "statuses", None) or [])[:limit]
            ]
        document_ids = list(
            dict.fromkeys(
                status["document_id"] for status in statuses if status.get("document_id")
            )
        )
        documents = []
        if document_ids:
            _requested, fetched = await _fetch_custom_emoji_documents(cl, document_ids)
            documents = _custom_emoji_documents_payload(fetched)
        return json.dumps(
            {
                "kind": kind,
                "count": len(statuses),
                "statuses": statuses,
                "documents": documents,
            },
            ensure_ascii=False,
            indent=2,
        )
    except ValueError as e:
        return f"Error: {e}"
    except Exception as e:
        return log_and_format_error("list_emoji_statuses", e, kind=kind, limit=limit)


@mcp.tool(
    annotations=ToolAnnotations(
        title="Set Emoji Status",
        openWorldHint=True,
        destructiveHint=True,
        idempotentHint=True,
    )
)
@with_account(readonly=False)
async def set_emoji_status(
    document_id: Union[int, str],
    until: Optional[Union[str, int]] = None,
    account: str = None,
) -> str:
    """Set a validated custom emoji as the account status, optionally until a date."""
    try:
        document_id = _coerce_int64(document_id, name="document_id", positive=True)
        cl = get_client(account)
        await _require_premium(cl)
        _requested, documents = await _fetch_custom_emoji_documents(cl, [document_id])
        if not documents:
            raise ValueError(f"document_id {document_id} is not a valid custom emoji.")
        parsed_until = _parse_until(until)
        await cl(
            functions.account.UpdateEmojiStatusRequest(
                emoji_status=types.EmojiStatus(
                    document_id=document_id,
                    until=parsed_until,
                )
            )
        )
        return json.dumps(
            {
                "document_id": document_id,
                "document_id_str": str(document_id),
                "until": parsed_until.isoformat() if parsed_until else None,
            },
            ensure_ascii=False,
            indent=2,
        )
    except ValueError as e:
        return f"Error: {e}"
    except Exception as e:
        return log_and_format_error("set_emoji_status", e, document_id=document_id, until=until)


@mcp.tool(
    annotations=ToolAnnotations(
        title="Clear Emoji Status",
        openWorldHint=True,
        destructiveHint=True,
        idempotentHint=True,
    )
)
@with_account(readonly=False)
async def clear_emoji_status(account: str = None) -> str:
    """Clear the account's custom-emoji status."""
    try:
        cl = get_client(account)
        await cl(functions.account.UpdateEmojiStatusRequest(emoji_status=types.EmojiStatusEmpty()))
        return "Emoji status cleared."
    except Exception as e:
        return log_and_format_error("clear_emoji_status", e)


@mcp.tool(
    annotations=ToolAnnotations(
        title="Install Custom Emoji Set", openWorldHint=True, destructiveHint=True
    )
)
@with_account(readonly=False)
async def install_custom_emoji_set(
    short_name: Optional[str] = None,
    set_id: Optional[Union[int, str]] = None,
    access_hash: Optional[Union[int, str]] = None,
    archived: bool = False,
    account: str = None,
) -> str:
    """Install a custom-emoji set after verifying that it is not a normal sticker pack."""
    try:
        stickerset = _resolve_sticker_set_input(
            short_name=short_name, set_id=set_id, access_hash=access_hash
        )
        cl = get_client(account)
        _result, set_payload = await _get_custom_emoji_set_result(cl, stickerset)
        await cl(
            functions.messages.InstallStickerSetRequest(stickerset=stickerset, archived=archived)
        )
        return json.dumps(
            {"installed": True, "archived": archived, "set": set_payload},
            ensure_ascii=False,
            indent=2,
        )
    except ValueError as e:
        return f"Error: {e}"
    except Exception as e:
        return log_and_format_error(
            "install_custom_emoji_set", e, short_name=short_name, set_id=set_id
        )


@mcp.tool(
    annotations=ToolAnnotations(
        title="Uninstall Custom Emoji Set", openWorldHint=True, destructiveHint=True
    )
)
@with_account(readonly=False)
async def uninstall_custom_emoji_set(
    short_name: Optional[str] = None,
    set_id: Optional[Union[int, str]] = None,
    access_hash: Optional[Union[int, str]] = None,
    account: str = None,
) -> str:
    """Uninstall a custom-emoji set after verifying its type."""
    try:
        stickerset = _resolve_sticker_set_input(
            short_name=short_name, set_id=set_id, access_hash=access_hash
        )
        cl = get_client(account)
        _result, set_payload = await _get_custom_emoji_set_result(cl, stickerset)
        await cl(functions.messages.UninstallStickerSetRequest(stickerset=stickerset))
        return json.dumps(
            {"uninstalled": True, "set": set_payload},
            ensure_ascii=False,
            indent=2,
        )
    except ValueError as e:
        return f"Error: {e}"
    except Exception as e:
        return log_and_format_error(
            "uninstall_custom_emoji_set", e, short_name=short_name, set_id=set_id
        )


@mcp.tool(
    annotations=ToolAnnotations(
        title="Archive Custom Emoji Set",
        openWorldHint=True,
        destructiveHint=True,
        idempotentHint=True,
    )
)
@with_account(readonly=False)
async def set_custom_emoji_set_archived(
    archived: bool,
    short_name: Optional[str] = None,
    set_id: Optional[Union[int, str]] = None,
    access_hash: Optional[Union[int, str]] = None,
    account: str = None,
) -> str:
    """Archive or unarchive a verified custom-emoji set without uninstalling it."""
    try:
        stickerset = _resolve_sticker_set_input(
            short_name=short_name, set_id=set_id, access_hash=access_hash
        )
        cl = get_client(account)
        _result, set_payload = await _get_custom_emoji_set_result(cl, stickerset)
        await cl(
            functions.messages.ToggleStickerSetsRequest(
                stickersets=[stickerset],
                archive=True if archived else None,
                unarchive=True if not archived else None,
            )
        )
        return json.dumps(
            {"archived": archived, "set": set_payload},
            ensure_ascii=False,
            indent=2,
        )
    except ValueError as e:
        return f"Error: {e}"
    except Exception as e:
        return log_and_format_error(
            "set_custom_emoji_set_archived", e, short_name=short_name, set_id=set_id
        )


__all__ = [
    "search_custom_emojis",
    "get_custom_emoji_documents",
    "list_recent_custom_emojis",
    "list_custom_emoji_sets",
    "list_archived_custom_emoji_sets",
    "list_custom_emoji_groups",
    "list_emoji_reactions",
    "search_custom_emoji_sets",
    "get_custom_emoji_set",
    "pick_random_custom_emoji",
    "inspect_message_custom_emojis",
    "preview_custom_emoji",
    "download_custom_emoji",
    "send_custom_emoji_message",
    "send_custom_emoji_reaction",
    "get_my_emoji_status",
    "list_emoji_statuses",
    "set_emoji_status",
    "clear_emoji_status",
    "install_custom_emoji_set",
    "uninstall_custom_emoji_set",
    "set_custom_emoji_set_archived",
]

"""Telegram-safe text splitting."""

from __future__ import annotations


def utf16_units(value: str) -> int:
    """Return the number of UTF-16 code units used by Telegram limits."""
    return len(value.encode("utf-16-le")) // 2


def telegram_text_chunks(value: str, limit: int = 4096) -> list[str]:
    """Split text without changing it, respecting Telegram's UTF-16 limit."""
    if limit <= 0:
        raise ValueError("limit must be positive")
    if not value:
        return [""]

    chunks: list[str] = []
    start = 0
    units = 0
    for index, char in enumerate(value):
        char_units = 2 if ord(char) > 0xFFFF else 1
        if units + char_units > limit:
            chunks.append(value[start:index])
            start = index
            units = 0
        units += char_units
    chunks.append(value[start:])
    return chunks

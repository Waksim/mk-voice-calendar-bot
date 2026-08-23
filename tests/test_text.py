from tg_voice_transcriber_bot.text import telegram_text_chunks, utf16_units


def test_chunks_preserve_text_and_utf16_limit():
    value = ("абв😀" * 1500) + "конец"
    chunks = telegram_text_chunks(value)

    assert "".join(chunks) == value
    assert all(utf16_units(chunk) <= 4096 for chunk in chunks)


def test_empty_text_is_sendable():
    assert telegram_text_chunks("") == [""]

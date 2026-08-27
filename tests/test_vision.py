import asyncio
import json
import logging
import struct
import sys
import threading
from types import ModuleType, SimpleNamespace

import httpx
import pytest

from tg_voice_transcriber_bot.vision import (
    GeminiVisionProvider,
    OpenAICompatibleVisionProvider,
    RapidOcrProvider,
    VisionImage,
    VisionInputError,
    VisionObservation,
    VisionProviderChain,
    VisionProviderError,
    VisionResult,
    VisionStage,
    VisionUnavailableError,
)


def png(width=64, height=32, *, extra=b""):
    return (
        b"\x89PNG\r\n\x1a\n"
        + struct.pack(">I", 13)
        + b"IHDR"
        + struct.pack(">II", width, height)
        + b"\x08\x02\x00\x00\x00"
        + extra
    )


IMAGE = VisionImage(png(), "image/png")


class Provider:
    def __init__(self, name, model, *, result=None, error=None, delay=0):
        self.provider_name = name
        self.model = model
        self.result = result
        self.error = error
        self.delay = delay
        self.calls = 0

    async def analyze(self, image):
        self.calls += 1
        if self.delay:
            await asyncio.sleep(self.delay)
        if self.error:
            raise self.error
        return self.result


def local_provider(result=None, error=None, delay=0):
    return Provider(
        "Local OCR",
        "rapidocr/pp-ocrv5-cyrillic",
        result=result
        or VisionObservation(
            "Изображение обработано локальным OCR.", "локальный текст"
        ),
        error=error,
        delay=delay,
    )


def chain(stages, local=None, **changes):
    arguments = {
        "local_ocr": local or local_provider(),
        "local_timeout_seconds": 0.2,
    }
    arguments.update(changes)
    return VisionProviderChain(stages, **arguments)


def test_public_contract_is_observation_only_with_chain_diagnostics():
    cloud = Provider(
        "Gemini API",
        "gemini-3.7-flash",
        result=VisionObservation("Скриншот приложения.", "Сб 29 августа"),
    )

    result = asyncio.run(
        chain([VisionStage(cloud, 0.2)]).analyze(IMAGE)
    )

    assert result == VisionResult(
        description="Скриншот приложения.",
        visible_text="Сб 29 августа",
        provider="Gemini API",
        model="gemini-3.7-flash",
        used_local_ocr=False,
    )
    assert set(VisionObservation.__dataclass_fields__) == {
        "description",
        "visible_text",
    }


def test_chain_tries_cloud_fallbacks_in_order_without_calling_local():
    failed = Provider(
        "First API",
        "vision-one",
        error=VisionProviderError("response contained PRIVATE-TEXT"),
    )
    working = Provider(
        "Second API",
        "vision-two",
        result=VisionObservation("Описание", "Текст"),
    )
    local = local_provider()

    result = asyncio.run(
        chain(
            [VisionStage(failed, 0.2), VisionStage(working, 0.2)],
            local,
        ).analyze(IMAGE)
    )

    assert result.provider == "Second API"
    assert result.used_local_ocr is False
    assert failed.calls == 1
    assert working.calls == 1
    assert local.calls == 0


def test_local_ocr_is_mandatory_after_api_errors_and_timeouts():
    failed = Provider(
        "Broken API",
        "vision-broken",
        error=RuntimeError("private upstream body"),
    )
    hanging = Provider("Slow API", "vision-slow", delay=0.05)
    local = local_provider(
        VisionObservation("OCR без визуального описания", "29 августа\n8:00")
    )

    result = asyncio.run(
        chain(
            [VisionStage(failed, 0.2), VisionStage(hanging, 0.001)],
            local,
        ).analyze(IMAGE)
    )

    assert result.visible_text == "29 августа\n8:00"
    assert result.provider == "Local OCR"
    assert result.used_local_ocr is True
    assert local.calls == 1


def test_description_only_cloud_result_continues_to_local_ocr_and_merges():
    cloud = Provider(
        "Cloud API",
        "vision-model",
        result=VisionObservation("Экран бронирования корта", ""),
    )
    local = local_provider(
        VisionObservation("", "29 августа\n8:00–10:00")
    )

    result = asyncio.run(
        chain([VisionStage(cloud, 0.2)], local).analyze(IMAGE)
    )

    assert result.description == "Экран бронирования корта"
    assert result.visible_text == "29 августа\n8:00–10:00"
    assert result.provider == "Local OCR"
    assert result.used_local_ocr is True
    assert cloud.calls == local.calls == 1


def test_chain_reports_content_free_error_only_after_local_ocr_fails():
    secret = "SECRET-VISIBLE-TEXT"
    cloud = Provider("Cloud API", "cloud-model", error=RuntimeError(secret))
    local = local_provider(error=RuntimeError(secret))

    with pytest.raises(VisionUnavailableError) as raised:
        asyncio.run(
            chain([VisionStage(cloud, 0.2)], local).analyze(IMAGE)
        )

    assert secret not in str(raised.value)
    assert cloud.calls == local.calls == 1


def test_logs_never_include_provider_exception_or_observed_text(caplog):
    secret = "DO-NOT-LOG-THIS-OCR-TEXT"
    cloud = Provider("Cloud API", "cloud-model", error=RuntimeError(secret))
    local = local_provider(
        VisionObservation("private description", secret)
    )

    with caplog.at_level(logging.INFO, logger="tg_voice_transcriber_bot.vision"):
        asyncio.run(chain([VisionStage(cloud, 0.2)], local).analyze(IMAGE))

    assert secret not in caplog.text
    assert "private description" not in caplog.text
    assert "error_type=RuntimeError" in caplog.text


@pytest.mark.parametrize(
    ("image", "message"),
    [
        (VisionImage(b"not-an-image", "image/png"), "does not match"),
        (VisionImage(png(), "image/gif"), "unsupported"),
        (VisionImage(b"", "image/png"), "empty"),
        (VisionImage(png(10_000, 10_000), "image/png"), "pixel count"),
        (VisionImage(png(extra=b"x" * 100), "image/png"), "too large"),
    ],
)
def test_image_safety_limits_are_checked_before_any_provider(image, message):
    cloud = Provider(
        "Cloud API",
        "cloud-model",
        result=VisionObservation("Описание", "Текст"),
    )
    limit = 50 if message == "too large" else 1_000

    with pytest.raises(VisionInputError, match=message):
        asyncio.run(
            chain(
                [VisionStage(cloud, 0.2)],
                max_image_bytes=limit,
                max_pixels=20_000_000,
            ).analyze(image)
        )

    assert cloud.calls == 0


def test_output_is_cleaned_and_bounded_before_reaching_planner():
    cloud = Provider(
        "Cloud API",
        "cloud-model",
        result=VisionObservation("  desc\x00ription  ", "  " + "x" * 100),
    )

    result = asyncio.run(
        chain(
            [VisionStage(cloud, 0.2)],
            max_description_chars=20,
            max_visible_text_chars=30,
        ).analyze(IMAGE)
    )

    assert result.description == "description"
    assert len(result.visible_text) == 30
    assert result.visible_text.endswith("… output truncated …]")


def test_gemini_wire_request_contains_image_and_two_field_schema_only():
    observed = {}

    def handler(request: httpx.Request) -> httpx.Response:
        observed["request"] = request
        observed["payload"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "candidates": [
                    {
                        "content": {
                            "parts": [
                                {
                                    "text": json.dumps(
                                        {
                                            "description": "Скриншот бронирования.",
                                            "visible_text": "29 августа 8:00",
                                        },
                                        ensure_ascii=False,
                                    )
                                }
                            ]
                        }
                    }
                ]
            },
        )

    async def scenario():
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(handler)
        ) as client:
            provider = GeminiVisionProvider(
                "gemini-secret",
                model="gemini-3.7-flash",
                timeout_seconds=5,
                client=client,
            )
            return await provider.analyze(IMAGE)

    result = asyncio.run(scenario())
    request = observed["request"]
    payload = observed["payload"]

    assert result == VisionObservation(
        "Скриншот бронирования.", "29 августа 8:00"
    )
    assert request.headers["x-goog-api-key"] == "gemini-secret"
    assert "gemini-secret" not in str(request.url)
    image_part = payload["contents"][0]["parts"][1]["inlineData"]
    assert image_part["mimeType"] == "image/png"
    assert image_part["data"]
    schema = payload["generationConfig"]["responseJsonSchema"]
    assert set(schema["properties"]) == {"description", "visible_text"}
    serialized_schema = json.dumps(schema)
    assert "start_at" not in serialized_schema
    assert "address" not in serialized_schema


def test_openai_compatible_request_is_multimodal_and_observation_only():
    observed = {}

    def handler(request: httpx.Request) -> httpx.Response:
        observed["request"] = request
        observed["payload"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "description": "Экран оплаты.",
                                    "visible_text": "Оплата прошла успешно!",
                                },
                                ensure_ascii=False,
                            )
                        }
                    }
                ]
            },
        )

    async def scenario():
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(handler)
        ) as client:
            provider = OpenAICompatibleVisionProvider(
                "router-secret",
                endpoint_url="https://openrouter.ai/api/v1/chat/completions",
                provider_name="OpenRouter",
                model="google/gemma-4-31b-it:free",
                timeout_seconds=5,
                strict_json_schema=True,
                client=client,
            )
            return await provider.analyze(IMAGE)

    result = asyncio.run(scenario())
    payload = observed["payload"]

    assert result.visible_text == "Оплата прошла успешно!"
    assert observed["request"].headers["authorization"] == "Bearer router-secret"
    user_content = payload["messages"][1]["content"]
    assert user_content[1]["type"] == "image_url"
    assert user_content[1]["image_url"]["url"].startswith(
        "data:image/png;base64,"
    )
    schema = payload["response_format"]["json_schema"]["schema"]
    assert set(schema["properties"]) == {"description", "visible_text"}


def test_provider_calendar_fields_are_rejected_and_local_ocr_takes_over():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "description": "Описание",
                                    "visible_text": "Текст",
                                    "start_at": "2026-08-29T08:00:00+03:00",
                                }
                            )
                        }
                    }
                ]
            },
        )

    async def scenario():
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(handler)
        ) as client:
            provider = OpenAICompatibleVisionProvider(
                "router-secret",
                endpoint_url="https://openrouter.ai/api/v1/chat/completions",
                provider_name="OpenRouter",
                model="google/gemma-4-31b-it:free",
                timeout_seconds=5,
                client=client,
            )
            return await chain(
                [VisionStage(provider, 0.2)], local_provider()
            ).analyze(IMAGE)

    result = asyncio.run(scenario())
    assert result.used_local_ocr is True
    assert result.visible_text == "локальный текст"


def test_rapidocr_adapter_supports_legacy_and_modern_result_shapes():
    legacy_engine = lambda _: (
        [
            [[[0, 0]], "Первая строка", 0.99],
            [[[0, 1]], "Шум", 0.1],
            [[[0, 2]], "Вторая строка", 0.95],
        ],
        0.01,
    )
    provider = RapidOcrProvider(engine=legacy_engine, min_confidence=0.5)

    result = asyncio.run(provider.analyze(IMAGE))

    assert result.visible_text == "Первая строка\nВторая строка"
    assert result.description == ""

    chained = asyncio.run(chain([], provider).analyze(IMAGE))
    assert chained.description == ""
    assert chained.visible_text == "Первая строка\nВторая строка"
    assert chained.used_local_ocr is True


def test_rapidocr_factory_pins_read_only_cyrillic_ppocrv5_models(monkeypatch):
    observed_params = []

    class FakeRapidOCR:
        def __init__(self, *, params):
            observed_params.append(params)

        def __call__(self, data):
            return SimpleNamespace(txts=["текст"], scores=[0.99])

    fake_module = ModuleType("rapidocr")
    fake_module.EngineType = SimpleNamespace(ONNXRUNTIME="onnxruntime")
    fake_module.LangCls = SimpleNamespace(CH="ch")
    fake_module.LangDet = SimpleNamespace(CH="ch")
    fake_module.LangRec = SimpleNamespace(ESLAV="eslav", CYRILLIC="cyrillic")
    fake_module.ModelType = SimpleNamespace(MOBILE="mobile")
    fake_module.OCRVersion = SimpleNamespace(PPOCRV5="PP-OCRv5")
    fake_module.RapidOCR = FakeRapidOCR
    monkeypatch.setitem(sys.modules, "rapidocr", fake_module)
    provider = RapidOcrProvider(model_root_dir="/opt/rapidocr-models")

    result = asyncio.run(provider.analyze(IMAGE))

    assert result.visible_text == "текст"
    assert len(observed_params) == 1
    params = observed_params[0]
    assert params["Global.model_root_dir"] == "/opt/rapidocr-models"
    assert params["Global.log_level"] == "critical"
    assert params["Det.engine_type"] == "onnxruntime"
    assert params["Det.lang_type"] == "ch"
    assert params["Det.model_type"] == "mobile"
    assert params["Det.ocr_version"] == "PP-OCRv5"
    assert params["Cls.lang_type"] == "ch"
    assert params["Cls.model_type"] == "mobile"
    assert params["Cls.ocr_version"] == "PP-OCRv5"
    assert params["Rec.lang_type"] == "cyrillic"
    assert params["Rec.model_type"] == "mobile"
    assert params["Rec.ocr_version"] == "PP-OCRv5"


def test_rapidocr_uses_eslav_model_only_when_cyrillic_finds_no_text(monkeypatch):
    observed_languages = []

    class FakeRapidOCR:
        def __init__(self, *, params):
            self.language = params["Rec.lang_type"]
            observed_languages.append(self.language)

        def __call__(self, data):
            if self.language == "cyrillic":
                return SimpleNamespace(txts=None, scores=None)
            return SimpleNamespace(txts=("резервный текст",), scores=(0.99,))

    fake_module = ModuleType("rapidocr")
    fake_module.EngineType = SimpleNamespace(ONNXRUNTIME="onnxruntime")
    fake_module.LangCls = SimpleNamespace(CH="ch")
    fake_module.LangDet = SimpleNamespace(CH="ch")
    fake_module.LangRec = SimpleNamespace(ESLAV="eslav", CYRILLIC="cyrillic")
    fake_module.ModelType = SimpleNamespace(MOBILE="mobile")
    fake_module.OCRVersion = SimpleNamespace(PPOCRV5="PP-OCRv5")
    fake_module.RapidOCR = FakeRapidOCR
    monkeypatch.setitem(sys.modules, "rapidocr", fake_module)
    provider = RapidOcrProvider(model_root_dir="/opt/rapidocr-models")

    result = asyncio.run(provider.analyze(IMAGE))

    assert result.visible_text == "резервный текст"
    assert observed_languages == ["cyrillic", "eslav"]


def test_rapidocr_requires_baked_model_root_without_injected_engine():
    with pytest.raises(VisionInputError, match="Local OCR configuration"):
        RapidOcrProvider()


def test_local_ocr_timeout_is_bounded_after_cloud_fallbacks():
    local = local_provider(delay=0.05)

    with pytest.raises(VisionUnavailableError):
        asyncio.run(
            chain([], local, local_timeout_seconds=0.001).analyze(IMAGE)
        )

    assert local.calls == 1


def test_timed_out_local_ocr_never_runs_concurrently_on_same_engine():
    started = threading.Event()
    release = threading.Event()
    calls = []

    def blocking_engine(data):
        calls.append(data)
        started.set()
        release.wait(timeout=1)
        return SimpleNamespace(txts=("готово",), scores=(0.99,))

    provider = RapidOcrProvider(engine=blocking_engine)

    async def scenario():
        first = asyncio.create_task(
            chain([], provider, local_timeout_seconds=0.01).analyze(IMAGE)
        )
        while not started.is_set():
            await asyncio.sleep(0)
        with pytest.raises(VisionUnavailableError):
            await first
        with pytest.raises(VisionUnavailableError):
            await chain([], provider, local_timeout_seconds=0.1).analyze(IMAGE)
        assert len(calls) == 1
        release.set()
        await asyncio.sleep(0.01)

    asyncio.run(scenario())


def test_invalid_http_response_is_never_copied_into_error():
    secret = "PRIVATE-PROVIDER-BODY"

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text=secret)

    async def scenario():
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(handler)
        ) as client:
            provider = OpenAICompatibleVisionProvider(
                "router-secret",
                endpoint_url="https://openrouter.ai/api/v1/chat/completions",
                provider_name="OpenRouter",
                model="vision-model",
                timeout_seconds=5,
                client=client,
            )
            with pytest.raises(VisionProviderError) as raised:
                await provider.analyze(IMAGE)
            return str(raised.value)

    message = asyncio.run(scenario())
    assert secret not in message
    assert "500" in message

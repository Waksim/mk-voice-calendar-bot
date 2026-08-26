"""Bounded, secret-safe OpenRouter structured-output provider."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping, Sequence
from copy import deepcopy
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
import json
import logging
import math
import re
import time
from typing import Any, Awaitable, Callable
from urllib.parse import quote

import httpx

from .gemini import (
    CALENDAR_PLANNER_SYSTEM_INSTRUCTION,
    GeminiApiError,
    GeminiError,
    GeminiRateLimitError,
    ProviderAuthenticationError,
    ProviderCreditError,
    ProviderPermanentError,
    _PLANNER_CALL_ID,
    _RATE_LIMIT_ERROR_OBSERVED,
    _RATE_LIMIT_OBSERVED,
    _allowed_event_ids,
    _calendar_operation_input,
    _calendar_prompt,
    _copy_history_steps,
    _diagnostic_fingerprint,
    _log_structured_output_failure,
    _safe_diagnostic_label,
    _safe_diagnostic_token,
    _validate_input,
)
from .intent import (
    CALENDAR_INTENT_SCHEMA,
    CALENDAR_OPERATION_SCHEMA,
    normalize_calendar_intent,
    normalize_calendar_operation_plan,
)


LOGGER = logging.getLogger("tg_voice_transcriber_bot.planner.openrouter")


class OpenRouterApiError(GeminiApiError):
    """A sanitized OpenRouter request or response failure."""


class OpenRouterCreditError(OpenRouterApiError, ProviderCreditError):
    """The OpenRouter account has insufficient credits (HTTP 402)."""


class OpenRouterAuthenticationError(
    OpenRouterApiError, ProviderAuthenticationError
):
    """OpenRouter rejected the API credential or its model access."""


class OpenRouterRequestRejectedError(OpenRouterApiError, ProviderPermanentError):
    """OpenRouter rejected this request or the configured model access."""


class OpenRouterRateLimitError(GeminiRateLimitError, OpenRouterApiError):
    """OpenRouter rate limiting remained after bounded retries."""


_WIRE_UNSUPPORTED_CONSTRAINTS = frozenset(
    {
        "exclusiveMaximum",
        "exclusiveMinimum",
        "format",
        "maxItems",
        "maxLength",
        "maxProperties",
        "maximum",
        "minItems",
        "minLength",
        "minProperties",
        "minimum",
        "multipleOf",
        "pattern",
        "uniqueItems",
    }
)


def _portable_strict_schema(value: Any) -> Any:
    """Keep the cross-provider strict subset; semantic checks remain local."""

    if isinstance(value, Mapping):
        return {
            key: _portable_strict_schema(item)
            for key, item in value.items()
            if key not in _WIRE_UNSUPPORTED_CONSTRAINTS
        }
    if isinstance(value, list):
        return [_portable_strict_schema(item) for item in value]
    return deepcopy(value)


def _strict_operation_wire_schema() -> dict[str, Any]:
    """Encode optional patch keys using the strict-schema nullable pattern."""

    schema = _portable_strict_schema(CALENDAR_OPERATION_SCHEMA)
    patch_schema = schema["properties"]["operations"]["items"]["properties"][
        "patch"
    ]["anyOf"][0]
    properties = patch_schema["properties"]
    patch_schema["required"] = list(properties)
    for field, property_schema in tuple(properties.items()):
        properties[field] = {
            "anyOf": [property_schema, {"type": "null"}],
        }
    return schema


_OPENROUTER_CALENDAR_INTENT_SCHEMA = _portable_strict_schema(
    CALENDAR_INTENT_SCHEMA
)
_OPENROUTER_CALENDAR_OPERATION_SCHEMA = _strict_operation_wire_schema()

_PATCH_WIRE_INSTRUCTION = """
<openrouter_patch_encoding>
For every update operation, the patch object must contain every schema field.
Rule 7's "only changed fields" means only non-null patch values. Use null for
fields that must remain unchanged. To clear location,
description, or recurrence_rrule, leave that patch field null and put its name
in clear_fields. Never invent values merely to replace nulls.
</openrouter_patch_encoding>
""".strip()


class OpenRouterApi:
    """OpenRouter Chat Completions implementation of ``GeminiProvider``."""

    _BASE_URL = "https://openrouter.ai/api/v1"
    _DEFAULT_MODEL = "nvidia/nemotron-3-super-120b-a12b:free"
    _MAX_REQUEST_BYTES = 64 * 1024
    _MAX_RESPONSE_BYTES = 1_000_000
    _REASONING_EFFORTS = frozenset({"low", "medium", "high"})

    def __init__(
        self,
        api_key: str,
        *,
        timeout_seconds: float,
        timezone: str,
        model: str = _DEFAULT_MODEL,
        reasoning_effort: str = "medium",
        max_tokens: int = 8192,
        max_retries: int = 2,
        max_retry_delay_seconds: float = 30,
        client: httpx.AsyncClient | None = None,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        if not isinstance(api_key, str) or not api_key.strip():
            raise OpenRouterApiError("OpenRouter API key is empty")
        normalized_api_key = api_key.strip()
        if not re.fullmatch(r"[A-Za-z0-9._-]+", normalized_api_key):
            raise OpenRouterApiError("OpenRouter API key is invalid")
        if not isinstance(model, str) or not re.fullmatch(
            r"[A-Za-z0-9][A-Za-z0-9._-]*/"
            r"[A-Za-z0-9][A-Za-z0-9._-]*"
            r"(?::[A-Za-z0-9][A-Za-z0-9._-]*)?",
            model,
        ):
            raise OpenRouterApiError("Configured OpenRouter model name is invalid")
        if reasoning_effort not in self._REASONING_EFFORTS:
            raise OpenRouterApiError("OpenRouter reasoning effort is invalid")
        if (
            isinstance(timeout_seconds, bool)
            or not isinstance(timeout_seconds, (int, float))
            or timeout_seconds <= 0
            or not math.isfinite(float(timeout_seconds))
            or isinstance(max_retries, bool)
            or not isinstance(max_retries, int)
            or max_retries < 0
            or isinstance(max_tokens, bool)
            or not isinstance(max_tokens, int)
            or not 1 <= max_tokens <= 65_536
            or not isinstance(max_retry_delay_seconds, (int, float))
            or isinstance(max_retry_delay_seconds, bool)
            or not math.isfinite(float(max_retry_delay_seconds))
            or max_retry_delay_seconds < 0
        ):
            raise OpenRouterApiError("OpenRouter API configuration is invalid")

        self._api_key = normalized_api_key
        self.model = model
        self.timeout_seconds = float(timeout_seconds)
        self.timezone = timezone
        self.reasoning_effort = reasoning_effort
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self.max_retry_delay_seconds = float(max_retry_delay_seconds)
        self._sleep = sleep
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(
            timeout=httpx.Timeout(
                connect=min(10.0, self.timeout_seconds),
                read=self.timeout_seconds,
                write=min(20.0, self.timeout_seconds),
                pool=min(10.0, self.timeout_seconds),
            ),
            limits=httpx.Limits(max_connections=3, max_keepalive_connections=2),
        )

    @property
    def _chat_url(self) -> str:
        return f"{self._BASE_URL}/chat/completions"

    @property
    def _key_url(self) -> str:
        return f"{self._BASE_URL}/key"

    @property
    def _credits_url(self) -> str:
        return f"{self._BASE_URL}/credits"

    @property
    def _model_url(self) -> str:
        return f"{self._BASE_URL}/model/{quote(self.model, safe='/')}"

    @property
    def _is_free_model(self) -> bool:
        return self.model.endswith(":free")

    @staticmethod
    def _selected_provider(router_mapping: Mapping[str, Any]) -> Any:
        endpoints = router_mapping.get("endpoints")
        endpoints_mapping = endpoints if isinstance(endpoints, Mapping) else {}
        available = endpoints_mapping.get("available")
        if not isinstance(available, list):
            return None
        for endpoint in available[:100]:
            if isinstance(endpoint, Mapping) and endpoint.get("selected") is True:
                return endpoint.get("provider")
        return None

    @staticmethod
    def _response_diagnostics(response: httpx.Response) -> dict[str, str]:
        diagnostics = {
            "generation_id": _safe_diagnostic_token(
                response.headers.get("x-generation-id")
            ),
            "provider": "none",
            "router_strategy": "none",
            "router_region": "none",
            "router_attempt": "none",
            "error_code": "none",
        }
        # Metadata is diagnostic only. Do not parse an unexpectedly large
        # provider body before the normal bounded response validator runs.
        if len(response.content) > OpenRouterApi._MAX_RESPONSE_BYTES:
            return diagnostics
        try:
            body = response.json()
        except (ValueError, RecursionError):
            return diagnostics
        body_mapping = body if isinstance(body, Mapping) else {}
        error = body_mapping.get("error")
        error_mapping = error if isinstance(error, Mapping) else {}
        error_metadata = error_mapping.get("metadata")
        metadata_mapping = (
            error_metadata if isinstance(error_metadata, Mapping) else {}
        )
        router_metadata = body_mapping.get("openrouter_metadata")
        router_mapping = (
            router_metadata if isinstance(router_metadata, Mapping) else {}
        )
        attempts = router_mapping.get("attempts")
        last_attempt = (
            attempts[-1]
            if isinstance(attempts, list)
            and attempts
            and isinstance(attempts[-1], Mapping)
            else {}
        )
        generation_id = (
            response.headers.get("x-generation-id")
            or body_mapping.get("id")
        )
        provider = (
            OpenRouterApi._selected_provider(router_mapping)
            or last_attempt.get("provider")
            or metadata_mapping.get("provider_name")
            or body_mapping.get("provider")
        )
        return {
            "generation_id": _safe_diagnostic_token(generation_id),
            "provider": _safe_diagnostic_label(provider),
            "router_strategy": _safe_diagnostic_token(
                router_mapping.get("strategy")
            ),
            "router_region": _safe_diagnostic_token(router_mapping.get("region")),
            "router_attempt": _safe_diagnostic_token(
                router_mapping.get("attempt")
            ),
            "error_code": _safe_diagnostic_token(error_mapping.get("code")),
        }

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    @classmethod
    def _ensure_request_size(cls, payload: Mapping[str, Any]) -> int:
        try:
            serialized = json.dumps(
                payload,
                ensure_ascii=False,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        except (TypeError, ValueError, UnicodeError):
            raise OpenRouterApiError(
                "OpenRouter API request is not serializable"
            ) from None
        if len(serialized) > cls._MAX_REQUEST_BYTES:
            raise OpenRouterApiError("OpenRouter API request is too large")
        return len(serialized)

    @staticmethod
    def _retry_after(response: httpx.Response) -> float | None:
        value = response.headers.get("retry-after")
        if value is None:
            return None
        try:
            seconds = float(value)
        except ValueError:
            try:
                retry_at = parsedate_to_datetime(value)
            except (TypeError, ValueError, OverflowError):
                return None
            if retry_at.tzinfo is None or retry_at.utcoffset() is None:
                retry_at = retry_at.replace(tzinfo=timezone.utc)
            seconds = (
                retry_at.astimezone(timezone.utc) - datetime.now(timezone.utc)
            ).total_seconds()
        if not math.isfinite(seconds):
            return None
        return max(0.0, seconds)

    def _retry_delay(
        self,
        response: httpx.Response | None,
        attempt: int,
        *,
        rate_limited: bool = False,
    ) -> float:
        retry_after = self._retry_after(response) if response is not None else None
        if retry_after is not None:
            delay = retry_after
        elif rate_limited:
            delay = float(10 * (2**attempt))
        else:
            delay = float(2**attempt)
        return min(delay, self.max_retry_delay_seconds)

    @staticmethod
    def _retryable_status(status_code: int) -> bool:
        return status_code in {408, 429, 524, 529} or 500 <= status_code <= 599

    async def _request(
        self,
        method: str,
        url: str,
        *,
        payload: Mapping[str, Any] | None = None,
        authenticated: bool = True,
    ) -> httpx.Response:
        request_bytes = (
            self._ensure_request_size(payload) if payload is not None else 0
        )
        rate_limited = [False]
        try:
            async with asyncio.timeout(self.timeout_seconds):
                return await self._request_with_retries(
                    method,
                    url,
                    payload=payload,
                    authenticated=authenticated,
                    rate_limited=rate_limited,
                    request_bytes=request_bytes,
                )
        except TimeoutError:
            LOGGER.warning(
                "OpenRouter API deadline exhausted; call_id=%s model=%s "
                "rate_limit_seen=%s timeout=%.3fs",
                _PLANNER_CALL_ID.get(),
                self.model,
                rate_limited[0],
                self.timeout_seconds,
            )
            if rate_limited[0]:
                raise OpenRouterRateLimitError(
                    "OpenRouter API rate limit exceeded"
                ) from None
            raise OpenRouterApiError("OpenRouter API request timed out") from None

    async def _request_with_retries(
        self,
        method: str,
        url: str,
        *,
        payload: Mapping[str, Any] | None,
        authenticated: bool,
        rate_limited: list[bool],
        request_bytes: int,
    ) -> httpx.Response:
        headers = {"accept": "application/json"}
        if authenticated:
            headers["authorization"] = f"Bearer {self._api_key}"
        if payload is not None:
            headers["content-type"] = "application/json"
        endpoint = (
            "chat_completions"
            if url == self._chat_url
            else "key"
            if url == self._key_url
            else "credits"
            if url == self._credits_url
            else "model"
        )
        if endpoint == "chat_completions":
            headers["x-openrouter-metadata"] = "enabled"

        for attempt in range(self.max_retries + 1):
            attempt_started = time.monotonic()
            try:
                response = await self._client.request(
                    method,
                    url,
                    headers=headers,
                    json=dict(payload) if payload is not None else None,
                )
            except httpx.TimeoutException as exc:
                LOGGER.warning(
                    "OpenRouter HTTP attempt failed; call_id=%s model=%s "
                    "endpoint=%s attempt=%d request_bytes=%d elapsed=%.3fs "
                    "error_type=%s",
                    _PLANNER_CALL_ID.get(),
                    self.model,
                    endpoint,
                    attempt + 1,
                    request_bytes,
                    time.monotonic() - attempt_started,
                    type(exc).__name__,
                )
                # The provider may have completed and billed a timed-out
                # generation. Do not start an indistinguishable second one.
                raise OpenRouterApiError(
                    "OpenRouter API request timed out"
                ) from None
            except httpx.HTTPError as exc:
                retry_delay = (
                    self._retry_delay(None, attempt)
                    if attempt < self.max_retries
                    else None
                )
                LOGGER.warning(
                    "OpenRouter HTTP attempt failed; call_id=%s model=%s "
                    "endpoint=%s attempt=%d request_bytes=%d elapsed=%.3fs "
                    "error_type=%s retry_delay=%s",
                    _PLANNER_CALL_ID.get(),
                    self.model,
                    endpoint,
                    attempt + 1,
                    request_bytes,
                    time.monotonic() - attempt_started,
                    type(exc).__name__,
                    f"{retry_delay:.3f}s" if retry_delay is not None else "none",
                )
                if attempt < self.max_retries:
                    await self._sleep(retry_delay or 0)
                    continue
                raise OpenRouterApiError(
                    f"OpenRouter API transport error: {type(exc).__name__}"
                ) from None

            status = response.status_code
            diagnostics = self._response_diagnostics(response)
            retry_delay = (
                self._retry_delay(
                    response,
                    attempt,
                    rate_limited=status == 429,
                )
                if self._retryable_status(status) and attempt < self.max_retries
                else None
            )
            log_method = LOGGER.info if 200 <= status < 300 else LOGGER.warning
            log_method(
                "OpenRouter HTTP response; call_id=%s model=%s endpoint=%s "
                "attempt=%d status=%d request_bytes=%d elapsed=%.3fs "
                "response_bytes=%d "
                "generation_id=%s provider=%s router_strategy=%s "
                "router_region=%s router_attempt=%s error_code=%s "
                "retry_delay=%s",
                _PLANNER_CALL_ID.get(),
                self.model,
                endpoint,
                attempt + 1,
                status,
                request_bytes,
                time.monotonic() - attempt_started,
                len(response.content),
                diagnostics["generation_id"],
                diagnostics["provider"],
                diagnostics["router_strategy"],
                diagnostics["router_region"],
                diagnostics["router_attempt"],
                diagnostics["error_code"],
                f"{retry_delay:.3f}s" if retry_delay is not None else "none",
            )
            if 200 <= status < 300:
                return response
            if status == 402:
                raise OpenRouterCreditError(
                    "OpenRouter API credits are exhausted"
                ) from None
            if status == 401:
                raise OpenRouterAuthenticationError(
                    "OpenRouter API credential or access was rejected"
                ) from None
            if status == 403:
                raise OpenRouterRequestRejectedError(
                    "OpenRouter rejected the request or model access"
                ) from None
            if status == 429:
                rate_limited[0] = True
                if _RATE_LIMIT_OBSERVED.get() is not None:
                    _RATE_LIMIT_ERROR_OBSERVED.set(
                        OpenRouterRateLimitError(
                            "OpenRouter API rate limit exceeded"
                        )
                    )
                    _RATE_LIMIT_OBSERVED.set(True)
            if self._retryable_status(status) and attempt < self.max_retries:
                await self._sleep(retry_delay or 0)
                continue
            if status == 429:
                raise OpenRouterRateLimitError(
                    "OpenRouter API rate limit exceeded"
                ) from None
            raise OpenRouterApiError(
                f"OpenRouter API HTTP status {status}"
            ) from None

        raise OpenRouterApiError("OpenRouter API request failed")

    @classmethod
    def _response_json(cls, response: httpx.Response) -> dict[str, Any]:
        if len(response.content) > cls._MAX_RESPONSE_BYTES:
            raise OpenRouterApiError("OpenRouter API response was too large")
        try:
            body = response.json()
        except ValueError:
            raise OpenRouterApiError(
                "OpenRouter API returned invalid JSON"
            ) from None
        if not isinstance(body, dict):
            raise OpenRouterApiError(
                "OpenRouter API returned an invalid envelope"
            )
        return body

    async def validate(self) -> None:
        key_response = await self._request("GET", self._key_url)
        key_body = self._response_json(key_response)
        key_data = key_body.get("data")
        if not isinstance(key_data, Mapping):
            raise OpenRouterApiError("OpenRouter API key validation failed")
        limit_remaining = key_data.get("limit_remaining")
        if (
            limit_remaining is not None
            and (
                isinstance(limit_remaining, bool)
                or not isinstance(limit_remaining, (int, float))
                or not math.isfinite(float(limit_remaining))
            )
        ):
            raise OpenRouterApiError("OpenRouter API key validation failed")
        if limit_remaining is not None and float(limit_remaining) <= 0:
            raise OpenRouterCreditError(
                "OpenRouter API key spending limit is exhausted"
            )

        credit_response = await self._request("GET", self._credits_url)
        credit_body = self._response_json(credit_response)
        credit_data = credit_body.get("data")
        if not isinstance(credit_data, Mapping):
            raise OpenRouterApiError("OpenRouter credit validation failed")
        total_credits = credit_data.get("total_credits")
        total_usage = credit_data.get("total_usage")
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            for value in (total_credits, total_usage)
        ):
            raise OpenRouterApiError("OpenRouter credit validation failed")
        remaining_credits = float(total_credits) - float(total_usage)
        if remaining_credits < 0 or (
            remaining_credits == 0 and not self._is_free_model
        ):
            raise OpenRouterCreditError("OpenRouter API credits are exhausted")

        model_response = await self._request("GET", self._model_url)
        model_body = self._response_json(model_response)
        model_data = model_body.get("data")
        model_id = (
            model_data.get("id")
            if isinstance(model_data, Mapping)
            else model_body.get("id")
        )
        if model_id != self.model:
            raise OpenRouterApiError("Configured OpenRouter model is unavailable")
        supported = (
            model_data.get("supported_parameters")
            if isinstance(model_data, Mapping)
            else None
        )
        required_parameters = {
            "max_tokens",
            "reasoning",
            "response_format",
            "structured_outputs",
        }
        if not isinstance(supported, list) or not required_parameters.issubset(
            {value for value in supported if isinstance(value, str)}
        ):
            raise OpenRouterApiError(
                "Configured OpenRouter model lacks required parameters"
            )
        reasoning = (
            model_data.get("reasoning")
            if isinstance(model_data, Mapping)
            else None
        )
        supported_efforts = (
            reasoning.get("supported_efforts")
            if isinstance(reasoning, Mapping)
            else None
        )
        if not isinstance(supported_efforts, list) or self.reasoning_effort not in {
            value for value in supported_efforts if isinstance(value, str)
        }:
            raise OpenRouterApiError(
                "Configured OpenRouter model lacks requested reasoning effort"
            )

    def _structured_payload(
        self,
        *,
        messages: Sequence[Mapping[str, str]],
        schema: Mapping[str, Any],
        schema_name: str,
    ) -> dict[str, Any]:
        return {
            "model": self.model,
            "messages": [dict(message) for message in messages],
            "reasoning": {"effort": self.reasoning_effort},
            "max_tokens": self.max_tokens,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": schema_name,
                    "strict": True,
                    "schema": deepcopy(dict(schema)),
                },
            },
            "provider": {
                "require_parameters": True,
                "data_collection": "allow",
            },
        }

    @staticmethod
    def _decode_operation_wire_payload(payload: Any) -> Any:
        """Turn strict-schema null sentinels back into an optional patch."""

        if not isinstance(payload, Mapping):
            return payload
        decoded = deepcopy(dict(payload))
        operations = decoded.get("operations")
        if not isinstance(operations, list):
            return decoded
        for operation in operations:
            if not isinstance(operation, dict):
                continue
            patch = operation.get("patch")
            if isinstance(patch, Mapping):
                compact_patch = {
                    key: value for key, value in patch.items() if value is not None
                }
                operation["patch"] = compact_patch or None
        return decoded

    @staticmethod
    def _step_text(step: Mapping[str, Any]) -> str | None:
        content = step.get("content")
        if not isinstance(content, Sequence) or isinstance(
            content, (str, bytes, bytearray)
        ):
            return None
        parts = [
            item.get("text")
            for item in content
            if isinstance(item, Mapping)
            and item.get("type") == "text"
            and isinstance(item.get("text"), str)
        ]
        text = "".join(parts)
        return text if text else None

    @classmethod
    def _history_messages(
        cls, history_steps: Sequence[Mapping[str, Any]]
    ) -> list[dict[str, str]]:
        """Convert only text input/output steps; discard native thoughts."""

        messages: list[dict[str, str]] = []
        for step in _copy_history_steps(history_steps):
            step_type = step.get("type")
            role = (
                "user"
                if step_type == "user_input"
                else "assistant"
                if step_type == "model_output"
                else None
            )
            if role is None:
                # Includes Gemini thought/signature steps.  No unrecognized
                # fields are serialized into the OpenRouter request.
                continue
            text = cls._step_text(step)
            if text is not None:
                messages.append({"role": role, "content": text})
        return messages

    @staticmethod
    def _completion_text(body: Mapping[str, Any]) -> str:
        choices = body.get("choices")
        if not isinstance(choices, list) or len(choices) != 1:
            raise OpenRouterApiError(
                "OpenRouter API returned no structured output"
            )
        choice = choices[0]
        if not isinstance(choice, Mapping):
            raise OpenRouterApiError(
                "OpenRouter API returned no structured output"
            )
        if choice.get("finish_reason") not in {None, "stop"}:
            raise OpenRouterApiError("OpenRouter completion was incomplete")
        message = choice.get("message")
        if not isinstance(message, Mapping):
            raise OpenRouterApiError(
                "OpenRouter API returned no structured output"
            )
        content = message.get("content")
        if isinstance(content, str):
            text = content
        elif isinstance(content, list):
            text = "".join(
                item.get("text", "")
                for item in content
                if isinstance(item, Mapping)
                and item.get("type") == "text"
                and isinstance(item.get("text"), str)
            )
        else:
            text = ""
        if not text.strip():
            raise OpenRouterApiError(
                "OpenRouter API returned no structured output"
            )
        return text

    @staticmethod
    def _completion_diagnostics(body: Mapping[str, Any]) -> dict[str, Any]:
        choices = body.get("choices")
        choice = (
            choices[0]
            if isinstance(choices, list)
            and choices
            and isinstance(choices[0], Mapping)
            else {}
        )
        usage = body.get("usage")
        usage_mapping = usage if isinstance(usage, Mapping) else {}
        completion_details = usage_mapping.get("completion_tokens_details")
        completion_details_mapping = (
            completion_details if isinstance(completion_details, Mapping) else {}
        )
        prompt_details = usage_mapping.get("prompt_tokens_details")
        prompt_details_mapping = (
            prompt_details if isinstance(prompt_details, Mapping) else {}
        )
        router_metadata = body.get("openrouter_metadata")
        router_mapping = (
            router_metadata if isinstance(router_metadata, Mapping) else {}
        )
        attempts = router_mapping.get("attempts")
        safe_attempts = []
        if isinstance(attempts, list):
            for attempt in attempts[:10]:
                if isinstance(attempt, Mapping):
                    safe_attempts.append(
                        {
                            "provider": _safe_diagnostic_label(
                                attempt.get("provider")
                            ),
                            "model": _safe_diagnostic_token(attempt.get("model")),
                            "status": _safe_diagnostic_token(
                                attempt.get("status")
                            ),
                        }
                    )
        pipeline = router_mapping.get("pipeline")
        safe_pipeline = []
        if isinstance(pipeline, list):
            for stage in pipeline[:10]:
                if isinstance(stage, Mapping):
                    safe_pipeline.append(
                        {
                            "type": _safe_diagnostic_token(stage.get("type")),
                            "name": _safe_diagnostic_label(stage.get("name")),
                        }
                    )
        safe_usage: dict[str, int | float] = {
            key: value
            for key in (
                "prompt_tokens",
                "completion_tokens",
                "total_tokens",
            )
            if isinstance((value := usage_mapping.get(key)), int)
            and not isinstance(value, bool)
            and value >= 0
        }
        nested_token_fields = {
            "reasoning_tokens": completion_details_mapping.get(
                "reasoning_tokens",
                usage_mapping.get("reasoning_tokens"),
            ),
            "cached_tokens": prompt_details_mapping.get(
                "cached_tokens",
                usage_mapping.get("cached_tokens"),
            ),
        }
        safe_usage.update(
            {
                key: value
                for key, value in nested_token_fields.items()
                if isinstance(value, int)
                and not isinstance(value, bool)
                and value >= 0
            }
        )
        cost = usage_mapping.get("cost")
        if (
            isinstance(cost, int)
            and not isinstance(cost, bool)
            and cost >= 0
        ) or (isinstance(cost, float) and math.isfinite(cost) and cost >= 0):
            safe_usage["cost"] = cost
        return {
            "generation_id": _safe_diagnostic_token(body.get("id")),
            "returned_model": _safe_diagnostic_token(body.get("model")),
            "provider": _safe_diagnostic_label(
                OpenRouterApi._selected_provider(router_mapping)
                or body.get("provider")
            ),
            "finish_reason": _safe_diagnostic_token(choice.get("finish_reason")),
            "usage": safe_usage,
            "router_strategy": _safe_diagnostic_token(
                router_mapping.get("strategy")
            ),
            "router_region": _safe_diagnostic_token(router_mapping.get("region")),
            "router_attempt": _safe_diagnostic_token(
                router_mapping.get("attempt")
            ),
            "router_is_byok": (
                router_mapping.get("is_byok")
                if isinstance(router_mapping.get("is_byok"), bool)
                else None
            ),
            "attempts": safe_attempts,
            "pipeline": safe_pipeline,
        }

    async def _complete_structured(
        self,
        *,
        messages: Sequence[Mapping[str, str]],
        schema: Mapping[str, Any],
        schema_name: str,
    ) -> tuple[Any, str]:
        payload = self._structured_payload(
            messages=messages,
            schema=schema,
            schema_name=schema_name,
        )
        response = await self._request("POST", self._chat_url, payload=payload)
        try:
            body = self._response_json(response)
        except OpenRouterApiError as exc:
            LOGGER.warning(
                "OpenRouter completion envelope rejected; call_id=%s "
                "model=%s response_bytes=%d error=%s",
                _PLANNER_CALL_ID.get(),
                self.model,
                len(response.content),
                str(exc),
            )
            raise
        completion_diagnostics = self._completion_diagnostics(body)
        metadata_json = json.dumps(
            completion_diagnostics,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        try:
            text = self._completion_text(body)
        except OpenRouterApiError as exc:
            LOGGER.warning(
                "OpenRouter completion envelope rejected; call_id=%s "
                "model=%s response_bytes=%d metadata=%s error=%s",
                _PLANNER_CALL_ID.get(),
                self.model,
                len(response.content),
                metadata_json,
                str(exc),
            )
            raise
        output_bytes, output_fingerprint = _diagnostic_fingerprint(text)
        LOGGER.info(
            "OpenRouter completion envelope; call_id=%s model=%s "
            "response_bytes=%d output_bytes=%d output_fingerprint=%s "
            "metadata=%s",
            _PLANNER_CALL_ID.get(),
            self.model,
            len(response.content),
            output_bytes,
            output_fingerprint,
            metadata_json,
        )
        try:
            return json.loads(text), text
        except json.JSONDecodeError as exc:
            _log_structured_output_failure(
                provider="OpenRouter",
                model=self.model,
                phase="json_decode",
                output=text,
                reason=f"line={exc.lineno} column={exc.colno} position={exc.pos}",
            )
            raise OpenRouterApiError(
                "OpenRouter API returned invalid structured JSON"
            ) from None

    async def extract_event(
        self,
        transcript: str,
        *,
        reference_time: datetime,
        account: str,
    ) -> dict[str, Any]:
        try:
            _validate_input(transcript, reference_time)
            prompt = _calendar_prompt(
                transcript,
                reference_time=reference_time,
                account=account,
                timezone=self.timezone,
            )
        except GeminiError as exc:
            raise OpenRouterApiError(str(exc)) from None

        structured, output_text = await self._complete_structured(
            messages=[{"role": "user", "content": prompt}],
            schema=_OPENROUTER_CALENDAR_INTENT_SCHEMA,
            schema_name="calendar_intent",
        )
        try:
            return normalize_calendar_intent(
                structured,
                expected_timezone=self.timezone,
            )
        except ValueError as exc:
            _log_structured_output_failure(
                provider="OpenRouter",
                model=self.model,
                phase="semantic_validation",
                output=output_text,
                reason=str(exc),
            )
            raise OpenRouterApiError(
                f"OpenRouter returned an invalid calendar event: {exc}"
            ) from None

    async def plan_calendar_actions(
        self,
        transcript: str,
        *,
        reference_time: datetime,
        account: str,
        application_state: Mapping[str, Any],
        recent_conversation: Sequence[Mapping[str, Any]],
        history_steps: Sequence[Mapping[str, Any]] = (),
    ) -> dict[str, Any]:
        try:
            _validate_input(transcript, reference_time)
            if not isinstance(application_state, Mapping):
                raise GeminiError("Application state must be an object")
            if isinstance(
                recent_conversation, (str, bytes, bytearray)
            ) or not isinstance(recent_conversation, Sequence):
                raise GeminiError("Recent conversation must be an array")
            history_messages = self._history_messages(history_steps)
            current_input = _calendar_operation_input(
                transcript,
                reference_time=reference_time,
                account=account,
                timezone=self.timezone,
                application_state=application_state,
                recent_conversation=recent_conversation,
            )
            current_text = self._step_text(current_input)
            if current_text is None:
                raise GeminiError("Calendar planner input is invalid")
        except GeminiError as exc:
            raise OpenRouterApiError(str(exc)) from None

        messages = [
            {
                "role": "system",
                "content": (
                    f"{CALENDAR_PLANNER_SYSTEM_INSTRUCTION}\n\n"
                    f"{_PATCH_WIRE_INSTRUCTION}"
                ),
            },
            *history_messages,
            {"role": "user", "content": current_text},
        ]
        structured, output_text = await self._complete_structured(
            messages=messages,
            schema=_OPENROUTER_CALENDAR_OPERATION_SCHEMA,
            schema_name="calendar_operation_plan",
        )
        structured = self._decode_operation_wire_payload(structured)
        try:
            normalized = normalize_calendar_operation_plan(
                structured,
                _allowed_event_ids(application_state),
                expected_timezone=self.timezone,
            )
        except ValueError as exc:
            _log_structured_output_failure(
                provider="OpenRouter",
                model=self.model,
                phase="semantic_validation",
                output=output_text,
                reason=str(exc),
            )
            raise OpenRouterApiError(
                f"OpenRouter returned an invalid calendar plan: {exc}"
            ) from None

        normalized["_interaction_input"] = deepcopy(current_input)
        normalized["_interaction_steps"] = [
            {
                "type": "model_output",
                "content": [{"type": "text", "text": output_text}],
            }
        ]
        return normalized


__all__ = [
    "OpenRouterApi",
    "OpenRouterApiError",
    "OpenRouterAuthenticationError",
    "OpenRouterCreditError",
    "OpenRouterRateLimitError",
    "OpenRouterRequestRejectedError",
]

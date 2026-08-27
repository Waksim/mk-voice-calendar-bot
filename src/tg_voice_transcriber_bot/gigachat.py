"""Direct, bounded and secret-safe GigaChat v1 structured-output provider."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping, Sequence
from copy import deepcopy
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
import json
import logging
import math
from pathlib import Path
import re
import ssl
import time
from typing import Any, Awaitable, Callable
from urllib.parse import urlencode, urlsplit
import uuid

import httpx

from .gemini import (
    CALENDAR_PLANNER_SYSTEM_INSTRUCTION,
    PLANNER_MODEL_FIELD,
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
    _history_has_image_observations,
    _log_structured_output_failure,
    _safe_diagnostic_token,
    _validate_input,
    _validate_planner_input,
)
from .intent import normalize_calendar_intent, normalize_calendar_operation_plan
from .openrouter import (
    _OPENROUTER_CALENDAR_INTENT_SCHEMA,
    _OPENROUTER_CALENDAR_OPERATION_SCHEMA,
    OpenRouterApi,
)


LOGGER = logging.getLogger("tg_voice_transcriber_bot.planner.gigachat")

_EXTRACT_FUNCTION_NAME = "extract_calendar_event"
_PLAN_FUNCTION_NAME = "plan_calendar_actions"


def _unwrap_nullable_schema(value: Mapping[str, Any]) -> tuple[Any, bool]:
    """Return the non-null branch of GigaChat-incompatible nullable unions."""

    branches = value.get("anyOf")
    if branches is None:
        return value, False
    if not isinstance(branches, list) or len(branches) != 2:
        raise GigaChatConfigurationError(
            "GigaChat function schema contains an unsupported union"
        )
    null_branches = [
        branch
        for branch in branches
        if isinstance(branch, Mapping) and branch.get("type") == "null"
    ]
    non_null_branches = [branch for branch in branches if branch not in null_branches]
    if len(null_branches) != 1 or len(non_null_branches) != 1:
        raise GigaChatConfigurationError(
            "GigaChat function schema contains an unsupported union"
        )
    return non_null_branches[0], True


def _giga_function_schema(value: Any) -> Any:
    """Convert the portable strict schema to GigaChat function parameters.

    GigaChat rejects the portable ``anyOf: [T, null]`` encoding.  A nullable
    object property is represented instead as an optional property: unwrap its
    non-null branch and remove its name from that object's ``required`` list.
    """

    if isinstance(value, Mapping):
        unwrapped, _ = _unwrap_nullable_schema(value)
        if unwrapped is not value:
            return _giga_function_schema(unwrapped)

        converted = {
            key: _giga_function_schema(item)
            for key, item in value.items()
            if key not in {"properties", "required"}
        }
        properties = value.get("properties")
        if isinstance(properties, Mapping):
            nullable_properties: set[str] = set()
            converted_properties: dict[str, Any] = {}
            for name, property_schema in properties.items():
                if not isinstance(name, str):
                    raise GigaChatConfigurationError(
                        "GigaChat function schema contains an invalid property"
                    )
                if isinstance(property_schema, Mapping):
                    unwrapped_property, nullable = _unwrap_nullable_schema(
                        property_schema
                    )
                else:
                    unwrapped_property, nullable = property_schema, False
                if nullable:
                    nullable_properties.add(name)
                converted_properties[name] = _giga_function_schema(
                    unwrapped_property
                )
            converted["properties"] = converted_properties

            required = value.get("required")
            if required is not None:
                if not isinstance(required, list) or any(
                    not isinstance(name, str) for name in required
                ):
                    raise GigaChatConfigurationError(
                        "GigaChat function schema has an invalid required list"
                    )
                converted["required"] = [
                    name for name in required if name not in nullable_properties
                ]
        elif "required" in value:
            raise GigaChatConfigurationError(
                "GigaChat function schema has required fields without properties"
            )
        return converted
    if isinstance(value, list):
        return [_giga_function_schema(item) for item in value]
    return deepcopy(value)


class GigaChatApiError(GeminiApiError):
    """A sanitized GigaChat request or response failure."""


class GigaChatAuthenticationError(
    GigaChatApiError, ProviderAuthenticationError
):
    """GigaChat rejected the configured OAuth credentials or bearer token."""


class GigaChatQuotaError(GigaChatApiError, ProviderCreditError):
    """The GigaChat account cannot serve the request due to quota or billing."""


class GigaChatRequestRejectedError(GigaChatApiError, ProviderPermanentError):
    """GigaChat rejected a validly authenticated request permanently."""


class GigaChatConfigurationError(GigaChatApiError, ProviderPermanentError):
    """The configured stable model, endpoint, or CA bundle is unusable."""


class GigaChatRateLimitError(GeminiRateLimitError, GigaChatApiError):
    """GigaChat rate limiting remained after the bounded retry budget."""


_GIGACHAT_CALENDAR_INTENT_SCHEMA = _giga_function_schema(
    _OPENROUTER_CALENDAR_INTENT_SCHEMA
)
_GIGACHAT_CALENDAR_OPERATION_SCHEMA = _giga_function_schema(
    _OPENROUTER_CALENDAR_OPERATION_SCHEMA
)


class GigaChatApi:
    """HTTP GigaChat v1 implementation of :class:`GeminiProvider`.

    The provider deliberately supports one stable model alias.  OAuth and API
    traffic share a one-slot semaphore because the corporate credential is also
    shared by every structured-output request made through this instance.
    """

    _DEFAULT_BASE_URL = "https://api.giga.chat/v1"
    _DEFAULT_AUTH_URL = "https://ngw.devices.sberbank.ru:9443/api/v2/oauth"
    _STABLE_MODEL = "GigaChat-2-Max"
    _RETRYABLE_STATUS_CODES = frozenset({429, 500, 502, 503, 504})
    _MAX_REQUEST_BYTES = 64 * 1024
    _MAX_RESPONSE_BYTES = 1_000_000

    def __init__(
        self,
        credentials: str,
        *,
        ca_bundle_path: str | Path,
        timeout_seconds: float,
        timezone: str,
        scope: str = "GIGACHAT_API_CORP",
        model: str = _STABLE_MODEL,
        base_url: str = _DEFAULT_BASE_URL,
        auth_url: str = _DEFAULT_AUTH_URL,
        user_agent: str = "tg-voice-calendar-bot/1.0",
        temperature: float = 0.1,
        max_tokens: int = 8192,
        max_retries: int = 2,
        max_retry_delay_seconds: float = 30.0,
        token_refresh_skew_seconds: float = 60.0,
        client: httpx.AsyncClient | None = None,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        clock: Callable[[], float] = time.time,
        uuid4: Callable[[], uuid.UUID] = uuid.uuid4,
    ) -> None:
        normalized_credentials = self._validate_credentials(credentials)
        normalized_scope = self._validate_scope(scope)
        if model != self._STABLE_MODEL:
            raise GigaChatConfigurationError(
                "Configured GigaChat model must use the stable alias"
            )
        normalized_base_url = self._validate_https_url(base_url, field="base URL")
        normalized_auth_url = self._validate_https_url(auth_url, field="OAuth URL")
        normalized_user_agent = self._validate_user_agent(user_agent)
        ca_path = Path(ca_bundle_path).expanduser()
        if not ca_path.is_file():
            raise GigaChatConfigurationError("GigaChat CA bundle is unavailable")
        if (
            isinstance(timeout_seconds, bool)
            or not isinstance(timeout_seconds, (int, float))
            or not math.isfinite(float(timeout_seconds))
            or timeout_seconds <= 0
            or isinstance(temperature, bool)
            or not isinstance(temperature, (int, float))
            or not math.isfinite(float(temperature))
            or not 0 < float(temperature) <= 2
            or isinstance(max_tokens, bool)
            or not isinstance(max_tokens, int)
            or not 1 <= max_tokens <= 65_536
            or isinstance(max_retries, bool)
            or not isinstance(max_retries, int)
            or not 0 <= max_retries <= 10
            or isinstance(max_retry_delay_seconds, bool)
            or not isinstance(max_retry_delay_seconds, (int, float))
            or not math.isfinite(float(max_retry_delay_seconds))
            or max_retry_delay_seconds < 0
            or isinstance(token_refresh_skew_seconds, bool)
            or not isinstance(token_refresh_skew_seconds, (int, float))
            or not math.isfinite(float(token_refresh_skew_seconds))
            or token_refresh_skew_seconds < 0
            or not callable(sleep)
            or not callable(clock)
            or not callable(uuid4)
        ):
            raise GigaChatConfigurationError("GigaChat API configuration is invalid")

        self._credentials = normalized_credentials
        self.scope = normalized_scope
        self.model = model
        self.base_url = normalized_base_url
        self.auth_url = normalized_auth_url
        self.user_agent = normalized_user_agent
        self.ca_bundle_path = ca_path.resolve()
        self.timeout_seconds = float(timeout_seconds)
        self.timezone = timezone
        self.temperature = float(temperature)
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self.max_retry_delay_seconds = float(max_retry_delay_seconds)
        self.token_refresh_skew_seconds = float(token_refresh_skew_seconds)
        self._sleep = sleep
        self._clock = clock
        self._uuid4 = uuid4
        self._semaphore = asyncio.Semaphore(1)
        self._token_lock = asyncio.Lock()
        self._access_token: str | None = None
        self._access_token_expires_at = 0.0
        self._owns_client = client is None
        if client is None:
            try:
                ssl_context = ssl.create_default_context(
                    cafile=str(self.ca_bundle_path)
                )
            except (OSError, ssl.SSLError):
                raise GigaChatConfigurationError(
                    "GigaChat CA bundle could not be loaded"
                ) from None
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(
                    connect=min(10.0, self.timeout_seconds),
                    read=self.timeout_seconds,
                    write=min(20.0, self.timeout_seconds),
                    pool=min(10.0, self.timeout_seconds),
                ),
                limits=httpx.Limits(max_connections=1, max_keepalive_connections=1),
                verify=ssl_context,
                trust_env=False,
            )
        else:
            self._client = client

    @staticmethod
    def _validate_credentials(value: str) -> str:
        if not isinstance(value, str):
            raise GigaChatAuthenticationError("GigaChat OAuth credentials are invalid")
        normalized = value.strip()
        if (
            not normalized
            or len(normalized) > 4096
            or re.fullmatch(r"[A-Za-z0-9+/=_-]+", normalized) is None
        ):
            raise GigaChatAuthenticationError("GigaChat OAuth credentials are invalid")
        return normalized

    @staticmethod
    def _validate_scope(value: str) -> str:
        if not isinstance(value, str):
            raise GigaChatConfigurationError("GigaChat OAuth scope is invalid")
        normalized = value.strip()
        if re.fullmatch(r"GIGACHAT_API_[A-Z0-9_]{2,32}", normalized) is None:
            raise GigaChatConfigurationError("GigaChat OAuth scope is invalid")
        return normalized

    @staticmethod
    def _validate_https_url(value: str, *, field: str) -> str:
        if not isinstance(value, str):
            raise GigaChatConfigurationError(f"GigaChat {field} is invalid")
        normalized = value.strip().rstrip("/")
        parsed = urlsplit(normalized)
        if (
            parsed.scheme != "https"
            or not parsed.netloc
            or parsed.username is not None
            or parsed.password is not None
            or parsed.query
            or parsed.fragment
        ):
            raise GigaChatConfigurationError(f"GigaChat {field} is invalid")
        return normalized

    @staticmethod
    def _validate_user_agent(value: str) -> str:
        if not isinstance(value, str):
            raise GigaChatConfigurationError("GigaChat User-Agent is invalid")
        normalized = value.strip()
        if (
            not normalized
            or len(normalized) > 200
            or any(ord(character) < 33 or ord(character) > 126 for character in normalized)
        ):
            raise GigaChatConfigurationError("GigaChat User-Agent is invalid")
        return normalized

    @property
    def _chat_url(self) -> str:
        return f"{self.base_url}/chat/completions"

    @property
    def _models_url(self) -> str:
        return f"{self.base_url}/models"

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    @classmethod
    def _encode_json(cls, payload: Mapping[str, Any]) -> bytes:
        try:
            encoded = json.dumps(
                payload,
                ensure_ascii=False,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        except (TypeError, ValueError, UnicodeError, RecursionError):
            raise GigaChatApiError("GigaChat API request is not serializable") from None
        if len(encoded) > cls._MAX_REQUEST_BYTES:
            raise GigaChatApiError("GigaChat API request is too large")
        return encoded

    @classmethod
    def _response_json(cls, response: httpx.Response) -> dict[str, Any]:
        if len(response.content) > cls._MAX_RESPONSE_BYTES:
            raise GigaChatApiError("GigaChat API response was too large")
        try:
            body = response.json()
        except (ValueError, RecursionError):
            raise GigaChatApiError("GigaChat API returned invalid JSON") from None
        if not isinstance(body, dict):
            raise GigaChatApiError("GigaChat API returned an invalid envelope")
        return body

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
            if retry_at.tzinfo is None:
                retry_at = retry_at.replace(tzinfo=timezone.utc)
            seconds = (retry_at - datetime.now(timezone.utc)).total_seconds()
        return max(0.0, seconds) if math.isfinite(seconds) else None

    def _retry_delay(self, response: httpx.Response, attempt: int) -> float:
        provider_delay = self._retry_after(response)
        fallback_delay = min(2.0**attempt, self.max_retry_delay_seconds)
        return min(
            provider_delay if provider_delay is not None else fallback_delay,
            self.max_retry_delay_seconds,
        )

    async def _send_with_retries(
        self,
        method: str,
        url: str,
        *,
        headers: Mapping[str, str],
        content: bytes | None,
        endpoint: str,
    ) -> httpx.Response:
        request_bytes = len(content or b"")
        for attempt in range(self.max_retries + 1):
            started = time.monotonic()
            try:
                response = await self._client.request(
                    method,
                    url,
                    headers=dict(headers),
                    content=content,
                )
            except httpx.TimeoutException:
                LOGGER.warning(
                    "GigaChat HTTP request timed out; call_id=%s endpoint=%s "
                    "model=%s attempt=%d request_bytes=%d elapsed=%.3fs",
                    _PLANNER_CALL_ID.get(),
                    endpoint,
                    self.model,
                    attempt + 1,
                    request_bytes,
                    time.monotonic() - started,
                )
                raise GigaChatApiError("GigaChat API request timed out") from None
            except httpx.HTTPError:
                LOGGER.warning(
                    "GigaChat HTTP transport failed; call_id=%s endpoint=%s "
                    "model=%s attempt=%d request_bytes=%d elapsed=%.3fs",
                    _PLANNER_CALL_ID.get(),
                    endpoint,
                    self.model,
                    attempt + 1,
                    request_bytes,
                    time.monotonic() - started,
                )
                raise GigaChatApiError("GigaChat API transport failed") from None

            status = response.status_code
            log_method = LOGGER.info if 200 <= status < 300 else LOGGER.warning
            log_method(
                "GigaChat HTTP response; call_id=%s endpoint=%s model=%s "
                "attempt=%d status=%d request_bytes=%d response_bytes=%d "
                "elapsed=%.3fs",
                _PLANNER_CALL_ID.get(),
                endpoint,
                self.model,
                attempt + 1,
                status,
                request_bytes,
                len(response.content),
                time.monotonic() - started,
            )
            if (
                status in self._RETRYABLE_STATUS_CODES
                and attempt < self.max_retries
            ):
                if status == 429:
                    self._observe_rate_limit()
                await self._sleep(self._retry_delay(response, attempt))
                continue
            return response
        raise GigaChatApiError("GigaChat API request failed")

    @staticmethod
    def _observe_rate_limit() -> GigaChatRateLimitError:
        error = GigaChatRateLimitError("GigaChat API rate limit exceeded")
        if _RATE_LIMIT_OBSERVED.get() is not None:
            _RATE_LIMIT_OBSERVED.set(True)
            _RATE_LIMIT_ERROR_OBSERVED.set(error)
        return error

    @staticmethod
    def _raise_for_status(
        response: httpx.Response, *, oauth: bool = False
    ) -> None:
        status = response.status_code
        if 200 <= status < 300:
            return
        if status in {401, 403} or (oauth and status in {400, 422}):
            raise GigaChatAuthenticationError(
                "GigaChat OAuth credential or access was rejected"
            )
        if status == 402:
            raise GigaChatQuotaError("GigaChat account quota is unavailable")
        if status == 429:
            raise GigaChatApi._observe_rate_limit()
        if status == 404:
            raise GigaChatConfigurationError(
                "Configured GigaChat endpoint or model is unavailable"
            )
        if status in {400, 409, 413, 415, 422}:
            raise GigaChatRequestRejectedError("GigaChat rejected the API request")
        raise GigaChatApiError(f"GigaChat API HTTP status {status}")

    def _oauth_headers(self) -> dict[str, str]:
        request_id = self._uuid4()
        if not isinstance(request_id, uuid.UUID) or request_id.version != 4:
            raise GigaChatConfigurationError("GigaChat RqUID generator is invalid")
        return {
            "Authorization": f"Basic {self._credentials}",
            "RqUID": str(request_id),
            "Accept": "application/json",
            "Content-Type": "application/x-www-form-urlencoded",
            "User-Agent": self.user_agent,
        }

    async def _fetch_access_token(self) -> tuple[str, float]:
        content = urlencode({"scope": self.scope}).encode("ascii")
        response = await self._send_with_retries(
            "POST",
            self.auth_url,
            headers=self._oauth_headers(),
            content=content,
            endpoint="oauth",
        )
        self._raise_for_status(response, oauth=True)
        body = self._response_json(response)
        token = body.get("access_token")
        if (
            not isinstance(token, str)
            or not token
            or len(token) > 16_384
            or any(character.isspace() for character in token)
        ):
            raise GigaChatAuthenticationError(
                "GigaChat OAuth returned an invalid access token"
            )

        now = self._clock()
        expires_at = body.get("expires_at")
        expires_in = body.get("expires_in")
        if isinstance(expires_at, (int, float)) and not isinstance(expires_at, bool):
            expiration = float(expires_at)
            if expiration > 10_000_000_000:
                expiration /= 1000.0
        elif isinstance(expires_in, (int, float)) and not isinstance(expires_in, bool):
            expiration = now + float(expires_in)
        else:
            raise GigaChatAuthenticationError(
                "GigaChat OAuth returned no token expiration"
            )
        if not math.isfinite(expiration) or expiration <= now:
            raise GigaChatAuthenticationError(
                "GigaChat OAuth returned an expired access token"
            )
        return token, expiration

    def _token_is_fresh(self) -> bool:
        return bool(self._access_token) and (
            self._access_token_expires_at - self.token_refresh_skew_seconds
            > self._clock()
        )

    async def _get_access_token(self, *, force_refresh: bool = False) -> str:
        if not force_refresh and self._token_is_fresh():
            assert self._access_token is not None
            return self._access_token
        async with self._token_lock:
            if not force_refresh and self._token_is_fresh():
                assert self._access_token is not None
                return self._access_token
            token, expiration = await self._fetch_access_token()
            self._access_token = token
            self._access_token_expires_at = expiration
            return token

    def _api_headers(self, token: str) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {token}",
            "Accept": "application/json",
            "Content-Type": "application/json",
            "User-Agent": self.user_agent,
        }

    async def _authorized_request(
        self,
        method: str,
        url: str,
        *,
        payload: Mapping[str, Any] | None,
        endpoint: str,
    ) -> httpx.Response:
        content = self._encode_json(payload) if payload is not None else None
        async with self._semaphore:
            token = await self._get_access_token()
            for auth_attempt in range(2):
                response = await self._send_with_retries(
                    method,
                    url,
                    headers=self._api_headers(token),
                    content=content,
                    endpoint=endpoint,
                )
                if response.status_code != 401:
                    self._raise_for_status(response)
                    return response
                if auth_attempt == 0:
                    if token == self._access_token:
                        self._access_token = None
                        self._access_token_expires_at = 0.0
                    token = await self._get_access_token(force_refresh=True)
                    continue
                self._raise_for_status(response)
        raise GigaChatAuthenticationError(
            "GigaChat OAuth credential or access was rejected"
        )

    def _structured_payload(
        self,
        *,
        messages: Sequence[Mapping[str, str]],
        schema: Mapping[str, Any],
        function_name: str,
        function_description: str,
    ) -> dict[str, Any]:
        return {
            "model": self.model,
            "messages": [dict(message) for message in messages],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream": False,
            "functions": [
                {
                    "name": function_name,
                    "description": function_description,
                    "parameters": deepcopy(dict(schema)),
                }
            ],
            "function_call": {"name": function_name},
        }

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
        safe_usage = {
            key: value
            for key in (
                "prompt_tokens",
                "completion_tokens",
                "total_tokens",
                "precached_prompt_tokens",
            )
            if isinstance((value := usage_mapping.get(key)), int)
            and not isinstance(value, bool)
            and value >= 0
        }
        return {
            "actual_model": _safe_diagnostic_token(body.get("model")),
            "finish_reason": _safe_diagnostic_token(choice.get("finish_reason")),
            "usage": safe_usage,
        }

    @staticmethod
    def _function_arguments(
        body: Mapping[str, Any], *, expected_function_name: str
    ) -> tuple[dict[str, Any], str]:
        choices = body.get("choices")
        if not isinstance(choices, list) or len(choices) != 1:
            raise GigaChatApiError("GigaChat API returned no function call")
        choice = choices[0]
        if not isinstance(choice, Mapping):
            raise GigaChatApiError("GigaChat API returned no function call")
        if choice.get("finish_reason") != "function_call":
            raise GigaChatApiError("GigaChat function completion was incomplete")
        message = choice.get("message")
        function_call = (
            message.get("function_call") if isinstance(message, Mapping) else None
        )
        if not isinstance(function_call, Mapping):
            raise GigaChatApiError("GigaChat API returned no function call")
        if function_call.get("name") != expected_function_name:
            raise GigaChatApiError("GigaChat API returned an unexpected function")
        arguments = function_call.get("arguments")
        if not isinstance(arguments, Mapping):
            raise GigaChatApiError("GigaChat API returned invalid function arguments")
        try:
            copied_arguments = deepcopy(dict(arguments))
            output_text = json.dumps(
                copied_arguments,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
                allow_nan=False,
            )
        except (TypeError, ValueError, UnicodeError, RecursionError):
            raise GigaChatApiError(
                "GigaChat API returned invalid function arguments"
            ) from None
        return copied_arguments, output_text

    async def _complete_structured(
        self,
        *,
        messages: Sequence[Mapping[str, str]],
        schema: Mapping[str, Any],
        function_name: str,
        function_description: str,
    ) -> tuple[Any, str]:
        payload = self._structured_payload(
            messages=messages,
            schema=schema,
            function_name=function_name,
            function_description=function_description,
        )
        response = await self._authorized_request(
            "POST",
            self._chat_url,
            payload=payload,
            endpoint="chat_completions",
        )
        body = self._response_json(response)
        diagnostics = self._completion_diagnostics(body)
        metadata_json = json.dumps(
            diagnostics,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        try:
            arguments, output_text = self._function_arguments(
                body,
                expected_function_name=function_name,
            )
        except GigaChatApiError as exc:
            LOGGER.warning(
                "GigaChat function envelope rejected; call_id=%s model=%s "
                "response_bytes=%d metadata=%s error=%s",
                _PLANNER_CALL_ID.get(),
                self.model,
                len(response.content),
                metadata_json,
                str(exc),
            )
            raise
        output_bytes, output_fingerprint = _diagnostic_fingerprint(output_text)
        LOGGER.info(
            "GigaChat function completion received; call_id=%s model=%s "
            "response_bytes=%d output_bytes=%d output_fingerprint=%s metadata=%s",
            _PLANNER_CALL_ID.get(),
            self.model,
            len(response.content),
            output_bytes,
            output_fingerprint,
            metadata_json,
        )
        return arguments, output_text

    async def validate(self) -> None:
        response = await self._authorized_request(
            "GET",
            self._models_url,
            payload=None,
            endpoint="models",
        )
        body = self._response_json(response)
        models = body.get("data")
        if not isinstance(models, list):
            raise GigaChatConfigurationError(
                "GigaChat models endpoint returned an invalid catalog"
            )
        aliases = {
            model.get("id")
            for model in models
            if isinstance(model, Mapping) and isinstance(model.get("id"), str)
        }
        if self.model not in aliases:
            raise GigaChatConfigurationError(
                "Configured GigaChat model is unavailable"
            )

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
            raise GigaChatApiError(str(exc)) from None

        structured, output_text = await self._complete_structured(
            messages=[{"role": "user", "content": prompt}],
            schema=_GIGACHAT_CALENDAR_INTENT_SCHEMA,
            function_name=_EXTRACT_FUNCTION_NAME,
            function_description=(
                "Extract the requested Google Calendar event from the user command."
            ),
        )
        try:
            return normalize_calendar_intent(
                structured,
                expected_timezone=self.timezone,
            )
        except ValueError:
            _log_structured_output_failure(
                provider="GigaChat",
                model=self.model,
                phase="semantic_validation",
                output=output_text,
                reason="semantic_validation_failed",
            )
            raise GigaChatApiError(
                "GigaChat returned an invalid calendar event"
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
        input_kind: str = "text",
        image_observations: Sequence[Mapping[str, Any]] = (),
    ) -> dict[str, Any]:
        try:
            normalized_observations = _validate_planner_input(
                transcript,
                reference_time,
                input_kind=input_kind,
                image_observations=image_observations,
            )
            if not isinstance(application_state, Mapping):
                raise GeminiError("Application state must be an object")
            if isinstance(
                recent_conversation, (str, bytes, bytearray)
            ) or not isinstance(recent_conversation, Sequence):
                raise GeminiError("Recent conversation must be an array")
            native_history = _copy_history_steps(history_steps)
            reuse_image_evidence = bool(
                normalized_observations
            ) and _history_has_image_observations(native_history)
            history_messages = OpenRouterApi._history_messages(native_history)
            current_input = _calendar_operation_input(
                transcript,
                reference_time=reference_time,
                account=account,
                timezone=self.timezone,
                application_state=application_state,
                recent_conversation=recent_conversation,
                input_kind=input_kind,
                image_observations=(
                    () if reuse_image_evidence else normalized_observations
                ),
                image_evidence_in_history=reuse_image_evidence,
            )
            current_text = OpenRouterApi._step_text(current_input)
            if current_text is None:
                raise GeminiError("Calendar planner input is invalid")
        except GeminiError as exc:
            raise GigaChatApiError(str(exc)) from None

        messages = [
            {
                "role": "system",
                "content": CALENDAR_PLANNER_SYSTEM_INSTRUCTION,
            },
            *history_messages,
            {"role": "user", "content": current_text},
        ]
        structured, output_text = await self._complete_structured(
            messages=messages,
            schema=_GIGACHAT_CALENDAR_OPERATION_SCHEMA,
            function_name=_PLAN_FUNCTION_NAME,
            function_description=(
                "Plan the requested Google Calendar read or mutation operations."
            ),
        )
        try:
            normalized = normalize_calendar_operation_plan(
                structured,
                _allowed_event_ids(application_state),
                expected_timezone=self.timezone,
            )
        except ValueError:
            _log_structured_output_failure(
                provider="GigaChat",
                model=self.model,
                phase="semantic_validation",
                output=output_text,
                reason="semantic_validation_failed",
            )
            raise GigaChatApiError(
                "GigaChat returned an invalid calendar plan"
            ) from None

        normalized["_interaction_input"] = deepcopy(current_input)
        normalized["_interaction_steps"] = [
            {
                "type": "model_output",
                "content": [{"type": "text", "text": output_text}],
            }
        ]
        normalized[PLANNER_MODEL_FIELD] = self.model
        return normalized


__all__ = [
    "GigaChatApi",
    "GigaChatApiError",
    "GigaChatAuthenticationError",
    "GigaChatConfigurationError",
    "GigaChatQuotaError",
    "GigaChatRateLimitError",
    "GigaChatRequestRejectedError",
]

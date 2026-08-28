"""Loopback-only, secret-isolated runner for subscription-backed Codex CLI."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
import hashlib
import json
import logging
import os
from pathlib import Path
import re
import secrets
import signal
import tempfile
import time
from typing import Any

from aiohttp import web

from .intent import CALENDAR_INTENT_SCHEMA
from .openrouter import _portable_strict_schema, _strict_operation_wire_schema


LOGGER = logging.getLogger("tg_voice_transcriber_bot.codex_runner")
_MODEL_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,99}")
_MAX_REQUEST_BYTES = 192 * 1024
_MAX_PROMPT_BYTES = 96 * 1024
_MAX_SCHEMA_BYTES = 64 * 1024
_MAX_PROCESS_OUTPUT_BYTES = 1_000_000
_FINGERPRINT_KEY = os.urandom(32)
_RUNNER_TOKEN_RE = re.compile(r"[A-Za-z0-9_-]{32,256}")
_TASK_SCHEMAS: dict[str, Mapping[str, Any]] = {
    "extract_event": _portable_strict_schema(CALENDAR_INTENT_SCHEMA),
    "plan_calendar_actions": _strict_operation_wire_schema(),
}


class RunnerError(RuntimeError):
    def __init__(self, kind: str, message: str, *, status: int = 502) -> None:
        super().__init__(message)
        self.kind = kind
        self.status = status


def _environment_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        parsed = int(value)
    except ValueError:
        raise RuntimeError(f"{name} must be an integer") from None
    if parsed <= 0:
        raise RuntimeError(f"{name} must be positive")
    return parsed


def _fingerprint(value: bytes) -> tuple[int, str]:
    return (
        len(value),
        hashlib.blake2s(
            value,
            key=_FINGERPRINT_KEY,
            digest_size=8,
        ).hexdigest(),
    )


def _classify_failure(stderr: bytes) -> tuple[str, int]:
    text = stderr[-64_000:].decode("utf-8", errors="replace").casefold()
    if any(
        marker in text
        for marker in (
            "not logged in",
            "login required",
            "unauthorized",
            "authentication failed",
            "invalid refresh token",
            "401 unauthorized",
        )
    ):
        return "authentication", 401
    if any(
        marker in text
        for marker in (
            "usage limit",
            "quota exceeded",
            "quota exhausted",
        )
    ):
        return "quota", 429
    if any(
        marker in text
        for marker in (
            "rate limit",
            "too many requests",
            "status 429",
        )
    ):
        return "rate_limit", 429
    if any(
        marker in text
        for marker in (
            "model is not supported",
            "model not found",
            "unknown model",
            "invalid model",
        )
    ):
        return "configuration", 422
    return "execution", 502


def _assert_no_tool_events(stdout: bytes) -> None:
    """Reject a run if the supposedly data-only model attempted any tool."""

    forbidden_markers = (
        "command",
        "computer",
        "file_change",
        "image_generation",
        "mcp",
        "shell",
        "tool",
        "web_search",
    )
    for raw_line in stdout.splitlines():
        try:
            event = json.loads(raw_line)
        except (UnicodeDecodeError, json.JSONDecodeError):
            raise RunnerError(
                "execution", "Codex CLI emitted an invalid event stream"
            ) from None
        if not isinstance(event, Mapping):
            raise RunnerError("execution", "Codex CLI emitted an invalid event")
        event_type = str(event.get("type", "")).casefold()
        item = event.get("item")
        item_type = (
            str(item.get("type", "")).casefold()
            if isinstance(item, Mapping)
            else ""
        )
        combined = f"{event_type} {item_type}"
        if any(marker in combined for marker in forbidden_markers):
            raise RunnerError("execution", "Codex CLI attempted a forbidden tool")


class CodexExecutor:
    def __init__(
        self,
        *,
        binary: Path,
        codex_home: Path,
        timeout_seconds: int,
        model: str,
        reasoning_effort: str,
    ) -> None:
        _validate_model_settings(model, reasoning_effort)
        self.binary = binary
        self.codex_home = codex_home
        self.timeout_seconds = timeout_seconds
        self.model = model
        self.reasoning_effort = reasoning_effort
        self._lock = asyncio.Lock()

    @property
    def auth_path(self) -> Path:
        return self.codex_home / "auth.json"

    def live(self) -> bool:
        try:
            return (
                self.binary.is_file()
                and os.access(self.binary, os.X_OK)
                and self.codex_home.is_dir()
            )
        except OSError:
            return False

    def healthy(self) -> bool:
        try:
            return (
                self.live()
                and self.auth_path.is_file()
                and os.access(self.auth_path, os.R_OK | os.W_OK)
            )
        except OSError:
            return False

    def _process_environment(self) -> dict[str, str]:
        allowed_names = (
            "ALL_PROXY",
            "HTTPS_PROXY",
            "HTTP_PROXY",
            "LANG",
            "LC_ALL",
            "NODE_EXTRA_CA_CERTS",
            "NO_PROXY",
            "PATH",
            "SSL_CERT_DIR",
            "SSL_CERT_FILE",
            "TZ",
        )
        environment = {
            name: value
            for name in allowed_names
            if (value := os.environ.get(name)) is not None
        }
        environment["CODEX_HOME"] = str(self.codex_home)
        environment["HOME"] = "/tmp"
        environment["TMPDIR"] = "/tmp"
        return environment

    @staticmethod
    def _terminate(process: asyncio.subprocess.Process) -> None:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass

    async def _run_process(
        self,
        *arguments: str,
        stdin: bytes | None = None,
        cwd: str | None = None,
    ) -> tuple[bytes, bytes]:
        try:
            process = await asyncio.create_subprocess_exec(
                str(self.binary),
                *arguments,
                cwd=cwd,
                env=self._process_environment(),
                stdin=(
                    asyncio.subprocess.PIPE
                    if stdin is not None
                    else asyncio.subprocess.DEVNULL
                ),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                start_new_session=True,
            )
        except OSError:
            raise RunnerError(
                "configuration",
                "Codex CLI could not be started",
                status=503,
            ) from None
        started = time.monotonic()
        try:
            stdout, stderr = await asyncio.wait_for(
                process.communicate(stdin),
                timeout=self.timeout_seconds,
            )
        except TimeoutError:
            self._terminate(process)
            stdout, stderr = await process.communicate()
            stdout_size, stdout_fingerprint = _fingerprint(stdout)
            stderr_size, stderr_fingerprint = _fingerprint(stderr)
            LOGGER.warning(
                "Codex CLI timed out; elapsed=%.3fs stdout_bytes=%d "
                "stdout_fingerprint=%s stderr_bytes=%d stderr_fingerprint=%s",
                time.monotonic() - started,
                stdout_size,
                stdout_fingerprint,
                stderr_size,
                stderr_fingerprint,
            )
            raise RunnerError("timeout", "Codex CLI timed out", status=504) from None
        except asyncio.CancelledError:
            self._terminate(process)
            await process.communicate()
            raise
        stdout_size, stdout_fingerprint = _fingerprint(stdout)
        stderr_size, stderr_fingerprint = _fingerprint(stderr)
        if (
            stdout_size > _MAX_PROCESS_OUTPUT_BYTES
            or stderr_size > _MAX_PROCESS_OUTPUT_BYTES
        ):
            raise RunnerError("execution", "Codex CLI output was too large")
        if process.returncode != 0:
            kind, status = _classify_failure(stderr)
            LOGGER.warning(
                "Codex CLI failed; exit_code=%d elapsed=%.3fs "
                "stdout_bytes=%d stdout_fingerprint=%s stderr_bytes=%d "
                "stderr_fingerprint=%s failure_kind=%s",
                process.returncode,
                time.monotonic() - started,
                stdout_size,
                stdout_fingerprint,
                stderr_size,
                stderr_fingerprint,
                kind,
            )
            raise RunnerError(kind, "Codex CLI request failed", status=status)
        LOGGER.info(
            "Codex CLI completed; elapsed=%.3fs stdout_bytes=%d "
            "stdout_fingerprint=%s stderr_bytes=%d stderr_fingerprint=%s",
            time.monotonic() - started,
            stdout_size,
            stdout_fingerprint,
            stderr_size,
            stderr_fingerprint,
        )
        return stdout, stderr

    async def validate(self) -> None:
        if not self.healthy():
            raise RunnerError(
                "configuration",
                "Codex CLI or its auth cache is unavailable",
                status=503,
            )
        async with self._lock:
            stdout, stderr = await self._run_process("login", "status")
        status_output = b"\n".join((stdout, stderr))
        if "logged in using chatgpt" not in status_output.decode(
            "utf-8", errors="replace"
        ).casefold():
            raise RunnerError(
                "authentication",
                "Codex CLI is not logged in with ChatGPT",
                status=401,
            )

    async def execute(
        self,
        *,
        task_kind: str,
        prompt: str,
    ) -> Any:
        if not isinstance(task_kind, str):
            raise RunnerError(
                "configuration", "Codex task kind is invalid", status=422
            )
        schema = _TASK_SCHEMAS.get(task_kind)
        if schema is None:
            raise RunnerError(
                "configuration", "Codex task kind is invalid", status=422
            )
        if not self.healthy():
            raise RunnerError(
                "configuration",
                "Codex CLI or its auth cache is unavailable",
                status=503,
            )
        try:
            prompt_bytes = prompt.encode("utf-8")
        except UnicodeError:
            raise RunnerError(
                "configuration", "Codex prompt encoding is invalid", status=422
            ) from None
        try:
            schema_bytes = json.dumps(
                schema,
                ensure_ascii=False,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        except (TypeError, ValueError, UnicodeError):
            raise RunnerError(
                "configuration", "Codex output schema is invalid", status=422
            ) from None
        if not prompt_bytes or len(prompt_bytes) > _MAX_PROMPT_BYTES:
            raise RunnerError(
                "configuration", "Codex prompt size is invalid", status=422
            )
        if not schema_bytes or len(schema_bytes) > _MAX_SCHEMA_BYTES:
            raise RunnerError(
                "configuration", "Codex schema size is invalid", status=422
            )
        prompt_size, prompt_fingerprint = _fingerprint(prompt_bytes)
        LOGGER.info(
            "Codex CLI request accepted; task_kind=%s model=%s "
            "reasoning_effort=%s prompt_bytes=%d prompt_fingerprint=%s "
            "schema_bytes=%d",
            task_kind,
            self.model,
            self.reasoning_effort,
            prompt_size,
            prompt_fingerprint,
            len(schema_bytes),
        )

        async with self._lock:
            with tempfile.TemporaryDirectory(prefix="mk-calendar-codex-") as workdir:
                schema_path = Path(workdir) / "schema.json"
                output_path = Path(workdir) / "output.json"
                schema_path.write_bytes(schema_bytes)
                stdout, _ = await self._run_process(
                    "exec",
                    "--model",
                    self.model,
                    "--sandbox",
                    "read-only",
                    "--ephemeral",
                    "--ignore-user-config",
                    "--ignore-rules",
                    "--strict-config",
                    "--skip-git-repo-check",
                    "--json",
                    "--disable",
                    "shell_tool",
                    "--disable",
                    "unified_exec",
                    "--disable",
                    "apps",
                    "--disable",
                    "remote_plugin",
                    "--disable",
                    "plugins",
                    "--disable",
                    "browser_use",
                    "--disable",
                    "computer_use",
                    "--disable",
                    "image_generation",
                    "--disable",
                    "hooks",
                    "--disable",
                    "memories",
                    "--disable",
                    "multi_agent",
                    "--disable",
                    "multi_agent_v2",
                    "--disable",
                    "goals",
                    "--output-schema",
                    str(schema_path),
                    "--output-last-message",
                    str(output_path),
                    "--color",
                    "never",
                    "-c",
                    f'model_reasoning_effort="{self.reasoning_effort}"',
                    "-c",
                    'approval_policy="never"',
                    "-c",
                    'web_search="disabled"',
                    "-",
                    stdin=prompt_bytes,
                    cwd=workdir,
                )
                _assert_no_tool_events(stdout)
                try:
                    metadata = output_path.stat()
                    if (
                        not output_path.is_file()
                        or metadata.st_size <= 0
                        or metadata.st_size > _MAX_PROCESS_OUTPUT_BYTES
                    ):
                        raise OSError
                    output_bytes = output_path.read_bytes()
                except OSError:
                    raise RunnerError(
                        "execution", "Codex CLI returned no bounded output"
                    ) from None
        try:
            return json.loads(output_bytes)
        except (UnicodeDecodeError, json.JSONDecodeError):
            raise RunnerError(
                "execution", "Codex CLI returned invalid structured JSON"
            ) from None


def _validate_model_settings(model: Any, reasoning_effort: Any) -> None:
    if not isinstance(model, str) or not _MODEL_RE.fullmatch(model):
        raise RunnerError("configuration", "Codex model is invalid", status=422)
    if reasoning_effort not in {"low", "medium", "high", "xhigh", "max"}:
        raise RunnerError(
            "configuration", "Codex reasoning effort is invalid", status=422
        )


def _json_error(error: RunnerError) -> web.Response:
    return web.json_response({"error": error.kind}, status=error.status)


def _read_runner_token() -> str:
    direct = os.environ.get("CODEX_RUNNER_TOKEN")
    token_path = os.environ.get("CODEX_RUNNER_TOKEN_FILE")
    if direct is not None and token_path is not None:
        raise RuntimeError("Codex runner token has conflicting sources")
    if token_path is not None:
        path = Path(token_path)
        try:
            metadata = path.stat()
            if not path.is_file() or metadata.st_size > 1024:
                raise OSError
            token = path.read_text(encoding="utf-8").strip()
        except (OSError, UnicodeError):
            raise RuntimeError("Cannot read Codex runner token file") from None
    else:
        token = (direct or "").strip()
    if not _RUNNER_TOKEN_RE.fullmatch(token):
        raise RuntimeError("Codex runner token is invalid")
    return token


def build_application(executor: CodexExecutor, *, bearer_token: str) -> web.Application:
    application = web.Application(client_max_size=_MAX_REQUEST_BYTES)

    def require_authorization(request: web.Request) -> None:
        supplied = request.headers.get("Authorization", "")
        expected = f"Bearer {bearer_token}"
        if not secrets.compare_digest(supplied, expected):
            raise RunnerError(
                "authentication", "Codex runner authorization failed", status=401
            )

    async def health(_: web.Request) -> web.Response:
        if not executor.live():
            return web.json_response({"status": "unavailable"}, status=503)
        return web.json_response({"status": "ok"})

    async def request_payload(request: web.Request) -> dict[str, Any]:
        try:
            payload = await request.json(loads=json.loads)
        except (json.JSONDecodeError, UnicodeDecodeError, ValueError):
            raise RunnerError("configuration", "Request JSON is invalid", status=400)
        if not isinstance(payload, dict):
            raise RunnerError(
                "configuration", "Request envelope is invalid", status=400
            )
        return payload

    async def validate(request: web.Request) -> web.Response:
        try:
            require_authorization(request)
            payload = await request_payload(request)
            if payload:
                raise RunnerError(
                    "configuration", "Validation request is invalid", status=400
                )
            await executor.validate()
        except RunnerError as error:
            return _json_error(error)
        return web.json_response(
            {
                "status": "ok",
                "model": executor.model,
                "reasoning_effort": executor.reasoning_effort,
            }
        )

    async def execute(request: web.Request) -> web.Response:
        try:
            require_authorization(request)
            payload = await request_payload(request)
            if set(payload) != {"task_kind", "prompt"}:
                raise RunnerError(
                    "configuration", "Execution request is invalid", status=400
                )
            prompt = payload["prompt"]
            task_kind = payload["task_kind"]
            if not isinstance(prompt, str):
                raise RunnerError(
                    "configuration", "Execution payload is invalid", status=400
                )
            if not isinstance(task_kind, str) or task_kind not in _TASK_SCHEMAS:
                raise RunnerError(
                    "configuration", "Codex task kind is invalid", status=422
                )
            output = await executor.execute(
                task_kind=task_kind,
                prompt=prompt,
            )
        except RunnerError as error:
            return _json_error(error)
        return web.json_response({"output": output})

    application.router.add_get("/healthz", health)
    application.router.add_post("/v1/validate", validate)
    application.router.add_post("/v1/execute", execute)
    return application


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    logging.getLogger("aiohttp.access").setLevel(logging.WARNING)
    binary = Path(os.environ.get("CODEX_BINARY_PATH", "/usr/local/bin/codex"))
    codex_home = Path(os.environ.get("CODEX_HOME", "/var/lib/codex"))
    timeout_seconds = _environment_int("CODEX_RUNNER_TIMEOUT_SECONDS", 50)
    bearer_token = _read_runner_token()
    model = os.environ.get("CODEX_MODEL", "gpt-5.6-sol").strip()
    reasoning_effort = os.environ.get("CODEX_REASONING_EFFORT", "medium").strip().lower()
    port = _environment_int("CODEX_RUNNER_LISTEN_PORT", 8091)
    if port > 65535:
        raise RuntimeError("CODEX_RUNNER_LISTEN_PORT is invalid")
    host = os.environ.get("CODEX_RUNNER_LISTEN_HOST", "127.0.0.1")
    if host not in {"127.0.0.1", "::1"}:
        raise RuntimeError("Codex runner must listen on loopback")
    executor = CodexExecutor(
        binary=binary,
        codex_home=codex_home,
        timeout_seconds=timeout_seconds,
        model=model,
        reasoning_effort=reasoning_effort,
    )
    web.run_app(
        build_application(executor, bearer_token=bearer_token),
        host=host,
        port=port,
        access_log=None,
        print=None,
    )


if __name__ == "__main__":  # pragma: no cover
    main()

import asyncio
import json
from pathlib import Path

import pytest
from aiohttp import ClientSession, web

from tg_voice_transcriber_bot import codex_runner
from tg_voice_transcriber_bot.codex_runner import (
    CodexExecutor,
    RunnerError,
    build_application,
    _assert_no_tool_events,
    _classify_failure,
)


CALENDAR_RUNNER_EXTRACT_OUTPUT = {
    "action": "create",
    "events": [
        {
            "title": "Позвонить врачу",
            "start_at": "2026-08-29T10:00:00+03:00",
            "end_at": "2026-08-29T10:30:00+03:00",
            "all_day": False,
            "timezone": "Europe/Moscow",
            "location": None,
            "description": None,
            "recurrence_rrule": None,
        }
    ],
    "clarification_question": None,
    "confidence": 1.0,
}


def _ready_executor(tmp_path, *, timeout_seconds=1):
    binary = tmp_path / "codex"
    binary.write_text("#!/bin/sh\n", encoding="utf-8")
    binary.chmod(0o700)
    codex_home = tmp_path / "codex-home"
    codex_home.mkdir()
    auth_path = codex_home / "auth.json"
    auth_path.write_text('{"auth_mode":"chatgpt"}', encoding="utf-8")
    auth_path.chmod(0o600)
    return CodexExecutor(
        binary=binary,
        codex_home=codex_home,
        timeout_seconds=timeout_seconds,
        model="gpt-5.6-luna",
        reasoning_effort="high",
    )


def _assert_strict_object_schemas(node):
    if isinstance(node, dict):
        if node.get("type") == "object":
            properties = node.get("properties")
            assert isinstance(properties, dict)
            assert node.get("additionalProperties") is False
            assert set(node.get("required", [])) == set(properties)
        for value in node.values():
            _assert_strict_object_schemas(value)
    elif isinstance(node, list):
        for value in node:
            _assert_strict_object_schemas(value)


class FakeProcess:
    def __init__(self, *, stdout=b"", stderr=b"", returncode=0, pid=4321):
        self.stdout = stdout
        self.stderr = stderr
        self.returncode = returncode
        self.pid = pid
        self.stdin_values = []

    async def communicate(self, stdin=None):
        self.stdin_values.append(stdin)
        return self.stdout, self.stderr


def _safe_event_stream():
    events = [
        {"type": "thread.started", "thread_id": "thread-1"},
        {"type": "turn.started"},
        {
            "type": "item.completed",
            "item": {
                "id": "item-1",
                "type": "agent_message",
                "text": '{"action":"ignore"}',
            },
        },
        {
            "type": "turn.completed",
            "usage": {"input_tokens": 10, "output_tokens": 4},
        },
    ]
    return b"\n".join(json.dumps(event).encode() for event in events) + b"\n"


def test_runner_rejects_non_url_safe_token(monkeypatch):
    monkeypatch.setenv("CODEX_RUNNER_TOKEN", "я" * 32)

    with pytest.raises(RuntimeError, match="token is invalid"):
        codex_runner._read_runner_token()


def test_health_requires_executable_binary_and_writable_auth(tmp_path):
    executor = _ready_executor(tmp_path)
    assert executor.healthy() is True

    executor.auth_path.unlink()
    assert executor.healthy() is False
    executor.auth_path.write_text("{}", encoding="utf-8")
    assert executor.healthy() is True

    executor.binary.chmod(0o600)
    assert executor.healthy() is False


def test_validate_checks_chatgpt_login_status(tmp_path, monkeypatch):
    executor = _ready_executor(tmp_path)
    observed = []

    async def fake_run(*arguments, **kwargs):
        observed.append((arguments, kwargs))
        return b"Logged in using ChatGPT\n", b""

    monkeypatch.setattr(executor, "_run_process", fake_run)
    asyncio.run(executor.validate())

    assert observed == [(('login', 'status'), {})]


def test_validate_accepts_chatgpt_login_status_from_stderr(tmp_path, monkeypatch):
    executor = _ready_executor(tmp_path)

    async def fake_run(*_arguments, **_kwargs):
        return b"", b"Logged in using ChatGPT\n"

    monkeypatch.setattr(executor, "_run_process", fake_run)
    asyncio.run(executor.validate())


def test_validate_rejects_non_chatgpt_auth_status(tmp_path, monkeypatch):
    executor = _ready_executor(tmp_path)

    async def fake_run(*_arguments, **_kwargs):
        return b"Logged in using an API key\n", b""

    monkeypatch.setattr(executor, "_run_process", fake_run)

    with pytest.raises(RunnerError) as caught:
        asyncio.run(executor.validate())
    assert caught.value.kind == "authentication"
    assert caught.value.status == 401


def test_http_api_requires_bearer_and_keeps_model_and_schema_server_side():
    token = "runner-http-token-0123456789abcdef"

    class FakeExecutor:
        model = "gpt-5.6-luna"
        reasoning_effort = "high"

        def __init__(self):
            self.validation_calls = 0
            self.execution_calls = []

        def live(self):
            return True

        async def validate(self):
            self.validation_calls += 1

        async def execute(self, *, task_kind, prompt):
            self.execution_calls.append((task_kind, prompt))
            return {"action": "ignore"}

    async def scenario():
        executor = FakeExecutor()
        application = build_application(executor, bearer_token=token)
        runner = web.AppRunner(application)
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        server = site._server
        assert server is not None
        port = server.sockets[0].getsockname()[1]
        base_url = f"http://127.0.0.1:{port}"
        try:
            async with ClientSession() as client:
                response = await client.get(f"{base_url}/healthz")
                assert response.status == 200
                assert await response.json() == {"status": "ok"}

                response = await client.post(f"{base_url}/v1/validate", json={})
                assert response.status == 401
                assert await response.json() == {"error": "authentication"}

                headers = {"Authorization": f"Bearer {token}"}
                response = await client.post(
                    f"{base_url}/v1/validate", json={}, headers=headers
                )
                assert response.status == 200
                assert await response.json() == {
                    "status": "ok",
                    "model": "gpt-5.6-luna",
                    "reasoning_effort": "high",
                }

                response = await client.post(
                    f"{base_url}/v1/execute",
                    json={
                        "task_kind": "extract_event",
                        "prompt": "payload",
                        "schema": {"type": "object"},
                    },
                    headers=headers,
                )
                assert response.status == 400
                assert await response.json() == {"error": "configuration"}

                response = await client.post(
                    f"{base_url}/v1/execute",
                    json={"task_kind": "arbitrary", "prompt": "payload"},
                    headers=headers,
                )
                assert response.status == 422
                assert await response.json() == {"error": "configuration"}

                response = await client.post(
                    f"{base_url}/v1/execute",
                    json={
                        "task_kind": "extract_event",
                        "prompt": "payload",
                    },
                    headers=headers,
                )
                assert response.status == 200
                assert await response.json() == {
                    "output": {"action": "ignore"}
                }
        finally:
            await runner.cleanup()
        return executor

    executor = asyncio.run(scenario())
    assert executor.validation_calls == 1
    assert executor.execution_calls == [("extract_event", "payload")]


def test_execute_uses_stdin_schema_isolated_cwd_and_no_tool_flags(
    tmp_path, monkeypatch
):
    executor = _ready_executor(tmp_path)
    observed = {}
    process = FakeProcess(stdout=_safe_event_stream())
    output = {"action": "ignore", "operations": [], "confidence": 1.0}

    async def fake_create(*arguments, **kwargs):
        observed["arguments"] = arguments
        observed["kwargs"] = kwargs
        schema_path = Path(
            arguments[arguments.index("--output-schema") + 1]
        )
        output_path = Path(
            arguments[arguments.index("--output-last-message") + 1]
        )
        observed["schema"] = json.loads(schema_path.read_text(encoding="utf-8"))
        observed["cwd"] = kwargs["cwd"]
        output_path.write_text(json.dumps(output), encoding="utf-8")
        return process

    monkeypatch.setenv("OPENAI_API_KEY", "PRIVATE-OPENAI-KEY")
    monkeypatch.setenv("CODEX_API_KEY", "PRIVATE-CODEX-KEY")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://private.invalid")
    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_create)

    prompt = "Calendar payload with $(touch /tmp/never-run)"
    result = asyncio.run(
        executor.execute(
            task_kind="plan_calendar_actions",
            prompt=prompt,
        )
    )

    arguments = observed["arguments"]
    kwargs = observed["kwargs"]
    assert arguments[0] == str(executor.binary)
    assert arguments[1:3] == ("exec", "--model")
    assert "gpt-5.6-luna" in arguments
    assert arguments[-1] == "-"
    assert ("--sandbox", "read-only") == (
        arguments[arguments.index("--sandbox")],
        arguments[arguments.index("--sandbox") + 1],
    )
    for flag in (
        "--ephemeral",
        "--ignore-user-config",
        "--ignore-rules",
        "--strict-config",
        "--skip-git-repo-check",
        "--json",
    ):
        assert flag in arguments
    disabled = {
        arguments[index + 1]
        for index, value in enumerate(arguments)
        if value == "--disable"
    }
    assert {
        "shell_tool",
        "unified_exec",
        "apps",
        "remote_plugin",
        "plugins",
        "browser_use",
        "computer_use",
        "image_generation",
        "hooks",
        "memories",
        "multi_agent",
        "multi_agent_v2",
        "goals",
    } <= disabled
    assert kwargs["start_new_session"] is True
    assert kwargs["cwd"] == observed["cwd"]
    assert not Path(observed["cwd"]).exists()
    assert kwargs["env"]["CODEX_HOME"] == str(executor.codex_home)
    assert kwargs["env"]["HOME"] == "/tmp"
    assert "OPENAI_API_KEY" not in kwargs["env"]
    assert "CODEX_API_KEY" not in kwargs["env"]
    assert "OPENAI_BASE_URL" not in kwargs["env"]
    assert process.stdin_values == [prompt.encode("utf-8")]
    assert observed["schema"]["properties"]["action"]["type"] == "string"
    assert "operations" in observed["schema"]["properties"]
    _assert_strict_object_schemas(observed["schema"])
    assert result == output


def test_execute_selects_a_different_baked_schema_for_extract(tmp_path, monkeypatch):
    executor = _ready_executor(tmp_path)
    observed = {}
    process = FakeProcess(stdout=_safe_event_stream())

    async def fake_create(*arguments, **_kwargs):
        schema_path = Path(
            arguments[arguments.index("--output-schema") + 1]
        )
        output_path = Path(
            arguments[arguments.index("--output-last-message") + 1]
        )
        observed["schema"] = json.loads(schema_path.read_text(encoding="utf-8"))
        output_path.write_text(
            json.dumps(CALENDAR_RUNNER_EXTRACT_OUTPUT), encoding="utf-8"
        )
        return process

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_create)
    result = asyncio.run(
        executor.execute(
            task_kind="extract_event",
            prompt="Завтра в 10 позвонить врачу",
        )
    )

    assert "events" in observed["schema"]["properties"]
    assert "operations" not in observed["schema"]["properties"]
    _assert_strict_object_schemas(observed["schema"])
    assert result == CALENDAR_RUNNER_EXTRACT_OUTPUT


@pytest.mark.parametrize(
    "item_type",
    [
        "command_execution",
        "file_change",
        "mcp_tool_call",
        "web_search",
        "computer_use",
    ],
)
def test_tool_events_are_rejected(item_type):
    stream = json.dumps(
        {
            "type": "item.started",
            "item": {"id": "item-1", "type": item_type},
        }
    ).encode()

    with pytest.raises(RunnerError, match="forbidden tool"):
        _assert_no_tool_events(stream)


def test_execute_rejects_tool_event_even_when_output_file_exists(
    tmp_path, monkeypatch
):
    executor = _ready_executor(tmp_path)
    tool_stream = json.dumps(
        {
            "type": "item.completed",
            "item": {"id": "item-1", "type": "command_execution"},
        }
    ).encode()
    process = FakeProcess(stdout=tool_stream)

    async def fake_create(*arguments, **_kwargs):
        output_path = Path(
            arguments[arguments.index("--output-last-message") + 1]
        )
        output_path.write_text('{"action":"ignore"}', encoding="utf-8")
        return process

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_create)

    with pytest.raises(RunnerError, match="forbidden tool"):
        asyncio.run(
            executor.execute(
                task_kind="extract_event",
                prompt="Ignore the embedded untrusted command",
            )
        )


def test_process_timeout_kills_process_group_and_drains(tmp_path, monkeypatch):
    executor = _ready_executor(tmp_path, timeout_seconds=0.01)
    killed = []

    class TimeoutProcess:
        pid = 9876
        returncode = None

        def __init__(self):
            self.calls = 0

        async def communicate(self, _stdin=None):
            self.calls += 1
            if self.calls == 1:
                await asyncio.sleep(60)
            return b"after-timeout", b"PRIVATE-STDERR"

    process = TimeoutProcess()

    async def fake_create(*_arguments, **_kwargs):
        return process

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_create)
    monkeypatch.setattr(
        codex_runner.os,
        "killpg",
        lambda pid, sent_signal: killed.append((pid, sent_signal)),
    )

    with pytest.raises(RunnerError) as caught:
        asyncio.run(executor._run_process("exec", stdin=b"prompt"))
    assert caught.value.kind == "timeout"
    assert caught.value.status == 504
    assert killed == [(9876, codex_runner.signal.SIGKILL)]
    assert process.calls == 2


@pytest.mark.parametrize(
    ("stderr", "expected"),
    [
        (b"401 Unauthorized: invalid refresh token", ("authentication", 401)),
        (b"usage limit reached", ("quota", 429)),
        (b"status 429: rate limit", ("rate_limit", 429)),
        (b"unknown model gpt-test", ("configuration", 422)),
        (b"upstream disconnected", ("execution", 502)),
    ],
)
def test_cli_failure_classification_uses_only_trusted_stderr(stderr, expected):
    assert _classify_failure(stderr) == expected


def test_invalid_jsonl_event_is_rejected_without_echoing_it():
    private_value = b"PRIVATE-MODEL-OUTPUT"
    with pytest.raises(RunnerError) as caught:
        _assert_no_tool_events(private_value)
    assert private_value.decode() not in str(caught.value)

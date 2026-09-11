from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import socket
import threading
import time
import pytest

from repo_agent.models import FinalAnswer, ToolCall, ToolResult
from repo_agent.openai_provider import (
    OpenAIConfig,
    OpenAIConfigurationError,
    OpenAIHTTPClient,
    OpenAIHTTPError,
    OpenAIProtocolError,
    OpenAIProvider,
    OpenAIResponseTooLargeError,
    OpenAITimeoutError,
    OpenAITransportError,
    RemoteModelNotAllowedError,
    strict_response_tools,
)
from repo_agent.tools import TOOL_DEFINITIONS


ResponseSpec = tuple[int, Mapping[str, str], bytes]
Responder = Callable[[str, Mapping[str, str], bytes, int], ResponseSpec]


@contextmanager
def fake_openai_server(
    responder: Responder,
) -> Iterator[tuple[str, list[dict[str, object]]]]:
    requests: list[dict[str, object]] = []
    server_errors: list[BaseException] = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:  # noqa: N802 - stdlib handler API
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length)
            headers = {key.casefold(): value for key, value in self.headers.items()}
            index = len(requests)
            requests.append({"path": self.path, "headers": headers, "body": raw})
            try:
                status, response_headers, response_body = responder(
                    self.path,
                    headers,
                    raw,
                    index,
                )
            except BaseException as exc:
                server_errors.append(exc)
                status, response_headers, response_body = json_response(
                    {"error": {"code": "fake_server_assertion"}},
                    status=500,
                )
            try:
                self.send_response(status)
                for key, value in response_headers.items():
                    self.send_header(key, value)
                self.end_headers()
                self.wfile.write(response_body)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def log_message(self, format: str, *args: object) -> None:
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.daemon_threads = True
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        host, port = server.server_address
        yield f"http://{host}:{port}", requests
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
        if server_errors:
            raise server_errors[0]


def json_response(payload: object, status: int = 200) -> ResponseSpec:
    return (
        status,
        {"Content-Type": "application/json"},
        json.dumps(payload, separators=(",", ":")).encode("utf-8"),
    )


def config(base_url: str, **overrides: object) -> OpenAIConfig:
    values: dict[str, object] = {
        "api_key": "fake-provider-key",
        "base_url": base_url,
        "model": "fake-model",
        "timeout_seconds": 1.0,
    }
    values.update(overrides)
    return OpenAIConfig(**values)  # type: ignore[arg-type]


def test_config_loads_only_explicit_contract_and_redacts_key() -> None:
    api_key = "fake-key-that-must-not-appear"
    loaded = OpenAIConfig.from_env(
        {
            "REPO_AGENT_API_KEY": api_key,
            "REPO_AGENT_BASE_URL": "http://localhost:8123/v1/",
            "REPO_AGENT_MODEL": "test-model",
            "OPENAI_API_KEY": "must-be-ignored",
        }
    )

    assert loaded.base_url == "http://localhost:8123/v1"
    assert loaded.model == "test-model"
    assert api_key not in repr(loaded)
    assert api_key not in str(loaded)
    assert "must-be-ignored" not in repr(loaded)


def test_responses_usage_is_aggregated_across_rounds() -> None:
    def responder(path, headers, raw, index):
        del headers, raw
        assert path == "/v1/responses"
        if index == 0:
            return json_response(
                {
                    "output": [
                        {
                            "type": "function_call",
                            "call_id": "usage-call",
                            "name": "git_status",
                            "arguments": "{}",
                        }
                    ],
                    "usage": {
                        "input_tokens": 10,
                        "input_tokens_details": {"cached_tokens": 3},
                        "output_tokens": 4,
                        "total_tokens": 14,
                    },
                }
            )
        return json_response(
            {
                "output": [
                    {
                        "type": "message",
                        "content": [{"type": "output_text", "text": "done"}],
                    }
                ],
                "usage": {
                    "input_tokens": 20,
                    "output_tokens": 3,
                    "total_tokens": 23,
                },
            }
        )

    with fake_openai_server(responder) as (base_url, _):
        provider = OpenAIProvider(config(f"{base_url}/v1"))
        call = provider.next_step("inspect", ())
        assert isinstance(call, ToolCall)
        final = provider.next_step(
            "inspect",
            (ToolResult(call.id, call.name, True, "clean"),),
        )

    assert isinstance(final, FinalAnswer)
    assert provider.usage.response_count == 2
    assert provider.usage.reported_response_count == 2
    assert provider.usage.input_tokens == 30
    assert provider.usage.cached_input_tokens == 3
    assert provider.usage.output_tokens == 7
    assert provider.usage.total_tokens == 37
    assert provider.usage.complete is True


def test_missing_usage_is_not_reported_as_zero_tokens() -> None:
    def responder(path, headers, raw, index):
        del path, headers, raw, index
        return json_response(
            {
                "output": [
                    {
                        "type": "message",
                        "content": [{"type": "output_text", "text": "done"}],
                    }
                ]
            }
        )

    with fake_openai_server(responder) as (base_url, _):
        provider = OpenAIProvider(config(f"{base_url}/v1"))
        provider.next_step("inspect", ())

    assert provider.usage.response_count == 1
    assert provider.usage.reported_response_count == 0
    assert provider.usage.cached_input_tokens is None
    assert provider.usage.total_tokens is None
    assert provider.usage.complete is False


def test_missing_environment_names_are_reported_without_values() -> None:
    with pytest.raises(OpenAIConfigurationError) as raised:
        OpenAIConfig.from_env({"REPO_AGENT_API_KEY": "fake-only-key"})

    message = str(raised.value)
    assert "REPO_AGENT_BASE_URL" in message
    assert "REPO_AGENT_MODEL" in message
    assert "fake-only-key" not in message


@pytest.mark.parametrize(
    "url",
    [
        "https://api.example.invalid/v1",
        "http://api.example.invalid/v1",
    ],
)
def test_remote_endpoints_require_explicit_opt_in(url: str) -> None:
    with pytest.raises(RemoteModelNotAllowedError, match="allow_remote_model"):
        config(url)


def test_remote_endpoints_still_require_https_after_opt_in() -> None:
    with pytest.raises(OpenAIConfigurationError, match="HTTPS"):
        config("http://api.example.invalid/v1", allow_remote_model=True)

    allowed = config(
        "https://api.example.invalid/v1/",
        allow_remote_model=True,
    )
    assert allowed.base_url == "https://api.example.invalid/v1"


@pytest.mark.parametrize(
    "url",
    [
        "file:///tmp/model",
        "http://user:password@localhost:8000/v1",
        "http://localhost:8000/v1?token=fake",
        "http://localhost:8000/v1#fragment",
        "http://localhost:bad/v1",
        "http://localhost:8000/v1/../other",
    ],
)
def test_malformed_or_credential_bearing_urls_are_rejected(url: str) -> None:
    with pytest.raises(OpenAIConfigurationError):
        config(url)


def test_strict_tools_require_every_property_and_make_optionals_nullable() -> None:
    tools = strict_response_tools(TOOL_DEFINITIONS)
    search = next(tool for tool in tools if tool["name"] == "search")
    schema = search["parameters"]
    assert isinstance(schema, dict)

    assert search["strict"] is True
    assert schema["additionalProperties"] is False
    assert schema["required"] == ["pattern", "glob", "fixed_strings"]
    properties = schema["properties"]
    assert properties["pattern"]["type"] == "string"
    assert properties["glob"]["type"] == ["string", "null"]
    assert properties["fixed_strings"]["type"] == ["boolean", "null"]
    assert "default" not in properties["fixed_strings"]

    original = next(item for item in TOOL_DEFINITIONS if item.name == "search")
    assert original.to_provider_dict()["strict"] is False
    assert original.input_schema["required"] == ["pattern"]


def test_responses_provider_returns_native_tool_call_and_drops_optional_nulls() -> None:
    def responder(
        path: str,
        headers: Mapping[str, str],
        raw: bytes,
        index: int,
    ) -> ResponseSpec:
        assert path == "/v1/responses"
        assert index == 0
        assert headers["authorization"] == "Bearer fake-provider-key"
        assert headers["idempotency-key"] == "run-1:inspect:0:responses:0"
        request = json.loads(raw)
        assert request["model"] == "fake-model"
        assert request["parallel_tool_calls"] is False
        assert request["store"] is False
        assert all(tool["strict"] is True for tool in request["tools"])
        return json_response(
            {
                "output": [
                    {
                        "type": "function_call",
                        "call_id": "call-responses-1",
                        "name": "search",
                        "arguments": json.dumps(
                            {
                                "pattern": "TODO",
                                "glob": None,
                                "fixed_strings": None,
                            }
                        ),
                    }
                ]
            }
        )

    with fake_openai_server(responder) as (base_url, requests):
        provider = OpenAIProvider(
            config(f"{base_url}/v1"),
            idempotency_key="run-1:inspect:0",
        )
        decision = provider.next_step("find TODO markers", ())

    assert decision == ToolCall(
        "call-responses-1",
        "search",
        {"pattern": "TODO"},
    )
    assert provider.api_kind == "responses"
    assert len(requests) == 1


def test_idempotency_key_rejects_header_injection_without_echoing_value() -> None:
    bad_value = "run-1:inspect:0\r\nX-Leak: fake-provider-key"
    with pytest.raises(OpenAIConfigurationError) as raised:
        OpenAIProvider(
            config("http://127.0.0.1:8123"),
            idempotency_key=bad_value,
        )

    assert bad_value not in str(raised.value)
    assert "fake-provider-key" not in str(raised.value)


def test_responses_provider_round_trips_reasoning_and_function_output() -> None:
    reasoning = {
        "type": "reasoning",
        "id": "reasoning-provider-1",
        "summary": [],
    }
    function_call = {
        "type": "function_call",
        "id": "function-provider-1",
        "call_id": "exact-provider-call-id",
        "name": "git_status",
        "arguments": "{}",
        "status": "completed",
    }

    def responder(
        path: str,
        headers: Mapping[str, str],
        raw: bytes,
        index: int,
    ) -> ResponseSpec:
        assert path == "/responses"
        assert headers["idempotency-key"] == (
            f"run-a:inspect:1:responses:{index}"
        )
        request = json.loads(raw)
        if index == 0:
            assert request["input"][0]["role"] == "user"
            return json_response({"output": [reasoning, function_call]})

        assert index == 1
        continued = request["input"]
        assert continued[1] == reasoning
        assert continued[2] == function_call
        assert continued[3]["type"] == "function_call_output"
        assert continued[3]["call_id"] == "exact-provider-call-id"
        tool_result = json.loads(continued[3]["output"])
        assert tool_result == {
            "ok": True,
            "output": "## main",
            "error": None,
            "exit_code": 0,
            "truncated": False,
        }
        return json_response(
            {
                "output": [
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [
                            {"type": "output_text", "text": "Status inspected."}
                        ],
                    }
                ]
            }
        )

    with fake_openai_server(responder) as (base_url, requests):
        provider = OpenAIProvider(
            config(base_url),
            idempotency_key="run-a:inspect:1",
        )
        call = provider.next_step("inspect status", ())
        assert call == ToolCall("exact-provider-call-id", "git_status", {})
        result = ToolResult(
            call_id=call.id,
            name=call.name,
            ok=True,
            output="## main",
            exit_code=0,
        )
        final = provider.next_step("inspect status", (result,))

    assert final == FinalAnswer("Status inspected.")
    assert len(requests) == 2


def test_provider_rejects_a_result_for_a_different_call_id_before_http() -> None:
    def responder(*_: object) -> ResponseSpec:
        return json_response(
            {
                "output": [
                    {
                        "type": "function_call",
                        "call_id": "expected-call",
                        "name": "git_status",
                        "arguments": "{}",
                    }
                ]
            }
        )

    with fake_openai_server(responder) as (base_url, requests):
        provider = OpenAIProvider(config(base_url))
        provider.next_step("inspect", ())
        mismatched = ToolResult("wrong-call", "git_status", True, "ok")
        with pytest.raises(OpenAIProtocolError, match="pending call"):
            provider.next_step("inspect", (mismatched,))

    assert len(requests) == 1


def test_provider_does_not_repeat_a_pending_call_without_its_result() -> None:
    def responder(*_: object) -> ResponseSpec:
        return json_response(
            {
                "output": [
                    {
                        "type": "function_call",
                        "call_id": "still-pending",
                        "name": "git_status",
                        "arguments": "{}",
                    }
                ]
            }
        )

    with fake_openai_server(responder) as (base_url, requests):
        provider = OpenAIProvider(config(base_url))
        provider.next_step("inspect", ())
        with pytest.raises(OpenAIProtocolError, match="no matching result"):
            provider.next_step("inspect", ())

    assert len(requests) == 1


def test_same_logical_round_reuses_derived_idempotency_key_on_retry() -> None:
    def responder(
        path: str,
        headers: Mapping[str, str],
        raw: bytes,
        index: int,
    ) -> ResponseSpec:
        if index == 0:
            return (200, {"Content-Type": "application/json"}, b"not-json")
        return json_response(
            {
                "output": [
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [
                            {"type": "output_text", "text": "retry succeeded"}
                        ],
                    }
                ]
            }
        )

    with fake_openai_server(responder) as (base_url, requests):
        provider = OpenAIProvider(
            config(base_url),
            idempotency_key="run-retry:plan:2",
        )
        with pytest.raises(OpenAIProtocolError):
            provider.next_step("plan change", ())
        final = provider.next_step("plan change", ())

    assert final == FinalAnswer("retry succeeded")
    assert [
        request["headers"]["idempotency-key"] for request in requests
    ] == [
        "run-retry:plan:2:responses:0",
        "run-retry:plan:2:responses:0",
    ]


def test_provider_resets_conversation_when_task_changes() -> None:
    def responder(
        path: str,
        headers: Mapping[str, str],
        raw: bytes,
        index: int,
    ) -> ResponseSpec:
        request = json.loads(raw)
        assert len(request["input"]) == 1
        prompt = request["input"][0]["content"][0]["text"]
        expected_task = "first task" if index == 0 else "second task"
        assert expected_task in prompt
        return json_response(
            {
                "output": [
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [
                            {"type": "output_text", "text": f"done {index}"}
                        ],
                    }
                ]
            }
        )

    with fake_openai_server(responder) as (base_url, requests):
        provider = OpenAIProvider(config(base_url))
        assert provider.next_step("first task", ()) == FinalAnswer("done 0")
        assert provider.next_step("second task", ()) == FinalAnswer("done 1")

    assert len(requests) == 2


def test_responses_provider_extracts_final_message_text() -> None:
    def responder(*_: object) -> ResponseSpec:
        return json_response(
            {
                "output": [
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [
                            {"type": "output_text", "text": "Inspection complete."}
                        ],
                    }
                ]
            }
        )

    with fake_openai_server(responder) as (base_url, _):
        provider = OpenAIProvider(config(base_url))
        decision = provider.next_step("inspect", ())

    assert decision == FinalAnswer("Inspection complete.")


def test_provider_falls_back_to_chat_only_for_unsupported_responses_endpoint() -> None:
    def responder(
        path: str,
        headers: Mapping[str, str],
        raw: bytes,
        index: int,
    ) -> ResponseSpec:
        expected_keys = [
            "run-chat:implement:0:responses:0",
            "run-chat:implement:0:chat:0",
            "run-chat:implement:0:chat:1",
        ]
        assert headers["idempotency-key"] == expected_keys[index]
        if index == 0:
            assert path == "/responses"
            return json_response({"error": {"code": "unknown_endpoint"}}, 404)
        assert path == "/chat/completions"
        request = json.loads(raw)
        assert request["parallel_tool_calls"] is False
        search = next(
            tool for tool in request["tools"] if tool["function"]["name"] == "search"
        )
        assert search["function"]["strict"] is True
        if index == 2:
            assistant = request["messages"][-2]
            tool_result = request["messages"][-1]
            assert assistant["role"] == "assistant"
            assert assistant["tool_calls"][0]["id"] == "call-chat-1"
            assert tool_result["role"] == "tool"
            assert tool_result["tool_call_id"] == "call-chat-1"
            assert json.loads(tool_result["content"])["output"] == "1 match"
            return json_response(
                {
                    "choices": [
                        {
                            "message": {
                                "role": "assistant",
                                "content": "FIXME inspection complete.",
                            }
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 20,
                        "prompt_tokens_details": {"cached_tokens": 6},
                        "completion_tokens": 3,
                        "total_tokens": 23,
                    },
                }
            )
        return json_response(
            {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "call-chat-1",
                                    "type": "function",
                                    "function": {
                                        "name": "search",
                                        "arguments": json.dumps(
                                            {
                                                "pattern": "FIXME",
                                                "glob": "*.py",
                                                "fixed_strings": False,
                                            }
                                        ),
                                    },
                                }
                            ],
                        }
                    }
                ],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 4,
                    "total_tokens": 14,
                },
            }
        )

    with fake_openai_server(responder) as (base_url, requests):
        provider = OpenAIProvider(
            config(base_url),
            idempotency_key="run-chat:implement:0",
        )
        decision = provider.next_step("find FIXME markers", ())
        assert isinstance(decision, ToolCall)
        result = ToolResult(
            decision.id,
            decision.name,
            True,
            "1 match",
            exit_code=0,
        )
        final = provider.next_step("find FIXME markers", (result,))

    assert decision == ToolCall(
        "call-chat-1",
        "search",
        {"pattern": "FIXME", "glob": "*.py", "fixed_strings": False},
    )
    assert provider.api_kind == "chat_completions"
    assert provider.usage.input_tokens == 30
    assert provider.usage.cached_input_tokens == 6
    assert provider.usage.output_tokens == 7
    assert provider.usage.total_tokens == 37
    assert provider.usage.complete is True
    assert final == FinalAnswer("FIXME inspection complete.")
    assert [request["path"] for request in requests] == [
        "/responses",
        "/chat/completions",
        "/chat/completions",
    ]


@pytest.mark.parametrize("status", [400, 401, 403, 429, 500])
def test_provider_does_not_fallback_for_non_capability_http_errors(status: int) -> None:
    def responder(*_: object) -> ResponseSpec:
        return json_response(
            {"error": {"message": "fake-provider-key", "code": "request_failed"}},
            status,
        )

    with fake_openai_server(responder) as (base_url, requests):
        provider = OpenAIProvider(config(base_url))
        with pytest.raises(OpenAIHTTPError) as raised:
            provider.next_step("inspect", ())

    assert raised.value.status_code == status
    assert "fake-provider-key" not in str(raised.value)
    assert len(requests) == 1


def test_invalid_json_is_a_sanitized_protocol_error() -> None:
    def responder(*_: object) -> ResponseSpec:
        return (200, {"Content-Type": "application/json"}, b"not-json-secret-value")

    with fake_openai_server(responder) as (base_url, _):
        client = OpenAIHTTPClient(config(base_url))
        with pytest.raises(OpenAIProtocolError) as raised:
            client.post_json("/responses", {"model": "fake"})

    assert "not-json-secret-value" not in str(raised.value)
    assert "fake-provider-key" not in str(raised.value)


def test_response_size_limit_is_enforced_before_json_parsing() -> None:
    oversized = b"{" + (b"x" * 512) + b"}"

    def responder(*_: object) -> ResponseSpec:
        return (200, {"Content-Type": "application/json"}, oversized)

    with fake_openai_server(responder) as (base_url, _):
        client = OpenAIHTTPClient(config(base_url, max_response_bytes=128))
        with pytest.raises(OpenAIResponseTooLargeError, match="size limit"):
            client.post_json("/responses", {"model": "fake"})


def test_timeout_is_sanitized_and_does_not_include_credentials() -> None:
    def responder(*_: object) -> ResponseSpec:
        time.sleep(0.15)
        return json_response({"output": []})

    with fake_openai_server(responder) as (base_url, _):
        client = OpenAIHTTPClient(config(base_url, timeout_seconds=0.03))
        with pytest.raises(OpenAITimeoutError) as raised:
            client.post_json("/responses", {"model": "fake"})

    assert "fake-provider-key" not in str(raised.value)


def test_network_failure_is_sanitized() -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as reservation:
        reservation.bind(("127.0.0.1", 0))
        port = reservation.getsockname()[1]

    client = OpenAIHTTPClient(
        config(f"http://127.0.0.1:{port}", timeout_seconds=0.2)
    )
    with pytest.raises(OpenAITransportError) as raised:
        client.post_json("/responses", {"model": "fake"})

    assert "fake-provider-key" not in str(raised.value)


def test_redirect_is_not_followed() -> None:
    def responder(
        path: str,
        headers: Mapping[str, str],
        raw: bytes,
        index: int,
    ) -> ResponseSpec:
        return (302, {"Location": "/captured"}, b"")

    with fake_openai_server(responder) as (base_url, requests):
        client = OpenAIHTTPClient(config(base_url))
        with pytest.raises(OpenAIHTTPError) as raised:
            client.post_json("/responses", {"model": "fake"})

    assert raised.value.status_code == 302
    assert [request["path"] for request in requests] == ["/responses"]


def test_request_size_limit_is_enforced_without_contacting_server() -> None:
    def responder(*_: object) -> ResponseSpec:
        raise AssertionError("oversized request must not reach the server")

    with fake_openai_server(responder) as (base_url, requests):
        client = OpenAIHTTPClient(config(base_url, max_request_bytes=64))
        with pytest.raises(OpenAITransportError, match="request exceeds"):
            client.post_json("/responses", {"input": "x" * 1000})

    assert requests == []


def test_unknown_tool_and_multiple_calls_are_protocol_errors() -> None:
    responses = [
        {
            "type": "function_call",
            "call_id": "call-1",
            "name": "not_registered",
            "arguments": "{}",
        },
        {
            "type": "function_call",
            "call_id": "call-2",
            "name": "git_status",
            "arguments": "{}",
        },
    ]

    def responder(*_: object) -> ResponseSpec:
        return json_response({"output": responses})

    with fake_openai_server(responder) as (base_url, _):
        provider = OpenAIProvider(config(base_url))
        with pytest.raises(OpenAIProtocolError, match="multiple tool calls"):
            provider.next_step("inspect", ())

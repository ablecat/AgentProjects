from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import FrozenInstanceError
from email.message import Message
from io import BytesIO
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import socket
import threading
import time
from urllib.error import HTTPError
import pytest

import repo_agent.openai_provider as provider_module
from repo_agent.models import FinalAnswer, ToolCall, ToolResult
from repo_agent.openai_provider import (
    APIKind,
    ESTIMATED_UTF8_BYTES_PER_TOKEN,
    MAX_TASK_CHARACTERS,
    MAX_RESPONSE_TOKEN_BUDGET,
    REQUEST_PROTOCOL_TOKEN_MARGIN,
    OpenAIConfig,
    OpenAIConfigurationError,
    OpenAIHTTPClient,
    OpenAIHTTPError,
    OpenAIProtocolError,
    OpenAIProvider,
    OpenAIRequestCapacityError,
    OpenAIResponseTooLargeError,
    OpenAITimeoutError,
    OpenAITransportError,
    RemoteModelNotAllowedError,
    _estimate_request_input_tokens,
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


def test_request_input_token_estimate_uses_exact_serialized_utf8_size() -> None:
    payload = {"model": "fake-model", "input": "\u6d4b\u8bd5"}
    encoded = json.dumps(
        payload,
        ensure_ascii=True,
        allow_nan=False,
        separators=(",", ":"),
    ).encode("utf-8")

    assert _estimate_request_input_tokens(payload) == (
        len(encoded) + ESTIMATED_UTF8_BYTES_PER_TOKEN - 1
    ) // ESTIMATED_UTF8_BYTES_PER_TOKEN + REQUEST_PROTOCOL_TOKEN_MARGIN


@pytest.mark.parametrize(
    "value",
    (False, True, 0, -1, 1.5, "100", MAX_RESPONSE_TOKEN_BUDGET + 1, 10**1000),
)
def test_response_token_budget_rejects_invalid_values(value: object) -> None:
    provider = OpenAIProvider(config("http://localhost:8123/v1"))

    with pytest.raises(OpenAIConfigurationError, match="response token budget"):
        provider.set_response_token_budget(value)  # type: ignore[arg-type]


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
        assert provider.last_response_usage is None
        call = provider.next_step("inspect", ())
        assert isinstance(call, ToolCall)
        first_usage = provider.last_response_usage
        assert first_usage is not None
        assert first_usage.response_count == 1
        assert first_usage.reported_response_count == 1
        assert first_usage.input_tokens == 10
        assert first_usage.cached_input_tokens == 3
        assert first_usage.output_tokens == 4
        assert first_usage.total_tokens == 14
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
    assert provider.last_response_usage is not first_usage
    assert provider.last_response_usage is not None
    assert provider.last_response_usage.response_count == 1
    assert provider.last_response_usage.reported_response_count == 1
    assert provider.last_response_usage.input_tokens == 20
    assert provider.last_response_usage.cached_input_tokens == 0
    assert provider.last_response_usage.output_tokens == 3
    assert provider.last_response_usage.total_tokens == 23
    with pytest.raises(FrozenInstanceError):
        first_usage.input_tokens = 999  # type: ignore[misc]
    with pytest.raises(AttributeError):
        provider.last_response_usage = first_usage  # type: ignore[misc]


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
    assert provider.last_response_usage is not None
    assert provider.last_response_usage.response_count == 1
    assert provider.last_response_usage.reported_response_count == 0
    assert provider.last_response_usage.input_tokens is None
    assert provider.last_response_usage.cached_input_tokens is None
    assert provider.last_response_usage.output_tokens is None
    assert provider.last_response_usage.total_tokens is None


def test_malformed_usage_marks_the_received_response_as_unreported() -> None:
    provider = OpenAIProvider(config("http://localhost:8123/v1"))
    provider._record_usage(
        {
            "usage": {
                "input_tokens": 2,
                "output_tokens": 3,
                "total_tokens": 5,
            }
        },
        "responses",
    )
    with pytest.raises(OpenAIProtocolError, match="usage is malformed"):
        provider._record_usage(
            {
                "usage": {
                    "input_tokens": 7,
                    "output_tokens": "invalid",
                    "total_tokens": 7,
                }
            },
            "responses",
        )

    assert provider.usage.response_count == 2
    assert provider.usage.reported_response_count == 1
    assert provider.usage.input_tokens == 2
    assert provider.usage.output_tokens == 3
    assert provider.usage.total_tokens == 5
    assert provider.usage.complete is False
    assert provider.last_response_usage == provider_module.ProviderUsage(
        1, 0, None, None, None, None
    )


@pytest.mark.parametrize(
    ("api_kind", "input_name", "output_name"),
    (
        ("responses", "input_tokens", "output_tokens"),
        ("chat_completions", "prompt_tokens", "completion_tokens"),
    ),
)
@pytest.mark.parametrize(
    ("input_tokens", "output_tokens", "total_tokens"),
    (
        (1_000_000, 2_000_000, 0),
        (7, 11, 19),
        (7, 11, 17),
    ),
)
def test_usage_total_must_equal_input_plus_output_without_partial_update(
    api_kind: APIKind,
    input_name: str,
    output_name: str,
    input_tokens: int,
    output_tokens: int,
    total_tokens: int,
) -> None:
    provider = OpenAIProvider(config("http://localhost:8123/v1"))
    provider._record_usage(
        {
            "usage": {
                input_name: 2,
                output_name: 3,
                "total_tokens": 5,
            }
        },
        api_kind,
    )
    with pytest.raises(OpenAIProtocolError, match="usage is malformed"):
        provider._record_usage(
            {
                "usage": {
                    input_name: input_tokens,
                    output_name: output_tokens,
                    "total_tokens": total_tokens,
                }
            },
            api_kind,
        )

    assert provider.usage.response_count == 2
    assert provider.usage.reported_response_count == 1
    assert provider.usage.input_tokens == 2
    assert provider.usage.output_tokens == 3
    assert provider.usage.total_tokens == 5
    assert provider.usage.complete is False
    assert provider.last_response_usage == provider_module.ProviderUsage(
        1, 0, None, None, None, None
    )


@pytest.mark.parametrize(
    ("api_kind", "input_name", "output_name", "details_name"),
    (
        (
            "responses",
            "input_tokens",
            "output_tokens",
            "input_tokens_details",
        ),
        (
            "chat_completions",
            "prompt_tokens",
            "completion_tokens",
            "prompt_tokens_details",
        ),
    ),
)
def test_cached_tokens_are_an_input_subset_and_are_not_added_to_total(
    api_kind: APIKind,
    input_name: str,
    output_name: str,
    details_name: str,
) -> None:
    provider = OpenAIProvider(config("http://localhost:8123/v1"))
    provider._record_usage(
        {
            "usage": {
                input_name: 10,
                details_name: {"cached_tokens": 4},
                output_name: 3,
                "total_tokens": 13,
            }
        },
        api_kind,
    )

    assert provider.usage.input_tokens == 10
    assert provider.usage.cached_input_tokens == 4
    assert provider.usage.output_tokens == 3
    assert provider.usage.total_tokens == 13


@pytest.mark.parametrize(
    ("api_kind", "input_name", "output_name", "details_name"),
    (
        (
            "responses",
            "input_tokens",
            "output_tokens",
            "input_tokens_details",
        ),
        (
            "chat_completions",
            "prompt_tokens",
            "completion_tokens",
            "prompt_tokens_details",
        ),
    ),
)
def test_cached_tokens_cannot_exceed_input_tokens(
    api_kind: APIKind,
    input_name: str,
    output_name: str,
    details_name: str,
) -> None:
    provider = OpenAIProvider(config("http://localhost:8123/v1"))

    with pytest.raises(OpenAIProtocolError, match="usage is malformed"):
        provider._record_usage(
            {
                "usage": {
                    input_name: 3,
                    details_name: {"cached_tokens": 4},
                    output_name: 2,
                    "total_tokens": 5,
                }
            },
            api_kind,
        )

    assert provider.usage.response_count == 1
    assert provider.usage.reported_response_count == 0
    assert provider.usage.total_tokens is None
    assert provider.usage.complete is False
    assert provider.last_response_usage == provider_module.ProviderUsage(
        1, 0, None, None, None, None
    )


def test_missing_environment_names_are_reported_without_values() -> None:
    with pytest.raises(OpenAIConfigurationError) as raised:
        OpenAIConfig.from_env({"REPO_AGENT_API_KEY": "fake-only-key"})

    message = str(raised.value)
    assert "REPO_AGENT_BASE_URL" in message
    assert "REPO_AGENT_MODEL" in message
    assert "fake-only-key" not in message


def test_config_rejects_api_keys_too_short_for_secret_scanning() -> None:
    with pytest.raises(OpenAIConfigurationError, match="too short"):
        config("http://localhost:8123/v1", api_key="short")


def test_config_canonicalizes_api_key_surrounding_whitespace() -> None:
    loaded = config(
        "http://localhost:8123/v1", api_key="  fake-provider-key  "
    )

    assert loaded.api_key == "fake-provider-key"


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
    ("url", "expected"),
    (
        ("https://api.deepseek.com", True),
        ("https://API.DEEPSEEK.COM:443/", True),
        ("https://api.deepseek.com/v1/", True),
        ("https://api.deepseek.com/v2", False),
        ("https://deepseek.com", False),
        ("http://api.deepseek.com/v1", False),
    ),
)
def test_official_deepseek_base_url_detection(url: str, expected: bool) -> None:
    assert provider_module.is_official_deepseek_base_url(url) is expected


def test_config_rejects_an_integer_too_large_for_a_platform_timeout() -> None:
    with pytest.raises(OpenAIConfigurationError, match="timeout_seconds"):
        config("http://localhost:8000/v1", timeout_seconds=10**1000)


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
        assert "max_output_tokens" not in request
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


def test_responses_provider_sends_explicit_reasoning_effort() -> None:
    def responder(
        _path: str,
        _headers: Mapping[str, str],
        raw: bytes,
        _index: int,
    ) -> ResponseSpec:
        request = json.loads(raw)
        assert request["reasoning"] == {"effort": "none"}
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
        provider = OpenAIProvider(config(base_url), reasoning_effort="none")
        assert provider.next_step("inspect", ()) == FinalAnswer("done")


def test_forced_tool_choice_is_consumed_by_one_responses_request() -> None:
    def responder(
        _path: str,
        _headers: Mapping[str, str],
        raw: bytes,
        index: int,
    ) -> ResponseSpec:
        request = json.loads(raw)
        if index == 0:
            assert request["tool_choice"] == {
                "type": "function",
                "name": "git_status",
            }
            return json_response(
                {
                    "output": [
                        {
                            "type": "function_call",
                            "call_id": "forced-call",
                            "name": "git_status",
                            "arguments": "{}",
                        }
                    ]
                }
            )
        assert "tool_choice" not in request
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
        provider = OpenAIProvider(config(base_url))
        provider.set_forced_tool_choice("git_status")
        call = provider.next_step("inspect", ())
        assert isinstance(call, ToolCall)
        assert provider.next_step(
            "inspect", (ToolResult(call.id, call.name, True, "clean"),)
        ) == FinalAnswer("done")


def test_official_deepseek_sequentializes_parallel_responses_calls() -> None:
    provider = OpenAIProvider(
        OpenAIConfig(
            api_key="fake-deepseek-key",
            base_url="https://api.deepseek.com",
            model="deepseek-v4-pro",
            allow_remote_model=True,
        )
    )
    payloads: list[Mapping[str, object]] = []

    def responder(_path, payload, **_kwargs):
        payloads.append(payload)
        if len(payloads) == 1:
            return {
                "output": [
                    {"type": "reasoning", "id": "reasoning-1", "content": []},
                    {
                        "type": "function_call",
                        "call_id": "first-call",
                        "name": "git_status",
                        "arguments": "{}",
                    },
                    {
                        "type": "function_call",
                        "call_id": "deferred-call",
                        "name": "git_status",
                        "arguments": "{}",
                    },
                ]
            }
        return {
            "output": [
                {
                    "type": "message",
                    "content": [{"type": "output_text", "text": "done"}],
                }
            ]
        }

    provider._client.post_json = responder  # type: ignore[method-assign]

    decision = provider.next_step("inspect", ())
    pending_replay_bytes = provider.pending_internal_tool_result_bytes
    assert pending_replay_bytes > 0
    assert provider.parallel_tool_call_violations == 1
    final = provider.next_step(
        "inspect",
        (ToolResult(decision.id, decision.name, True, "clean"),),
    )

    assert decision == ToolCall("first-call", "git_status", {})
    assert final == FinalAnswer("done")
    continued_input = payloads[1]["input"]
    assert isinstance(continued_input, list)
    replayed_calls = [
        item
        for item in continued_input
        if isinstance(item, Mapping) and item.get("type") == "function_call"
    ]
    assert [item["call_id"] for item in replayed_calls] == [
        "first-call",
        "deferred-call",
    ]
    replayed_outputs = [
        item
        for item in continued_input
        if isinstance(item, Mapping) and item.get("type") == "function_call_output"
    ]
    assert [item["call_id"] for item in replayed_outputs] == [
        "first-call",
        "deferred-call",
    ]
    deferred = json.loads(replayed_outputs[1]["output"])
    assert deferred["ok"] is False
    assert "not executed" in deferred["error"]
    assert provider.pending_internal_tool_result_bytes == 0
    assert provider.parallel_tool_call_violations == 1


def test_official_deepseek_rejects_too_many_parallel_responses_calls() -> None:
    provider = OpenAIProvider(
        OpenAIConfig(
            api_key="fake-deepseek-key",
            base_url="https://api.deepseek.com/v1",
            model="deepseek-v4-pro",
            allow_remote_model=True,
        )
    )
    call_count = provider_module.MAX_DEFERRED_RESPONSE_CALLS + 2
    provider._client.post_json = lambda *_args, **_kwargs: {  # type: ignore[method-assign]
        "output": [
            {
                "type": "function_call",
                "call_id": f"call-{index}",
                "name": "git_status",
                "arguments": "{}",
            }
            for index in range(call_count)
        ]
    }

    with pytest.raises(OpenAIProtocolError, match="too many parallel tool calls"):
        provider.next_step("inspect", ())

    assert provider.parallel_tool_call_violations == call_count - 1
    assert provider.pending_internal_tool_result_bytes == 0


def test_responses_provider_reports_incomplete_output_reason() -> None:
    def responder(*_: object) -> ResponseSpec:
        return json_response(
            {
                "status": "incomplete",
                "incomplete_details": {"reason": "max_output_tokens"},
                "output": [{"type": "reasoning", "content": []}],
                "usage": {
                    "input_tokens": 10,
                    "output_tokens": 20,
                    "total_tokens": 30,
                },
            }
        )

    with fake_openai_server(responder) as (base_url, _):
        provider = OpenAIProvider(config(base_url))
        with pytest.raises(OpenAIProtocolError, match="max_output_tokens"):
            provider.next_step("inspect", ())

    assert provider.usage.total_tokens == 30


def test_responses_token_budget_is_recomputed_and_consumed_each_round() -> None:
    budgets = (4_000, 3_000)
    output_limits: list[int] = []

    def responder(
        path: str,
        headers: Mapping[str, str],
        raw: bytes,
        index: int,
    ) -> ResponseSpec:
        del headers
        assert path == "/responses"
        request = json.loads(raw)
        if index < 2:
            output_limit = request["max_output_tokens"]
            assert type(output_limit) is int and output_limit > 0
            assert output_limit + _estimate_request_input_tokens(request) <= budgets[index]
            output_limits.append(output_limit)
        else:
            assert "max_output_tokens" not in request

        if index == 0:
            return json_response(
                {
                    "output": [
                        {
                            "type": "function_call",
                            "call_id": "budget-call",
                            "name": "git_status",
                            "arguments": "{}",
                        }
                    ]
                }
            )
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

    with fake_openai_server(responder) as (base_url, requests):
        provider = OpenAIProvider(config(base_url))
        provider.set_response_token_budget(budgets[0])
        call = provider.next_step("inspect", ())
        assert isinstance(call, ToolCall)
        provider.set_response_token_budget(budgets[1])
        final = provider.next_step(
            "inspect",
            (ToolResult(call.id, call.name, True, "clean"),),
        )
        uncapped = provider.next_step("another task", ())

    assert final == FinalAnswer("done")
    assert uncapped == FinalAnswer("done")
    assert output_limits[1] < output_limits[0]
    assert len(requests) == 3


def test_response_token_budget_can_be_exhausted_before_http() -> None:
    def responder(*_: object) -> ResponseSpec:
        return json_response({"output": []})

    with fake_openai_server(responder) as (base_url, requests):
        provider = OpenAIProvider(config(base_url))
        provider.set_response_token_budget(1_000)
        with pytest.raises(OpenAIRequestCapacityError, match="estimated request input"):
            provider.next_step("inspect", ())

    assert requests == []
    assert provider.usage.response_count == 0
    assert provider.usage.total_tokens is None
    assert provider.last_response_usage is None


def test_responses_budget_requires_sixteen_output_tokens_before_http(
    monkeypatch,
) -> None:
    estimated_payloads: list[dict[str, object]] = []

    def fixed_estimate(payload: Mapping[str, object]) -> int:
        estimated_payloads.append(dict(payload))
        return 100

    monkeypatch.setattr(
        provider_module, "_estimate_request_input_tokens", fixed_estimate
    )

    def responder(*_: object) -> ResponseSpec:
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

    with fake_openai_server(responder) as (base_url, requests):
        blocked = OpenAIProvider(config(base_url))
        blocked.set_response_token_budget(115)
        with pytest.raises(OpenAIRequestCapacityError, match="minimum output"):
            blocked.next_step("inspect", ())
        assert requests == []
        assert blocked.usage.response_count == 0
        assert blocked.usage.total_tokens is None

        allowed = OpenAIProvider(config(base_url))
        allowed.set_response_token_budget(116)
        assert allowed.next_step("inspect", ()) == FinalAnswer("done")

    assert len(requests) == 1
    request = json.loads(requests[0]["body"])
    assert request["max_output_tokens"] == 16
    assert all(
        {
            "model",
            "instructions",
            "input",
            "tools",
            "parallel_tool_calls",
            "store",
            "max_output_tokens",
        }.issubset(payload)
        for payload in estimated_payloads
    )


def test_chat_budget_can_retain_one_output_token(monkeypatch) -> None:
    monkeypatch.setattr(
        provider_module, "_estimate_request_input_tokens", lambda _payload: 100
    )
    payload: dict[str, object] = {"messages": []}

    provider_module._apply_response_token_budget(
        payload,
        field_name="max_completion_tokens",
        total_tokens=101,
    )

    assert payload["max_completion_tokens"] == 1
    with pytest.raises(OpenAIRequestCapacityError, match="minimum output"):
        provider_module._apply_response_token_budget(
            {"messages": []},
            field_name="max_completion_tokens",
            total_tokens=100,
        )


def test_complete_responses_payload_has_a_real_capacity_boundary_before_http() -> None:
    def responder(*_: object) -> ResponseSpec:
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

    with fake_openai_server(responder) as (base_url, requests):
        def attempt(task_size: int) -> tuple[bool, dict[str, object] | None]:
            before = len(requests)
            provider = OpenAIProvider(
                config(base_url),
                tool_definitions=(TOOL_DEFINITIONS[0],),
                system_prompt="Review the complete candidate.",
            )
            provider.set_response_token_budget(2_500)
            try:
                decision = provider.next_step("x" * task_size, ())
            except OpenAIRequestCapacityError:
                assert len(requests) == before
                assert provider.usage.response_count == 0
                return False, None
            assert decision == FinalAnswer("done")
            assert len(requests) == before + 1
            return True, json.loads(requests[-1]["body"])

        lower = 1
        upper = 40_000
        assert attempt(lower)[0] is True
        assert attempt(upper)[0] is False
        while lower + 1 < upper:
            midpoint = (lower + upper) // 2
            if attempt(midpoint)[0]:
                lower = midpoint
            else:
                upper = midpoint

        fits, request = attempt(lower)
        over_capacity, _ = attempt(upper)

    assert fits is True
    assert request is not None
    assert request["max_output_tokens"] >= 16
    assert upper == lower + 1
    assert over_capacity is False


def test_extended_task_limit_reaches_transport_with_complete_payload() -> None:
    def responder(*_: object) -> ResponseSpec:
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

    with fake_openai_server(responder) as (base_url, requests):
        provider_config = config(base_url)
        provider = OpenAIProvider(
            provider_config,
            tool_definitions=(TOOL_DEFINITIONS[0],),
            system_prompt="Review the complete bounded workflow context.",
        )
        decision = provider.next_step("x" * MAX_TASK_CHARACTERS, ())

    assert decision == FinalAnswer("done")
    assert len(requests) == 1
    assert len(requests[0]["body"]) < provider_config.max_request_bytes


def test_response_token_budget_survives_a_protocol_error_retry() -> None:
    output_limits: list[int] = []

    def responder(
        path: str,
        headers: Mapping[str, str],
        raw: bytes,
        index: int,
    ) -> ResponseSpec:
        del path, headers
        request = json.loads(raw)
        output_limits.append(request["max_output_tokens"])
        if index == 0:
            return (200, {"Content-Type": "application/json"}, b"not-json")
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

    with fake_openai_server(responder) as (base_url, requests):
        provider = OpenAIProvider(config(base_url))
        provider.set_response_token_budget(4_000)
        with pytest.raises(OpenAIProtocolError):
            provider.next_step("inspect", ())
        final = provider.next_step("inspect", ())

    assert final == FinalAnswer("done")
    assert output_limits[0] == output_limits[1]
    assert len(requests) == 2


def test_forced_tool_choice_survives_a_protocol_error_retry() -> None:
    tool_choices: list[object] = []

    def responder(
        _path: str,
        _headers: Mapping[str, str],
        raw: bytes,
        index: int,
    ) -> ResponseSpec:
        request = json.loads(raw)
        tool_choices.append(request["tool_choice"])
        if index == 0:
            return (200, {"Content-Type": "application/json"}, b"not-json")
        return json_response(
            {
                "output": [
                    {
                        "type": "function_call",
                        "call_id": "forced-retry",
                        "name": "git_status",
                        "arguments": "{}",
                    }
                ]
            }
        )

    with fake_openai_server(responder) as (base_url, requests):
        provider = OpenAIProvider(config(base_url))
        provider.set_forced_tool_choice("git_status")
        with pytest.raises(OpenAIProtocolError):
            provider.next_step("inspect", ())
        decision = provider.next_step("inspect", ())

    assert decision == ToolCall("forced-retry", "git_status", {})
    assert tool_choices == [
        {"type": "function", "name": "git_status"},
        {"type": "function", "name": "git_status"},
    ]
    assert len(requests) == 2


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
    total_budgets = (4_000, 4_000, 3_000)

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
        request = json.loads(raw)
        if index == 0:
            assert path == "/responses"
            assert "max_completion_tokens" not in request
            output_limit = request["max_output_tokens"]
            assert (
                output_limit + _estimate_request_input_tokens(request)
                <= total_budgets[index]
            )
            return json_response({"error": {"code": "unknown_endpoint"}}, 404)
        assert path == "/chat/completions"
        assert "max_output_tokens" not in request
        output_limit = request["max_completion_tokens"]
        assert (
            output_limit + _estimate_request_input_tokens(request)
            <= total_budgets[index]
        )
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
        provider.set_response_token_budget(total_budgets[0])
        decision = provider.next_step("find FIXME markers", ())
        assert isinstance(decision, ToolCall)
        result = ToolResult(
            decision.id,
            decision.name,
            True,
            "1 match",
            exit_code=0,
        )
        provider.set_response_token_budget(total_budgets[2])
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


class _ResponseHeaders:
    def __init__(self, content_lengths: list[object]) -> None:
        self._content_lengths = content_lengths

    def get_all(self, name: str) -> list[object] | None:
        assert name == "Content-Length"
        return list(self._content_lengths) or None


class _BoundedResponse:
    def __init__(self, headers: object, body: bytes) -> None:
        self.headers = headers
        self._body = body

    def read(self, size: int) -> bytes:
        return self._body[:size]


class _ErrorOpener:
    def __init__(self, error: HTTPError) -> None:
        self._error = error

    def open(self, *_args: object, **_kwargs: object) -> object:
        raise self._error


def _http_error(
    status: int,
    content_lengths: tuple[str, ...],
    body: bytes,
) -> HTTPError:
    headers = Message()
    for value in content_lengths:
        headers.add_header("Content-Length", value)
    return HTTPError(
        "http://localhost:8123/v1/responses",
        status,
        "model error",
        headers,
        BytesIO(body),
    )


def test_response_content_length_accepts_single_exact_value_and_mapping_fallback() -> None:
    client = OpenAIHTTPClient(
        config("http://localhost:8123/v1", max_response_bytes=8)
    )
    response = _BoundedResponse({"Content-Length": "2"}, b"{}")

    assert client._read_bounded(response) == b"{}"


def test_response_content_length_rejects_duplicate_values() -> None:
    client = OpenAIHTTPClient(
        config("http://localhost:8123/v1", max_response_bytes=8)
    )
    response = _BoundedResponse(_ResponseHeaders(["2", "2"]), b"{}")

    with pytest.raises(OpenAIProtocolError, match="Content-Length"):
        client._read_bounded(response)


@pytest.mark.parametrize(
    "value",
    ("", "+2", "-2", "2.0", " 2", "2 ", "\u0662", "0" * 129, b"2", None),
)
def test_response_content_length_rejects_invalid_values(value: object) -> None:
    client = OpenAIHTTPClient(
        config("http://localhost:8123/v1", max_response_bytes=8)
    )
    response = _BoundedResponse(_ResponseHeaders([value]), b"{}")

    with pytest.raises(OpenAIProtocolError, match="Content-Length"):
        client._read_bounded(response)


def test_response_content_length_rejects_declared_overflow() -> None:
    client = OpenAIHTTPClient(
        config("http://localhost:8123/v1", max_response_bytes=8)
    )
    response = _BoundedResponse(_ResponseHeaders(["9"]), b"{}")

    with pytest.raises(OpenAIResponseTooLargeError, match="size limit"):
        client._read_bounded(response)


def test_response_content_length_rejects_short_read() -> None:
    client = OpenAIHTTPClient(
        config("http://localhost:8123/v1", max_response_bytes=8)
    )
    response = _BoundedResponse(_ResponseHeaders(["3"]), b"{}")

    with pytest.raises(OpenAIProtocolError, match="Content-Length"):
        client._read_bounded(response)


@pytest.mark.parametrize("status", [400, 500])
@pytest.mark.parametrize(
    ("content_lengths", "body", "expected_error"),
    [
        (("2", "2"), b"{}", OpenAIProtocolError),
        (("invalid",), b"{}", OpenAIProtocolError),
        (("9",), b"{}", OpenAIResponseTooLargeError),
        (("3",), b"{}", OpenAIProtocolError),
    ],
)
def test_http_error_responses_enforce_content_length_contract(
    status: int,
    content_lengths: tuple[str, ...],
    body: bytes,
    expected_error: type[Exception],
) -> None:
    client = OpenAIHTTPClient(
        config("http://localhost:8123/v1", max_response_bytes=8)
    )
    client._opener = _ErrorOpener(  # type: ignore[assignment]
        _http_error(status, content_lengths, body)
    )

    with pytest.raises(expected_error, match="Content-Length|size limit"):
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

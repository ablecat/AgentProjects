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
    OpenAIToolArgumentsError,
    OpenAIUnknownToolError,
    OpenAITransportError,
    RemoteModelNotAllowedError,
    _encode_request_payload,
    _estimate_request_input_tokens,
    strict_response_tools,
    tool_result_replay_bytes,
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


def test_request_input_token_estimate_charges_each_serialized_utf8_byte() -> None:
    payload = {"model": "fake-model", "input": "\u6d4b\u8bd5"}
    encoded = json.dumps(
        payload,
        ensure_ascii=True,
        allow_nan=False,
        separators=(",", ":"),
    ).encode("utf-8")

    assert _estimate_request_input_tokens(payload) == (
        len(encoded) + REQUEST_PROTOCOL_TOKEN_MARGIN
    )


def test_tool_result_replay_bytes_bounds_responses_and_chat_unicode_envelopes() -> None:
    result = ToolResult(
        "unicode-call",
        "read_files",
        True,
        "\U0001f600\u754c" * 2_048,
        exit_code=0,
    )
    serialized = json.dumps(
        {
            "ok": result.ok,
            "output": result.output,
            "error": result.error,
            "exit_code": result.exit_code,
            "truncated": result.truncated,
        },
        ensure_ascii=True,
        allow_nan=False,
        separators=(",", ":"),
    )
    responses_envelope = {
        "input": [
            {
                "type": "function_call_output",
                "call_id": result.call_id,
                "output": serialized,
            }
        ]
    }
    chat_envelope = {
        "messages": [
            {
                "role": "tool",
                "tool_call_id": result.call_id,
                "content": serialized,
            }
        ]
    }
    response_bytes = len(_encode_request_payload(responses_envelope))
    chat_bytes = len(_encode_request_payload(chat_envelope))

    assert tool_result_replay_bytes(result) == max(response_bytes, chat_bytes)
    assert tool_result_replay_bytes(result) >= response_bytes
    assert tool_result_replay_bytes(result) >= chat_bytes
    assert tool_result_replay_bytes(result) > len(result.output.encode("utf-8")) * 3


@pytest.mark.parametrize(
    "request_input",
    (
        [
            {
                "role": "user",
                "content": [{"type": "input_text", "text": "x" * 1_000}],
            }
        ],
        [
            {
                "role": "user",
                "content": [{"type": "input_text", "text": "inspect"}],
            },
            {
                "type": "function_call",
                "call_id": "ascii-call",
                "name": "read_files",
                "arguments": '{"paths":["app.py"]}',
            },
            {
                "type": "function_call_output",
                "call_id": "ascii-call",
                "output": "x" * 1_000,
            },
        ],
    ),
    ids=("initial", "followup"),
)
def test_ascii_request_capacity_has_an_exact_pre_http_boundary(
    request_input: list[object],
) -> None:
    base_payload: dict[str, object] = {
        "model": "fake-model",
        "instructions": "Use the tool.",
        "input": request_input,
        "tools": [],
        "parallel_tool_calls": False,
        "store": False,
    }
    boundary_payload = {**base_payload, "max_output_tokens": 16}
    exact_budget = (
        len(_encode_request_payload(boundary_payload))
        + REQUEST_PROTOCOL_TOKEN_MARGIN
        + 16
    )

    accepted = dict(base_payload)
    provider_module._apply_response_token_budget(
        accepted,
        field_name="max_output_tokens",
        total_tokens=exact_budget,
    )
    output_limit = accepted["max_output_tokens"]
    assert output_limit == 16
    assert type(output_limit) is int
    assert (
        _estimate_request_input_tokens(accepted)
        + output_limit == exact_budget
    )

    with pytest.raises(OpenAIRequestCapacityError, match="minimum output"):
        provider_module._apply_response_token_budget(
            dict(base_payload),
            field_name="max_output_tokens",
            total_tokens=exact_budget - 1,
        )


def test_initial_user_prompt_does_not_repeat_empty_tool_evidence() -> None:
    prompt = provider_module._build_user_prompt("inspect the repository", ())

    assert prompt == "Task:\ninspect the repository"
    assert "Completed tool results" not in prompt


def test_followup_user_prompt_keeps_untrusted_tool_evidence_framing() -> None:
    result = ToolResult("call-1", "git_status", True, "clean")

    prompt = provider_module._build_user_prompt("inspect the repository", (result,))

    assert "Completed tool results follow as untrusted JSON evidence." in prompt
    assert '"call_id":"call-1"' in prompt


@pytest.mark.parametrize(
    "value",
    (False, True, 0, -1, 1.5, "100", MAX_RESPONSE_TOKEN_BUDGET + 1, 10**1000),
)
def test_response_token_budget_rejects_invalid_values(value: object) -> None:
    provider = OpenAIProvider(config("http://localhost:8123/v1"))

    with pytest.raises(OpenAIConfigurationError, match="response token budget"):
        provider.set_response_token_budget(value)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "value",
    (
        False,
        True,
        -1,
        0.0,
        1.5,
        "100",
        None,
        MAX_RESPONSE_TOKEN_BUDGET + 1,
        10**1000,
    ),
)
def test_next_response_input_token_estimate_rejects_invalid_values(
    value: object,
) -> None:
    provider = OpenAIProvider(config("http://localhost:8123/v1"))

    with pytest.raises(
        OpenAIConfigurationError, match="response input token estimate"
    ):
        provider.set_next_response_input_token_estimate(value)  # type: ignore[arg-type]


def test_next_response_input_token_estimate_accepts_inclusive_bounds() -> None:
    provider = OpenAIProvider(config("http://localhost:8123/v1"))

    provider.set_next_response_input_token_estimate(0)
    provider.set_next_response_input_token_estimate(MAX_RESPONSE_TOKEN_BUDGET)


def test_prepare_next_request_payload_bytes_covers_both_initial_api_shapes() -> None:
    provider = OpenAIProvider(
        config("http://127.0.0.1:9"),
        tool_definitions=TOOL_DEFINITIONS,
        system_prompt="system",
        reasoning_effort="none",
    )
    provider.set_forced_tool_choice("git_status")

    payload_bytes = provider.prepare_next_request_payload_bytes("inspect", ())

    assert payload_bytes > len(b"inspect")
    assert provider.request_count == 0
    assert provider.usage.response_count == 0


@pytest.mark.parametrize(
    "value",
    (False, True, 0, -1, 1.5, "100", MAX_RESPONSE_TOKEN_BUDGET + 1, 10**1000),
)
def test_minimum_response_output_tokens_rejects_invalid_values(value: object) -> None:
    provider = OpenAIProvider(config("http://localhost:8123/v1"))

    with pytest.raises(OpenAIConfigurationError, match="minimum response output"):
        provider.set_minimum_response_output_tokens(value)  # type: ignore[arg-type]


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
    assert provider.rejected_tool_call_violations == 0
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
            assert [tool["name"] for tool in request["tools"]] == [
                "git_status"
            ]
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
        assert {tool["name"] for tool in request["tools"]} == {
            definition.name for definition in TOOL_DEFINITIONS
        }
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


def test_official_deepseek_named_tool_request_is_non_thinking() -> None:
    provider = OpenAIProvider(
        OpenAIConfig(
            api_key="fake-deepseek-key",
            base_url="https://api.deepseek.com",
            model="deepseek-v4-pro",
            allow_remote_model=True,
        ),
        reasoning_effort="none",
    )
    payloads: list[Mapping[str, object]] = []

    def responder(_path, payload, **_kwargs):
        payloads.append(payload)
        return {
            "output": [
                {
                    "type": "function_call",
                    "call_id": "read-call",
                    "name": "git_status",
                    "arguments": "{}",
                }
            ],
            "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
        }

    provider._client.post_json = responder  # type: ignore[method-assign]
    provider.set_forced_tool_choice("git_status")

    assert provider.next_step("inspect", ()) == ToolCall(
        "read-call", "git_status", {}
    )
    request = payloads[0]
    assert request["reasoning"] == {"effort": "none"}
    assert request["tool_choice"] == {"type": "function", "name": "git_status"}
    assert [tool["name"] for tool in request["tools"]] == ["git_status"]


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
    assert provider.rejected_tool_call_violations == 0
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
    assert provider.rejected_tool_call_violations == 0


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
    assert provider.rejected_tool_call_violations == 1
    assert provider.pending_internal_tool_result_bytes == 0


def test_malformed_usage_counts_unexecuted_calls_without_parallel_double_count() -> None:
    provider = OpenAIProvider(
        OpenAIConfig(
            api_key="fake-deepseek-key",
            base_url="https://api.deepseek.com",
            model="deepseek-v4-pro",
            allow_remote_model=True,
        )
    )
    provider._client.post_json = lambda *_args, **_kwargs: {  # type: ignore[method-assign]
        "output": [
            {
                "type": "function_call",
                "call_id": f"usage-call-{index}",
                "name": "git_status",
                "arguments": "{}",
            }
            for index in range(2)
        ],
        "usage": {
            "input_tokens": 3,
            "output_tokens": "invalid",
            "total_tokens": 3,
        },
    }

    with pytest.raises(OpenAIProtocolError, match="usage is malformed"):
        provider.next_step("inspect", ())

    assert provider.parallel_tool_call_violations == 1
    assert provider.rejected_tool_call_violations == 1


def test_responses_provider_reports_incomplete_output_reason() -> None:
    def responder(*_: object) -> ResponseSpec:
        return json_response(
            {
                "status": "incomplete",
                "incomplete_details": {"reason": "max_output_tokens"},
                "output": [
                    {
                        "type": "function_call",
                        "call_id": "incomplete-call",
                        "name": "git_status",
                        "arguments": '{"unfinished":',
                    }
                ],
                "usage": {
                    "input_tokens": 10,
                    "output_tokens": 20,
                    "total_tokens": 30,
                },
            }
        )

    with fake_openai_server(responder) as (base_url, _):
        provider = OpenAIProvider(config(base_url))
        with pytest.raises(OpenAIProtocolError, match="max_output_tokens") as raised:
            provider.next_step("inspect", ())

    assert not isinstance(raised.value, OpenAIToolArgumentsError)
    assert provider.usage.complete is True
    assert provider.usage.total_tokens == 30
    assert provider.rejected_tool_call_violations == 0


def test_responses_malformed_nested_tool_arguments_use_dedicated_error() -> None:
    malformed = '{"path":"unterminated'

    def responder(*_: object) -> ResponseSpec:
        return json_response(
            {
                "status": "completed",
                "output": [
                    {
                        "type": "function_call",
                        "call_id": "responses-malformed-arguments",
                        "name": "git_status",
                        "arguments": malformed,
                    }
                ],
            }
        )

    with fake_openai_server(responder) as (base_url, _):
        provider = OpenAIProvider(config(base_url))
        with pytest.raises(
            OpenAIToolArgumentsError, match="tool arguments is invalid JSON"
        ) as raised:
            provider.next_step("inspect", ())

    assert malformed not in str(raised.value)
    assert provider.rejected_tool_call_violations == 1
    assert provider.recovered_tool_argument_violations == 0
    provider.mark_malformed_tool_arguments_recovered()
    assert provider.recovered_tool_argument_violations == 1
    with pytest.raises(OpenAIProtocolError, match="awaiting recovery"):
        provider.mark_malformed_tool_arguments_recovered()


def test_chat_malformed_nested_tool_arguments_use_dedicated_error() -> None:
    malformed = '{"path":"unterminated'

    def responder(
        path: str,
        _headers: Mapping[str, str],
        _raw: bytes,
        _index: int,
    ) -> ResponseSpec:
        if path == "/responses":
            return json_response({"error": {"code": "unknown_endpoint"}}, 404)
        return json_response(
            {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "chat-malformed-arguments",
                                    "type": "function",
                                    "function": {
                                        "name": "git_status",
                                        "arguments": malformed,
                                    },
                                }
                            ],
                        }
                    }
                ]
            }
        )

    with fake_openai_server(responder) as (base_url, _):
        provider = OpenAIProvider(config(base_url))
        with pytest.raises(
            OpenAIToolArgumentsError, match="tool arguments is invalid JSON"
        ) as raised:
            provider.next_step("inspect", ())

    assert malformed not in str(raised.value)
    assert provider.api_kind == "chat_completions"
    assert provider.rejected_tool_call_violations == 1


def test_unknown_tool_uses_dedicated_error_and_tracks_recovery() -> None:
    def responder(*_: object) -> ResponseSpec:
        return json_response(
            {
                "output": [
                    {
                        "type": "function_call",
                        "call_id": "unknown-tool-call",
                        "name": "not_registered",
                        "arguments": '{"unfinished":',
                    }
                ]
            }
        )

    with fake_openai_server(responder) as (base_url, _):
        provider = OpenAIProvider(config(base_url))
        with pytest.raises(
            OpenAIUnknownToolError, match="unknown or inactive tool"
        ) as raised:
            provider.next_step("inspect", ())

    assert isinstance(raised.value, OpenAIProtocolError)
    assert not isinstance(raised.value, OpenAIToolArgumentsError)
    assert provider.rejected_tool_call_violations == 1
    assert provider.unknown_tool_violations == 1
    assert provider.recovered_unknown_tool_violations == 0
    provider.mark_unknown_tool_recovered()
    assert provider.recovered_unknown_tool_violations == 1
    with pytest.raises(OpenAIProtocolError, match="awaiting recovery"):
        provider.mark_unknown_tool_recovered()


def test_outer_malformed_response_remains_a_general_protocol_error() -> None:
    def responder(*_: object) -> ResponseSpec:
        return 200, {"Content-Type": "application/json"}, b'{"output":'

    with fake_openai_server(responder) as (base_url, _):
        provider = OpenAIProvider(config(base_url))
        with pytest.raises(
            OpenAIProtocolError, match="model response is invalid JSON"
        ) as raised:
            provider.next_step("inspect", ())

    assert not isinstance(raised.value, OpenAIToolArgumentsError)


def test_responses_token_budget_is_recomputed_and_consumed_each_round() -> None:
    budgets = (6_000, 6_000)
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
    assert provider.request_count == 0
    assert provider.usage.response_count == 0
    assert provider.usage.total_tokens is None
    assert provider.last_response_usage is None


def test_responses_input_estimate_override_allows_full_payload_once() -> None:
    budget = 256
    input_estimate = 64
    task = "x" * 1_000

    def responder(path, _headers, _raw, _index):
        assert path == "/responses"
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
        provider = OpenAIProvider(
            config(base_url),
            tool_definitions=(TOOL_DEFINITIONS[0],),
            system_prompt="Review the complete candidate.",
        )
        provider.set_response_token_budget(budget)
        with pytest.raises(OpenAIRequestCapacityError):
            provider.next_step(task, ())
        assert requests == []

        provider.set_next_response_input_token_estimate(input_estimate)
        assert provider.next_step(task, ()) == FinalAnswer("done")

        request = json.loads(requests[0]["body"])
        assert _estimate_request_input_tokens(request) > budget
        assert request["max_output_tokens"] == budget - input_estimate

        provider.set_response_token_budget(budget)
        with pytest.raises(OpenAIRequestCapacityError):
            provider.next_step("y" * 1_000, ())

    assert len(requests) == 1


def test_input_estimate_override_applies_to_responses_and_chat_once() -> None:
    budget = 256
    input_estimate = 64
    task = "x" * 1_000

    def responder(path, _headers, _raw, _index):
        if path == "/responses":
            return json_response({"error": {"code": "unknown_endpoint"}}, 404)
        assert path == "/chat/completions"
        return json_response(
            {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "done",
                        }
                    }
                ]
            }
        )

    with fake_openai_server(responder) as (base_url, requests):
        blocked = OpenAIProvider(
            config(base_url),
            tool_definitions=(TOOL_DEFINITIONS[0],),
            system_prompt="Review the complete candidate.",
        )
        blocked.set_response_token_budget(budget)
        with pytest.raises(OpenAIRequestCapacityError):
            blocked.next_step(task, ())
        assert requests == []

        provider = OpenAIProvider(
            config(base_url),
            tool_definitions=(TOOL_DEFINITIONS[0],),
            system_prompt="Review the complete candidate.",
        )
        provider.set_response_token_budget(budget)
        provider.set_next_response_input_token_estimate(input_estimate)
        assert provider.next_step(task, ()) == FinalAnswer("done")

        responses_request = json.loads(requests[0]["body"])
        chat_request = json.loads(requests[1]["body"])
        assert _estimate_request_input_tokens(responses_request) > budget
        assert _estimate_request_input_tokens(chat_request) > budget
        assert responses_request["max_output_tokens"] == budget - input_estimate
        assert chat_request["max_completion_tokens"] == budget - input_estimate

        provider.set_response_token_budget(budget)
        with pytest.raises(OpenAIRequestCapacityError):
            provider.next_step("y" * 1_000, ())

    assert provider.api_kind == "chat_completions"
    assert [request["path"] for request in requests] == [
        "/responses",
        "/chat/completions",
    ]


def test_chat_fallback_capacity_error_records_the_completed_responses_attempt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def estimate(payload: Mapping[str, object]) -> int:
        return 100 if "input" in payload else 200

    monkeypatch.setattr(provider_module, "_estimate_request_input_tokens", estimate)

    def responder(path, _headers, _raw, _index):
        assert path == "/responses"
        return json_response({"error": {"code": "unknown_endpoint"}}, 404)

    with fake_openai_server(responder) as (base_url, requests):
        provider = OpenAIProvider(config(base_url))
        provider.set_response_token_budget(150)
        with pytest.raises(OpenAIRequestCapacityError, match="minimum output"):
            provider.next_step("inspect", ())

    assert [request["path"] for request in requests] == ["/responses"]
    assert provider.request_count == 1
    assert provider.usage.response_count == 1
    assert provider.usage.reported_response_count == 0
    assert provider.usage.complete is False
    assert provider.last_response_usage == provider_module.ProviderUsage(
        1, 0, None, None, None, None
    )


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


def test_phase_output_floor_is_enforced_before_http(monkeypatch) -> None:
    monkeypatch.setattr(
        provider_module, "_estimate_request_input_tokens", lambda _payload: 100
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
        blocked.set_response_token_budget(355)
        blocked.set_minimum_response_output_tokens(256)
        with pytest.raises(OpenAIRequestCapacityError, match="minimum output"):
            blocked.next_step("inspect", ())
        assert requests == []

        allowed = OpenAIProvider(config(base_url))
        allowed.set_response_token_budget(356)
        allowed.set_minimum_response_output_tokens(256)
        assert allowed.next_step("inspect", ()) == FinalAnswer("done")

    request = json.loads(requests[0]["body"])
    assert request["max_output_tokens"] == 256


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
        provider = OpenAIProvider(
            config(base_url),
            idempotency_key="run-budget:implement:0",
        )
        provider.set_response_token_budget(6_000)
        with pytest.raises(OpenAIProtocolError):
            provider.next_step("inspect", ())
        final = provider.next_step("inspect", ())

    assert final == FinalAnswer("done")
    assert output_limits[0] == output_limits[1]
    assert [
        request["headers"]["idempotency-key"] for request in requests
    ] == [
        "run-budget:implement:0:responses:0",
        "run-budget:implement:0:responses:1",
    ]
    assert len(requests) == 2


def test_forced_tool_choice_survives_a_protocol_error_retry() -> None:
    tool_choices: list[object] = []
    exposed_tool_names: list[list[object]] = []

    def responder(
        _path: str,
        _headers: Mapping[str, str],
        raw: bytes,
        index: int,
    ) -> ResponseSpec:
        request = json.loads(raw)
        tool_choices.append(request["tool_choice"])
        exposed_tool_names.append([tool["name"] for tool in request["tools"]])
        if index == 0:
            return json_response(
                {
                    "output": [
                        {
                            "type": "function_call",
                            "call_id": "unexposed-tool",
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
        provider = OpenAIProvider(
            config(base_url),
            idempotency_key="run-forced:implement:0",
        )
        provider.set_forced_tool_choice("git_status")
        with pytest.raises(
            OpenAIUnknownToolError, match="unknown or inactive tool"
        ):
            provider.next_step("inspect", ())
        decision = provider.next_step("inspect", ())
        provider.mark_unknown_tool_recovered()

    assert decision == ToolCall("forced-retry", "git_status", {})
    assert tool_choices == [
        {"type": "function", "name": "git_status"},
        {"type": "function", "name": "git_status"},
    ]
    assert exposed_tool_names == [["git_status"], ["git_status"]]
    assert provider.rejected_tool_call_violations == 1
    assert provider.unknown_tool_violations == 1
    assert provider.recovered_unknown_tool_violations == 1
    assert provider.parallel_tool_call_violations == 0
    assert [
        request["headers"]["idempotency-key"] for request in requests
    ] == [
        "run-forced:implement:0:responses:0",
        "run-forced:implement:0:responses:1",
    ]
    assert len(requests) == 2


def test_responses_forced_tool_choice_rejects_final_text_and_survives_retry() -> None:
    def responder(
        _path: str,
        _headers: Mapping[str, str],
        raw: bytes,
        index: int,
    ) -> ResponseSpec:
        request = json.loads(raw)
        assert request["tool_choice"] == {
            "type": "function",
            "name": "git_status",
        }
        if index == 0:
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
        return json_response(
            {
                "output": [
                    {
                        "type": "function_call",
                        "call_id": "forced-after-text",
                        "name": "git_status",
                        "arguments": "{}",
                    }
                ]
            }
        )

    with fake_openai_server(responder) as (base_url, requests):
        provider = OpenAIProvider(
            config(base_url),
            idempotency_key="forced-text:responses",
        )
        provider.set_forced_tool_choice("git_status")
        with pytest.raises(OpenAIProtocolError, match="forced tool"):
            provider.next_step("inspect", ())
        decision = provider.next_step("inspect", ())

    assert decision == ToolCall("forced-after-text", "git_status", {})
    assert provider.rejected_tool_call_violations == 1
    assert [
        request["headers"]["idempotency-key"] for request in requests
    ] == [
        "forced-text:responses:responses:0",
        "forced-text:responses:responses:1",
    ]


def test_chat_forced_tool_choice_filters_one_round_and_survives_retry() -> None:
    chat_attempt = 0

    def responder(
        path: str,
        _headers: Mapping[str, str],
        raw: bytes,
        _index: int,
    ) -> ResponseSpec:
        nonlocal chat_attempt
        request = json.loads(raw)
        if path == "/responses":
            assert request["tool_choice"] == {
                "type": "function",
                "name": "git_status",
            }
            assert [tool["name"] for tool in request["tools"]] == [
                "git_status"
            ]
            return json_response({"error": {"code": "unknown_endpoint"}}, 404)

        assert path == "/chat/completions"
        chat_attempt += 1
        if chat_attempt <= 2:
            assert request["tool_choice"] == {
                "type": "function",
                "function": {"name": "git_status"},
            }
            assert [
                tool["function"]["name"] for tool in request["tools"]
            ] == ["git_status"]
            name = "search" if chat_attempt == 1 else "git_status"
            arguments = (
                {
                    "pattern": "TODO",
                    "glob": None,
                    "fixed_strings": None,
                }
                if name == "search"
                else {}
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
                                        "id": f"chat-forced-{chat_attempt}",
                                        "type": "function",
                                        "function": {
                                            "name": name,
                                            "arguments": json.dumps(arguments),
                                        },
                                    }
                                ],
                            }
                        }
                    ]
                }
            )

        assert "tool_choice" not in request
        assert {
            tool["function"]["name"] for tool in request["tools"]
        } == {definition.name for definition in TOOL_DEFINITIONS}
        return json_response(
            {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "done",
                        }
                    }
                ]
            }
        )

    with fake_openai_server(responder) as (base_url, requests):
        provider = OpenAIProvider(
            config(base_url),
            idempotency_key="run-chat-sticky:implement:0",
        )
        provider.set_forced_tool_choice("git_status")
        with pytest.raises(
            OpenAIUnknownToolError, match="unknown or inactive tool"
        ):
            provider.next_step("inspect", ())
        assert provider.api_kind == "chat_completions"
        call = provider.next_step("inspect", ())
        assert call == ToolCall("chat-forced-2", "git_status", {})
        provider.mark_unknown_tool_recovered()
        final = provider.next_step(
            "inspect", (ToolResult(call.id, call.name, True, "clean"),)
        )

    assert final == FinalAnswer("done")
    assert [request["path"] for request in requests] == [
        "/responses",
        "/chat/completions",
        "/chat/completions",
        "/chat/completions",
    ]
    assert provider.rejected_tool_call_violations == 1
    assert provider.unknown_tool_violations == 1
    assert provider.recovered_unknown_tool_violations == 1
    assert provider.parallel_tool_call_violations == 0
    assert [
        request["headers"]["idempotency-key"] for request in requests
    ] == [
        "run-chat-sticky:implement:0:responses:0",
        "run-chat-sticky:implement:0:chat:0",
        "run-chat-sticky:implement:0:chat:1",
        "run-chat-sticky:implement:0:chat:2",
    ]


def test_chat_forced_tool_choice_rejects_final_text_and_survives_retry() -> None:
    chat_attempt = 0

    def responder(
        path: str,
        _headers: Mapping[str, str],
        raw: bytes,
        _index: int,
    ) -> ResponseSpec:
        nonlocal chat_attempt
        if path == "/responses":
            return json_response({"error": {"code": "unknown_endpoint"}}, 404)
        chat_attempt += 1
        request = json.loads(raw)
        assert request["tool_choice"] == {
            "type": "function",
            "function": {"name": "git_status"},
        }
        if chat_attempt == 1:
            return json_response(
                {
                    "choices": [
                        {
                            "message": {
                                "role": "assistant",
                                "content": "done",
                            }
                        }
                    ]
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
                                    "id": "chat-forced-after-text",
                                    "type": "function",
                                    "function": {
                                        "name": "git_status",
                                        "arguments": "{}",
                                    },
                                }
                            ],
                        }
                    }
                ]
            }
        )

    with fake_openai_server(responder) as (base_url, requests):
        provider = OpenAIProvider(
            config(base_url),
            idempotency_key="forced-text:chat",
        )
        provider.set_forced_tool_choice("git_status")
        with pytest.raises(OpenAIProtocolError, match="forced tool"):
            provider.next_step("inspect", ())
        assert provider.api_kind == "chat_completions"
        decision = provider.next_step("inspect", ())

    assert decision == ToolCall("chat-forced-after-text", "git_status", {})
    assert provider.rejected_tool_call_violations == 1
    assert [request["path"] for request in requests] == [
        "/responses",
        "/chat/completions",
        "/chat/completions",
    ]
    assert [
        request["headers"]["idempotency-key"] for request in requests
    ] == [
        "forced-text:chat:responses:0",
        "forced-text:chat:chat:0",
        "forced-text:chat:chat:1",
    ]


def test_chat_malformed_usage_counts_unexecuted_tool_call() -> None:
    def responder(
        path: str,
        _headers: Mapping[str, str],
        _raw: bytes,
        _index: int,
    ) -> ResponseSpec:
        if path == "/responses":
            return json_response({"error": {"code": "unknown_endpoint"}}, 404)
        return json_response(
            {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "chat-usage-call",
                                    "type": "function",
                                    "function": {
                                        "name": "git_status",
                                        "arguments": "{}",
                                    },
                                }
                            ],
                        }
                    }
                ],
                "usage": {
                    "prompt_tokens": "invalid",
                    "completion_tokens": 2,
                    "total_tokens": 2,
                },
            }
        )

    with fake_openai_server(responder) as (base_url, _):
        provider = OpenAIProvider(config(base_url))
        with pytest.raises(OpenAIProtocolError, match="usage is malformed"):
            provider.next_step("inspect", ())

    assert provider.api_kind == "chat_completions"
    assert provider.rejected_tool_call_violations == 1
    assert provider.parallel_tool_call_violations == 0


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


def test_rejected_apply_patch_arguments_are_compacted_in_responses_replay() -> None:
    rejected_diff = "diff --git a/app.py b/app.py\n" + ("+large-change\n" * 2_000)

    def responder(
        path: str,
        _headers: Mapping[str, str],
        raw: bytes,
        index: int,
    ) -> ResponseSpec:
        assert path == "/responses"
        request = json.loads(raw)
        if index == 0:
            return json_response(
                {
                    "output": [
                        {
                            "type": "function_call",
                            "id": "responses-patch-item",
                            "call_id": "responses-patch-call",
                            "name": "apply_patch",
                            "arguments": json.dumps(
                                {"unified_diff": rejected_diff},
                                separators=(",", ":"),
                            ),
                        }
                    ]
                }
            )

        assert index == 1
        replayed = request["input"]
        function_call = next(
            item
            for item in replayed
            if item.get("type") == "function_call"
        )
        assert function_call["call_id"] == "responses-patch-call"
        assert json.loads(function_call["arguments"]) == {
            "unified_diff": "[rejected patch omitted from replay]"
        }
        assert rejected_diff not in json.dumps(replayed)
        output = next(
            item
            for item in replayed
            if item.get("type") == "function_call_output"
        )
        serialized_result = json.loads(output["output"])
        assert serialized_result["ok"] is False
        assert serialized_result["error"] == "patch rejected by test"
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
        call = provider.next_step("apply the change", ())
        assert isinstance(call, ToolCall)
        final = provider.next_step(
            "apply the change",
            (
                ToolResult(
                    call.id,
                    call.name,
                    False,
                    "",
                    error="patch rejected by test",
                    exit_code=1,
                ),
            ),
        )

    assert final == FinalAnswer("done")


def test_rejected_write_files_arguments_are_compacted_for_both_replays() -> None:
    provider = OpenAIProvider(config("http://127.0.0.1:1"))
    raw_arguments = json.dumps(
        {"files": [{"path": "app.py", "content": "sensitive contents"}]}
    )
    provider._responses_input = [  # noqa: SLF001 - replay regression boundary
        {
            "type": "function_call",
            "call_id": "write-call",
            "name": "write_files",
            "arguments": raw_arguments,
        }
    ]
    provider._chat_messages = [  # noqa: SLF001 - replay regression boundary
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "write-call",
                    "type": "function",
                    "function": {
                        "name": "write_files",
                        "arguments": raw_arguments,
                    },
                }
            ],
        }
    ]

    provider._compact_rejected_mutation_call(  # noqa: SLF001
        "write-call", "write_files"
    )

    placeholder = {
        "files": [
            {
                "path": "__rejected_write_omitted__.txt",
                "content": "[rejected file contents omitted from replay]",
            }
        ]
    }
    assert json.loads(provider._responses_input[0]["arguments"]) == placeholder  # type: ignore[index]  # noqa: SLF001
    replayed = provider._chat_messages[0]["tool_calls"][0]["function"][  # type: ignore[index]  # noqa: SLF001
        "arguments"
    ]
    assert json.loads(replayed) == placeholder
    assert "sensitive contents" not in json.dumps(provider._responses_input)  # noqa: SLF001
    assert "sensitive contents" not in json.dumps(provider._chat_messages)  # noqa: SLF001


def test_rejected_apply_patch_arguments_are_compacted_in_chat_replay() -> None:
    rejected_diff = "diff --git a/app.py b/app.py\n" + ("+large-change\n" * 2_000)

    def responder(
        path: str,
        _headers: Mapping[str, str],
        raw: bytes,
        index: int,
    ) -> ResponseSpec:
        request = json.loads(raw)
        if index == 0:
            assert path == "/responses"
            return json_response({"error": {"code": "unknown_endpoint"}}, 404)

        assert path == "/chat/completions"
        if index == 1:
            return json_response(
                {
                    "choices": [
                        {
                            "message": {
                                "role": "assistant",
                                "content": None,
                                "tool_calls": [
                                    {
                                        "id": "chat-patch-call",
                                        "type": "function",
                                        "function": {
                                            "name": "apply_patch",
                                            "arguments": json.dumps(
                                                {"unified_diff": rejected_diff},
                                                separators=(",", ":"),
                                            ),
                                        },
                                    }
                                ],
                            }
                        }
                    ]
                }
            )

        assert index == 2
        assistant = request["messages"][-2]
        assert assistant["role"] == "assistant"
        tool_call = assistant["tool_calls"][0]
        assert tool_call["id"] == "chat-patch-call"
        assert json.loads(tool_call["function"]["arguments"]) == {
            "unified_diff": "[rejected patch omitted from replay]"
        }
        assert rejected_diff not in json.dumps(assistant)
        tool_result = request["messages"][-1]
        serialized_result = json.loads(tool_result["content"])
        assert serialized_result["ok"] is False
        assert serialized_result["error"] == "patch rejected by test"
        return json_response(
            {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "done",
                        }
                    }
                ]
            }
        )

    with fake_openai_server(responder) as (base_url, _):
        provider = OpenAIProvider(config(base_url))
        call = provider.next_step("apply the change", ())
        assert isinstance(call, ToolCall)
        final = provider.next_step(
            "apply the change",
            (
                ToolResult(
                    call.id,
                    call.name,
                    False,
                    "",
                    error="patch rejected by test",
                    exit_code=1,
                ),
            ),
        )

    assert final == FinalAnswer("done")


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


def test_successful_http_protocol_error_rotates_derived_idempotency_key() -> None:
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
        "run-retry:plan:2:responses:1",
    ]


def test_http_error_retry_reuses_derived_idempotency_key() -> None:
    def responder(
        _path: str,
        _headers: Mapping[str, str],
        _raw: bytes,
        index: int,
    ) -> ResponseSpec:
        if index == 0:
            return json_response({"error": {"code": "temporary_failure"}}, 503)
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
        provider = OpenAIProvider(
            config(base_url),
            idempotency_key="run-http-retry:plan:0",
        )
        with pytest.raises(OpenAIHTTPError):
            provider.next_step("inspect", ())
        assert provider.next_step("inspect", ()) == FinalAnswer("done")

    assert [
        request["headers"]["idempotency-key"] for request in requests
    ] == [
        "run-http-retry:plan:0:responses:0",
        "run-http-retry:plan:0:responses:0",
    ]


def test_transport_error_retry_reuses_derived_idempotency_key() -> None:
    provider = OpenAIProvider(
        config("http://127.0.0.1:8123"),
        idempotency_key="run-transport-retry:plan:0",
    )
    keys: list[str | None] = []

    def responder(
        _path: str,
        _payload: Mapping[str, object],
        *,
        idempotency_key: str | None = None,
    ) -> dict[str, object]:
        keys.append(idempotency_key)
        if len(keys) == 1:
            raise OpenAITransportError("temporary transport failure")
        return {
            "output": [
                {
                    "type": "message",
                    "content": [{"type": "output_text", "text": "done"}],
                }
            ]
        }

    provider._client.post_json = responder  # type: ignore[assignment]

    with pytest.raises(OpenAITransportError):
        provider.next_step("inspect", ())
    assert provider.next_step("inspect", ()) == FinalAnswer("done")
    assert keys == [
        "run-transport-retry:plan:0:responses:0",
        "run-transport-retry:plan:0:responses:0",
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
    total_budgets = (6_000, 6_000, 6_000)

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


def test_cancellation_check_prevents_chat_fallback_request() -> None:
    class Cancelled(RuntimeError):
        pass

    def responder(
        path: str,
        _headers: Mapping[str, str],
        _raw: bytes,
        _index: int,
    ) -> ResponseSpec:
        assert path == "/responses"
        return json_response({"error": {"code": "unknown_endpoint"}}, 404)

    checks = 0

    def check_cancelled() -> None:
        nonlocal checks
        checks += 1
        raise Cancelled("cancelled before fallback")

    with fake_openai_server(responder) as (base_url, requests):
        provider = OpenAIProvider(config(base_url))
        provider.set_cancellation_check(check_cancelled)
        with pytest.raises(Cancelled, match="before fallback"):
            provider.next_step("inspect", ())

    assert checks == 1
    assert provider.api_kind == "chat_completions"
    assert provider.request_count == 1
    assert [request["path"] for request in requests] == ["/responses"]


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
    assert provider.request_count == 1
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

    assert provider.rejected_tool_call_violations == 2
    assert provider.parallel_tool_call_violations == 0


def test_malformed_tool_call_is_counted_as_rejected() -> None:
    def responder(*_: object) -> ResponseSpec:
        return json_response(
            {
                "output": [
                    {
                        "type": "function_call",
                        "call_id": "malformed-call",
                        "name": "git_status",
                        "arguments": {},
                    }
                ]
            }
        )

    with fake_openai_server(responder) as (base_url, _):
        provider = OpenAIProvider(config(base_url))
        with pytest.raises(OpenAIProtocolError, match="JSON string") as raised:
            provider.next_step("inspect", ())

    assert not isinstance(raised.value, OpenAIToolArgumentsError)
    assert provider.rejected_tool_call_violations == 1
    assert provider.parallel_tool_call_violations == 0

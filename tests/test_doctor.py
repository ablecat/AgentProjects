from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import threading
from typing import Any

import pytest

from repo_agent.doctor import OpenAIDoctorError, run_doctor
from repo_agent.openai_provider import (
    OpenAIConfig,
    OpenAIHTTPError,
    OpenAIProtocolError,
)


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


def config(base_url: str) -> OpenAIConfig:
    return OpenAIConfig(
        api_key="fake-doctor-key",
        base_url=base_url,
        model="fake-doctor-model",
        timeout_seconds=1,
    )


def _challenge_from_responses_request(request: Mapping[str, Any]) -> str:
    tool = request["tools"][0]
    return tool["parameters"]["properties"]["challenge"]["enum"][0]


def _challenge_from_chat_request(request: Mapping[str, Any]) -> str:
    tool = request["tools"][0]["function"]
    return tool["parameters"]["properties"]["challenge"]["enum"][0]


def _confirmation_from_output(raw_output: str) -> str:
    return json.loads(raw_output)["confirmation"]


def test_responses_doctor_preserves_reasoning_call_and_exact_call_id() -> None:
    reasoning_item = {
        "type": "reasoning",
        "id": "reasoning-1",
        "summary": [{"type": "summary_text", "text": "Calling doctor tool."}],
    }
    function_call: dict[str, object] = {}

    def responder(
        path: str,
        headers: Mapping[str, str],
        raw: bytes,
        index: int,
    ) -> ResponseSpec:
        assert path == "/v1/responses"
        assert headers["authorization"] == "Bearer fake-doctor-key"
        request = json.loads(raw)
        assert request["parallel_tool_calls"] is False
        tool = request["tools"][0]
        assert tool["strict"] is True
        assert tool["parameters"]["additionalProperties"] is False
        assert tool["parameters"]["required"] == ["challenge"]

        if index == 0:
            challenge = _challenge_from_responses_request(request)
            function_call.update(
                {
                    "type": "function_call",
                    "id": "function-item-1",
                    "call_id": "exact-call-id-1",
                    "name": "repo_agent_doctor_check",
                    "arguments": json.dumps({"challenge": challenge}),
                    "status": "completed",
                }
            )
            return json_response({"output": [reasoning_item, function_call]})

        assert index == 1
        continued_input = request["input"]
        assert continued_input[0]["role"] == "user"
        assert continued_input[1] == reasoning_item
        assert continued_input[2] == function_call
        tool_output = continued_input[3]
        assert tool_output["type"] == "function_call_output"
        assert tool_output["call_id"] == "exact-call-id-1"
        confirmation = _confirmation_from_output(tool_output["output"])
        return json_response(
            {
                "output": [
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [
                            {"type": "output_text", "text": confirmation}
                        ],
                    }
                ]
            }
        )

    with fake_openai_server(responder) as (base_url, requests):
        result = run_doctor(config(f"{base_url}/v1"))

    assert result.ok is True
    assert result.api_kind == "responses"
    assert result.api == "responses"
    assert result.tool_call_verified is True
    assert result.final_response_verified is True
    assert result.http_requests == 2
    assert len(requests) == 2
    assert "fake-doctor-key" not in repr(result)


@pytest.mark.parametrize(
    "unsupported_status,error_code",
    [(404, "not_found"), (405, "method_not_allowed"), (400, "unsupported_endpoint")],
)
def test_doctor_falls_back_to_chat_for_clear_capability_errors(
    unsupported_status: int,
    error_code: str,
) -> None:
    assistant_message: dict[str, object] = {}

    def responder(
        path: str,
        headers: Mapping[str, str],
        raw: bytes,
        index: int,
    ) -> ResponseSpec:
        if index == 0:
            assert path == "/responses"
            return json_response(
                {"error": {"code": error_code}},
                status=unsupported_status,
            )

        assert path == "/chat/completions"
        request = json.loads(raw)
        assert request["parallel_tool_calls"] is False
        tool = request["tools"][0]
        assert tool["type"] == "function"
        assert tool["function"]["strict"] is True
        if index == 1:
            challenge = _challenge_from_chat_request(request)
            assistant_message.update(
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "exact-chat-call-id",
                            "type": "function",
                            "function": {
                                "name": "repo_agent_doctor_check",
                                "arguments": json.dumps({"challenge": challenge}),
                            },
                        }
                    ],
                }
            )
            return json_response({"choices": [{"message": assistant_message}]})

        assert index == 2
        messages = request["messages"]
        assert messages[-2] == assistant_message
        tool_result = messages[-1]
        assert tool_result["role"] == "tool"
        assert tool_result["tool_call_id"] == "exact-chat-call-id"
        confirmation = _confirmation_from_output(tool_result["content"])
        return json_response(
            {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": confirmation,
                        }
                    }
                ]
            }
        )

    with fake_openai_server(responder) as (base_url, requests):
        result = run_doctor(config(base_url))

    assert result.api_kind == "chat_completions"
    assert result.http_requests == 3
    assert [request["path"] for request in requests] == [
        "/responses",
        "/chat/completions",
        "/chat/completions",
    ]


@pytest.mark.parametrize("status", [400, 401, 403, 429, 500])
def test_doctor_does_not_fallback_for_non_capability_http_errors(status: int) -> None:
    def responder(*_: object) -> ResponseSpec:
        return json_response(
            {
                "error": {
                    "code": "request_failed",
                    "message": "fake-doctor-key must remain private",
                }
            },
            status=status,
        )

    with fake_openai_server(responder) as (base_url, requests):
        with pytest.raises(OpenAIHTTPError) as raised:
            run_doctor(config(base_url))

    assert raised.value.status_code == status
    assert "fake-doctor-key" not in str(raised.value)
    assert len(requests) == 1


def test_doctor_does_not_fallback_when_responses_returns_text_only() -> None:
    def responder(*_: object) -> ResponseSpec:
        return json_response(
            {
                "output": [
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [
                            {"type": "output_text", "text": "Everything works."}
                        ],
                    }
                ]
            }
        )

    with fake_openai_server(responder) as (base_url, requests):
        with pytest.raises(OpenAIDoctorError, match="native function call"):
            run_doctor(config(base_url))

    assert len(requests) == 1


def test_doctor_does_not_fallback_on_invalid_responses_json() -> None:
    def responder(*_: object) -> ResponseSpec:
        return (200, {"Content-Type": "application/json"}, b"not-json")

    with fake_openai_server(responder) as (base_url, requests):
        with pytest.raises(OpenAIProtocolError, match="invalid JSON"):
            run_doctor(config(base_url))

    assert len(requests) == 1


def test_doctor_rejects_wrong_challenge_without_a_second_request() -> None:
    def responder(*_: object) -> ResponseSpec:
        return json_response(
            {
                "output": [
                    {
                        "type": "function_call",
                        "call_id": "call-wrong-challenge",
                        "name": "repo_agent_doctor_check",
                        "arguments": json.dumps({"challenge": "wrong"}),
                    }
                ]
            }
        )

    with fake_openai_server(responder) as (base_url, requests):
        with pytest.raises(OpenAIDoctorError, match="preserve the challenge"):
            run_doctor(config(base_url))

    assert len(requests) == 1


def test_doctor_requires_final_text_to_confirm_the_tool_output() -> None:
    def responder(
        path: str,
        headers: Mapping[str, str],
        raw: bytes,
        index: int,
    ) -> ResponseSpec:
        request = json.loads(raw)
        if index == 0:
            challenge = _challenge_from_responses_request(request)
            return json_response(
                {
                    "output": [
                        {
                            "type": "function_call",
                            "call_id": "call-final-check",
                            "name": "repo_agent_doctor_check",
                            "arguments": json.dumps({"challenge": challenge}),
                        }
                    ]
                }
            )
        return json_response(
            {
                "output": [
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [
                            {"type": "output_text", "text": "unrelated text"}
                        ],
                    }
                ]
            }
        )

    with fake_openai_server(responder) as (base_url, requests):
        with pytest.raises(OpenAIDoctorError, match="did not confirm"):
            run_doctor(config(base_url))

    assert len(requests) == 2

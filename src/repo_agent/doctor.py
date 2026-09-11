"""Connectivity and native tool-calling verification for model endpoints."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
import json
import secrets
from typing import Literal

from .openai_provider import (
    APIKind,
    OpenAIConfig,
    OpenAIHTTPClient,
    OpenAIHTTPError,
    OpenAIProtocolError,
    _chat_content_text,
    _decode_json_value,
    _protocol_identifier,
    _responses_output_text,
)


_TOOL_NAME = "repo_agent_doctor_check"
_SYSTEM_PROMPT = (
    "You are verifying an Agent API connection. Follow the requested function "
    "call exactly. Do not reveal credentials or add unrelated content."
)


class OpenAIDoctorError(OpenAIProtocolError):
    """The endpoint responded, but failed the native tool-call handshake."""


@dataclass(frozen=True, slots=True)
class DoctorResult:
    """A safe-to-display summary of a successful doctor handshake."""

    ok: Literal[True]
    api_kind: APIKind
    model: str
    tool_call_verified: Literal[True]
    final_response_verified: Literal[True]
    http_requests: int
    message: str

    @property
    def api(self) -> APIKind:
        """Compatibility alias for display code."""

        return self.api_kind


def run_doctor(config: OpenAIConfig) -> DoctorResult:
    """Prove a complete native function call, preferring Responses.

    Chat Completions is attempted only when the initial Responses request
    receives a clear endpoint-capability error. Authentication, rate-limit,
    timeout, transport, JSON, and protocol failures are never hidden by a
    fallback attempt.
    """

    if not isinstance(config, OpenAIConfig):
        raise TypeError("config must be an OpenAIConfig")
    client = OpenAIHTTPClient(config)
    challenge = f"challenge-{secrets.token_hex(16)}"
    confirmation = f"REPO_AGENT_DOCTOR_OK_{secrets.token_hex(16)}"

    try:
        _run_responses_doctor(client, challenge, confirmation)
    except OpenAIHTTPError as exc:
        if not exc.endpoint_unsupported:
            raise
        _run_chat_doctor(client, challenge, confirmation)
        return DoctorResult(
            ok=True,
            api_kind="chat_completions",
            model=config.model,
            tool_call_verified=True,
            final_response_verified=True,
            http_requests=3,
            message="Chat Completions native function calling verified",
        )

    return DoctorResult(
        ok=True,
        api_kind="responses",
        model=config.model,
        tool_call_verified=True,
        final_response_verified=True,
        http_requests=2,
        message="Responses API native function calling verified",
    )


def doctor(config: OpenAIConfig) -> DoctorResult:
    """Alias suitable for direct imports by a future CLI command."""

    return run_doctor(config)


def _run_responses_doctor(
    client: OpenAIHTTPClient,
    challenge: str,
    confirmation: str,
) -> None:
    tool = _responses_tool(challenge)
    instructions = _doctor_instructions(confirmation)
    input_items: list[object] = [
        {
            "role": "user",
            "content": [
                {
                    "type": "input_text",
                    "text": (
                        f"Call {_TOOL_NAME} exactly once with challenge "
                        f"{challenge!r}. Do not answer with text before the "
                        "function result."
                    ),
                }
            ],
        }
    ]
    first = client.post_json(
        "/responses",
        {
            "model": client.config.model,
            "instructions": instructions,
            "input": deepcopy(input_items),
            "tools": [deepcopy(tool)],
            "tool_choice": {"type": "function", "name": _TOOL_NAME},
            "parallel_tool_calls": False,
            "store": False,
        },
    )
    output = first.get("output")
    if not isinstance(output, list) or not all(
        isinstance(item, Mapping) for item in output
    ):
        raise OpenAIDoctorError("Responses doctor output is malformed")
    function_call = _single_responses_function_call(output)
    call_id, arguments = _responses_call_details(function_call)
    _verify_challenge(arguments, challenge)

    tool_output = json.dumps(
        {"ok": True, "confirmation": confirmation},
        ensure_ascii=True,
        separators=(",", ":"),
    )
    continued_input: list[object] = deepcopy(input_items)
    continued_input.extend(deepcopy(output))
    continued_input.append(
        {
            "type": "function_call_output",
            "call_id": call_id,
            "output": tool_output,
        }
    )
    second = client.post_json(
        "/responses",
        {
            "model": client.config.model,
            "instructions": instructions,
            "input": continued_input,
            "tools": [deepcopy(tool)],
            "parallel_tool_calls": False,
            "store": False,
        },
    )
    second_output = second.get("output")
    if not isinstance(second_output, list):
        raise OpenAIDoctorError("Responses doctor continuation is malformed")
    if any(
        isinstance(item, Mapping) and item.get("type") == "function_call"
        for item in second_output
    ):
        raise OpenAIDoctorError("Responses doctor requested an unexpected second tool")
    final_text = _responses_output_text(second, second_output).strip()
    _verify_confirmation(final_text, confirmation)


def _run_chat_doctor(
    client: OpenAIHTTPClient,
    challenge: str,
    confirmation: str,
) -> None:
    response_tool = _responses_tool(challenge)
    chat_tool = {
        "type": "function",
        "function": {
            key: deepcopy(response_tool[key])
            for key in ("name", "description", "parameters", "strict")
        },
    }
    messages: list[dict[str, object]] = [
        {"role": "system", "content": _doctor_instructions(confirmation)},
        {
            "role": "user",
            "content": (
                f"Call {_TOOL_NAME} exactly once with challenge {challenge!r}. "
                "Do not answer with text before the function result."
            ),
        },
    ]
    first = client.post_json(
        "/chat/completions",
        {
            "model": client.config.model,
            "messages": deepcopy(messages),
            "tools": [deepcopy(chat_tool)],
            "tool_choice": {
                "type": "function",
                "function": {"name": _TOOL_NAME},
            },
            "parallel_tool_calls": False,
            "store": False,
        },
    )
    assistant_message = _first_chat_message(first)
    call_id, arguments = _chat_call_details(assistant_message)
    _verify_challenge(arguments, challenge)

    tool_output = json.dumps(
        {"ok": True, "confirmation": confirmation},
        ensure_ascii=True,
        separators=(",", ":"),
    )
    continued_messages: list[object] = [deepcopy(message) for message in messages]
    continued_messages.append(deepcopy(dict(assistant_message)))
    continued_messages.append(
        {
            "role": "tool",
            "tool_call_id": call_id,
            "content": tool_output,
        }
    )
    second = client.post_json(
        "/chat/completions",
        {
            "model": client.config.model,
            "messages": continued_messages,
            "tools": [deepcopy(chat_tool)],
            "parallel_tool_calls": False,
            "store": False,
        },
    )
    final_message = _first_chat_message(second)
    raw_calls = final_message.get("tool_calls")
    if isinstance(raw_calls, list) and raw_calls:
        raise OpenAIDoctorError("Chat doctor requested an unexpected second tool")
    final_text = _chat_content_text(final_message.get("content")).strip()
    _verify_confirmation(final_text, confirmation)


def _responses_tool(challenge: str) -> dict[str, object]:
    return {
        "type": "function",
        "name": _TOOL_NAME,
        "description": "Return the exact one-time doctor challenge.",
        "parameters": {
            "type": "object",
            "properties": {
                "challenge": {
                    "type": "string",
                    "enum": [challenge],
                    "description": "The exact challenge from the user request.",
                }
            },
            "required": ["challenge"],
            "additionalProperties": False,
        },
        "strict": True,
    }


def _doctor_instructions(confirmation: str) -> str:
    return (
        f"{_SYSTEM_PROMPT} After the function result arrives, read its "
        f"confirmation field and reply with exactly {confirmation!r}."
    )


def _single_responses_function_call(
    output: Sequence[Mapping[str, object]],
) -> Mapping[str, object]:
    calls = [item for item in output if item.get("type") == "function_call"]
    if len(calls) != 1:
        raise OpenAIDoctorError(
            "Responses doctor did not return exactly one native function call"
        )
    return calls[0]


def _responses_call_details(
    function_call: Mapping[str, object],
) -> tuple[str, Mapping[str, object]]:
    if function_call.get("name") != _TOOL_NAME:
        raise OpenAIDoctorError("Responses doctor returned the wrong function name")
    call_id = _protocol_identifier(function_call.get("call_id"), "tool call id")
    raw_arguments = function_call.get("arguments")
    if not isinstance(raw_arguments, str):
        raise OpenAIDoctorError("Responses doctor arguments are not a JSON string")
    arguments = _decode_json_value(raw_arguments, "doctor tool arguments")
    if not isinstance(arguments, Mapping):
        raise OpenAIDoctorError("Responses doctor arguments are not an object")
    return call_id, arguments


def _first_chat_message(payload: Mapping[str, object]) -> Mapping[str, object]:
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        raise OpenAIDoctorError("Chat doctor response is missing choices")
    first = choices[0]
    if not isinstance(first, Mapping):
        raise OpenAIDoctorError("Chat doctor choice is malformed")
    message = first.get("message")
    if not isinstance(message, Mapping) or message.get("role") != "assistant":
        raise OpenAIDoctorError("Chat doctor assistant message is malformed")
    return message


def _chat_call_details(
    assistant_message: Mapping[str, object],
) -> tuple[str, Mapping[str, object]]:
    raw_calls = assistant_message.get("tool_calls")
    if not isinstance(raw_calls, list) or len(raw_calls) != 1:
        raise OpenAIDoctorError(
            "Chat doctor did not return exactly one native function call"
        )
    raw_call = raw_calls[0]
    if not isinstance(raw_call, Mapping) or raw_call.get("type") != "function":
        raise OpenAIDoctorError("Chat doctor tool call is malformed")
    function = raw_call.get("function")
    if not isinstance(function, Mapping) or function.get("name") != _TOOL_NAME:
        raise OpenAIDoctorError("Chat doctor returned the wrong function name")
    call_id = _protocol_identifier(raw_call.get("id"), "tool call id")
    raw_arguments = function.get("arguments")
    if not isinstance(raw_arguments, str):
        raise OpenAIDoctorError("Chat doctor arguments are not a JSON string")
    arguments = _decode_json_value(raw_arguments, "doctor tool arguments")
    if not isinstance(arguments, Mapping):
        raise OpenAIDoctorError("Chat doctor arguments are not an object")
    return call_id, arguments


def _verify_challenge(arguments: Mapping[str, object], challenge: str) -> None:
    if set(arguments) != {"challenge"} or arguments.get("challenge") != challenge:
        raise OpenAIDoctorError("doctor function call did not preserve the challenge")


def _verify_confirmation(final_text: str, confirmation: str) -> None:
    if final_text != confirmation:
        raise OpenAIDoctorError(
            "doctor final response did not confirm the function result"
        )


__all__ = [
    "DoctorResult",
    "OpenAIDoctorError",
    "doctor",
    "run_doctor",
]

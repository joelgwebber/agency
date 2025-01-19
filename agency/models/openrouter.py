from __future__ import annotations

import json
from typing import Any, Dict, List, Literal, NotRequired, Optional, TypedDict, Union

import requests

from agency.keys import OPENROUTER_API_KEY
from agency.models.llm import LLM, Function, FunctionCall, Message, Role
from agency.models.openapi import OpenAPISchema


class OpenRouterLLM(LLM):
    """OpenRouter implementation of the LLM interface."""

    def __init__(self, model: str = "openai/gpt-3.5-turbo"):
        self.model = model

    def complete(
        self,
        messages: List[Message],
        functions: Optional[List[Function]] = None,
    ) -> Message:
        # Convert our messages to OpenRouter format
        or_messages = []

        # Always start with system message requiring JSON
        # This is particularly important because OpenRouter won't let you request json output, without
        # the word 'json' appearing somewhere in the prompt. Feels like a hack.
        or_messages.append(
            {
                "role": "system",
                "content": "Always return precisely one correctly-structured json output; never raw text.",
            }
        )

        # Convert our messages, ensuring content is never null
        for msg in messages:
            or_message: Dict[str, Any] = {}
            if msg.role == Role.TOOL and msg.function:
                # Tool responses need special formatting
                or_message = {
                    "role": "tool",
                    "content": json.dumps(msg.function.arguments),
                    "tool_call_id": msg.function.id,
                    "name": msg.function.name,
                }
            else:
                # Regular messages
                content = msg.content if msg.content is not None else ""
                or_message = {"role": msg.role.value, "content": content}

                # Add tool_calls for assistant messages with function calls
                if msg.role == Role.ASSISTANT and msg.function:
                    tool_call = {
                        "id": msg.function.id,
                        "type": "function",
                        "function": {
                            "name": msg.function.name,
                            "arguments": json.dumps(msg.function.arguments),
                        },
                    }
                    or_message["tool_calls"] = [tool_call]

            or_messages.append(or_message)

        # Convert our functions to OpenRouter format
        or_functions = (
            [
                {
                    "type": "function",
                    "function": {
                        "name": f.name,
                        "description": f.description,
                        "parameters": f.parameters,
                    },
                }
                for f in functions
            ]
            if functions
            else None
        )

        # Make the API call
        rsp = requests.post(
            url="https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "HTTP-Referer": "j15r.com",
                "X-Title": "agency",
            },
            json={
                "model": self.model,
                "response_format": {"type": "json_object"},
                "messages": or_messages,
                "tools": or_functions,
            },
        ).json()

        if "error" in rsp:
            raise Exception(f"OpenRouter API error: {rsp['error']}")

        # Extract the completion
        completion = rsp["choices"][0]["message"]

        # Convert completion to Message
        msg = Message(role=Role.ASSISTANT, content=completion.get("content", ""))

        # Add function call if present
        if "tool_calls" in completion:
            tool_calls = completion["tool_calls"]
            if len(tool_calls) > 1:
                raise Exception(f"Expected at most 1 tool call, got {len(tool_calls)}")
            tool_call = tool_calls[0]
            msg.function = FunctionCall(
                id=tool_call["id"],
                name=tool_call["function"]["name"],
                arguments=json.loads(tool_call["function"]["arguments"]),
            )

        return msg


# TODO: Use these types from the OpenRouter docs explicitly in the dynamic code above.


class ORError(TypedDict):
    error: Dict[str, Any]


# https://openrouter.ai/docs/requests
class ORRequest(TypedDict):
    # Either messages or prompt is required.
    messages: NotRequired[List[Message]]
    prompt: NotRequired[str]

    model: NotRequired[str]
    response_format: NotRequired[ORResponseFormat]
    stop: NotRequired[Union[str, List[str]]]
    stream: NotRequired[bool]

    max_tokens: NotRequired[int]
    temperature: NotRequired[float]

    tools: NotRequired[List[ORToolDesc]]
    tool_choice: NotRequired[Union[str, ORToolChoice]]  # TODO: WTF does this do?

    seed: NotRequired[int]
    top_p: NotRequired[int]
    top_k: NotRequired[int]
    frequency_penalty: NotRequired[float]
    presence_penalty: NotRequired[float]
    repetition_penalty: NotRequired[float]
    logit_bias: NotRequired[Dict[int, float]]
    top_logprobs: NotRequired[int]
    min_p: NotRequired[float]
    top_a: NotRequired[float]

    prediction: NotRequired[ORPrediction]

    transforms: NotRequired[List[str]]
    models: NotRequired[List[str]]
    route: NotRequired[Literal["fallback"]]
    provider_preferences: NotRequired[ORProviderPreferences]


class ORResponseFormat(TypedDict):
    type: Literal["json_object"]


# https://openrouter.ai/docs/responses
class ORResponse(TypedDict):
    id: str
    choices: List[ORChoice]
    created: int
    model: str
    object: Literal["chat.completion", "chat.completion.chunk"]
    system_fingerprint: NotRequired[str]
    usage: NotRequired[ORResponseUsage]


class ORChoice(TypedDict):
    finish_reason: NotRequired[str]
    error: NotRequired[str]
    text: NotRequired[str]  # For non-chat completion
    message: NotRequired[ORMessage]  # For non-streaming chats
    delta: NotRequired[ORDelta]  # For streaming chats


class ORMessage(TypedDict):
    role: Literal["user", "assistant", "system", "tool"]
    content: Union[str, List[ORPart]]
    name: NotRequired[str]  # For tool results
    tool_call_id: NotRequired[str]  # For tool results
    tool_calls: NotRequired[List[ORToolCall]]  # For tool calls


class ORToolCall(TypedDict):
    id: str
    type: Literal["function"]
    function: ORFunctionCall


class ORDelta(TypedDict):
    content: str
    role: NotRequired[str]
    tool_calls: NotRequired[List[ORToolCall]]


class ORFunctionCall(TypedDict):
    name: str
    arguments: str


class ORResponseUsage(TypedDict):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ORProviderPreferences(TypedDict):
    pass


class ORPrediction(TypedDict):
    type: Literal["content"]
    content: str


class ORTextPart(TypedDict):
    type: Literal["text"]
    text: str


class ORImagePart(TypedDict):
    type: Literal["image_url"]
    image_url: ORImageURL


ORPart = Union[ORTextPart, ORImagePart]


class ORImageURL(TypedDict):
    url: str
    detail: NotRequired[str]  # defaults to "auto"


class ORToolDesc(TypedDict):
    type: Literal["function"]
    function: ORFunctionDesc


class ORFunctionDesc(TypedDict):
    name: str
    description: NotRequired[str]
    parameters: OpenAPISchema


class ORToolChoice(TypedDict):
    type: Literal["function"]
    function: ORToolChoiceFunc


class ORToolChoiceFunc(TypedDict):
    name: str

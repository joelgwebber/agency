from __future__ import annotations

import json
from typing import (
    Any,
    Dict,
    List,
    Literal,
    NotRequired,
    Optional,
    TypedDict,
    Union,
    cast,
)

import requests

from agency.keys import OPENROUTER_API_KEY
from agency.models import Function, FunctionCall, Message, Model, Role
from agency.models.openapi import OpenAPISchema


class OpenRouter(Model):
    """OpenRouter implementation of the LLM interface."""

    _model_id: str

    def __init__(self, model_id: str = "openai/gpt-3.5-turbo"):
        self._model_id = model_id

    def complete(
        self,
        messages: List[Message],
        response: Optional[OpenAPISchema] = None,
        functions: Optional[List[Function]] = None,
    ) -> Message:
        """Complete a conversation using the OpenRouter API."""
        # Convert messages and functions to OpenRouter format
        or_messages = self._convert_messages(messages)
        or_functions = self._convert_functions(functions)

        # Build and send request
        request = self._build_request(or_messages, or_functions, response)
        rsp = requests.post(
            url="https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "HTTP-Referer": "j15r.com",
                "X-Title": "agency",
            },
            json=request,
        ).json()

        # Handle response
        return self._handle_response(rsp)

    def _convert_message(self, msg: Message) -> ORMessage:
        """Convert a single Message to OpenRouter format."""
        if msg.role == Role.TOOL and msg.function:
            return ORMessage(
                role="tool",
                content=json.dumps(msg.function.arguments),
                tool_call_id=msg.function.id,
                name=msg.function.name,
            )

        # Regular messages
        content = msg.content if msg.content is not None else ""
        if not isinstance(content, str):
            raise ValueError(f"Expected string content, got {type(content)}")

        or_message = ORMessage(role=msg.role.value, content=content)

        # Add tool_calls for assistant messages with function calls
        if msg.role == Role.ASSISTANT and msg.function:
            tool_call = ORToolCall(
                id=msg.function.id,
                type="function",
                function=ORFunctionCall(
                    name=msg.function.name, arguments=json.dumps(msg.function.arguments)
                ),
            )
            or_message["tool_calls"] = [tool_call]

        return or_message

    def _convert_messages(self, messages: List[Message]) -> List[ORMessage]:
        """Convert agency Messages to OpenRouter format."""
        return [self._convert_message(msg) for msg in messages]

    def _convert_functions(
        self, functions: Optional[List[Function]]
    ) -> Optional[List[ORToolDesc]]:
        """Convert agency Functions to OpenRouter format."""
        if not functions:
            return None

        return [
            ORToolDesc(
                type="function",
                function=ORFunctionDesc(
                    name=f.name, description=f.description, parameters=f.parameters
                ),
            )
            for f in functions
        ]

    def _build_request(
        self,
        messages: List[ORMessage],
        functions: Optional[List[ORToolDesc]],
        response: Optional[OpenAPISchema],
    ) -> ORRequest:
        """Build the OpenRouter API request."""
        request = ORRequest(
            model=self._model_id,
            messages=messages,
        )
        if functions:
            request["tools"] = functions
        if response:
            request["response_format"] = ORResponseFormat(
                type="json_schema",
                json_schema=ORResponseSchema(
                    name="response",  # TODO: Does having a real name help the model?
                    strict=True,
                    schema=response,
                ),
            )
        return request

    def _handle_response(self, rsp: Dict) -> Message:
        """Process OpenRouter API response into an agency Message."""
        if "error" in rsp:
            raise Exception(f"OpenRouter API error: {rsp['error']}")

        # Parse response and extract completion
        response: ORResponse = cast(ORResponse, rsp)
        if len(response["choices"]) == 0:
            raise Exception("OpenRouter provided no response choices")
        choice = response["choices"][0]
        if "message" not in choice:
            raise Exception("OpenRouter provided no message")

        completion: ORMessage = choice["message"]
        content = completion["content"]

        # Create base message
        msg = Message(role=Role.ASSISTANT)
        if content:
            if not isinstance(content, str):
                raise ValueError(
                    f"Expected string content in response, got {type(content)}"
                )
            msg.content = content

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


# OpenRouter API types:


class ORError(TypedDict):
    error: Dict[str, Any]


# https://openrouter.ai/docs/requests
class ORRequest(TypedDict):
    # Either messages or prompt is required.
    messages: NotRequired[List[ORMessage]]
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
    type: Literal["json_object", "json_schema"]
    json_schema: ORResponseSchema


class ORResponseSchema(TypedDict):
    name: NotRequired[str]
    strict: NotRequired[bool]
    schema: OpenAPISchema


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
    content: Optional[Union[str, List[ORPart]]]
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

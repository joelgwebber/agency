from __future__ import annotations

import json
from typing import Any, Dict, List, Literal, NotRequired, TypedDict, Union

import requests

from agency.keys import OPENROUTER_API_KEY

# Try to use this to keep it from dumping raw text or multiple json responses.
system_prompt: Message = {
    "role": "system",
    "content": """Always return precisely one correctly-structured json output; never raw text.""",
}


class Router:
    model: str

    def __init__(self, model: str = "openai/gpt-3.5-turbo"):
        self.model = model

    def send(self, input: List[Message], tools: List[ToolDesc]) -> Message:
        # Prepend the system input.
        messages = [system_prompt]
        messages.extend(input)

        rsp: Union[RouterError, RouterResponse] = requests.post(
            url="https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "HTTP-Referer": "j15r.com",
                "X-Title": "agency",
            },
            data=json.dumps(
                {
                    "model": self.model,
                    "response_format": {"type": "json_object"},
                    "messages": messages,
                    "tools": tools,
                }
            ),
        ).json()

        if "error" in rsp:
            raise Exception(f"error calling router: {rsp['error']}")

        choice = rsp["choices"][0]
        if "message" not in choice:
            raise Exception(f"expected 'message' in {rsp['choices']}")

        return choice["message"]


class RouterError(TypedDict):
    error: Dict[str, Any]


# https://openrouter.ai/docs/requests
class RouterRequest(TypedDict):
    # Either messages or prompt is required.
    messages: NotRequired[List[Message]]
    prompt: NotRequired[str]

    model: NotRequired[str]
    response_format: NotRequired[ResponseFormat]
    stop: NotRequired[Union[str, List[str]]]
    stream: NotRequired[bool]

    max_tokens: NotRequired[int]
    temperature: NotRequired[float]

    tools: NotRequired[List[ToolDesc]]
    tool_choice: NotRequired[Union[str, ToolChoice]]  # TODO: WTF does this do?

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

    prediction: NotRequired[Prediction]

    transforms: NotRequired[List[str]]
    models: NotRequired[List[str]]
    route: NotRequired[Literal["fallback"]]
    provider_preferences: NotRequired[ProviderPreferences]


class ResponseFormat(TypedDict):
    type: Literal["json_object"]


# https://openrouter.ai/docs/responses
class RouterResponse(TypedDict):
    id: str
    choices: List[Choice]
    created: int
    model: str
    object: Literal["chat.completion", "chat.completion.chunk"]
    system_fingerprint: NotRequired[str]
    usage: NotRequired[ResponseUsage]


class Choice(TypedDict):
    finish_reason: NotRequired[str]
    error: NotRequired[str]
    text: NotRequired[str]  # For non-chat completion
    message: NotRequired[Message]  # For non-streaming chats
    delta: NotRequired[Delta]  # For streaming chats


class Message(TypedDict):
    role: Literal["user", "assistant", "system", "tool"]
    content: Union[str, List[Part]]
    name: NotRequired[str]  # For tool results
    tool_call_id: NotRequired[str]  # For tool results
    tool_calls: NotRequired[List[ToolCall]]  # For tool calls


class ToolCall(TypedDict):
    id: str
    type: Literal["function"]
    function: FunctionCall


class Delta(TypedDict):
    content: str
    role: NotRequired[str]
    tool_calls: NotRequired[List[ToolCall]]


class FunctionCall(TypedDict):
    name: str
    arguments: str


class ResponseUsage(TypedDict):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ProviderPreferences(TypedDict):
    pass


class Prediction(TypedDict):
    type: Literal["content"]
    content: str


class TextPart(TypedDict):
    type: Literal["text"]
    text: str


class ImagePart(TypedDict):
    type: Literal["image_url"]
    image_url: ImageURL


Part = Union[TextPart, ImagePart]


class ImageURL(TypedDict):
    url: str
    detail: NotRequired[str]  # defaults to "auto"


class ToolDesc(TypedDict):
    type: Literal["function"]
    function: FunctionDesc


class FunctionDesc(TypedDict):
    name: str
    description: NotRequired[str]
    parameters: OpenAPISchema


class OpenAPISchema(TypedDict):
    type: str
    format: NotRequired[str]
    description: NotRequired[str]
    enum: NotRequired[list[Any]]
    properties: NotRequired[Dict[str, OpenAPISchema]]
    required: NotRequired[List[str]]
    items: NotRequired[OpenAPISchema]


class ToolChoice(TypedDict):
    type: Literal["function"]
    function: ToolChoiceFunc


class ToolChoiceFunc(TypedDict):
    name: str

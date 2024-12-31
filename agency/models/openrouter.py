from __future__ import annotations

import json
from typing import List, Optional

import requests

from agency.keys import OPENROUTER_API_KEY
from agency.models.llm import (
    LLM,
    Function,
    FunctionCall,
    FunctionResult,
    Message,
    Response,
)


class OpenRouterLLM(LLM):
    """OpenRouter implementation of the LLM interface."""

    def __init__(self, model: str = "openai/gpt-3.5-turbo"):
        self.model = model

    def complete(
        self,
        messages: List[Message],
        functions: Optional[List[Function]] = None,
        function_result: Optional[FunctionResult] = None,
    ) -> Response:
        # Convert our messages to OpenRouter format
        or_messages = []

        # Always start with system message requiring JSON
        # This is particularly important because OpenRouter won't let you request json outpu, without
        # the word 'json' appearing somewhere in the prompt. Feels like a hack.
        or_messages.append(
            {
                "role": "system",
                "content": "Always return precisely one correctly-structured json output; never raw text.",
            }
        )

        # Convert our messages
        for msg in messages:
            or_messages.append({"role": msg.role.value, "content": msg.content})

        # Add function result if provided
        if function_result:
            or_messages.append(
                {
                    "role": "tool",
                    "content": json.dumps(function_result.args),
                    "name": function_result.call_id,
                    "tool_call_id": function_result.call_id,
                }
            )

        # Convert our functions to OpenRouter format
        or_functions = [
            {
                "type": "function",
                "function": {
                    "name": f.name,
                    "description": f.description,
                    "parameters": f.parameters,
                },
            }
            for f in functions
        ] if functions else None

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

        # Handle function calls
        if "tool_calls" in completion:
            tool_calls = completion["tool_calls"]
            if len(tool_calls) > 1:
                raise Exception(f"Expected at most 1 tool call, got {len(tool_calls)}")
            tool_call = tool_calls[0]
            return Response(
                function=FunctionCall(
                    name=tool_call["function"]["name"],
                    args=json.loads(tool_call["function"]["arguments"]),
                    call_id=tool_call["id"],
                )
            )

        # Handle content
        return Response(content=completion["content"])

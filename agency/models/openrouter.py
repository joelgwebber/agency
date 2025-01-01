from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

import requests

from agency.keys import OPENROUTER_API_KEY
from agency.models.llm import LLM, Function, FunctionCall, Message, Role


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

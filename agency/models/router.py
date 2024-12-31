from __future__ import annotations

from typing import List

from agency.models.llm import LLM, Function, FunctionResult, Message, Role, ToolCall
from agency.models.openrouter import OpenRouterLLM


class Router:
    """Wrapper around LLM implementations that handles message conversion."""

    _llm: LLM

    def __init__(self, model: str = "openai/gpt-3.5-turbo"):
        self._llm = OpenRouterLLM(model)

    def send(self, messages: List[Message], functions: List[Function]) -> Message:
        # Convert messages to our format
        llm_messages = []
        result = None
        for msg in messages:
            if isinstance(msg, dict) and msg.get("role") == "tool":
                # Handle tool results
                llm_messages.append(
                    Message(role=Role.USER, content=str(msg.get("content", "")))
                )
                result = FunctionResult(
                    args=msg.get("content", {}), call_id=msg.get("tool_call_id", "")
                )
            else:
                # Handle normal messages
                content = msg.content if isinstance(msg, Message) else ""
                role_str = msg.get("role", "user") if isinstance(msg, dict) else "user"
                llm_messages.append(
                    Message(role=Role[role_str.upper()], content=content)
                )

        # Get completion
        response = self._llm.complete(
            messages=llm_messages, functions=functions, function_result=result
        )

        # Convert response back to old format
        if response.function:
            return Message(
                role=Role.ASSISTANT,
                content="",
                tool_call=ToolCall(
                    id=response.function.call_id,
                    name=response.function.name,
                    arguments=response.function.args,
                ),
            )
        else:
            return Message(role=Role.ASSISTANT, content=response.content or "")

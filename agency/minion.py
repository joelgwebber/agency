import json
import traceback
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

from jinja2 import Environment

from agency.models import Message, Part, TextPart, ToolDesc
from agency.tool import Tool, ToolCall, ToolDecl, ToolResult


@dataclass
class MinionDecl(ToolDecl):
    """Extension of a tool declaration to be used with minions, giving them a jinja template and list of available
    tools they can use."""

    template: str
    tools: List[ToolDecl]


class Minion(Tool):
    decl: ToolDecl
    _history: List[Message]
    _tools: List[ToolDesc]

    def __init__(self, decl: MinionDecl, tools: List[Tool]):
        self.decl = decl
        self._history = []
        self._template = Environment().from_string(decl.template)
        self._tools = [
            {
                "type": "function",
                "function": tool.decl.to_func(),
            }
            for tool in tools
        ]

    def invoke(self, req: ToolCall) -> ToolResult:
        try:
            message: Message
            if not req.result_tool_id:
                # Initial request to this tool.
                prompt = self._template.render(req.args)
                message = Message(
                    role="user", content=[TextPart(type="text", text=prompt)]
                )
            else:
                # Getting a response from a tool invocation.
                if not req.result_call_id:
                    raise Exception("expected call_id for tool request")
                message = Message(
                    role="tool",
                    name=req.result_tool_id,
                    tool_call_id=req.result_call_id,
                    content=json.dumps(req.args),
                )

            # Append to history and complete with the underlying model.
            self._history.append(message)
            completion = req.context.router.send(self._history, self._tools)
            self._history.append(completion)

            # Handle any tool calls requested by the model.
            if "tool_calls" in completion:
                tool_calls = completion["tool_calls"]
                if len(tool_calls) > 1:
                    raise Exception(
                        f"Only one tool call per completion supported: {completion}"
                    )
                func = tool_calls[0]["function"]
                id = tool_calls[0]["id"]
                return ToolResult(
                    args=json.loads(func["arguments"]),
                    call_tool_id=func["name"],
                    call_id=id,
                )

            response_args = _parse_content(completion["content"])
            return ToolResult(response_args)

        except Exception as e:
            # Catch exceptions, log them, and send them to the model in hopes it will sort itself.
            msg = f"""exception calling {self.decl.id}: {e}
                     {"\n".join(traceback.format_exception(e))}"""
            return ToolResult({"error": msg})


def _parse_content(content: Union[str, List[Part]]) -> Dict[str, Any]:
    """Parse content into a dictionary, handling both string and Part list inputs.

    Args:
        content: Either a JSON string or a list of content parts

    Returns:
        Parsed dictionary from the JSON content

    Raises:
        Exception: If no valid JSON is found or multiple JSON objects are found
    """
    valid_jsons = []

    # Handle direct string input
    if isinstance(content, str):
        result = _try_parse_json(content)
        if result is None:
            raise Exception("No valid JSON dictionary found")
        return result

    # Handle list of parts
    for part in content:
        if part["type"] == "text":
            result = _try_parse_json(part["text"].strip())
            if result is not None:
                valid_jsons.append(result)

    if len(valid_jsons) == 0:
        raise Exception("No valid JSON dictionary found in any content parts")
    if len(valid_jsons) > 1:
        raise Exception(
            f"Found multiple ({len(valid_jsons)}) JSON objects: {valid_jsons}"
        )

    return valid_jsons[0]


def _try_parse_json(text: str) -> Optional[Dict[str, Any]]:
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
        return None
    except json.JSONDecodeError as e:
        print(f"--- error decoding json: {e}\n{text}")
        return None

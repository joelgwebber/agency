import json
import traceback
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from jinja2 import Environment
from jinja2.environment import Template

from agency.models import Function, FunctionCall, Message, Model, Role
from agency.tool import Tool, ToolCall, ToolDecl, ToolResult


@dataclass
class MinionDecl(ToolDecl):
    """Extension of a tool declaration to be used with minions, giving them a jinja template and list of available
    tools they can use."""

    template: str
    tools: List[ToolDecl]


class Minion(Tool):
    decl: ToolDecl
    _model: Model
    _template: Template
    _tools: List[Function]
    _history: List[Message]

    def __init__(
        self, decl: ToolDecl, model: Model, template: str, tools: List[ToolDecl]
    ):
        self.decl = decl
        self._model = model
        self._template = Environment().from_string(template)
        self._tools = [decl.to_func() for decl in tools]
        self._history = []

    def invoke(self, req: ToolCall) -> ToolResult:
        try:
            message: Message
            if not req.result_tool_id:
                # Initial request to this tool.
                prompt = self._template.render(req.args)
                message = Message(role=Role.USER, content=prompt)
            else:
                # Getting a response from a tool invocation.
                if not req.result_call_id:
                    raise Exception("expected call_id for tool request")
                message = Message(
                    role=Role.TOOL,
                    function=FunctionCall(
                        name=req.result_tool_id,
                        id=req.result_call_id,
                        arguments=req.args,
                    ),
                )

            # Append to history and complete with the underlying model.
            self._history.append(message)
            completion = self._model.complete(self._history, self._tools)
            self._history.append(completion)

            # Handle any tool calls requested by the model.
            if completion.function:
                func = completion.function
                return ToolResult(
                    args=func.arguments,
                    call_tool_id=func.name,
                    call_id=func.id,
                )

            # Parse the response and return it as this minion's result.
            # TODO: We'd probably be better off asking the LLM to call a "return" function.
            if completion.content:
                response_args = _parse_content(completion.content)
                return ToolResult(response_args)
            return ToolResult({})

        except Exception as e:
            # Catch exceptions, log them, and send them to the model in hopes it will sort itself.
            msg = f"""exception calling {self.decl.id}: {e}
                     {"\n".join(traceback.format_exception(e))}"""
            return ToolResult({"error": msg})


def _parse_content(content: str) -> Dict[str, Any]:
    """Parse content into a dictionary, handling both string and Part list inputs.

    Args:
        content: Either a JSON string or a list of content parts

    Returns:
        Parsed dictionary from the JSON content

    Raises:
        Exception: If no valid JSON is found or multiple JSON objects are found
    """

    # Handle direct string input
    result = _try_parse_json(content)
    if result is None:
        raise Exception("No valid JSON dictionary found")
    return result

    # valid_jsons = []
    # Handle list of parts
    # for part in content:
    #     if part["type"] == "text":
    #         result = _try_parse_json(part["text"].strip())
    #         if result is not None:
    #             valid_jsons.append(result)
    #
    # if len(valid_jsons) == 0:
    #     raise Exception("No valid JSON dictionary found in any content parts")
    # if len(valid_jsons) > 1:
    #     raise Exception(
    #         f"Found multiple ({len(valid_jsons)}) JSON objects: {valid_jsons}"
    #     )
    #
    # return valid_jsons[0]


def _try_parse_json(text: str) -> Optional[Dict[str, Any]]:
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
        return None
    except json.JSONDecodeError as e:
        print(f"--- error decoding json: {e}\n{text}")
        return None

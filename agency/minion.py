import json
import traceback
from dataclasses import dataclass
from typing import List

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

        # Initialize history with the system message.
        self._history = [
            Message(
                Role.SYSTEM,
                "Always invoke functions or return a structured JSON result. Never return raw text.",
            )
        ]

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
            completion = self._model.complete(
                self._history, self.decl.returns.to_openapi(), self._tools
            )
            self._history.append(completion)

            # Handle any tool calls requested by the model.
            if completion.function:
                func = completion.function
                return ToolResult(
                    args=func.arguments,
                    call_tool_id=func.name,
                    call_id=func.id,
                )

            # Otherwise we have a result.
            if completion.content:
                try:
                    result = json.loads(completion.content)
                    return ToolResult(result)
                except json.JSONDecodeError:
                    print(">>>", completion.content)

            return ToolResult({"error": "Expected structured output"})

        except Exception as e:
            # Catch exceptions, log them, and send them to the model in hopes it will sort itself.
            print(">>>", self._history)
            msg = f"""exception calling {self.decl.id}: {e}
                     {"\n".join(traceback.format_exception(e))}"""
            return ToolResult({"error": msg})

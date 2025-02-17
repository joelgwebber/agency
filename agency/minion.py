import json
import traceback
from dataclasses import dataclass
from typing import List

from jinja2 import Environment
from jinja2.environment import Template

from agency.models import Function, FunctionCall, Message, Model, Role
from agency.tool import Stack, Tool, ToolDecl


@dataclass
class MinionDecl(ToolDecl):
    """Extension of a tool declaration to be used with minions, giving them a jinja template and list of available
    tools they can use."""

    template: str
    tools: List[ToolDecl]


class Minion(Tool):
    _decl: ToolDecl
    _model: Model
    _template: Template
    _funcs: List[Function]

    def __init__(self, decl: ToolDecl, model: Model, template: str, *tools: ToolDecl):
        self._decl = decl
        self._model = model
        self._template = Environment().from_string(template)
        self._funcs = [decl.to_func() for decl in tools]

        # Initialize history with the system message.
        self._history = [
            Message(
                Role.SYSTEM,
                "Always invoke functions or return a structured JSON result. Never return raw text.",
            )
        ]

    @property
    def decl(self) -> ToolDecl:
        return self._decl

    def invoke(self, stack: Stack):
        frame = stack.top()
        try:
            message: Message
            if not frame.result_tool_id:
                # Initial request to this tool.
                prompt = self._template.render(frame.args)
                message = Message(role=Role.USER, content=prompt)
            else:
                # Getting a response from a tool invocation.
                if not frame.result_call_id:
                    raise Exception("expected call_id for tool request")
                message = Message(
                    role=Role.TOOL,
                    function=FunctionCall(
                        name=frame.result_tool_id,
                        id=frame.result_call_id,
                        arguments=frame.args,
                    ),
                )

            # Append to history and complete with the underlying model.
            frame.history.append(message)
            completion = self._model.complete(
                frame.history, self.decl.returns.to_openapi(), self._funcs
            )
            frame.history.append(completion)

            # Handle any tool calls requested by the model.
            if completion.function:
                func = completion.function
                stack.invoke(func.name, func.arguments, func.id)
                return

            # Otherwise we have a result.
            if completion.content:
                try:
                    stack.respond(json.loads(completion.content))
                    return
                except json.JSONDecodeError:
                    print(">>>", completion.content)

            stack.error("Expected structured output")

        except Exception as e:
            # Catch exceptions, log them, and send them to the model in hopes it will sort itself.
            print(">>>", frame.history)
            msg = f"""exception calling {self.decl.id}: {e}
                     {"\n".join(traceback.format_exception(e))}"""
            stack.error(msg)

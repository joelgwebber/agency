from __future__ import annotations

from typing import List

from agency.models.model import Function, Message, Model, Role
from agency.tool import Frame, Stack, ToolDecl
from agency.toolbox import Toolbox
from agency.utils import trunc


class Agent:
    _sys_prompt: Message
    _model: Model
    _funcs: List[Function]
    _toolbox: Toolbox

    def __init__(
        self, sys_prompt: str, model: Model, toolbox: Toolbox, *tools: ToolDecl
    ):
        self._sys_prompt = Message(Role.SYSTEM, sys_prompt)
        self._model = model
        self._funcs = [decl.to_func() for decl in tools]
        self._toolbox = toolbox

    def start(self) -> Stack:
        stack = Stack(frames=[Frame("", {}, "", [self._sys_prompt])])
        return stack

    def ask(self, stack: Stack, question: str) -> str:
        frame = stack.bottom()
        frame.history.append(Message(Role.USER, question))

        response = ""
        while True:
            result = self._model.complete(frame.history, None, self._funcs)
            if result.function:
                stack.invoke(
                    result.function.name, result.function.arguments, result.function.id
                )
                self._exec_tools(stack)
            elif result.content:
                response = result.content
                break

        return response

    def _exec_tools(self, stack: Stack):
        """
        The tool stack works as follows:
        - Each tool.dispatch() returns a ToolResponse
        - If response.call_id is None: tool is done, return result to previous tool
        - If response.call_id is set: push that tool onto stack and continue
        """
        while stack.depth() > 1:
            frame = stack.top()

            # debug
            args = frame.result_args if frame.result_args else frame.args
            print(
                f"--> invoking {frame.tool_id} <- {frame.result_tool_id}({frame.result_call_id})\n{trunc(str(args), 120)}"
            )

            # TODO: Catch any exception and return it to the caller as a
            # structured error response, focusing on LLM self-repair.
            tool = self._toolbox.tool_by_id(frame.tool_id)
            if not tool:
                raise Exception("expected frame.tool to be defined")
            tool.invoke(stack)


# result: Optional[ToolResult] = None
# if result.call_tool_id:
#     # It wants to call another tool; push it on the stack.
#     tool = self._toolbox.tool_by_id(result.call_tool_id)
#     stack.invoke(tool, result.args, result.call_id or "")
# else:
#     # The Tool is done; pop it off the stack and pass the response to the underlying frame.
#     last_frame = stack.pop()
#     if stack.depth() > 0:
#         stack.top().respond(
#             last_frame.tool_id, last_frame.call_id, result.args
#         )

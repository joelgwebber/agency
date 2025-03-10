from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from agency.minion import Except
from agency.schema import parse_val, schema_for
from agency.tool import (
    ExceptToolId,
    ResultToolId,
    Tool,
    ToolCall,
    ToolContext,
    ToolResult,
)
from agency.utils import trunc


@dataclass
class Frame:
    tool: Tool
    tool_id: str
    args: Dict[str, Any]
    call_id: str

    result_tool_id: Optional[str] = field(default=None)
    result_call_id: Optional[str] = field(default=None)
    result_args: Optional[Dict[str, Any]] = field(default=None)

    def respond(self, result_tool_id: str, result_call_id: str, result: Dict[str, Any]):
        self.result_tool_id = result_tool_id
        self.result_call_id = result_call_id
        self.result_args = result


class Agency:
    _stack: List[Frame]
    _toolbox: Dict[str, Tool]

    def __init__(self, tools: List[Tool]):
        self._stack = []
        self._toolbox = {tool.decl.id: tool for tool in tools}

    def ask(self, tool_id: str, args: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a tool request, handling nested tool calls via the stack.

        The tool stack works as follows:
        - Each tool.dispatch() returns a ToolResponse
        - If response.call_id is None: tool is done, return result to previous tool
        - If response.call_id is set: push that tool onto stack and continue

        Args:
            tool_id: ID of the tool to execute
            args: Arguments to pass to the tool

        Returns:
            The final result after all nested tool calls complete

        Raises:
            Exception: If tool_id is not found
        """
        response: Optional[ToolResult] = None
        self.push_tool(tool_id, args, "")
        while len(self._stack) > 0:
            frame = self._stack[-1]
            args = frame.result_args if frame.result_args else frame.args
            print(
                f"--> invoking {frame.tool_id} <- {frame.result_tool_id}({frame.result_call_id})\n{trunc(str(args), 120)}"
            )

            # TODO: Catch any exception and return it to the caller as a
            # structured error response, focusing on LLM self-repair.
            response = frame.tool.invoke(
                ToolCall(
                    name=frame.tool_id,
                    args=args,
                    context=ToolContext(),
                    result_tool_id=frame.result_tool_id,
                    result_call_id=frame.result_call_id,
                )
            )

            if response.call_tool_id == ResultToolId:
                # The Tool is done; pop it off the stack and pass the response to the underlying frame.
                last_frame = self._stack.pop()
                if len(self._stack) > 0:
                    self._stack[-1].respond(
                        last_frame.tool_id, last_frame.call_id, response.args
                    )
            elif response.call_tool_id == ExceptToolId:
                ex = parse_val(response.args, schema_for(Except))
                raise Exception(ex.message)
            else:
                # It wants to call another tool; push it on the stack.
                self.push_tool(
                    response.call_tool_id, response.args, response.call_id or ""
                )

        return response.args if response else {}

    def push_tool(self, tool_id: str, args: Dict[str, Any], call_id: str):
        """Push a tool onto the stack by its ID."""
        self._stack.append(
            Frame(
                tool=self.tool_by_id(tool_id),
                tool_id=tool_id,
                args=args,
                call_id=call_id,
            )
        )

    def tool_by_id(self, tool_id: str) -> Tool:
        if tool_id in self._toolbox:
            return self._toolbox[tool_id]
        raise Exception(f"no such tool: {tool_id}")

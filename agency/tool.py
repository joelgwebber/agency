from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol

from agency.models import Function
from agency.models.model import Message
from agency.schema import Schema


@dataclass
class Stack:
    frames: List[Frame]

    def depth(self) -> int:
        return len(self.frames)

    def bottom(self) -> Frame:
        return self.frames[0]

    def top(self) -> Frame:
        return self.frames[-1]

    def push(self, frame: Frame):
        self.frames.append(frame)

    def pop(self) -> Frame:
        return self.frames.pop()

    def invoke(self, tool_id: str, args: Dict[str, Any], call_id: str):
        self.push(
            Frame(
                tool_id=tool_id,
                args=args,
                call_id=call_id,
                history=[],
            )
        )

    def respond(self, args: Dict[str, Any]):
        callee = self.pop()
        self.top().respond(callee.tool_id, callee.call_id, args)

    def error(self, msg: str):
        self.respond({"error": msg})


@dataclass
class Frame:
    tool_id: str
    args: Dict[str, Any]
    call_id: str
    history: List[Message]

    result_tool_id: Optional[str] = field(default=None)
    result_call_id: Optional[str] = field(default=None)
    result_args: Optional[Dict[str, Any]] = field(default=None)

    def respond(self, result_tool_id: str, result_call_id: str, result: Dict[str, Any]):
        self.result_tool_id = result_tool_id
        self.result_call_id = result_call_id
        self.result_args = result


@dataclass
class ToolDecl:
    """Declaration for a tool that can be used by a language model (via a Minion).
    Their ids must be unique within the context of a single Minion."""

    id: str
    desc: str
    params: Schema
    returns: Schema

    def to_func(self) -> Function:
        return Function(
            name=self.id,
            description=self.desc,
            parameters=self.params.to_openapi(),
        )


class Tool(Protocol):
    """Protocol defining the interface that all tools must implement.

    This protocol ensures that all tools have a declaration describing their
    interface and an invoke method to handle requests.

    Attributes:
        decl: Tool declaration containing metadata about the tool's interface
    """

    @property
    def decl(self) -> ToolDecl: ...

    def invoke(self, stack: Stack): ...


# @dataclass
# class ToolCall:
#     """Represents a request to invoke a tool with specific arguments.
#
#     This class encapsulates all the information needed to make a tool call, including any arguments.
#
#     Attributes:
#         id: Unique identifier for this tool call
#         name: Name of the tool to call
#         arguments: Dictionary of arguments to pass to the tool
#         result_tool_id: ID of the tool that produced this call's input
#         result_call_id: ID of the specific call that produced this call's input
#     """
#
#     name: str
#     args: Dict[str, Any]
#     result_tool_id: Optional[str] = None
#     result_call_id: Optional[str] = None
#
#
# @dataclass
# class ToolResult:
#     """Represents the result of a tool invocation.
#
#     Tools return this class to provide their output and optionally request
#     another tool invocation.
#
#     Attributes:
#         args: Dictionary containing the tool's output
#         call_tool_id: ID of another tool to call with these args
#         call_id: ID to associate with the tool result
#     """
#
#     args: Dict[str, Any]
#     call_tool_id: Optional[str] = field(default=None)
#     call_id: Optional[str] = field(default=None)

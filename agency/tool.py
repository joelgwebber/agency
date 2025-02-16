from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Protocol

from agency.models import Function
from agency.schema import Schema

# ResultToolId = "__result__"
# ExceptToolId = "__except__"


@dataclass
class ToolContext:
    """Context passed to all tool dispatches containing shared resources.

    This context is provided to every tool invocation and contains resources
    that are shared across the entire tool execution chain.
    """

    pass


@dataclass
class ToolCall:
    """Represents a request to invoke a tool with specific arguments.

    This class encapsulates all the information needed to make a tool call,
    including the execution context and any arguments.

    Attributes:
        id: Unique identifier for this tool call
        name: Name of the tool to call
        arguments: Dictionary of arguments to pass to the tool
        context: The shared context for this tool execution (optional)
        result_tool_id: ID of the tool that produced this call's input
        result_call_id: ID of the specific call that produced this call's input
    """

    name: str
    args: Dict[str, Any]
    context: Optional[ToolContext] = None
    result_tool_id: Optional[str] = None
    result_call_id: Optional[str] = None


@dataclass
class ToolResult:
    """Represents the result of a tool invocation.

    Tools return this class to provide their output and optionally request
    another tool invocation.

    Attributes:
        args: Dictionary containing the tool's output
        call_tool_id: ID of another tool to call with these args
        call_id: ID to associate with the tool result
    """

    args: Dict[str, Any]
    call_tool_id: Optional[str] = field(default=None)
    call_id: Optional[str] = field(default=None)


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

    decl: ToolDecl

    def invoke(self, req: ToolCall) -> ToolResult: ...

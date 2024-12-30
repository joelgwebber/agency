from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Protocol

from agency.router import Router
from agency.tools.tools import ToolDecl


@dataclass
class ToolContext:
    """Context passed to all tool dispatches containing shared resources.

    This context is provided to every tool invocation and contains resources
    that are shared across the entire tool execution chain.

    Attributes:
        router: The router instance used for invoking language models
    """

    router: Router


@dataclass
class ToolCall:
    """Represents a request to invoke a tool with specific arguments.

    This class encapsulates all the information needed to make a tool call,
    including the execution context and any arguments.

    Attributes:
        context: The shared context for this tool execution
        args: Dictionary of arguments to pass to the tool
        result_tool_id: ID of the tool that produced this call's input
        result_call_id: ID of the specific call that produced this call's input
    """

    context: ToolContext
    args: Dict[str, Any]
    result_tool_id: Optional[str] = field(default=None)
    result_call_id: Optional[str] = field(default=None)


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


class Tool(Protocol):
    """Protocol defining the interface that all tools must implement.

    This protocol ensures that all tools have a declaration describing their
    interface and an invoke method to handle requests.

    Attributes:
        decl: Tool declaration containing metadata about the tool's interface
    """

    decl: ToolDecl

    def invoke(self, req: ToolCall) -> ToolResult: ...

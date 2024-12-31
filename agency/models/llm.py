from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional

from agency.models.openapi import OpenAPISchema


class Role(Enum):
    """Valid roles for LLM messages."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


@dataclass
class ToolCall:
    """A tool/function call from the LLM."""

    id: str
    name: str
    arguments: Dict[str, Any]


@dataclass
class Message:
    """A message in the LLM conversation."""

    role: Role
    content: str
    tool_call: Optional[ToolCall] = None
    tool_call_id: Optional[str] = None


@dataclass
class Function:
    """Description of a function the LLM can call."""

    name: str
    description: str
    parameters: OpenAPISchema


@dataclass
class FunctionCall:
    """A function call requested by the LLM."""

    name: str
    args: Dict[str, Any]
    call_id: str


@dataclass
class FunctionResult:
    """Result of a function call, to be passed back to the LLM."""

    args: Dict[str, Any]
    call_id: str


@dataclass
class Response:
    """Response from the LLM."""

    content: Optional[str] = None
    function: Optional[FunctionCall] = None


class LLM:
    """Interface for language model implementations."""

    def complete(
        self,
        messages: List[Message],
        functions: Optional[List[Function]] = None,
        function_result: Optional[FunctionResult] = None,
    ) -> Response:
        """Get a completion from the language model.

        Args:
            messages: The conversation history
            functions: Available functions the LLM can call
            function_result: Result from previous function call

        Returns:
            The model's response, either content or a function call
        """
        raise NotImplementedError

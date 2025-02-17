from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional

from .openapi import OpenAPISchema


class Role(Enum):
    """Valid roles for LLM messages."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


@dataclass
class FunctionCall:
    """A function call requested by the LLM."""

    id: str
    name: str
    arguments: Dict[str, Any]


@dataclass
class Message:
    """A message in the LLM conversation."""

    role: Role
    content: Optional[str] = None
    function: Optional[FunctionCall] = None


@dataclass
class Function:
    """Description of a function the LLM can call."""

    name: str
    description: str
    parameters: OpenAPISchema


class Model:
    """Interface for language model implementations."""

    def complete(
        self,
        messages: List[Message],
        response: Optional[OpenAPISchema] = None,
        functions: Optional[List[Function]] = None,
    ) -> Message:
        """Get a completion from the language model.

        Args:
            messages: The conversation history
            functions: Available functions the LLM can call

        Returns:
            The model's response, either content or a function call
        """
        raise NotImplementedError

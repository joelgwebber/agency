from dataclasses import dataclass
from typing import List

import pytest

from agency.agency import Agency
from agency.schema import Schema, Type
from agency.tool import Tool, ToolCall, ToolDecl, ToolResult

_str_schema = Schema(typ=Type.String, desc="test string param")


@dataclass
class MockTool(Tool):
    """A mock tool that returns predefined responses for testing."""

    decl: ToolDecl
    responses: List[ToolResult]
    response_index: int = 0

    def invoke(self, req: ToolCall) -> ToolResult:
        if self.response_index >= len(self.responses):
            raise Exception(f"Mock tool {self.decl.id} ran out of responses")
        result = self.responses[self.response_index]
        self.response_index += 1
        return result


def test_simple_tool_execution():
    """Test executing a single tool that returns a direct result."""
    mock_tool = MockTool(
        decl=ToolDecl(
            id="mock1", desc="Mock Tool 1", params=_str_schema, returns=_str_schema
        ),
        responses=[ToolResult(args={"result": "success"})],
    )

    agency = Agency(tools=[mock_tool])
    result = agency.ask("mock1", {"input": "test"})

    assert result == {"result": "success"}


def test_nested_tool_execution():
    """Test executing nested tools where one tool calls another."""
    mock_tool1 = MockTool(
        decl=ToolDecl(
            id="mock1", desc="Mock Tool 1", params=_str_schema, returns=_str_schema
        ),
        responses=[
            # First call another tool
            ToolResult(
                args={"nested_input": "test"}, call_tool_id="mock2", call_id="call1"
            ),
            # Then return final result after getting mock2's response
            ToolResult(args={"final": "done"}),
        ],
    )

    mock_tool2 = MockTool(
        decl=ToolDecl(
            id="mock2", desc="Mock Tool 2", params=_str_schema, returns=_str_schema
        ),
        responses=[ToolResult(args={"nested_result": "success"})],
    )

    agency = Agency(tools=[mock_tool1, mock_tool2])
    result = agency.ask("mock1", {"input": "test"})

    assert result == {"final": "done"}


def test_deep_nested_tool_execution():
    """Test executing deeply nested tools (3 levels)."""
    mock_tool1 = MockTool(
        decl=ToolDecl(
            id="mock1", desc="Mock Tool 1", params=_str_schema, returns=_str_schema
        ),
        responses=[
            ToolResult(args={"level1": "test"}, call_tool_id="mock2", call_id="call1"),
            ToolResult(args={"final": "done"}),
        ],
    )

    mock_tool2 = MockTool(
        decl=ToolDecl(
            id="mock2", desc="Mock Tool 2", params=_str_schema, returns=_str_schema
        ),
        responses=[
            ToolResult(args={"level2": "test"}, call_tool_id="mock3", call_id="call2"),
            ToolResult(
                args={"level2_complete": "success"}
            ),  # Response after mock3 returns
        ],
    )

    mock_tool3 = MockTool(
        decl=ToolDecl(
            id="mock3", desc="Mock Tool 3", params=_str_schema, returns=_str_schema
        ),
        responses=[ToolResult(args={"level3": "success"})],
    )

    agency = Agency(tools=[mock_tool1, mock_tool2, mock_tool3])
    result = agency.ask("mock1", {"input": "test"})

    assert result == {"final": "done"}


def test_tool_not_found():
    """Test error handling when requesting a non-existent tool."""
    agency = Agency(tools=[])

    with pytest.raises(Exception) as exc:
        agency.ask("nonexistent", {})
    assert "no such tool" in str(exc.value)


def test_tool_throws_exception():
    """Test error handling when a tool raises an exception."""

    def raise_error(self, req: ToolCall) -> ToolResult:
        raise ValueError("Simulated error")

    def error_invoke(req: ToolCall) -> ToolResult:
        raise ValueError("Simulated error")

    mock_tool = MockTool(
        decl=ToolDecl(
            id="error", desc="Error Tool", params=_str_schema, returns=_str_schema
        ),
        responses=[],
    )
    mock_tool.invoke = error_invoke

    agency = Agency(tools=[mock_tool])

    with pytest.raises(ValueError) as exc:
        agency.ask("error", {})
    assert "Simulated error" in str(exc.value)

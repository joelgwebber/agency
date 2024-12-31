from unittest.mock import patch

import pytest

from agency.models.llm import Function, FunctionCall, Message, Role
from agency.models.openrouter import OpenRouterLLM


@pytest.fixture
def llm():
    return OpenRouterLLM("test-model")


def test_complete_content(llm):
    with patch("requests.post") as mock_post:
        mock_post.return_value.json.return_value = {
            "choices": [{"message": {"content": '{"result": "test response"}'}}]
        }

        messages = [Message(role=Role.USER, content="test prompt")]
        response = llm.complete(messages)

        assert response.content == '{"result": "test response"}'
        assert response.function is None
        assert response.role == Role.ASSISTANT

        # Verify API call
        mock_post.assert_called_once()
        call_args = mock_post.call_args.kwargs["json"]
        assert call_args["model"] == "test-model"
        assert len(call_args["messages"]) == 2  # System + user message
        assert call_args["messages"][1]["content"] == "test prompt"


def test_complete_tool_call(llm):
    with patch("requests.post") as mock_post:
        mock_post.return_value.json.return_value = {
            "choices": [
                {
                    "message": {
                        "content": "",
                        "tool_calls": [
                            {
                                "id": "call-123",
                                "function": {
                                    "name": "test_function",
                                    "arguments": '{"arg": "value"}',
                                },
                                "type": "function",
                            }
                        ],
                    }
                }
            ]
        }

        messages = [Message(role=Role.USER, content="test prompt")]
        functions = [
            Function(
                name="test_function",
                description="A test function",
                parameters={"type": "object", "properties": {}},
            )
        ]
        response = llm.complete(messages, functions)

        assert response.content == ""
        assert response.function == FunctionCall(
            id="call-123", name="test_function", arguments={"arg": "value"}
        )

        # Verify API call
        mock_post.assert_called_once()
        call_args = mock_post.call_args.kwargs["json"]
        assert call_args["tools"][0]["function"]["name"] == "test_function"


def test_multiple_tool_calls_raises(llm):
    with patch("requests.post") as mock_post:
        mock_post.return_value.json.return_value = {
            "choices": [
                {
                    "message": {
                        "tool_calls": [
                            {
                                "id": "call-1",
                                "function": {"name": "func1", "arguments": "{}"},
                            },
                            {
                                "id": "call-2",
                                "function": {"name": "func2", "arguments": "{}"},
                            },
                        ]
                    }
                }
            ]
        }

        messages = [Message(role=Role.USER, content="test prompt")]
        with pytest.raises(Exception) as exc:
            llm.complete(messages)
        assert "Expected at most 1 tool call" in str(exc.value)


def test_api_error_raises(llm):
    with patch("requests.post") as mock_post:
        mock_post.return_value.json.return_value = {"error": "API Error Message"}

        messages = [Message(role=Role.USER, content="test prompt")]
        with pytest.raises(Exception) as exc:
            llm.complete(messages)
        assert "OpenRouter API error" in str(exc.value)

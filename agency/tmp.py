import json
from typing import List

from agency.router import Message, Router, ToolDesc

r = Router()

record_note: ToolDesc = {
    "type": "function",
    "function": {
        "name": "record-note",
        "description": "records a note",
        "parameters": {
            "type": "object",
            "properties": {
                "id": {"type": "string", "description": ""},
                "text": {"type": "string", "description": ""},
            },
            "required": ["id", "text"],
        },
    },
}

history: List[Message] = [
    {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": "You are a research assistant, helping your human understand any topic.\n    - Use the notebook to record notes as you go, looking them up by subject when needed\n    - DO NOT use any other information to answer questions\n    - Always cite your sources\n    - Always respond with json objects\n    Current question: test",
            }
        ],
    },
    {
        "role": "assistant",
        "content": [],
        "tool_calls": [
            {
                # "index": 0,
                "id": "call_emJeh4sCcswRQv1A8ULiaGel",
                "type": "function",
                "function": {
                    "name": "record-note",
                    "arguments": '{"id":"test","text":"This is a test note for the research assistant. Feel free to ask questions or request information."}',
                },
            }
        ],
    },
    {
        "role": "tool",
        "name": "record-note",
        "tool_call_id": "call_emJeh4sCcswRQv1A8ULiaGel",
        "content": [{"type": "text", "text": '{"question": "test"}'}],
    },
]

json.dumps(
    {
        "model": "openai/gpt-3.5-turbo",
        "response_format": {"type": "json_object"},
        "messages": history,
        "tools": [record_note],
    }
)

rsp = r.send(history, [record_note])

# ----------------------------------------------------------------------------------------------------------------------

weather: ToolDesc = {
    "type": "function",
    "function": {
        "name": "weather",
        "description": "gets the current weather",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "The city and state, e.g. San Francisco, CA",
                },
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
            },
            "required": ["location"],
        },
    },
}

msgs: List[Message] = [
    {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": "What's the weather like in NYC?",
            }
        ],
    }
]

msgs.append(r.send(msgs, [weather]))

if "tool_calls" in msgs[-1]:
    id = msgs[-1]["tool_calls"][0]["id"]
    msgs.append(
        {
            "role": "tool",
            "name": "weather",
            "tool_call_id": id,
            "content": json.dumps(
                {
                    "temperature": 20,
                    "unit": "C",
                    "precipitation": 0,
                }
            ),
            # [
            #     {
            #         "type": "text",
            #         "text": json.dumps(
            #             {
            #                 "temperature": 20,
            #                 "unit": "C",
            #                 "precipitation": 0,
            #             }
            #         ),
            #     }
            # ],
        }
    )
    msgs.append(r.send(msgs, [weather]))

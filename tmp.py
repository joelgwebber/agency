import requests

from agency.keys import OPENROUTER_API_KEY

or_req = {
    "model": "openai/o3-mini",
    "messages": [
        {
            "role": "user",
            "content": """
Answer the following question in general terms, with the goal of creating starting points for
further research. Make sure to follow the explicit response schema.

What is the average flight velocity of an unladen swallow?
""",
        }
    ],
    "response_format": {
        "type": "json_schema",
        "json_schema": {
            "name": "response",
            "strict": True,
            "schema": {
                "type": "object",
                "format": "object",
                "description": "",
                "properties": {
                    "answer": {"type": "string", "format": "string", "description": ""},
                    # "search_terms": {
                    #     "type": "array",
                    #     "format": "array",
                    #     "description": "",
                    #     "items": {
                    #         "type": "string",
                    #         "format": "string",
                    #         "description": "",
                    #     },
                    # },
                },
                "required": ["answer"],
            },
        },
    },
}


rsp = requests.post(
    url="https://openrouter.ai/api/v1/chat/completions",
    headers={
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "HTTP-Referer": "j15r.com",
        "X-Title": "agency",
    },
    json=or_req,
).json()

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date
from typing import Any, Generator, List, Literal, Optional, cast

from google.cloud.aiplatform_v1beta1 import FunctionCall, FunctionResponse
from proto.marshal.collections.maps import MapComposite
from proto.marshal.collections.repeated import RepeatedComposite
from vertexai.generative_models import ChatSession, Content, GenerativeModel, Part

from agency.tools import Tool, ToolBox
from agency.utils import part_is_text

# Give today's date, because it often gets that wrong.
# Explicitly request that it use recipes to fuel CoT reasoning.
_sys_suffix = f"""
Today's date is {date.today()}.

When asked a question, ALWAYS do the following:
- ALWAYS call `find_recipes` BEFORE ANY OTHER FUNCTION, to determine the best approach.
- Clearly enumerate the steps you plan to take, AFTER getting ideas from `find_recipes`.
- Summarize the instructions from recipes that inform your plan.
- Use any appropriate ideas and constraints from recipes to inform function calls.
- Complete the request using the available functions.
- Use your existing context as much as possible to avoid duplicate work.
- Use the notebook to record notes as you go, looking them up by subject when needed.
- DO NOT use any other information to answer questions.
"""


def ex_user(*parts: Part) -> Content:
    return Content(role="user", parts=list(parts))


def ex_model(*parts: Part) -> Content:
    return Content(role="model", parts=list(parts))


def ex_text(text: str) -> Part:
    return Part.from_text(text)


def ex_call(fn: str, **kwargs) -> Part:
    return Part.from_dict(
        {
            "function_call": {
                "name": fn,
                "args": kwargs,
            }
        }
    )


def ex_resp(fn: str, result: Any) -> Part:
    return Part.from_dict(
        {"function_response": {"name": fn, "response": {"content": {"result": result}}}}
    )


# We seed the chat history with these examples, because it allows us to structure them
# exactly as they would look in the history. This appears to work well as way of merging
# few-shot prompting with function calling.
init_shots = [
    # Initial request/response with find_recipes call.
    ex_user(
        ex_text(
            """
            The following are some examples of expected requests and responses.
            <EXAMPLE>
            I'm an account rep focusing on retention for SaaS customers.
            Tell me which customers I should reach out to this week, and explain why I should choose them.
            """
        )
    ),
    ex_model(
        ex_text(
            "I will use `find_recipes` to find the best way to identify customers at risk of churn."
        ),
        ex_call("find_recipes", number=5, goal="Identify customers at risk of churn"),
        ex_text("</EXAMPLE>"),
    ),
    ex_user(
        ex_text("<EXAMPLE>"),
        ex_resp(
            "find_recipes",
            [
                "these would be recipes",
                "that you should use to inform your plan",
            ],
        ),
    ),
    ex_model(
        Part.from_text(
            """
            Based on these recipes, I will do the following:
            - Things I will do in my plan
            - to satisfy the request.
            </EXAMPLE>
            """
        ),
    ),
]


@dataclass
class Thought:
    typ: Literal["message", "tool"]
    tool: str
    content: str


class Agency:
    _chat: ChatSession
    _inputs: List[Part]
    _toolbox: ToolBox
    _log: List[List[str]]

    def __init__(
        self,
        tools: List[Tool],
        sys_instr: str,
        history: Optional[List[Content]] = None,
    ):
        model = GenerativeModel(
            # "gemini-1.5-pro-preview-0514" has VERY low quota (3/m)
            # "gemini-1.0-pro-001" 32k context length often breaks session summaries
            "gemini-1.5-flash-preview-0514",
            system_instruction=[Part.from_text(sys_instr), Part.from_text(_sys_suffix)],
        )

        if history == None:
            history = init_shots

        self._toolbox = ToolBox(tools)
        self._chat = model.start_chat(response_validation=False, history=history)
        self._inputs = []
        self._log = []

    @property
    def history(self) -> List[Content]:
        return self._chat.history

    @property
    def inputs(self) -> List[Part]:
        return self._inputs

    def ask(self, question: str) -> None:
        self.new_question(question)
        while not self.is_finished():
            for thought in self.think():
                match thought.typ:
                    case "message":
                        print(thought.content)
                    case "tool":
                        print(thought.tool, thought.content)

    def new_question(self, question: str):
        self._inputs.append(Part.from_text(question))

    def is_finished(self) -> bool:
        return len(self._inputs) == 0

    def think(self) -> Generator[Thought, None, None]:
        new_inputs = []
        outputs = self._send(self._inputs)
        for part in outputs:
            if part_is_text(part):
                # Sometimes we get whitespace-only.
                if part.text.strip() != "":
                    yield Thought("message", "", part.text)
            else:
                call = part.function_call
                rsp_part = self._toolbox.dispatch(call)
                rsp = rsp_part.function_response
                new_inputs.append(rsp_part)
                yield Thought("tool", call.name, _print_tool(call, rsp))

        self._inputs = new_inputs

    def _send(self, parts: List[Part]) -> List[Part]:
        rsp = self._chat.send_message(
            cast(List, parts),
            tools=[self._toolbox.lang_tools],
            generation_config={"temperature": 0},
        )

        if isinstance(rsp, Iterable):
            raise Exception("streaming messages not yet supported")

        cand = rsp.candidates[0]
        return cand.content.parts


def _print_proto(m) -> str:
    match m:
        case MapComposite():
            return ", ".join([f"{k} = {_print_proto(m[k])}" for k in m])
        case RepeatedComposite():
            return ", ".join([_print_proto(v) for v in m])
    return str(m)


def _print_tool(call: FunctionCall, rsp: FunctionResponse) -> str:
    return f"""{_print_proto(call.args)}\n{_print_proto(rsp.response)}"""

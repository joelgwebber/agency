from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, cast

from vertexai.generative_models import Content
from vertexai.generative_models import GenerativeModel as GModel
from vertexai.generative_models import Part as GPart


class PartType(Enum):
    Text = "text"
    Call = "call"
    Response = "response"


@dataclass
class Part:
    type: PartType


@dataclass
class TextPart(Part):
    text: str

    def __init__(self, text: str):
        Part.__init__(self, PartType.Text)
        self.text = text


@dataclass
class CallPart(Part):
    name: str
    args: Dict[str, Any]

    def __init__(self, name: str, args: Dict[str, Any]):
        Part.__init__(self, PartType.Call)
        self.name = name
        self.args = args


@dataclass
class ResponsePart(Part):
    name: str
    returned: Dict[str, Any]

    def __init__(self, name: str, returned: Dict[str, Any]):
        Part.__init__(self, PartType.Response)
        self.name = name
        self.returned = returned


class Model(ABC):
    @abstractmethod
    def send(self, input: List[Part]) -> List[Part]: ...


class GeminiModel(Model):
    _gemini: GModel

    def __init__(self, system: str):
        # Use Gemini 1.5 Flash.
        self._gemini = GModel(
            "gemini-1.5-flash-preview-0514",
            system_instruction=[GPart.from_text(system)],
        )

    def send(self, input: List[Part]) -> List[Part]:
        ginput = Content(role="user", parts=[part_to_gpart(part) for part in input])
        rsp = self._gemini.generate_content([ginput])
        return [gpart_to_part(gpart) for gpart in rsp.candidates[0].content.parts]


def part_to_gpart(part: Part) -> GPart:
    match part.type:
        case PartType.Text:
            text_part = cast(TextPart, part)
            return GPart.from_text(text_part.text)
        case PartType.Response:
            rsp_part = cast(ResponsePart, part)
            return GPart.from_function_response(rsp_part.name, rsp_part.returned)
        case PartType.Call:
            raise Exception("function calls only go the other way")


def gpart_to_part(gpart: GPart) -> Part:
    if gpart_is_text(gpart):
        return TextPart(gpart.text)
    fn = gpart.function_call
    return CallPart(fn.name, fn.args)


# TODO: Is there really no better way to do this?!
def gpart_is_text(part: GPart) -> bool:
    try:
        part.text
        return True
    except:
        return False

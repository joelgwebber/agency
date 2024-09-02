from __future__ import annotations

from datetime import datetime
from typing import List

import pytz
from google.cloud.aiplatform_v1beta1 import FunctionCall, FunctionResponse
from IPython.display import Markdown, display
from proto.marshal.collections.maps import MapComposite
from proto.marshal.collections.repeated import RepeatedComposite
from vertexai.generative_models import Part
from vertexai.language_models import TextEmbeddingInput, TextEmbeddingModel


def embed(recipe: str) -> List[float]:
    # This is the most recent embedding model I'm aware of.
    # It has a 2k input limit, so we might have to break some things up.
    model = TextEmbeddingModel.from_pretrained("text-embedding-004")

    # SEMANTIC_SIMILARITY seems to cluster too tightly, leading to repeated entries.
    inputs: List = [TextEmbeddingInput(recipe, "CLASSIFICATION")]

    embeddings = model.get_embeddings(inputs)
    return embeddings[0].values


def print_proto(m) -> str:
    match m:
        case MapComposite():
            return ", ".join([f"{k} = {print_proto(m[k])}" for k in m])
        case RepeatedComposite():
            return ", ".join([print_proto(v) for v in m])
    return str(m)


def print_tool(call: FunctionCall, rsp: FunctionResponse) -> str:
    return f"""{print_proto(call.args)}\n{print_proto(rsp.response)}"""


# TODO: Is there really no better way to do this?!
def part_is_text(part: Part) -> bool:
    try:
        part.text
        return True
    except:
        return False


def markdown(md: str):
    if running_in_notebook():
        display(Markdown(md))
    else:
        print(md)


def running_in_notebook():
    # Note: This import doesn't type-check, but it's actually there in practice.
    # This whole thing is a mess, but there doesn't appear to be a better way.
    try:
        from IPython import get_ipython  # pyright: ignore

        if "IPKernelApp" not in get_ipython().config:  # pyright: ignore
            raise ImportError("not in a notebook")
        return True
    except Exception:
        return False


class timestamp(datetime):
    """Override of built-in datetime to force UTC."""

    def __new__(
        cls,
        year,
        month,
        day,
        hour=0,
        minute=0,
        second=0,
        microsecond=0,
        tzinfo=pytz.UTC,
    ):
        return super().__new__(
            cls, year, month, day, hour, minute, second, microsecond, tzinfo=tzinfo
        )

    @classmethod
    def fromisoformat(cls, date_string) -> timestamp:
        d = datetime.fromisoformat(date_string)
        return timestamp(
            d.year, d.month, d.day, d.hour, d.minute, d.second, d.microsecond
        )

    @staticmethod
    def zero() -> timestamp:
        return timestamp(1970, 1, 1)

    def is_zero(self) -> bool:
        return self == timestamp.zero()

    def sql(self) -> str:
        """Produces a SQL-formatted TIMESTAMP() expression for the given Python timestamp object."""
        return f'TIMESTAMP("{self.year}-{self.month:02}-{self.day:02} {self.hour:02}:{self.minute:02}:{self.second:02}")'

from __future__ import annotations

from datetime import datetime

import pytz
from IPython.display import Markdown, display
from vertexai.generative_models import Part


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

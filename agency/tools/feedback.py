from dataclasses import dataclass
from typing import List

from agency.schema import (
    Schema,
    Type,
    parse_val,
    prop,
    schema,
    schema_for,
    serialize_val,
)
from agency.tool import Stack, Tool, ToolDecl
from agency.tools.logstore import LogStore
from agency.utils import timestamp


@dataclass
class SubmitFeedback(Tool):
    @schema
    class Params:
        expectation: str = prop("What the user expected")
        context: str = prop(
            """What actually happened, with any context useful to reproduce.
            Be as verbose as necessary to allow for reproduction of the issue."""
        )

    _store: LogStore

    @property
    def decl(self) -> ToolDecl:
        return ToolDecl(
            "submit-feedback",
            "Submits feedback on an interaction",
            schema_for(SubmitFeedback.Params),
            Schema(Type.Object, ""),
        )

    def invoke(self, stack: Stack):
        args = parse_val(stack.top().args, self.decl.params)
        self._store.append(f"""Expected: {args.expectation}\nContext: {args.context}""")
        stack.respond({})


@dataclass
class GetFeedback(Tool):
    @schema
    class Params:
        query: str = prop("What the user expected")
        begin: timestamp = prop("The beginning time range")
        end: timestamp = prop("The ending time range")

    @schema
    class Returns:
        feedback: List[str]

    @property
    def decl(self) -> ToolDecl:
        return ToolDecl(
            "get-feedback",
            "Gets user feedback within a specified time range",
            schema_for(GetFeedback.Params),
            schema_for(GetFeedback.Returns),
        )

    _store: LogStore

    def invoke(self, stack: Stack):
        args = parse_val(stack.top().args, self.decl.params)
        result = self._store.query(args.query, args.begin, args.end)
        stack.respond(serialize_val(GetFeedback.Returns(result), self.decl.returns))

from dataclasses import dataclass
from typing import List

from agency.schema import Schema, Type, parse_val, prop, schema, schema_for
from agency.tool import Tool, ToolCall, ToolDecl, ToolResult
from agency.tools.logstore import LogStore
from agency.utils import timestamp


@dataclass
class SubmitFeedback(Tool):
    @schema()
    class Params:
        expectation: str = prop("What the user expected")
        context: str = prop(
            """What actually happened, with any context useful to reproduce.
            Be as verbose as necessary to allow for reproduction of the issue."""
        )

    decl = ToolDecl(
        "submit-feedback",
        "Submits feedback on an interaction",
        schema_for(Params),
        Schema(Type.Object, ""),
    )

    _store: LogStore

    def invoke(self, req: ToolCall) -> ToolResult:
        args = parse_val(req.args, SubmitFeedback.decl.params)
        self._store.append(f"""Expected: {args.expectation}\nContext: {args.context}""")
        return ToolResult({})


@dataclass
class GetFeedback(Tool):
    @schema()
    class Params:
        query: str = prop("What the user expected")
        begin: timestamp = prop("The beginning time range")
        end: timestamp = prop("The ending time range")

    @schema()
    class Returns:
        feedback: List[str]

    decl = ToolDecl(
        "get-feedback",
        "Gets user feedback within a specified time range",
        schema_for(Params),
        schema_for(Returns),
    )

    _store: LogStore

    def invoke(self, req: ToolCall) -> ToolResult:
        args = parse_val(req.args, GetFeedback.decl.params)
        result = self._store.query(args.query, args.begin, args.end)
        return ToolResult(dict(GetFeedback.Returns(result)))

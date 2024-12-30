from dataclasses import dataclass

from agency.schema import parse_val, prop, schema, schema_for
from agency.tool import Tool, ToolCall, ToolDecl, ToolResult
from agency.tools.logstore import LogStore
from agency.utils import timestamp


@dataclass
class SubmitFeedback(Tool):
    @schema()
    class Args:
        expectation: str = prop("What the user expected")
        context: str = prop(
            """What actually happened, with any context useful to reproduce.
            Be as verbose as necessary to allow for reproduction of the issue."""
        )

    decl = ToolDecl(
        "submit-feedback",
        "Submits feedback on an interaction",
        schema_for(Args),
    )

    _store: LogStore

    def invoke(self, req: ToolCall) -> ToolResult:
        args = parse_val(req.args, SubmitFeedback.decl.params)
        self._store.append(f"""Expected: {args.expectation}\nContext: {args.context}""")
        return ToolResult({})


@dataclass
class GetFeedback(Tool):
    @schema()
    class Args:
        query: str = prop("What the user expected")
        begin: timestamp = prop("The beginning time range")
        end: timestamp = prop("The ending time range")

    decl = ToolDecl(
        "get-feedback",
        "Gets user feedback within a specified time range",
        schema_for(Args),
    )

    _store: LogStore

    def invoke(self, req: ToolCall) -> ToolResult:
        args = parse_val(req.args, GetFeedback.decl.params)
        result = self._store.query(args.query, args.begin, args.end)
        return ToolResult({"feedback": result})

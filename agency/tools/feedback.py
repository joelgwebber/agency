from typing import List

import chromadb
import chromadb.api

from agency.embedding import embed_text
from agency.tools.annotations import prop, schema, schema_for
from agency.tools.tools import ToolDecl, parse_val
from agency.types import Tool, ToolCall, ToolResult
from agency.utils import timestamp


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

    _coll: chromadb.Collection

    def __init__(self, coll: chromadb.Collection):
        self._coll = coll

    def invoke(self, req: ToolCall) -> ToolResult:
        args = parse_val(req.args, SubmitFeedback.decl.params)
        when = timestamp.now().timestamp()
        doc = f"Expected:\n{args.expectation}\nContext:\n{args.context}"
        self._coll.add(
            ids=str(when),
            documents=doc,
            embeddings=embed_text(doc).tolist(),
            metadatas={"when": when},
        )
        return ToolResult({})


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

    _coll: chromadb.Collection

    def __init__(self, coll: chromadb.Collection):
        self._coll = coll

    def invoke(self, req: ToolCall) -> ToolResult:
        args = parse_val(req.args, GetFeedback.decl.params)
        rsp = self._coll.query(
            query_embeddings=embed_text(args.query).tolist(),
            where={
                "$and": [
                    {"when": {"$gte": args.begin.timestamp()}},
                    {"when": {"$lt": args.end.timestamp()}},
                ]
            },
        )

        result: List[List[str]] = []
        if rsp["documents"] is not None and rsp["metadatas"] is not None:
            docs = rsp["documents"][0]
            metas = rsp["metadatas"][0]
            for i in range(0, len(docs)):
                when = timestamp.fromtimestamp(float(metas[i]["when"]))
                result.append([when.isoformat(), docs[i]])

        return ToolResult({"feedback": result})

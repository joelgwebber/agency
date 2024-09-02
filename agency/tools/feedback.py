from typing import List

import chromadb
import chromadb.api

from agency.tools import Tool
from agency.tools.annotations import prop, schema, decl
from agency.utils import timestamp, embed


@schema()
class SubmitFeedbackArgs:
    expectation: str = prop("What the user expected")
    context: str = prop(
        """What actually happened, with any context useful to reproduce.
        Be as verbose as necessary to allow for reproduction of the issue."""
    )


@schema()
class GetFeedbackArgs:
    query: str = prop("What the user expected")
    begin: timestamp = prop("The beginning time range")
    end: timestamp = prop("The ending time range")


class Feedback(Tool):
    _feedback_coll: chromadb.Collection

    def __init__(self, dbclient: chromadb.api.ClientAPI):
        Tool.__init__(self)

        self.declare(self.submit_feedback)
        self.declare(self.get_feedback)

        self._feedback_coll = dbclient.create_collection(
            name="feedback", get_or_create=True
        )

    @decl("submit_feedback", "Submits feedback on an interaction")
    def submit_feedback(self, args: SubmitFeedbackArgs) -> None:
        when = timestamp.now().timestamp()
        doc = f"Expected:\n{args.expectation}\nContext:\n{args.context}"
        self._feedback_coll.add(
            ids=str(when),
            documents=doc,
            embeddings=embed(doc),
            metadatas={"when": when},
        )

    @decl("get_feedback", "Gets user feedback within a specified time range")
    def get_feedback(self, args: GetFeedbackArgs) -> List[List[str]]:
        rsp = self._feedback_coll.query(
            query_embeddings=embed(args.query),
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

        return result

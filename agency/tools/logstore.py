import os
from typing import List

import chromadb
import chromadb.api

from agency.embedding import embed_text
from agency.utils import timestamp


class LogStore:
    _coll: chromadb.Collection
    _work_dir: str

    def __init__(self, dbclient: chromadb.api.ClientAPI, dir: str, name: str):
        self._coll = dbclient.get_or_create_collection(
            name=name,
            embedding_function=None,  # Use raw embeddings
            metadata={"dimension": 384},  # Set dimension for 384-vectors
        )
        self._work_dir = os.path.join(dir, name)
        os.makedirs(self._work_dir, exist_ok=True)

    def append(self, doc: str) -> None:
        when = timestamp.now()
        with open(os.path.join(self._work_dir, f"{when}.md"), "w") as file:
            file.write(doc)
            file.close()

        self._coll.add(
            ids=str(when),
            documents=doc,
            embeddings=embed_text(doc).tolist(),
            metadatas={"when": when.timestamp()},
        )

    def query(self, query: str, begin: timestamp, end: timestamp) -> List[List[str]]:
        rsp = self._coll.query(
            query_embeddings=embed_text(query).tolist(),
            where={
                "$and": [
                    {"when": {"$gte": begin.timestamp()}},
                    {"when": {"$lt": end.timestamp()}},
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

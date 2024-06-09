from typing import List

import chromadb
import chromadb.api
from vertexai.language_models import TextEmbeddingInput, TextEmbeddingModel

from agency.tools import Func, Schema, Tool, Type


class Notebook(Tool):
    _embed_model: TextEmbeddingModel
    _notebook_coll: chromadb.Collection

    def __init__(
        self, embed_model: TextEmbeddingModel, dbclient: chromadb.api.ClientAPI
    ):
        Tool.__init__(self)

        self._embed_model = embed_model
        self._notebook_coll = dbclient.create_collection(
            name="notebook", get_or_create=True
        )

        self._add_func(
            Func(
                self.record_note,
                "record_note",
                "Records a note in the notebook for later research.",
                {"text": Schema(Type.String, "The document text")},
            ),
        )

        self._add_func(
            Func(
                self.lookup_notes,
                "lookup_notes",
                "Looks up notes in the notebook.",
                {
                    "reference": Schema(Type.String, "Reference text"),
                    "max_results": Schema(
                        Type.Integer, "Maximum number of documents to return"
                    ),
                },
            ),
        )

    def record_note(self, text: str):
        # TODO: Use a stronger hash.
        doc_hash = f"{hash(text):016x}"
        self._notebook_coll.add(
            ids=doc_hash, embeddings=self._embed(text), documents=text
        )

    def lookup_notes(self, reference: str, max_results: int):
        vec = self._embed(reference)
        rsp = self._notebook_coll.query(
            query_embeddings=[vec], n_results=int(max_results)
        )
        if rsp["documents"] is not None:
            return rsp["documents"][0]
        return []

    def _embed(self, text: str) -> List[float]:
        # SEMANTIC_SIMILARITY seems to cluster too tightly, leading to repeated entries.
        inputs: List = [TextEmbeddingInput(text, "CLASSIFICATION")]
        embeddings = self._embed_model.get_embeddings(inputs)
        return embeddings[0].values

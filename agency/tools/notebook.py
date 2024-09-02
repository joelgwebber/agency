from typing import List

import chromadb
import chromadb.api
from vertexai.language_models import TextEmbeddingInput, TextEmbeddingModel

from agency.tools import Tool
from agency.tools.annotations import decl, prop, schema


@schema()
class RecordNoteArgs:
    text: str = prop("The document text")


@schema()
class LookupNotesArgs:
    reference: str = prop("Reference text")
    max_results: int = prop("Maximum number of documents to return", default=5)


# A tool for recording ephemeral notes for later consultation.
# The notes are stored in a vector database with computed embeddings for later retrieval.
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

        self.declare(self.record_note)
        self.declare(self.lookup_notes)

    @decl("record_note", "Records a note in the notebook for later research.")
    def record_note(self, args: RecordNoteArgs):
        # TODO: Use a stronger hash.
        doc_hash = f"{hash(args.text):016x}"
        self._notebook_coll.add(
            ids=doc_hash, embeddings=self._embed(args.text), documents=args.text
        )

    @decl("lookup_note", "Looks up notes in the notebook.")
    def lookup_notes(self, args: LookupNotesArgs):
        vec = self._embed(args.reference)
        rsp = self._notebook_coll.query(
            query_embeddings=[vec], n_results=int(args.max_results)
        )
        if rsp["documents"] is not None:
            return rsp["documents"][0]
        return []

    def _embed(self, text: str) -> List[float]:
        # SEMANTIC_SIMILARITY seems to cluster too tightly, leading to repeated entries.
        inputs: List = [TextEmbeddingInput(text, "CLASSIFICATION")]
        embeddings = self._embed_model.get_embeddings(inputs)
        return embeddings[0].values

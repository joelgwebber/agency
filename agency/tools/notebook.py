from typing import Dict, List

import chromadb
import chromadb.api
from vertexai.language_models import TextEmbeddingModel

from agency.tools import Tool
from agency.tools.annotations import decl, prop, schema
from agency.tools.docstore import Doc, Docstore


@schema()
class RecordNoteArgs:
    id: str = prop("unique note id")
    text: str = prop("note text")
    labels: Dict[str, str] = prop("labels and values to associate with this note")


@schema()
class UpdateNoteArgs:
    id: str = prop("note id to update")
    new_id: str = prop("new note id (may be the same)")
    text: str = prop("new note text")
    labels: Dict[str, str] = prop("new labels and values to associate with this note")


@schema()
class RemoveNoteArgs:
    id: str = prop("note id to remove")


@schema()
class LookupNotesArgs:
    reference: str = prop("reference text")
    max_results: int = prop("maximum number of documents to return", default=5)


# A tool for recording ephemeral notes for later consultation.
# The notes are stored in a vector database with computed embeddings for later retrieval.
class Notebook(Tool):
    _store: Docstore

    def __init__(
        self,
        embed_model: TextEmbeddingModel,
        dbclient: chromadb.api.ClientAPI,
        dir: str,
    ):
        Tool.__init__(self)
        self._store = Docstore(embed_model, dbclient, "notebook", dir)
        self.declare(self.record_note)
        self.declare(self.remove_note)
        self.declare(self.lookup_notes)

    @decl(
        "record_note",
        "Records a note in the notebook for later research. Try to use a simple semantic id.",
    )
    def record_note(self, args: RecordNoteArgs) -> None:
        self._store.create(_clean(args.id), _clean(args.text), args.labels)

    @decl(
        "update_note",
        "Updates a note in the notebook, changing its name and content",
    )
    def update_note(self, args: UpdateNoteArgs) -> None:
        self._store.update(
            _clean(args.id), _clean(args.new_id), _clean(args.text), args.labels
        )

    @decl(
        "remove_note",
        "Removes a note from the notebook",
    )
    def remove_note(self, args: RemoveNoteArgs) -> None:
        self._store.delete(_clean(args.id))

    @decl("lookup_note", "Looks up notes in the notebook.")
    def lookup_notes(self, args: LookupNotesArgs) -> List[Doc]:
        return self._store.find(args.reference, args.max_results)


def _clean(text: str) -> str:
    """LMs sometimes generate unnecessarily escaped characters."""
    return text.replace("\\n", "\n").replace("\\", "")

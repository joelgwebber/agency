"""Tools for recording notes for later consultation.

The notes are stored in a vector database with computed embeddings for later retrieval.
"""

from dataclasses import dataclass
from typing import Dict, List

from agency.schema import Schema, Type, parse_val, prop, schema, schema_for
from agency.tool import Tool, ToolCall, ToolDecl, ToolResult
from agency.tools.docstore import Docstore


@dataclass
class RecordNote(Tool):
    @schema()
    class Params:
        id: str = prop("unique note id")
        text: str = prop("note text")
        labels: Dict[str, str] = prop(
            "labels and values to associate with this note", default_factory=lambda: {}
        )

    decl = ToolDecl(
        "record-note",
        "Records a note in the notebook for later research. Use simple semantic ids.",
        schema_for(Params),
        Schema(Type.Object, ""),
    )

    store: Docstore

    def invoke(self, req: ToolCall) -> ToolResult:
        args = parse_val(req.args, RecordNote.decl.params)
        self.store.create(_clean(args.id), _clean(args.text), args.labels)
        return ToolResult({})


@dataclass
class UpdateNote(Tool):
    @schema()
    class Params:
        id: str = prop("note id to update")
        new_id: str = prop("new note id (may be the same)")
        text: str = prop("new note text")
        labels: Dict[str, str] = prop(
            "new labels and values to associate with this note"
        )

    decl = ToolDecl(
        "update-note",
        "Updates a note from the notebook.",
        schema_for(Params),
        Schema(Type.Object, ""),
    )

    store: Docstore

    def invoke(self, req: ToolCall) -> ToolResult:
        args = parse_val(req.args, UpdateNote.decl.params)
        self.store.update(
            _clean(args.id), _clean(args.new_id), _clean(args.text), args.labels
        )
        return ToolResult({})


@dataclass
class RemoveNote(Tool):
    @schema()
    class Params:
        id: str = prop("note id to remove")

    decl = ToolDecl(
        "remove-note",
        "Removes a note from the notebook.",
        schema_for(Params),
        Schema(Type.Object, ""),
    )

    store: Docstore

    def invoke(self, req: ToolCall) -> ToolResult:
        args = parse_val(req.args, RemoveNote.decl.params)
        self.store.delete(_clean(args.id))
        return ToolResult({})


@dataclass
class LookupNotes(Tool):
    @schema()
    class Params:
        reference: str = prop("reference text")
        max_results: int = prop("maximum number of documents to return", default=5)

    @schema()
    class Returns:
        notes: List[str] = prop("notes matching the reference text")

    decl = ToolDecl(
        "lookup-notes",
        "Looks up notes in the notebook.",
        schema_for(Params),
        schema_for(Returns),
    )

    store: Docstore

    def invoke(self, req: ToolCall) -> ToolResult:
        args = parse_val(req.args, LookupNotes.decl.params)
        docs = self.store.find(args.reference, args.max_results)
        return ToolResult({"notes": docs})


def _clean(text: str) -> str:
    """LMs sometimes generate unnecessarily escaped characters."""
    return text.replace("\\n", "\n").replace("\\", "")

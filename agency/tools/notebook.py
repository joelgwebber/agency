"""Tools for recording notes for later consultation.

The notes are stored in a vector database with computed embeddings for later retrieval.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

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
from agency.tools.docstore import Docstore


@dataclass
class RecordNote(Tool):
    @schema
    class Params:
        id: str = prop("unique note id")
        text: str = prop("note text")
        labels: Dict[str, str] = prop(
            "labeled string values to associate with this note",
            default_factory=lambda: {},
        )

    store: Docstore

    @property
    def decl(self) -> ToolDecl:
        return ToolDecl(
            "record-note",
            "Records a note in the notebook for later research. Use simple semantic ids.",
            schema_for(RecordNote.Params),
            Schema(Type.Object, ""),
        )

    def invoke(self, stack: Stack):
        args = parse_val(stack.top().args, self.decl.params)
        self.store.create(_clean(args.id), _clean(args.text), args.labels)
        stack.respond({})


@dataclass
class UpdateNote(Tool):
    @schema
    class Params:
        id: str = prop("note id to update")
        new_id: str = prop("new note id (may be the same)")
        text: str = prop("new note text")
        labels: Dict[str, str] = prop(
            "new labels and values to associate with this note"
        )

    store: Docstore

    @property
    def decl(self) -> ToolDecl:
        return ToolDecl(
            "update-note",
            "Updates a note from the notebook.",
            schema_for(UpdateNote.Params),
            Schema(Type.Object, ""),
        )

    def invoke(self, stack: Stack):
        args = parse_val(stack.top().args, self.decl.params)
        self.store.update(
            _clean(args.id), _clean(args.new_id), _clean(args.text), args.labels
        )
        stack.respond({})


@dataclass
class RemoveNote(Tool):
    @schema
    class Params:
        id: str = prop("note id to remove")

    store: Docstore

    def invoke(self, stack: Stack):
        args = parse_val(stack.top().args, self.decl.params)
        self.store.delete(_clean(args.id))
        stack.respond({})

    @property
    def decl(self) -> ToolDecl:
        return ToolDecl(
            "remove-note",
            "Removes a note from the notebook.",
            schema_for(RemoveNote.Params),
            Schema(Type.Object, ""),
        )


@dataclass
class LookupNotes(Tool):
    @schema
    class Params:
        reference: str = prop("reference text")
        max_results: int = prop("maximum number of documents to return", default=5)

    @schema
    class Returns:
        notes: List[str] = prop("notes matching the reference text")

    store: Docstore

    @property
    def decl(self) -> ToolDecl:
        return ToolDecl(
            "lookup-notes",
            "Looks up notes in the notebook.",
            schema_for(LookupNotes.Params),
            schema_for(LookupNotes.Returns),
        )

    def invoke(self, stack: Stack):
        args = parse_val(stack.top().args, self.decl.params)
        docs = self.store.find(args.reference, args.max_results)
        stack.respond(serialize_val(LookupNotes.Returns(notes=docs), self.decl.returns))


def _clean(text: str) -> str:
    """LMs sometimes generate unnecessarily escaped characters."""
    return text.replace("\\n", "\n").replace("\\", "")

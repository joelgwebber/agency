import os

import chromadb

from agency import Agency
from agency.keys import TAVILY_API_KEY
from agency.minion import MinionDecl
from agency.schema import schema, schema_for
from agency.tools.browse import Browse
from agency.tools.docstore import Docstore
from agency.tools.feedback import GetFeedback, LogStore, SubmitFeedback
from agency.tools.notebook import LookupNotes, RecordNote, RemoveNote, UpdateNote
from agency.tools.search import Search
from agency.ui import AgencyUI

tool_name = "world"
dbclient = chromadb.PersistentClient(os.path.join(tool_name, "chroma"))
knowledge = Docstore(dbclient, tool_name, "knowledge")
feedback = LogStore(dbclient, tool_name, "feedback")


@schema()
class WorldBuilderArgs:
    question: str


WorldBuilder = MinionDecl(
    "world-builder",
    """A world-building assistant.""",
    schema_for(WorldBuilderArgs),
    """You are a world-building assistant, helping to construct fictional settings and background material.
    - Maintain consistency with existing material in the notebook
    - Notes should be in Markdown format
    - Link to other notes using [[note-id]] syntax

    Current request: {{ question }}""",
    [
        RecordNote.decl,
        UpdateNote.decl,
        RemoveNote.decl,
        LookupNotes.decl,
        SubmitFeedback.decl,
        GetFeedback.decl,
    ],
)

tools = [
    Browse(),
    Search(TAVILY_API_KEY),
    RecordNote(knowledge),
    UpdateNote(knowledge),
    RemoveNote(knowledge),
    LookupNotes(knowledge),
    SubmitFeedback(feedback),
    GetFeedback(feedback),
]

agency = Agency(tools, [WorldBuilder])
AgencyUI(agency, WorldBuilder.id).run()

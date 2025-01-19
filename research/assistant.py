from __future__ import annotations

import os

import chromadb

from agency import Agency
from agency.keys import TAVILY_API_KEY
from agency.minion import Minion
from agency.tools.annotations import schema, schema_for
from agency.tools.browse import Browse
from agency.tools.docstore import Docstore
from agency.tools.feedback import GetFeedback, SubmitFeedback
from agency.tools.notebook import LookupNotes, RecordNote, RemoveNote, UpdateNote
from agency.tools.search import Search
from agency.tools.tools import ToolDecl
from agency.ui import AgencyUI

tool_name = "research"
dbclient = chromadb.PersistentClient(os.path.join(tool_name, "chroma"))
fb_coll = dbclient.get_or_create_collection("feedback")
notebook = Docstore(dbclient, tool_name, "notebook")


@schema()
class ResearchArgs:
    question: str


GeneralKnowledge = Minion(
    ToolDecl(
        "general-knowledge",
        "Answers questions about general knowledge, useful as a starting point for further research",
        schema_for(ResearchArgs),
    ),
    """Answer the following question in general terms, with the goal of creating starting points for further research.
    Response format: {"answer": "(answer)", "search_terms": ["(term0)", "(term1)", ...]}""",
    [],
)


ResearchAssistant = Minion(
    ToolDecl(
        "research-assistant",
        "A research assistant",
        schema_for(ResearchArgs),
    ),
    """You are a research assistant, helping your human understand any topic.
    - Use the notebook to record notes as you go, looking them up by subject when needed
    - Notebook entries are in markdown format, using [[wiki-links]] to connect them.
    - DO NOT use any other information to answer questions
    - Always cite your sources
    Current question: {{ question }}""",
    [
        Search.decl,
        Browse.decl,
        RecordNote.decl,
        UpdateNote.decl,
        RemoveNote.decl,
        LookupNotes.decl,
        GeneralKnowledge.decl,
    ],
)

tools = [
    ResearchAssistant,
    GeneralKnowledge,
    Browse(),
    Search(TAVILY_API_KEY),
    RecordNote(notebook),
    UpdateNote(notebook),
    RemoveNote(notebook),
    LookupNotes(notebook),
    SubmitFeedback(fb_coll),
    GetFeedback(fb_coll),
]


def run():
    agency = Agency(tools)
    AgencyUI(agency, ResearchAssistant.decl.id).run()


run()

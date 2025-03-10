from __future__ import annotations

import os

import chromadb

from agency import Agency
from agency.keys import TAVILY_API_KEY
from agency.minion import Minion
from agency.models.openrouter import OpenRouter
from agency.schema import schema, schema_for
from agency.tool import ToolDecl
from agency.tools.browse import Browse
from agency.tools.docstore import Docstore
from agency.tools.feedback import GetFeedback, LogStore, SubmitFeedback
from agency.tools.files import EditFile, ReadFile
from agency.tools.notebook import LookupNotes, RecordNote, RemoveNote, UpdateNote
from agency.tools.search import Search
from agency.ui import AgencyUI

tool_name = "research"
dbclient = chromadb.PersistentClient(os.path.join(tool_name, "chroma"))
feedback = LogStore(dbclient, tool_name, "feedback")
notebook = Docstore(dbclient, tool_name, "notebook")
model = OpenRouter("anthropic/claude-3.5-sonnet")


@schema()
class ResearchParams:
    question: str


@schema()
class ResearchResults:
    answer: str


@schema()
class KnowledgeResults:
    answer: str
    search_terms: str


GeneralKnowledge = Minion(
    ToolDecl(
        "general-knowledge",
        "Answers questions about general knowledge, useful as a starting point for further research",
        schema_for(ResearchParams),
        schema_for(KnowledgeResults),
    ),
    model,
    """Answer the following question in general terms, with the goal of creating starting points for further research.
    Response format: {"answer": "(answer)", "search_terms": ["(term0)", "(term1)", ...]}:
    {{ question }}""",
    [],
)


ResearchAssistant = Minion(
    ToolDecl(
        "research-assistant",
        "A research assistant",
        schema_for(ResearchParams),
        schema_for(ResearchResults),
    ),
    model,
    """You are a research assistant, helping your human understand any topic.
    - Use the notebook to record notes as you go, looking them up by subject when needed
    - Notebook entries are in markdown format, using [[wiki-links]] to connect them.
    - DO NOT use any other information to answer questions
    - Always cite your sources
    Current question: {{ question }}""",
    [
        GeneralKnowledge.decl,
        Search.decl,
        Browse.decl,
        # RecordNote.decl,
        # UpdateNote.decl,
        # RemoveNote.decl,
        # LookupNotes.decl,
        ReadFile.decl,
        EditFile.decl,
    ],
)


def run():
    agency = Agency(
        [
            ResearchAssistant,
            GeneralKnowledge,
            Browse(),
            Search(TAVILY_API_KEY),
            RecordNote(notebook),
            UpdateNote(notebook),
            RemoveNote(notebook),
            LookupNotes(notebook),
            ReadFile("research/src"),
            EditFile("research/src"),
            SubmitFeedback(feedback),
            GetFeedback(feedback),
        ]
    )
    AgencyUI(agency, ResearchAssistant.decl.id).run()


run()

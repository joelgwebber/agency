from __future__ import annotations

import os
from typing import List

import chromadb

from agency import Agent
from agency.keys import TAVILY_API_KEY
from agency.minion import Minion
from agency.models.openrouter import OpenRouter
from agency.schema import schema, schema_for
from agency.tool import ToolDecl
from agency.toolbox import Toolbox
from agency.tools.browse import Browse
from agency.tools.docstore import Docstore
from agency.tools.feedback import GetFeedback, LogStore, SubmitFeedback
from agency.tools.files import EditFile, ReadFile
from agency.tools.notebook import LookupNotes, RecordNote, RemoveNote, UpdateNote
from agency.tools.search import Search
from agency.ui import AgentUI

# Shared resources
agent_name = "research"
dbclient = chromadb.PersistentClient(os.path.join(agent_name, "chroma"))
feedback = LogStore(dbclient, agent_name, "feedback")
notebook = Docstore(dbclient, agent_name, "notebook")
sonnet = OpenRouter("anthropic/claude-3.5-sonnet")

# Tools
search = Search(TAVILY_API_KEY)
browse = Browse()
record_note = RecordNote(notebook)
update_note = UpdateNote(notebook)
remove_note = RemoveNote(notebook)
lookup_notes = LookupNotes(notebook)
read_file = ReadFile("research/src")
edit_file = EditFile("research/src")
submit_feedback = SubmitFeedback(feedback)
get_feedback = GetFeedback(feedback)


# General knowledge minion
@schema
class KnowledgeParams:
    question: str


@schema
class KnowledgeResults:
    answer: str
    search_terms: List[str]


GeneralKnowledge = Minion(
    ToolDecl(
        "general-knowledge",
        "Answers questions about general knowledge, useful as a starting point for further research",
        schema_for(KnowledgeParams),
        schema_for(KnowledgeResults),
    ),
    sonnet,
    """Answer the following question in general terms, with the goal of creating starting points for further research:
    {{ question }}""",
)


# Research minion
@schema
class ResearchParams:
    question: str


@schema
class ResearchResults:
    answer: str


ResearchAssistant = Minion(
    ToolDecl(
        "research-assistant",
        "A research assistant",
        schema_for(ResearchParams),
        schema_for(ResearchResults),
    ),
    sonnet,
    """You are a research assistant, helping your human understand any topic.
    - Use the notebook to record notes as you go, looking them up by subject when needed
    - Notebook entries are in markdown format, using [[wiki-links]] to connect them.
    - DO NOT use any other information to answer questions
    - Always cite your sources
    Current question: {{ question }}""",
    search.decl,
    browse.decl,
    record_note.decl,
    update_note.decl,
    remove_note.decl,
    lookup_notes.decl,
)


# Code minion
@schema
class CodeParams:
    question: str


@schema
class CodeResults:
    answer: str


CodeAssistant = Minion(
    ToolDecl(
        "code-assistant",
        "Operates on source files in a local repository",
        schema_for(CodeParams),
        schema_for(CodeResults),
    ),
    sonnet,
    """ """,
)


# Run the agent
toolbox = Toolbox(
    GeneralKnowledge,
    ResearchAssistant,
    search,
    browse,
    record_note,
    update_note,
    remove_note,
    lookup_notes,
    read_file,
    edit_file,
    submit_feedback,
    get_feedback,
)


def run():
    agent = Agent(
        """You are a general assistant, that can use tools to answer the user's questions.""",
        sonnet,
        toolbox,
        GeneralKnowledge.decl,
        ResearchAssistant.decl,
    )
    AgentUI(agent).run()


run()

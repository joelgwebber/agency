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
o3_mini = OpenRouter("openai/o3-mini")

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
class Knowledge(Minion):
    @schema
    class Params:
        question: str

    @schema
    class Results:
        answer: str
        search_terms: List[str]

    Template = """Answer the following question in general terms, with the goal of creating starting points for further research: {{ question }}"""

    def __init__(self):
        Minion.__init__(self, o3_mini, Knowledge.Template)

    @property
    def decl(self) -> ToolDecl:
        return ToolDecl(
            "general-knowledge",
            "Answers questions about general knowledge, useful as a starting point for further research",
            schema_for(Knowledge.Params),
            schema_for(Knowledge.Results),
        )


# Research minion
class Research(Minion):
    @schema
    class Params:
        question: str

    @schema
    class Results:
        answer: str

    Template = """
        You are a research assistant, helping your human understand any topic.
        - Use the notebook to record notes as you go, looking them up by subject when needed
        - Notebook entries are in markdown format, using [[wiki-links]] to connect them.
        - DO NOT use any other information to answer questions
        - Always cite your sources
        Current question: {{ question }}
    """

    def __init__(self):
        Minion.__init__(
            self,
            o3_mini,
            Research.Template,
            search.decl,
            browse.decl,
            record_note.decl,
            update_note.decl,
            remove_note.decl,
            lookup_notes.decl,
        )

    @property
    def decl(self) -> ToolDecl:
        return ToolDecl(
            "research-assistant",
            "A research assistant",
            schema_for(Research.Params),
            schema_for(Research.Results),
        )


# Code minion
class Code(Minion):
    @schema
    class Params:
        question: str

    @schema
    class Results:
        answer: str

    Template = """TODO"""

    def __init__(self):
        Minion.__init__(self, o3_mini, Code.Template)

    @property
    def decl(self):
        return ToolDecl(
            "code-assistant",
            "Operates on source files in a local repository",
            schema_for(Code.Params),
            schema_for(Code.Results),
        )


# Run the agent
toolbox = Toolbox(
    Knowledge(),
    Research(),
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
        o3_mini,
        toolbox,
        Knowledge().decl,
        Research().decl,
    )
    AgentUI(agent).run()


run()

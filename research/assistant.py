import os

import chromadb

from agency import Agency
from agency.keys import TAVILY_API_KEY
from agency.minion import MinionDecl
from agency.schema import schema, schema_for
from agency.tools.browse import Browse
from agency.tools.docstore import Docstore
from agency.tools.feedback import FeedbackStore, GetFeedback, SubmitFeedback
from agency.tools.notebook import LookupNotes, RecordNote, RemoveNote, UpdateNote
from agency.tools.search import Search
from agency.ui import AgencyUI

tool_name = "research"
dbclient = chromadb.PersistentClient(os.path.join(tool_name, "chroma"))
feedback = FeedbackStore(dbclient, tool_name, "feedback")
notebook = Docstore(dbclient, tool_name, "notebook")


@schema()
class ResearchArgs:
    question: str


ResearchAssistant = MinionDecl(
    "research-assistant",
    "A research assistant",
    schema_for(ResearchArgs),
    """You are a research assistant, helping your human understand any topic.
    - Use the notebook to record notes as you go, looking them up by subject when needed
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
        SubmitFeedback.decl,
        GetFeedback.decl,
    ],
)

tools = [
    Browse(),
    Search(TAVILY_API_KEY),
    RecordNote(notebook),
    UpdateNote(notebook),
    RemoveNote(notebook),
    LookupNotes(notebook),
    SubmitFeedback(feedback),
    GetFeedback(feedback),
]

agency = Agency(tools, [ResearchAssistant])
AgencyUI(agency, ResearchAssistant.id).run()

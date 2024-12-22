import os
from typing import List

import chromadb
from vertexai.generative_models import GenerativeModel, Part
from vertexai.language_models import TextEmbeddingModel

from agency import Agency
from agency.agency import Minion, required_instructions
from agency.tools import Tool
from agency.tools.browse import Browse
from agency.tools.notebook import Notebook
from agency.tools.recipes import Recipes
from agency.tools.search import Search
from agency.ui import AgencyUI

TAVILY_API_KEY = os.environ["TAVILY_API_KEY"]

work_dir = "work/world"
dbclient = chromadb.PersistentClient(work_dir + "/chroma")

# This is the most recent embedding model I'm aware of.
# It has a 2k input limit, so we might have to break some things up.
embed_model = TextEmbeddingModel.from_pretrained("text-embedding-004")

sys_instr = """
You are an assistant helping to construct fictional settings and background material. When generating content and
answering questions, always maintain consistency with existing material, available as project knowledge.

Notes and recipes are in Markdown format, and can be linked using the [[note-id]] syntax. When creating and updating
notes, try to ensure that links are created for other notes known to exist.
"""

sys_suffix = """
DO NOT write or update notes until explicitly asked to do so. Before making changes, give a simple description of the
changes you intend to make, and ask for confirmation.
"""

# Use Gemini 1.5 Flash.
model = GenerativeModel(
    "gemini-1.5-flash-preview-0514",
    system_instruction=[
        Part.from_text(sys_instr),
        Part.from_text(required_instructions),
        Part.from_text(sys_suffix),
    ],
)


class Builder(Minion):
    def __init__(self):
        super().__init__(
            [
                Recipes(embed_model, dbclient, "world/recipes"),
                Notebook(embed_model, dbclient, "world/knowledge"),
                Search(TAVILY_API_KEY),
                Browse(),
            ]
        )


agency = Agency(model, Builder())
AgencyUI(agency).run()

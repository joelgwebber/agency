import os
from typing import List

import chromadb
from vertexai.generative_models import GenerativeModel, Part
from vertexai.language_models import TextEmbeddingModel

from agency import Agency
from agency.agency import required_instructions
from agency.tools import Tool
from agency.tools.browse import Browse
from agency.tools.notebook import Notebook
from agency.tools.recipes import Recipes
from agency.tools.search import Search

TAVILY_API_KEY = os.environ["TAVILY_API_KEY"]

work_dir = "work/research"
dbclient = chromadb.PersistentClient(work_dir + "/chroma")

# This is the most recent embedding model I'm aware of.
# It has a 2k input limit, so we might have to break some things up.
embed_model = TextEmbeddingModel.from_pretrained("text-embedding-004")

tools: List[Tool] = [
    Recipes(embed_model, dbclient, "recipes", "research/recipes"),
    Notebook(embed_model, dbclient),
    Search(TAVILY_API_KEY),
    Browse(),
]

sys_instr = "You are a research assistant, helping your human understand any topic."
sys_suffix = """
- Use the notebook to record notes as you go, looking them up by subject when needed.
- DO NOT use any other information to answer questions.
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

agency = Agency(model, tools)

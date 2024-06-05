import os
from typing import List

import chromadb
from vertexai.language_models import TextEmbeddingModel

from agency import Agency
from agency.tools import Tool
from agency.tools.browse import Browse
from agency.tools.notebook import Notebook
from agency.tools.recipes import Recipes
from agency.tools.search import Search

TAVILY_API_KEY = os.environ["TAVILY_API_KEY"]

dbclient = chromadb.PersistentClient("work/chroma")

# This is the most recent embedding model I'm aware of.
# It has a 2k input limit, so we might have to break some things up.
embed_model = TextEmbeddingModel.from_pretrained("text-embedding-004")

tools: List[Tool] = [
    Recipes(embed_model, dbclient, "recipes"),
    Notebook(embed_model, dbclient),
    Search(TAVILY_API_KEY),
    Browse(),
]

agency = Agency(
    tools,
    f"""You are a research assistant, helping your human understand any topic.""",
)

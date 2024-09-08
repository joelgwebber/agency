from typing import List

import chromadb
import chromadb.api
from vertexai.language_models import TextEmbeddingModel

from agency.tools import Tool
from agency.tools.annotations import decl, prop, schema
from agency.tools.docstore import Docstore


@schema()
class FindRecipesArgs:
    goal: str = prop("A brief description of the goal")
    number: int = prop("Number of recipes to find, minimum 3", default=10)


class Recipes(Tool):
    _store: Docstore

    def __init__(
        self,
        embed_model: TextEmbeddingModel,
        dbclient: chromadb.api.ClientAPI,
        *dirs: str,
    ):
        Tool.__init__(self)
        self._store = Docstore(embed_model, dbclient, "recipes", None, *dirs)
        self.declare(self.find_recipes)

    # TODO: Make 'number' an int. Gotta figure out how to cast the proto args in dispatch() properly.
    @decl(
        "find_recipes",
        "Finds recipes that explain how to approach various tasks. ALWAYS call this first for a new question.",
    )
    def find_recipes(self, args: FindRecipesArgs) -> List[str]:
        return self._store.find(args.goal, args.number)

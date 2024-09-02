import os
from glob import glob
from hashlib import md5
from typing import List

import chromadb
import chromadb.api
from vertexai.language_models import TextEmbeddingInput, TextEmbeddingModel

from agency.tools import Tool
from agency.tools.annotations import decl, prop, schema


@schema()
class FindRecipesArgs:
    goal: str = prop("A brief description of the goal")
    number: int = prop("Number of recipes to find, minimum 3", default=10)


class Recipes(Tool):
    _recipes_coll: chromadb.Collection
    _embed_model: TextEmbeddingModel

    def __init__(
        self,
        embed_model: TextEmbeddingModel,
        dbclient: chromadb.api.ClientAPI,
        *dirs: str,
    ):
        Tool.__init__(self)

        self._embed_model = embed_model

        # TODO: Perform garbage collection for stale entries. Otherwise the database
        # gets cluttered up with old recipes and versions of them.
        #
        # For now, just wipe all the recipes for quick updates.
        # dbclient.delete_collection("recipes")

        self._recipes_coll = dbclient.create_collection(
            name="recipes", get_or_create=True
        )

        # Update recipes from disk contents.
        #
        self._update("thinker/recipes")
        for dir in dirs:
            self._update(dir)

        self.declare(self.find_recipes)

    # TODO: Make 'number' an int. Gotta figure out how to cast the proto args in dispatch() properly.
    @decl(
        "find_recipes",
        "Finds recipes that explain how to approach various tasks. ALWAYS call this first for a new question.",
    )
    def find_recipes(self, args: FindRecipesArgs) -> List[str]:
        vec = self._embed(args.goal)
        rsp = self._recipes_coll.query(
            query_embeddings=[vec], n_results=int(args.number)
        )
        if rsp["documents"] is not None:
            return dedupe(rsp["documents"][0])
        return []

    def _update(self, dir: str):
        # TODO: Group embed calls for efficiency.
        pattern = os.path.join(dir, f"*.md")
        for file_path in glob(pattern):
            print(f"--- recipe: {file_path}")
            with open(file_path, "r") as file:
                self._ensure(file.read())

    def _ensure(self, recipe: str):
        recipe_hash = md5(recipe.encode(), usedforsecurity=False).hexdigest()

        # Already extant?
        result = self._recipes_coll.get(ids=recipe_hash)
        if result["documents"] is not None:
            if len(result["documents"]) > 0:
                return

        # Nope. Embed and add it.
        print(f"    [re-]embedding {recipe_hash}")
        embedding = self._embed(recipe)
        self._recipes_coll.add(
            ids=recipe_hash,
            documents=recipe,
            embeddings=embedding,
        )

    def _embed(self, recipe: str) -> List[float]:
        # SEMANTIC_SIMILARITY seems to cluster too tightly, leading to repeated entries.
        inputs: List = [TextEmbeddingInput(recipe, "CLASSIFICATION")]
        embeddings = self._embed_model.get_embeddings(inputs)
        return embeddings[0].values


def dedupe(arr: List[str]) -> List[str]:
    m = {k: True for k in arr}
    return list(m.keys())

import os
from glob import glob
from hashlib import md5
from typing import Dict, List, Optional, Tuple, TypedDict

import chromadb.api
from chromadb import Metadata
from vertexai.language_models import TextEmbeddingInput, TextEmbeddingModel


class Doc(TypedDict):
    id: str
    labels: Dict[str, str]
    text: str


class Docstore:
    _embed_model: TextEmbeddingModel
    _coll: chromadb.Collection
    _work_dir: Optional[str]

    def __init__(
        self,
        embed_model: TextEmbeddingModel,
        dbclient: chromadb.api.ClientAPI,
        name: str,
        dir: str,
    ):
        # TODO: Perform garbage collection for stale entries. Otherwise the database
        #   gets cluttered up with old docs and versions of them.
        # For now, just wipe the database after making manual file changes.
        self._embed_model = embed_model
        self._coll = dbclient.create_collection(name=name, get_or_create=True)

        # Update recipes from disk contents.
        self._work_dir = dir
        self._load_dir(dir)

    def exists(self, id: str) -> Tuple[bool, Dict[str, str]]:
        result = self._coll.get(ids=id, include=["metadatas"])
        meta = result["metadatas"]
        if meta is not None and len(meta) > 0:
            return True, meta_labels(meta[0])
        return False, {}

    def create(self, id: str, text: str, labels: Dict[str, str]) -> None:
        # If a directory was specified, write the doc to disk.
        if self._work_dir:
            os.makedirs(self._work_dir, exist_ok=True)
            with open(os.path.join(self._work_dir, f"{id}.md"), "w") as file:
                header = "\n".join([f"{k}: {labels[k]}" for k in labels])
                file.write("---\n" + header + "\n---\n" + text)
                file.close()

        self._index_doc(id, text, labels)

    def delete(self, id: str) -> None:
        if not self.exists(id):
            raise Exception(f"note {id} does not exist")
        self._coll.delete(ids=[id])
        os.unlink(self._doc_file(id))

    def update(self, id: str, new_id: str, text: str, labels: Dict[str, str]) -> None:
        if self.exists(id):
            self.delete(id)
        self.create(new_id, text, labels)

    def find(self, query: str, number: int) -> List[Doc]:
        vec = self._embed(query)
        rsp = self._coll.query(
            query_embeddings=[vec],
            n_results=int(number),
            include=["documents", "metadatas", "uris"],
        )

        results: List[Doc] = []
        if (
            rsp["documents"] is not None
            and rsp["metadatas"] is not None
            and rsp["ids"] is not None
        ):
            docs = rsp["documents"][0]
            metas = rsp["metadatas"][0]
            ids = rsp["ids"][0]
            for i in range(0, len(docs)):
                # Convert labels; skip "hash" to avoid confusing the model.
                labels = meta_labels(metas[i])
                del labels["hash"]
                results.append(Doc(id=ids[i], labels=labels, text=docs[i]))
            return results
        return []

    def _doc_file(self, id: str) -> str:
        name = f"{id}.md"
        if self._work_dir:
            return os.path.join(self._work_dir, name)
        return name

    def _load_dir(self, dir: str):
        # TODO: Group embed calls for efficiency.
        pattern = os.path.join(dir, f"*.md")
        for file_path in glob(pattern):
            id = file_id(file_path)
            self._load_doc(id, file_path)

    def _load_doc(self, id: str, file_path: str):
        with open(file_path, "r") as file:
            content = file.read()

        # Parse the header if present.
        labels = {}
        text = content.strip()
        parts = content.split("---", 2)
        if len(parts) == 3:
            # File has a header
            header = parts[1].strip()
            labels = {
                k.strip(): v.strip()
                for k, v in (
                    line.split(":", 1) for line in header.split("\n") if ":" in line
                )
            }
            text = parts[2].strip()

        self._index_doc(id, text, labels)

    def _index_doc(self, id: str, text: str, labels: Dict[str, str]):
        # Already extant?
        text_hash = md5((f"{id} : {text}").encode(), usedforsecurity=False).hexdigest()
        exists, old_labels = self.exists(id)
        if exists and ("hash" in old_labels) and (text_hash == old_labels["hash"]):
            return

        # Nope. Embed and add it.
        print(f"--- [re-]embedding {id}\n    {text_hash} : {old_labels}")
        labels["hash"] = text_hash
        embedding = self._embed(text)
        self._coll.add(
            ids=id,
            documents=text,
            embeddings=embedding,
            metadatas=labels,
        )

    def _embed(self, text: str) -> List[float]:
        # SEMANTIC_SIMILARITY seems to cluster too tightly, leading to repeated entries.
        inputs: List = [TextEmbeddingInput(text, "CLASSIFICATION")]
        embeddings = self._embed_model.get_embeddings(inputs)
        return embeddings[0].values


def file_id(file: str) -> str:
    base = os.path.basename(file)
    return os.path.splitext(base)[0]


def meta_labels(meta: Metadata) -> Dict[str, str]:
    labels: Dict[str, str] = {}
    for key in meta:
        labels[key] = str(meta[key])
    return labels

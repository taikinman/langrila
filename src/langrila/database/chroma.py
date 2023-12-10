import sys

python_version = sys.version_info
# NOTE : Python version < 3.10 is bundled by lower version sqlite client, so in that case sqlite modules is override
# https://docs.trychroma.com/troubleshooting#sqlite
__import__("pysqlite3")
sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")

from pathlib import Path
from typing import Optional

import chromadb

from ..base import BaseModule
from ..result import RetrievalResult
from ..usage import Usage


class ChromaCollectionModule(BaseModule):
    def __init__(
        self,
        persistence_directory: str,
        collection_name: str,
        embedder: Optional[BaseModule] = None,
    ):
        self.embedder = embedder
        self.persistence_directory = Path(persistence_directory)
        self.collection_name = collection_name

    def run(
        self,
        documents: list[str],
        metadatas: Optional[list[dict[str, str]]]=None,
        embeddings: Optional[list[list[float]]] = None,
    ) -> None:
        if embeddings is None:
            if self.embedder is not None:
                embeddings = self.embedder(documents).embeddings
            else:
                raise AttributeError(
                    "attribute embedder must be the instance of the class inheriting BaseModule."
                )

        ids = [str(i) for i in range(len(documents))]

        client = chromadb.PersistentClient(path=self.persistence_directory.as_posix())

        # recreation collection
        try:
            client.delete_collection(name=self.collection_name)
        except ValueError:
            pass

        collection = client.get_or_create_collection(
            name=self.collection_name, metadata={"hnsw:space": "cosine"}
        )

        collection.upsert(ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas)

    def as_retriever(
        self, n_results: int = 4, threshold_similarity: float = 0.8, return_only_relevant_docs: bool = False
    ) -> "ChromaRetrievalModule":
        return ChromaRetrievalModule(
            embedder=self.embedder,
            persistence_directory=self.persistence_directory,
            collection_name=self.collection_name,
            n_results=n_results,
            threshold_similarity=threshold_similarity,
            return_only_relevant_docs=return_only_relevant_docs,
        )


class ChromaRetrievalModule(BaseModule):
    def __init__(
        self,
        embedder: BaseModule,
        persistence_directory: str,
        collection_name: str,
        n_results: int = 4,
        threshold_similarity: float = 0.8,
        return_only_relevant_docs: bool = False,
    ):
        assert isinstance(
            embedder, BaseModule
        ), "embedder must be the instance of the class inheriting BaseModule."
        self.embedder = embedder
        self.n_results = n_results
        self.threshold_similarity = threshold_similarity
        self.persistence_directory = persistence_directory
        self.collection_name = collection_name
        self.return_only_relevant_docs = return_only_relevant_docs
        self.n_results = n_results

    def run(
        self,
        query: str,
        where: Optional[dict] = None,
    ) -> dict:
        query_embed = self.embedder(query)

        client = chromadb.PersistentClient(path=self.persistence_directory.as_posix())
        collection = client.get_collection(name=self.collection_name)

        retrieved = collection.query(
            query_embeddings=query_embed.embeddings[0], n_results=self.n_results, where=where
        )

        _results = self.filter_with_distance(retrieved)

        results = RetrievalResult(
            ids=_results["ids"],
            documents=_results["documents"],
            metadatas=_results["metadatas"],
            similarities=_results["similarities"],
            usage=Usage(
                prompt_tokens=query_embed.usage.prompt_tokens,
                completion_tokens=0,
            ),
        )

        if self.return_only_relevant_docs:
            return "\n\n".join(results["documents"])

        return results

    def filter_with_distance(self, retrieval_results):

        results = dict(
            ids = [
                d
                for d, dist in zip(retrieval_results["ids"][0], retrieval_results["distances"][0])
                if dist <= 1-self.threshold_similarity
            ],
            documents=[
                d
                for d, dist in zip(
                    retrieval_results["documents"][0], retrieval_results["distances"][0]
                )
                if dist <= 1-self.threshold_similarity
            ],
            metadatas=[
                md
                for md, dist in zip(
                    retrieval_results["metadatas"][0], retrieval_results["distances"][0]
                )
                if dist <= 1-self.threshold_similarity
            ],
            similarities=[
                1-dist for dist in retrieval_results["distances"][0] if dist <= 1-self.threshold_similarity
            ]
        )

        return results

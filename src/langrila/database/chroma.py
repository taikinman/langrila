from typing import Any

import chromadb
from chromadb import DEFAULT_DATABASE, DEFAULT_TENANT
from chromadb.api import ClientAPI

from ..result import RetrievalResults
from .base import (
    BaseEmbeddingModule,
    BaseLocalCollectionModule,
    BaseLocalRetrievalModule,
)


class ChromaLocalCollectionModule(BaseLocalCollectionModule):
    def __init__(
        self,
        persistence_directory: str,
        collection_name: str,
        metadata: dict[str, str] | None = None,
        embedder: BaseEmbeddingModule | None = None,
        logger: Any | None = None,
        limit_collection_size: int = 10000,
        tenant: str = DEFAULT_TENANT,
        database: str = DEFAULT_DATABASE,
    ):
        super().__init__(
            persistence_directory=persistence_directory,
            collection_name=collection_name,
            embedder=embedder,
            logger=logger,
            limit_collection_size=limit_collection_size,
        )
        self.metadata = metadata or {"hnsw:space": "cosine"}
        self.tenant = tenant
        self.database = database

    def _glob(self, client: ClientAPI) -> list[str]:
        return [c.name for c in client.list_collections() if self.collection_name in c.name]

    def _exists(self, client: ClientAPI, collection_name: str) -> bool:
        return len([name for name in self._glob(client=client) if name == collection_name]) > 0

    def _create_collection(self, client: ClientAPI, collection_name: str) -> None:
        self.collection = client.create_collection(name=collection_name, metadata=self.metadata)

    def _delete_record(self, client: ClientAPI, collection_name: str, ids: list[str | int]) -> None:
        if not hasattr(self, "collection"):
            self.collection = client.get_collection(name=collection_name)
        self.collection.delete(ids=[str(i) for i in ids])

    def _upsert(
        self,
        client: ClientAPI,
        collection_name: str,
        ids: list[str | int],
        documents: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict[str, str]],
        **kwargs,
    ) -> None:
        if not hasattr(self, "collection"):
            self.collection = client.get_collection(name=collection_name)

        self.collection.upsert(
            ids=[str(i) for i in ids],
            embeddings=embeddings,
            documents=documents,
            metadatas=[m | {"document": doc} for m, doc in zip(metadatas, documents, strict=True)],
            **kwargs,
        )

        return

    def _delete_collection(self, client: ClientAPI, collection_name: str) -> None:
        client.delete_collection(name=collection_name)

    def get_client(self) -> ClientAPI:
        return chromadb.PersistentClient(
            path=self.persistence_directory.as_posix(), tenant=self.tenant, database=self.database
        )

    def as_retriever(
        self,
        n_results: int = 4,
        score_threshold: float = 0.5,
    ) -> "ChromaLocalRetrievalModule":
        return ChromaLocalRetrievalModule(
            embedder=self.embedder,
            persistence_directory=self.persistence_directory,
            collection_name=self.collection_name,
            n_results=n_results,
            score_threshold=score_threshold,
            logger=self.logger,
        )


class ChromaLocalRetrievalModule(BaseLocalRetrievalModule):
    def __init__(
        self,
        persistence_directory: str,
        collection_name: str,
        embedder: BaseEmbeddingModule | None = None,
        n_results: int = 4,
        score_threshold: float = 0.5,
        logger: Any | None = None,
        ascending: bool = True,
        tenant: str = DEFAULT_TENANT,
        database: str = DEFAULT_DATABASE,
    ):
        super().__init__(
            persistence_directory=persistence_directory,
            collection_name=collection_name,
            embedder=embedder,
            n_results=n_results,
            score_threshold=score_threshold,
            logger=logger,
            ascending=ascending,
        )
        self.tenant = tenant
        self.database = database

    def get_client(self) -> ClientAPI:
        return chromadb.PersistentClient(
            path=self.persistence_directory.as_posix(), tenant=self.tenant, database=self.database
        )

    def _glob(self, client: ClientAPI) -> list[str]:
        return [c.name for c in client.list_collections() if self.collection_name in c.name]

    def _retrieve(
        self,
        client: ClientAPI,
        collection_name: str,
        query_vector: list[float],
        n_results: int,
        score_threshold: float,
        filter: Any | None = None,
        **kwargs,
    ) -> RetrievalResults:
        collection = client.get_collection(name=collection_name)

        retrieved = collection.query(
            query_embeddings=query_vector,
            where=filter,
            include=["metadatas", "documents", "distances"],
            n_results=n_results,
            **kwargs,
        )

        ids = []
        scores = []
        documents = []
        metadatas = []
        collections = []

        for _id, dist, document, metadata in zip(
            retrieved["ids"][0],
            retrieved["distances"][0],
            retrieved["documents"][0],
            retrieved["metadatas"][0],
            strict=True,
        ):
            if dist <= score_threshold:
                ids.append(int(_id))
                scores.append(dist)
                documents.append(document)
                metadatas.append(metadata)
                collections.append(collection_name)

        return RetrievalResults(
            ids=ids,
            scores=scores,
            documents=documents,
            metadatas=metadatas,
            collections=collections,
        )

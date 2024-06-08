from typing import Any, Optional

from qdrant_client import AsyncQdrantClient, QdrantClient, models
from qdrant_client.models import Distance, VectorParams

from ..result import RetrievalResults
from .base import (
    BaseEmbeddingModule,
    BaseLocalCollectionModule,
    BaseLocalRetrievalModule,
    BaseRemoteCollectionModule,
    BaseRemoteRetrievalModule,
)


class QdrantLocalCollectionModule(BaseLocalCollectionModule):
    def __init__(
        self,
        persistence_directory: str,
        collection_name: str,
        vector_size: int,
        distance: Distance = Distance.COSINE,
        embedder: BaseEmbeddingModule | None = None,
        logger: Any | None = None,
        on_disk: bool = False,
    ):
        super().__init__(
            persistence_directory=persistence_directory,
            collection_name=collection_name,
            embedder=embedder,
            logger=logger,
        )
        self.vector_size = vector_size
        self.distance = distance
        self.on_disk = on_disk

    def _glob(self, client: QdrantClient) -> list[str]:
        return [
            c.name for c in client.get_collections().collections if self.collection_name in c.name
        ]

    def _exists(self, client: QdrantClient, collection_name: str) -> bool:
        return client.collection_exists(collection_name=collection_name)

    def _create_collection(self, client: QdrantClient, collection_name: str, **kwargs) -> None:
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=self.vector_size,
                distance=self.distance,
                on_disk=self.on_disk,
            ),
            **kwargs,
        )

    def _upsert(
        self,
        client: QdrantClient,
        collection_name: str,
        ids: list[str | int],
        embeddings: list[list[float]],
        metadatas: list[dict[str, str]],
    ) -> None:
        client.upsert(
            collection_name=collection_name,
            points=models.Batch(
                ids=ids,
                vectors=embeddings,
                payloads=metadatas,
            ),
        )

        return

    def _delete(self, client: QdrantClient, collection_name: str) -> None:
        client.delete_collection(collection_name=collection_name)

    def get_client(self) -> QdrantClient:
        return QdrantClient(path=self.persistence_directory)

    def as_retriever(
        self,
        n_results: int = 4,
        score_threshold: float = 0.8,
    ) -> "QdrantLocalRetrievalModule":
        return QdrantLocalRetrievalModule(
            embedder=self.embedder,
            persistence_directory=self.persistence_directory,
            collection_name=self.collection_name,
            n_results=n_results,
            score_threshold=score_threshold,
            logger=self.logger,
        )


class QdrantRemoteCollectionModule(BaseRemoteCollectionModule):
    def __init__(
        self,
        url: str,
        collection_name: str,
        vector_size: int,
        port: str = "6333",
        distance: Distance = Distance.COSINE,
        embedder: BaseEmbeddingModule | None = None,
        logger: Any | None = None,
        on_disk: bool = False,
    ):
        self.vector_size = vector_size
        self.distance = distance
        self.on_disk = on_disk

        super().__init__(
            collection_name=collection_name,
            embedder=embedder,
            logger=logger,
            url=url,
            port=port,
        )

    def _glob(self, client: QdrantClient) -> list[str]:
        return [
            c.name for c in client.get_collections().collections if self.collection_name in c.name
        ]

    def _exists(self, client: QdrantClient, collection_name: str) -> bool:
        return client.collection_exists(collection_name=collection_name)

    def _create_collection(self, client: QdrantClient, collection_name: str, **kwargs) -> None:
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=self.vector_size,
                distance=self.distance,
                on_disk=self.on_disk,
            ),
            **kwargs,
        )

    def _upsert(
        self,
        client: QdrantClient,
        collection_name: str,
        ids: list[str | int],
        embeddings: list[list[float]],
        metadatas: list[dict[str, str]],
    ) -> None:
        client.upsert(
            collection_name=collection_name,
            points=models.Batch(
                ids=ids,
                vectors=embeddings,
                payloads=metadatas,
            ),
        )

        return

    def _delete(self, client: QdrantClient, collection_name: str) -> None:
        client.delete_collection(collection_name=collection_name)

    async def _aglob(self, client: AsyncQdrantClient) -> list[str]:
        return [
            c.name
            for c in (await client.get_collections()).collections
            if self.collection_name in c.name
        ]

    async def _aexists(self, client: AsyncQdrantClient, collection_name: str) -> bool:
        return await client.collection_exists(collection_name=collection_name)

    async def _acreate_collection(
        self, client: AsyncQdrantClient, collection_name: str, **kwargs
    ) -> None:
        await client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=self.vector_size,
                distance=self.distance,
                on_disk=self.on_disk,
            ),
            **kwargs,
        )

    async def _aupsert(
        self,
        client: AsyncQdrantClient,
        collection_name: str,
        ids: list[str | int],
        embeddings: list[list[float]],
        metadatas: list[dict[str, str]],
    ) -> None:
        await client.upsert(
            collection_name=collection_name,
            points=models.Batch(
                ids=ids,
                vectors=embeddings,
                payloads=metadatas,
            ),
        )

        return

    async def _adelete(self, client: AsyncQdrantClient, collection_name: str) -> None:
        await client.delete_collection(collection_name=collection_name)

    def get_client(self) -> QdrantClient:
        return QdrantClient(url=self.url, port=self.port)

    def get_async_client(self) -> AsyncQdrantClient:
        return AsyncQdrantClient(url=self.url, port=self.port)

    def as_retriever(
        self,
        n_results: int = 4,
        score_threshold: float = 0.8,
    ) -> "QdrantRemoteRetrievalModule":
        return QdrantRemoteRetrievalModule(
            embedder=self.embedder,
            url=self.url,
            port=self.port,
            collection_name=self.collection_name,
            n_results=n_results,
            score_threshold=score_threshold,
            logger=self.logger,
        )


class QdrantLocalRetrievalModule(BaseLocalRetrievalModule):
    def __init__(
        self,
        persistence_directory: str,
        collection_name: str,
        embedder: BaseEmbeddingModule | None = None,
        n_results: int = 4,
        score_threshold: float = 0.8,
        logger: Any | None = None,
    ):
        super().__init__(
            persistence_directory=persistence_directory,
            collection_name=collection_name,
            embedder=embedder,
            n_results=n_results,
            score_threshold=score_threshold,
            logger=logger,
        )

    def get_client(self) -> QdrantClient:
        return QdrantClient(path=self.persistence_directory)

    def _glob(self, client: QdrantClient) -> list[str]:
        return [
            c.name for c in client.get_collections().collections if self.collection_name in c.name
        ]

    def _retrieve(
        self,
        client: QdrantClient,
        collection_name: str,
        query_vector: list[float],
        n_results: int,
        score_threshold: float,
        filter: Any | None = None,
        **kwargs,
    ) -> RetrievalResults:
        retrieved = client.search(
            collection_name=collection_name,
            query_vector=query_vector,
            query_filter=filter,
            score_threshold=score_threshold,
            with_vectors=False,
            limit=n_results,
            **kwargs,
        )

        ids = [r.id for r in retrieved]
        scores = [r.score for r in retrieved]
        documents = [r.payload["document"] for r in retrieved]
        metadatas = [r.payload["metadata"] for r in retrieved]
        collections = [r.payload["collection"] for r in retrieved]

        return RetrievalResults(
            ids=ids,
            scores=scores,
            documents=documents,
            metadatas=metadatas,
            collections=collections,
        )


class QdrantRemoteRetrievalModule(BaseRemoteRetrievalModule):
    def __init__(
        self,
        url: str,
        collection_name: str,
        port: str = "6333",
        embedder: BaseEmbeddingModule | None = None,
        n_results: int = 4,
        score_threshold: float = 0.8,
        logger: Any | None = None,
    ):
        super().__init__(
            collection_name=collection_name,
            embedder=embedder,
            n_results=n_results,
            score_threshold=score_threshold,
            logger=logger,
            url=url,
            port=port,
        )

    def get_client(self) -> QdrantClient:
        return QdrantClient(url=self.url, port=self.port)

    def get_async_client(self) -> AsyncQdrantClient:
        return AsyncQdrantClient(url=self.url, port=self.port)

    def _glob(self, client: QdrantClient) -> list[str]:
        return [
            c.name for c in client.get_collections().collections if self.collection_name in c.name
        ]

    def _retrieve(
        self,
        client: QdrantClient,
        collection_name: str,
        query_vector: list[float],
        n_results: int,
        score_threshold: float,
        filter: Any | None = None,
        **kwargs,
    ) -> RetrievalResults:
        retrieved = client.search(
            collection_name=collection_name,
            query_vector=query_vector,
            query_filter=filter,
            score_threshold=score_threshold,
            with_vectors=False,
            limit=n_results,
            **kwargs,
        )

        ids = [r.id for r in retrieved]
        scores = [r.score for r in retrieved]
        documents = [r.payload["document"] for r in retrieved]
        metadatas = [r.payload["metadata"] for r in retrieved]
        collections = [r.payload["collection"] for r in retrieved]

        return RetrievalResults(
            ids=ids,
            scores=scores,
            documents=documents,
            metadatas=metadatas,
            collections=collections,
        )

    async def _aglob(self, client: AsyncQdrantClient) -> list[str]:
        return [
            c.name
            for c in (await client.get_collections()).collections
            if self.collection_name in c.name
        ]

    async def _aretrieve(
        self,
        client: AsyncQdrantClient,
        collection_name: str,
        query_vector: list[float],
        n_results: int,
        score_threshold: float,
        filter: Any | None = None,
        **kwargs,
    ) -> RetrievalResults:
        retrieved = await client.search(
            collection_name=collection_name,
            query_vector=query_vector,
            query_filter=filter,
            score_threshold=score_threshold,
            with_vectors=False,
            limit=n_results,
            **kwargs,
        )

        ids = [r.id for r in retrieved]
        scores = [r.score for r in retrieved]
        documents = [r.payload["document"] for r in retrieved]
        metadatas = [r.payload["metadata"] for r in retrieved]
        collections = [r.payload["collection"] for r in retrieved]

        return RetrievalResults(
            ids=ids,
            scores=scores,
            documents=documents,
            metadatas=metadatas,
            collections=collections,
        )

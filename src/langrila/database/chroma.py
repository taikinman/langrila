from typing import Any

import chromadb
from chromadb import DEFAULT_DATABASE, DEFAULT_TENANT, Settings
from chromadb.api import AsyncClientAPI, ClientAPI
from chromadb.api.types import URI, Image, Include, OneOrMany
from chromadb.types import Where, WhereDocument

from ..core.model import LLMModel
from ..core.retrieval import RetrievalResults
from .base import (
    BaseLocalCollectionModule,
    BaseLocalRetrievalModule,
    BaseRemoteCollectionModule,
    BaseRemoteRetrievalModule,
)


class ChromaLocalCollectionModule(BaseLocalCollectionModule):
    def __init__(
        self,
        persistence_directory: str,
        collection_name: str,
        metadata: dict[str, Any] | None = None,
        embedder: LLMModel | None = None,
        logger: Any | None = None,
        settings: Settings | None = None,
        tenant: str = DEFAULT_TENANT,
        database: str = DEFAULT_DATABASE,
        batch_size: int = 100,
    ):
        super().__init__(
            persistence_directory=persistence_directory,
            collection_name=collection_name,
            embedder=embedder,
            logger=logger,
            batch_size=batch_size,
        )
        self.metadata = metadata or {"hnsw:space": "cosine"}
        self.settings = settings
        self.tenant = tenant
        self.database = database

    def _exists(self, client: ClientAPI, collection_name: str) -> bool:
        return len([c.name for c in client.list_collections() if c.name == collection_name]) > 0

    def _create_collection(self, client: ClientAPI, collection_name: str) -> None:
        client.create_collection(name=collection_name, metadata=self.metadata)

    def _delete_record(
        self,
        client: ClientAPI,
        collection_name: str,
        ids: list[str | int],
        filter: Where | None = None,
        where_document: WhereDocument | None = None,
    ) -> None:
        collection = client.get_collection(name=collection_name)

        collection.delete(ids=[str(i) for i in ids], where=filter, where_document=where_document)

    def _upsert(
        self,
        client: ClientAPI,
        collection_name: str,
        ids: list[str | int],
        documents: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict[str, str]],
        images: OneOrMany[Image] | None = None,
        uris: OneOrMany[URI] | None = None,
    ) -> None:
        collection = client.get_collection(name=collection_name)

        collection.upsert(
            ids=[str(i) for i in ids],
            embeddings=embeddings,
            documents=documents,
            metadatas=[m | {"document": doc} for m, doc in zip(metadatas, documents, strict=True)],
            images=images,
            uris=uris,
        )

        return

    def _delete_collection(self, client: ClientAPI, collection_name: str) -> None:
        client.delete_collection(name=collection_name)

    def get_client(self) -> ClientAPI:
        return chromadb.PersistentClient(
            path=self.persistence_directory.as_posix(),
            settings=self.settings,
            tenant=self.tenant,
            database=self.database,
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
            settings=self.settings,
            tenant=self.tenant,
            database=self.database,
            n_results=n_results,
            score_threshold=score_threshold,
            logger=self.logger,
        )


class ChromaRemoteCollectionModule(BaseRemoteCollectionModule):
    def __init__(
        self,
        host: str,
        collection_name: str,
        port: str = "8000",
        ssl: bool = False,
        headers: dict[str, str] | None = None,
        settings: Settings | None = None,
        metadata: dict[str, Any] | None = None,
        embedder: LLMModel | None = None,
        logger: Any | None = None,
        tenant: str = DEFAULT_TENANT,
        database: str = DEFAULT_DATABASE,
        batch_size: int = 100,
    ):
        super().__init__(
            url=host,
            port=port,
            collection_name=collection_name,
            embedder=embedder,
            logger=logger,
            batch_size=batch_size,
        )
        self.metadata = metadata or {"hnsw:space": "cosine"}
        self.tenant = tenant
        self.database = database
        self.ssl = ssl
        self.headers = headers
        self.settings = settings

    def _exists(self, client: ClientAPI, collection_name: str) -> bool:
        return len([c.name for c in client.list_collections() if c.name == collection_name]) > 0

    def _create_collection(self, client: ClientAPI, collection_name: str) -> None:
        client.create_collection(name=collection_name, metadata=self.metadata)

    def _delete_record(
        self,
        client: ClientAPI,
        collection_name: str,
        ids: list[str | int],
        filter: Where | None = None,
        where_document: WhereDocument | None = None,
    ) -> None:
        collection = client.get_collection(name=collection_name)
        collection.delete(ids=[str(i) for i in ids], where=filter, where_document=where_document)

    def _upsert(
        self,
        client: ClientAPI,
        collection_name: str,
        ids: list[str | int],
        documents: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict[str, str]],
        images: OneOrMany[Image] | None = None,
        uris: OneOrMany[URI] | None = None,
    ) -> None:
        collection = client.get_collection(name=collection_name)

        collection.upsert(
            ids=[str(i) for i in ids],
            embeddings=embeddings,
            documents=documents,
            metadatas=[m | {"document": doc} for m, doc in zip(metadatas, documents, strict=True)],
            images=images,
            uris=uris,
        )

        return

    def _delete_collection(self, client: ClientAPI, collection_name: str) -> None:
        client.delete_collection(name=collection_name)

    def get_client(self) -> ClientAPI:
        if hasattr(self, "client") and isinstance(self.client, ClientAPI):
            return self.client

        self.client = chromadb.HttpClient(
            host=self.url,
            port=self.port,
            ssl=self.ssl,
            headers=self.headers,
            settings=self.settings,
            tenant=self.tenant,
            database=self.database,
        )

        return self.client

    async def get_async_client(self) -> AsyncClientAPI:
        if hasattr(self, "client") and isinstance(self.client, AsyncClientAPI):
            return self.client

        self.client = await chromadb.AsyncHttpClient(
            host=self.url,
            port=self.port,
            ssl=self.ssl,
            headers=self.headers,
            settings=self.settings,
            tenant=self.tenant,
            database=self.database,
        )

        return self.client

    def as_retriever(
        self,
        n_results: int = 4,
        score_threshold: float = 0.5,
    ) -> "ChromaRemoteRetrievalModule":
        return ChromaRemoteRetrievalModule(
            embedder=self.embedder,
            host=self.url,
            port=self.port,
            ssl=self.ssl,
            headers=self.headers,
            settings=self.settings,
            collection_name=self.collection_name,
            n_results=n_results,
            score_threshold=score_threshold,
            logger=self.logger,
            tenant=self.tenant,
            database=self.database,
        )

    async def _create_collection_async(self, client: AsyncClientAPI, collection_name: str) -> None:
        await client.create_collection(name=collection_name, metadata=self.metadata)

    async def _exists_async(self, client: AsyncClientAPI, collection_name: str) -> bool:
        return (
            len([c.name for c in await client.list_collections() if c.name == collection_name]) > 0
        )

    async def _delete_collection_async(self, client: AsyncClientAPI, collection_name: str) -> None:
        await client.delete_collection(name=collection_name)

    async def _delete_record_async(
        self,
        client: AsyncClientAPI,
        collection_name: str,
        ids: list[str | int],
        filter: Where | None = None,
        where_document: WhereDocument | None = None,
    ) -> None:
        collection = await client.get_collection(name=collection_name)

        await collection.delete(
            ids=[str(i) for i in ids], where=filter, where_document=where_document
        )

    async def _upsert_async(
        self,
        client: AsyncClientAPI,
        collection_name: str,
        ids: list[str | int],
        documents: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict[str, str]],
        images: OneOrMany[Image] | None = None,
        uris: OneOrMany[URI] | None = None,
    ) -> None:
        collection = await client.get_collection(name=collection_name)

        await collection.upsert(
            ids=[str(i) for i in ids],
            embeddings=embeddings,
            documents=documents,
            metadatas=[m | {"document": doc} for m, doc in zip(metadatas, documents, strict=True)],
            images=images,
            uris=uris,
        )

        return


class ChromaLocalRetrievalModule(BaseLocalRetrievalModule):
    def __init__(
        self,
        persistence_directory: str,
        collection_name: str,
        embedder: LLMModel | None = None,
        n_results: int = 4,
        score_threshold: float = 0.5,
        logger: Any | None = None,
        ascending: bool = True,
        settings: Settings | None = None,
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
        self.settings = settings
        self.tenant = tenant
        self.database = database

    def get_client(self) -> ClientAPI:
        return chromadb.PersistentClient(
            path=self.persistence_directory.as_posix(),
            settings=self.settings,
            tenant=self.tenant,
            database=self.database,
        )

    def _retrieve(
        self,
        client: ClientAPI,
        collection_name: str,
        query_vector: list[float],
        n_results: int,
        score_threshold: float,
        filter: Any | None = None,
        query_images: OneOrMany[Image] | None = None,
        query_uris: OneOrMany[URI] | None = None,
        where_document: WhereDocument | None = None,
        include: Include | None = None,
    ) -> RetrievalResults:
        collection = client.get_collection(name=collection_name)

        retrieved = collection.query(
            query_embeddings=query_vector,
            where=filter,
            include=include or ["metadatas", "documents", "distances"],
            n_results=n_results,
            query_images=query_images,
            query_uris=query_uris,
            where_document=where_document,
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


class ChromaRemoteRetrievalModule(BaseRemoteRetrievalModule):
    def __init__(
        self,
        host: str,
        collection_name: str,
        port: str = "8000",
        ssl: bool = False,
        headers: dict[str, str] | None = None,
        settings: Settings | None = None,
        embedder: LLMModel | None = None,
        n_results: int = 4,
        score_threshold: float = 0.5,
        logger: Any | None = None,
        ascending: bool = True,
        tenant: str = DEFAULT_TENANT,
        database: str = DEFAULT_DATABASE,
    ):
        super().__init__(
            url=host,
            port=port,
            collection_name=collection_name,
            embedder=embedder,
            n_results=n_results,
            score_threshold=score_threshold,
            logger=logger,
            ascending=ascending,
        )
        self.ssl = ssl
        self.headers = headers
        self.settings = settings
        self.tenant = tenant
        self.database = database

    def get_client(self) -> ClientAPI:
        if hasattr(self, "client") and isinstance(self.client, ClientAPI):
            return self.client

        self.client = chromadb.HttpClient(
            host=self.url,
            port=self.port,
            ssl=self.ssl,
            headers=self.headers,
            settings=self.settings,
            tenant=self.tenant,
            database=self.database,
        )

        return self.client

    async def get_async_client(self) -> AsyncClientAPI:
        if hasattr(self, "client") and isinstance(self.client, AsyncClientAPI):
            return self.client

        self.client = await chromadb.AsyncHttpClient(
            host=self.url,
            port=self.port,
            ssl=self.ssl,
            headers=self.headers,
            settings=self.settings,
            tenant=self.tenant,
            database=self.database,
        )

        return self.client

    def _retrieve(
        self,
        client: ClientAPI,
        collection_name: str,
        query_vector: list[float],
        n_results: int,
        score_threshold: float,
        filter: Any | None = None,
        query_images: OneOrMany[Image] | None = None,
        query_uris: OneOrMany[URI] | None = None,
        where_document: WhereDocument | None = None,
        include: Include | None = None,
    ) -> RetrievalResults:
        collection = client.get_collection(name=collection_name)

        retrieved = collection.query(
            query_embeddings=query_vector,
            where=filter,
            include=include or ["metadatas", "documents", "distances"],
            n_results=n_results,
            query_images=query_images,
            query_uris=query_uris,
            where_document=where_document,
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

    async def _retrieve_async(
        self,
        client: AsyncClientAPI,
        collection_name: str,
        query_vector: list[float],
        n_results: int,
        score_threshold: float,
        filter: Any | None = None,
        query_images: OneOrMany[Image] | None = None,
        query_uris: OneOrMany[URI] | None = None,
        where_document: WhereDocument | None = None,
        include: Include | None = None,
    ) -> RetrievalResults:
        collection = await client.get_collection(name=collection_name)

        retrieved = await collection.query(
            query_embeddings=query_vector,
            where=filter,
            include=include or ["metadatas", "documents", "distances"],
            n_results=n_results,
            query_images=query_images,
            query_uris=query_uris,
            where_document=where_document,
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

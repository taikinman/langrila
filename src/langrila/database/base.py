import math
from abc import ABC, abstractmethod
from itertools import cycle
from pathlib import Path
from typing import Any, Generator, Optional

from ..base import BaseEmbeddingModule
from ..logger import DefaultLogger
from ..result import EmbeddingResults, RetrievalResult
from ..usage import Usage
from ..utils import make_batch


class _BaseCollectionModule(ABC):
    def __init__(
        self,
        collection_name: str,
        embedder: BaseEmbeddingModule = None,
        logger: Any | None = None,
    ):
        self.embedder = embedder
        self.collection_name = collection_name
        self.logger = logger or DefaultLogger()
        self.limit_collection_size: int = 10000

    @abstractmethod
    def get_client(self) -> Any:
        """
        return the client object
        """
        raise NotImplementedError

    def get_async_client(self) -> Any:
        """
        return the async client object
        """
        raise NotImplementedError

    @abstractmethod
    def glob(self, client: Any) -> list[str] | Generator[str, None, None]:
        """
        return the list of collection names
        """
        raise NotImplementedError

    @abstractmethod
    def _create_collection(self, client: Any, collection_name: str, **kwargs) -> None:
        raise NotImplementedError

    async def _acreate_collection(self, client: Any, collection_name: str, **kwargs) -> None:
        raise NotImplementedError

    @abstractmethod
    def _exists(self, client: Any, collection_name: str) -> bool:
        raise NotImplementedError

    async def _aexists(self, client: Any, collection_name: str) -> bool:
        raise NotImplementedError

    @abstractmethod
    def _upsert(
        self,
        client: Any,
        collection_name: str,
        ids: list[str | int],
        embeddings: list[list[float]],
        metadatas: list[dict[str, str]],
    ) -> None:
        raise NotImplementedError

    async def _aupsert(
        self,
        client: Any,
        collection_name: str,
        ids: list[str | int],
        embeddings: list[list[float]],
        metadatas: list[dict[str, str]],
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    def _delete(self, client: Any, collection_name: str) -> None:
        raise NotImplementedError

    def as_retriever(
        self,
        n_results: int,
        score_threshold: float,
    ) -> "BaseLocalRetrievalModule|BaseRemoteRetrievalModule":
        raise NotImplementedError

    def create_collection(self, suffix: str = "", **kwargs) -> None:
        client = self.get_client()
        colelction_name = self.collection_name + suffix
        self._create_collection(client=client, collection_name=colelction_name, **kwargs)

    async def acreate_collection(self, suffix: str = "", **kwargs) -> None:
        client = self.get_client()
        colelction_name = self.collection_name + suffix
        await self._acreate_collection(client=client, collection_name=colelction_name, **kwargs)

    def exists(self) -> bool:
        client = self.get_client()
        return self._exists(client=client, collection_name=self.collection_name)

    def upsert(
        self,
        ids: list[str | int],
        embeddings: list[list[float]],
        metadatas: list[dict[str, str]],
    ) -> None:
        client = self.get_client()
        self._upsert(
            client=client,
            collection_name=self.collection_name,
            ids=ids,
            embeddings=embeddings,
            metadatas=metadatas,
        )

    async def aupsert(
        self,
        ids: list[str | int],
        embeddings: list[list[float]],
        metadatas: list[dict[str, str]],
    ) -> None:
        client = self.get_client()
        await self._aupsert(
            client=client,
            collection_name=self.collection_name,
            ids=ids,
            embeddings=embeddings,
            metadatas=metadatas,
        )

    def delete(self) -> None:
        client = self.get_client()

        for collection_name in self.glob(client=client):
            if self._exists(client=client, collection_name=collection_name):
                self._delete(client=client, collection_name=collection_name)

    def run(
        self,
        documents: list[str],
        metadatas: Optional[list[dict[str, str]]] = None,
        **kwargs,
    ) -> None:
        client = self.get_client()

        if metadatas is not None and len(documents) != len(metadatas):
            raise ValueError(
                "The length of documents and metadatas must be the same. "
                f"Got {len(documents)} documents and {len(metadatas)} metadatas."
            )

        if metadatas is None:
            metadatas = [{} for _ in range(len(documents))]

        length = len(documents)
        ids = []
        _ = range(self.limit_collection_size)
        for i in cycle(_):
            ids.append(i)
            if len(ids) >= length:
                break

        # batchfy due to memory usage
        total_idx: int = 0
        collection_index: int = 0
        batch_size: int = 100

        documents_batch = make_batch(documents, batch_size=batch_size)
        metadatas_batch = make_batch(metadatas, batch_size=batch_size)
        ids_batch = make_batch(ids, batch_size=batch_size)

        n_batches: int = math.ceil(len(documents) / batch_size)

        for i, (doc_batch, metadata_batch, id_batch) in enumerate(
            zip(documents_batch, metadatas_batch, ids_batch, strict=True)
        ):
            embedding_batch: EmbeddingResults = self.embedder.run(doc_batch)

            if total_idx == 0:
                suffix: str = f"_{collection_index}"
                collection_name: str = self.collection_name + suffix
                if not self._exists(client=client, collection_name=collection_name):
                    self._create_collection(
                        client=client, collection_name=collection_name, **kwargs
                    )
                    self.logger.info(
                        f"Create {n_batches} batches for collection {collection_name}."
                    )

            self.logger.info(f"[batch {i+1}/{n_batches}] Upsert points...")

            metadata_batch = [
                {"metadata": metadata, "document": document, "collection": collection_name}
                for metadata, document in zip(metadata_batch, doc_batch, strict=True)
            ]

            self._upsert(
                client=client,
                collection_name=collection_name,
                ids=id_batch,
                embeddings=embedding_batch.embeddings,
                metadatas=metadata_batch,
            )

            total_idx += len(doc_batch)

            # collection size is limited to 10000 avoiding memory error
            if total_idx >= self.limit_collection_size:
                collection_index += 1
                total_idx = 0

    async def arun(
        self,
        documents: list[str],
        metadatas: Optional[list[dict[str, str]]] = None,
        **kwargs,
    ) -> None:
        client = self.get_async_client()

        if metadatas is not None and len(documents) != len(metadatas):
            raise ValueError(
                "The length of documents and metadatas must be the same. "
                f"Got {len(documents)} documents and {len(metadatas)} metadatas."
            )

        if metadatas is None:
            metadatas = [{} for _ in range(len(documents))]

        length = len(documents)
        ids = []
        _ = range(self.limit_collection_size)
        for i in cycle(_):
            ids.append(i)
            if len(ids) >= length:
                break

        # batchfy due to memory usage
        total_idx: int = 0
        collection_index: int = 0
        batch_size: int = 100

        documents_batch = make_batch(documents, batch_size=batch_size)
        metadatas_batch = make_batch(metadatas, batch_size=batch_size)
        ids_batch = make_batch(ids, batch_size=batch_size)

        n_batches: int = math.ceil(len(documents) / batch_size)

        for i, (doc_batch, metadata_batch, id_batch) in enumerate(
            zip(documents_batch, metadatas_batch, ids_batch, strict=True)
        ):
            embedding_batch: EmbeddingResults = await self.embedder.arun(doc_batch)

            if total_idx == 0:
                suffix: str = f"_{collection_index}"
                collection_name: str = self.collection_name + suffix
                if not await self._aexists(client=client, collection_name=collection_name):
                    await self._acreate_collection(
                        client=client, collection_name=collection_name, **kwargs
                    )
                    self.logger.info(
                        f"Create {n_batches} batches for collection {collection_name}."
                    )

            self.logger.info(f"[batch {i+1}/{n_batches}] Upsert points...")

            metadata_batch = [
                {"metadata": metadata, "document": document, "collection": collection_name}
                for metadata, document in zip(metadata_batch, doc_batch, strict=True)
            ]

            await self._aupsert(
                client=client,
                collection_name=collection_name,
                ids=id_batch,
                embeddings=embedding_batch.embeddings,
                metadatas=metadata_batch,
            )

            total_idx += len(doc_batch)

            # collection size is limited to 10000 avoiding memory error
            if total_idx >= self.limit_collection_size:
                collection_index += 1
                total_idx = 0


class _BaseRetrievalModule(ABC):
    def __init__(
        self,
        collection_name: str,
        embedder: BaseEmbeddingModule = None,
        logger: Any | None = None,
        n_results: int = 4,
        score_threshold: float = 0.8,
        reverse: bool = True,
    ):
        self.embedder = embedder
        self.collection_name = collection_name
        self.logger = logger or DefaultLogger()
        self.n_results = n_results
        self.score_threshold = score_threshold
        self.reverse = reverse

    @abstractmethod
    def get_client(self) -> Any:
        """
        return the client object
        """
        raise NotImplementedError

    def get_async_client(self) -> Any:
        """
        return the async client object
        """
        raise NotImplementedError

    @abstractmethod
    def glob(self, client: Any) -> list[str] | Generator[str, None, None]:
        """
        return the list of collection names
        """
        raise NotImplementedError

    @abstractmethod
    def _retrieve(
        self,
        client: Any,
        collection_name: str,
        query_vector: list[float],
        n_results: int,
        score_threshold: float,
        filter: Any | None = None,
        **kwargs,
    ) -> RetrievalResult:
        raise NotImplementedError

    async def _aretrieve(
        self,
        client: Any,
        collection_name: str,
        query_vector: list[float],
        n_results: int,
        score_threshold: float,
        filter: Any | None = None,
        **kwargs,
    ) -> RetrievalResult:
        raise NotImplementedError

    def run(self, query: str, filter: Any | None = None, **kwargs) -> RetrievalResult:
        client = self.get_client()

        embed: EmbeddingResults = self.embedder.run(query)
        usage: Usage = embed.usage

        collection_names: list[str] | Generator[str, None, None]
        collection_names = self.glob(client=client)

        ids = []
        scores = []
        documents = []
        metadatas = []
        collections = []

        for collection_name in collection_names:
            self.logger.info(f"Retrieve from {collection_name}...")
            retrieved: RetrievalResult = self._retrieve(
                client=client,
                collection_name=collection_name,
                query_vector=embed.embeddings[0],
                filter=filter,
                n_results=self.n_results,
                score_threshold=self.score_threshold,
                **kwargs,
            )

            ids.extend(retrieved.ids)
            scores.extend(retrieved.scores)
            documents.extend(retrieved.documents)
            metadatas.extend(retrieved.metadatas)
            collections.extend(retrieved.collections)

        self.logger.info("Sort results...")
        sort_indices = sorted(range(len(scores)), key=scores.__getitem__, reverse=self.reverse)

        # top-k results
        results = RetrievalResult(
            ids=[ids[i] for i in sort_indices][: self.n_results],
            scores=[scores[i] for i in sort_indices][: self.n_results],
            documents=[documents[i] for i in sort_indices][: self.n_results],
            metadatas=[metadatas[i] for i in sort_indices][: self.n_results],
            collections=[collections[i] for i in sort_indices][: self.n_results],
            usage=usage,
        )

        return results

    async def arun(self, query: str, filter: Any | None = None, **kwargs) -> RetrievalResult:
        client = self.get_async_client()

        embed: EmbeddingResults = await self.embedder.arun(query)
        usage: Usage = embed.usage

        collection_names: list[str] | Generator[str, None, None]
        collection_names = self.glob(client=client)

        ids = []
        scores = []
        documents = []
        metadatas = []
        collections = []

        # sweep all collections one by one to avoid memory error
        for collection_name in collection_names:
            self.logger.info(f"Retrieve from {collection_name}...")
            retrieved: RetrievalResult = await self._aretrieve(
                client=client,
                collection_name=collection_name,
                query_vector=embed.embeddings[0],
                filter=filter,
                n_results=self.n_results,
                score_threshold=self.score_threshold,
                **kwargs,
            )

            ids.extend(retrieved.ids)
            scores.extend(retrieved.scores)
            documents.extend(retrieved.documents)
            metadatas.extend(retrieved.metadatas)
            collections.extend(retrieved.collections)

        self.logger.info("Sort results...")
        sort_indices = sorted(range(len(scores)), key=scores.__getitem__, reverse=True)

        # top-k results
        results = RetrievalResult(
            ids=[ids[i] for i in sort_indices][: self.n_results],
            scores=[scores[i] for i in sort_indices][: self.n_results],
            documents=[documents[i] for i in sort_indices][: self.n_results],
            metadatas=[metadatas[i] for i in sort_indices][: self.n_results],
            collections=[collections[i] for i in sort_indices][: self.n_results],
            usage=usage,
        )

        return results


class BaseLocalCollectionModule(_BaseCollectionModule):
    def __init__(
        self,
        persistence_directory: str,
        collection_name: str,
        embedder: BaseEmbeddingModule = None,
        logger: Any | None = None,
    ):
        self.persistence_directory = Path(persistence_directory)
        super().__init__(collection_name=collection_name, embedder=embedder, logger=logger)


class BaseRemoteCollectionModule(_BaseCollectionModule):
    def __init__(
        self,
        url: str,
        collection_name: str,
        port: str = "6333",
        embedder: BaseEmbeddingModule = None,
        logger: Any | None = None,
    ):
        self.url = url
        self.port = port
        super().__init__(collection_name=collection_name, embedder=embedder, logger=logger)


class BaseLocalRetrievalModule(_BaseRetrievalModule):
    def __init__(
        self,
        embedder: BaseEmbeddingModule,
        persistence_directory: str,
        collection_name: str,
        n_results: int = 4,
        score_threshold: float = 0.8,
        logger: Any | None = None,
    ):
        assert isinstance(
            embedder, BaseEmbeddingModule
        ), "embedder must be the instance of the class inheriting BaseEmbeddingModule."

        super().__init__(
            collection_name=collection_name,
            embedder=embedder,
            n_results=n_results,
            score_threshold=score_threshold,
            logger=logger,
        )
        self.persistence_directory = Path(persistence_directory)


class BaseRemoteRetrievalModule(_BaseRetrievalModule):
    def __init__(
        self,
        embedder: BaseEmbeddingModule,
        url: str,
        collection_name: str,
        port: str = "6333",
        n_results: int = 4,
        score_threshold: float = 0.8,
        logger: Any | None = None,
    ):
        assert isinstance(
            embedder, BaseEmbeddingModule
        ), "embedder must be the instance of the class inheriting BaseEmbeddingModule."

        super().__init__(
            collection_name=collection_name,
            embedder=embedder,
            n_results=n_results,
            score_threshold=score_threshold,
            logger=logger,
        )
        self.url = url
        self.port = port

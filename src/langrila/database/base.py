import math
import time
from abc import ABC, abstractmethod
from itertools import cycle
from pathlib import Path
from typing import Any, Generator, Optional

from tqdm import tqdm

from ..base import BaseEmbeddingModule
from ..logger import DefaultLogger
from ..result import EmbeddingResults, RetrievalResults
from ..usage import Usage
from ..utils import make_batch


class AbstractLocalCollectionModule(ABC):
    """
    Base class for collection module.
    Collection limits the number of its records (<= 10000 records) to keep memory error away.
    If you can include records over limitation, collection will be automatically divided into multiple collection.
    """

    @abstractmethod
    def get_client(self) -> Any:
        """
        return the client object
        """
        raise NotImplementedError

    @abstractmethod
    def _glob(self, client: Any) -> list[str] | Generator[str, None, None]:
        """
        return the list of collection names
        """
        raise NotImplementedError

    @abstractmethod
    def _create_collection(self, client: Any, collection_name: str, **kwargs) -> None:
        raise NotImplementedError

    @abstractmethod
    def _exists(self, client: Any, collection_name: str) -> bool:
        raise NotImplementedError

    @abstractmethod
    def _upsert(
        self,
        client: Any,
        collection_name: str,
        ids: list[str | int],
        documents: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict[str, str]],
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    def _delete(self, client: Any, collection_name: str) -> None:
        raise NotImplementedError

    @abstractmethod
    def as_retriever(
        self,
        n_results: int,
        score_threshold: float,
    ):
        raise NotImplementedError


class AbstractRemoteCollectionModule(ABC):
    """
    Base class for collection module.
    Collection limits the number of its records (<= 10000 records) to keep memory error away.
    If you can include records over limitation, collection will be automatically divided into multiple collection.
    """

    @abstractmethod
    def get_client(self) -> Any:
        """
        return the client object
        """
        raise NotImplementedError

    @abstractmethod
    def get_async_client(self) -> Any:
        """
        return the async client object
        """
        raise NotImplementedError

    @abstractmethod
    def _glob(self, client: Any) -> list[str] | Generator[str, None, None]:
        """
        return the list of collection names
        """
        raise NotImplementedError

    @abstractmethod
    async def _aglob(
        self,
        client: Any,
    ) -> list[str] | Generator[str, None, None]:
        """
        return the list of collection names
        """
        raise NotImplementedError

    @abstractmethod
    def _create_collection(self, client: Any, collection_name: str, **kwargs) -> None:
        raise NotImplementedError

    @abstractmethod
    async def _acreate_collection(self, client: Any, collection_name: str, **kwargs) -> None:
        raise NotImplementedError

    @abstractmethod
    def _exists(self, client: Any, collection_name: str) -> bool:
        raise NotImplementedError

    @abstractmethod
    async def _aexists(self, client: Any, collection_name: str) -> bool:
        raise NotImplementedError

    @abstractmethod
    def _upsert(
        self,
        client: Any,
        collection_name: str,
        ids: list[str | int],
        documents: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict[str, str]],
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    async def _aupsert(
        self,
        client: Any,
        collection_name: str,
        ids: list[str | int],
        documents: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict[str, str]],
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    async def _adelete(self, client: Any, collection_name: str) -> None:
        raise NotImplementedError

    @abstractmethod
    def _delete(self, client: Any, collection_name: str) -> None:
        raise NotImplementedError

    @abstractmethod
    def as_retriever(
        self,
        n_results: int,
        score_threshold: float,
    ):
        raise NotImplementedError


class BaseLocalCollectionModule(AbstractLocalCollectionModule):
    def __init__(
        self,
        persistence_directory: Path | str,
        collection_name: str,
        embedder: BaseEmbeddingModule = None,
        logger: Any | None = None,
        limit_collection_size: int = 10000,
    ):
        self.persistence_directory = Path(persistence_directory)
        self.embedder = embedder
        self.collection_name = collection_name
        self.logger = logger or DefaultLogger()
        assert limit_collection_size % 100 == 0, "limit_collection_size must be multiple of 100."
        self.limit_collection_size: int = limit_collection_size

    def create_collection(self, suffix: str = "", **kwargs) -> None:
        client = self.get_client()
        colelction_name = self.collection_name + suffix
        self._create_collection(client=client, collection_name=colelction_name, **kwargs)

    def exists(self) -> bool:
        client = self.get_client()
        return self._exists(client=client, collection_name=self.collection_name)

    def upsert(
        self,
        ids: list[str | int],
        documents: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict[str, str]],
    ) -> None:
        client = self.get_client()
        self._upsert(
            client=client,
            collection_name=self.collection_name,
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
        )

    def delete(self) -> None:
        client = self.get_client()

        for collection_name in self._glob(client=client):
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

        for i, (doc_batch, metadata_batch, id_batch) in tqdm(
            enumerate(zip(documents_batch, metadatas_batch, ids_batch, strict=True)),
            total=n_batches,
        ):
            embedding_batch: EmbeddingResults = self.embedder.run(doc_batch)

            if total_idx == 0:
                suffix: str = f"_{collection_index}"
                collection_name: str = self.collection_name + suffix
                if not self._exists(client=client, collection_name=collection_name):
                    self._create_collection(
                        client=client, collection_name=collection_name, **kwargs
                    )
                    self.logger.info(f"Create collection {collection_name}.")

            # self.logger.info(f"[batch {i+1}/{n_batches}] Upsert points...")

            metadata_batch = [
                {"metadata": metadata, "collection": collection_name} for metadata in metadata_batch
            ]

            n_retries = 0
            while n_retries < 3:
                try:
                    self._upsert(
                        client=client,
                        collection_name=collection_name,
                        ids=id_batch,
                        documents=doc_batch,
                        embeddings=embedding_batch.embeddings,
                        metadatas=metadata_batch,
                    )
                    break
                except Exception as e:
                    self.logger.error(f"Error: {e}")
                    n_retries += 1
                    time.sleep(10)

                    if n_retries == 3:
                        raise e

            total_idx += len(doc_batch)

            # collection size is limited to 10000 avoiding memory error
            if total_idx >= self.limit_collection_size:
                collection_index += 1
                total_idx = 0


class BaseRemoteCollectionModule(BaseLocalCollectionModule, AbstractRemoteCollectionModule):
    def __init__(
        self,
        url: str,
        collection_name: str,
        port: str = "6333",
        embedder: BaseEmbeddingModule = None,
        logger: Any | None = None,
        limit_collection_size: int = 10000,
    ):
        self.url = url
        self.port = port
        self.embedder = embedder
        self.collection_name = collection_name
        self.logger = logger or DefaultLogger()
        self.limit_collection_size: int = limit_collection_size

    async def acreate_collection(self, suffix: str = "", **kwargs) -> None:
        client = self.get_async_client()
        colelction_name = self.collection_name + suffix
        await self._acreate_collection(client=client, collection_name=colelction_name, **kwargs)

    async def aexists(self) -> bool:
        client = self.get_async_client()
        return await self._aexists(client=client, collection_name=self.collection_name)

    async def aupsert(
        self,
        ids: list[str | int],
        documents: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict[str, str]],
    ) -> None:
        client = self.get_async_client()
        await self._aupsert(
            client=client,
            collection_name=self.collection_name,
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
        )

    async def adelete(self) -> None:
        client = self.get_async_client()
        for collection_name in await self._aglob(client=client):
            if await self._aexists(client=client, collection_name=collection_name):
                await self._adelete(client=client, collection_name=collection_name)

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

        for i, (doc_batch, metadata_batch, id_batch) in tqdm(
            enumerate(zip(documents_batch, metadatas_batch, ids_batch, strict=True)),
            total=n_batches,
        ):
            embedding_batch: EmbeddingResults = await self.embedder.arun(doc_batch)

            if total_idx == 0:
                suffix: str = f"_{collection_index}"
                collection_name: str = self.collection_name + suffix
                if not await self._aexists(client=client, collection_name=collection_name):
                    await self._acreate_collection(
                        client=client, collection_name=collection_name, **kwargs
                    )
                    self.logger.info(f"Create collection {collection_name}.")

            # self.logger.info(f"[batch {i+1}/{n_batches}] Upsert points...")

            metadata_batch = [
                {"metadata": metadata, "collection": collection_name} for metadata in metadata_batch
            ]

            n_retries = 0
            while n_retries < 3:
                try:
                    await self._aupsert(
                        client=client,
                        collection_name=collection_name,
                        ids=id_batch,
                        documents=doc_batch,
                        embeddings=embedding_batch.embeddings,
                        metadatas=metadata_batch,
                    )
                    break
                except Exception as e:
                    self.logger.error(f"Error: {e}")
                    n_retries += 1
                    time.sleep(10)

                    if n_retries == 3:
                        raise e

            total_idx += len(doc_batch)

            # collection size is limited to 10000 avoiding memory error
            if total_idx >= self.limit_collection_size:
                collection_index += 1
                total_idx = 0


class AbstractLocalRetrievalModule(ABC):
    @abstractmethod
    def get_client(self) -> Any:
        """
        return the async client object
        """
        raise NotImplementedError

    @abstractmethod
    def _glob(self, client: Any) -> list[str] | Generator[str, None, None]:
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
    ) -> RetrievalResults:
        raise NotImplementedError


class AbstractRemoteRetrievalModule(AbstractLocalRetrievalModule):
    @abstractmethod
    def get_client(self) -> Any:
        raise NotImplementedError

    @abstractmethod
    def get_async_client(self) -> Any:
        """
        return the async client object
        """
        raise NotImplementedError

    @abstractmethod
    def _glob(self, client: Any) -> list[str] | Generator[str, None, None]:
        raise NotImplementedError

    @abstractmethod
    async def _aglob(self, client: Any) -> list[str] | Generator[str, None, None]:
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
    ) -> RetrievalResults:
        raise NotImplementedError

    @abstractmethod
    async def _aretrieve(
        self,
        client: Any,
        collection_name: str,
        query_vector: list[float],
        n_results: int,
        score_threshold: float,
        filter: Any | None = None,
        **kwargs,
    ) -> RetrievalResults:
        raise NotImplementedError


class BaseLocalRetrievalModule(AbstractLocalRetrievalModule):
    def __init__(
        self,
        persistence_directory: Path | str,
        collection_name: str,
        embedder: BaseEmbeddingModule = None,
        n_results: int = 4,
        score_threshold: float = 0.8,
        logger: Any | None = None,
        ascending: bool = False,
    ):
        self.persistence_directory = Path(persistence_directory)
        self.embedder = embedder
        self.collection_name = collection_name
        self.n_results = n_results
        self.score_threshold = score_threshold
        self.logger = logger or DefaultLogger()
        self.ascending = ascending

    def run(self, query: str, filter: Any | None = None, **kwargs) -> RetrievalResults:
        client = self.get_client()

        embed: EmbeddingResults = self.embedder.run(query)
        usage: Usage = embed.usage

        collection_names: list[str] | Generator[str, None, None]
        collection_names = self._glob(client=client)

        ids = []
        scores = []
        documents = []
        metadatas = []
        collections = []

        for collection_name in collection_names:
            self.logger.info(f"Retrieve from collection {collection_name}...")
            retrieved: RetrievalResults = self._retrieve(
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
        sort_indices = sorted(
            range(len(scores)), key=scores.__getitem__, reverse=not self.ascending
        )

        # top-k results
        results = RetrievalResults(
            ids=[ids[i] for i in sort_indices][: self.n_results],
            scores=[scores[i] for i in sort_indices][: self.n_results],
            documents=[documents[i] for i in sort_indices][: self.n_results],
            metadatas=[metadatas[i] for i in sort_indices][: self.n_results],
            collections=[collections[i] for i in sort_indices][: self.n_results],
            usage=usage,
        )

        return results


class BaseRemoteRetrievalModule(BaseLocalRetrievalModule, AbstractRemoteRetrievalModule):
    def __init__(
        self,
        url: str,
        collection_name: str,
        port: str = "6333",
        embedder: BaseEmbeddingModule = None,
        n_results: int = 4,
        score_threshold: float = 0.8,
        logger: Any | None = None,
        ascending: bool = False,
    ):
        self.url = url
        self.port = port
        self.embedder = embedder
        self.n_results = n_results
        self.score_threshold = score_threshold
        self.collection_name = collection_name
        self.ascending = ascending
        self.logger = logger or DefaultLogger()

    async def arun(self, query: str, filter: Any | None = None, **kwargs) -> RetrievalResults:
        client = self.get_async_client()

        embed: EmbeddingResults = await self.embedder.arun(query)
        usage: Usage = embed.usage

        collection_names: list[str] | Generator[str, None, None]
        collection_names = await self._aglob(client=client)

        ids = []
        scores = []
        documents = []
        metadatas = []
        collections = []

        # sweep all collections one by one to avoid memory error
        for collection_name in collection_names:
            self.logger.info(f"Retrieve from collection {collection_name}...")
            retrieved: RetrievalResults = await self._aretrieve(
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
        sort_indices = sorted(
            range(len(scores)), key=scores.__getitem__, reverse=not self.ascending
        )

        # top-k results
        results = RetrievalResults(
            ids=[ids[i] for i in sort_indices][: self.n_results],
            scores=[scores[i] for i in sort_indices][: self.n_results],
            documents=[documents[i] for i in sort_indices][: self.n_results],
            metadatas=[metadatas[i] for i in sort_indices][: self.n_results],
            collections=[collections[i] for i in sort_indices][: self.n_results],
            usage=usage,
        )

        return results

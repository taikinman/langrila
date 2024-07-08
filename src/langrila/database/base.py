import asyncio
import math
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Optional

from tqdm import tqdm

from langrila import Usage

from ..base import BaseEmbeddingModule
from ..logger import DefaultLogger
from ..result import EmbeddingResults, RetrievalResults
from ..utils import make_batch


class AbstractLocalCollectionModule(ABC):
    """
    Base class for collection module.
    """

    @abstractmethod
    def get_client(self) -> Any:
        """
        return the client object
        """
        raise NotImplementedError

    @abstractmethod
    def _create_collection(self, client: Any, collection_name: str) -> None:
        """
        create a collection
        """
        raise NotImplementedError

    @abstractmethod
    def _exists(self, client: Any, collection_name: str) -> bool:
        """
        check if the collection exists
        """
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
        **kwargs,
    ) -> None:
        """
        upsert embeddings, documents and metadatas.
        """
        raise NotImplementedError

    @abstractmethod
    def _delete_collection(self, client: Any, collection_name: str) -> None:
        """
        delete the collection
        """
        raise NotImplementedError

    @abstractmethod
    def _delete_record(self, client: Any, collection_name: str, **kwargs) -> None:
        """
        delete the record from the collection. kwargs depends on the clieint.
        """
        raise NotImplementedError

    @abstractmethod
    def as_retriever(
        self,
        n_results: int,
        score_threshold: float,
    ) -> "BaseLocalRetrievalModule":
        """
        return the retrieval module
        """
        raise NotImplementedError


class AbstractRemoteCollectionModule(ABC):
    """
    Base class for collection module.
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
    def _create_collection(self, client: Any, collection_name: str) -> None:
        """
        create a collection
        """
        raise NotImplementedError

    @abstractmethod
    async def _acreate_collection(self, client: Any, collection_name: str) -> None:
        """
        create a collection
        """
        raise NotImplementedError

    @abstractmethod
    def _exists(self, client: Any, collection_name: str) -> bool:
        """
        check if the collection exists
        """
        raise NotImplementedError

    @abstractmethod
    async def _aexists(self, client: Any, collection_name: str) -> bool:
        """
        check if the collection exists
        """
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
        **kwargs,
    ) -> None:
        """
        upsert embeddings, documents and metadatas.
        """
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
        **kwargs,
    ) -> None:
        """
        upsert embeddings, documents and metadatas.
        """
        raise NotImplementedError

    @abstractmethod
    async def _adelete_collection(self, client: Any, collection_name: str) -> None:
        """
        delete the collection
        """
        raise NotImplementedError

    @abstractmethod
    def _delete_collection(self, client: Any, collection_name: str) -> None:
        """
        delete the collection
        """
        raise NotImplementedError

    @abstractmethod
    async def _adelete_record(self, client: Any, collection_name: str, **kwargs) -> None:
        """
        delete the record from the collection. kwargs depends on the clieint.
        """
        raise NotImplementedError

    @abstractmethod
    def _delete_record(self, client: Any, collection_name: str, **kwargs) -> None:
        """
        delete the record from the collection. kwargs depends on the clieint.
        """
        raise NotImplementedError

    @abstractmethod
    def as_retriever(
        self,
        n_results: int,
        score_threshold: float,
    ) -> "BaseRemoteRetrievalModule":
        """
        return the retrieval module
        """
        raise NotImplementedError


class BaseLocalCollectionModule(AbstractLocalCollectionModule):
    def __init__(
        self,
        persistence_directory: Path | str,
        collection_name: str,
        embedder: BaseEmbeddingModule | None = None,
        logger: Any | None = None,
    ):
        self.persistence_directory = Path(persistence_directory)
        self.embedder = embedder
        self.collection_name = collection_name
        self.logger = logger or DefaultLogger()

    def create_collection(self) -> None:
        client = self.get_client()
        self._create_collection(client=client, collection_name=self.collection_name)

    def exists(self) -> bool:
        client = self.get_client()
        return self._exists(client=client, collection_name=self.collection_name)

    def upsert(
        self,
        ids: list[str | int],
        documents: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict[str, str]],
        **kwargs,
    ) -> None:
        self._verify_metadata(metadatas)
        client = self.get_client()
        self._upsert(
            client=client,
            collection_name=self.collection_name,
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
            **kwargs,
        )

    def delete_collection(self) -> None:
        client = self.get_client()

        if self._exists(client=client, collection_name=self.collection_name):
            self._delete_collection(client=client, collection_name=self.collection_name)

    def delete_record(self, **kwargs) -> None:
        client = self.get_client()
        self._delete_record(client=client, collection_name=self.collection_name, **kwargs)

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
        else:
            # check if the key 'collection' or 'document' is included in metadatas
            self._verify_metadata(metadatas)

        ids = [i for i in range(len(documents))]

        # batchfy due to memory usage
        batch_size: int = 100

        documents_batch = make_batch(documents, batch_size=batch_size)
        metadatas_batch = make_batch(metadatas, batch_size=batch_size)
        ids_batch = make_batch(ids, batch_size=batch_size)

        n_batches: int = math.ceil(len(documents) / batch_size)

        if not self._exists(client=client, collection_name=self.collection_name):
            self._create_collection(client=client, collection_name=self.collection_name)
            self.logger.info(f"Create collection {self.collection_name}.")

        for doc_batch, metadata_batch, id_batch in tqdm(
            zip(documents_batch, metadatas_batch, ids_batch, strict=True),
            total=n_batches,
        ):
            embedding_batch: EmbeddingResults = self.embedder.run(doc_batch)

            n_retries = 0
            while n_retries < 3:
                try:
                    self._upsert(
                        client=client,
                        collection_name=self.collection_name,
                        ids=id_batch,
                        documents=doc_batch,
                        embeddings=embedding_batch.embeddings,
                        metadatas=metadata_batch,
                        **kwargs,
                    )
                    break
                except Exception as e:
                    self.logger.error(f"Error: {e}")
                    n_retries += 1
                    time.sleep(10)

                    if n_retries == 3:
                        raise e

        if hasattr(self, "_save_on_last"):
            self._save_on_last(client=client)

    def _verify_metadata(self, metadatas: list[dict[str, str]]) -> None:
        """
        check if the key 'document' is included in metadatas
        """
        metadata_keys = [metadata for metadata in metadatas if "document" in metadata]
        if len(metadata_keys) > 0:
            raise ValueError("The key 'document' is reserved. Use another key.")


class BaseRemoteCollectionModule(BaseLocalCollectionModule, AbstractRemoteCollectionModule):
    def __init__(
        self,
        url: str,
        collection_name: str,
        port: str,
        embedder: BaseEmbeddingModule = None,
        logger: Any | None = None,
    ):
        self.url = url
        self.port = port
        self.embedder = embedder
        self.collection_name = collection_name
        self.logger = logger or DefaultLogger()

    async def acreate_collection(self) -> None:
        client = self.get_async_client()
        await self._acreate_collection(client=client, collection_name=self.collection_name)

    async def aexists(self) -> bool:
        client = self.get_async_client()
        return await self._aexists(client=client, collection_name=self.collection_name)

    async def aupsert(
        self,
        ids: list[str | int],
        documents: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict[str, str]],
        **kwargs,
    ) -> None:
        self._verify_metadata(metadatas)
        client = self.get_async_client()
        await self._aupsert(
            client=client,
            collection_name=self.collection_name,
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
            **kwargs,
        )

    async def adelete_collection(self) -> None:
        client = self.get_async_client()
        if await self._aexists(client=client, collection_name=self.collection_name):
            await self._adelete_collection(client=client, collection_name=self.collection_name)

    async def adelete_record(self, **kwargs) -> None:
        client = self.get_async_client()
        await self._adelete_record(client=client, collection_name=self.collection_name, **kwargs)

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
        else:
            # check if the key 'collection' or 'document' is included in metadatas
            self._verify_metadata(metadatas)

        ids = [i for i in range(len(documents))]

        # batchfy due to memory usage
        batch_size: int = 100

        documents_batch = make_batch(documents, batch_size=batch_size)
        metadatas_batch = make_batch(metadatas, batch_size=batch_size)
        ids_batch = make_batch(ids, batch_size=batch_size)

        n_batches: int = math.ceil(len(documents) / batch_size)

        if not await self._aexists(client=client, collection_name=self.collection_name):
            await self._acreate_collection(client=client, collection_name=self.collection_name)
            self.logger.info(f"Create collection {self.collection_name}.")

        for doc_batch, metadata_batch, id_batch in tqdm(
            zip(documents_batch, metadatas_batch, ids_batch, strict=True),
            total=n_batches,
        ):
            embedding_batch: EmbeddingResults = await self.embedder.arun(doc_batch)

            n_retries = 0
            while n_retries < 3:
                try:
                    await self._aupsert(
                        client=client,
                        collection_name=self.collection_name,
                        ids=id_batch,
                        documents=doc_batch,
                        embeddings=embedding_batch.embeddings,
                        metadatas=metadata_batch,
                        **kwargs,
                    )
                    break
                except Exception as e:
                    self.logger.error(f"Error: {e}")
                    n_retries += 1
                    await asyncio.sleep(10)

                    if n_retries == 3:
                        raise e

        if hasattr(self, "_save_on_last"):
            self._save_on_last(client=client)


class AbstractLocalRetrievalModule(ABC):
    @abstractmethod
    def get_client(self) -> Any:
        """
        return the async client object
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
        score_threshold: float = 0.5,
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

        self.logger.info(f"Retrieve from collection {self.collection_name}...")
        retrieved: RetrievalResults = self._retrieve(
            client=client,
            collection_name=self.collection_name,
            query_vector=embed.embeddings[0],
            filter=filter,
            n_results=self.n_results,
            score_threshold=self.score_threshold,
            **kwargs,
        )

        retrieved.usage = embed.usage  # override

        return retrieved


class BaseRemoteRetrievalModule(BaseLocalRetrievalModule, AbstractRemoteRetrievalModule):
    def __init__(
        self,
        url: str,
        collection_name: str,
        port: str,
        embedder: BaseEmbeddingModule = None,
        n_results: int = 4,
        score_threshold: float = 0.5,
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

        self.logger.info(f"Retrieve from collection {self.collection_name}...")
        retrieved: RetrievalResults = await self._aretrieve(
            client=client,
            collection_name=self.collection_name,
            query_vector=embed.embeddings[0],
            filter=filter,
            n_results=self.n_results,
            score_threshold=self.score_threshold,
            **kwargs,
        )

        retrieved.usage = embed.usage  # override

        return retrieved

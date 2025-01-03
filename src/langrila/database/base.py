import asyncio
import inspect
import math
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Optional

from tqdm import tqdm

from ..core.embedding import EmbeddingResults
from ..core.logger import DEFAULT_LOGGER as default_logger
from ..core.model import LLMModel
from ..core.retrieval import RetrievalResults
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
        **kwargs: Any,
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
    def _delete_record(self, client: Any, collection_name: str, **kwargs: Any) -> None:
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
    async def _create_collection_async(self, client: Any, collection_name: str) -> None:
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
    async def _exists_async(self, client: Any, collection_name: str) -> bool:
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
        **kwargs: Any,
    ) -> None:
        """
        upsert embeddings, documents and metadatas.
        """
        raise NotImplementedError

    @abstractmethod
    async def _upsert_async(
        self,
        client: Any,
        collection_name: str,
        ids: list[str | int],
        documents: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict[str, str]],
        **kwargs: Any,
    ) -> None:
        """
        upsert embeddings, documents and metadatas.
        """
        raise NotImplementedError

    @abstractmethod
    async def _delete_collection_async(self, client: Any, collection_name: str) -> None:
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
    async def _delete_record_async(self, client: Any, collection_name: str, **kwargs: Any) -> None:
        """
        delete the record from the collection. kwargs depends on the clieint.
        """
        raise NotImplementedError

    @abstractmethod
    def _delete_record(self, client: Any, collection_name: str, **kwargs: Any) -> None:
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
        embedder: LLMModel | None = None,  # type: ignore
        logger: Any | None = None,
        batch_size: int = 100,
    ):
        self.persistence_directory = Path(persistence_directory)
        self.embedder = embedder
        self.collection_name = collection_name
        self.logger = logger or default_logger
        self.batch_size = batch_size

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
        **kwargs: Any,
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

    def delete_record(self, **kwargs: Any) -> None:
        client = self.get_client()
        self._delete_record(client=client, collection_name=self.collection_name, **kwargs)

    def run(
        self,
        documents: list[str],
        metadatas: Optional[list[dict[str, str]]] = None,
        **kwargs: Any,
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

        documents_batch = make_batch(documents, batch_size=self.batch_size)
        metadatas_batch = make_batch(metadatas, batch_size=self.batch_size)
        ids_batch = make_batch(ids, batch_size=self.batch_size)

        n_batches: int = math.ceil(len(documents) / self.batch_size)

        if not self._exists(client=client, collection_name=self.collection_name):
            self._create_collection(client=client, collection_name=self.collection_name)
            self.logger.info(f"Create collection {self.collection_name}.")

        for doc_batch, metadata_batch, id_batch in tqdm(
            zip(documents_batch, metadatas_batch, ids_batch, strict=True),
            total=n_batches,
        ):
            if self.embedder is None:
                raise ValueError("embedder is not set.")

            embedding_batch: EmbeddingResults = self.embedder.embed_text(doc_batch)

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

    def _verify_metadata(self, metadatas: list[dict[str, str]]) -> None:
        """
        check if the key 'document' is included in metadatas
        """
        metadata_keys = [metadata for metadata in metadatas if "document" in metadata]
        if len(metadata_keys) > 0:
            raise ValueError("The key 'document' is reserved. Use another key.")


class BaseRemoteCollectionModule(BaseLocalCollectionModule, AbstractRemoteCollectionModule):  # type: ignore[misc]
    def __init__(
        self,
        url: str,
        collection_name: str,
        port: str,
        embedder: LLMModel | None = None,  # type: ignore
        logger: Any | None = None,
        batch_size: int = 100,
    ) -> None:
        self.url = url
        self.port = port
        self.embedder = embedder
        self.collection_name = collection_name
        self.logger = logger or default_logger
        self.batch_size = batch_size

    async def create_collection_async(self) -> None:
        if inspect.iscoroutinefunction(self.get_async_client):
            client = await self.get_async_client()
        else:
            client = self.get_async_client()

        await self._create_collection_async(client=client, collection_name=self.collection_name)

    async def exists_async(self) -> bool:
        if inspect.iscoroutinefunction(self.get_async_client):
            client = await self.get_async_client()
        else:
            client = self.get_async_client()

        return await self._exists_async(client=client, collection_name=self.collection_name)

    async def upsert_async(
        self,
        ids: list[str | int],
        documents: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict[str, str]],
        **kwargs: Any,
    ) -> None:
        self._verify_metadata(metadatas)
        if inspect.iscoroutinefunction(self.get_async_client):
            client = await self.get_async_client()
        else:
            client = self.get_async_client()

        await self._upsert_async(
            client=client,
            collection_name=self.collection_name,
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
            **kwargs,
        )

    async def delete_collection_async(self) -> None:
        if inspect.iscoroutinefunction(self.get_async_client):
            client = await self.get_async_client()
        else:
            client = self.get_async_client()

        if await self._exists_async(client=client, collection_name=self.collection_name):
            await self._delete_collection_async(client=client, collection_name=self.collection_name)

    async def delete_record_async(self, **kwargs: Any) -> None:
        if inspect.iscoroutinefunction(self.get_async_client):
            client = await self.get_async_client()
        else:
            client = self.get_async_client()

        await self._delete_record_async(
            client=client, collection_name=self.collection_name, **kwargs
        )

    async def run_async(
        self,
        documents: list[str],
        metadatas: Optional[list[dict[str, str]]] = None,
        **kwargs: Any,
    ) -> None:
        if inspect.iscoroutinefunction(self.get_async_client):
            client = await self.get_async_client()
        else:
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

        documents_batch = make_batch(documents, batch_size=self.batch_size)
        metadatas_batch = make_batch(metadatas, batch_size=self.batch_size)
        ids_batch = make_batch(ids, batch_size=self.batch_size)

        n_batches: int = math.ceil(len(documents) / self.batch_size)

        if not await self._exists_async(client=client, collection_name=self.collection_name):
            await self._create_collection_async(client=client, collection_name=self.collection_name)
            self.logger.info(f"Create collection {self.collection_name}.")

        for doc_batch, metadata_batch, id_batch in tqdm(
            zip(documents_batch, metadatas_batch, ids_batch, strict=True),
            total=n_batches,
        ):
            if self.embedder is None:
                raise ValueError("embedder is not set.")

            embedding_batch: EmbeddingResults = await self.embedder.embed_text_async(doc_batch)

            n_retries = 0
            while n_retries < 3:
                try:
                    await self._upsert_async(
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
        **kwargs: Any,
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
        **kwargs: Any,
    ) -> RetrievalResults:
        raise NotImplementedError

    @abstractmethod
    async def _retrieve_async(
        self,
        client: Any,
        collection_name: str,
        query_vector: list[float],
        n_results: int,
        score_threshold: float,
        filter: Any | None = None,
        **kwargs: Any,
    ) -> RetrievalResults:
        raise NotImplementedError


class BaseLocalRetrievalModule(AbstractLocalRetrievalModule):
    def __init__(
        self,
        persistence_directory: Path | str,
        collection_name: str,
        embedder: LLMModel | None = None,  # type: ignore
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
        self.logger = logger or default_logger
        self.ascending = ascending

    def run(self, query: str, filter: Any | None = None, **kwargs: Any) -> RetrievalResults:
        client = self.get_client()

        if self.embedder is None:
            raise ValueError("embedder is not set.")

        embed: EmbeddingResults = self.embedder.embed_text(query)

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
        embedder: LLMModel | None = None,  # type: ignore
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
        self.logger = logger or default_logger

    async def run_async(
        self, query: str, filter: Any | None = None, **kwargs: Any
    ) -> RetrievalResults:
        if inspect.iscoroutinefunction(self.get_async_client):
            client = await self.get_async_client()
        else:
            client = self.get_async_client()

        if self.embedder is None:
            raise ValueError("embedder is not set.")

        embed: EmbeddingResults = await self.embedder.embed_text_async(query)

        self.logger.info(f"Retrieve from collection {self.collection_name}...")
        retrieved: RetrievalResults = await self._retrieve_async(
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


class BaseMetadataFilter(ABC):
    @abstractmethod
    def run(self, metadata: dict[str, str]) -> bool:
        raise NotImplementedError

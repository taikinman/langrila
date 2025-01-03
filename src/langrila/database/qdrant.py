import os
from typing import (
    Any,
    Awaitable,
    Callable,
    Mapping,
    Sequence,
    Union,
)

from qdrant_client import AsyncQdrantClient, QdrantClient, models
from qdrant_client.conversions import common_types as types

from ..core.model import LLMModel
from ..core.retrieval import RetrievalResults
from .base import (
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
        vectors_config: Union[types.VectorParams, Mapping[str, types.VectorParams]],
        embedder: LLMModel | None = None,
        logger: Any | None = None,
        sparse_vectors_config: Mapping[str, types.SparseVectorParams] | None = None,
        on_disk_payload: bool | None = None,
        hnsw_config: types.HnswConfigDiff | None = None,
        optimizers_config: types.OptimizersConfigDiff | None = None,
        wal_config: types.WalConfigDiff | None = None,
        quantization_config: types.QuantizationConfig | None = None,
        init_from: types.InitFrom | None = None,
        timeout: int | None = None,
        force_disable_check_same_thread: bool = False,
        batch_size: int = 100,
    ):
        super().__init__(
            persistence_directory=persistence_directory,
            collection_name=collection_name,
            embedder=embedder,
            logger=logger,
            batch_size=batch_size,
        )
        self.vectors_config = vectors_config
        self.sparse_vectors_config = sparse_vectors_config
        self.on_disk_payload = on_disk_payload
        self.hnsw_config = hnsw_config
        self.optimizers_config = optimizers_config
        self.wal_config = wal_config
        self.quantization_config = quantization_config
        self.init_from = init_from
        self.timeout = timeout
        self.force_disable_check_same_thread = force_disable_check_same_thread

    def _exists(self, client: QdrantClient, collection_name: str) -> bool:
        return client.collection_exists(collection_name=collection_name)

    def _create_collection(self, client: QdrantClient, collection_name: str) -> None:
        client.create_collection(
            collection_name=collection_name,
            vectors_config=self.vectors_config,
            sparse_vectors_config=self.sparse_vectors_config,
            on_disk_payload=self.on_disk_payload,
            hnsw_config=self.hnsw_config,
            optimizers_config=self.optimizers_config,
            wal_config=self.wal_config,
            quantization_config=self.quantization_config,
            init_from=self.init_from,
            timeout=self.timeout,
        )

    def _upsert(
        self,
        client: QdrantClient,
        collection_name: str,
        ids: list[str | int],
        documents: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict[str, str]],
        wait: bool = True,
        ordering: types.WriteOrdering | None = None,
        shard_key_selector: types.ShardKeySelector | None = None,
    ) -> None:
        client.upsert(
            collection_name=collection_name,
            points=models.Batch(
                ids=ids,
                vectors=embeddings,
                payloads=[
                    m | {"document": doc} for m, doc in zip(metadatas, documents, strict=True)
                ],
            ),
            wait=wait,
            ordering=ordering,
            shard_key_selector=shard_key_selector,
        )

        return

    def _delete_collection(self, client: QdrantClient, collection_name: str) -> None:
        client.delete_collection(collection_name=collection_name)

    def _delete_record(
        self,
        client: QdrantClient,
        collection_name: str,
        points_selector: types.PointsSelector,
        wait: bool = True,
        ordering: types.WriteOrdering | None = None,
        shard_key_selector: types.ShardKeySelector | None = None,
    ) -> None:
        client.delete(
            collection_name=collection_name,
            points_selector=points_selector,
            wait=wait,
            ordering=ordering,
            shard_key_selector=shard_key_selector,
        )

    def get_client(self) -> QdrantClient:
        return QdrantClient(
            path=self.persistence_directory,
            force_disable_check_same_thread=self.force_disable_check_same_thread,
        )

    def as_retriever(
        self,
        n_results: int = 4,
        score_threshold: float = 0.5,
        ascending: bool = False,
    ) -> "QdrantLocalRetrievalModule":
        return QdrantLocalRetrievalModule(
            embedder=self.embedder,
            persistence_directory=self.persistence_directory,
            collection_name=self.collection_name,
            n_results=n_results,
            score_threshold=score_threshold,
            logger=self.logger,
            force_disable_check_same_thread=self.force_disable_check_same_thread,
            ascending=ascending,
        )


class QdrantRemoteCollectionModule(BaseRemoteCollectionModule):
    def __init__(
        self,
        url: str,
        collection_name: str,
        port: str = "6333",
        embedder: LLMModel | None = None,
        logger: Any | None = None,
        https: bool | None = None,
        api_key_env_name: str | None = None,
        host: str | None = None,
        grpc_port: str | None = "6334",
        grpc_options: dict[str, Any] | None = None,
        auth_token_provider: Union[Callable[[], str], Callable[[], Awaitable[str]]] | None = None,
        vectors_config: Union[types.VectorParams, Mapping[str, types.VectorParams]] = None,
        sparse_vectors_config: Mapping[str, types.SparseVectorParams] | None = None,
        shard_number: int | None = None,
        sharding_method: types.ShardingMethod | None = None,
        replication_factor: int | None = None,
        write_consistency_factor: int | None = None,
        on_disk_payload: bool | None = None,
        hnsw_config: types.HnswConfigDiff | None = None,
        optimizers_config: types.OptimizersConfigDiff | None = None,
        wal_config: types.WalConfigDiff | None = None,
        quantization_config: types.QuantizationConfig | None = None,
        init_from: types.InitFrom | None = None,
        timeout: int | None = None,
        batch_size: int = 100,
    ):
        super().__init__(
            collection_name=collection_name,
            embedder=embedder,
            logger=logger,
            url=url,
            port=port,
            batch_size=batch_size,
        )
        self.https = https
        self.api_key_env_name = api_key_env_name
        self.host = host
        self.grpc_port = grpc_port
        self.grpc_options = grpc_options
        self.auth_token_provider = auth_token_provider
        self.vectors_config = vectors_config
        self.sparse_vectors_config = sparse_vectors_config
        self.shard_number = shard_number
        self.sharding_method = sharding_method
        self.replication_factor = replication_factor
        self.write_consistency_factor = write_consistency_factor
        self.on_disk_payload = on_disk_payload
        self.hnsw_config = hnsw_config
        self.optimizers_config = optimizers_config
        self.wal_config = wal_config
        self.quantization_config = quantization_config
        self.init_from = init_from
        self.timeout = timeout

    def _exists(self, client: QdrantClient, collection_name: str) -> bool:
        return client.collection_exists(collection_name=collection_name)

    def _create_collection(self, client: QdrantClient, collection_name: str) -> None:
        client.create_collection(
            collection_name=collection_name,
            vectors_config=self.vectors_config,
            sparse_vectors_config=self.sparse_vectors_config,
            shard_number=self.shard_number,
            sharding_method=self.sharding_method,
            replication_factor=self.replication_factor,
            write_consistency_factor=self.write_consistency_factor,
            on_disk_payload=self.on_disk_payload,
            hnsw_config=self.hnsw_config,
            optimizers_config=self.optimizers_config,
            wal_config=self.wal_config,
            quantization_config=self.quantization_config,
            init_from=self.init_from,
            timeout=self.timeout,
        )

    def _upsert(
        self,
        client: QdrantClient,
        collection_name: str,
        ids: list[str | int],
        documents: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict[str, str]],
        wait: bool = True,
        ordering: types.WriteOrdering | None = None,
        shard_key_selector: types.ShardKeySelector | None = None,
    ) -> None:
        client.upsert(
            collection_name=collection_name,
            points=models.Batch(
                ids=ids,
                vectors=embeddings,
                payloads=[
                    m | {"document": doc} for m, doc in zip(metadatas, documents, strict=True)
                ],
            ),
            wait=wait,
            ordering=ordering,
            shard_key_selector=shard_key_selector,
        )

        return

    def _delete_collection(self, client: QdrantClient, collection_name: str) -> None:
        client.delete_collection(collection_name=collection_name)

    def _delete_record(
        self,
        client: QdrantClient,
        collection_name: str,
        points_selector: types.PointsSelector,
        wait: bool = True,
        ordering: types.WriteOrdering | None = None,
        shard_key_selector: types.ShardKeySelector | None = None,
    ) -> None:
        client.delete(
            collection_name=collection_name,
            points_selector=points_selector,
            wait=wait,
            ordering=ordering,
            shard_key_selector=shard_key_selector,
        )

    async def _exists_async(self, client: AsyncQdrantClient, collection_name: str) -> bool:
        return await client.collection_exists(collection_name=collection_name)

    async def _create_collection_async(
        self, client: AsyncQdrantClient, collection_name: str
    ) -> None:
        await client.create_collection(
            collection_name=collection_name,
            vectors_config=self.vectors_config,
            sparse_vectors_config=self.sparse_vectors_config,
            shard_number=self.shard_number,
            sharding_method=self.sharding_method,
            replication_factor=self.replication_factor,
            write_consistency_factor=self.write_consistency_factor,
            on_disk_payload=self.on_disk_payload,
            hnsw_config=self.hnsw_config,
            optimizers_config=self.optimizers_config,
            wal_config=self.wal_config,
            quantization_config=self.quantization_config,
            init_from=self.init_from,
            timeout=self.timeout,
        )

    async def _upsert_async(
        self,
        client: AsyncQdrantClient,
        collection_name: str,
        ids: list[str | int],
        documents: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict[str, str]],
        wait: bool = True,
        ordering: types.WriteOrdering | None = None,
        shard_key_selector: types.ShardKeySelector | None = None,
    ) -> None:
        await client.upsert(
            collection_name=collection_name,
            points=models.Batch(
                ids=ids,
                vectors=embeddings,
                payloads=[
                    m | {"document": doc} for m, doc in zip(metadatas, documents, strict=True)
                ],
            ),
            wait=wait,
            ordering=ordering,
            shard_key_selector=shard_key_selector,
        )

        return

    async def _delete_collection_async(
        self, client: AsyncQdrantClient, collection_name: str
    ) -> None:
        await client.delete_collection(collection_name=collection_name)

    async def _delete_record_async(
        self,
        client: AsyncQdrantClient,
        collection_name: str,
        points_selector: types.PointsSelector,
        wait: bool = True,
        ordering: types.WriteOrdering | None = None,
        shard_key_selector: types.ShardKeySelector | None = None,
    ) -> None:
        await client.delete(
            collection_name=collection_name,
            points_selector=points_selector,
            wait=wait,
            ordering=ordering,
            shard_key_selector=shard_key_selector,
        )

    def get_client(self) -> QdrantClient:
        if hasattr(self, "client") and isinstance(self.client, QdrantClient):
            return self.client

        self.client = QdrantClient(
            url=self.url,
            port=self.port,
            https=self.https,
            api_key=os.getenv(self.api_key_env_name) if self.api_key_env_name else None,
            host=self.host,
            grpc_port=self.grpc_port,
            grpc_options=self.grpc_options,
            auth_token_provider=self.auth_token_provider,
            timeout=self.timeout,
        )
        return self.client

    def get_async_client(self) -> AsyncQdrantClient:
        if hasattr(self, "client") and isinstance(self.client, AsyncQdrantClient):
            return self.client

        self.client = AsyncQdrantClient(
            url=self.url,
            port=self.port,
            https=self.https,
            api_key=os.getenv(self.api_key_env_name) if self.api_key_env_name else None,
            host=self.host,
            grpc_port=self.grpc_port,
            grpc_options=self.grpc_options,
            auth_token_provider=self.auth_token_provider,
            timeout=self.timeout,
        )
        return self.client

    def as_retriever(
        self,
        n_results: int = 4,
        score_threshold: float = 0.5,
        ascending: bool = False,
    ) -> "QdrantRemoteRetrievalModule":
        return QdrantRemoteRetrievalModule(
            embedder=self.embedder,
            url=self.url,
            port=self.port,
            collection_name=self.collection_name,
            n_results=n_results,
            score_threshold=score_threshold,
            logger=self.logger,
            https=self.https,
            api_key_env_name=self.api_key_env_name,
            host=self.host,
            grpc_port=self.grpc_port,
            grpc_options=self.grpc_options,
            auth_token_provider=self.auth_token_provider,
            timeout=self.timeout,
            ascending=ascending,
        )


class QdrantLocalRetrievalModule(BaseLocalRetrievalModule):
    def __init__(
        self,
        persistence_directory: str,
        collection_name: str,
        embedder: LLMModel | None = None,
        n_results: int = 4,
        score_threshold: float = 0.5,
        logger: Any | None = None,
        ascending: bool = False,
        force_disable_check_same_thread: bool = False,
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
        self.force_disable_check_same_thread = force_disable_check_same_thread

    def get_client(self) -> QdrantClient:
        return QdrantClient(
            path=self.persistence_directory,
            force_disable_check_same_thread=self.force_disable_check_same_thread,
        )

    def _retrieve(
        self,
        client: QdrantClient,
        collection_name: str,
        query_vector: list[float],
        n_results: int,
        score_threshold: float,
        filter: Any | None = None,
        search_params: types.SearchParams | None = None,
        offset: int | None = None,
        with_payload: Union[bool, Sequence[str], types.PayloadSelector] = True,
        with_vectors: Union[bool, Sequence[str]] = False,
        append_payload: bool = True,
        consistency: types.ReadConsistency | None = None,
        shard_key_selector: types.ShardKeySelector | None = None,
        timeout: int | None = None,
    ) -> RetrievalResults:
        retrieved = client.search(
            collection_name=collection_name,
            query_vector=query_vector,
            query_filter=filter,
            score_threshold=score_threshold,
            limit=n_results,
            search_params=search_params,
            offset=offset,
            with_payload=with_payload,
            with_vectors=with_vectors,
            append_payload=append_payload,
            consistency=consistency,
            shard_key_selector=shard_key_selector,
            timeout=timeout,
        )

        ids = [r.id for r in retrieved]
        scores = [r.score for r in retrieved]
        documents = [r.payload["document"] for r in retrieved]
        metadatas = [r.payload for r in retrieved]
        collections = [collection_name for _ in retrieved]

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
        embedder: LLMModel | None = None,
        n_results: int = 4,
        score_threshold: float = 0.5,
        logger: Any | None = None,
        ascending: bool = False,
        https: bool | None = None,
        api_key_env_name: str | None = None,
        host: str | None = None,
        grpc_port: str | None = "6334",
        grpc_options: dict[str, Any] | None = None,
        auth_token_provider: Union[Callable[[], str], Callable[[], Awaitable[str]]] | None = None,
        timeout: int | None = None,
    ):
        super().__init__(
            collection_name=collection_name,
            embedder=embedder,
            n_results=n_results,
            score_threshold=score_threshold,
            logger=logger,
            url=url,
            port=port,
            ascending=ascending,
        )
        self.https = https
        self.api_key_env_name = api_key_env_name
        self.host = host
        self.grpc_port = grpc_port
        self.grpc_options = grpc_options
        self.auth_token_provider = auth_token_provider
        self.timeout = timeout

    def get_client(self) -> QdrantClient:
        if hasattr(self, "client") and isinstance(self.client, QdrantClient):
            return self.client

        self.client = QdrantClient(
            url=self.url,
            port=self.port,
            https=self.https,
            api_key=os.getenv(self.api_key_env_name) if self.api_key_env_name else None,
            host=self.host,
            grpc_port=self.grpc_port,
            grpc_options=self.grpc_options,
            auth_token_provider=self.auth_token_provider,
            timeout=self.timeout,
        )
        return self.client

    def get_async_client(self) -> AsyncQdrantClient:
        if hasattr(self, "client") and isinstance(self.client, AsyncQdrantClient):
            return self.client

        self.client = AsyncQdrantClient(
            url=self.url,
            port=self.port,
            https=self.https,
            api_key=os.getenv(self.api_key_env_name) if self.api_key_env_name else None,
            host=self.host,
            grpc_port=self.grpc_port,
            grpc_options=self.grpc_options,
            auth_token_provider=self.auth_token_provider,
            timeout=self.timeout,
        )
        return self.client

    def _retrieve(
        self,
        client: QdrantClient,
        collection_name: str,
        query_vector: list[float],
        n_results: int,
        score_threshold: float,
        filter: Any | None = None,
        search_params: types.SearchParams | None = None,
        offset: int | None = None,
        with_payload: Union[bool, Sequence[str], types.PayloadSelector] = True,
        with_vectors: Union[bool, Sequence[str]] = False,
        append_payload: bool = True,
        consistency: types.ReadConsistency | None = None,
        shard_key_selector: types.ShardKeySelector | None = None,
        timeout: int | None = None,
    ) -> RetrievalResults:
        retrieved = client.search(
            collection_name=collection_name,
            query_vector=query_vector,
            query_filter=filter,
            score_threshold=score_threshold,
            limit=n_results,
            search_params=search_params,
            offset=offset,
            with_payload=with_payload,
            with_vectors=with_vectors,
            append_payload=append_payload,
            consistency=consistency,
            shard_key_selector=shard_key_selector,
            timeout=timeout,
        )

        ids = [r.id for r in retrieved]
        scores = [r.score for r in retrieved]
        documents = [r.payload["document"] for r in retrieved]
        metadatas = [r.payload for r in retrieved]
        collections = [collection_name for _ in retrieved]

        return RetrievalResults(
            ids=ids,
            scores=scores,
            documents=documents,
            metadatas=metadatas,
            collections=collections,
        )

    async def _retrieve_async(
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
        metadatas = [r.payload for r in retrieved]
        collections = [collection_name for _ in retrieved]

        return RetrievalResults(
            ids=ids,
            scores=scores,
            documents=documents,
            metadatas=metadatas,
            collections=collections,
        )

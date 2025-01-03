import math
from pathlib import Path
from typing import Any

import numpy as np
from usearch.index import DTypeLike, Index, Matches, MetricLike, ProgressCallback

from ..core.model import LLMModel
from ..core.retrieval import RetrievalResults
from .base import (
    BaseLocalCollectionModule,
    BaseLocalRetrievalModule,
    BaseMetadataFilter,
)
from .metadata_store.sqlite import SQLiteMetadataStore


class UsearchLocalCollectionModule(BaseLocalCollectionModule):
    def __init__(
        self,
        persistence_directory: str,
        collection_name: str,
        ndim: int,
        metric: MetricLike = "cos",
        dtype: DTypeLike | None = "f32",
        connectivity: int | None = None,
        expansion_add: int | None = None,
        expansion_search: int | None = None,
        multi: bool = False,
        view: bool = False,
        enable_key_lookups: bool = True,
        embedder: LLMModel | None = None,
        logger: Any | None = None,
        batch_size: int = 100,
    ):
        super().__init__(
            persistence_directory=persistence_directory,
            collection_name=collection_name,
            embedder=embedder,
            logger=logger,
            batch_size=batch_size,
        )
        self.ndim = ndim
        self.metric = metric
        self.dtype = dtype
        self.connectivity = connectivity
        self.expansion_add = expansion_add
        self.expansion_search = expansion_search
        self.multi = multi
        self.view = view
        self.enable_key_lookups = enable_key_lookups
        self._collection_path = (
            Path(self.persistence_directory) / self.collection_name / "vectordb.usearch"
        )

        self.metadata_store = SQLiteMetadataStore(
            persistence_directory=persistence_directory, collection_name=collection_name
        )

    def get_client(self) -> Index:
        if hasattr(self, "client"):
            return self.client

        self.client = Index(
            ndim=self.ndim,
            metric=self.metric,
            dtype=self.dtype,
            connectivity=self.connectivity,
            expansion_add=self.expansion_add,
            expansion_search=self.expansion_search,
            multi=self.multi,
            view=self.view,
            enable_key_lookups=self.enable_key_lookups,
            path=self._collection_path,
        )

        return self.client

    def _exists(self, **kwargs) -> bool:
        return self._collection_path.exists()

    def _create_collection(self, client: Index, **kwargs) -> None:
        self._collection_path.parent.mkdir(parents=True, exist_ok=True)
        client.save(self._collection_path)

    def _delete_record(self, client: Index, ids: list[int] | np.ndarray, **kwargs) -> None:
        client.remove(ids, **kwargs)
        self.metadata_store.delete_records(ids=ids)

    def _upsert(
        self,
        client: Index,
        collection_name: str,  # not used
        ids: list[str | int],
        documents: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict[str, str]],
        copy: bool = False,
        threads: int = 0,
        log: bool = False,
        progress: ProgressCallback | None = None,
    ) -> None:
        is_contained = client.contains(ids)

        if is_contained.any():
            # FIXME : search result may be different when new vectors are inserted after existing vectors are removed.
            # Instead, rebuilding the index is recommended using `delete_collection` before upserting.
            # Or use exact search to avoid this issue when search time.
            duplicate_keys = np.where(is_contained)[0]
            client.remove(np.array(ids)[duplicate_keys])
            self.metadata_store.delete_records(ids=ids)

        client.add(
            keys=ids,
            vectors=np.array(embeddings),
            copy=copy,
            threads=threads,
            log=log,
            progress=progress,
        )

        self.metadata_store.store(
            ids=ids,
            metadatas=[m | {"document": doc} for m, doc in zip(metadatas, documents, strict=True)],
        )

        self._save_on_last(client)

        return

    def _save_on_last(self, client: Index) -> None:
        client.save(self._collection_path)

    def _delete_collection(self, client: Index, **kwargs) -> None:
        client.reset()
        self.metadata_store.delete_table()

    def as_retriever(
        self,
        n_results: int = 4,
        score_threshold: float = 0.5,
        n_results_coef: float = 5.0,
    ) -> "UsearchLocalRetrievalModule":
        return UsearchLocalRetrievalModule(
            persistence_directory=self.persistence_directory,
            collection_name=self.collection_name,
            ndim=self.ndim,
            metric=self.metric,
            dtype=self.dtype,
            connectivity=self.connectivity,
            expansion_add=self.expansion_add,
            expansion_search=self.expansion_search,
            multi=self.multi,
            view=self.view,
            enable_key_lookups=self.enable_key_lookups,
            embedder=self.embedder,
            n_results=n_results,
            score_threshold=score_threshold,
            logger=self.logger,
            n_results_coef=n_results_coef,
        )


class UsearchLocalRetrievalModule(BaseLocalRetrievalModule):
    def __init__(
        self,
        persistence_directory: str,
        collection_name: str,
        ndim: int,
        metric: MetricLike = "cos",
        dtype: DTypeLike | None = "f32",
        connectivity: int | None = None,
        expansion_add: int | None = None,
        expansion_search: int | None = None,
        multi: bool = False,
        view: bool = False,
        enable_key_lookups: bool = True,
        embedder: LLMModel | None = None,
        n_results: int = 4,
        score_threshold: float = 0.5,
        logger: Any | None = None,
        n_results_coef: float = 5.0,  # It's multiplied by n_results to get the number of results to fetch before filtering
    ):
        super().__init__(
            persistence_directory=persistence_directory,
            collection_name=collection_name,
            embedder=embedder,
            n_results=n_results,
            score_threshold=score_threshold,
            logger=logger,
        )

        self.ndim = ndim
        self.metric = metric
        self.dtype = dtype
        self.connectivity = connectivity
        self.expansion_add = expansion_add
        self.expansion_search = expansion_search
        self.multi = multi
        self.view = view
        self.enable_key_lookups = enable_key_lookups
        self._collection_path = (
            Path(self.persistence_directory) / self.collection_name / "vectordb.usearch"
        )

        self.metadata_store = SQLiteMetadataStore(
            persistence_directory=persistence_directory, collection_name=collection_name
        )
        self.n_results_coef = n_results_coef

    def get_client(self) -> Index:
        if hasattr(self, "client"):
            return self.client

        self.client = Index(
            ndim=self.ndim,
            metric=self.metric,
            dtype=self.dtype,
            connectivity=self.connectivity,
            expansion_add=self.expansion_add,
            expansion_search=self.expansion_search,
            multi=self.multi,
            view=self.view,
            enable_key_lookups=self.enable_key_lookups,
            path=self._collection_path,
        )

        return self.client

    def _retrieve(
        self,
        client: Index,
        collection_name: str,
        query_vector: list[float],
        n_results: int,
        score_threshold: float,
        filter: BaseMetadataFilter | None = None,
        radius: float = math.inf,
        threads: int = 0,
        exact: bool = False,
        log: bool = False,
        progress: ProgressCallback | None = None,
    ) -> RetrievalResults:
        matches: Matches = client.search(
            vectors=np.array(query_vector),
            count=int(n_results * self.n_results_coef) if filter else n_results,
            radius=radius,
            threads=threads,
            exact=exact,
            log=log,
            progress=progress,
        )

        ids = matches.keys.tolist()
        scores = matches.distances.tolist()

        # filter by score threshold as list
        ids = [i for i, s in zip(ids, scores, strict=True) if s <= score_threshold]
        scores = [s for s in scores if s <= score_threshold]
        collections = [collection_name for _ in ids]

        metadatas = self.metadata_store.fetch(ids)
        documents = [m["document"] for m in metadatas]

        # filter by metadata
        if filter:
            use_indices = [i for i, metadata in enumerate(metadatas) if filter.run(metadata)][
                :n_results
            ]
            ids = [ids[i] for i in use_indices]
            documents = [documents[i] for i in use_indices]
            metadatas = [metadatas[i] for i in use_indices]
            scores = [scores[i] for i in use_indices]
            collections = [collections[i] for i in use_indices]

        return RetrievalResults(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
            scores=scores,
            collections=collections,
        )

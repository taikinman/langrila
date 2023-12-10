from pathlib import Path
from typing import Optional

import numpy as np
from qdrant_client import QdrantClient, models
from qdrant_client.conversions import common_types as types
from qdrant_client.models import Distance, VectorParams

from ..base import BaseModule
from ..result import RetrievalResult


class QdrantLocalCollectionModule(BaseModule):
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
                embeddings = np.array(self.embedder(documents).embeddings)
            else:
                raise AttributeError(
                    "attribute embedder must be the instance of the class inheriting BaseModule."
                )

        if not isinstance(embeddings, np.ndarray):
            embeddings = np.array(embeddings)

        client = QdrantClient(path=self.persistence_directory)
        client.recreate_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(size=embeddings.shape[1], distance=Distance.COSINE),
        )

        client.upload_records(
            collection_name=self.collection_name,
            records=[
                models.Record(
                    id=idx,
                    vector=embeddings[idx].tolist(),
                    payload={"document": doc, "metadata": metadatas[idx]},
                )
                if metadatas is not None
                else models.Record(
                    id=idx,
                    vector=embeddings[idx].tolist(),
                    payload={"document": doc, "metadata": None},
                )
                for idx, doc in enumerate(documents)
            ],
        )

    def as_retriever(
        self,
        n_results: int = 4,
        threshold_similarity: float = 0.8,
        with_vectors: bool = True,
    ) -> "QdrantLocalRetrievalModule":
        return QdrantLocalRetrievalModule(
            embedder=self.embedder,
            persistence_directory=self.persistence_directory,
            collection_name=self.collection_name,
            n_results=n_results,
            threshold_similarity=threshold_similarity,
            with_vectors=with_vectors,
        )


class QdrantLocalRetrievalModule(BaseModule):
    def __init__(
        self,
        embedder: BaseModule,
        persistence_directory: str,
        collection_name: str,
        n_results: int = 4,
        threshold_similarity: float = 0.8,
        with_vectors: bool = True,
    ):
        assert isinstance(
            embedder, BaseModule
        ), "embedder must be the instance of the class inheriting BaseModule."
        self.embedder = embedder
        self.n_results = n_results
        self.threshold_similarity = threshold_similarity
        self.persistence_directory = persistence_directory
        self.collection_name = collection_name
        self.n_results = n_results
        self.with_vectors = with_vectors

    def run(self, query: str, filter: Optional[types.Filter] = None):
        client = QdrantClient(path=self.persistence_directory)

        embed = self.embedder(query)

        retrieved = client.search(
            collection_name=self.collection_name,
            query_vector=embed.embeddings[0],
            query_filter=filter,
            score_threshold=self.threshold_similarity,
            with_vectors=self.with_vectors,
            limit=self.n_results,
        )

        results = RetrievalResult(
            ids=[r.id for r in retrieved],
            similarities=[r.score for r in retrieved],
            documents=[r.payload["document"] for r in retrieved],
            metadatas=[r.payload["metadata"] for r in retrieved],
            usage=embed.usage,
        )

        # results = {
        #     "ids": [r.id for r in retrieved],
        #     "distances": [1 - r.score for r in retrieved],
        #     "documents": [r.payload["document"] for r in retrieved],
        #     "metadatas": [r.payload["metadata"] for r in retrieved],
        # }

        return results


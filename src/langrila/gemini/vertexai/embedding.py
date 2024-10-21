from typing import Any, Sequence

from google.auth import credentials as auth_credentials

from ...base import BaseEmbeddingModule
from ...result import EmbeddingResults
from ...usage import Usage
from ..gemini_utils import get_client


class VertexAIEmbeddingModule(BaseEmbeddingModule):
    def __init__(
        self,
        model_name: str | None = None,
        title: str | None = None,
        dimensions: int | None = None,
        task_type: str | None = None,
        api_key_env_name: str | None = None,
        project_id_env_name: str | None = None,
        location_env_name: str | None = None,
        experiment: str | None = None,
        experiment_description: str | None = None,
        experiment_tensorboard: str | bool | None = None,
        staging_bucket: str | None = None,
        credentials: auth_credentials.Credentials | None = None,
        encryption_spec_key_name: str | None = None,
        network: str | None = None,
        service_account: str | None = None,
        endpoint_env_name: str | None = None,
        request_metadata: Sequence[tuple[str, str]] | None = None,
        **kwargs: Any,
    ):
        self.api_type = "vertexai"
        self.model_name = model_name
        self.title = title
        self.dimensions = dimensions
        self.task_type = task_type

        self._client = get_client(
            api_key_env_name=api_key_env_name,
            api_type=self.api_type,
            project_id_env_name=project_id_env_name,
            location_env_name=location_env_name,
            experiment=experiment,
            experiment_description=experiment_description,
            experiment_tensorboard=experiment_tensorboard,
            staging_bucket=staging_bucket,
            credentials=credentials,
            encryption_spec_key_name=encryption_spec_key_name,
            network=network,
            service_account=service_account,
            endpoint_env_name=endpoint_env_name,
            request_metadata=request_metadata,
        )

    def _get_embedding_kwargs(self, **kwargs):
        _kwargs = {}
        _kwargs["model_name"] = kwargs.get("model_name") or self.model_name
        _kwargs["title"] = kwargs.get("title") or self.title

        if kwargs.get("task_type") or self.task_type:
            _kwargs["task_type"] = kwargs.get("task_type") or self.task_type

        if kwargs.get("dimensions") or self.dimensions:
            _kwargs["output_dimensionality"] = kwargs.get("dimensions") or self.dimensions

        return _kwargs

    def run(
        self,
        text: str | list[str],
        model_name: str | None = None,
        dimensions: int | None = None,
        task_type: str | None = None,
        title: str | None = None,
    ) -> EmbeddingResults:
        if not isinstance(text, list):
            text = [text]

        embedding_kwargs = self._get_embedding_kwargs(
            model_name=model_name,
            dimensions=dimensions,
            task_type=task_type,
            title=title,
        )

        embeddings = self._client.embed_text(texts=text, **embedding_kwargs)
        prompt_usage = Usage(
            model_name=embedding_kwargs.get("model_name"),
            prompt_tokens=int(sum([emb.statistics.token_count for emb in embeddings])),
        )
        return EmbeddingResults(
            text=text,
            embeddings=[emb.values for emb in embeddings],
            usage=prompt_usage,
        )

    async def arun(
        self,
        text: str | list[str],
        model_name: str | None = None,
        dimensions: int | None = None,
        task_type: str | None = None,
        title: str | None = None,
    ) -> EmbeddingResults:
        if not isinstance(text, list):
            text = [text]

        embedding_kwargs = self._get_embedding_kwargs(
            model_name=model_name,
            dimensions=dimensions,
            task_type=task_type,
            title=title,
        )

        embeddings = await self._client.embed_text_async(texts=text, **embedding_kwargs)
        prompt_usage = Usage(
            model_name=embedding_kwargs.get("model_name"),
            prompt_tokens=int(sum([emb.statistics.token_count for emb in embeddings])),
        )

        return EmbeddingResults(
            text=text,
            embeddings=[emb.values for emb in embeddings],
            usage=prompt_usage,
        )
